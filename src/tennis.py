# -*- coding: utf-8 -*-
import datetime
import copy
import random
import functools
import unittest
from contextlib import closing
from typing import Optional

import common as co
import log
import dba
import tennis_time as tt
import score as sc
import ratings
import bet
import bet_coefs
import oncourt_players
import stat_cont as st


@functools.total_ordering
class RoundGroup(object):
    names = ("qual", "mdraw")

    def __init__(self, value):
        self.value = value.strip()
        assert (
            isinstance(self.value, str) and self.value in RoundGroup.names
        ), "unexpected rg value: '{}'".format(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, RoundGroup):
            return self.value == other.value
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, str):
            idx_other = RoundGroup.names.index(other)
        else:
            idx_other = RoundGroup.names.index(other.value)
        return RoundGroup.names.index(self.value) < idx_other


@functools.total_ordering
class Round(object):
    robin_names = [
        "Robin",
        "Rubber 1",
        "Rubber 2",
        "Rubber 3",
        "Rubber 4",
        "Rubber 5",
        "N/A",
    ]
    names = (
        ["Pre-q", "q-First", "q-Second", "Qualifying", "First"]
        + robin_names
        + ["Second", "Third", "Fourth", "1/4", "1/2", "Final", "Bronze"]
    )
    to_oncourt_id = {
        "Pre-q": 0,
        "q-First": 1,
        "q-Second": 2,
        "Qualifying": 3,
        "First": 4,
        "Second": 5,
        "Third": 6,
        "Fourth": 7,
        "Robin": 8,
        "1/4": 9,
        "1/2": 10,
        "Final": 12,
        "Bronze": 11,
        "Rubber 1": 13,
        "Rubber 2": 14,
        "Rubber 3": 15,
        "Rubber 4": 16,
        "Rubber 5": 17,
        "N/A": 20,
    }

    def __init__(self, value, details=None):
        self.value = co.to_ascii(value.strip())
        self._detailing = details
        assert self.value in Round.names, "unexpected rnd value: {}".format(self.value)

    @property
    def detailing(self):
        return self._detailing

    @detailing.setter
    def detailing(self, value):
        self._detailing = value

    def __str__(self):
        return self.value if not self.detailing else self.value + ";" + self.detailing

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return "{}('{}', {})".format(
            self.__class__.__name__,
            self.value,
            "None" if not self.detailing else "'" + self.detailing + "'",
        )

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other  # and not self.detailing
        elif isinstance(other, Round):
            return self.value == other.value and self.detailing == other.detailing
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if other is None:
            return False
        idx_other = (
            Round.names.index(other)
            if isinstance(other, str)
            else Round.names.index(other.value)
        )
        idx_me = Round.names.index(self.value)
        if idx_other == idx_me:
            return (
                False
                if isinstance(other, str)
                else str(self.detailing) < str(other.detailing)
            )
        else:
            return idx_me < idx_other

    def group(self):
        return RoundGroup("qual") if self.qualification() else RoundGroup("mdraw")

    def main_draw(self):
        return self.value not in ("Pre-q", "q-First", "q-Second", "Qualifying")

    def robin(self):
        return self.value in Round.robin_names

    def rubber(self):
        return self.value.startswith("Rubber")

    def pre_qualification(self):
        return self.value == "Pre-q"

    def qualification(self):
        return self.value in ("q-First", "q-Second", "Qualifying")

    def final(self):
        return self.value == "Final"

    def oncourt_id(self):
        return Round.to_oncourt_id[self.value]

    @staticmethod
    def from_oncourt_id(oncourt_id):
        for name, ident in Round.to_oncourt_id.items():
            if ident == oncourt_id:
                return Round(name)

    def get_next(self, matches_encountered=None):
        if self.value not in ("Final", "Bronze", "Rubber 5"):
            if self.robin():
                idx = Round.robin_names.index(self.value)
                return Round(Round.robin_names[idx + 1], self.detailing)
            else:
                if self.value == "q-First":
                    if matches_encountered is not None and matches_encountered <= 4:
                        return Round("Qualifying")
                    else:
                        return Round("q-Second")
                if self.value == "Qualifying":
                    return Round("First", self.detailing)
                if self.value == "Second":
                    if matches_encountered is not None and matches_encountered <= 8:
                        return Round("1/4", self.detailing)
                    else:
                        return Round("Third", self.detailing)
                if self.value == Round("Third"):
                    if matches_encountered is not None and matches_encountered <= 8:
                        return Round("1/4", self.detailing)
                    else:
                        return Round("Fourth", self.detailing)
                idx = Round.names.index(self.value)
                return Round(Round.names[idx + 1], self.detailing)

    def define_detailing(self, lhs_past_rounds, rhs_past_rounds):
        if self.qualification() or self.pre_qualification():
            return
        lhs_qual = any([r.qualification() for r in lhs_past_rounds])
        rhs_qual = any([r.qualification() for r in rhs_past_rounds])
        if lhs_qual and rhs_qual:
            self.detailing = "qual vs qual"
            return
        if self.value == "First" or self.robin():
            if lhs_qual or rhs_qual:
                self.detailing = "vs qual"
            return  # need not analyze for bye
        lhs_bye = "First" not in lhs_past_rounds and not any(
            [r.robin() for r in lhs_past_rounds]
        )
        rhs_bye = "First" not in rhs_past_rounds and not any(
            [r.robin() for r in rhs_past_rounds]
        )
        assert not (
            lhs_qual and lhs_bye
        ), "lhs_qual AND! lhs_bye. lhs_past_rounds: {}".format(lhs_past_rounds)
        assert not (
            rhs_qual and rhs_bye
        ), "rhs_qual AND! rhs_bye. rhs_past_rounds: {}".format(rhs_past_rounds)
        if lhs_bye and rhs_bye:
            self.detailing = "bye vs bye"
        elif (lhs_qual or rhs_qual) and (lhs_bye or rhs_bye):
            self.detailing = "bye vs qual"
        elif lhs_qual or rhs_qual:
            self.detailing = "vs qual"
        elif lhs_bye or rhs_bye:
            self.detailing = "vs bye"


class RoundTest(unittest.TestCase):
    def test_cmp(self):
        self.assertTrue(Round("Rubber 3") < "Rubber 5")
        self.assertTrue(Round("Rubber 3") > "Rubber 2")
        self.assertTrue("Rubber 3" < Round("Rubber 5"))
        self.assertTrue("Rubber 3" > Round("Rubber 2"))

        self.assertTrue("Rubber 3" == Round("Rubber 3"))
        self.assertTrue(Round("Rubber 3") == "Rubber 3")

        self.assertFalse(Round("Rubber 2").qualification())

        self.assertTrue(Round("First", details="vs qual") == "First")


def get_rnd_metric(rnd):
    return get_rnd_metric.rnd_to_metric[rnd]


get_rnd_metric.rnd_to_metric = {
    "Pre-q": 0.0,
    "q-First": 0.2,
    "q-Second": 0.4,
    "Qualifying": 0.6,
    "First": 0.9,
    "Robin": 1.0,
    "Rubber 1": 1.05,
    "Rubber 2": 1.1,
    "Rubber 3": 1.15,
    "Rubber 4": 1.2,
    "Rubber 5": 1.25,
    "Second": 1.8,
    "Third": 2.5,
    "Fourth": 3.0,
    "1/4": 3.5,
    "1/2": 4.2,
    "Bronze": 4.3,
    "Final": 5.2,
}


def match_keys_combinations(level, surface, rnd):
    args = []
    if level:
        args.append({"level": level})
    if surface:
        args.append({"surface": surface})
    if rnd:
        args.append({"rnd": rnd})
        args.append({"rg": rnd.group()})
    return co.keys_combinations(args)


class Match(object):
    def __init__(
        self, first_player=None, second_player=None, score=None, rnd=None, date=None
    ):
        self.first_player = copy.copy(first_player)
        self.second_player = copy.copy(second_player)
        self.score = copy.copy(score)
        self.rnd = copy.copy(rnd)
        self.date = date
        self.first_draw_status = (
            None  # after init try may be '' as mark of need not init
        )
        self.second_draw_status = (
            None  # after init try may be '' as mark of need not init
        )
        self.head_to_head = None
        self.stat = None
        self.offer = bet.Offer(bet.get_pinnacle_company())
        self.decided_tiebreak_info = None
        self.set_items = None  # here may be Dict[int, SetItems]

    def flip(self):
        self.first_player, self.second_player = self.second_player, self.first_player
        if self.score:
            self.score = self.score.fliped()
        self.first_draw_status, self.second_draw_status = (
            self.second_draw_status,
            self.first_draw_status,
        )
        if self.offer:
            self.offer.flip()
        if self.stat:
            self.stat.flip()
        if hasattr(self, "detailed_score"):
            import detailed_score

            self.detailed_score = detailed_score.detailed_score_side_reversed(
                self.detailed_score
            )

    def set_decided_tiebreak_info(self, tour, rnd: Round):
        """:return True if ok, False if error"""
        try:
            score, dectieinfo = sc.get_decided_tiebreak_info_ext(tour, rnd, self.score)
            if not score.valid():
                log.warn(
                    f"notchecked invalid score {score} {score.error} "
                    f"\n\t{tour.sex} {tour.date} {tour.name} "
                    f"{self.first_player} {self.second_player}"
                )
                return False
            else:
                self.decided_tiebreak_info = dectieinfo
                self.score = score
                return True
        except co.TennisScoreSuperTieError as err:
            log.warn(f"notchecked {err} {self.first_player} {self.second_player}")
            return False

    @property
    def decided_tiebreak(self):
        if self.decided_tiebreak_info is not None:
            return self.decided_tiebreak_info.beg_scr is not None

    def paired(self):
        result = None
        if self.first_player is not None:
            if "/" in self.first_player.name:
                return True
            result = False

        if self.second_player is not None:
            if "/" in self.second_player.name:
                return True
            result = False
        return result

    def single(self):
        return not self.paired()

    def is_paired_participant(self, player):
        if self.first_player is not None:
            names = self.first_player.name.split("/")
            if len(names) == 2 and player.name in names:
                return True
        if self.second_player is not None:
            names = self.second_player.name.split("/")
            if len(names) == 2 and player.name in names:
                return True
        return False

    def detailed_score_correspond(self):
        """соответствует ли детальный счет просто счету"""
        if hasattr(self, "detailed_score"):
            etalon_score_t = tuple(s for s in self.score)
            detscore_t = self.detailed_score.final_score()
            return etalon_score_t == detscore_t

    def __str__(self):
        return "{}{}{}{} - {}{}{}".format(
            "" if self.date is None else str(self.date) + " ",
            "" if self.rnd is None else str(self.rnd) + " ",
            "" if not self.first_draw_status else "(" + self.first_draw_status + ") ",
            self.first_player,
            "" if not self.second_draw_status else "(" + self.second_draw_status + ") ",
            self.second_player,
            "" if not self.score else " " + str(self.score),
        )

    def first_player_bet_chance(self):
        if self.offer and self.offer.win_coefs:
            return self.offer.win_coefs.chances()[0]

    def is_first_favorite(self, min_chance=0.7, min_diff=None):
        import ratings_std

        if min_diff is None:
            min_diff = ratings_std.default_min_diff
        if self.first_player is None or self.second_player is None:
            return None
        if self.offer.win_coefs:
            return self.offer.win_coefs.chances()[0] >= min_chance
        return co.LT == ratings_std.compare_rank(
            self.first_player.rating.rank("std"),
            self.second_player.rating.rank("std"),
            min_diff=min_diff,
        )

    def is_second_favorite(self, min_chance=0.7, min_diff=None):
        import ratings_std

        if min_diff is None:
            min_diff = ratings_std.default_min_diff
        if self.first_player is None or self.second_player is None:
            return None
        if self.offer.win_coefs:
            return self.offer.win_coefs.chances()[1] >= min_chance
        return co.GT == ratings_std.compare_rank(
            self.first_player.rating.rank("std"),
            self.second_player.rating.rank("std"),
            min_diff=min_diff,
        )

    def favorite_player(self, min_chance=0.7, min_diff=None):
        import ratings_std

        if min_diff is None:
            min_diff = ratings_std.default_min_diff
        if self.is_first_favorite(min_chance=min_chance, min_diff=min_diff):
            return self.first_player
        elif self.is_second_favorite(min_chance=min_chance, min_diff=min_diff):
            return self.second_player

    def favorite_firsted_offer(self):
        import ratings_std

        def is_flip_by_rating(first_player, second_player):
            cmp_val = ratings_std.compare_rank(
                first_player.rating.rank("std"),
                second_player.rating.rank("std"),
                min_diff=1,
            )
            if cmp_val == co.LT:
                result = False
            elif cmp_val == co.GT:
                result = True
            elif cmp_val == co.EQ:
                result = random.choice((True, False))
            else:
                result = None
            return result

        result_offer = copy.deepcopy(self.offer)
        if not result_offer.win_coefs:
            if not self.first_player or not self.second_player:
                return None, None
            is_flip = is_flip_by_rating(self.first_player, self.second_player)
            if is_flip:
                result_offer.flip()
            return result_offer, is_flip

        left_chance, right_chance = result_offer.win_coefs.chances()
        compare = co.cmp(left_chance, right_chance)
        is_flip = False
        if compare == -1:
            is_flip = True
        elif compare == 0 and self.first_player and self.second_player:
            is_flip = is_flip_by_rating(self.first_player, self.second_player)
        if is_flip:
            result_offer.flip()
        return result_offer, is_flip

    def is_ranks_both_below(self, value, rtg_name="std"):
        """True если у обоих рейтинг below (worse) указанному"""
        if self.first_player and self.second_player:
            cmp_r1 = self.first_player.rating.cmp_rank(value, rtg_name)
            cmp_r2 = self.second_player.rating.cmp_rank(value, rtg_name)
            if cmp_r1 and cmp_r2:
                return cmp_r1 == ratings.BELOW and cmp_r2 == ratings.BELOW

    def is_ranks_both_above(self, value, rtg_name="std"):
        """True если у обоих рейтинг above (better) указанному"""
        if self.first_player and self.second_player:
            cmp_r1 = self.first_player.rating.cmp_rank(value, rtg_name)
            cmp_r2 = self.second_player.rating.cmp_rank(value, rtg_name)
            if cmp_r1 and cmp_r2:
                return cmp_r1 == ratings.ABOVE and cmp_r2 == ratings.ABOVE

    def is_ranks_dif_wide(self, max_dif, rtg_name="std"):
        if self.first_player and self.second_player:
            r1 = self.first_player.rating.rank(rtg_name)
            r2 = self.second_player.rating.rank(rtg_name)
            if r1 is not None and r2 is not None:
                return abs(r1 - r2) > max_dif

    def read_ratings(self, sex, date=None, surfaces=("all",)):
        if self.first_player:
            self.first_player.read_rating(sex, date=date, surfaces=surfaces)

        if self.second_player:
            self.second_player.read_rating(sex, date=date, surfaces=surfaces)

    def read_birth_date(self, sex):
        if self.first_player is not None:
            self.first_player.read_birth_date(sex)

        if self.second_player is not None:
            self.second_player.read_birth_date(sex)

    def read_pers_det(self, sex):
        if self.first_player:
            self.first_player.read_pers_det(sex)

        if self.second_player:
            self.second_player.read_pers_det(sex)

    def fill_offer(self, sex, tour_id, tour_date, alter_bettor):
        """alter_bettor: None - only try1 default bettor;
        False try1 default then if fail try2 alter bettor;
        True try2 alter bettor.
        """
        if alter_bettor:
            bettor_id = bet_coefs.ALTER_BETTOR
        else:
            bettor_id = bet_coefs.DEFAULT_BETTOR
        db_offers = bet_coefs.db_offers(sex, tour_date, bettor_id)
        predicate = (
            lambda o: o.tour_id == tour_id
            and o.rnd == self.rnd
            and (
                (
                    o.first_player_id == self.first_player.ident
                    and o.second_player_id == self.second_player.ident
                )
                or (
                    o.first_player_id == self.second_player.ident
                    and o.second_player_id == self.first_player.ident
                )
            )
        )
        if self.rnd is None:
            predicate = lambda o: o.tour_id == tour_id and (
                (
                    o.first_player_id == self.first_player.ident
                    and o.second_player_id == self.second_player.ident
                )
                or (
                    o.first_player_id == self.second_player.ident
                    and o.second_player_id == self.first_player.ident
                )
            )
        db_offer = co.find_first(db_offers, predicate)
        if db_offer is not None:
            if self.rnd is None and db_offer.rnd is not None:
                self.rnd = db_offer.rnd
            self.offer.company = db_offer.company
            if db_offer.win_coefs:
                self.offer.win_coefs = db_offer.win_coefs
            if db_offer.total_coefs:
                self.offer.total_coefs = db_offer.total_coefs
            if db_offer.sets_coefs:
                self.offer.sets_coefs = db_offer.sets_coefs
            if db_offer.handicap_coefs:
                self.offer.handicap_coefs = db_offer.handicap_coefs
            if (
                db_offer.first_player_id == self.second_player.ident
                and db_offer.second_player_id == self.first_player.ident
            ):
                self.offer.flip()
            return True  # ok
        elif alter_bettor is False:  # it was try1 failed
            if bet_coefs.ALL_BETTORS:
                return self.fill_offer(sex, tour_id, tour_date, alter_bettor=True)
            else:
                log.warn("bet_coefs.ALL_BETTORS=False with try2 in {}".format(self))
                return False

    def same_players_by_ident(self, other_match):
        return (
            self.first_player.ident is not None
            and self.second_player.ident is not None
            and self.first_player.ident
            in (other_match.first_player.ident, other_match.second_player.ident)
            and self.second_player.ident
            in (other_match.first_player.ident, other_match.second_player.ident)
        )

    def same_players_by_name(self, other_match):
        return (
            self.first_player.name is not None
            and self.second_player.name is not None
            and self.first_player.name
            in (other_match.first_player.name, other_match.second_player.name)
            and self.second_player.name
            in (other_match.first_player.name, other_match.second_player.name)
        )

    def read_draw_statuses_from_db(self, sex, tour_id):
        def seed_sql(player_id):
            return """select SEEDING
                      from Seed_{0}
                      where ID_T_S = {1}
                        and ID_P_S = {2};""".format(
                sex, tour_id, player_id
            )

        def fetch_draw_status(player_id):
            cursor.execute(seed_sql(player_id))
            row = cursor.fetchone()
            if row is not None and len(row) == 1:
                seed = row[0].strip().lower()
                if "q" in seed or "ll" in seed:
                    seed = "qual"
                return seed

        def first_rnd_bye_sql(player_id):
            """we supposed today mode is sufficient"""
            return """SELECT today.TOUR 
                      FROM today_{0} AS today 
                      WHERE today.TOUR = {1} and
                        today.ROUND = {2} and
                        today.ID1 = {3} and
                        today.RESULT = 'bye';""".format(
                sex, tour_id, Round("First").oncourt_id(), player_id
            )

        def fetch_first_rnd_bye(player_id):
            cursor.execute(first_rnd_bye_sql(player_id))
            return cursor.fetchone() is not None

        if self.rnd is None or self.rnd.qualification():
            return
        with closing(dba.get_connect().cursor()) as cursor:
            self.first_draw_status = fetch_draw_status(self.first_player.ident)
            self.second_draw_status = fetch_draw_status(self.second_player.ident)
            if self.rnd == "Second":
                if self.first_draw_status != "qual":
                    if fetch_first_rnd_bye(self.first_player.ident):
                        self.first_draw_status = "bye"
                    if self.second_draw_status != "qual":
                        if fetch_first_rnd_bye(self.second_player.ident):
                            self.second_draw_status = "bye"

    def incomer_side(self):
        """opponent of qual at First, or bye at Second"""
        if self.first_draw_status is None or self.second_draw_status is None:
            return None

        if self.rnd == "First":
            if self.first_draw_status == "qual" and self.second_draw_status != "qual":
                return co.RIGHT
            if self.second_draw_status == "qual" and self.first_draw_status != "qual":
                return co.LEFT

        if self.rnd == "Second":
            if self.first_draw_status == "bye" and self.second_draw_status != "bye":
                return co.LEFT
            if self.second_draw_status == "bye" and self.first_draw_status != "bye":
                return co.RIGHT


class Player(object):
    def __init__(
        self,
        ident=None,
        name=None,
        cou=None,
        birth_date=None,
        lefty: Optional[bool] = None,
        disp_names=None,
    ):
        self.ident = ident
        self.name = co.to_ascii(name.strip()) if name is not None else None
        self.cou = co.to_ascii(cou.strip()) if cou else None
        self.rating = ratings.Rating()
        self.birth_date = birth_date
        self.lefty: Optional[bool] = lefty
        self.disp_names = disp_names

    def __str__(self):
        rank = self.rating.rank("std")
        return "{}{} ({}){}{}".format(
            self.name,
            "" if self.birth_date is None else " " + str(self.age()) + "y",
            self.cou,
            "" if rank is None else " " + str(rank),
            "" if not self.lefty else " lefty",
        )

    def __hash__(self):
        return hash((self.ident, self.name, self.cou))

    def __eq__(self, other):
        if other is None:
            return False
        if self.ident is not None and other.ident is not None:
            return self.ident == other.ident
        else:
            return self.name == other.name and self.cou == other.cou

    def __ne__(self, other):
        return not self.__eq__(other)

    def unknown(self):
        """unknown player used in oncourt as player versus bye player"""
        return self.ident == 3700

    def disp_name(self, key, default=None):
        if self.disp_names:
            return self.disp_names.get(key, default)
        return default

    def age(self, ondate=None):
        if self.birth_date is None:
            return None
        todate = ondate or datetime.date.today()
        delta = todate - self.birth_date
        return round(float(delta.days) / 365.0, 1)

    def read_rating(self, sex, date=None, surfaces=("all",)):
        if not self.ident:
            return
        if date is None:
            rating_date = datetime.date.today()
            if rating_date.isoweekday() == 1:
                # в понед-к (по крайней мере утром) oncourt еще не иммеет свежие рейтинги
                rating_date = rating_date - datetime.timedelta(days=7)
            else:
                rating_date = tt.past_monday_date(rating_date)
        else:
            rating_date = date
        self.rating.read(sex, self.ident, surfaces=surfaces, date=rating_date)

    def read_pers_det(self, sex):
        if self.ident is None:
            return
        player = co.find_first(
            oncourt_players.players(sex), lambda p: p.ident == self.ident
        )
        if player is not None:
            self.lefty = player.lefty

    def read_birth_date(self, sex):
        if not self.ident or self.birth_date is not None:
            return
        sql = """SELECT DATE_P
                 FROM Players_{}
                 WHERE ID_P = {};""".format(
            sex, self.ident
        )
        with closing(dba.get_connect().cursor()) as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
        if row:
            birth = row[0]
            if birth:
                self.birth_date = birth.date() if birth is not None else None


class Surface(object):
    def __init__(self, name):
        name_low = co.to_ascii(name.strip()).lower()
        if ("carpet" in name_low and "outdoor" not in name_low) or (
            "hard" in name_low and ("indoor" in name_low or name_low.startswith("i."))
        ):
            self.name = "Carpet"
        elif "grass" in name_low:
            self.name = "Grass"
        elif "clay" in name_low:
            self.name = "Clay"
        elif "hard" in name_low or ("carpet" in name_low and "outdoor" in name_low):
            self.name = "Hard"
        elif "acrylic" in name_low:
            self.name = "Acrylic"
        else:
            raise co.TennisSurfaceError("unexpected surface '{}'".format(name))

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Surface):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class SurfaceTest(unittest.TestCase):
    def test_cmp(self):
        self.assertTrue("Clay" == Surface("Clay"))
        self.assertTrue(Surface("Carpet") == "Carpet")


class Level(object):
    names = ("main", "chal", "team", "teamworld", "junior", "future", "masters", "gs")

    def __init__(self, name):
        assert name in Level.names, "unexpected level: '{}'".format(name)
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Level):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def gs(self):
        return self.name == "gs"

    @property
    def masters(self):
        return self.name == "masters"

    @property
    def main(self):
        return self.name == "main"

    @property
    def chal(self):
        return self.name == "chal"

    @property
    def junior(self):
        return self.name == "junior"

    @property
    def future(self):
        return self.name == "future"

    @property
    def team(self):
        return self.name == "team"

    @property
    def teamworld(self):
        return self.name == "teamworld"


def soft_level(level, rnd, qualification=None):
    """
    усредненный уровень

    >>> soft_level(level='masters', rnd=None, qualification=True)
    'main'
    >>> soft_level(level=Level('masters'), rnd=None, qualification=True)
    'main'
    >>> soft_level(level='masters', rnd=Round('q-First'), qualification=None)
    'main'
    >>> soft_level(level=Level('main'), rnd=Round('q-First'))
    'qual'
    >>> soft_level(level=Level('gs'), rnd=Round('q-First'))
    'qual'
    >>> soft_level(level='teamworld', rnd=None, qualification=True)
    'qual'
    >>> soft_level(level='team', rnd=None, qualification=True)
    'qual'
    >>> soft_level(level=Level('chal'), rnd=Round('q-First'))
    'chal'
    """
    qual = qualification
    if qual is None and rnd is not None:
        qual = rnd.qualification() or rnd.pre_qualification()

    if qual and level not in ("masters", "chal", "future"):
        result = "qual"
    elif level in ("masters", "gs", "teamworld"):
        result = "main"
    else:
        result = str(level)
    return result


class HeadToHead(object):
    """method flip need not here: if match.flip() will be called,
    then self.fst_player, self.snd_player will be swaped as references,
    hence direct() will return fliped value"""

    TIME_DISCOUNT_FACTOR = 0.8

    def __init__(self, sex, match, completed_only=False):
        self.tour_match_aset = []
        self.fst_player = match.first_player
        self.snd_player = match.second_player
        self.date = match.date
        self._make(sex, match, completed_only)

    def __str__(self):
        return co.to_align_text(
            [
                (t, m.rnd, m.first_player.name, m.score)
                for t, m, _ in self.tour_match_aset
            ]
        )

    def __getitem__(self, index):
        return self.tour_match_aset[index]

    def __bool__(self):
        return bool(self.tour_match_aset)

    __nonzero__ = __bool__

    def __len__(self):
        return len(self.tour_match_aset)

    def __iter__(self):
        return iter(self.tour_match_aset)

    def __reversed__(self):
        return reversed(self.tour_match_aset)

    def direct(self, default_value=None):
        """DIRECT = H2H(1, 2) - H2H(2, 1)
        H2H(i, j) = percentage of matches won by player i against player j
        So, 1 - all dominate of first; -1 - all dominate of second;
            0 - equal ballance or no matches at all
        If empty H2H then return default_value
        """

        def time_discount_coef(tour_date):
            if tour_date is None or self.date is None:
                return 1.0  # no discount
            delta = self.date - tour_date
            delta_years = float(delta.days) / 365.0
            return min(
                self.TIME_DISCOUNT_FACTOR, self.TIME_DISCOUNT_FACTOR ** delta_years
            )

        def win_counts():
            fst_cnt, snd_cnt = 0, 0
            fst_sum, snd_sum = 0.0, 0.0
            for tour, match, _ in self.tour_match_aset:
                disc_coef = time_discount_coef(tour.date)
                if match.first_player == self.fst_player:
                    fst_cnt += 1
                    fst_sum += disc_coef
                elif match.first_player == self.snd_player:
                    snd_sum += disc_coef
                    snd_cnt += 1
            return fst_cnt, fst_sum, snd_cnt, snd_sum

        fst_cnt, fst_sum, snd_cnt, snd_sum = win_counts()
        if (fst_cnt, snd_cnt) == (0, 0):
            return default_value
        if snd_cnt == 0:
            fst_adv_ratio = min(1.0, fst_sum)
            return fst_adv_ratio
        elif fst_cnt == 0:
            snd_adv_ratio = min(1.0, snd_sum)
            return -snd_adv_ratio
        else:
            fst_adv_ratio = float(fst_sum) / float(fst_sum + snd_sum)
            return fst_adv_ratio - (1.0 - fst_adv_ratio)

    def _make(self, sex, match, completed_only):
        from tournament import Tournament

        if (
            not match.first_player
            or not match.second_player
            or not match.first_player.ident
            or not match.second_player.ident
            or not match.date
        ):
            return
        tour_date_max = tt.past_monday_date(match.date)
        sql = """select Rounds.NAME_R, Courts.NAME_C, tours.ID_T, tours.NAME_T, 
                        tours.RANK_T, tours.DATE_T, tours.PRIZE_T, tours.COUNTRY_T,
                        games.ID1_G, games.ID2_G, games.RESULT_G, games.DATE_G
                 from Tours_{0} AS tours, Games_{0} AS games, Rounds, Courts
                 where games.ID_T_G = tours.ID_T
                   and Rounds.ID_R = games.ID_R_G
                   and tours.ID_C_T = Courts.ID_C
                   and tours.DATE_T < {1}
                   and ((games.ID1_G = {2} and games.ID2_G = {3}) or
                        (games.ID2_G = {2} and games.ID1_G = {3}) )
                   and (tours.NAME_T Not Like '%juniors%')
                   ;""".format(
            sex,
            dba.msaccess_date(tour_date_max),
            match.first_player.ident,
            match.second_player.ident,
        )
        with closing(dba.get_connect().cursor()) as cursor:
            for (
                rnd_name,
                surf_name,
                tour_id,
                tour_name,
                tour_rank,
                tour_time,
                money,
                cou,
                p1st_id,
                p2nd_id,
                score_name,
                match_dtime,
            ) in cursor.execute(sql):
                score = sc.Score(score_name)
                if score.retired and completed_only:
                    continue
                tour = Tournament(
                    ident=tour_id,
                    name=tour_name,
                    sex=sex,
                    surface=Surface(surf_name),
                    rank=tour_rank,
                    date=tour_time.date() if tour_time else None,
                    money=money,
                    cou=cou,
                )
                if tour.level == "future":
                    continue
                match_date = match_dtime.date() if match_dtime is not None else None
                if (
                    p1st_id == match.first_player.ident
                    and p2nd_id == match.second_player.ident
                ):
                    m = Match(
                        match.first_player,
                        match.second_player,
                        score=score,
                        rnd=Round(rnd_name),
                        date=match_date,
                    )
                else:
                    m = Match(
                        match.second_player,
                        match.first_player,
                        score=score,
                        rnd=Round(rnd_name),
                        date=match_date,
                    )
                avgset = self._get_avgset(tour, m)
                self.tour_match_aset.append((tour, m, avgset))
        self.tour_match_aset.sort(key=lambda i: i[0].date, reverse=True)
        self._remove_current_match(match.date)

    def _remove_current_match(self, current_match_date):
        """
        актуально для больших шлемов, т.к. условие отбора по времени в методе _make
        говорит что надо брать турниры прошлой недели и так далее в прошлое.
        Но для GrSlam на прошл. неделе мы можем найти этот же шлем (с current_match_date)
        """
        if len(self.tour_match_aset) > 0:
            if (
                current_match_date is not None
                and self.tour_match_aset[0][1].date == current_match_date
            ):
                del self.tour_match_aset[0]

    @staticmethod
    def _get_avgset(tour, match):
        """here tour does not contain matches"""
        import total

        return total.get_tour_local_avgset(
            tour, match.rnd, mode=total.LocalAvgsetMode.curyear_firstly, min_size=15
        )

    def total_target_evaled(self, target_best_of_five, target_avgset):
        import total

        sumer = st.Sumator()
        sets_sumer = st.Sumator()
        for tour, match, aset in self.tour_match_aset:
            if aset is not None:
                avgset_coef = target_avgset / aset
            else:
                year = max(2005, tour.date.year)
                aset = total.generic_avgset(
                    tour.sex,
                    tour.surface,
                    tour.level == "chal",
                    match.rnd.qualification(),
                    min(2017, year),
                )
                if aset is None:
                    log.error(
                        "{} H2H {}-{} fail avgset at t-date{}".format(
                            tour.sex, self.fst_player, self.snd_player, tour.date
                        )
                    )
                    avgset_coef = 1.0
                else:
                    avgset_coef = target_avgset / aset
            total_value = match.score.normalized_total(treat_bo5_as_bo3=False)
            sets_value = match.score.sets_count()
            best_of_five = match.score.best_of_five()
            if (best_of_five, target_best_of_five) == (False, True):
                total_value = total.normbo3_to_normbo5(total_value)
                sets_value = total.setsbo3_to_setsbo5(sets_value)
            elif (best_of_five, target_best_of_five) == (True, False):
                total_value = total.normbo5_to_normbo3(total_value)
                sets_value = total.setsbo5_to_setsbo3(sets_value)
            sumer.hit(total_value * avgset_coef)
            sets_sumer.hit(sets_value * avgset_coef)
        return sumer.average(), sets_sumer.average()

    def recently_won_player_id(self, max_history_days=25):
        if not self.tour_match_aset:
            return None
        match = self.tour_match_aset[0][1]
        if match.date:
            now_date = datetime.date.today()
            if (now_date - match.date) <= datetime.timedelta(days=max_history_days):
                return match.first_player.ident


class HeadToHeadTest(unittest.TestCase):
    def test_ito_kukushkin(self):
        plr1 = Player(ident=8313, name="Tatsuma Ito", cou="JPN")
        plr2 = Player(ident=9043, name="Mikhail Kukushkin", cou="KAZ")
        match = Match(
            first_player=plr1,
            second_player=plr2,
            score=sc.Score("4-6 7-6(3) 7-6(6)"),
            rnd=Round("Second"),
            date=datetime.date(2017, 2, 1),
        )
        h2hdir = HeadToHead(sex="atp", match=match, completed_only=True).direct()
        self.assertEqual(h2hdir, None)

    # def test_recent_winner(self):
    #     """ actual only when date was 2019,10,18 and recent h2h match was 2019,9,26 """
    #     plr1 = Player(ident=6590, name="Cagla Buyukakcay", cou="TUR")
    #     plr2 = Player(ident=8951, name="Veronica Cepede Royg", cou="PAR")
    #     match = Match(first_player=plr1, second_player=plr2,
    #                   score=None, rnd=Round('Second'),
    #                   date=datetime.date(2019, 10, 18))
    #     h2h = HeadToHead(sex='wta', match=match, completed_only=True)
    #     recent_winner_id = h2h.recently_won_player_id()
    #     self.assertEqual(recent_winner_id, plr1.ident)


if __name__ == "__main__":
    import doctest

    log.initialize(co.logname(__file__, test=True), "debug", None)
    dba.open_connect()
    doctest.testmod()
    unittest.main()
    dba.close_connect()
