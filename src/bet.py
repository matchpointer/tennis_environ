from typing import NamedTuple

from loguru import logger as log
import common as co
import score as sc
import stat_cont as st


class Company(NamedTuple):
    ident: int
    name: str


companies = [
    Company(ident=1, name="Mar"),
    Company(ident=2, name="Pin"),
    Company(ident=99999, name="BC"),
]


def get_marathon_company():
    return co.find_first(companies, lambda c: c.name == "Mar")


def get_pinnacle_company():
    return co.find_first(companies, lambda c: c.name == "Pin")


def get_betcity_company():
    return co.find_first(companies, lambda c: c.name == "BC")


def get_company_by_id(ident):
    return co.find_first(companies, lambda c: c.ident == ident)


def chance_to_book_coef(chance, book_side_margin):
    return (1.0 - book_side_margin) / chance


class TotalCoefs(object):
    def __init__(self, line_value=None, less_coef=None, greater_coef=None):
        def bo5_to_bo3(total_value):
            result = None
            if (
                25 < total_value <= 32
            ):  # f5: [18, 32] -> [12, 21]. f5(x) = (9/14)x + 6/14
                result = float(round((9 * total_value + 6) / 14.0))
            elif 33 < total_value:  # g5: [33, 53] -> [22, 27]. g5(x) = (1/4)x + 55/4
                result = float(round((total_value + 55) / 4.0))
            if 23.1 < result:
                result = 23.5
            elif 22.1 < result:
                result = 22.5
            else:
                result = 21.5
            return result

        self.line_value = line_value  # if line_value <= 25 else bo5_to_bo3(line_value)
        self.less_coef = less_coef
        self.greater_coef = greater_coef

    def __bool__(self):
        return (
            self.line_value is not None
            and self.line_value > 13.0
            and self.less_coef is not None
            and self.greater_coef is not None
        )

    __nonzero__ = __bool__

    def __str__(self):
        return "total {} {}/{}".format(
            self.line_value, round(self.less_coef, 2), round(self.greater_coef, 2)
        )

    def integer_line(self):
        return (
            self.line_value is not None
            and abs(float(round(self.line_value, 0)) - self.line_value) < 0.1
        )

    def best_of_five(self):
        return self.line_value is not None and self.line_value > 26.0


class TotalSetsCoefs(object):
    def __init__(self, line_value=None, less_coef=None, greater_coef=None):
        self.line_value = line_value
        self.less_coef = less_coef
        self.greater_coef = greater_coef

    def __bool__(self):
        return (
            self.line_value is not None
            and self.less_coef is not None
            and self.greater_coef is not None
        )

    __nonzero__ = __bool__

    def __str__(self):
        return "total-sets {} {}/{}".format(
            round(self.line_value, 1),
            round(self.less_coef, 2),
            round(self.greater_coef, 2),
        )


class WinCoefs(object):
    def __init__(self, first_coef=None, second_coef=None):
        self.first_coef = first_coef
        self.second_coef = second_coef

    def __bool__(self):
        return (
            self.first_coef is not None
            and self.first_coef > 1
            and self.second_coef is not None
            and self.second_coef > 1
        )

    __nonzero__ = __bool__

    def __str__(self):
        return "win: {}/{}".format(self.first_coef, self.second_coef)

    def flip(self):
        self.first_coef, self.second_coef = self.second_coef, self.first_coef

    def chances(self):
        if self.first_coef is None or self.second_coef is None:
            return None, None
        prob_first = 1.0 / self.first_coef
        prob_second = 1.0 / self.second_coef
        margin = (prob_first + prob_second) - 1.0
        return (
            prob_first - (margin * prob_first) / (prob_first + prob_second),
            prob_second - (margin * prob_second) / (prob_first + prob_second),
        )

    def bookmaker_margin(self):
        if self.first_coef is None or self.second_coef is None:
            return None
        prob_first = 1.0 / self.first_coef
        prob_second = 1.0 / self.second_coef
        return (prob_first + prob_second) - 1.0


class HandicapCoefs(object):
    def __init__(
        self, first_value=None, first_coef=None, second_value=None, second_coef=None
    ):
        self.first_value = first_value
        self.first_coef = first_coef
        self.second_value = second_value
        self.second_coef = second_coef

    def __bool__(self):
        return (
            self.first_value is not None
            and self.first_coef is not None
            and self.second_value is not None
            and self.second_coef is not None
        )

    __nonzero__ = __bool__

    def __str__(self):
        return "handicap: {}/{} {}/{}".format(
            self.first_value, self.first_coef, self.second_value, self.second_coef
        )

    def flip(self):
        self.first_coef, self.second_coef = self.second_coef, self.first_coef
        self.first_value, self.second_value = self.second_value, self.first_value


class SetsCoefs(object):
    def __init__(self, scoreset_dict=None):
        """can be inited from dict([((2,0), 1.9), ((2,1), 2.8),
        ((1,2), 4.5), ((0,2), 6.7)])
        """
        self.coef_from_scoreset = {}
        if scoreset_dict:
            assert (
                len(scoreset_dict) >= 4
            ), "not enough args when init SetsCoefs {}".format(scoreset_dict)
            for scoreset_coef in scoreset_dict.items():
                assert (
                    len(scoreset_coef) == 2
                ), "not pair encountered {} when init SetsCoefs".format(scoreset_coef)
                self.coef_from_scoreset[scoreset_coef[0]] = scoreset_coef[1]

    def max_sets_number(self):
        sets_scores = list(self.coef_from_scoreset.keys())
        if len(sets_scores):
            return max(
                max([s[0] for s in sets_scores]), max([s[1] for s in sets_scores])
            )
        else:
            return 2

    def __str__(self):
        return "sets {}".format(self.coef_from_scoreset)

    def __getitem__(self, index):
        return self.coef_from_scoreset[index]

    def __setitem__(self, index, value):
        self.coef_from_scoreset[index] = value

    def __bool__(self):
        return len(self.coef_from_scoreset) > 0

    __nonzero__ = __bool__

    def __len__(self):
        return len(self.coef_from_scoreset)

    def __iter__(self):
        return iter(self.coef_from_scoreset)

    def __contains__(self, index):
        return index in self.coef_from_scoreset.keys()

    def items(self):
        return list(self.coef_from_scoreset.items())

    def keys(self):
        return list(self.coef_from_scoreset.keys())

    def values(self):
        return list(self.coef_from_scoreset.values())

    def flip(self):
        if (2, 0) in self.coef_from_scoreset.keys():
            val = self.coef_from_scoreset[(2, 0)]
            self.coef_from_scoreset[(2, 0)] = self.coef_from_scoreset[(0, 2)]
            self.coef_from_scoreset[(0, 2)] = val
        if (2, 1) in self.coef_from_scoreset.keys():
            val = self.coef_from_scoreset[(2, 1)]
            self.coef_from_scoreset[(2, 1)] = self.coef_from_scoreset[(1, 2)]
            self.coef_from_scoreset[(1, 2)] = val


def create_sets_coefs(coefs):
    if not all(coefs):
        return None
    if len(coefs) == 4:
        d = dict(
            [
                ((2, 0), float(coefs[0])),
                ((2, 1), float(coefs[1])),
                ((1, 2), float(coefs[2])),
                ((0, 2), float(coefs[3])),
            ]
        )
        return SetsCoefs(d)
    elif len(coefs) == 6:
        d = dict(
            [
                ((3, 0), float(coefs[0])),
                ((3, 1), float(coefs[1])),
                ((3, 2), float(coefs[2])),
                ((2, 3), float(coefs[3])),
                ((1, 3), float(coefs[4])),
                ((0, 3), float(coefs[5])),
            ]
        )
        return SetsCoefs(d)
    raise co.TennisError("can not create SetsCoefs from '{}'".format(coefs))


class Offer(object):
    def __init__(self, company):
        self.company = company
        self.win_coefs = None
        self.total_coefs = None
        self.sets_coefs = None
        self.handicap_coefs = None

    def __bool__(self):
        return (
            bool(self.win_coefs)
            or bool(self.total_coefs)
            or bool(self.sets_coefs)
            or bool(self.handicap_coefs)
        )

    __nonzero__ = __bool__

    def __str__(self):
        return "{} {} {} {} {}".format(
            self.company.name,
            self.total_coefs or "",
            self.win_coefs or "",
            self.sets_coefs or "",
            ("\n\t" + str(self.handicap_coefs)) if self.handicap_coefs else "",
        )

    def flip(self):
        if self.win_coefs:
            self.win_coefs.flip()
        if self.sets_coefs:
            self.sets_coefs.flip()
        if self.handicap_coefs:
            self.handicap_coefs.flip()

    def insert_sql(self, sex, first_id, second_id, tour_id, rnd_id):
        def attr_val(obj, attrname):
            if obj is None:
                return "null"
            return getattr(obj, attrname)

        def key_val(obj, dictname, key):
            if obj is None:
                return "null"
            dictionary = getattr(obj, dictname)
            return dictionary[key] if key in dictionary else "null"

        return """insert into Odds_{} (ID_B_O, ID1_O, ID2_O, ID_T_O, ID_R_O, 
                          K1, K2, TOTAL, KTM, KTB, F1, F2, KF1, KF2, 
                          K20, K21, K12, K02,
                          K30, K31, K32, K23, K13, K03)
                  values ({}, {}, {}, {}, {}, 
                          {}, {}, {}, {}, {}, {}, {}, {}, {}, 
                          {}, {}, {}, {}, 
                          {}, {}, {}, {}, {}, {});""".format(
            sex,
            self.company.ident,
            first_id,
            second_id,
            tour_id,
            rnd_id,
            attr_val(self.win_coefs, "first_coef"),
            attr_val(self.win_coefs, "second_coef"),
            attr_val(self.total_coefs, "line_value"),
            attr_val(self.total_coefs, "less_coef"),
            attr_val(self.total_coefs, "greater_coef"),
            attr_val(self.handicap_coefs, "first_value"),
            attr_val(self.handicap_coefs, "second_value"),
            attr_val(self.handicap_coefs, "first_coef"),
            attr_val(self.handicap_coefs, "second_coef"),
            key_val(self.sets_coefs, "coef_from_scoreset", (2, 0)),
            key_val(self.sets_coefs, "coef_from_scoreset", (2, 1)),
            key_val(self.sets_coefs, "coef_from_scoreset", (1, 2)),
            key_val(self.sets_coefs, "coef_from_scoreset", (0, 2)),
            key_val(self.sets_coefs, "coef_from_scoreset", (3, 0)),
            key_val(self.sets_coefs, "coef_from_scoreset", (3, 1)),
            key_val(self.sets_coefs, "coef_from_scoreset", (3, 2)),
            key_val(self.sets_coefs, "coef_from_scoreset", (2, 3)),
            key_val(self.sets_coefs, "coef_from_scoreset", (1, 3)),
            key_val(self.sets_coefs, "coef_from_scoreset", (0, 3)),
        )

    def best_of_five(self):
        if self.sets_coefs:
            return self.sets_coefs.max_sets_number() > 2
        if self.total_coefs:
            return self.total_coefs.best_of_five()


class DbOffer(Offer):
    def __init__(self, company, tour_id, rnd, first_player_id, second_player_id):
        Offer.__init__(self, company)
        self.tour_id = tour_id
        self.rnd = rnd
        self.first_player_id = first_player_id
        self.second_player_id = second_player_id

    def __eq__(self, other):
        return (self.tour_id == other.tour_id and self.rnd == other.rnd) and (
            (
                self.first_player_id == other.first_player_id
                and self.second_player_id == other.second_player_id
            )
            or (
                self.first_player_id == other.second_player_id
                and self.second_player_id == other.first_player_id
            )
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class Pick(object):
    def __init__(
        self,
        company,
        coef,
        value=None,
        margin=None,
        probability=None,
        explain=None,
        money=None,
        is_live=False,
    ):
        self.company = company
        self.coef = coef
        self.value = value
        self.margin = margin
        self.probability = probability
        self.explain = explain
        self.money = money
        self.is_live = is_live

    def __str__(self):
        return "{} {} coef {} mrg {} prob {}".format(
            self.company.name,
            "-" if self.value is None else self.value,
            self.coef,
            "-" if self.margin is None else "{0:.2f}".format(self.margin),
            "-" if self.probability is None else "{0:.2f}".format(self.probability),
        )

    def to_str(self, match):
        raise co.TennisError("forgoten implementation of to_str")


class PickTotalSets(Pick):
    def __init__(
        self, company, coef, value, is_less, margin=None, probability=None, explain=None
    ):
        Pick.__init__(self, company, coef, value, margin, probability, explain)
        self.is_less = is_less

    def to_str(self, match):
        return "{} {} {}".format(
            self.__class__.__name__, "less" if self.is_less else "greater", str(self)
        )


class PickTotal(Pick):
    def __init__(
        self,
        company,
        coef,
        value,
        is_less,
        margin=None,
        probability=None,
        explain=None,
        money=None,
    ):
        Pick.__init__(self, company, coef, value, margin, probability, explain, money)
        self.is_less = is_less

    def to_str(self, match):
        return "{} {} {}".format(
            self.__class__.__name__, "less" if self.is_less else "greater", str(self)
        )

    def result(self, match):
        """return: True if win, False if loss, None if bet returned"""
        games_sum = match.score.games_count()
        if games_sum == 0:
            return None
        cmpr = co.cmp(games_sum, self.value)
        if not match.score.retired:
            if cmpr == 0:
                return None
            elif cmpr == -1:
                return self.is_less
            else:
                return not self.is_less
        if cmpr >= 0:  # games_sum >= value
            return not self.is_less
        full_score = sc.completed_score(match.score, best_of_five=self.value > 25.6)
        games_sum = full_score.games_count()
        cmpr = co.cmp(games_sum, self.value)
        if cmpr <= 0:  # games_sum <= value
            return None
        else:
            return not self.is_less


class PickWin(Pick):
    """
    родительский атрибут value не используется
    """

    def __init__(
        self,
        company,
        coef,
        sider_id,
        margin=None,
        probability=None,
        explain=None,
        money=None,
    ):
        Pick.__init__(self, company, coef, None, margin, probability, explain, money)
        self.sider_id = sider_id

    def to_str(self, match):
        if match.first_player.ident == self.sider_id:
            winner = match.first_player.name
        else:
            winner = match.second_player.name
        return "{} {} {}".format(self.__class__.__name__, winner, str(self))

    def result(self, match):
        """
        return: True if win, False if loss, None if bet returned
        """
        if match.score.sets_count(full=True) >= 1:
            (win_sets, loss_sets) = match.score.sets_score(full=True)
            if win_sets > loss_sets:
                # first player is match winner
                return match.first_player.ident == self.sider_id
            elif win_sets < loss_sets:
                # second player is match winner
                return match.second_player.ident == self.sider_id


class PickHandicap(Pick):
    def __init__(
        self,
        company,
        coef,
        value,
        sider_id,
        margin=None,
        probability=None,
        explain=None,
        money=None,
    ):
        Pick.__init__(self, company, coef, value, margin, probability, explain, money)
        self.sider_id = sider_id

    def to_str(self, match):
        sider = (
            match.first_player.name
            if match.first_player.ident == self.sider_id
            else match.second_player.name
        )
        return "{} {} {}".format(self.__class__.__name__, sider, str(self))

    def result(self, match):
        """
        return: True if win, False if loss, None if bet returned
        """
        if not match.score.retired:
            advantage = match.score.games_advantage()
            if match.second_player.ident == self.sider_id:
                advantage = -advantage
            if -advantage == self.value:
                return None
            return -advantage < self.value


class PickSetsScore(Pick):
    def __init__(
        self,
        company,
        coef,
        value,
        sider_id,
        margin=None,
        probability=None,
        explain=None,
        money=None,
    ):
        Pick.__init__(self, company, coef, value, margin, probability, explain, money)
        self.sider_id = sider_id

    def to_str(self, match):
        sider = match.first_player.name
        if self.sider_id == match.second_player.ident:
            sider = match.second_player.name
        return "{} {} {}".format(self.__class__.__name__, sider, str(self))

    def result(self, match):
        """return: True if win, False if loss, None if bet returned"""
        win_sets, loss_sets = match.score.sets_score(full=True)
        if self.sider_id == match.second_player.ident:
            win_sets, loss_sets = loss_sets, win_sets
        if not match.score.retired:
            assert max(win_sets, loss_sets) == max(
                self.value[0], self.value[1]
            ), "invalid score bet {} for match score {}".format(
                self.value, (win_sets, loss_sets)
            )
            return self.value == (win_sets, loss_sets)
        # незавершеный матч - два рез-та: проигрыш или возврат
        if self.value is not None:
            bet_win_sets, bet_loss_sets = self.value
            if (win_sets > bet_win_sets) or (loss_sets > bet_loss_sets):
                return False


def exist_conflict_picks(picks):
    size = len(picks)
    if size < 2:
        return False
    for i in range(size - 1):
        for j in range(i + 1, size):
            if (
                isinstance(picks[i], PickWin)
                and isinstance(picks[j], PickWin)
                and picks[i].sider_id != picks[j].sider_id
            ):
                return True
            if isinstance(picks[i], PickSetsScore) and isinstance(
                picks[j], PickSetsScore
            ):
                win_i, loss_i = picks[i].value
                win_j, loss_j = picks[j].value
                if (
                    picks[i].sider_id == picks[j].sider_id
                    and (co.cmp(win_i, loss_i) * co.cmp(win_j, loss_j)) < 0
                ):
                    return True
                if (
                    picks[i].sider_id != picks[j].sider_id
                    and (co.cmp(win_i, loss_i) * co.cmp(win_j, loss_j)) > 0
                ):
                    return True
    return False


class Sager:
    """Отслеживает максимальную денежную просадку (sag)
    и макс. значение банка (max_money)
    """

    def __init__(self, start_money):
        self.sag = 0.0
        self.max_money = start_money

    def set_current_money(self, current_money):
        if current_money > self.max_money:
            self.max_money = current_money
        else:
            sag_current = self.max_money - current_money
            if sag_current > self.sag:
                self.sag = sag_current

    def __repr__(self):
        return "$sag:{0:.1f}".format(self.sag)


class FlatProfiter(object):
    money_betsize = 1.0

    def __init__(self):
        self.money_delta = 0.0
        self.win_loss = st.WinLoss()

    def __repr__(self):
        return "roi:{0} $dif:{1:.2f} WL:{2}".format(
            self.roi_ratio() or "", self.money_delta, self.win_loss
        )

    @property
    def size(self):
        return self.win_loss.size

    def calculate_bet(self, coef, iswin):
        self.win_loss.hit(iswin)
        if iswin:
            self.money_delta += self.money_betsize * (coef - 1.0)
        else:
            self.money_delta -= self.money_betsize

    def roi_ratio(self):
        money_in = self.win_loss.size * self.money_betsize
        money_out = self.money_delta
        if money_in > 0:
            return round(money_out / money_in, 3)


class FlatLimitedProfiter(FlatProfiter):
    def __init__(self, start_money=50):
        super(FlatLimitedProfiter, self).__init__()
        self.start_money = start_money
        self.sager = Sager(float(start_money))

    def bankrot(self):
        return self.current_money() < self.money_betsize

    def current_money(self):
        return self.start_money + self.money_delta

    def calculate_bet(self, coef, iswin):
        if not self.bankrot():
            super(FlatLimitedProfiter, self).calculate_bet(coef, iswin)
            self.sager.set_current_money(self.current_money())

    def __repr__(self):
        result = super(FlatLimitedProfiter, self).__repr__()
        return result + " " + str(self.sager)

    def __lt__(self, other):
        return self.roi_ratio() >= other.roi_ratio()


def show_best_traders(traders, top=3):
    sorted_traders = sorted(traders)
    top_size = min(top, len(traders))
    print("---------- top %d----------" % top_size)
    for i in range(top_size):
        print("%s" % sorted_traders[i])


def write_traders_to_file(sex, traders):
    filename = "./{}_race_traders.txt".format(sex)
    with open(filename, "w") as fhandle:
        for trader in sorted(traders):
            fhandle.write("{}\n".format(trader))


def traders_reporting(traders, dirname):
    log.info("start traders_reporting at " + dirname)
    for trader in traders:
        trader.reporting(dirname)
    log.info("finish traders_reporting")
