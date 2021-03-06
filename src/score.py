# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import copy
import itertools
from enum import Enum
from typing import Tuple, List, Optional, TYPE_CHECKING

from side import Side
import lev
import common as co
from loguru import logger as log
import feature

if TYPE_CHECKING:
    from tennis import Round


class Winsetway(Enum):
    serve = 1
    receive = 2
    tie_opener = 3
    tie_not_opener = 4


def exist_point_increment(
    prev_tuple: Tuple[int, int], next_tuple: Tuple[int, int]
) -> bool:
    """input arguments are simple 2-tuples"""
    if (prev_tuple[0] + 1) == next_tuple[0] and prev_tuple[1] == next_tuple[1]:
        return True
    if (prev_tuple[1] + 1) == next_tuple[1] and prev_tuple[0] == next_tuple[0]:
        return True
    return False


def winsetway(det_score, setnum):
    """вернем пару (isleftwinner, winsetway) или None если неясно"""

    def check_next_set_begin(setnum_last_item):
        set_items = list(det_score.set_items(setnum + 1))
        if set_items:
            fst_key, _ = set_items[0]
            if fst_key[-1] != (0, 0):
                log.error(
                    "next set must beginwith00 for setn {} in {}".format(
                        setnum, det_score
                    )
                )
                return
            prev_key, prev_dgame = setnum_last_item
            if len(fst_key) != (1 + len(prev_key)):
                log.error(
                    "next set must be wider for setn {} in {}".format(setnum, det_score)
                )
            set_prev, set_prev_full = prev_key[-1], fst_key[-2]
            if not exist_point_increment(set_prev, set_prev_full):
                log.error(
                    "next set must be 1increm for setn {} in {}".format(
                        setnum, det_score
                    )
                )
            left_winprevset = set_prev_full[0] > set_prev_full[1]
            if prev_dgame.left_wingame is not left_winprevset:
                log.error(
                    "next set bad winner for setn {} in {}".format(setnum, det_score)
                )

    set_items = list(det_score.set_items(setnum))
    if set_items:
        last_key, last_dgame = set_items[-1]
        if last_dgame.tiebreak:
            assert last_key[-1] == (
                6,
                6,
            ), "invalidtie at setnum {} in dscore {}".format(setnum, det_score)
            wsetway = (
                Winsetway.tie_opener
                if last_dgame.opener_wingame
                else Winsetway.tie_not_opener
            )
        else:
            wsetway = (
                Winsetway.serve if last_dgame.opener_wingame else Winsetway.receive
            )
        check_next_set_begin((last_key, last_dgame))
        return last_dgame.left_wingame, wsetway


Event = Enum("Event", "match_begin set_begin changeover")


def at_keyscore_event(keyscore):
    """keyscore - счет в стиле detailed_score (т.е. перед розыгрышем).
        если перед розыгрышем есть событие Event, то вернем это событие

    >>> at_keyscore_event(((0, 0),)).name
    'match_begin'
    >>> at_keyscore_event(((1, 0),)) is None
    True
    >>> at_keyscore_event(((0, 2),)) is None
    True
    >>> at_keyscore_event(((2, 1),)).name
    'changeover'
    """
    if not keyscore:
        return None
    if keyscore[-1] == (0, 0):
        return Event.match_begin if len(keyscore) == 1 else Event.set_begin
    curset_sum = sum(keyscore[-1])
    if curset_sum > 1 and (curset_sum % 2) == 1:
        return Event.changeover


def keyscore_fliped(keyscore):
    """keyscore - счет в стиле detailed_score (т.е. перед розыгрышем).
    вернем счет где левое и правое поменялось местами"""
    return tuple((tuple(reversed(s)) for s in keyscore))


def get_curset_opener(cursetnum, prevset_opener: Side, all_score):
    """
    here all_score can be also Score object

    >>> get_curset_opener(cursetnum=2, prevset_opener=co.LEFT, all_score=((6, 2), (6, 3)))
    Side('LEFT')
    >>> get_curset_opener(cursetnum=2, prevset_opener=co.LEFT, all_score=((6, 1), (6, 3)))
    Side('RIGHT')
    """
    assert 1 <= cursetnum <= 5
    assert prevset_opener in (co.LEFT, co.RIGHT)
    prevscr = all_score[cursetnum - 2]
    prevset_lastg_opener = (
        prevset_opener if sum(prevscr) % 2 == 1 else prevset_opener.fliped()
    )
    return prevset_lastg_opener.fliped()


def get_curset_opener_by_next(cursetnum, nextset_opener: Side, all_score):
    """
    here all_score can be also Score object

    >>> get_curset_opener_by_next(1, nextset_opener=co.LEFT, all_score=((6, 2), (6, 3)))
    Side('LEFT')
    >>> get_curset_opener_by_next(1, nextset_opener=co.LEFT, all_score=((6, 1), (6, 3)))
    Side('RIGHT')
    """
    curscr = all_score[cursetnum - 1]
    curset_lastg_opener = nextset_opener.fliped()
    curset_opener = (
        curset_lastg_opener if sum(curscr) % 2 == 1 else curset_lastg_opener.fliped()
    )
    return curset_opener


def serve_quadrant_at(num_ingame):
    """вернем serve quadrant, котор. будет играться при счете (x, y),
         где x,y - int с единичными приращениями

    >>> serve_quadrant_at((0, 0))
    'DEUCE'
    >>> serve_quadrant_at((1, 0))
    'ADV'
    >>> serve_quadrant_at((1, 1))
    'DEUCE'
    """
    return co.DEUCE if ((num_ingame[0] + num_ingame[1]) % 2) == 0 else co.ADV


class QuadrantPoint(object):
    def __init__(self, quadrant, srv_side: Side, srv_winpoint=None):
        self.quadrant = quadrant
        self.srv_side = srv_side
        self.srv_winpoint = srv_winpoint


def tie_quadrant_points(
    num_ingame_from, srv_side_from: Side, num_points, side_winpoint=None
):
    tie_open_side = tie_opener_side(num_ingame_from, srv_side_from)
    for _ in range(num_points):
        quadrant = serve_quadrant_at(num_ingame_from)
        srv_side = tie_serve_side_at(num_ingame_from, tie_open_side)
        if side_winpoint is None:
            srv_winpoint = None
        else:
            srv_winpoint = srv_side == side_winpoint
        yield QuadrantPoint(
            quadrant=quadrant, srv_side=srv_side, srv_winpoint=srv_winpoint
        )
        num_ingame_from = (num_ingame_from[0] + 1, num_ingame_from[1])  # emulate next


def tie_opener_side(num_ingame, srv_side: Side):
    """По текущему num_ingame (2-tuple) и текущему подающему вернем открывателя

    >>> tie_opener_side((0, 0), srv_side=co.LEFT)
    Side('LEFT')
    >>> tie_opener_side((0, 0), srv_side=co.RIGHT)
    Side('RIGHT')
    >>> tie_opener_side((1, 0), srv_side=co.LEFT)
    Side('RIGHT')
    >>> tie_opener_side((1, 1), srv_side=co.LEFT)
    Side('RIGHT')
    >>> tie_opener_side((1, 1), srv_side=co.RIGHT)
    Side('LEFT')
    """
    return srv_side if tie_opener_serve_at(num_ingame) else srv_side.fliped()


def tie_serve_side_at(num_ingame, tie_open_side: Side) -> Side:
    """По текущему num_ingame (2-tuple) и открывателю вернем текущего подающего

    >>> tie_serve_side_at((0, 0), tie_open_side=co.LEFT)
    Side('LEFT')
    >>> tie_serve_side_at((0, 1), tie_open_side=co.LEFT)
    Side('RIGHT')
    >>> tie_serve_side_at((1, 1), tie_open_side=co.LEFT)
    Side('RIGHT')
    >>> tie_serve_side_at((1, 1), tie_open_side=co.RIGHT)
    Side('LEFT')
    >>> tie_serve_side_at((1, 5), tie_open_side=co.LEFT)
    Side('RIGHT')
    >>> tie_serve_side_at((1, 6), tie_open_side=co.LEFT)
    Side('LEFT')
    """
    if tie_opener_serve_at(num_ingame):
        return tie_open_side
    else:
        return tie_open_side.fliped()


def get_tie_open_side(num_ingame, serve_side: Side) -> Optional[Side]:
    """По текущему num_ingame (2-tuple) и подающему вернем открывателя тайбрейка

    >>> get_tie_open_side((0, 0), serve_side=co.LEFT)
    Side('LEFT')
    >>> get_tie_open_side((0, 1), serve_side=co.LEFT)
    Side('RIGHT')
    >>> get_tie_open_side((1, 1), serve_side=co.LEFT)
    Side('RIGHT')
    >>> get_tie_open_side((1, 1), serve_side=co.RIGHT)
    Side('LEFT')
    >>> get_tie_open_side((1, 5), serve_side=co.LEFT)
    Side('RIGHT')
    >>> get_tie_open_side((1, 5), serve_side=co.RIGHT)
    Side('LEFT')
    """
    if serve_side is None:
        return None  # bad input, hence bad output
    summa = (num_ingame[0] + num_ingame[1]) % 4
    if summa in (0, 3):
        return serve_side
    else:
        return serve_side.fliped()


def tie_opener_serve_at(num_ingame):
    """подает ли открыватель тайбрейка при счете num_ingame (2-tuple)"""
    return ((num_ingame[0] + num_ingame[1]) % 4) in (0, 3)


def tie_serve_number_at(num_ingame) -> int:
    """По текущему num_ingame (2-tuple)
        вернем 1 если подающий начинает свою пару подач,
        вернем 2 если подающий заканчивает свою пару подач (далее смена)

    >>> tie_serve_number_at((0, 0))
    1
    >>> tie_serve_number_at((0, 1))
    1
    >>> tie_serve_number_at((1, 0))
    1
    >>> tie_serve_number_at((1, 1))
    2
    >>> tie_serve_number_at((2, 1))
    1
    >>> tie_serve_number_at((2, 2))
    2
    """
    summa = num_ingame[0] + num_ingame[1]
    if summa == 0:
        return 1
    elif (summa % 2) == 0:
        return 2
    else:
        return 1


def tie_normalized(num_ingame: Tuple[int, int], is_super: bool) -> Tuple[int, int]:
    """По текущему num_ingame (2-tuple)
        вернем не выходящий за пределы (7, 7)  для super_tie: (10, 10)

    >>> tie_normalized((8, 7), is_super=False)
    (6, 5)
    >>> tie_normalized((9, 8), is_super=False)
    (7, 6)
    >>> tie_normalized((7, 7), is_super=False)
    (5, 5)
    """
    limit = 10 if is_super else 7
    summa = num_ingame[0] + num_ingame[1]
    if summa >= limit * 2:
        dif = summa - limit * 2
        n_four = (dif // 4) + 1
        return num_ingame[0] - 2 * n_four, num_ingame[1] - 2 * n_four
    return num_ingame


def tie_begin6_score(det_game):
    if det_game.tiebreak and det_game.valid:
        left_sum, right_sum = 0, 0
        for pnt_num, pnt in enumerate(det_game, start=1):
            if pnt_num > 6:
                break
            if pnt.win(left=True):
                left_sum += 1
            else:
                right_sum += 1
        return left_sum, right_sum


Scr = Tuple[int, int]
nil_scr = (-1, -1)


class Score(object):
    set_re = re.compile(
        r"(?P<win_games>\d\d?)-(?P<loss_games>\d\d?)(?P<tie_break>\(\d\d?\))?"
    )
    decsupertie_setnum = 3

    def __init__(self, text, retired=None, decsupertie=None):
        pure_text = co.strip(text)
        self.wl_games = []  # list of 2-tuples
        self.retired = self.__init_retired(pure_text, retired=retired)
        # decsupertie = True if supertiebreak instead set3 (set3 stored as (1,0) or (0,1))
        # __str__ outs set3 in traditional style (tennisbetsite, oncourt) (sample 12-10)
        self.decsupertie = self.__init_decsupertie(pure_text, decsupertie)
        self.supertie_scr = None
        self.idx_tiemin = {}  # set_idx -> tie_loser_result_points_num
        self.error = None
        if pure_text:
            self.__init_parse(pure_text)
            self._check_valid()

    @staticmethod
    def from_pairs(pairs, retired=False, decsupertie=False):
        """init from (s1_left, s1_right), (s2_left, s2_right) ..."""
        obj = Score(text="")
        obj.retired = retired
        obj.decsupertie = decsupertie
        obj.wl_games = [(fst_scr, snd_scr) for fst_scr, snd_scr in pairs]
        obj._check_valid()
        return obj

    def __str__(self):
        result = ""
        for idx, (win, loss) in enumerate(self.wl_games):
            set_txt = "{}-{} ".format(win, loss)
            if (
                self.decsupertie
                and idx == (self.decsupertie_setnum - 1)
                and idx == (len(self.wl_games) - 1)
            ):
                assert self.supertie_scr is not None, "unformat none decsuptie scr"
                set_txt = "{}-{} ".format(self.supertie_scr[0], self.supertie_scr[1])
            elif (win, loss) in ((7, 6), (6, 7), (6, 6)):
                tie_min_result = self.idx_tiemin.get(idx)
                if tie_min_result is not None:
                    set_txt = "{}-{}({}) ".format(win, loss, tie_min_result)
            result += set_txt
        if self.retired:
            result += "ret. "
        if self.decsupertie:
            result += "decsupertie"
        return result.rstrip()

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, str(self))

    def __eq__(self, other):
        if other is None:
            return False
        if self.wl_games != other.wl_games:
            return False
        return (
            self.retired == other.retired
            and self.decsupertie == other.decsupertie
            and self.supertie_scr == other.supertie_scr
        )

    def __getitem__(self, index):
        return self.wl_games[index]

    def __setitem__(self, index, value):
        self.wl_games[index] = value

    def __len__(self):
        return len(self.wl_games)

    def __iter__(self):
        return iter(self.wl_games)

    def valid(self):
        return bool(self.wl_games) and not self.error

    # was coefs to try:
    # f1: [18, 35] -> [12, 20]. f1(x) = (8/17)x + 60/17
    # g1: [36, 53] -> [21, 27]. g1(x) = (6/17)x + 141/17
    # f2: [18, 35] -> [12, 21]. f2(x) = (9/17)x + 42/17
    # g2: [36, 53] -> [22, 27]. g2(x) = (5/17)x + 194/17
    # f4: [18, 33] -> [12, 21]. f4(x) = (9/15)x + 18/15
    # g4: [34, 53] -> [22, 27]. g4(x) = (5/19)x + 248/19
    def normalized_total(self, treat_bo5_as_bo3=True):
        if self.best_of_five():
            best_of_five_total = self.normalized_total_impl(min_value=18, max_value=53)
            if treat_bo5_as_bo3:
                if best_of_five_total <= 32:
                    # f5: [18, 32] -> [12, 21]. f5(x) = (9/14)x + 6/14
                    return int(round(float(9 * best_of_five_total + 6) / 14.0))
                else:  # g5: [33, 53] -> [22, 27]. g5(x) = (1/4)x + 55/4
                    return int(round(float(best_of_five_total + 55) / 4.0))
            else:
                return best_of_five_total
        else:
            return self.normalized_total_impl(min_value=12, max_value=27)

    def normalized_total_impl(self, min_value, max_value):
        result = 0
        for (win, loss) in self.wl_games:
            result += min(13, win + loss)
        result = max(min_value, result)
        return min(max_value, result)

    def games_count(self, load_mode=False, max_setnum=None):
        result = 0
        for setnum, (win, loss) in enumerate(self.wl_games, start=1):
            result += win + loss
            if load_mode and (win, loss) in ((7, 6), (6, 7)):
                result += 1  # с точки зрения нагрузки тайбрейк - это два гейма
            if max_setnum is not None and setnum >= max_setnum:
                break
        return result

    def pairs_iter(self):
        """yields seq of (x: int, y: int) as given in common score text"""
        for idx, (win, loss) in enumerate(self.wl_games):
            if (
                self.decsupertie
                and idx == (self.decsupertie_setnum - 1)
                and idx == (len(self.wl_games) - 1)
            ):
                assert self.supertie_scr is not None, "unformat none decsuptie scr"
                yield self.supertie_scr[0], self.supertie_scr[1]
            else:
                yield win, loss

    def sum_all(self):
        result = 0
        for x, y in self.pairs_iter():
            result += x + y
        return result

    def games_advantage(self):
        diff = 0
        for (win, loss) in self.wl_games:
            diff += win - loss
        return diff

    def best_of_five(self):
        sets_count = self.sets_count()
        return sets_count > 3 or (
            sets_count == 3
            and (
                # first two sets are straight
                (
                    self.wl_games[0][0] > self.wl_games[0][1]
                    and self.wl_games[1][0] > self.wl_games[1][1]
                )
                or (
                    self.wl_games[0][0] < self.wl_games[0][1]
                    and self.wl_games[1][0] < self.wl_games[1][1]
                )
            )
        )

    def sets_count(self, full=False):
        if not full:
            return len(self)
        return sum((1 for (win, loss) in self.wl_games if is_full_set((win, loss))))

    def sets_score(self, full=False):
        """Return (win_sets_count, loss_sets_count)"""
        win_sets_count, loss_sets_count = 0, 0
        for win, loss in self.wl_games:
            if (not full and (win > 0 or loss > 0)) or (
                full and is_full_set((win, loss))
            ):
                if win > loss:
                    win_sets_count += 1
                elif win < loss:
                    loss_sets_count += 1
        return win_sets_count, loss_sets_count

    def winer_side(self):
        if not self.retired:
            sets_scr = self.sets_score(full=False)
            if sets_scr[0] != sets_scr[1]:
                return co.side(sets_scr[0] > sets_scr[1])

    def remove_zero_sets(self):
        while (0, 0) in self.wl_games:
            self.wl_games.remove((0, 0))

    def tie_loser_result(self, setnum):
        if self.decsupertie and setnum == self.decsupertie_setnum:
            return min(*self.supertie_scr) if self.supertie_scr else None
        return self.idx_tiemin.get(setnum - 1)

    def live_update_tie_loser_result(self, ingame):
        if ingame is None:
            return
        setnum = len(self)
        if setnum >= 1 and self[setnum - 1] == (6, 6):
            if ingame[0].isdigit() and ingame[1].isdigit():
                pts1, pts2 = int(ingame[0]), int(ingame[1])
                if abs(pts1 - pts2) <= 10:
                    self.idx_tiemin[setnum - 1] = min(pts1, pts2)

    @staticmethod
    def __parse_retired(text):
        return "w/o" in text or "ret" in text or "n/p" in text or "def" in text

    def __init_retired(self, text, retired):
        retired_in_text = self.__parse_retired(text)
        if retired_in_text:
            if retired is False:
                raise ValueError("ambigious retired: arg=False, text: {}".format(text))
            result = True
        else:
            result = bool(retired)
        return result

    @staticmethod
    def __init_decsupertie(text, decsupertie):
        decsupertie_in_text = "decsupertie" in text
        if decsupertie_in_text:
            if decsupertie is False:
                raise ValueError(
                    "ambigious decsupertie: arg=False, text: {}".format(text)
                )
            result = True
        else:
            result = bool(decsupertie)
        return result

    def __init_parse(self, text):
        sets = text.split(" ")
        for set_idx, s in enumerate(sets):
            if not self.__parse_retired(s) and s != "decsupertie" and len(s) > 0:
                wl_tiemin = self.__set_parse(s)
                if wl_tiemin is not None:
                    if self.decsupertie and set_idx == (self.decsupertie_setnum - 1):
                        self.supertie_scr = (wl_tiemin[0], wl_tiemin[1])
                        suptie_gscrore = (
                            (1, 0) if wl_tiemin[0] > wl_tiemin[1] else (0, 1)
                        )
                        self.wl_games.append(suptie_gscrore)
                        if not self.retired and max(wl_tiemin[0], wl_tiemin[1]) < 10:
                            raise co.TennisScoreSuperTieError(
                                f"unformat decsupertie {text}"
                            )
                    else:
                        self.wl_games.append((wl_tiemin[0], wl_tiemin[1]))
                        if (
                            wl_tiemin[2] and len(wl_tiemin[2]) > 2
                        ):  # is tie loser result
                            tiemin = wl_tiemin[2][1:-1]  # get val from in parentnes ()
                            self.idx_tiemin[set_idx] = int(tiemin)

    def __set_parse(self, text):
        match = Score.set_re.match(text)
        if match:
            return (
                int(match.group("win_games")),
                int(match.group("loss_games")),
                match.group("tie_break"),
            )
        else:
            self.error = "set '{}' is not matched with regexp".format(text)
            return None

    def fliped(self):
        """вернем с обменом сторонами (левого с правым)"""
        result = Score.from_pairs(
            [(s, f) for (f, s) in self.wl_games], retired=self.retired
        )
        if self.idx_tiemin:
            result.idx_tiemin = copy.copy(self.idx_tiemin)
        result.error = self.error
        result.decsupertie = self.decsupertie
        if self.supertie_scr:
            result.supertie_scr = co.reversed_tuple(self.supertie_scr)
        return result

    def tupled(self):
        return tuple(self.wl_games)

    def _check_valid(self):
        """assign to self.error if problems detected"""
        (win_sets, loss_sets) = (0, 0)
        for win, loss in self.wl_games:
            if abs(win - loss) > 6:
                self.error = "invalid set {}-{} (diff more than 6)".format(win, loss)
            if win >= loss:
                win_sets += 1
            else:
                loss_sets += 1
        if (
            (win_sets <= loss_sets) or max(win_sets, loss_sets) < 2
        ) and not self.retired:
            self.error = (
                "win sets num: {} disbalance with " "loss sets num: {} (no ret.)"
            ).format(win_sets, loss_sets)
        n_sets = len(self.wl_games)
        if n_sets > 5:
            self.error = "too long score (more than five sets)"
        elif n_sets > 0:
            (win_last, loss_last) = self.wl_games[-1]
            minval, maxval = min(win_last, loss_last), max(win_last, loss_last)
            if not self.retired and (
                win_last <= loss_last or (win_last < 6 and not self.decsupertie)
            ):
                self.error = "invalid last set {}-{}".format(win_last, loss_last)
            elif (
                maxval >= 10
                and minval < (maxval - 2)
                and not (n_sets == 3 and self.decsupertie)
            ):
                self.error = "invalid lastset {}-{}".format(win_last, loss_last)

        elif n_sets == 0 and not self.error:
            self.error = "empty score"

    def is_symmetric(self):
        for scr in self:
            if scr[0] != scr[1]:
                return False
        return True

    @staticmethod
    def sets_score_variants(best_of_five=False):
        if best_of_five:
            return (3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3)
        else:
            return (2, 0), (2, 1), (1, 2), (0, 2)

    def features(self) -> List[feature.Feature]:
        result: List[feature.Feature] = []
        for idx in range(5):
            fst_name = f"s{idx + 1}_fst_games"
            snd_name = f"s{idx + 1}_snd_games"
            if idx < len(self.wl_games):
                scr = self.wl_games[idx]
                pair = feature.make_pair(
                    fst_name=fst_name,
                    snd_name=snd_name,
                    fst_value=scr[0],
                    snd_value=scr[1],
                )
            else:
                pair = feature.make_pair(fst_name=fst_name, snd_name=snd_name)
            co.add_pair(result, pair[0], pair[1])
        result.append(
            feature.RigidFeature("s1_tie_loser_result", self.tie_loser_result(setnum=1))
        )
        return result


# gone from st_cont.py
def complete_sets_score_keys(histogram, best_of_five=False):
    assert (
        histogram.default_factory == int
    ), "unexpected (int expected) histo type: {}".format(histogram.default_factory)
    for sets_score in Score.sets_score_variants(best_of_five):
        if sets_score not in histogram:
            histogram[sets_score] = 0


__text_to_num = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "60": 5}


def is_text_correct(text):
    return text in __text_to_num


def text_from_num(number):
    for text, num in __text_to_num.items():
        if num == number:
            return text


def ingame_to_num(text_ingame, tiebreak):
    """return num_ingame 2-tuple

    >>> ingame_to_num(('0', '30'), tiebreak=False)
    (0, 2)
    >>> ingame_to_num(('A', '40'), tiebreak=False)
    (4, 3)
    >>> ingame_to_num(('40', 'A'), tiebreak=False)
    (3, 4)
    >>> ingame_to_num(('3', '4'), tiebreak=True)
    (3, 4)
    """
    if tiebreak:
        return int(text_ingame[0]), int(text_ingame[1])
    else:
        return (__text_to_num[text_ingame[0]], __text_to_num[text_ingame[1]])


def num_ingame_to_win(num_ingame, tiebreak):
    """return 2-tuple (x, y),
        где x - сколько очков подряд должен выиграть левый для победы в гейме,
            y - аналогично для правого игрока. num_ingame - исходный счет.

    >>> num_ingame_to_win((0, 3), tiebreak=False)
    (5, 1)
    >>> num_ingame_to_win((0, 0), tiebreak=False)
    (4, 4)
    >>> num_ingame_to_win((0, 1), tiebreak=False)
    (4, 3)
    >>> num_ingame_to_win((3, 4), tiebreak=False)
    (3, 1)
    >>> num_ingame_to_win((4, 3), tiebreak=False)
    (1, 3)
    >>> num_ingame_to_win((6, 5), tiebreak=True)
    (1, 3)
    """
    min_for_win = 7 if tiebreak else 4
    fst_for_win = max(min_for_win, num_ingame[1] + 2)
    snd_for_win = max(min_for_win, num_ingame[0] + 2)
    return fst_for_win - num_ingame[0], snd_for_win - num_ingame[1]


def num_ingame_breakpoint_count(num_ingame, left_service, tiebreak=False):
    """сколько BP имеет принимающий при данном счете

    >>> num_ingame_breakpoint_count((3, 3), left_service=True)
    0
    >>> num_ingame_breakpoint_count((0, 3), left_service=True)
    3
    >>> num_ingame_breakpoint_count((2, 3), left_service=True)
    1
    >>> num_ingame_breakpoint_count((3, 4), left_service=True)
    1
    >>> num_ingame_breakpoint_count((3, 3), left_service=False)
    0
    >>> num_ingame_breakpoint_count((3, 0), left_service=False)
    3
    >>> num_ingame_breakpoint_count((3, 2), left_service=False)
    1
    >>> num_ingame_breakpoint_count((4, 3), left_service=False)
    1
    """
    x_towin, y_towin = num_ingame_to_win(num_ingame, tiebreak=tiebreak)
    if left_service:
        if y_towin != 1 or x_towin < 2:
            return 0
        else:
            return num_ingame[1] - num_ingame[0]
    else:
        if x_towin != 1 or y_towin < 2:
            return 0
        else:
            return num_ingame[0] - num_ingame[1]


def num_ingame_bgpoint_counts(num_ingame, tiebreak):
    """return (left_edge_points, right_edge_points)

    >>> num_ingame_bgpoint_counts((3, 3), tiebreak=False)
    (0, 0)
    >>> num_ingame_bgpoint_counts((4, 3), tiebreak=False)
    (1, 0)
    >>> num_ingame_bgpoint_counts((3, 4), tiebreak=False)
    (0, 1)
    >>> num_ingame_bgpoint_counts((3, 0), tiebreak=False)
    (3, 0)
    >>> num_ingame_bgpoint_counts((0, 3), tiebreak=False)
    (0, 3)

    >>> num_ingame_bgpoint_counts((6, 0), tiebreak=True)
    (6, 0)
    >>> num_ingame_bgpoint_counts((0, 6), tiebreak=True)
    (0, 6)
    >>> num_ingame_bgpoint_counts((1, 6), tiebreak=True)
    (0, 5)
    >>> num_ingame_bgpoint_counts((3, 0), tiebreak=True)
    (0, 0)
    >>> num_ingame_bgpoint_counts((6, 7), tiebreak=True)
    (0, 1)
    >>> num_ingame_bgpoint_counts((7, 6), tiebreak=True)
    (1, 0)
    >>> num_ingame_bgpoint_counts((8, 9), tiebreak=True)
    (0, 1)
    >>> num_ingame_bgpoint_counts((9, 8), tiebreak=True)
    (1, 0)
    """
    return (
        num_ingame_breakpoint_count(num_ingame, left_service=False, tiebreak=tiebreak),
        num_ingame_breakpoint_count(num_ingame, left_service=True, tiebreak=tiebreak),
    )


def differed(num_tuple):
    if num_tuple is None:
        return None
    min_val = min(num_tuple)
    if min_val > 0:
        return num_tuple[0] - min_val, num_tuple[1] - min_val
    return num_tuple


def prev_game_score(score):
    result = Score(str(score))
    for set_idx in reversed(range(len(result))):
        left_games, right_games = result[set_idx]
        if left_games == 0 and right_games == 0:
            continue
        if left_games == right_games:
            return None  # can not get not ambigious prev game score
        if left_games > right_games:
            result[set_idx] = (left_games - 1, right_games)
        else:
            result[set_idx] = (left_games, right_games - 1)
        return result


def is_full_set(in_set, is_decided=None, decided_tie_info=None):
    win_games, loss_games = in_set
    minimum = min(win_games, loss_games)
    maximum = max(win_games, loss_games)
    if maximum >= (minimum + 2) and maximum >= 6:
        return True
    if decided_tie_info is None or not is_decided:
        return in_set in ((7, 6), (6, 7))
    # tiebreak in decided set:
    if (maximum - minimum) != 1:
        return False
    return decided_tie_info.beg_scr == (minimum, minimum)


def completed_score(score, best_of_five=False):
    """счет дополняется до формально верного (для случаев если незавершен)"""

    def left_filled_set(in_set):
        """left must be winner"""
        win_games, loss_games = in_set
        if win_games < 6 or win_games <= loss_games:
            win_games = max(6, win_games, loss_games + 1)
        if win_games == 6 and loss_games == 5:
            win_games = 7
        return win_games, loss_games

    result = copy.deepcopy(score)
    sets_count = result.sets_count()
    if sets_count >= 1 and not is_full_set(result[sets_count - 1]):
        result[sets_count - 1] = left_filled_set(result[sets_count - 1])
    sets_win, sets_loss = result.sets_score()
    while len(result.wl_games) < 5:
        result.wl_games.append((0, 0))
    if sets_win == 2:
        if best_of_five:
            result[sets_win + sets_loss] = (6, 0)
    elif sets_win == 1:
        result[sets_win + sets_loss] = (6, 0)
        if best_of_five:
            result[sets_win + sets_loss + 1] = (6, 0)
    elif sets_win == 0:
        result[sets_win + sets_loss] = (6, 0)
        result[sets_win + sets_loss + 1] = (6, 0)
        if best_of_five:
            result[sets_win + sets_loss + 2] = (6, 0)
    result.remove_zero_sets()
    return result


def tough_score(score, best_of_five=None):
    if best_of_five is None:
        best_of_five = score.best_of_five()
    n_games = score.games_count()
    return n_games >= 47 if best_of_five else n_games >= 31


def setidx_by_name(setname, best_of_five=False):
    result = None
    if setname == "open":
        result = 0
    elif setname == "second" and not best_of_five:
        result = 1
    elif setname == "decided":
        result = 4 if best_of_five else 2
    return result


def set_tail_gaps(start, end, left_semiopen=False):
    """start, end - пары начального и конечного счетов теннисной партии.
    Вернем итератор для start <= elem < end (если left_semiopen),
    Вернем итератор для start < elem <= end (если not left_semiopen)
    """

    def before_end(start, end):
        delta = 0
        emax, emin = max(end), min(end)
        if emax > emin:
            bigidx = 0 if emax == end[0] else 1
            if end[bigidx] > start[bigidx]:
                delta = -1
            if emax > (emin + 1) and end[bigidx] > (start[bigidx] + 1) and emax != 6:
                delta = -2
            if bigidx == 0:
                return end[0] + delta, end[1]
            else:
                return end[0], end[1] + delta
        return end

    def do_step(beg, end):
        delta_x = end[0] - beg[0]
        delta_y = end[1] - beg[1]
        if delta_x >= delta_y:
            return beg[0] + 1, beg[1]
        else:
            return beg[0], beg[1] + 1

    def steps(beg, end):
        if left_semiopen:
            while beg < end:
                yield beg
                beg = do_step(beg, end)
        else:
            while beg < end:
                beg = do_step(beg, end)
                yield beg

    soft_end = before_end(start, end)
    return itertools.chain(steps(start, soft_end), steps(soft_end, end))


def score_tail_gap_iter(start_score, end_score, left_semiopen=False):
    """на входе типы tuple. Вернем элементы start_score < elem <= end_score"""
    eq_sets = [(b == e) for b, e in zip(start_score, end_score)]
    if (
        len(start_score) > len(end_score)
        or start_score >= end_score
        or (
            len(start_score) >= 1
            and len(end_score) >= 1
            and start_score[0] == tuple(reversed(end_score[0]))
        )  # first set reversed
    ):
        raise NotImplementedError("start: {} end: {}".format(start_score, end_score))
    n_false = sum((1 for es in eq_sets if not es))
    if n_false > 1 or (n_false == 1 and eq_sets.index(False) != (len(eq_sets) - 1)):
        raise NotImplementedError(
            "start: {} end: {} [2]".format(start_score, end_score)
        )
    n_beg_eq = len(eq_sets) - n_false
    while n_beg_eq < len(end_score):
        if n_beg_eq == len(start_score):
            start_score = start_score + ((0, 0),)
        sc_start = start_score[n_beg_eq]
        sc_end = end_score[n_beg_eq]
        if sc_start > sc_end:
            raise NotImplementedError(
                "start: {} end: {} [3]".format(start_score, end_score)
            )
        for sc_current in set_tail_gaps(sc_start, sc_end, left_semiopen):
            start_score = start_score[0:n_beg_eq] + (sc_current,)
            yield start_score
        n_beg_eq += 1


class TieInfo:
    __slots__ = ('beg_scr', 'is_super')

    def __init__(self, beg_scr, is_super: bool):
        #  beg_scr - score in dec set (sample: (6, 6)) when starts tiebreak (if exist)
        self.beg_scr = beg_scr
        self.is_super = is_super

    def is_tiebreak(self):
        return self.beg_scr is not None

    def is_supertiebreak(self):
        return self.beg_scr is not None and self.is_super

    def is_usial_tiebreak(self):
        return self.is_tiebreak() and not self.is_supertiebreak()

    def at_scr(self):
        return self.beg_scr

    def is_fmt66_over(self, dec_set):
        return (
            self.is_usial_tiebreak()
            and self.at_scr() == (6, 6)
            and dec_set
            and max(*dec_set) > 7
        )

    def is_fmt1212_over(self, dec_set):
        return (
            self.is_usial_tiebreak()
            and self.at_scr() == (12, 12)
            and dec_set
            and max(*dec_set) > 13
        )

    def __bool__(self):
        return self.beg_scr is not None

    def __eq__(self, other):
        return self.beg_scr == other.beg_scr and self.is_super == other.is_super

    def __repr__(self):
        return "beg:{} super:{}".format(self.beg_scr, int(self.is_super))


absent_tie_info = TieInfo(beg_scr=None, is_super=False)  # for win by advantage set
default_tie_info = TieInfo(beg_scr=(6, 6), is_super=False)
super_tie_66_info = TieInfo(beg_scr=(6, 6), is_super=True)
super_tie_1212_info = TieInfo(beg_scr=(12, 12), is_super=True)


def decided_tiebreak(
    sex, year, tour_name, qualification, level=None, best_of_five=None, money=None
) -> Optional[TieInfo]:
    def ao_wildcard_tiebreak():
        if (sex == "wta" and year >= 2017) or (sex == "atp" and year >= 2018):
            return True
        return False

    if year >= 2022 and level == lev.gs:
        return super_tie_66_info
    if tour_name is not None and 'us-open' in tour_name:
        return default_tie_info
    if tour_name is None:
        return None

    if "wildcard" in tour_name and "australian-open" in tour_name:
        return default_tie_info if ao_wildcard_tiebreak() else absent_tie_info
    if "olympics" in tour_name:
        return default_tie_info if year >= 2016 else absent_tie_info

    if qualification is None:
        return None

    if "wimbledon" in tour_name:
        if year >= 2022:
            return super_tie_66_info
        elif year >= 2019:
            return TieInfo(beg_scr=(12, 12), is_super=False)
        return absent_tie_info

    elif "french-open" in tour_name:
        if year >= 2022:
            return super_tie_66_info
        if year <= 2016:
            return absent_tie_info
        if year >= 2017 and qualification:
            return default_tie_info
        return absent_tie_info

    elif "australian-open" in tour_name:
        if year >= 2019:
            return super_tie_66_info
        if year <= 2016:
            return absent_tie_info
        if (
            (sex == "wta" and year >= 2018 and qualification)
            or (sex == "atp" and year >= 2019 and qualification)
        ):
            return default_tie_info
        return absent_tie_info

    elif sex == "wta" and qualification and level != lev.gs:
        if (
            level == "future"
            or (level == "chal" and money is not None and money < 115000)
        ):
            return TieInfo(beg_scr=(0, 0), is_super=True)

    elif level == "teamworld" and sex == "atp":
        if best_of_five is None:
            return None
        elif best_of_five is False:
            # если команда winner уже известна, то формат сокращается (std)
            return default_tie_info
        return default_tie_info if year >= 2016 else absent_tie_info

    elif level == "teamworld" and sex == "wta" and year <= 2017:
        return absent_tie_info
    else:
        return default_tie_info


def get_decided_tiebreak_info_ext(
    tour, rnd: Round, score: Score
) -> Tuple[Score, TieInfo]:
    """:return Tuple[may be edited Score, dec TieInfo]
    can raise TennisScoreSuperTieError.
    if in score no exist dec set, than return Tuple[in score, default TieInfo]"""
    n_sets = score.sets_count()
    best_of_five = False if tour.sex == "wta" else score.best_of_five()
    dec_set_num = 5 if best_of_five else 3
    dec_set = score[dec_set_num - 1] if n_sets == dec_set_num else None
    if dec_set is None:
        return score, default_tie_info

    is_dec_supertie_scr_uniq = None
    if is_dec_supertie_scr(dec_set, score.retired, unique_mode=True):
        is_dec_supertie_scr_uniq = True
    is_dec_supertie_scr_lowrank = None
    if (
        tour.sex == "wta"
        and (
            tour.level in ("future", "chal")
            or "all-lower-level-tournaments" in tour.name
        )
        and is_dec_supertie_scr(dec_set, score.retired)
    ):
        is_dec_supertie_scr_lowrank = True
    dec_tie_fmt = decided_tiebreak(
        tour.sex,
        tour.date.year,
        tour.name,
        qualification=rnd.qualification(),
        level=tour.level,
        best_of_five=best_of_five,
        money=tour.money,
    )

    if is_dec_supertie_scr_uniq:
        res_dectieinfo = TieInfo(beg_scr=(0, 0), is_super=True)
    elif (
        not dec_tie_fmt.is_tiebreak()
        or dec_tie_fmt.is_fmt66_over(dec_set)
        or dec_tie_fmt.is_fmt1212_over(dec_set)
    ):
        res_dectieinfo = TieInfo(None, False)
    elif is_dec_supertie_scr_lowrank:
        res_dectieinfo = TieInfo(beg_scr=(0, 0), is_super=True)
    else:
        res_dectieinfo = dec_tie_fmt

    res_score = score
    if res_dectieinfo.beg_scr == (0, 0):
        if not score.retired and max(dec_set) < 10:
            return score, default_tie_info  # dec super tie impossible
        scr_txt = str(score)
        try:
            res_score = Score(scr_txt, decsupertie=True)
        except co.TennisScoreSuperTieError as err:
            raise co.TennisScoreSuperTieError(
                f"{err} sex:{tour.sex} {tour.date} tour:{tour.name} "
                f"rnd:{rnd} lev:{tour.level} ${tour.money}"
            )
    return res_score, res_dectieinfo


def is_dec_supertie_scr(in_set, retired, unique_mode=False):
    """most widly used in third set WTA ITF"""
    win_games, loss_games = in_set
    minval = min(win_games, loss_games)
    maxval = max(win_games, loss_games)
    if unique_mode:
        return maxval >= (minval + 3) and maxval == 10
    if maxval >= (minval + 2) and maxval >= 10:
        return True
    if retired and maxval >= (minval + 3) and maxval >= 8:
        return True


def paired_match_sets_count(tour, score, full=False):
    result = 0
    for set_idx in range(len(score)):
        win, loss = score[set_idx]
        if win > 0 or loss > 0:
            set_num = set_idx + 1
            if set_num == 3 and "wimbledon" in tour.name:
                if not full or (full and max(win, loss) >= 6):
                    result += 1
            elif set_num == 3 and max(win, loss) >= 10:
                break  # champion tie (it is not set)
            elif not full or (full and max(win, loss) >= 6):
                result += 1
    return result


def paired_match_games_count(tour, score, load_mode=False):
    result = 0
    for set_idx in range(len(score)):
        win, loss = score[set_idx]
        if win > 0 or loss > 0:
            set_num = set_idx + 1
            if set_num == 3 and max(win, loss) >= 10 and "wimbledon" not in tour.name:
                result += 3  # champion tie (it is not set, assume as 3 games)
            else:
                result += win + loss
                if load_mode and (win, loss) in ((7, 6), (6, 7)):
                    result += 1  # с точки зрения нагрузки тайбрейк - это два гейма
    return result


def breaks_advantage(insetscore, srv_side: Side):
    """breaks advantage for left: 0 - no adv. n>0 left leads n breaks. n<0 left trails.

    >>> breaks_advantage((1, 1), srv_side=co.LEFT)
    0
    >>> breaks_advantage((0, 1), srv_side=co.LEFT)
    0
    >>> breaks_advantage((2, 0), srv_side=co.LEFT)
    1
    >>> breaks_advantage((1, 2), srv_side=co.RIGHT)
    -1
    >>> breaks_advantage((3, 0), srv_side=co.LEFT)
    2
    >>> breaks_advantage((1, 4), srv_side=co.RIGHT)
    -2
    >>> breaks_advantage((0, 6), srv_side=co.LEFT)
    -3
    """
    diff = insetscore[0] - insetscore[1]
    if diff == 0:
        return 0
    if diff < 0:
        return -breaks_advantage(co.reversed_tuple(insetscore), srv_side.fliped())
    # only left may have advantage
    if srv_side.is_left():
        return (diff + 1) // 2
    else:
        return diff // 2


def strong_lead_side(cur_score, ingame, srv_side: Side) -> Optional[Side]:
    """определяем явного лидера. ingame - в текстовом формате.

    >>> strong_lead_side(Score('5-1'), ingame=None, srv_side=co.RIGHT)
    Side('LEFT')
    >>> strong_lead_side(Score('2-6 5-1'), ingame=None, srv_side=co.RIGHT)

    >>> strong_lead_side(Score('6-3 2-1'), ingame=None, srv_side=co.RIGHT)
    Side('LEFT')
    >>> strong_lead_side(Score('6-3 2-6 4-1'), ingame=None, srv_side=co.LEFT)
    Side('LEFT')
    >>> strong_lead_side(Score('3-6 2-1'), ingame=('40', '0'), srv_side=co.RIGHT)

    """

    def get_prev_sets_lead_side():
        if sets_score[0] == sets_score[1]:
            return None
        return co.LEFT if sets_score[0] > sets_score[1] else co.RIGHT

    def ingame_brk_threat():
        """угроза брейка: 1 левый угрожает, -1 правый, 0 - нет"""
        if ingame is None:
            return 0
        if ingame in (("0", "40"), ("15", "40")) and srv_side.is_left():
            return -1
        if ingame in (("40", "0"), ("40", "15")) and srv_side.is_right():
            return 1
        return 0

    def get_cur_set_lead_side(scr):
        brk_adv = breaks_advantage(scr, srv_side)
        threat_brk_adv = ingame_brk_threat()
        adv = brk_adv + threat_brk_adv
        if adv > 0:
            return co.LEFT
        elif adv < 0:
            return co.RIGHT

    if cur_score is None:
        return None

    sets_score = cur_score.sets_score(full=True)
    prev_sets_lead_side = get_prev_sets_lead_side()

    cur_set = cur_score[-1]
    if is_full_set(cur_set):
        cur_set_lead_side = None
    else:
        cur_set_lead_side = get_cur_set_lead_side(cur_set)

    if not prev_sets_lead_side and not cur_set_lead_side:
        return None
    elif prev_sets_lead_side == cur_set_lead_side:
        return prev_sets_lead_side
    elif not prev_sets_lead_side:
        return cur_set_lead_side
    elif not cur_set_lead_side:
        return prev_sets_lead_side
    return None

