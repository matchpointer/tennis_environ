from collections import defaultdict, namedtuple
from operator import itemgetter
import operator
import re
import functools
import unittest
from typing import Optional, List

from recordclass import recordclass

import common as co
from loguru import logger as log
from side import Side
import report_line as rl
import score as sc


class LastInrowCount(object):
    def __init__(self, init_count=0):
        self.count = init_count  # sample: -2 loss two last. +3 wins three last

    @property
    def size(self):
        return self.count

    def hit(self, cnt):
        """cnt: bool_iswin or +int_count of wins or -int_count of loss"""
        if cnt is None:  # unknown game(s) result
            self.count = 0  # clear state
            return
        if isinstance(cnt, bool):
            cnt = 1 if cnt else -1
        if cnt > 0:  # win
            if self.count <= 0:
                self.count = cnt
            else:
                self.count += cnt
        elif cnt < 0:  # loss
            if self.count >= 0:
                self.count = cnt
            else:
                self.count -= abs(cnt)

    @staticmethod
    def glue(cnt1, cnt2):
        """return integrated cnt where cnt1, cnt2 are taken from neighbour sets"""
        if cnt1 is None or cnt2 is None:  # unknown set(s) result
            return None
        if cnt2 == 0:
            return cnt1  # cnt2 set is ignored as empty yet
        sign1, sign2 = co.sign(cnt1), co.sign(cnt2)
        if sign1 == sign2:
            return sign1 * (abs(cnt1) + abs(cnt2))
        return cnt2


class Sumator(object):
    str_as_average = True
    average_precision = 3  # digits after comma

    def __init__(self, init_sum=0, init_count: int = 0):
        self.sum = init_sum
        self.count = init_count

    @classmethod
    def from_seq(cls, iterable):
        result = cls()
        for val in iterable:
            result.hit(val)
        return result

    @property
    def size(self) -> int:
        return self.count

    def __str__(self):
        if Sumator.str_as_average:
            return co.formated(
                self.average(), self.count, round_digits=Sumator.average_precision
            )
        else:
            return co.formated(self.sum, self.count)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.sum, self.count)

    def hit(self, value):
        self.sum += value
        self.count += 1

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Sumator(self.sum + other, self.count + 1)
        else:
            assert isinstance(
                other, Sumator
            ), "invalid other Sumator type: '{}'".format(type(other))
            return Sumator(self.sum + other.sum, self.count + other.count)

    def __iadd__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.hit(other)
        else:
            assert isinstance(
                other, Sumator
            ), "invalid other Sumator type: '{}'".format(type(other))
            self.sum += other.sum
            self.count += other.count
        return self

    def __eq__(self, other):
        if isinstance(self.sum, float) or isinstance(other.sum, float):
            return co.equal_float(self.sum, other.sum) and self.count == other.count
        else:
            return self.sum == other.sum and self.count == other.count

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.sum, self.count))

    def __bool__(self):
        return self.count > 0

    __nonzero__ = __bool__

    def average(self):
        if self.count > 0:
            return float(self.sum) / float(self.count)

    @property
    def value(self):
        return self.average()


@functools.total_ordering
class WinLoss(object):
    text_re = re.compile(
        r"(?P<value>-?\d+\.\d+)(?P<percent_sign>%?) +\((?P<size>\d+)\) *"
    )

    def __init__(self, win_count: int = 0, loss_count: int = 0):
        self.win_count = win_count
        self.loss_count = loss_count

    @classmethod
    def from_ratio(cls, win_ratio, size):
        win_count = int(round(float(size) * win_ratio))
        loss_count = size - win_count
        return cls(win_count, loss_count)

    @classmethod
    def from_text(cls, text):
        match = cls.text_re.match(text)
        if match:
            multiplier = 0.01 if match.group("percent_sign") else 1.0
            win_ratio = float(match.group("value")) * multiplier
            size = int(match.group("size"))
            return cls.from_ratio(win_ratio, size)
        raise co.TennisError("unparsed WinLoss text: '{}'".format(text))

    @classmethod
    def from_iter(cls, iterable):
        """make object from iter of booleans-like"""
        win_count, loss_count = 0, 0
        for val in iterable:
            if val:
                win_count += 1
            else:
                loss_count += 1
        return cls(win_count, loss_count)

    def add_win(self, count: int):
        self.win_count += count

    def add_loss(self, count: int):
        self.loss_count += count

    def __bool__(self):
        return self.size > 0

    __nonzero__ = __bool__

    def __eq__(self, other):
        """compare win ratios"""
        if not self and not other:
            return True
        this_sum = self.win_count + self.loss_count
        other_sum = other.win_count + other.loss_count
        return (self.win_count * other_sum) == (other.win_count * this_sum)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """compare win ratios"""
        if not self:
            return True
        if not other:
            return False
        this_sum = self.win_count + self.loss_count
        other_sum = other.win_count + other.loss_count
        return (self.win_count * other_sum) < (other.win_count * this_sum)

    def __str__(self):
        return co.percented_text(self.win_count, self.size)

    def ratio_size_str(self, precision=2):
        if self.win_count == 0 and self.loss_count == 0:
            return "0 (0)"
        return f"{self.ratio:.{precision}f} ({self.size})"

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.win_count, self.loss_count
        )

    def __hash__(self):
        return hash((self.win_count, self.loss_count))

    def __add__(self, other):
        if isinstance(other, (bool, int)):
            result = WinLoss(self.win_count, self.loss_count)
            result.hit(other)
            return result
        else:
            assert isinstance(other, WinLoss), "invalid other type: '{}'".format(
                type(other)
            )
            return WinLoss(
                self.win_count + other.win_count, self.loss_count + other.loss_count
            )

    def __iadd__(self, other):
        if isinstance(other, (bool, int)):
            self.hit(other)
        else:
            assert isinstance(other, WinLoss), "invalid other type: '{}'".format(
                type(other)
            )
            self.win_count += other.win_count
            self.loss_count += other.loss_count
        return self

    def __sub__(self, other):
        if isinstance(other, (bool, int)):
            result = WinLoss(self.win_count, self.loss_count)
            if other:
                result.win_count -= 1
            else:
                result.loss_count -= 1
            return result
        else:
            assert isinstance(other, WinLoss), "invalid other type: '{}'".format(
                type(other)
            )
            return WinLoss(
                self.win_count - other.win_count, self.loss_count - other.loss_count
            )

    def __isub__(self, other):
        if isinstance(other, (bool, int)):
            if other:
                self.win_count -= 1
            else:
                self.loss_count -= 1
        else:
            assert isinstance(other, WinLoss), "invalid other type: '{}'".format(
                type(other)
            )
            self.win_count -= other.win_count
            self.loss_count -= other.loss_count
        return self

    @property
    def size(self) -> int:
        return self.win_count + self.loss_count

    @property
    def ratio(self):
        all_count = self.win_count + self.loss_count
        if all_count > 0:
            return float(self.win_count) / float(all_count)

    value = ratio

    def hit(self, iswin):
        if iswin:
            self.win_count += 1
        else:
            self.loss_count += 1

    def reversed(self):
        return WinLoss(win_count=self.loss_count, loss_count=self.win_count)


class WinLossTest(unittest.TestCase):
    def test_compare(self):
        wl1 = WinLoss.from_text("41.2% (17)")
        wl2 = WinLoss.from_text("37.8% (339)")
        self.assertTrue(wl1 > wl2)
        self.assertTrue(wl2 < wl1)
        self.assertEqual([wl2, wl1], sorted([wl1, wl2]))

    def test_sum(self):
        wl1 = WinLoss(7, 3)
        wl2 = WinLoss(6, 4)
        wl3 = WinLoss(0, 0)
        dct = {1: wl1, 2: wl2, 3: wl3}
        s = sum((dct[i] for i in range(1, 4)), WinLoss())
        self.assertEqual(wl1 + wl2 + wl3, s)


class WinLossPairValue(object):
    """РґР»СЏ РЅР°РєРѕРїР»РµРЅРёСЏ Р·РЅР°С‡РµРЅРёР№ (value) СЃ СЂР°Р·Р±РёРІРєРѕР№ РЅР° РґРІР° РІР°СЂРёР°РЅС‚Р°:
    РІС‹РёРіСЂС‹С€ РёР»Рё РїСЂРѕРёРіСЂС‹С€.
    """

    def __init__(self):
        self.winloss = WinLoss()
        self.win_value = Sumator()
        self.loss_value = Sumator()

    def __iadd__(self, value_iswin):
        value, iswin = value_iswin[0], value_iswin[1]
        self.winloss.hit(iswin)
        if iswin:
            self.win_value += value
        else:
            self.loss_value += value
        return self

    def __str__(self):
        return "{0} winval: {1:5.2f} lossval: {2:5.2f}".format(
            self.winloss, self.win_value.average(), self.loss_value.average()
        )

    @property
    def size(self):
        return self.winloss.size


class Histogram(defaultdict):
    def __init__(self, default_factory=int, iterable=None):
        """default_factory must supply operations: +=, +"""
        super(Histogram, self).__init__(
            default_factory, iterable if iterable is not None else {}
        )

    def hit(self, keys, value=1):
        for key in keys:
            self[key] += value

    def __bool__(self):
        return any(self.values())

    __nonzero__ = __bool__

    def __add__(self, other):
        assert (
            self.default_factory == other.default_factory
        ), "add requires default_factory equal. \nself: {}\nother: {}".format(
            self.default_factory, other.default_factory
        )
        result = Histogram(self.default_factory)
        for k in self.keys() | other.keys():
            result[k] = self[k] + other[k]
        return result


def create_histogram():
    return Histogram()


def create_winloss_histogram():
    return Histogram(WinLoss)


def create_summator_histogram():
    return Histogram(Sumator)


def create_winloss_pair_value_histogram():
    return Histogram(WinLossPairValue)


class SetsScoreHistogramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.histos = [
            Histogram(int, {(2, 0): 5, (2, 1): 3, (1, 2): 1, (0, 2): 2}),
            Histogram(int, {(2, 0): 6, (2, 1): 4, (1, 2): 2, (0, 2): 3}),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.histos = []

    def test_init_and_sum(self):
        self.assertEqual(self.histos[0][(2, 0)], 5)
        self.assertEqual(self.histos[0][(2, 1)], 3)
        self.assertEqual(self.histos[0][(1, 2)], 1)
        self.assertEqual(self.histos[0][(0, 2)], 2)

        self.assertEqual(self.histos[1][(2, 0)], 6)
        self.assertEqual(self.histos[1][(2, 1)], 4)
        self.assertEqual(self.histos[1][(1, 2)], 2)
        self.assertEqual(self.histos[1][(0, 2)], 3)

        start_histo = Histogram()
        sc.complete_sets_score_keys(start_histo)
        histo = sum(self.histos, start_histo)
        self.assertEqual(type(histo), Histogram)
        self.assertEqual(histo[(2, 0)], 11)
        self.assertEqual(histo[(2, 1)], 7)
        self.assertEqual(histo[(1, 2)], 3)
        self.assertEqual(histo[(0, 2)], 5)


class QuadServiceStatError(co.TennisError):
    pass


class QuadServiceStatLateError(QuadServiceStatError):
    pass


ThresholdedResult = namedtuple("ThresholdedResult", "side quadrant ratio")


class BreakupTrack:
    __slots__ = ("is_fst_breakups",)

    def __init__(self):
        self.is_fst_breakups: List[bool] = []

    def is_any_breakup(self):
        return bool(self.is_fst_breakups)

    def is_fst_breakup(self):
        return any(self.is_fst_breakups)

    def is_snd_breakup(self):
        return not all(self.is_fst_breakups)

    def put(self, scr, left_service: bool, left_win_game: bool):
        """ input: last live score about finished game """
        if left_service is left_win_game:
            return  # hold
        x, y = scr
        if x <= y and left_service:
            self.is_fst_breakups.append(False)
        elif y <= x and left_service is False:
            self.is_fst_breakups.append(True)


class BreakupTracker:
    MAX_SETNUM = 3  # for time & memory economy

    def __init__(self):
        # setnum -> BreakupTrack
        self.track_from_setnum = defaultdict(BreakupTrack)

    def is_fst_breakup_list(self, setnum: int) -> Optional[List[int]]:
        if setnum is not None:
            return self.track_from_setnum[setnum].is_fst_breakups

    def is_fst_breakup(self, setnum):
        if setnum is not None:
            return self.track_from_setnum[setnum].is_fst_breakup()

    def is_snd_breakup(self, setnum):
        if setnum is not None:
            return self.track_from_setnum[setnum].is_snd_breakup()

    def is_any_breakup(self, setnum):
        if setnum is not None:
            return self.track_from_setnum[setnum].is_any_breakup()

    def close_previous(
        self, prev_score, prev_left_service: bool, prev_left_wingame: bool
    ):
        scr = prev_score[-1]
        if scr != (6, 6):  # we want skip tie
            setnum = len(prev_score)
            if setnum <= self.MAX_SETNUM:
                self.track_from_setnum[setnum].put(
                    scr, prev_left_service, prev_left_wingame
                )


class HoldStat(object):
    """holds and breaks counts done by both players"""

    def __init__(self):
        # setnum -> (left WinLoss, right WinLoss)
        #           win count = holds count, loss count = breaks count
        self.wlpair_from_setnum = defaultdict(lambda: (WinLoss(), WinLoss()))

    def result_pair(self, setnum=None, as_float=True):
        """return winloss_pair if not as_float, otherwise (left_ratio, right_ratio)"""
        if setnum is not None:
            return self.wlpair_from_setnum[setnum]
        fst, snd = WinLoss(), WinLoss()
        for sn in self.wlpair_from_setnum:
            fst_wl, snd_wl = self.wlpair_from_setnum[sn]
            fst += fst_wl
            snd += snd_wl
        return (fst.ratio, snd.ratio) if as_float else (fst, snd)

    def close_previous(self, prev_score, prev_left_service, prev_left_wingame):
        if prev_score[-1] != (6, 6):  # we want skip tie
            setnum = len(prev_score)
            fst_snd = self.wlpair_from_setnum[setnum]
            if prev_left_service:
                fst_snd[0].hit(prev_left_wingame)
            else:
                fst_snd[1].hit(not prev_left_wingame)


class HoldStatTest(unittest.TestCase):
    def test_close_previous(self):
        stat1 = HoldStat()
        stat1.close_previous(
            sc.Score("1-1"), prev_left_service=True, prev_left_wingame=True
        )  # 1/1, 0/0
        stat1.close_previous(
            sc.Score("2-1"), prev_left_service=False, prev_left_wingame=False
        )  # 1/1, 1/1
        stat1.close_previous(
            sc.Score("2-2"), prev_left_service=True, prev_left_wingame=False
        )  # 1/2, 1/1
        stat1.close_previous(
            sc.Score("2-3"), prev_left_service=False, prev_left_wingame=False
        )  # 1/2, 2/2
        stat1.close_previous(
            sc.Score("2-4"), prev_left_service=True, prev_left_wingame=True
        )  # 2/3, 2/2
        stat1.close_previous(
            sc.Score("6-6"), prev_left_service=True, prev_left_wingame=False
        )  # skip tie
        res_pair = stat1.result_pair(setnum=1, as_float=False)
        self.assertEqual(res_pair, (WinLoss(2, 1), WinLoss(2, 0)))


class MissBigPointStat(object):
    def __init__(self):
        # setnum -> (left_cnt, right_cnt)
        self.sp_srv_dct = defaultdict(lambda: (0, 0))
        self.sp_rcv_dct = defaultdict(lambda: (0, 0))
        self.mp_srv_dct = defaultdict(lambda: (0, 0))
        self.mp_rcv_dct = defaultdict(lambda: (0, 0))

        # tmp means current game. (left_cnt, right_cnt) :
        self.tmp_sp_srv = (0, 0)
        self.tmp_sp_rcv = (0, 0)
        self.tmp_mp_srv = (0, 0)
        self.tmp_mp_rcv = (0, 0)

        self.tmp_sp_chance_side = None
        self.tmp_mp_chance_side = None

    def clear_tmp(self):
        self.tmp_sp_srv = (0, 0)
        self.tmp_sp_rcv = (0, 0)
        self.tmp_mp_srv = (0, 0)
        self.tmp_mp_rcv = (0, 0)
        self.tmp_sp_chance_side = None
        self.tmp_mp_chance_side = None

    def get_scope(self, side):
        if self.tmp_sp_chance_side in (side, co.ANY):
            scope = co.SET
            if self.tmp_mp_chance_side in (side, co.ANY):
                scope = co.MATCH
            return scope

    def inc_tmp_count(self, scope, srv, left, value):
        def inced_pair(pair):
            return (pair[0] + value, pair[1]) if left else (pair[0], pair[1] + value)

        if scope == co.SET:
            if srv:
                self.tmp_sp_srv = inced_pair(self.tmp_sp_srv)
            else:
                self.tmp_sp_rcv = inced_pair(self.tmp_sp_rcv)
        elif scope == co.MATCH:
            if srv:
                self.tmp_mp_srv = inced_pair(self.tmp_mp_srv)
            else:
                self.tmp_mp_rcv = inced_pair(self.tmp_mp_rcv)

    def get_tmp_count(self, scope, srv, left):
        def get_val(pair):
            return pair[0] if left else pair[1]

        if scope == co.SET:
            return get_val(self.tmp_sp_srv) if srv else get_val(self.tmp_sp_rcv)
        elif scope == co.MATCH:
            return get_val(self.tmp_mp_srv) if srv else get_val(self.tmp_mp_rcv)

    def get_counts(self, scope, srv, setnum=None):
        def impl(dictionary):
            if setnum is not None:
                return dictionary[setnum]
            fst, snd = 0, 0
            for sn in dictionary:
                val_t = dictionary[sn]
                fst += val_t[0]
                snd += val_t[1]
            return fst, snd

        if scope == co.SET:
            return impl(self.sp_srv_dct) if srv else impl(self.sp_rcv_dct)
        elif scope == co.MATCH:
            return impl(self.mp_srv_dct) if srv else impl(self.mp_rcv_dct)

    def _inc_count(self, scope, srv, setnum, left, value):
        def impl(dictionary):
            val_t = dictionary[setnum]
            if left:
                dictionary[setnum] = (val_t[0] + value, val_t[1])
            else:
                dictionary[setnum] = (val_t[0], val_t[1] + value)

        if scope == co.SET:
            if srv:
                impl(self.sp_srv_dct)
            else:
                impl(self.sp_rcv_dct)
        elif scope == co.MATCH:
            if srv:
                impl(self.mp_srv_dct)
            else:
                impl(self.mp_rcv_dct)

    def close_previous(self, prev_score, prev_left_service, prev_left_wingame):
        miss_side = co.RIGHT if prev_left_wingame else co.LEFT
        miss_side_left = miss_side.is_left()
        srv_side = co.side(prev_left_service)
        srv = miss_side == srv_side
        m_val = self.get_tmp_count(co.MATCH, srv, miss_side_left)
        if m_val > 0:
            setnum = len(prev_score)
            self._inc_count(co.MATCH, srv, setnum, miss_side_left, m_val)
        else:
            s_val = self.get_tmp_count(co.SET, srv, miss_side_left)
            if s_val > 0:
                setnum = len(prev_score)
                self._inc_count(co.SET, srv, setnum, miss_side_left, s_val)
        self.clear_tmp()

    def open_fresh(self, score, ingame, left_service, tiebreak, best_of_five):
        import miss_points

        score_t = score.tupled()
        self.tmp_sp_chance_side = miss_points.setball_chance_side(score_t[-1], tiebreak)
        if self.tmp_sp_chance_side is None:
            return
        self.tmp_mp_chance_side = miss_points.matchball_chance_side(
            score_t, best_of_five, self.tmp_sp_chance_side
        )

    def continue_current(
        self, prev_score, prev_ingame, prev_left_service, ingame, left_service, tiebreak
    ):
        def impl(edges, left_service):
            if not tiebreak:
                if edges[0] > 0:
                    scope = self.get_scope(co.LEFT)
                    if scope is not None:
                        self.inc_tmp_count(
                            scope, left_service, left=True, value=edges[0]
                        )
                elif edges[1] > 0:
                    scope = self.get_scope(co.RIGHT)
                    if scope is not None:
                        self.inc_tmp_count(
                            scope, not left_service, left=False, value=edges[1]
                        )
            else:
                pass  # TODO tempo

        if self.tmp_sp_chance_side is None:
            return
        num_ingame = sc.ingame_to_num(ingame, tiebreak=tiebreak)
        edges_cur = sc.num_ingame_bgpoint_counts(num_ingame, tiebreak=tiebreak)
        if edges_cur == (0, 0):
            return
        num_prev_ingame = sc.ingame_to_num(prev_ingame, tiebreak=tiebreak)
        edges_prev = sc.num_ingame_bgpoint_counts(num_prev_ingame, tiebreak=tiebreak)
        if edges_prev == (0, 0):
            impl(edges_cur, left_service)
        elif self.same_case(num_prev_ingame, num_ingame):
            return  # prev case already processed
        else:
            impl(edges_cur, left_service)

    @staticmethod
    def same_case(num_ingame1, num_ingame2):
        def max_where_t(num_ingame):
            if num_ingame[0] > num_ingame[1]:
                return num_ingame[0], 0
            elif num_ingame[0] < num_ingame[1]:
                return num_ingame[1], 1
            else:
                raise co.TennisError("bad num_ingame {} max_where_t".format(num_ingame))

        max_where_1 = max_where_t(num_ingame1)
        max_where_2 = max_where_t(num_ingame2)
        return max_where_1 == max_where_2


class AllowBreakPointStat:
    def __init__(self):
        self.bpcount_from_setnum = defaultdict(lambda: (0, 0))

    def get_compare(self, setnum):
        from detailed_score_misc import allow_breakpoints_compare

        val_t = self.bpcount_from_setnum[setnum]
        return allow_breakpoints_compare(val_t[0], val_t[1])

    def tostring(self, setnum=None):
        fst, snd = self.get_counts(setnum)
        return "abp{} {}-{}".format("" if setnum is None else str(setnum), fst, snd)

    def get_counts(self, setnum=None):
        if setnum is not None:
            return self.bpcount_from_setnum[setnum]
        fst, snd = 0, 0
        for sn in self.bpcount_from_setnum:
            val_t = self.bpcount_from_setnum[sn]
            fst += val_t[0]
            snd += val_t[1]
        return fst, snd

    def _inc_count(self, setnum, left, value):
        val_t = self.bpcount_from_setnum[setnum]
        if left:
            self.bpcount_from_setnum[setnum] = (val_t[0] + value, val_t[1])
        else:
            self.bpcount_from_setnum[setnum] = (val_t[0], val_t[1] + value)

    def close_previous(
        self, prev_left_wingame, prev_score, prev_ingame, prev_left_service
    ):
        bpcnt = self.allow_breakpoint_count_at(prev_ingame, prev_left_service)
        if bpcnt > 0:
            return  # already counted
        if prev_left_wingame is not prev_left_service:
            # service was not hold
            bpcnt = 1
            if prev_left_service and prev_ingame == ("0", "30"):
                bpcnt += 1  # probably was 2 bp
            elif not prev_left_service and prev_ingame == ("30", "0"):
                bpcnt += 1  # probably was 2 bp
        if bpcnt > 0:
            self._inc_count(setnum=len(prev_score), left=prev_left_service, value=bpcnt)

    def continue_current(self, prev_score, prev_ingame, prev_left_service, ingame):
        cur_bpcnt = self.allow_breakpoint_count_at(ingame, prev_left_service)
        if not cur_bpcnt:
            return
        prev_bpcnt = self.allow_breakpoint_count_at(prev_ingame, prev_left_service)
        if not prev_bpcnt:
            result_cnt = cur_bpcnt
        else:
            result_cnt = 0
            if prev_ingame != ingame and (
                (prev_left_service and ingame == ("40", "A"))
                or (not prev_left_service and ingame == ("A", "40"))
            ):
                result_cnt = 1
        if result_cnt > 0:
            self._inc_count(
                setnum=len(prev_score), left=prev_left_service, value=result_cnt
            )

    @staticmethod
    def allow_breakpoint_count_at(ingame, left_service):
        num_ingame = sc.ingame_to_num(ingame, tiebreak=False)
        return sc.num_ingame_breakpoint_count(num_ingame, left_service)


class AllowBreakPointStatTest(unittest.TestCase):
    def test_continue_bp(self):
        stat1 = AllowBreakPointStat()

        prev_score = sc.Score("5-2")
        prev_left_service = False
        prev_ingame = ("30", "30")
        ingame = ("40", "30")
        stat1.continue_current(prev_score, prev_ingame, prev_left_service, ingame)
        self.assertEqual((0, 1), stat1.get_counts())


class MatchLastInrow(object):
    def __init__(self):
        # here setnum -> LastInrowCount
        self.fst_inrowgames_from_setnum = defaultdict(LastInrowCount)

    def fst_count(self, setnum):
        return self.fst_inrowgames_from_setnum[setnum].count

    def close_game(self, setnum, prev_left_wingame):
        if setnum >= 3:
            return  # memory optimization
        self.fst_inrowgames_from_setnum[setnum].hit(1 if prev_left_wingame else -1)


class SetOpenerMatchTracker:
    """target is answer who is serve at concrete score"""

    def __init__(self):
        self.setnum_dict = defaultdict(SetOpenerHitCounter)

    def put(self, setnum, scr, is_left_service):
        """True if puted value is majority consilienced, False - not consilienced,
        None if not proof evidence"""
        if scr in ((6, 6), (12, 12)) or is_left_service is None:
            return
        if sum(scr) % 2 == 0:
            self.setnum_dict[setnum].put(is_left_service)
            opener_side = self.setnum_dict[setnum].get_opener_side()
            if opener_side is not None:
                return opener_side == co.side(is_left_service)
        else:
            self.setnum_dict[setnum].put(not is_left_service)
            opener_side = self.setnum_dict[setnum].get_opener_side()
            if opener_side is not None:
                return opener_side == co.side(not is_left_service)

    def get_opener_side(self, setnum, scr=(0, 0), at=True):
        """at=False is analog of service shift after scr (True is exactly at scr)"""
        set_open_side = self.setnum_dict[setnum].get_opener_side()
        if set_open_side is not None:
            if at:
                return set_open_side if sum(scr) % 2 == 0 else set_open_side.fliped()
            else:
                return set_open_side if sum(scr) % 2 == 1 else set_open_side.fliped()

    def is_left_opener(self, setnum, scr=(0, 0), at=True):
        """at=False is analog of service shift after scr (True is exactly at scr)"""
        opener_side = self.get_opener_side(setnum, scr, at)
        if opener_side is not None:
            return opener_side == co.LEFT


class TestSetOpener(unittest.TestCase):
    def test_match_track(self):
        trk = SetOpenerMatchTracker()
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, None)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, None)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)

        r = trk.put(setnum=2, scr=(0, 0), is_left_service=False)
        self.assertEqual(r, None)

        r = trk.get_opener_side(setnum=1, scr=(5, 3), at=True)
        self.assertEqual(r, co.LEFT)

        r = trk.get_opener_side(setnum=1, scr=(5, 3), at=False)
        self.assertEqual(r, co.RIGHT)

        r = trk.get_opener_side(setnum=1, scr=(5, 4), at=True)
        self.assertEqual(r, co.RIGHT)

        r = trk.get_opener_side(setnum=1, scr=(0, 0), at=True)
        self.assertEqual(r, co.LEFT)


class SetOpenerHitCounter:
    """target is work around noised live data about who is serve at current score"""

    min_vs_zero_count = 3
    min_vs_one_count = 6
    min_dominate_all_count = 10
    min_dominate_ratio = 0.8

    def __init__(self):
        self.left_side_count = 0
        self.right_side_count = 0

    def put(self, is_left_opener):
        if is_left_opener:
            self.left_side_count += 1
        else:
            self.right_side_count += 1

    def get_opener_side(self):
        all_count = self.left_side_count + self.right_side_count
        if all_count >= self.min_vs_zero_count and (
            self.left_side_count == 0 or self.right_side_count == 0
        ):
            return co.side(self.left_side_count >= self.right_side_count)
        if all_count > self.min_vs_one_count and (
            self.left_side_count == 1 or self.right_side_count == 1
        ):
            return co.side(self.left_side_count >= self.right_side_count)
        if all_count >= self.min_dominate_all_count:
            if self.left_side_count >= self.right_side_count:
                ratio = float(self.left_side_count) / float(all_count)
                if ratio >= self.min_dominate_ratio:
                    return co.LEFT
            else:
                ratio = float(self.right_side_count) / float(all_count)
                if ratio >= self.min_dominate_ratio:
                    return co.RIGHT


EdgeScr = recordclass("EdgeScr", ("x", "y", "left_serve"))


class EdgeScrTrack:
    __slots__ = ('scores',)

    def __init__(self):
        self.scores = []

    def put(self, x: int, y: int, left_serve: bool):
        if (x != 5 and y != 5) or x == y:
            return
        edge_scr = EdgeScr(x=x, y=y, left_serve=left_serve)
        if edge_scr not in self.scores:
            self.scores.append(edge_scr)

    def serveonset_count(self, side: Side):
        result = 0
        for edsc in self.scores:
            if (
                (edsc.x == 5 and edsc.y < 5 and edsc.left_serve and side.is_left())
                or
                (edsc.x < 5 and edsc.y == 5 and not edsc.left_serve and side.is_right())
                or
                (edsc.x == 6 and edsc.y == 5 and edsc.left_serve and side.is_left())
                or
                (edsc.x == 5 and edsc.y == 6 and not edsc.left_serve and side.is_right())
            ):
                result += 1
        return result


class EdgeScores:
    __slots__ = ('track_sn1', 'track_sn2')

    def __init__(self):
        self.track_sn1 = EdgeScrTrack()
        self.track_sn2 = EdgeScrTrack()

    def put(self, setnum: int, x: int, y: int, left_serve: bool):
        if setnum == 1:
            self.track_sn1.put(x, y, left_serve)
        elif setnum == 2:
            self.track_sn2.put(x, y, left_serve)

    def serveonset_count(self, setnum: int, side: Side):
        if setnum == 1:
            return self.track_sn1.serveonset_count(side)
        elif setnum == 2:
            return self.track_sn2.serveonset_count(side)


class QuadServiceStat(object):
    def __init__(self, det_score_items=None):
        # here _dct are dict{setnum -> WinLoss}
        self.fst_deuce_wl_dct = defaultdict(WinLoss)  # first(left) player
        self.fst_adv_wl_dct = defaultdict(WinLoss)  # first(left) player
        self.snd_deuce_wl_dct = defaultdict(WinLoss)
        self.snd_adv_wl_dct = defaultdict(WinLoss)
        if det_score_items is not None:
            self.apply_detailed_games(det_score_items)
        self.lastinrow = MatchLastInrow()  # who win last games serie in each set
        self.breakup_tracker = BreakupTracker()
        self.startgame_points_count = 0
        self.edge_scores = EdgeScores()

    def get_all(self, setnum=None):
        if setnum is None:
            fst_deuce_wl = sum(
                (self.fst_deuce_wl_dct[i] for i in range(1, 6)), WinLoss()
            )
            fst_adv_wl = sum((self.fst_adv_wl_dct[i] for i in range(1, 6)), WinLoss())
            snd_deuce_wl = sum(
                (self.snd_deuce_wl_dct[i] for i in range(1, 6)), WinLoss()
            )
            snd_adv_wl = sum((self.snd_adv_wl_dct[i] for i in range(1, 6)), WinLoss())
            return fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl
        else:
            return (
                self.fst_deuce_wl_dct[setnum],
                self.fst_adv_wl_dct[setnum],
                self.snd_deuce_wl_dct[setnum],
                self.snd_adv_wl_dct[setnum],
            )

    def win_counts(self, setnum):
        fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl = self.get_all(setnum=setnum)
        fst_wl = fst_deuce_wl + fst_adv_wl
        snd_wl = snd_deuce_wl + snd_adv_wl
        return (
            fst_wl.win_count + snd_wl.loss_count,
            snd_wl.win_count + fst_wl.loss_count,
        )

    def srv_win_loss(self, side, quadrant=None, setnum=None):
        fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl = self.get_all(setnum=setnum)
        if side.is_left():
            deuce_wl = fst_deuce_wl
            adv_wl = fst_adv_wl
        elif side.is_right():
            deuce_wl = snd_deuce_wl
            adv_wl = snd_adv_wl
        else:
            # return (left_result, right_result) for both
            return fst_deuce_wl + fst_adv_wl, snd_deuce_wl + snd_adv_wl
        if quadrant is None:
            return deuce_wl + adv_wl
        elif quadrant == co.DEUCE:
            return deuce_wl
        elif quadrant == co.ADV:
            return adv_wl
        else:
            raise co.TennisError("invalid quad {} in srv_win_loss".format(quadrant))

    def __str__(self):
        fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl = self.get_all()
        return "{} {} ({} {}) ({} {})".format(
            (fst_deuce_wl + fst_adv_wl).ratio_size_str(),  # for left
            (snd_deuce_wl + snd_adv_wl).ratio_size_str(),
            self.abbr_winloss_text(fst_deuce_wl, co.LEFT, co.DEUCE),
            self.abbr_winloss_text(fst_adv_wl, co.LEFT, co.ADV),
            self.abbr_winloss_text(snd_deuce_wl, co.RIGHT, co.DEUCE),
            self.abbr_winloss_text(snd_adv_wl, co.RIGHT, co.ADV),
        )

    @staticmethod
    def abbr_winloss_text(win_loss, side, quadrant):
        return "{}{}{}".format(
            "L" if side.is_left() else "R",
            "D" if quadrant == co.DEUCE else "A",
            win_loss.ratio_size_str(),
        )

    def thresholded_result(self, threshold, min_size):
        def update_result(current_result, win_loss, side, quadrant):
            if win_loss.size >= min_size and win_loss.ratio <= threshold:
                if current_result is None or current_result.ratio > win_loss.ratio:
                    return ThresholdedResult(
                        side=side, quadrant=quadrant, ratio=win_loss.ratio
                    )
            return current_result

        fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl = self.get_all()
        thr_result = None
        thr_result = update_result(thr_result, fst_deuce_wl, co.LEFT, co.DEUCE)
        thr_result = update_result(thr_result, fst_adv_wl, co.LEFT, co.ADV)
        thr_result = update_result(thr_result, snd_deuce_wl, co.RIGHT, co.DEUCE)
        thr_result = update_result(thr_result, snd_adv_wl, co.RIGHT, co.ADV)
        return thr_result

    def apply_detailed_games(self, det_score_items):
        for key, det_game in det_score_items:
            setnum = len(key)
            for point in det_game:
                if point.left_opener:
                    deuce_wl = self.fst_deuce_wl_dct[setnum]
                    adv_wl = self.fst_adv_wl_dct[setnum]
                else:
                    deuce_wl = self.snd_deuce_wl_dct[setnum]
                    adv_wl = self.snd_adv_wl_dct[setnum]
                ingame = point.num_score()
                if sc.serve_quadrant_at(ingame) == co.DEUCE:
                    deuce_wl.hit(point.win())
                else:
                    adv_wl.hit(point.win())

    def update_with(
        self, prev_score, prev_ingame, prev_left_service,
        score, ingame, left_service
    ):
        def empty(scr):
            return scr is None or len(scr) == 0

        if empty(prev_score) and empty(score):
            return
        if empty(prev_score):
            return self.__open_fresh(score, ingame, left_service)
        if empty(score):
            return
        try:
            prev_left_wingame = self.__get_prev_finish_info(prev_score, score)
            if prev_left_wingame is not None:
                self.__close_previous(
                    prev_left_wingame, prev_score, prev_ingame, prev_left_service
                )
                self.__open_fresh(score, ingame, left_service)
            else:
                self.__continue_current(
                    prev_score,
                    prev_ingame if prev_ingame else ("0", "0"),
                    prev_left_service,
                    ingame,
                )
        except QuadServiceStatLateError:
            self.__open_fresh(
                score, ingame, left_service
            )  # KeyError here will abort all

        except (QuadServiceStatError, KeyError, TypeError, ValueError) as err:
            log.exception(
                f"QuadError: {err} {err.__class__.__name__}\n"
                f"prsc {prev_score} pri {prev_ingame} prsrv {prev_left_service}"
                f"\nsc {score} in {ingame} srv {left_service}",
            )

    @staticmethod
    def is_tiebreak(score, ingame):
        def usial_ingame():
            if ingame and ingame != ("0", "0"):
                return sc.is_text_correct(ingame[0]) and sc.is_text_correct(ingame[1])

        return (
            len(score) > 0
            and score[-1] in ((6, 6), (7, 6), (6, 7))
            and usial_ingame() in (False, None)
        )

    def __fill(self, setnum, num_ingame_from, num_ingame_to, is_left_player):
        diff_ingame = (
            num_ingame_to[0] - num_ingame_from[0],
            num_ingame_to[1] - num_ingame_from[1],
        )
        min_common = min(diff_ingame)
        if min_common:
            diff_ingame = (diff_ingame[0] - min_common, diff_ingame[1] - min_common)
        num_points, win = 0, None
        if diff_ingame[0]:
            num_points = diff_ingame[0]
            win = is_left_player
        elif diff_ingame[1]:
            num_points = diff_ingame[1]
            win = not is_left_player
        if num_points:
            deuce_wl = self.fst_deuce_wl_dct[setnum]
            adv_wl = self.fst_adv_wl_dct[setnum]
            if not is_left_player:
                deuce_wl = self.snd_deuce_wl_dct[setnum]
                adv_wl = self.snd_adv_wl_dct[setnum]
            is_from_deuce = ((num_ingame_from[0] + num_ingame_from[1]) % 2) == 0
            if is_from_deuce:
                deuce_points = (num_points + 1) // 2
                adv_points = num_points // 2
            else:
                deuce_points = num_points // 2
                adv_points = (num_points + 1) // 2

            if win:
                deuce_wl.add_win(deuce_points)
                adv_wl.add_win(adv_points)
            else:
                deuce_wl.add_loss(deuce_points)
                adv_wl.add_loss(adv_points)

    def __fill_tie(self, setnum, num_ingame_from, num_ingame_to, srv_side_from):
        def fill_quad_points(quad_points):
            for quad_point in quad_points:
                if quad_point.srv_side == co.LEFT:
                    deuce_wl = self.fst_deuce_wl_dct[setnum]
                    adv_wl = self.fst_adv_wl_dct[setnum]
                else:
                    deuce_wl = self.snd_deuce_wl_dct[setnum]
                    adv_wl = self.snd_adv_wl_dct[setnum]
                wl = deuce_wl if quad_point.quadrant == co.DEUCE else adv_wl
                wl.hit(quad_point.srv_winpoint)

        diff_ingame = (
            num_ingame_to[0] - num_ingame_from[0],
            num_ingame_to[1] - num_ingame_from[1],
        )
        min_common = min(diff_ingame)
        only_one_side_win = min_common == 0 and diff_ingame[0] != diff_ingame[1]
        if not only_one_side_win:
            return
        side_winpoint = co.LEFT if diff_ingame[0] > diff_ingame[1] else co.RIGHT
        fill_quad_points(
            sc.tie_quadrant_points(
                num_ingame_from,
                srv_side_from,
                num_points=sum(num_ingame_to) - sum(num_ingame_from),
                side_winpoint=side_winpoint,
            )
        )

    def __close_previous(
        self, prev_left_wingame, prev_score, prev_ingame, prev_left_service
    ):
        def get_num_ingame_to(num_ingame_from, points_to_win):
            if prev_left_wingame:
                return num_ingame_from[0] + points_to_win[0], num_ingame_from[1]
            else:
                return num_ingame_from[0], num_ingame_from[1] + points_to_win[1]

        if prev_score is None or prev_ingame is None or prev_left_service is None:
            return

        sn = len(prev_score)
        self.lastinrow.close_game(sn, prev_left_wingame)
        if self.is_tiebreak(prev_score, prev_ingame):
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=True)
            points_to_win = sc.num_ingame_to_win(num_ingame_from, tiebreak=True)
            num_ingame_to = get_num_ingame_to(num_ingame_from, points_to_win)
            self.__fill_tie(
                sn, num_ingame_from, num_ingame_to, co.side(prev_left_service)
            )
        else:
            prev_inset_scr = prev_score[-1]
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=False)
            points_to_win = sc.num_ingame_to_win(num_ingame_from, tiebreak=False)
            num_ingame_to = get_num_ingame_to(num_ingame_from, points_to_win)
            self.__fill(sn, num_ingame_from, num_ingame_to, prev_left_service)
            self.breakup_tracker.close_previous(
                prev_score, prev_left_service, prev_left_wingame
            )
            if sn <= 2:
                self.edge_scores.put(
                    setnum=sn, x=prev_inset_scr[0], y=prev_inset_scr[1],
                    left_serve=prev_left_service)

    def __continue_current(self, prev_score, prev_ingame, prev_left_service,
                           ingame):
        if prev_ingame == ingame:
            return
        sn = len(prev_score)
        if self.is_tiebreak(prev_score, prev_ingame):
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=True)
            num_ingame_to = sc.ingame_to_num(ingame, tiebreak=True)
            self.__fill_tie(
                sn, num_ingame_from, num_ingame_to, co.side(prev_left_service)
            )
        else:
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=False)
            num_ingame_to = sc.ingame_to_num(ingame, tiebreak=False)
            self.__fill(sn, num_ingame_from, num_ingame_to, prev_left_service)
            if prev_score[0] == (0, 0):
                self.startgame_points_count += 1

    def __open_fresh(self, score, ingame, left_service):
        if ingame in (None, ("0", "0")):
            return
        sn = len(score)
        if self.is_tiebreak(score, ingame):
            num_ingame_to = sc.ingame_to_num(ingame, tiebreak=True)
            tie_open_side = sc.tie_opener_side(num_ingame_to, co.side(left_service))
            self.__fill_tie(sn, (0, 0), num_ingame_to, tie_open_side)
        else:
            num_ingame_to = sc.ingame_to_num(ingame, tiebreak=False)
            self.__fill(sn, (0, 0), num_ingame_to, left_service)

    @staticmethod
    def __get_prev_finish_info(prev_score, score):
        """return left_wingame if prev finished, None if not finished.
        raise QuadServiceStatError if nonsense data"""
        prev_len, cur_len = len(prev_score), len(score)
        if prev_len < cur_len and cur_len >= 2:
            prev_full_set_sc = score[-2]
            is_prev_full_set = sc.is_full_set(prev_full_set_sc)
            if not is_prev_full_set:
                raise QuadServiceStatLateError(
                    "prevscore {} toscore {}".format(prev_score, score))
            if (
                sc.exist_point_increment(prev_score[-1], prev_full_set_sc)
                or score[-1] == (0, 0)
                # возм. также случай score[-1] == внутри-tie счет из завершенного tie
                # min(score[-1]) == prev_score.idx_tiemin[prev_len-1]
                # тогда надо как-то обойти молчанием этот выход наружу внутри-tie счета
            ):
                return prev_full_set_sc[0] > prev_full_set_sc[1]
        elif prev_len > cur_len:
            raise QuadServiceStatError(
                "prevscore {} toscore {}".format(prev_score, score)
            )
        elif prev_len == cur_len:
            # set is continues
            prev_set_sc, cur_set_sc = prev_score[-1], score[-1]
            if prev_set_sc == cur_set_sc:
                return None  # game is continues
            elif sc.exist_point_increment(prev_set_sc, cur_set_sc):
                return prev_set_sc[0] < cur_set_sc[0]  # finished nicely
            else:
                if prev_set_sc[0] == cur_set_sc[0]:
                    return False  # right win prev game
                elif prev_set_sc[1] == cur_set_sc[1]:
                    return True  # left win prev game
                else:
                    # we are late and do not know who did win
                    raise QuadServiceStatLateError(
                        f"prevscore {prev_score} toscore {score} (set is continues)")


class QuadServiceStatTest(unittest.TestCase):
    def test_close_by_right(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("5-1"),
            prev_ingame=("30", "40"),
            prev_left_service=False,
            score=sc.Score("5-2"),
            ingame=("0", "0"),
            left_service=True,
        )

        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))

        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.LEFT))

    def test_close_by_left_then_fresh_right(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("5-1"),
            prev_ingame=("30", "40"),
            prev_left_service=False,
            score=sc.Score("6-1 0-0"),
            ingame=("0", "40"),
            left_service=True,
        )
        # closing: right broken by 3 points
        self.assertEqual(WinLoss(0, 2), qstat1.srv_win_loss(co.RIGHT, co.ADV))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))

        # freshing: left loss 3 points
        self.assertEqual(WinLoss(0, 2), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.LEFT, co.DEUCE))

    def test_close_by_left_then_fresh_right_tiebreak(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("6-4 5-6"),
            prev_ingame=("30", "40"),
            prev_left_service=True,
            score=sc.Score("6-4 6-6"),
            ingame=("0", "2"),
            left_service=True,
        )
        # closing: left hold by win 3 points: ADV(2), DEUCE(1)
        # freshing: right by win 1 point: DEUCE(1)
        # freshing: left by loss 1 point: ADV(0)
        self.assertEqual(WinLoss(2, 1), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.LEFT, co.DEUCE))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))

    def test_close_by_left_then_fresh_left_tiebreak(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("6-4 5-6"),
            prev_ingame=("30", "40"),
            prev_left_service=True,
            score=sc.Score("6-4 6-6"),
            ingame=("2", "0"),
            left_service=True,
        )
        # closing: left hold by win 3 points: ADV(2), DEUCE(1)
        # freshing: right opener by loss 1 point: DEUCE(0)
        # freshing: left by win 1 point: ADV(1)
        self.assertEqual(WinLoss(3, 0), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.LEFT, co.DEUCE))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))


class TotalSlices(object):
    def __init__(self):
        self._totcollect_from_key = defaultdict(TotalCollector)

    def get_keys(self):
        return list(self._totcollect_from_key.keys())

    def get_max_keys_cardinality(self):
        return max([k.cardinality() for k in self._totcollect_from_key.keys()])

    def addition(self, score, keys):
        normalized_total = score.normalized_total()
        for key in keys:
            self._totcollect_from_key[key].addition(normalized_total)

    def substract(self, score, keys):
        normalized_total = score.normalized_total()
        for key in keys:
            self._totcollect_from_key[key].substract(normalized_total)
            if sum(self._totcollect_from_key[key].slots_hits) == 0:
                del self._totcollect_from_key[key]

    def report_line_list(self, fun_name, keys):
        items = []
        for key in keys:
            if key in self._totcollect_from_key:
                totcollect = self._totcollect_from_key[key]
                items.append(
                    rl.ReportLine(
                        key=key,
                        value=getattr(totcollect, fun_name)(),
                        size=totcollect.hits_count(),
                    )
                )
        return rl.ReportLineList(items=items)

    def write_report(
        self,
        filename,
        memfun_name,
        sort_asc,
        sort_keyfun=itemgetter(1),
        threshold=0,
        head="",
    ):
        """Write to filename patterns: 'key value (size)\n'.
        Sort on value (default) according to sort_asc."""
        if len(self._totcollect_from_key) == 0:
            return
        key_val_num = []
        for key, totcollector in self._totcollect_from_key.items():
            if totcollector.hits_count() >= threshold:
                val = getattr(totcollector, memfun_name)()
                key_val_num.append((str(key), val, totcollector.hits_count()))
        if sort_asc is True:
            key_val_num.sort(key=sort_keyfun)
        elif sort_asc is False:
            key_val_num.sort(key=sort_keyfun, reverse=True)

        key_len_max = max([len(str(k)) for k in self._totcollect_from_key.keys()])
        with open(filename, "w") as fhandle:
            if head:
                fhandle.write(head + "\n")
            fmt = "{0:_<" + str(key_len_max + 2) + "}  {1:5.2f} ({2})\n"
            for key, value, num in key_val_num:
                fhandle.write(fmt.format(key, value, num))

    def value(self, key, memfun_name):
        totcollector = self._totcollect_from_key[key]
        return getattr(totcollector, memfun_name)()

    def size(self, key):
        totcollector = self._totcollect_from_key[key]
        return totcollector.hits_count()

    def get_collector(self, key):
        return self._totcollect_from_key[key]


def slices_table_report(
    fun_name, slices, titles, idx1, idx2, min_diff, min_size, filter_keys, filename
):
    """output slices data which |slices[idx1].value[k] - slices[idx2].value[k]| > min_diff
    for k in filter_keys.
    If filter_keys is None then used keys from slices[idx1].get_keys(),
                            slices[idx2].get_keys() with max key cardinality.
    """
    assert len(slices) == len(titles), "len(slices) == len(titles) not supplied"

    def keys_selected(fun_name, filter_keys, slices, idx1, idx2, min_diff, min_size):
        assert len(slices) > max(idx1, idx2), "len(slices) >= 3 not supplied"
        selected = []
        if filter_keys:
            use_keys = filter_keys
        else:
            use_keys = list(set(slices[idx1].get_keys()) & set(slices[idx2].get_keys()))
            max_key_cardinality = max([k.cardinality() for k in use_keys])
            use_keys = [k for k in use_keys if k.cardinality() == max_key_cardinality]
        for filter_key in use_keys:
            if (
                slices[idx1].size(filter_key) >= min_size
                and slices[idx2].size(filter_key) >= min_size
            ):
                value1 = slices[idx1].value(filter_key, fun_name)
                value2 = slices[idx2].value(filter_key, fun_name)
                if abs(value1 - value2) >= min_diff:
                    selected.append(filter_key)
        return selected

    sel_keys = keys_selected(
        fun_name, filter_keys, slices, idx1, idx2, min_diff, min_size
    )
    if len(sel_keys) > 0:
        key_len_max = max([len(str(k)) for k in sel_keys])
        titles_line = " " * (key_len_max + 2)
        for title in titles:
            titles_line += title + "\t"
        key_fmt = "{0:_<" + str(key_len_max + 2) + "} "
        with open(filename, "w") as fh:
            fh.write(titles_line + "\n")
            for filter_key in sorted(sel_keys):
                fh.write(key_fmt.format(filter_key))
                for slc in slices:  # columns
                    value = slc.value(filter_key, fun_name)
                    size = slc.size(filter_key)
                    fh.write(
                        "{0:5.2f} ({1}) {2}".format(
                            value, size, " " * (5 - len(str(size)))
                        )
                    )
                fh.write("\n")


class TotalCollector(object):
    slots_count = 16
    begin_slot_value = 12

    def __init__(self):
        self.slots_hits = []
        for _ in range(TotalCollector.slots_count):
            self.slots_hits.append(0)

    def addition(self, normalized_total):
        slot_idx = normalized_total - TotalCollector.begin_slot_value
        assert slot_idx < TotalCollector.slots_count, "unexpected slot_idx: {}".format(
            slot_idx
        )
        self.slots_hits[slot_idx] += 1

    def substract(self, normalized_total):
        slot_idx = normalized_total - TotalCollector.begin_slot_value
        assert slot_idx < TotalCollector.slots_count, "unexpected slot_idx: {}".format(
            slot_idx
        )
        assert (
            self.slots_hits[slot_idx] > 0
        ), "unexpected slot_idx: {} with zero count".format(slot_idx)
        self.slots_hits[slot_idx] -= 1

    def hits_count(self):
        return sum(self.slots_hits)

    def mean_value(self):
        hits_num = self.hits_count()
        assert hits_num > 0, "can not def mean_value without hits"
        result = 0
        slot_value = TotalCollector.begin_slot_value
        for idx in range(TotalCollector.slots_count):
            result += slot_value * self.slots_hits[idx]
            slot_value += 1
        return float(result) / float(hits_num)

    def estimate_value(self, up_idx=7):
        hits_num = self.hits_count()
        assert hits_num > 0, "can not def mean_value without hits"
        assert up_idx < TotalCollector.slots_count, "up_idx is out of range"

        low_flip_sum = 0
        for idx in range(up_idx):
            low_flip_sum += (
                co.centered_int_flip(idx + 1, 1, up_idx) * self.slots_hits[idx]
            )
        if low_flip_sum == 0:
            return 999.0

        up_sum = 0
        for idx in range(up_idx, TotalCollector.slots_count):
            up_sum += (idx + 1) * self.slots_hits[idx]

        return float(up_sum) / float(low_flip_sum)

    def estimate_value2(self, up_idx=7):
        hits_num = self.hits_count()
        assert hits_num > 0, "can not def estimate_value2 without hits"
        assert up_idx < TotalCollector.slots_count, "up_idx is out of range"

        (low_sum, low_hits_num) = (0, 0)
        for idx in range(up_idx):
            low_sum += (idx + 1) * self.slots_hits[idx]
            low_hits_num += self.slots_hits[idx]
        if low_hits_num == 0:
            return 999.0

        (up_sum, up_hits_num) = (0, 0)
        for idx in range(up_idx, TotalCollector.slots_count):
            up_sum += (idx + 1) * self.slots_hits[idx]
            up_hits_num += self.slots_hits[idx]
        if up_hits_num == 0:
            return 0.0

        return (float(up_sum) / float(up_hits_num)) / co.centered_float_flip(
            float(low_sum) / float(low_hits_num), 1, up_idx
        )

    def estimate_value2_test(self, up_idx=7):
        hits_num = self.hits_count()
        assert hits_num > 0, "can not def estimate_value2 without hits"
        assert up_idx < TotalCollector.slots_count, "up_idx is out of range"

        (low_sum, low_hits_num) = (0, 0)
        for idx in range(up_idx):
            low_sum += (idx + 1) * self.slots_hits[idx]
            low_hits_num += self.slots_hits[idx]
        if low_hits_num == 0:
            return 999.0

        (up_sum, up_hits_num) = (0, 0)
        for idx in range(up_idx, TotalCollector.slots_count):
            up_sum += (idx + 1) * self.slots_hits[idx]
            up_hits_num += self.slots_hits[idx]
        if up_hits_num == 0:
            return 0.0

        low = float(low_sum) / float(low_hits_num)
        low_fliped = co.centered_float_flip(low, 1, up_idx)
        up = float(up_sum) / float(up_hits_num)
        estimate = up / low_fliped
        return estimate, low, low_fliped, low_sum, low_hits_num, up, up_sum, up_hits_num

    def ratio_by_cmpfun(self, cmpfun):
        true_count, false_count = (0, 0)
        slot_value = TotalCollector.begin_slot_value
        for idx in range(TotalCollector.slots_count):
            if cmpfun(slot_value):
                true_count += self.slots_hits[idx]
            else:
                false_count += self.slots_hits[idx]
            slot_value += 1
        assert (
            true_count > 0 or false_count > 0
        ), "empty TotalCollector when ratio_by_cmpfun"
        return float(true_count) / float(true_count + false_count)

    def ratio_by_le(self, value):
        return self.ratio_by_cmpfun(lambda x: operator.le(x, value))

    def ratio_by_le_18(self):
        return self.ratio_by_le(18)

    def ratio_by_le_19(self):
        return self.ratio_by_le(19)

    def ratio_by_le_20(self):
        return self.ratio_by_le(20)

    def ratio_by_le_21(self):
        return self.ratio_by_le(21)

    def ratio_by_le_22(self):
        return self.ratio_by_le(22)

    def ratio_by_le_23(self):
        return self.ratio_by_le(23)


if __name__ == "__main__":
    unittest.main()
