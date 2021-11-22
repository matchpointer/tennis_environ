from collections import defaultdict, namedtuple
import operator
import re
import functools

import common as co
import log
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

    @staticmethod
    def from_seq(iterable):
        result = Sumator()
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

    def add_win(self, count: int):
        self.win_count += count

    def add_loss(self, count: int):
        self.loss_count += count

    @staticmethod
    def create_from_ratio(win_ratio, size):
        win_count = int(round(float(size) * win_ratio))
        loss_count = size - win_count
        return WinLoss(win_count, loss_count)

    @staticmethod
    def create_from_text(text):
        match = WinLoss.text_re.match(text)
        if match:
            multiplier = 0.01 if match.group("percent_sign") else 1.0
            win_ratio = float(match.group("value")) * multiplier
            size = int(match.group("size"))
            return WinLoss.create_from_ratio(win_ratio, size)
        raise co.TennisError("unparsed WinLoss text: '{}'".format(text))

    @staticmethod
    def create_from_iter(iterable):
        """make object from iter of booleans"""
        win_count, loss_count = 0, 0
        for val in iterable:
            if val:
                win_count += 1
            else:
                loss_count += 1
        return WinLoss(win_count, loss_count)

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


class QuadServiceStatError(co.TennisError):
    pass


class QuadServiceStatLateError(QuadServiceStatError):
    pass


ThresholdedResult = namedtuple("ThresholdedResult", "side quadrant ratio")


class BreakupTrack:
    __slots__ = ("is_fst_breakup", "is_snd_breakup")

    def __init__(self):
        self.is_fst_breakup = None
        self.is_snd_breakup = None

    def put(self, scr, left_service: bool, left_win_game: bool):
        if self.is_fst_breakup and self.is_snd_breakup:
            return
        if left_service is left_win_game:
            return  # hold
        x, y = scr
        if x <= y and left_service:
            self.is_snd_breakup = True
        elif y <= x and left_service is False:
            self.is_fst_breakup = True


class BreakupTracker:
    MAX_SETNUM = 2  # for time & memory economy

    def __init__(self):
        # setnum -> BreakupTrack
        self.track_from_setnum = defaultdict(BreakupTrack)

    def is_fst_breakup(self, setnum):
        if setnum is not None:
            return self.track_from_setnum[setnum].is_fst_breakup

    def is_snd_breakup(self, setnum):
        if setnum is not None:
            return self.track_from_setnum[setnum].is_snd_breakup

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


class AllowBreakPointStat(object):
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


class QuadServiceStat(object):
    def __init__(self, det_score_items=None):
        # here _dct are dict{setnum -> WinLoss}
        self.fst_deuce_wl_dct = defaultdict(WinLoss)  # first(left) player
        self.fst_adv_wl_dct = defaultdict(WinLoss)  # first(left) player
        self.snd_deuce_wl_dct = defaultdict(WinLoss)
        self.snd_adv_wl_dct = defaultdict(WinLoss)
        if det_score_items is not None:
            self.apply_detailed_games(det_score_items)
        self.allow_bp_stat = AllowBreakPointStat()
        self.lastinrow = MatchLastInrow()  # who win last games serie in each set
        self.breakup_tracker = BreakupTracker()

    def maxmin_for_tie(
        self, back_side, back_opener, min_size=19, min_thresh=0.58, max_thresh=0.58
    ):
        def idx_to_side(idx):
            return co.LEFT if idx <= 1 else co.RIGHT

        fst_deuce_wl, fst_adv_wl, snd_deuce_wl, snd_adv_wl = self.get_all()
        if (
            fst_deuce_wl.size < min_size
            or fst_adv_wl.size < min_size
            or snd_deuce_wl.size < min_size
            or snd_adv_wl.size < min_size
        ):
            return None
        values = (
            fst_deuce_wl.ratio,
            fst_adv_wl.ratio,
            snd_deuce_wl.ratio,
            snd_adv_wl.ratio,
        )
        max_idx, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if max_val < max_thresh or idx_to_side(max_idx) != back_side:
            return None
        if back_opener:
            max_num = 1 if max_idx in (0, 2) else 4  # opener tie num
        else:
            max_num = 2 if max_idx in (1, 3) else 3  # other tie num

        min_idx, min_val = min(enumerate(values), key=operator.itemgetter(1))
        if min_val <= min_thresh and idx_to_side(min_idx) != back_side:
            if back_opener:
                min_num = 2 if min_idx in (1, 3) else 3  # other tie num
            else:
                min_num = 1 if min_idx in (0, 2) else 4  # opener tie num
            return max_num, min_num
        else:
            return (max_num,)

    def allow_bp_counts(self, setnum=None):
        return self.allow_bp_stat.get_counts(setnum=setnum)

    def allow_bp_compare(self, setnum):
        return self.allow_bp_stat.get_compare(setnum)

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
        self, prev_score, prev_ingame, prev_left_service, score, ingame, left_service
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
            log.error(
                f"QuadError: {err} {err.__class__.__name__}\n"
                f"prsc {prev_score} pri {prev_ingame} prsrv {prev_left_service}"
                f"\nsc {score} in {ingame} srv {left_service}",
                exc_info=True,
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
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=False)
            points_to_win = sc.num_ingame_to_win(num_ingame_from, tiebreak=False)
            num_ingame_to = get_num_ingame_to(num_ingame_from, points_to_win)
            self.__fill(sn, num_ingame_from, num_ingame_to, prev_left_service)
            self.allow_bp_stat.close_previous(
                prev_left_wingame, prev_score, prev_ingame, prev_left_service
            )
            self.breakup_tracker.close_previous(
                prev_score, prev_left_service, prev_left_wingame
            )

    def __continue_current(self, prev_score, prev_ingame, prev_left_service, ingame):
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
            prev_inset_scr = prev_score[-1]
            num_ingame_from = sc.ingame_to_num(prev_ingame, tiebreak=False)
            num_ingame_to = sc.ingame_to_num(ingame, tiebreak=False)
            self.__fill(sn, num_ingame_from, num_ingame_to, prev_left_service)
            self.allow_bp_stat.continue_current(
                prev_score, prev_ingame, prev_left_service, ingame
            )

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
            if sc.exist_point_increment(
                prev_score[-1], prev_full_set_sc
            ) and sc.is_full_set(prev_full_set_sc):
                return prev_full_set_sc[0] > prev_full_set_sc[1]
            else:
                raise QuadServiceStatLateError(
                    "prevscore {} toscore {}".format(prev_score, score)
                )
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
                        "prevscore {} toscore {}".format(prev_score, score)
                    )
