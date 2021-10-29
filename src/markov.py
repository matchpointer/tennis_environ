r""" module for calculate importances and probabilities.
"""

import copy
import unittest
from collections import defaultdict, namedtuple
from typing import Optional, List

import common as co
import score as sc
import log
from side import Side
import matchstat


def _prob_round(prob):
    if prob is not None:
        return round(prob, 2)


class ProbsPair(object):
    def __init__(self, first, second=None):
        assert first is not None, "first arg is None in ProbsPair"
        self.first = _prob_round(first)
        self.second = _prob_round(second) if second is not None else self.first

    def __eq__(self, other):
        return (self.first, self.second) == (other.first, other.second)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return self.first is not None or self.second is not None

    __nonzero__ = __bool__

    def __hash__(self):
        return hash((self.first, self.second))

    def __str__(self):
        return "({}, {})".format(self.first, self.second)

    def __getitem__(self, index):
        if index == 0:
            return self.first
        elif index == 1:
            return self.second
        else:
            raise IndexError("bad ProbsPair index %d" % index)

    def __len__(self):
        return 2

    def __iter__(self):
        return iter((self.first, self.second))

    def reversed(self):
        return ProbsPair(first=self.second, second=self.first)


def make_probs_dict(win_probs, hold_probs):
    return {
        co.POINT: ProbsPair(win_probs[0], win_probs[1]),
        co.GAME: ProbsPair(hold_probs[0], hold_probs[1]),
    }


def reversed_probs_dict(probs_dict):
    result = {}
    for key, probs in probs_dict.items():
        result[key] = probs.reversed()
    return result


class CurrentState(object):
    def __init__(self, state, left_opener):
        self.state = state
        self.left_opener = left_opener


class CurrentContext(object):
    def __init__(self, match_cur_state, set_cur_state, game_cur_state, probs_dict):
        self.match_cur_state = match_cur_state
        self.set_cur_state = set_cur_state
        self.game_cur_state = game_cur_state
        self.probs_dict = probs_dict

    def point_importance(self):
        mcs, scs, gcs = self.match_cur_state, self.set_cur_state, self.game_cur_state
        if (
            not gcs.up_next
            or not gcs.down_next
            or not scs.up_next
            or not scs.down_next
            or not mcs.up_next
            or not mcs.down_next
        ):
            return None
        wmup = gcs.up_next.left_win_prob * (
            scs.up_next.left_win_prob * mcs.up_next.left_win_prob
            + (1.0 - scs.up_next.left_win_prob) * mcs.down_next.left_win_prob
        )
        wmdn = gcs.down_next.left_win_prob * (
            scs.down_next.left_win_prob * mcs.up_next.left_win_prob
            + (1.0 - scs.down_next.left_win_prob) * mcs.down_next.left_win_prob
        )
        return wmup - wmdn


class State(object):
    allfield_states = None
    probstates_from_probs = None
    initprob_from_probs = None

    """ состояние при счете left_count, right_count,
    """

    def __init__(self, left_count, right_count, left_win_prob=None):
        self.left_count = left_count
        self.right_count = right_count
        self._normalize_counts()

        # probability of win whole Unit from this state
        self.left_win_prob = left_win_prob
        win_side = self.terminated()
        if win_side == co.LEFT:
            self.left_win_prob = 1.0
        elif win_side == co.RIGHT:
            self.left_win_prob = 0.0

        self.up_next = None
        self.down_next = None

    @staticmethod
    def min_for_win():
        raise NotImplementedError()

    @staticmethod
    def prob_key(probs_dict):
        raise NotImplementedError()

    @classmethod
    def get_init_prob(cls, probs_dict):
        pkey = cls.prob_key(probs_dict)
        return cls.initprob_from_probs[pkey]

    def left_upstep_prob(self, probs_dict):
        raise NotImplementedError()

    @property
    def counts(self):
        return self.left_count, self.right_count

    @property
    def opener_side(self):
        """who serving in begining of this cur unit if left opened units-container"""
        if ((self.left_count + self.right_count) % 2) == 0:
            return co.LEFT
        else:
            return co.RIGHT

    def terminated(self):
        raise NotImplementedError()

    def _normalize_counts(self):
        pass

    def __hash__(self):
        return hash((self.left_count, self.right_count, self.opener_side))

    def __str__(self):
        return "({},{}) op:{} P:{}".format(
            self.left_count, self.right_count, self.opener_side, self.left_win_prob
        )

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.left_count, self.right_count
        )

    def __eq__(self, other):
        return (
            self.left_count == other.left_count
            and self.right_count == other.right_count
            and self.opener_side == other.opener_side
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if (self.left_count, self.right_count) < (other.left_count, other.right_count):
            return True
        if (self.left_count, self.right_count) == (other.left_count, other.right_count):
            return self.opener_side < other.opener_side
        return False

    @classmethod
    def step_forward_state(cls, from_state, advance_side):
        if from_state.terminated():
            return None

        if advance_side.is_left():
            return cls(from_state.left_count + 1, from_state.right_count)
        else:
            return cls(from_state.left_count, from_state.right_count + 1)

    @classmethod
    def build_allfield_states(cls):
        def step_advance():
            adv_states = set()
            for state in result_states:
                if state.terminated():
                    continue
                for advance_side in (co.LEFT, co.RIGHT):
                    new_state = state.step_forward_state(state, advance_side)
                    if new_state is not None and new_state not in adv_states:
                        adv_states.add(new_state)
            prev_len = len(result_states)
            result_states.update(adv_states)
            return prev_len < len(result_states)

        result_states = {cls(0, 0)}
        grow = True
        while grow:
            grow = step_advance()
        cls.allfield_states = copy.copy(result_states)
        return result_states

    @classmethod
    def build_all(cls, probs_dict):
        cls.build_prob_states(probs_dict)
        cls.build_initprob(probs_dict)

    @classmethod
    def build_prob_states(cls, probs_dict):
        key = cls.prob_key(probs_dict)
        if cls.probstates_from_probs[key]:
            return  # already done
        start_states, nostart_states = cls.split_start_states(probs_dict)
        term_states, states = split_term_states(nostart_states)
        prob_states = start_states + term_states
        make_prob_states(states, prob_states, probs_dict)
        assert not states, "notprobedstates_len: %d" % len(states)
        cls.probstates_from_probs[key] = prob_states

    @classmethod
    def check_prob_states(cls, probs_dict):
        """return empty str if OK, else errors string"""
        result = ""
        key = cls.prob_key(probs_dict)
        if cls.probstates_from_probs[key]:
            for probstate in cls.probstates_from_probs[key]:
                if probstate.left_win_prob is None:
                    result += str(probstate) + "\n"
        return result

    @classmethod
    def build_initprob(cls, probs_dict):
        key = cls.prob_key(probs_dict)
        if cls.initprob_from_probs[key]:
            return  # already done
        for state in cls.probstates_from_probs[key]:
            if state.counts == (0, 0):
                cls.initprob_from_probs[key] = state.left_win_prob
                break

    @classmethod
    def split_start_states(cls, probs_dict):
        """return (start_states, nostart_states)"""
        field_states = copy.copy(cls.allfield_states)
        start_states = cls.start_states(probs_dict)
        # log_states(start_states, "start_states", with_classname=True) # tempo
        for start_state in start_states:
            field_states.remove(start_state)
        return start_states, list(field_states)

    @classmethod
    def start_states(cls, probs_dict):
        raise NotImplementedError()


class GameState(State):
    """обычный гейм"""

    allfield_states = None
    probstates_from_probs = defaultdict(list)
    initprob_from_probs = defaultdict(lambda: None)

    def __init__(self, left_count, right_count, left_win_prob=None):
        super(GameState, self).__init__(left_count, right_count, left_win_prob)

    @staticmethod
    def min_for_win():
        return 4

    @staticmethod
    def prob_key(probs_dict):
        return probs_dict[co.POINT]  # point win prob at srv

    def left_upstep_prob(self, probs_dict):
        win_probs = probs_dict[co.POINT]
        return win_probs[0] if self.opener_side.is_left() else 1.0 - win_probs[1]

    @property
    def opener_side(self):
        """who serving in this cur game"""
        return co.LEFT

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if abs(self.left_count - self.right_count) <= 1:
            return None
        if self.min_for_win() <= self.left_count and self.left_count > self.right_count:
            return co.LEFT
        if (
            self.min_for_win() <= self.right_count
            and self.right_count > self.left_count
        ):
            return co.RIGHT

    def _normalize_counts(self):
        if self.left_count == self.right_count:
            if self.left_count >= self.min_for_win():
                self.left_count = self.min_for_win() - 1
                self.right_count = self.min_for_win() - 1
        else:
            max_count = max(self.left_count, self.right_count)
            if max_count > self.min_for_win():
                diff = max_count - self.min_for_win()
                self.left_count -= diff
                self.right_count -= diff

    @classmethod
    def start_states(cls, probs_dict):
        win_probs = probs_dict[co.POINT]
        pa = win_probs[0]
        pwin_from_40_40 = (pa * pa) / (pa * pa + (1.0 - pa) * (1.0 - pa))
        state_40_40 = GameState(3, 3, left_win_prob=pwin_from_40_40)

        delim = 1.0 - 2.0 * pa + 2.0 * pa * pa
        pwin_from_ADR = (pa * pa * pa) / delim
        state_40_A = GameState(3, 4, left_win_prob=pwin_from_ADR)

        pwin_from_ADS = pa * (1.0 - pa + pa * pa) / delim
        state_A_40 = GameState(4, 3, left_win_prob=pwin_from_ADS)

        return [
            GameState(4, 0),
            GameState(4, 1),
            GameState(4, 2),
            GameState(0, 4),
            GameState(1, 4),
            GameState(2, 4),
            state_40_40,
            state_40_A,
            state_A_40,
        ]


class TiebreakState(State):
    allfield_states = None
    probstates_from_probs = defaultdict(list)
    initprob_from_probs = defaultdict(lambda: None)

    def __init__(self, left_count, right_count, left_win_prob=None):
        super(TiebreakState, self).__init__(left_count, right_count, left_win_prob)

    @staticmethod
    def min_for_win():
        return 7

    @staticmethod
    def prob_key(probs_dict):
        return probs_dict[co.POINT]

    def left_upstep_prob(self, probs_dict):
        win_probs = probs_dict[co.POINT]
        return win_probs[0] if self.opener_side.is_left() else 1.0 - win_probs[1]

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if abs(self.left_count - self.right_count) <= 1:
            return None
        if self.min_for_win() <= self.left_count and self.left_count > self.right_count:
            return co.LEFT
        if (
            self.min_for_win() <= self.right_count
            and self.right_count > self.left_count
        ):
            return co.RIGHT

    def _normalize_counts(self):
        while (self.left_count + self.right_count) >= 14:
            self.left_count -= 2
            self.right_count -= 2

    @property
    def opener_side(self):
        """who serving in this point if left opened tie"""
        return sc.tie_serve_side_at((self.left_count, self.right_count), co.LEFT)

    @classmethod
    def start_states(cls, probs_dict):
        win_probs = probs_dict[co.POINT]
        pa = win_probs[0]
        qa = 1.0 - pa
        pb = win_probs[1]
        qb = 1.0 - pb

        pwin_from_6_6 = (pa * qb) / (pa * qb + qa * pb)
        state_6_6 = cls(6, 6, left_win_prob=pwin_from_6_6)

        pwin_from_7_6 = pwin_from_6_6 * (1.0 - qa * qb) / pa
        state_7_6 = cls(7, 6, left_win_prob=pwin_from_7_6)

        pwin_from_6_7 = pwin_from_6_6 * qb
        state_6_7 = cls(6, 7, left_win_prob=pwin_from_6_7)

        return [
            cls(7, 0),
            cls(7, 1),
            cls(7, 2),
            cls(7, 3),
            cls(7, 4),
            cls(7, 5),
            cls(0, 7),
            cls(1, 7),
            cls(2, 7),
            cls(3, 7),
            cls(4, 7),
            cls(5, 7),
            state_6_6,
            state_7_6,
            state_6_7,
        ]


class SetState(State):
    allfield_states = None
    probstates_from_probs = defaultdict(list)
    initprob_from_probs = defaultdict(lambda: None)

    """ обычный сет
    """

    def __init__(self, left_count, right_count, left_win_prob=None):
        super(SetState, self).__init__(left_count, right_count, left_win_prob)

    @staticmethod
    def min_for_win():
        return 6

    @staticmethod
    def prob_key(probs_dict):
        return probs_dict[co.POINT], probs_dict[co.GAME]

    def left_upstep_prob(self, probs_dict):
        hold_probs = probs_dict[co.GAME]
        return hold_probs[0] if self.opener_side.is_left() else 1.0 - hold_probs[1]

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if self.min_for_win() <= self.left_count and self.left_count >= (
            self.right_count + 2
        ):
            return co.LEFT
        if (
            self.min_for_win() + 1
        ) == self.left_count and self.right_count == self.min_for_win():
            return co.LEFT  # left win tie

        if self.min_for_win() <= self.right_count and self.right_count >= (
            self.left_count + 2
        ):
            return co.RIGHT
        if (
            self.min_for_win() + 1
        ) == self.right_count and self.left_count == self.min_for_win():
            return co.RIGHT  # right win tie

    @classmethod
    def start_states(cls, probs_dict):
        win_probs = probs_dict[co.POINT]
        pwin_from_6_6 = tiebreak_win_prob(win_probs)
        state_6_6 = cls(6, 6, left_win_prob=pwin_from_6_6)
        return [
            cls(6, 0),
            cls(6, 1),
            cls(6, 2),
            cls(6, 3),
            cls(6, 4),
            cls(7, 5),
            cls(0, 6),
            cls(1, 6),
            cls(2, 6),
            cls(3, 6),
            cls(4, 6),
            cls(5, 7),
            cls(7, 6),
            cls(6, 7),
            state_6_6,
        ]


class TielessSetState(State):
    allfield_states = None
    probstates_from_probs = defaultdict(list)
    initprob_from_probs = defaultdict(lambda: None)

    """ сет без тайбрейка
    """

    def __init__(self, left_count, right_count, left_win_prob=None):
        super(TielessSetState, self).__init__(left_count, right_count, left_win_prob)

    @staticmethod
    def min_for_win():
        return 6

    @staticmethod
    def prob_key(probs_dict):
        return probs_dict[co.POINT], probs_dict[co.GAME]

    def left_upstep_prob(self, probs_dict):
        hold_probs = probs_dict[co.GAME]
        return hold_probs[0] if self.opener_side.is_left() else 1.0 - hold_probs[1]

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if self.min_for_win() <= self.left_count and self.left_count >= (
            self.right_count + 2
        ):
            return co.LEFT
        if self.min_for_win() <= self.right_count and self.right_count >= (
            self.left_count + 2
        ):
            return co.RIGHT

    @classmethod
    def start_states(cls, probs_dict):
        hold_probs = probs_dict[co.GAME]
        pa, pb = hold_probs
        qa, qb = 1.0 - pa, 1.0 - pb
        pwin_from_6_6 = (pa * qb) / (pa * qb + qa * pb)
        state_6_6 = cls(6, 6, left_win_prob=pwin_from_6_6)
        return [
            cls(6, 0),
            cls(6, 1),
            cls(6, 2),
            cls(6, 3),
            cls(6, 4),
            cls(7, 5),
            cls(0, 6),
            cls(1, 6),
            cls(2, 6),
            cls(3, 6),
            cls(4, 6),
            cls(5, 7),
            state_6_6,
        ]

    def _normalize_counts(self):
        while (self.left_count + self.right_count) > 12:
            self.left_count -= 1
            self.right_count -= 1


class MatchBo3State(State):
    allfield_states = None
    probstates_from_probs = defaultdict(list)
    initprob_from_probs = defaultdict(lambda: None)

    """ стандартный матч
    """

    def __init__(self, left_count, right_count, left_win_prob=None):
        super(MatchBo3State, self).__init__(left_count, right_count, left_win_prob)

    @staticmethod
    def min_for_win():
        return 2

    @staticmethod
    def prob_key(probs_dict):
        return probs_dict[co.POINT], probs_dict[co.GAME]

    def left_upstep_prob(self, probs_dict):
        setw_prob = SetState.get_init_prob(probs_dict)
        return setw_prob if self.opener_side.is_left() else 1.0 - setw_prob

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if self.min_for_win() <= self.left_count and self.left_count > self.right_count:
            return co.LEFT
        if (
            self.min_for_win() <= self.right_count
            and self.right_count > self.left_count
        ):
            return co.RIGHT

    @classmethod
    def start_states(cls, probs_dict):
        paset = SetState.get_init_prob(probs_dict)
        state_1_1 = cls(1, 1, left_win_prob=paset)
        return [cls(2, 0), cls(2, 1), cls(0, 2), cls(1, 2), state_1_1]


def split_term_states(states):
    """return (terminated_states, noterminated_states)"""
    term_states = []
    states_len = len(states)
    for i in reversed(range(states_len)):
        if states[i].terminated():
            term_states.append(copy.deepcopy(states[i]))
            del states[i]
    return term_states, states


def make_prob_state(state, prob_states, probs_dict):
    """define left_win_prob in state. return True if success, False if failed"""
    # assert state.left_win_prob is None, "already probed state %s" % state
    state_next_win = state.step_forward_state(state, advance_side=co.LEFT)
    if state_next_win not in prob_states:
        return False
    next_win_idx = prob_states.index(state_next_win)
    state.up_next = prob_states[next_win_idx]

    state_next_loss = state.step_forward_state(state, advance_side=co.RIGHT)
    if state_next_loss not in prob_states:
        return False
    next_loss_idx = prob_states.index(state_next_loss)
    state.down_next = prob_states[next_loss_idx]

    up_prob = state.left_upstep_prob(probs_dict)
    state.left_win_prob = (
        up_prob * state.up_next.left_win_prob
        + (1.0 - up_prob) * state.down_next.left_win_prob
    )
    return True


def make_prob_states(states, prob_states, probs_dict):
    """transfer probed items from states into prob_states"""
    is_transfer = True
    while is_transfer:
        is_transfer = False
        states_len = len(states)
        for i in reversed(range(states_len)):
            if make_prob_state(states[i], prob_states, probs_dict):
                prob_states.append(copy.deepcopy(states[i]))
                del states[i]
                is_transfer = True


def content_classes():
    return (
        GameState,
        TiebreakState,
        SetState,
        TielessSetState,
        MatchBo3State,
    )


Prob = namedtuple("Prob", ["win_point", "hold"])


def initialize(probs: List[Prob]):
    def init_prob():
        prob_dct = make_probs_dict(
            win_probs=(prob.win_point, prob.win_point),
            hold_probs=(prob.hold, prob.hold),
        )
        GameState.build_all(probs_dict=prob_dct)
        TiebreakState.build_all(probs_dict=prob_dct)
        SetState.build_all(probs_dict=prob_dct)

    build_allfields()
    for prob in probs:
        init_prob()


def context_iter(det_score, best_of_five, decided_tiebreak, win_probs, hold_probs):
    def set_state_class(setnum):
        decided = setnum == 5 if best_of_five else 3
        if decided:
            return SetState if decided_tiebreak else TielessSetState
        else:
            return SetState

    def match_state_class():
        return MatchBo3State

    def pre_build(probs_dict):
        rev_probs_dict = reversed_probs_dict(probs_dict)
        GameState.build_all(probs_dict)
        GameState.build_all(rev_probs_dict)
        final_key_score = det_score.final_score()
        sets_count = len(final_key_score)
        match_state_cls = match_state_class()
        left_match_opener = None
        for setnum in range(1, sets_count + 1):
            set_state_cls = set_state_class(setnum)
            set_items = list(det_score.set_items(setnum))
            if not set_items:
                raise co.TennisError("context_iter empty set num {}".format(setnum))
            left_set_opener = set_items[0][1].left_opener
            if setnum == 1:
                left_match_opener = copy.copy(left_set_opener)
            if set_items[-1][1].tiebreak:
                assert (
                    left_set_opener == set_items[-1][1].left_opener
                ), "context_iter bad tie opener set num {}".format(setnum)
                if left_set_opener:
                    TiebreakState.build_all(probs_dict)
                else:
                    TiebreakState.build_all(rev_probs_dict)

            if left_set_opener:
                set_state_cls.build_all(probs_dict)
            else:
                set_state_cls.build_all(rev_probs_dict)
        if left_match_opener:
            match_state_cls.build_all(probs_dict)
        else:
            match_state_cls.build_all(rev_probs_dict)

    def get_contexts(probs_dict):
        rev_probs_dict = reversed_probs_dict(probs_dict)
        final_key_score = det_score.final_score()
        sets_count = len(final_key_score)
        match_state_cls = match_state_class()
        left_match_opener = None
        for setnum in range(1, sets_count + 1):
            set_state_cls = set_state_class(setnum)
            set_items = list(det_score.set_items(setnum))
            if not set_items:
                raise co.TennisError("context_iter empty set num {}".format(setnum))
            left_set_opener = set_items[0][1].left_opener
            if setnum == 1:
                left_match_opener = copy.copy(left_set_opener)
            if set_items[-1][1].tiebreak:
                assert (
                    left_set_opener == set_items[-1][1].left_opener
                ), "context_iter bad tie opener set num {}".format(setnum)

            if left_set_opener:
                set_state_cls.build_all(probs_dict)
            else:
                set_state_cls.build_all(rev_probs_dict)
        if left_match_opener:
            match_state_cls.build_all(probs_dict)
        else:
            match_state_cls.build_all(rev_probs_dict)

    # TODO
    probs_dict = make_probs_dict(win_probs, hold_probs)
    pre_build(probs_dict)


def build_allfields():
    for cls in content_classes():
        cls.build_allfield_states()


def tiebreak_win_prob(win_probs):
    """вернем вероятность выигрыша тайбрейка игроком A.
    win_probs[0] - вероятность выигрыша розыгрыша на своей подаче игроком A
    win_probs[1] - вероятность выигрыша розыгрыша на своей подаче игроком B.
    начинает игрок A. After Paul K. Newton and Joseph B. Keller
    """
    pa, pb = win_probs
    qa = 1.0 - pa
    qb = 1.0 - pb
    rich_prob = {}  # dict: score -> probability of rich this score
    rich_prob[(7, 0)] = (pa ** 3) * (qb ** 4)
    rich_prob[(0, 7)] = (qa ** 3) * (pb ** 4)
    rich_prob[(7, 1)] = 3.0 * (pa ** 3) * qa * (qb ** 4) + 4.0 * (pa ** 4) * pb * (
        qb ** 3
    )
    rich_prob[(1, 7)] = 3.0 * (qa ** 3) * pa * (pb ** 4) + 4.0 * (qa ** 4) * qb * (
        pb ** 3
    )
    rich_prob[(7, 2)] = (
        16.0 * (pa ** 4) * qa * pb * (qb ** 3)
        + 6.0 * (pa ** 5) * (pb ** 2) * (qb ** 2)
        + 6.0 * (pa ** 3) * (qa ** 2) * (qb ** 4)
    )
    rich_prob[(2, 7)] = (
        16.0 * (qa ** 4) * pa * qb * (pb ** 3)
        + 6.0 * (qa ** 5) * (qb ** 2) * (pb ** 2)
        + 6.0 * (qa ** 3) * (pa ** 2) * (pb ** 4)
    )
    rich_prob[(7, 3)] = (
        40.0 * (pa ** 3) * (qa ** 2) * pb * (qb ** 4)
        + 10.0 * (pa ** 2) * (qa ** 3) * (qb ** 5)
        + 4.0 * (pa ** 5) * (pb ** 3) * (qb ** 2)
        + 30.0 * (pa ** 4) * qa * (pb ** 2) * (qb ** 3)
    )
    rich_prob[(3, 7)] = (
        40.0 * (qa ** 3) * (pa ** 2) * qb * (pb ** 4)
        + 10.0 * (qa ** 2) * (pa ** 3) * (pb ** 5)
        + 4.0 * (qa ** 5) * (qb ** 3) * (pb ** 2)
        + 30.0 * (qa ** 4) * pa * (qb ** 2) * (pb ** 3)
    )
    rich_prob[(7, 4)] = (
        50.0 * (pa ** 4) * qa * (pb ** 3) * (qb ** 3)
        + 5.0 * (pa ** 5) * (pb ** 4) * (qb ** 2)
        + 50.0 * (pa ** 2) * (qa ** 3) * pb * (qb ** 5)
        + 5.0 * pa * (qa ** 4) * (qb ** 6)
        + 100.0 * (pa ** 3) * (qa ** 2) * (pb ** 2) * (qb ** 4)
    )
    rich_prob[(4, 7)] = (
        50.0 * (qa ** 4) * pa * (qb ** 3) * (pb ** 3)
        + 5.0 * (qa ** 5) * (qb ** 4) * (pb ** 2)
        + 50.0 * (qa ** 2) * (pa ** 3) * qb * (pb ** 5)
        + 5.0 * qa * (pa ** 4) * (pb ** 6)
        + 100.0 * (qa ** 3) * (pa ** 2) * (qb ** 2) * (pb ** 4)
    )
    rich_prob[(7, 5)] = (
        30.0 * (pa ** 2) * (qa ** 4) * pb * (qb ** 5)
        + pa * (qa ** 5) * (qb ** 6)
        + 200.0 * (pa ** 4) * (qa ** 2) * (pb ** 3) * (qb ** 3)
        + 75.0 * (pa ** 5) * qa * (pb ** 4) * (qb ** 2)
        + 150.0 * (pa ** 3) * (qa ** 3) * (pb ** 2) * (qb ** 4)
        + 6.0 * (pa ** 6) * (pb ** 5) * qb
    )
    rich_prob[(5, 7)] = (
        30.0 * (qa ** 2) * (pa ** 4) * qb * (pb ** 5)
        + qa * (pa ** 5) * (pb ** 6)
        + 200.0 * (qa ** 4) * (pa ** 2) * (qb ** 3) * (pb ** 3)
        + 75.0 * (qa ** 5) * pa * (qb ** 4) * (pb ** 2)
        + 150.0 * (qa ** 3) * (pa ** 3) * (qb ** 2) * (pb ** 4)
        + 6.0 * (qa ** 6) * (qb ** 5) * pb
    )
    rich_prob[(6, 6)] = 1.0 - sum(
        (rich_prob[(i, 7)] + rich_prob[(7, i)] for i in range(6))
    )
    return sum((rich_prob[(7, i)] for i in range(6))) + rich_prob[(6, 6)] * pa * qb / (
        1.0 - pa * pb - qa * qb
    )


def log_allfields():
    for cls in content_classes():
        log_states(cls.allfield_states, head=cls.__name__, with_classname=True)


def log_states(states, head="", with_classname=False):
    result = "-----%s (%d)------" % (head, len(states))
    for state in sorted(states):
        result += "\n\t%s" % state
        if with_classname:
            result += " " + state.__class__.__name__
    log.info(result)


def set_win_left_prob(
    set_state: SetState, game_state: GameState, g_st_diff: bool
) -> Optional[float]:
    if not set_state.up_next or not set_state.down_next:
        return None
    if g_st_diff:
        g_left_win_prob = 1.0 - game_state.left_win_prob
        g_left_lose_prob = game_state.left_win_prob
    else:
        g_left_win_prob = game_state.left_win_prob
        g_left_lose_prob = 1.0 - game_state.left_win_prob

    result = (
        g_left_win_prob * set_state.up_next.left_win_prob
        + g_left_lose_prob * set_state.down_next.left_win_prob
    )
    return result


def set_win_prob2(
    sex: str,
    surface,
    level,
    is_qual: bool,
    set_scr: sc.Scr,
    game_scr: sc.Scr,
    game_opener: Side,
    target_side: Side,
) -> Optional[float]:
    """usial game, not tie. returns prob for target_side"""
    key = skey(surface, level, is_qual)
    if (sex, key) in gen_to_prob:
        prob = gen_to_prob[(sex, key)]
        return set_win_prob(prob, set_scr, game_scr, game_opener, target_side)


def set_win_prob(
    prob: Prob, set_scr: sc.Scr, game_scr: sc.Scr, game_opener: Side, target_side: Side
) -> Optional[float]:
    """usial game, not tie. returns prob for target_side. Assume set_opener is LEFT"""
    prob_dct = make_probs_dict(
        win_probs=(prob.win_point, prob.win_point), hold_probs=(prob.hold, prob.hold)
    )
    key = SetState.prob_key(prob_dct)
    set_opener = game_opener if sum(set_scr) % 2 == 0 else game_opener.fliped()
    inp = f"s_scr {set_scr} g_opener {game_opener} g_scr {game_scr} set-op {set_opener}"
    if set_opener.is_right():
        log.error(f"set_win_prob set_opener must be left {inp}")
        # raise co.TennisDebugError(f'set_win_prob set_opener must be left {inp}')
        return
    s_st = co.find_first(
        SetState.probstates_from_probs[key],
        (
            lambda s: s.left_count == set_scr[0]
            and s.right_count == set_scr[1]
            and s.opener_side == game_opener
        ),
    )
    if s_st is None:
        log.error(f"set_win_prob notfound set_state {inp}")
        return
    key = GameState.prob_key(prob_dct)
    if game_opener.is_left():
        g_st_diff = False
    else:
        g_st_diff = True
    g_st = co.find_first(
        GameState.probstates_from_probs[key],
        (
            lambda s: s.left_count == game_scr[0]
            and s.right_count == game_scr[1]
            and s.opener_side == co.LEFT
        ),
    )
    if g_st is None:
        log.error(f"set_win_prob notfound game_state {inp}")
        return

    res_prob = set_win_left_prob(s_st, g_st, g_st_diff)
    if target_side.is_right() and res_prob is not None:
        res_prob = 1.0 - res_prob
    # log.debug(f'set_win_prob {res_prob} {inp} trg {target_side}')
    return res_prob


def tie_win_prob2(
    sex: str,
    surface,
    level,
    is_qual: bool,
    game_scr: sc.Scr,
    game_opener: Side,
    srv_side: Side,
    target_side: Side,
) -> Optional[float]:
    key = skey(surface, level, is_qual)
    if (sex, key) in gen_to_prob:
        prob = gen_to_prob[(sex, key)]
        return tie_win_prob(prob, game_scr, game_opener, srv_side, target_side)


def tie_win_prob(
    prob: Prob, game_scr: sc.Scr, game_opener: Side, srv_side: Side, target_side: Side
) -> Optional[float]:
    """returns prob for target_side"""
    if game_opener.is_right():
        srv_side = srv_side.fliped()
    prob_dct = make_probs_dict(
        win_probs=(prob.win_point, prob.win_point), hold_probs=(prob.hold, prob.hold)
    )
    key = TiebreakState.prob_key(prob_dct)
    g_st = co.find_first(
        TiebreakState.probstates_from_probs[key],
        (
            lambda s: s.left_count == game_scr[0]
            and s.right_count == game_scr[1]
            and s.opener_side == srv_side
        ),
    )
    if g_st is None:
        log.error(
            f"tie_win_prob notfound g_opener {game_opener} "
            f"g_scr {game_scr} srv_side {srv_side}"
        )
        return

    res_prob = g_st.left_win_prob
    if target_side != game_opener and res_prob is not None:
        res_prob = 1.0 - res_prob
    return res_prob


class TieWinProbTest(unittest.TestCase):
    def test_tie_win_prob_02(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(0, 2),
            game_opener=Side("RIGHT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 02 res_prob for R {res_pr}")
        self.assertTrue(0.1 < res_pr < 0.5)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(0, 2),
            game_opener=Side("RIGHT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 02 res_prob2 for L {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)

    def test_tie_win_prob_30(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 30 res_prob {res_pr}")
        self.assertTrue(0.6 < res_pr < 1)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 30 res_prob2 for R {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)

    def test_tie_win_prob_00(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(0, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 00 res_prob {res_pr}")
        self.assertTrue(abs(res_pr - 0.5) < 0.001)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(0, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 00 res_prob2 for R {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)


class SetWinLeftProbTest(unittest.TestCase):
    def test_set_win_prob_54(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(4, 5),
            game_opener=Side("RIGHT"),
            game_scr=(3, 0),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr is not None)
        self.assertTrue(0.7 < res_pr < 1, f"{res_pr} <= 0.7")

        res_pr2 = set_win_prob(
            prob=prob,
            set_scr=(4, 5),
            game_opener=Side("RIGHT"),
            game_scr=(0, 3),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        self.assertTrue(0.45 < res_pr2 < 0.6)

    def test_set_win_prob_53(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(5, 3),
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        self.assertTrue(0.7 < res_pr < 1)

    def test_set_win_prob_00(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(0, 0),
            game_opener=Side("LEFT"),
            game_scr=(0, 0),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"res_pr_00: {res_pr}")
        self.assertTrue(0.51 < res_pr < 1)


gen_to_prob = dict()  # (sex, key) -> Prob


def skey(surface, level, is_qual: bool) -> co.StructKey:
    if level in ("teamworld", "future") and is_qual:
        is_qual = False
    lev_val = "q-" + str(level) if is_qual else str(level)
    key = co.StructKey(level=lev_val, surface=str(surface))
    return key


def initialize_gen():
    prob_set = set()
    for sex in ("wta", "atp"):
        for surface in ("Hard", "Clay", "Carpet", "Grass"):
            for level in ("future", "chal", "main", "masters", "gs", "teamworld"):
                for is_qual in (False, True):
                    if level in ("teamworld", "future") and is_qual:
                        continue
                    lev_val = "q-" + level if is_qual else level
                    key = co.StructKey(level=lev_val, surface=surface)
                    p_win_pnt_sv = matchstat.generic_result(sex, "service_win", key)
                    p_hold_sv = matchstat.generic_result(sex, "service_hold", key)
                    if p_win_pnt_sv and p_hold_sv:
                        prob = Prob(
                            win_point=_prob_round(p_win_pnt_sv.value),
                            hold=_prob_round(p_hold_sv.value),
                        )
                        gen_to_prob[(sex, key)] = prob
                        prob_set.add(prob)
    initialize(probs=list(prob_set))


if __name__ == "__main__":
    import doctest

    log.initialize(
        co.logname(__file__, test=True), file_level="debug", console_level="info"
    )
    initialize_gen()
    initialize(probs=[Prob(win_point=0.63, hold=0.65)])
    # log_allfields()
    doctest.testmod()
    unittest.main()
