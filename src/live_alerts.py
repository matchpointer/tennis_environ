# -*- coding: utf-8 -*-
r"""
в модуле логика обработки Alert subclasses.
"""
import copy
from collections import namedtuple
import unittest
from typing import Optional


from side import Side
import common as co
import log
import ratings
import clf_common as cco
from score import Score, Scr, setidx_by_name
import clf_decided_00
import clf_secondset_00
from live import LiveMatch
import betfair_client
import impgames_stat
import inset_keep_recovery

import tieloss_affect_stat
import set2_after_set1loss_stat
import feature
from report_line import SizedValue


def remain_soft_main_skip_cond(match):
    return match.soft_level != "main"


def remain_ge_chal_skip_cond(match):
    return match.level not in ("chal", "main", "gs", "masters", "teamworld")


AlertMessage = namedtuple(
    "AlertMessage",
    "hashid text fst_name snd_name back_side case_name sex summary_href "
    "fst_betfair_name snd_betfair_name prob",
)


class Alert(object):
    """уведомление"""

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self.__eq__(other)

    def case_name(self):
        return ""

    def make_message(self, match, back_side: Side, prob=None, comment=""):
        self_text = "{} {} {}{} {}".format(
            match.sex,
            match.level,
            match.surface,
            " qual" if match.qualification else "",
            str(self),
        )
        back_text = "[" + ("1" if back_side.is_left() else "2") + "]"
        prob_text = ("P=" + str(round(prob, 3))) if prob is not None else ""
        comment_text = "" if not comment else " /" + comment + "/"
        return AlertMessage(
            hashid=match.hashid(add_text=self_text),
            text=(
                f"{match.tostring()} {back_text} {prob_text} {comment_text} {self_text}"
            ),
            fst_name=match.first_player.name if match.first_player else "",
            snd_name=match.second_player.name if match.second_player else "",
            back_side=back_side,
            case_name=self.case_name(),
            sex=match.sex,
            summary_href=match.summary_href(),
            fst_betfair_name=match.first_player.disp_name("betfair", ""),
            snd_betfair_name=match.second_player.disp_name("betfair", ""),
            prob=prob,
        )


class AlertSet(Alert):
    """уведомление о конкретном сете"""

    def __init__(self, setname: str, back_opener: Optional[bool]):
        super(AlertSet, self).__init__()
        self.setname = setname
        self.back_opener = back_opener

    def __str__(self):
        return "{} {} BO{}".format(
            self.setname[0:3],
            self.__class__.__name__[-14:],
            "-" if self.back_opener is None else int(self.back_opener),
        )

    def __repr__(self):
        return "{}(setname={}, back_opener={})".format(
            self.__class__.__name__, self.setname, self.back_opener
        )

    def __eq__(self, other):
        return (
            self.__class__.__name__ == other.__class__.__name__
            and self.setname == other.setname
            and self.back_opener == other.back_opener
        )

    def find_alertion(self, match: LiveMatch):
        """AlertMessage (if fired), False (gone), True (attention)"""
        result = self._condition_common(match)
        if result in (False, True, None):
            return result
        trg_srv_side = result
        result = self._condition_now(match, trg_srv_side)
        if isinstance(result, tuple):
            return self.make_message(match, *result)
        if result in (co.LEFT, co.RIGHT):
            return self.make_message(match, result)
        return result

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        raise NotImplementedError()

    def _condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if not match.first_player or not match.second_player:
            return False
        setnum = match.score.sets_count(full=False)
        if setnum == 0:
            return None
        insetscore = match.score[setnum - 1]
        trg_setidx = setidx_by_name(self.setname, best_of_five=match.best_of_five)
        if trg_setidx is None:
            return None
        if (setnum - 1) < trg_setidx:
            if setnum == trg_setidx and max(insetscore) >= 5:
                return True  # attention: next set may be our
            return None
        elif (setnum - 1) > trg_setidx:
            return False
        set_opener_side = match.curset_opener_side()
        return set_opener_side

    def back_side(self, trg_opener_side: Side) -> Side:
        if self.back_opener is None:
            return co.ANY
        return trg_opener_side if self.back_opener else trg_opener_side.fliped()


class AlertSetDecidedCloser(AlertSet):
    """уведомление о решающем сете, где можно ставить на закрывателя сета."""

    def __init__(self):
        super(AlertSetDecidedCloser, self).__init__(
            setname="decided", back_opener=False
        )

    def _condition_now(self, match, set_opener_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score:
            inset = match.score[-1]
            if inset in ((0, 0), (0, 1), (1, 0), (1, 1)):
                return self.back_side(set_opener_side)
        return True


class AlertSetDecidedCloserClf(AlertSetDecidedCloser):
    """
    увед-ние о реш. сете, и можно ставить на закр-ля сета.
    По анализу данных классификатором scikit-learn
    """

    def __init__(
        self,
        max_rating: int = cco.RANK_STD_BOTH_ABOVE,
        max_dif_rating: int = cco.RANK_STD_MAX_DIF,
    ):
        super().__init__()
        self.rtg_side_cond = ratings.SideCondition(max_rating, max_dif_rating)
        self.dec_begin_adv_dif = 0.1
        self.dec_begin_min_size = 10

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if clf_decided_00.skip_cond(match):
            return False
        if self.rtg_side_cond.condition_both_below(match) in (True, None):
            return False
        if match.left_service is None:
            return None
        return super()._condition_common(match)

    def case_name(self):
        """return '{setname}_00'"""
        return "{}_00".format(self.setname)

    @staticmethod
    def get_opener_closer(match, set_opener_side):
        """:return (opener_id, closer_id)"""
        opener_id, closer_id = match.first_player.ident, match.second_player.ident
        if set_opener_side == co.RIGHT:
            opener_id, closer_id = match.second_player.ident, match.first_player.ident
        return opener_id, closer_id

    @staticmethod
    def dec_begin_svalues(match, set_opener_side):
        """:return (opener_sv, closer_sv)"""
        opener_id, closer_id = AlertSetDecidedCloserClf.get_opener_closer(
            match, set_opener_side
        )
        opener_sv = inset_keep_recovery.read_player_sized_value(
            match.sex, inset_keep_recovery.BEGIN_ASPECT, "decided", opener_id
        )
        closer_sv = inset_keep_recovery.read_player_sized_value(
            match.sex, inset_keep_recovery.BEGIN_ASPECT, "decided", closer_id
        )
        return opener_sv, closer_sv

    def dec_begin_opener_adv(self, opener_sv, closer_sv):
        """:return is_opener_adv (True, False, None)"""
        if (
            opener_sv.size >= self.dec_begin_min_size
            and closer_sv.size >= self.dec_begin_min_size
        ):
            if opener_sv.value >= (closer_sv.value + self.dec_begin_adv_dif):
                return True
            if closer_sv.value >= (opener_sv.value + self.dec_begin_adv_dif):
                return False

    def tieloss_affect_opener_adv(self, match: LiveMatch, set_opener_side: Side):
        """:return is_opener_adv (True, False, None)"""
        set2_scr = match.score[1]
        if set2_scr not in ((6, 7), (7, 6)):
            return None
        is_opener_tie_win = (set_opener_side == co.LEFT) is (set2_scr[0] > set2_scr[1])
        opener_id, closer_id = self.get_opener_closer(match, set_opener_side)
        check_id = closer_id if is_opener_tie_win else opener_id
        is_aff = tieloss_affect_stat.is_player_affect_high(match.sex, check_id)
        if is_aff:
            return closer_id == check_id

    def decision(
        self,
        match,
        set_opener_side,
        clf_back_side,
        clf_prob,
        opener_has_tieaff_adv,
        opener_has_begin_adv,
        opener_beg_sv,
        closer_beg_sv,
    ):
        # now instead allow_back(match, backing_side, self, clf_prob, self.case_name())

        def contra(var1, var2):
            return var1 != var2 and var1 is not None and var2 is not None

        def beg_vals_text(back_side):
            back_sv, opp_sv = opener_beg_sv, closer_beg_sv
            if back_side != set_opener_side:
                back_sv, opp_sv = closer_beg_sv, opener_beg_sv
            return f"bckbg{back_sv} oppbg{opp_sv}"

        def fire_ret_by(opener_has_adv, prob, comment=""):
            back_side = set_opener_side if opener_has_adv else set_opener_side.fliped()
            if not self.rtg_side_cond.condition(match, back_side) or not allow_back(
                match, back_side, self, prob
            ):
                return False
            return back_side, prob, comment + beg_vals_text(back_side)

        if (clf_back_side, opener_has_tieaff_adv) == (None, None):
            return False
        opener_has_clf_adv = None
        if clf_back_side is not None:
            opener_has_clf_adv = clf_back_side == set_opener_side
        if opener_has_clf_adv is opener_has_tieaff_adv:
            return fire_ret_by(opener_has_clf_adv, clf_prob, "tieaff ")
        if contra(opener_has_clf_adv, opener_has_tieaff_adv):
            log.info(
                f"{self.__class__.__name__} reject {match.name} "
                f"tieloss aff VS backing_side {clf_back_side}"
            )
            return False  # contradict
        if opener_has_clf_adv is None and (
            (
                opener_has_tieaff_adv is opener_has_begin_adv
                and opener_has_tieaff_adv is not None
            )
            or opener_has_tieaff_adv is False
        ):
            return fire_ret_by(opener_has_tieaff_adv, 0.51, "tieaff ")

        if opener_has_clf_adv is not None and opener_has_tieaff_adv is None:
            if contra(opener_has_clf_adv, opener_has_begin_adv) and clf_prob < 0.7:
                log.info(
                    f"{self.__class__.__name__} reject {match.name} "
                    f"clf {clf_back_side} VS begin"
                )
                return False  # contradict
            return fire_ret_by(opener_has_clf_adv, clf_prob)
        return False

    def _condition_now(self, match, set_opener_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score:
            inset = match.score[-1]
            if inset == (0, 0):
                opener_beg_sv, closer_beg_sv = self.dec_begin_svalues(
                    match, set_opener_side
                )
                opener_has_begin_adv = self.dec_begin_opener_adv(
                    opener_beg_sv, closer_beg_sv
                )
                opener_has_tieaff_adv = self.tieloss_affect_opener_adv(
                    match, set_opener_side
                )
                clf_back_side, prob = clf_decided_00.match_has_min_proba(
                    match, set_opener_side
                )
                return self.decision(
                    match,
                    set_opener_side,
                    clf_back_side,
                    prob,
                    opener_has_tieaff_adv,
                    opener_has_begin_adv,
                    opener_beg_sv,
                    closer_beg_sv,
                )
        return True


class AlertSetAfterTie(AlertSet):
    def __init__(self, setname, max_rating, max_dif_rating):
        super(AlertSetAfterTie, self).__init__(setname=setname, back_opener=None)
        self.rtg_side_cond = ratings.SideCondition(max_rating, max_dif_rating)
        set_idx = setidx_by_name(setname)
        if set_idx is None:
            log.error(f"can not get setnum for setidx: {setname} assume 0")
            set_idx = 1
        self.trg_setnum = set_idx + 1

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        raise NotImplementedError()

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if self.rtg_side_cond.condition_both_below(match) in (True, None):
            log.info(
                "{} reject {} max both rtg".format(self.__class__.__name__, match.name)
            )
            return False
        setnum_full = match.score.sets_count(full=True)
        if setnum_full < (self.trg_setnum - 2):
            return None  # too early now
        elif setnum_full == (self.trg_setnum - 2):
            if len(match.score) == (setnum_full + 1) and match.score[-1] == (6, 6):
                return True  # after this tie will be our event
            return None
        elif setnum_full == (self.trg_setnum - 1):
            if match.score[setnum_full - 1] not in ((7, 6), (6, 7)):
                return False
            if len(match.score) == setnum_full or match.score[-1] == (0, 0):
                srv_side = match.curset_opener_side()
                return srv_side if srv_side is not None else co.LEFT
            return False
        elif setnum_full > (self.trg_setnum - 1):
            return False


class AlertSetAfterTieAffect(AlertSetAfterTie):
    def __init__(self, setname, max_rating=450, max_dif_rating=300):
        super(AlertSetAfterTieAffect, self).__init__(
            setname, max_rating, max_dif_rating
        )

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if match.soft_level not in ("main", "chal"):  # for chal
            return False
        if match.sex == "atp" and match.best_of_five:
            return False
        if not match.first_player or not match.second_player:
            return False
        is_fst_aff = tieloss_affect_stat.is_player_affect_high(
            match.sex, match.first_player.ident
        )
        is_snd_aff = tieloss_affect_stat.is_player_affect_high(
            match.sex, match.second_player.ident
        )
        if not is_fst_aff and not is_snd_aff:
            return False
        return super(AlertSetAfterTieAffect, self)._condition_common(match)

    def _condition_now(self, match, set_opener_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score:
            inset = match.score[-1]
            if inset == (0, 0):
                prev_trg_scr = match.score[self.trg_setnum - 2]
                if prev_trg_scr not in ((7, 6), (6, 7)):
                    log.error(
                        "{} {} NOT TIE SCORE: {}".format(
                            self.__class__.__name__, match.name, prev_trg_scr
                        )
                    )
                    return False
                losstie_side = (
                    co.LEFT if prev_trg_scr[0] < prev_trg_scr[1] else co.RIGHT
                )
                if not self.rtg_side_cond.condition(match, losstie_side.fliped()):
                    return False
                if losstie_side == co.LEFT:
                    is_aff = tieloss_affect_stat.is_player_affect_high(
                        match.sex, match.first_player.ident
                    )
                else:
                    is_aff = tieloss_affect_stat.is_player_affect_high(
                        match.sex, match.second_player.ident
                    )

                if is_aff and self.extra_condition(
                    match, set_opener_side, losstie_side
                ):
                    backing_side, prob = losstie_side.fliped(), 0.61
                    if allow_back(match, backing_side, self, prob):
                        return backing_side, prob
                return False
        return True

    def extra_condition(self, match, set_opener_side, losstie_side):
        if self.setname == "decided":
            if losstie_side != set_opener_side:
                return False
        elif self.setname == "second":
            tiewin_side = losstie_side.fliped()
            tiewin_pid = (
                match.first_player.ident
                if tiewin_side == co.LEFT
                else match.second_player.ident
            )
            sv = set2_after_set1loss_stat.read_player_sized_value(
                match.sex, "set2win_after_set1win", tiewin_pid
            )
            if (
                sv
                and sv.size > 20
                and (sv.value + 0.03)
                < set2_after_set1loss_stat.read_generic_value(
                    match.sex, "set2win_after_set1win"
                )
            ):
                log.info(
                    "{} reject {} by set2win_after_set1win {}".format(
                        self.__class__.__name__, match.name, sv.value
                    )
                )
                return False
        fst_chance = match.first_player_bet_chance()
        if fst_chance is not None:
            wintie_side_chance = (
                fst_chance if losstie_side == co.RIGHT else (1.0 - fst_chance)
            )
            if wintie_side_chance < 0.49:
                return False  # we avoid backing underdog here
        return True


class AlertSet2WinClf(AlertSet):
    def __init__(
        self,
        max_rating: int = cco.RANK_STD_BOTH_ABOVE,
        max_dif_rating: int = cco.RANK_STD_MAX_DIF,
    ):
        super().__init__(setname="second", back_opener=None)
        self.rtg_side_cond = ratings.SideCondition(max_rating, max_dif_rating)
        self.trg_setnum = 2

    def case_name(self):
        return "secondset_00"

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if self.rtg_side_cond.condition_both_below(match) in (True, None):
            return False
        result = super()._condition_common(match)
        if result in (co.LEFT, co.RIGHT):
            if clf_secondset_00.skip_cond(match):
                return False
        return result

    def _condition_now(self, match, set_opener_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score and len(match.score) == 2:
            inset = match.score[1]
            if sum(inset) > 1:
                return False  # we are late
            if inset == (0, 0):
                backing_side, prob = clf_secondset_00.match_has_min_proba(
                    match, set_opener_side
                )
                if (
                    backing_side is not None
                    and self.rtg_side_cond.condition(match, backing_side)
                    and allow_back(match, backing_side, self, prob)
                ):
                    return backing_side, prob
                return False
        return True


class AlertSc(Alert):
    """уведомление о конкретном счете в конкретном сете"""

    def __init__(
        self, setname: str, insetscore: Scr, srv_side: Side, back_srv: Optional[bool]
    ):
        super(AlertSc, self).__init__()
        self.setname = setname
        self.insetscore = insetscore
        self.srv_side = srv_side
        self.back_srv = back_srv

    def case_name(self):
        """return '{setname}_XYrcv' or '{setname}_XYsrv'"""
        scr_part = "{}{}".format(min(self.insetscore), max(self.insetscore))
        if self.back_srv is None:
            srv_part = "any"
        elif self.back_srv:
            srv_part = "srv"
        else:
            srv_part = "rcv"
        return "{}_{}{}".format(self.setname, scr_part, srv_part)

    def __str__(self):
        return "{} {} {} S{} BS{}".format(
            self.setname,
            self.__class__.__name__[-14:],
            self.insetscore,
            self.srv_side.abbr_text(),
            "-" if self.back_srv is None else int(self.back_srv),
        )

    def __repr__(self):
        return "{}(setname={}, insetscore={}, serve_side={}, back_srv={})".format(
            self.__class__.__name__,
            self.setname,
            self.insetscore,
            self.srv_side,
            self.back_srv,
        )

    def __eq__(self, other):
        return (
            self.__class__.__name__ == other.__class__.__name__
            and self.setname == other.setname
            and self.insetscore == other.insetscore
            and self.srv_side == other.srv_side
            and self.back_srv == other.back_srv
        )

    def find_alertion(self, match: LiveMatch):
        """AlertMessage (if fired), False (gone), True (attention)"""
        result = self._condition_common(match)
        if result in (False, True, None):
            return result
        trg_srv_side = result
        result = self._condition_next(match, trg_srv_side)
        if isinstance(result, tuple):
            return self.make_message(match, *result)
        if result in (co.LEFT, co.RIGHT):
            return self.make_message(match, result)
        result = self._condition_now(match, trg_srv_side)
        if isinstance(result, tuple):
            return self.make_message(match, *result)
        if result in (co.LEFT, co.RIGHT):
            return self.make_message(match, result)
        return result

    def _condition_now(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze current match state.
        Return (back side, prob) (if fired), False (if gone), True (if attention)"""
        raise NotImplementedError()

    def _condition_next(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze for next game from current match state.
        Return (back side, prob) (if fired), False (if gone), True (if attention)"""
        raise NotImplementedError()

    def _condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if not match.first_player or not match.second_player:
            return False
        setnum = match.score.sets_count(full=False)
        if setnum == 0:
            return None
        insetscore = match.score[setnum - 1]
        trg_setidx = setidx_by_name(self.setname, best_of_five=match.best_of_five)
        if trg_setidx is None:
            return None
        if (setnum - 1) < trg_setidx:
            if setnum == trg_setidx and max(insetscore) >= 5:
                return True  # attention: next set may be our
            return None
        elif (setnum - 1) > trg_setidx or (
            (setnum - 1) == trg_setidx and sum(insetscore) > sum(self.insetscore)
        ):
            return False
        lack_games_num = sum(self.insetscore) - sum(insetscore)
        trg_srv_left = match.left_service is ((lack_games_num % 2) == 0)
        if not self.srv_side.is_any() and self.srv_side.is_left() is not trg_srv_left:
            return False  # wrong server
        if lack_games_num >= 2:
            return None  # delay decision
        return co.side(trg_srv_left)

    def back_side(self, trg_srv_side: Side) -> Side:
        if self.back_srv is None:
            return co.ANY
        return trg_srv_side if self.back_srv else trg_srv_side.fliped()


class AlertScEqual(AlertSc):
    def __init__(self, setname: str, insetscore: Scr, back_srv: Optional[bool]):
        super(AlertScEqual, self).__init__(
            setname=setname, insetscore=insetscore, srv_side=co.ANY, back_srv=back_srv
        )
        assert insetscore[0] == insetscore[1], "invalid equal score {} ".format(
            insetscore
        )

    def _condition_now(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (
            None,
            ("0", "0"),
            ("15", "0"),
            ("0", "15"),
            ("15", "15"),
        ):
            return self.back_side(trg_srv_side)

    def _condition_next(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        insetscore = match.score[-1]
        # analyze trg_insetscore minus 1 game = insetscore
        cur_srv_side = trg_srv_side.fliped()  # inverse serve in prev game
        ingame_adv_side = ingame_advance_side(match.ingame)
        backing_side = self.back_side(trg_srv_side)
        if (insetscore[0] + 1, insetscore[1]) == self.insetscore and (
            (cur_srv_side.is_left() and match.ingame not in (("0", "40"), ("15", "40")))
            or (cur_srv_side.is_right() and ingame_adv_side == co.LEFT)
        ):
            return backing_side
        if (insetscore[0], insetscore[1] + 1) == self.insetscore and (
            (
                cur_srv_side.is_right()
                and match.ingame not in (("40", "0"), ("40", "15"))
            )
            or (cur_srv_side.is_left() and ingame_adv_side == co.RIGHT)
        ):
            return backing_side


class AlertScEqualSetend(AlertScEqual):
    def __init__(
        self,
        setname: str,
        insetscore: Scr,
        back_srv: bool,
        min_proba: float,
        min_size: int,
        max_lay_ratio: float,
    ):
        super().__init__(setname=setname, insetscore=insetscore, back_srv=back_srv)
        assert insetscore[0] == insetscore[1], "invalid equal score {} ".format(
            insetscore
        )
        self.min_proba = min_proba
        self.min_size = min_size
        self.max_lay_ratio = max_lay_ratio

    def _setend_adv_proba(self, match, trg_srv_side):
        fst_srv = trg_srv_side == co.LEFT
        res = impgames_stat.get_players_pair_result(
            sex=match.sex,
            timename="all",
            fst_plr_id=match.first_player.ident,
            snd_plr_id=match.second_player.ident,
            fst_srv=fst_srv,
            set_names=(self.setname, self.setname),
            aspect_name="setend",
            min_size=self.min_size,
        )
        if res is None:
            return None
        if self.back_srv:
            proba = res.fst_proba if fst_srv else res.snd_proba
            lay_ratio = res.snd_wl.ratio
        else:
            proba = res.snd_proba if fst_srv else res.fst_proba
            lay_ratio = res.fst_wl.ratio
        if (proba < self.min_proba) or (lay_ratio > self.max_lay_ratio):
            return None
        return proba

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if match.sex == "atp" and self.setname == "open" and match.best_of_five:
            return False
        trg_srv_side = super()._condition_common(match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            if self._setend_adv_proba(match, trg_srv_side) is None:
                return False
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (None, ("0", "0")):
            backing_side = self.back_side(trg_srv_side)
            proba = self._setend_adv_proba(match, trg_srv_side)
            if proba is not None and allow_back(match, backing_side, self, proba):
                return backing_side, proba
            return False

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        backing_side = super()._condition_next(match, trg_srv_side)
        if backing_side in (co.LEFT, co.RIGHT):
            proba = self._setend_adv_proba(match, trg_srv_side)
            if proba is not None and allow_back(match, backing_side, self, proba):
                return backing_side, proba
            return False
        return backing_side


class AlertScEqualTieValue(AlertScEqual):
    feature_names = {
        "open": "s1_tie_value",
        "decided": "sd_tie_value",
    }

    def __init__(
        self,
        setname,
        max_rating,
        max_dif_rating,
        min_back_value,
        max_lay_value,
        min_size,
    ):
        super().__init__(setname=setname, insetscore=(6, 6), back_srv=None)
        self.rtg_side_cond = ratings.SideCondition(max_rating, max_dif_rating)
        self.min_back_value = min_back_value
        self.max_lay_value = max_lay_value
        self.min_size = min_size
        assert (
            setname in AlertScEqualTieValue.feature_names.keys()
        ), "unexpected setname {}".format(setname)

    def feature_name(self):
        return self.feature_names[self.setname]

    @staticmethod
    def comment(match, feat_name, back_side):
        f1, f2 = feature.player_features(match.features, feat_name)
        if back_side == co.LEFT:
            back_val, opp_val = f1.value, f2.value
        elif back_side == co.RIGHT:
            back_val, opp_val = f2.value, f1.value
        else:
            raise co.TennisError("unexpect back_side '{}'".format(back_side))
        return "bkv{:.2f} opv{:.2f}".format(back_val, opp_val)

    def get_back_side(self, match, feat_name):
        back_side = None
        try:
            f1, f2 = feature.player_features(match.features, feat_name)
            if (
                f1.value is None
                or f2.value is None
                or f1.value.size < self.min_size
                or f2.value.size < self.min_size
            ):
                return
            if (
                f1.value.value >= self.min_back_value
                and f2.value.value <= self.max_lay_value
                and self.rtg_side_cond.condition(match, co.LEFT)
                and (
                    self.setname == "decided"
                    or tieloss_affect_stat.is_player_affect_above_avg(
                        match.sex, match.first_player.ident
                    )
                    is False
                )
            ):
                back_side = co.LEFT
            elif (
                f2.value.value >= self.min_back_value
                and f1.value.value <= self.max_lay_value
                and self.rtg_side_cond.condition(match, co.RIGHT)
                and (
                    self.setname == "decided"
                    or tieloss_affect_stat.is_player_affect_above_avg(
                        match.sex, match.second_player.ident
                    )
                    is False
                )
            ):
                back_side = co.RIGHT
        except cco.FeatureError:
            pass
        return back_side

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        feat_name = self.feature_name()
        cache_val = match.get_cache_value(feat_name)
        if cache_val is False:
            return False
        if self.setname == "decided" and not match.decided_tiebreak:
            match.set_cache_value(feat_name, value=False)
            return False
        trg_srv_side = super()._condition_common(match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            if cache_val is None:
                backing_side = self.get_back_side(match, feat_name)
                if backing_side is None:
                    match.set_cache_value(feat_name, value=False)
                    return False
                else:
                    match.set_cache_value(feat_name, value=backing_side)
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (
            None,
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
        ):
            feat_name = self.feature_name()
            cache_val = match.get_cache_value(feat_name)
            if cache_val is False:
                return False
            elif cache_val is None:
                backing_side = self.get_back_side(match, feat_name)
            else:
                backing_side = cache_val
            proba = 0.61
            if allow_back(match, backing_side, self, proba):
                return backing_side, proba, self.comment(match, feat_name, backing_side)
            return False

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        return None  # disable


class AlertScEqualTieRatio(AlertScEqual):
    def __init__(
        self,
        setname: str,
        max_rating: int,
        max_dif_rating: int,
        min_proba: float,
        max_lay_ratio,
        min_size: int,
    ):
        super().__init__(setname=setname, insetscore=(6, 6), back_srv=None)
        self.rtg_side_cond = ratings.SideCondition(max_rating, max_dif_rating)

        if setname == "open":
            self.plr_feature_name = "s1_tie_ratio"
        elif setname == "decided":
            self.plr_feature_name = "sd_tie_ratio"
        else:
            raise co.TennisError("unexpected setname {}".format(setname))

        self.min_proba = min_proba
        self.max_lay_ratio = max_lay_ratio
        self.min_size = min_size

    def get_feature_name(self, match):
        return self.plr_feature_name

    def sized_values(self, match):
        try:
            featname = self.get_feature_name(match)
            if featname:
                f1, f2 = feature.player_features(match.features, featname)
                return f1.value, f2.value
        except cco.FeatureError:
            pass
        return SizedValue(), SizedValue()

    def comment(self, match, back_side):
        sv1, sv2 = self.sized_values(match)
        if back_side == co.LEFT:
            back_sval, opp_sval = sv1, sv2
        elif back_side == co.RIGHT:
            back_sval, opp_sval = sv2, sv1
        else:
            raise co.TennisError("unexpect back_side '{}'".format(back_side))
        return f"BK{back_sval} OP{opp_sval}"

    def get_backside_prob(self, match):
        sv1, sv2 = self.sized_values(match)
        if sv1.size < self.min_size or sv2.size < self.min_size:
            return None, None
        proba1, proba2 = co.twoside_values(sv1, sv2)
        if proba1 < self.min_proba and proba2 < self.min_proba:
            return None, None
        if (
            proba1 >= self.min_proba
            and sv2.value <= self.max_lay_ratio
            and self.rtg_side_cond.condition(match, co.LEFT)
            and (
                self.setname == "decided"
                or tieloss_affect_stat.is_player_affect_above_avg(
                    match.sex, match.first_player.ident
                )
                is False
            )
        ):
            return co.LEFT, proba1
        if (
            proba2 >= self.min_proba
            and sv1.value <= self.max_lay_ratio
            and self.rtg_side_cond.condition(match, co.RIGHT)
            and (
                self.setname == "decided"
                or tieloss_affect_stat.is_player_affect_above_avg(
                    match.sex, match.second_player.ident
                )
                is False
            )
        ):
            return co.RIGHT, proba2

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        featname = self.get_feature_name(match)
        cache_val = match.get_cache_value(featname)
        if cache_val is False:
            return False
        if self.setname == "decided" and not match.decided_tiebreak:
            match.set_cache_value(featname, value=False)
            return False
        trg_srv_side = super()._condition_common(match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            if cache_val is None:
                backing_side, prob = None, None
                backing_side_prob = self.get_backside_prob(match)
                if backing_side_prob is not None:
                    backing_side, prob = backing_side_prob
                if backing_side is None:
                    match.set_cache_value(featname, value=False)
                    return False
                else:
                    match.set_cache_value(featname, value=(backing_side, prob))
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (
            None,
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
        ):
            featname = self.get_feature_name(match)
            if not featname:
                return False
            cache_val = match.get_cache_value(featname)
            if cache_val is False:
                return False
            elif cache_val is None:
                backing_side, prob = self.get_backside_prob(match)
            else:
                backing_side, prob = cache_val
            if backing_side is None:
                return False
            assert backing_side in (co.LEFT, co.RIGHT), (
                f"alert_tie_ratio bad back {backing_side} in {match.name} "
                f"featname: {featname} cache:{match.get_cache_value(featname)}"
            )
            if allow_back(match, backing_side, self, self.min_proba):
                return backing_side, self.min_proba, self.comment(match, backing_side)
            return False

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        return None  # disable


class AlertScPursue(AlertSc):
    """Зеркальная логика реализуется зеркально сконструируемым объектом.
    Т.е. например left_srv_side(3, 4) и (4, 3)right_srv_side - в разных об-тах
    """

    def __init__(
        self,
        setname,
        insetscore,
        back_srv=False,
        srv_leader=False,
        skip_cond=remain_soft_main_skip_cond,
    ):
        if insetscore[0] < insetscore[1]:
            srv_side = co.RIGHT if srv_leader else co.LEFT
        else:
            srv_side = co.LEFT if srv_leader else co.RIGHT
        super().__init__(
            setname=setname, insetscore=insetscore, srv_side=srv_side, back_srv=back_srv
        )
        self.skip_cond = skip_cond

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        trg_srv_side = AlertSc._condition_common(self, match)
        if trg_srv_side in (co.LEFT, co.RIGHT) and trg_srv_side != self.srv_side:
            return False
        if self.skip_cond(match):
            return False
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (
            None,
            ("0", "0"),
            ("15", "0"),
            ("0", "15"),
            ("15", "15"),
        ):
            return self.back_side(trg_srv_side)

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""

        def get_cur_insetscore(break_route):
            if break_route:
                if trg_srv_side.is_left():
                    return self.insetscore[0] - 1, self.insetscore[1]
                else:
                    return self.insetscore[0], self.insetscore[1] - 1
            else:
                if trg_srv_side.is_left():
                    return self.insetscore[0], self.insetscore[1] - 1
                else:
                    return self.insetscore[0] - 1, self.insetscore[1]

        cur_srv_side = trg_srv_side.fliped()  # inverse serve in prev game
        ingame_adv_side = ingame_advance_side(match.ingame)
        ingame_normal_sides = ingame_normalgo_sides(match.ingame)
        for break_route in (False, True):
            cur_insetscore = get_cur_insetscore(break_route)
            if cur_insetscore == match.score[-1]:
                if not break_route and cur_srv_side in ingame_normal_sides:
                    return self.back_side(trg_srv_side)
                # break route. лидер cur_srv_side. догоняющий ingame_adv_side
                elif (
                    break_route
                    and ingame_adv_side is not None
                    and ingame_adv_side != cur_srv_side
                ):
                    return self.back_side(trg_srv_side)


class AlertScPursueClf(AlertScPursue):
    """Зеркальная логика реализуется зеркально сконструируемым объектом.
    Т.е. например right_srv_side(0, 3) и (3, 0)left_srv_side - в разных об-тах.
    Our case: back player at 0-3 and receive.
    """

    rtg_side_cond = ratings.SideCondition(
        max_rating=cco.RANK_STD_BOTH_ABOVE, max_dif_rating=cco.RANK_STD_MAX_DIF
    )

    def __init__(
        self,
        setname,
        clf_fun,
        insetscore,
        back_srv,
        srv_leader,
        skip_cond=remain_soft_main_skip_cond,
    ):
        super(AlertScPursueClf, self).__init__(
            setname=setname,
            insetscore=insetscore,
            back_srv=back_srv,
            srv_leader=srv_leader,
            skip_cond=skip_cond,
        )
        self.clf_fun = clf_fun

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if self.rtg_side_cond.condition_both_below(match) in (True, None):
            return False
        trg_srv_side = AlertScPursue._condition_common(self, match)
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if (
            match.score
            and match.score[-1] == self.insetscore
            and match.ingame
            in (None, ("0", "0"), ("15", "0"), ("0", "15"), ("15", "15"))
        ):
            pursue_side = (
                co.LEFT if match.score[-1][0] < match.score[-1][1] else co.RIGHT
            )
            if not self.rtg_side_cond.condition(match, pursue_side):
                log.info(
                    "{} reject {} {} pursue rtg".format(
                        self.__class__.__name__, match.name, self.insetscore
                    )
                )
                return False
            set_opener_side = match.curset_opener_side()
            backing_side, prob = self.clf_fun(match, set_opener_side)
            if (backing_side is not None) and allow_back(
                match, backing_side, self, prob
            ):
                if (
                    self.insetscore in ((0, 1), (1, 0))
                    and self.setname == "decided"
                    and allow_back_inset_keep_recovery(
                        match,
                        backing_side,
                        "recovery",
                        "keep",
                        "decided",
                        lambda b_reco, o_keep: (b_reco - 0.39) > (o_keep - 0.60),
                    )
                    is False
                ):
                    log.info(
                        "{} reject by inset_keep_recovery:\n{}".format(
                            self.__class__.__name__, match.tostring()
                        )
                    )
                    return False
                return backing_side, prob
            else:
                return False
        return True

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention).
        Here trg_srv_side is our opponent.
        """
        return None  # disable early alert as we will use betfair bot


class AlertScPursueSet1_Clf(AlertScPursue):
    """Зеркальная логика реализуется зеркально сконструируемым объектом.
    Т.е. например right_srv_side(0, 3) и (3, 0)left_srv_side - в разных об-тах.
    Our case: back player at 0-3 and receive.
    """

    def __init__(self, insetscore, clf_fun, skip_cond):
        super(AlertScPursueSet1_Clf, self).__init__(
            setname="open", insetscore=insetscore, back_srv=False, srv_leader=True
        )
        self.clf_fun = clf_fun
        self.skip_cond = skip_cond
        self.rtg_side_cond = ratings.SideCondition(max_rating=350, max_dif_rating=220)

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if self.skip_cond(match):
            return False
        if self.rtg_side_cond.condition_both_below(match) in (True, None):
            return False
        trg_srv_side = AlertScPursue._condition_common(self, match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            backing_side = self.back_side(trg_srv_side)
            if not self.rtg_side_cond.condition(match, backing_side):
                return False
        return trg_srv_side

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if (
            match.score
            and match.score[-1] == self.insetscore
            and match.ingame in (None, ("0", "0"))
        ):
            set_open_side = match.curset_opener_side()
            if set_open_side is None:
                return False
            backing_side, prob = self.clf_fun(match, set_open_side=set_open_side)
            if (backing_side is not None) and allow_back(
                match, backing_side, self, prob
            ):
                return backing_side, prob
            else:
                return False
        return True

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention).
        Here trg_srv_side is our opponent.
        """
        return None  # disable early alert as we will use betfair bot


class AlertSc56toTie(AlertScPursue):
    """Зеркальная логика реализуется зеркально сконструируемым объектом.
    Т.е. например right_srv_side(6, 5) и (5, 6)left_srv_side - в разных об-тах.
    Our cases: back player who receive at 6-5 then open on tie (back_setopener=true).
               back player who serve at 5-6 then close on tie (back_setopener=false).
    """

    def __init__(
        self,
        setname: str,
        insetscore,
        back_setopener,
        min_proba,
        min_size,
        max_rating,
        max_dif_rating,
    ):
        assert abs(insetscore[0] - insetscore[1]) == 1, f"bad insetscore {insetscore}"
        super().__init__(
            setname=setname,
            insetscore=insetscore,
            back_srv=not back_setopener,
            srv_leader=False,
            skip_cond=remain_ge_chal_skip_cond,
        )
        self.min_proba = min_proba
        self.min_size = min_size
        if setname == "open":
            self.plr_feature_name = "s1_tie_ratio"
        elif setname == "decided":
            self.plr_feature_name = "sd_tie_ratio"
        elif setname == "second":
            self.plr_feature_name = "s2_under_tie_ratio"
        else:
            raise co.TennisError("unexpected setname {}".format(setname))

        self.rtg_side_cond = ratings.SideCondition(
            max_rating, max_dif_rating, rtg_name="elo"
        )

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if match.sex == "atp" and match.best_of_five:
            return False
        trg_srv_side = super()._condition_common(match)
        if trg_srv_side not in (co.LEFT, co.RIGHT):
            return trg_srv_side
        cache_val = match.get_cache_value(self.plr_feature_name)
        if cache_val is False:
            return False
        elif cache_val is None:
            trg_back_side = self.back_side(trg_srv_side)
            if not self.rtg_side_cond.condition(match, trg_back_side):
                match.set_cache_value(self.plr_feature_name, False)
                return False
            proba = self._get_back_side_advantage_proba(match, trg_back_side)
            if proba is not None:
                match.set_cache_value(self.plr_feature_name, trg_back_side)
            else:
                match.set_cache_value(self.plr_feature_name, False)
                log.info(
                    f"{self.__class__.__name__} reject early "
                    f"trg_srv_side {trg_srv_side} sn {self.setname} "
                    f"iss {self.insetscore} "
                    f"back: {trg_back_side}\n\t{match.tostring()}"
                )
                return False
        return trg_srv_side

    def _get_back_side_advantage_proba(self, match, trg_back_side):
        try:
            f1, f2 = feature.player_features(match.features, self.plr_feature_name)
            if (
                not f1
                or not f2
                or f1.value.size < self.min_size
                or f2.value.size < self.min_size
            ):
                return None
            proba1, proba2 = co.twoside_values(f1.value, f2.value)
            proba = proba1 if trg_back_side == co.LEFT else proba2
            if proba > self.min_proba:
                return proba
        except cco.FeatureError:
            return None

    def _condition_now(self, match, trg_srv_side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if (
            match.score
            and match.score[-1] == self.insetscore
            and match.ingame
            in (
                None,
                ("0", "0"),
            )
        ):
            backing_side = self.back_side(trg_srv_side)
            proba = self._get_back_side_advantage_proba(match, backing_side)
            if proba is None:
                log.info(
                    "{} reject now trg_srv_side {}\n\t{}".format(
                        self.__class__.__name__, trg_srv_side, match.tostring()
                    )
                )
                return False

            if backing_side is not None and allow_back(
                match, backing_side, self, proba
            ):
                return backing_side, proba
            else:
                return False
        return True

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention).
        Here trg_srv_side is our opponent.
        """
        return None  # disable early alert as we have some change-over time


class AlertScPursueBreakDownEarly(AlertScPursue):
    """Зеркальная логика реализуется зеркально сконструируемым объектом.
    Т.е. например right_srv_side(0, 1) и (1, 0)left_srv_side - в разных об-тах
    """

    def __init__(self, setname, insetscore):
        super(AlertScPursueBreakDownEarly, self).__init__(
            setname=setname, insetscore=insetscore, back_srv=False, srv_leader=True
        )

    def _condition_common(self, match):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if match.level in ("chal", "team"):
            return False
        trg_srv_side = super(AlertScPursueBreakDownEarly, self)._condition_common(match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            backing_side = self.back_side(trg_srv_side)
            if match.h2h_direct is not None and (
                (backing_side.is_left() and match.h2h_direct <= -0.95)
                or (backing_side.is_right() and match.h2h_direct >= 0.95)
            ):
                log.info(
                    "{} reject back bad h2h: {}\n{}".format(
                        self.__class__.__name__, match.h2h_direct, match.tostring()
                    )
                )
                return False  # h2h is strong against back player
            fst_bet_chance = match.first_player_bet_chance()
            if fst_bet_chance is None:
                return False
            if fst_bet_chance < 0.1 or fst_bet_chance > 0.9:
                log.info(
                    "{} reject too big favor: {} {}\n{}".format(
                        self.__class__.__name__,
                        backing_side,
                        fst_bet_chance,
                        match.tostring(),
                    )
                )
                return False
            if (backing_side.is_left() and fst_bet_chance > 0.5) or (
                backing_side.is_right() and fst_bet_chance < 0.5
            ):
                log.info(
                    "{} reject back {} lftchance: {}\n{}".format(
                        self.__class__.__name__,
                        backing_side,
                        fst_bet_chance,
                        match.tostring(),
                    )
                )
                return False  # we looked for: back player is NOT favorite
            vs_incomer_side = match.vs_incomer_advantage_side()
            if backing_side.is_oppose(vs_incomer_side):
                log.info(
                    "{} rejected by vs_incomer_side {} {}".format(
                        self.__class__.__name__, backing_side, match.tostring()
                    )
                )
                return False
        return trg_srv_side

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""

        def get_cur_insetscore():
            # break_route:
            if trg_srv_side.is_left():
                return self.insetscore[0] - 1, self.insetscore[1]
            else:
                return self.insetscore[0], self.insetscore[1] - 1

        cur_srv_side = trg_srv_side.fliped()  # inverse serve in prev game
        ingame_adv_side = ingame_advance_side(match.ingame)
        cur_insetscore = get_cur_insetscore()
        if cur_insetscore == match.score[-1]:
            if ingame_adv_side is not None and ingame_adv_side != cur_srv_side:
                return self.back_side(trg_srv_side)


class AlertTest(unittest.TestCase):
    class FakedPlayer:
        def __init__(self, ident, name):
            self.ident = ident
            self.name = name
            self.rating = ratings.Rating()

    @staticmethod
    def fliped_match(match):
        result = copy.deepcopy(match)
        result.first_player = copy.deepcopy(match.second_player)
        result.second_player = copy.deepcopy(match.first_player)
        result.score = result.score.fliped()
        result.left_service = (
            None if result.left_service is None else not result.left_service
        )
        result.ingame = tuple(reversed(result.ingame))
        return result

    @staticmethod
    def make_match_abstract(
        test_side_adv, scr, left_service, ingame=("0", "0"), tied_side_adv=co.LEFT
    ):
        lm = LiveMatch(live_event=None)
        lm.score = copy.deepcopy(scr)
        lm.left_service = left_service
        lm.ingame = ingame
        lm.first_player = AlertTest.FakedPlayer(ident=-1, name="Faked Name1")
        lm.second_player = AlertTest.FakedPlayer(ident=-2, name="Faked Name2")
        return lm

    def test_equal_abstract(self):
        alert = AlertScEqual(setname="decided", insetscore=(4, 4), back_srv=False)
        match = self.make_match_abstract(
            test_side_adv=co.RIGHT,
            scr=Score("6-7 7-5 3-4"),
            left_service=False,
            ingame=("40", "0"),
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage) and "[2]" in result.text)

        alert = AlertScEqual(setname="open", insetscore=(4, 4), back_srv=True)
        match = self.make_match_abstract(
            test_side_adv=co.RIGHT,
            scr=Score("4-4"),
            left_service=False,
            ingame=("0", "0"),
        )

    def test_pursue_pre00_bencic_putintseva_02(self):
        """expect: _find_alertion() NOT return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(0, 2),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("7-5 4-6 0-0"), left_service=True
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is not False)

    def test_pursue_pre01_bencic_putintseva_02(self):
        """expect: _find_alertion() NOT return False at 0-1"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(0, 2),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("7-5 4-6 0-1"), left_service=False
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is not False)

    def test_pursue_alert_bencic_putintseva_02(self):
        """expect: _find_alertion() NOT return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(0, 2),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("7-5 4-6 0-2"), left_service=True
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage))
        if isinstance(result, AlertMessage):
            self.assertEqual(co.LEFT, result.back_side)
        else:
            print("we got result: {}".format(result))

    def test_pursue_pre_barty_kvitova_20(self):
        """expect: _find_alertion() NOT return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(2, 0),
            back_srv=False,
            srv_leader=True,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("6-1 5-7 0-0"), left_service=True
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is not False)

    def test_pursue_pre_barty_kvitova_20_reject(self):
        """expect: _find_alertion() NOT return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(2, 0),
            back_srv=False,
            srv_leader=True,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("6-1 5-7 0-0"), left_service=False
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is False)

    def test_pursue_pre_barty_kvitova_30(self):
        """expect: _find_alertion() NOT return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(3, 0),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("6-1 5-7 0-0"), left_service=True
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is not False)

    def test_pursue_pre_barty_kvitova_30_reject(self):
        """expect: _find_alertion() return False in set begin"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(3, 0),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("6-1 5-7 0-0"), left_service=False
        )
        result = alert.find_alertion(match)
        self.assertTrue(result is False)

    def test_pursue_barty_kvitova_30_alert(self):
        """expect: _condition_now() return alert object"""
        alert = AlertScPursue(
            setname="decided",
            insetscore=(3, 0),
            back_srv=True,
            srv_leader=False,
            skip_cond=lambda _m: False,
        )
        match = self.make_match_abstract(
            test_side_adv=co.LEFT, scr=Score("6-1 5-7 3-0"), left_service=False
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage))
        if isinstance(result, AlertMessage):
            self.assertEqual(co.RIGHT, result.back_side)
        else:
            print("we got result: {}".format(result))

    def test_pursue_abstract(self):
        """expect: _condition_next() return alert object"""
        alert = AlertScPursue(
            setname="decided", insetscore=(4, 3), skip_cond=lambda _m: False
        )

        match = self.make_match_abstract(
            test_side_adv=co.LEFT,
            scr=Score("6-4 1-6 3-3"),
            left_service=True,
            ingame=("40", "15"),
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage))
        if isinstance(result, AlertMessage):
            self.assertTrue("[1]" in result.text)
        else:
            print("we got result: {}".format(result))

        match = self.make_match_abstract(
            test_side_adv=co.LEFT,
            scr=Score("6-4 1-6 4-2"),
            left_service=True,
            ingame=("15", "40"),
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage) and "[1]" in result.text)

    def test_pursue_abstract_inverse(self):
        """expect: _condition_next() return alert object"""
        alert = AlertScPursue(
            setname="decided", insetscore=(3, 4), skip_cond=lambda _m: False
        )

        match = self.make_match_abstract(
            test_side_adv=co.RIGHT,
            scr=Score("6-4 1-6 3-3"),
            left_service=False,
            ingame=("15", "40"),
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage) and "[2]" in result.text)

        match = self.make_match_abstract(
            test_side_adv=co.RIGHT,
            scr=Score("6-4 1-6 2-4"),
            left_service=False,
            ingame=("40", "15"),
        )
        result = alert.find_alertion(match)
        self.assertTrue(isinstance(result, AlertMessage) and "[2]" in result.text)


def allow_back_inset_keep_recovery(
    match, backing_side, back_aspect, opp_aspect, setname, okfun
):
    back_id, opp_id = match.first_player.ident, match.second_player.ident
    if backing_side == co.RIGHT:
        back_id, opp_id = match.second_player.ident, match.first_player.ident
    back_sv = inset_keep_recovery.read_player_sized_value(
        match.sex, back_aspect, setname, back_id
    )
    opp_sv = inset_keep_recovery.read_player_sized_value(
        match.sex, opp_aspect, setname, opp_id
    )
    min_size = 9 if match.sex == "wta" else 12
    if back_sv.size >= min_size and opp_sv.size >= min_size:
        return okfun(back_sv.value, opp_sv.value)


DEFAULT_MARKET_NAME = "MATCH_ODDS"
SET_WINNER_MARKET_NAME = "SET_WINNER"

TotalMatchedBackCoef = namedtuple("TotalMatchedBackCoef", "total_matched back_coef")


min_total_matched_back_coef = {
    (DEFAULT_MARKET_NAME, "wta"): TotalMatchedBackCoef(
        total_matched=4000.0, back_coef=1.33
    ),
    (DEFAULT_MARKET_NAME, "atp"): TotalMatchedBackCoef(
        total_matched=4000.0, back_coef=1.33
    ),
    (SET_WINNER_MARKET_NAME, "wta"): TotalMatchedBackCoef(
        total_matched=450.0, back_coef=1.25
    ),
    (SET_WINNER_MARKET_NAME, "atp"): TotalMatchedBackCoef(
        total_matched=450.0, back_coef=1.25
    ),
}


def admit_total_matched_back_coef(market_name, sex, total_matched, back_coef):
    threshold = min_total_matched_back_coef[(market_name, sex)]
    return total_matched >= threshold.total_matched and back_coef >= threshold.back_coef


def get_market_name(match, alert):
    if isinstance(alert, AlertSetAfterTieAffect):
        if match.score and match.score.sets_count(full=True) == 1:
            return SET_WINNER_MARKET_NAME
    return DEFAULT_MARKET_NAME


def allow_current_coefs(match, alert, backing_side, prob=None):
    market_name = get_market_name(match, alert)
    summary_href = match.summary_href()
    res = betfair_client.send_message(
        market_name,
        match.sex,
        "current_coefs",
        match.first_player.name,
        match.second_player.name,
        back_side=backing_side,
        summary_href="" if summary_href is None else summary_href,
    )
    if res:
        fst_coef, snd_coef, total_matched = res
        coef = fst_coef if backing_side == co.LEFT else snd_coef
        if coef >= 0 and total_matched >= 0:
            result = admit_total_matched_back_coef(
                market_name, match.sex, total_matched, coef
            )
            if not result:
                log.info(
                    "reject by coef {:.3f} matched: {} {} prob {}".format(
                        coef, total_matched, match, prob
                    )
                )
            return result


def allow_back(match, backing_side, alert, prob, case_name=None):
    if backing_side.is_oppose(match.recently_won_h2h_side()):
        log.info(
            "{} rejected by recently_won_h2h_side {} {}".format(
                alert.__class__.__name__,
                backing_side.fliped(),
                match.tostring(extended=True),
            )
        )
        return False

    if feature.is_player_feature(
        match.features, "retired", is_first=backing_side.is_left()
    ) or feature.is_player_feature(
        match.features, "absence", is_first=backing_side.is_left()
    ):
        log.info(
            f"{alert.__class__.__name__} rejected by ret/abs "
            f"back {backing_side} {match.tostring(extended=True)}"
        )
        return False

    if backing_side.is_oppose(match.last_res_advantage_side()):
        log.info(
            "{} rejected by last_results_side {} {}".format(
                alert.__class__.__name__,
                backing_side.fliped(),
                match.tostring(extended=True),
            )
        )
        return False

    vs_incomer_adv_side = match.vs_incomer_advantage_side()
    if backing_side.is_oppose(vs_incomer_adv_side):
        log.info(
            "{} rejected by vs_incomer_adv_side {}".format(
                alert.__class__.__name__, match.tostring(extended=True)
            )
        )
        return False

    if back_in_blacklist(match, backing_side, case_name):
        log.info(
            "{} rejected by blacklist {}".format(
                alert.__class__.__name__, match.tostring()
            )
        )
        return False

    # if betfair_client.is_initialized():
    #     allow_betfair = allow_current_coefs(match, alert, backing_side, prob)
    #     if allow_betfair is False:
    #         return False
    #     if (match.level == 'chal' or match.qualification) and allow_betfair is None:
    #         return False  # if betfair does silence than market possible is absent
    return True


def back_in_blacklist(match, backing_side: Side, case_name=None):
    backplr = match.first_player if backing_side.is_left() else match.second_player
    if backplr is None:
        return False
    return backplr.name in back_in_blacklist.blacknames_dct[(match.sex, case_name)]


# # scheme for getting blacklist (list of id for bad players in our case)
# MIN_SIZE = 23 #  20 if wta
# psize = df.groupby(['snd_pid']).size() # got Series, index - pids, values - sizes
# psize_min = psize[psize >= MIN_SIZE]
# pid_min = list(psize_min.index) # list of pid (for these players each size >= MIN_SIZE)
# grouped = df[LABEL_NAME].groupby(df['snd_pid'])
# black_pids = []
# for pid, group in grouped:
#   avg = group.mean() # avg is player's ratio of LABEL_NAME
#   if avg <= 0.5 and pid in pid_min:
#       black_pids.append(pid)
back_in_blacklist.blacknames_dct = {
    ("wta", None): [
        "Kyoka Okamura",
        "Josephine Boualem",
        "Tamara Curovic",
        "Elena Bogdan",
    ],
    ("atp", None): [
        "Thomaz Bellucci",
        "Marek Gengel",
        "Andrew Whittington",
        "Pierre Faivre",
        "Santiago Gonzalez",
        # low motivation:
        "Bernard Tomic",
    ],
    ("atp", "decided_00"): [],
    ("wta", "decided_00"): [],
}


def back_in_whitelist(match, backing_side: Side):
    backplr = match.first_player if backing_side.is_left() else match.second_player
    return (
        backplr is not None
        and backplr.name in back_in_whitelist.whitenames_dct[match.sex]
    )


back_in_whitelist.whitenames_dct = {"wta": ["Antonia Lottner"], "atp": []}


def is_light_set_win(back_side, match, sets_count=None):
    """if sets_count is not None then use it (optimization).
    check only first and second sets.
    """

    def is_light_set_win_num(setnum):
        if back_side in (co.LEFT, co.ANY) and match.score[setnum - 1] in (
            (6, 0),
            (6, 1),
            (5, 0),
        ):
            return True
        if back_side in (co.RIGHT, co.ANY) and match.score[setnum - 1] in (
            (0, 6),
            (1, 6),
            (0, 5),
        ):
            return True
        return False

    if sets_count is None:
        sets_count = match.score.sets_count(full=False)
    if sets_count >= 1 and is_light_set_win_num(setnum=1):
        return True
    if sets_count >= 2 and is_light_set_win_num(setnum=2):
        return True
    return False


def ingame_normalgo_sides(ingame):
    """return (LEFT,) or (RIGHT,) or (LEFT, RIGHT). If too early return empty tuple ()"""

    def is_too_early(fst, snd):
        return (fst, snd) in (("0", "0"), ("0", "15"), ("15", "0"), ("15", "15"))

    def is_left_advance_2points(fst, snd):
        return (fst, snd) in (("40", "0"), ("40", "15"), ("30", "0"))

    if ingame:
        if is_too_early(ingame[0], ingame[1]):
            return ()
        is_left_adv_2points = is_left_advance_2points(ingame[0], ingame[1])
        if is_left_adv_2points:
            return (co.LEFT,)
        is_right_adv_2points = is_left_advance_2points(ingame[1], ingame[0])
        if is_right_adv_2points:
            return (co.RIGHT,)
        return co.LEFT, co.RIGHT


def ingame_advance_side(ingame, oneballpoint=False):
    def is_left_advance(fst, snd):
        if oneballpoint:
            return (fst, snd) in (("40", "0"), ("40", "15"), ("40", "30"), ("A", "40"))
        else:
            return (fst, snd) in (("40", "0"), ("40", "15"))

    if ingame:
        if is_left_advance(ingame[0], ingame[1]):
            return co.LEFT
        if is_left_advance(ingame[1], ingame[0]):
            return co.RIGHT
