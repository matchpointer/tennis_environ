r"""
module gives Alert with subclasses.
"""
from collections import namedtuple
from typing import Optional


from side import Side
import common as co
import log
import ratings
import clf_common as cco
from score import Scr, setidx_by_name
import clf_decided_00
import clf_secondset_00
from live import LiveMatch
import betfair_client
import inset_keep_recovery

import tieloss_affect_stat
import set2_after_set1loss_stat
import feature
from report_line import SizedValue
import predicts_db


def remain_soft_main_skip_cond(match):
    return match.soft_level != "main"


def remain_ge_chal_skip_cond(match):
    return match.level not in ("chal", "main", "gs", "masters", "teamworld")


AlertMessage = namedtuple(
    "AlertMessage",
    "hashid text fst_id snd_id fst_name snd_name back_side "
    "case_name sex summary_href fst_betfair_name snd_betfair_name "
    "prob book_prob comment",
)


class AlertData:
    def __init__(self, back_side: Side, proba: float, comment: str = ''):
        self.back_side = back_side
        self.proba = proba
        self.comment = comment


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
        back_text = f"[{'1' if back_side.is_left() else '2'}]"
        prob_text = f"P={round(prob, 3) if prob is not None else '_'}"
        book_prob = (match.first_player_bet_chance() if back_side.is_left()
                     else 1. - match.first_player_bet_chance())
        return AlertMessage(
            hashid=match.hashid(add_text=self_text),
            text=(
                f"{match.tostring()} {back_text} {prob_text} {comment} {self_text}"
            ),
            fst_id=match.first_player.ident,
            snd_id=match.second_player.ident,
            fst_name=match.first_player.name if match.first_player else "",
            snd_name=match.second_player.name if match.second_player else "",
            back_side=back_side,
            case_name=self.case_name(),
            sex=match.sex,
            summary_href=match.summary_href(),
            fst_betfair_name=match.first_player.disp_name("betfair", ""),
            snd_betfair_name=match.second_player.disp_name("betfair", ""),
            prob=prob,
            book_prob=book_prob,
            comment=comment,
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
        """ return AlertMessage (if fired), False (gone), True (attention)"""
        result = self._condition_common(match)
        if result in (False, True, None):
            return result
        trg_srv_side = result
        result = self._condition_now(match, trg_srv_side)
        if isinstance(result, tuple):
            return self.make_message(match, *result)
        if isinstance(result, AlertData):
            return self.make_message(
                match, result.back_side, result.proba, result.comment)
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


def tieloss_affect_side(
        match: LiveMatch, setnum: int, check_side: Side = None
) -> Optional[Side]:
    """ setnum >= 2 is currently gaming. Prev set is possible ended with tiebreak.
        если передан check_side, то проверяется только он.
        возращается сторона, подверженная аффекту и проигравшая tie in prevset
    """
    if setnum < 2:
        log.error(f"tieloss_affect_side bad setnum : {setnum} match: {match}")
        return None
    prevset_scr = match.score[setnum - 2]
    if prevset_scr not in ((6, 7), (7, 6)):
        return None
    if prevset_scr[0] > prevset_scr[1]:
        check_id = match.second_player.ident
        tocheck_side = co.RIGHT
    else:
        check_id = match.first_player.ident
        tocheck_side = co.LEFT
    if check_side and check_side == tocheck_side:
        isaff = tieloss_affect_stat.is_player_affect_high(match.sex, check_id)
        if isaff:
            return check_side


class AlertSetDecided(AlertSet):
    """уведомление о решающем сете"""

    def __init__(self, back_opener: Optional[bool]):
        super(AlertSetDecided, self).__init__(
            setname="decided", back_opener=back_opener
        )

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score:
            inset = match.score[-1]
            if inset in ((0, 0), (0, 1), (1, 0), (1, 1)):
                return self.back_side(set_opener_side)
        return True


class AlertSetDecidedWinClf(AlertSetDecided):
    """
    увед-ние о реш. сете, где есть рекомендация о предпочтительном игроке от
    классификатора
    """

    def __init__(
        self,
        back_opener: Optional[bool],
        strict_min_probas: Optional[cco.PosNeg],
    ):
        super().__init__(back_opener=back_opener)
        self.strict_min_probas = strict_min_probas
        self.dec_begin_adv_dif = 0.1
        self.dec_begin_min_size = 10

    def _condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if clf_decided_00.skip_cond(match):
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
        opener_id, closer_id = AlertSetDecidedWinClf.get_opener_closer(
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

    @staticmethod
    def tieloss_affect_opener_adv(match: LiveMatch, set_opener_side: Side):
        """ return True (set opener advantage), False (set closer advantage),
                   None if nobody has advantage """
        aff_side = tieloss_affect_side(match, setnum=3)
        if aff_side is None:
            return None
        return aff_side == set_opener_side.fliped()

    def decision(
        self,
        match,
        set_opener_side,
        clf_back_side,
        clf_prob,
        opener_has_tieaff_adv,
        opener_has_begin_adv,
    ):
        def codirect(var1: Optional[bool], var2: Optional[bool]):
            if var1 is not None and var2 is not None:
                return var1 == var2

        def strict_fired():
            if self.strict_min_probas:
                variant = cco.Variant.find_match_variant(
                    match, clf_decided_00.work_variants)
                if variant is not None:
                    return (
                        (clf_back_side == set_opener_side
                         and self.strict_min_probas.neg >= variant.min_probas.neg)
                        or
                        (clf_back_side != set_opener_side
                         and self.strict_min_probas.pos >= variant.min_probas.pos)
                    )

        def checked_return(back_side: Side, prob, comment: str, rejected: bool):
            is_fired = (
                not self.strict_min_probas
                or strict_fired()
                or (match.sex == 'wta' and '+' in comments)
            )
            allow_res = None
            if is_fired:
                if not rejected:
                    allow_res = allow_back(match, back_side, self, prob)
                    if allow_res:
                        return back_side, prob, comment
                    rejected = True
            if rejected:
                if allow_res is not None and allow_res.comment:
                    comment += ' ' + allow_res.comment
                log.info(
                    f"{self.__class__.__name__} rejected {match.name}"
                    f" back_side {back_side} {comment}"
                )
                predicts_db.write_rejected(
                    match, 'decided_00', back_side, reason=comment)
            return False

        if clf_back_side is None:
            return False
        opener_has_clf_adv = clf_back_side == set_opener_side
        tieaff_codirect = codirect(opener_has_clf_adv, opener_has_tieaff_adv)
        begadv_codirect = codirect(opener_has_clf_adv, opener_has_begin_adv)
        comments = ""
        if tieaff_codirect is not None:
            comments += ("+tieaff" if tieaff_codirect else "-tieaff")
        if begadv_codirect is not None:
            comments += (" +begadv" if begadv_codirect else " -begadv")

        if tieaff_codirect is False:
            return checked_return(clf_back_side, clf_prob, comments, rejected=True)
        if begadv_codirect is False:
            if clf_prob < 0.7:
                comments += " cP<0.7"
                return checked_return(clf_back_side, clf_prob, comments, rejected=True)
            else:
                comments += " cP>0.7"
        return checked_return(clf_back_side, clf_prob, comments, rejected=False)

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
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

    def _condition_common(self, match: LiveMatch):
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

    def _condition_common(self, match: LiveMatch):
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

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
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
                and (sv.value + 0.03) < set2_after_set1loss_stat.read_generic_value(
                    match.sex, "set2win_after_set1win")
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
    def __init__(self):
        super().__init__(setname="second", back_opener=None)
        self.trg_setnum = 2

    def case_name(self):
        return "secondset_00"

    def _condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        result = super()._condition_common(match)
        if result in (co.LEFT, co.RIGHT):
            if clf_secondset_00.skip_cond(match):
                return False
        return result

    def _condition_now(self, match: LiveMatch, set_opener_side: Side):
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
                if backing_side is None:
                    return False
                allow_res = allow_back(match, backing_side, self, prob)
                is_tieaff = tieloss_affect_side(
                        match, setnum=2, check_side=backing_side) == backing_side
                if not allow_res or is_tieaff:
                    comment = allow_res.comment
                    if is_tieaff:
                        comment += ' ' + 'tieaff'
                    log.info(
                        f"{self.__class__.__name__} rejected {match.name}"
                        f" back: {backing_side} {comment}"
                    )
                    predicts_db.write_rejected(
                        match, self.case_name(), backing_side, reason=comment)
                    return False
                return backing_side, prob
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
        if isinstance(result, AlertData):
            return self.make_message(
                match, result.back_side, result.proba, result.comment)
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
    """ assume back_srv=True, back_srv=False - are implemented in two different objects"""
    def __init__(
        self,
        setname: str,
        insetscore: Scr,
        back_srv: bool,
        min_back_ratio: float,
        max_lay_ratio: float,
        min_size: int,
    ):
        super().__init__(setname=setname, insetscore=insetscore, back_srv=back_srv)
        assert insetscore[0] == insetscore[1], "invalid equal score {} ".format(
            insetscore
        )
        self.min_back_ratio = min_back_ratio
        self.max_lay_ratio = max_lay_ratio
        self.min_size = min_size

    def _setend_back_has_adv(self, match, trg_srv_side):
        """ return (flag, proba) where flag: has adv. Possible returns:
            (True, proba: float) - exist adv,
            (False, None) - not exist adv, thresholds are not implemented,
            (None, None) - not found features, or < min_size data.
            Эта реализация интересуется только srv hold обоих игроков.
            stage1 (st1) гейм при 4-4 (aspect='setend' у открывателя).
            stage2 (st2) след.гейм (asp='onstaymatch' если st2 откр oppo иначе 'onmatch').
            proba вычисляется приблизительно грубо как эмпир. вер-ть ok_st1 and ok_st2
        """
        st1_feat_name = f"{self.setname}_setend_srv"
        st2_aspect = 'onstaymatch' if self.back_srv else 'onmatch'
        st2_feat_name = f"{self.setname}_{st2_aspect}_srv"
        try:
            st1_feat = feature.player_feature(match.features, st1_feat_name,
                                              is_first=trg_srv_side.is_left())
            st2_feat = feature.player_feature(match.features, st2_feat_name,
                                              is_first=trg_srv_side.is_right())
            if st1_feat.value.size < self.min_size or st2_feat.value.size < self.min_size:
                return None, None
        except cco.FeatureError:
            return None, None
        if (
            self.back_srv
            and st1_feat.value.value >= self.min_back_ratio
            and st2_feat.value.value <= self.max_lay_ratio
        ):
            proba = st1_feat.value.value * (1 - st2_feat.value.value)
            return True, proba
        elif (
            not self.back_srv
            and st1_feat.value.value <= self.max_lay_ratio
            and st2_feat.value.value >= self.min_back_ratio
        ):
            proba = (1 - st1_feat.value.value) * st2_feat.value.value
            return True, proba
        return False, None

    def _condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will serve), False (gone), True (attention)"""
        if match.sex == "atp" and self.setname == "open" and match.best_of_five:
            return False
        # тут можно спросить match.cache ["decided_44"] is False -> return False
        # если там True, то возможно имеет смысл ждать.
        # Если None то готовим значение (False - нет смысла ни для одной из сторон,
        # иначе True)
        trg_srv_side = super()._condition_common(match)
        if trg_srv_side in (co.LEFT, co.RIGHT):
            is_adv, proba = self._setend_back_has_adv(match, trg_srv_side)
            if (is_adv, proba) == (None, None):
                return False
            if (is_adv, proba) == (False, None):
                log.info(
                    f"{self.case_name()} reject {match.name}"
                    f" trg_srv_side:{trg_srv_side} scr {match.score}"
                )
                return False
        return trg_srv_side

    def _write_predict_db(self, match, backing_side: Side, proba: float):
        predicts_db.write_predict(
            match, case_name=self.case_name(),
            back_side=backing_side, proba=proba,
            comments='OPENER' if self.back_srv else 'CLOSER')

    def _condition_now(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        if match.score[-1] == self.insetscore and match.ingame in (None, ("0", "0")):
            backing_side = self.back_side(trg_srv_side)
            is_adv, proba = self._setend_back_has_adv(match, trg_srv_side)
            if is_adv:
                self._write_predict_db(match, backing_side, proba)
            if is_adv and allow_back(match, backing_side, self, proba):
                return backing_side, proba
            if is_adv is False:
                log.info(
                    f"{self.case_name()} reject now {match.name}"
                    f" trg_srv_side:{trg_srv_side}"
                )
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

        self.min_proba = min_proba  # it is 'mixed' proba
        self.max_lay_ratio = max_lay_ratio
        self.min_back_ratio = 0.5
        self.min_size = min_size
        self.max_oppo_rank_adv_dif = 60

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

    def get_backside_proba(self, match):
        sv1, sv2 = self.sized_values(match)
        if sv1.size < self.min_size or sv2.size < self.min_size:
            return None, None
        proba1, proba2 = co.twoside_values(sv1, sv2)
        if proba1 < self.min_proba and proba2 < self.min_proba:
            return None, None
        srfrnkcmp = match.get_ranks_cmp(rtg_name="elo_alt", is_surface=True)
        if (
            proba1 >= self.min_proba
            and sv1.value >= self.min_back_ratio
            and sv2.value <= self.max_lay_ratio
            and self.rtg_side_cond.condition(match, co.LEFT)
            and not srfrnkcmp.side_prefer(co.RIGHT, self.max_oppo_rank_adv_dif)
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
            and sv2.value >= self.min_back_ratio
            and sv1.value <= self.max_lay_ratio
            and self.rtg_side_cond.condition(match, co.RIGHT)
            and not srfrnkcmp.side_prefer(co.LEFT, self.max_oppo_rank_adv_dif)
            and (
                self.setname == "decided"
                or tieloss_affect_stat.is_player_affect_above_avg(
                    match.sex, match.second_player.ident
                )
                is False
            )
        ):
            return co.RIGHT, proba2

    def _condition_common(self, match: LiveMatch):
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
                backing_side, proba = None, None
                backing_side_prob = self.get_backside_proba(match)
                if backing_side_prob is not None:
                    backing_side, proba = backing_side_prob
                if backing_side is None:
                    match.set_cache_value(featname, value=False)
                    return False
                else:
                    match.set_cache_value(
                        featname,
                        value=AlertData(backing_side, proba,
                                        self.comment(match, backing_side))
                    )
        return trg_srv_side

    def _condition_now(self, match: LiveMatch, trg_srv_side: Side):
        """Analyze current match state.
        Return AlertData (if fired), False (if reject), None (if nothing hapen)"""
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
            if cache_val is False or cache_val is None:
                return False
            alert_data = cache_val
            if allow_back(match, alert_data.back_side, self, alert_data.proba):
                return alert_data
            return False

    def _condition_next(self, match, trg_srv_side):
        """Analyze for next game from current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        return None  # disable


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


class AllowBackResult:
    def __init__(self, code: bool, comment: str):
        self.code = code
        self.comment = comment

    def __bool__(self):
        return bool(self.code)

    def __str__(self):
        return self.comment

    def __repr__(self):
        return f'AllowBackResult({self.code}, {self.comment})'


def allow_back(match, backing_side, alert, prob, case_name=None) -> AllowBackResult:
    if backing_side.is_oppose(match.recently_won_h2h_side()):
        log.info(
            "{} rejected by recently_won_h2h_side {} {}".format(
                alert.__class__.__name__,
                backing_side.fliped(),
                match.tostring(extended=True),
            )
        )
        return AllowBackResult(False, 'recently_won_h2h')

    is_retired = feature.is_player_feature(
        match.features, "retired", is_first=backing_side.is_left())
    is_absence = feature.is_player_feature(
        match.features, "absence", is_first=backing_side.is_left())
    if is_retired or is_absence:
        log.info(
            f"{alert.__class__.__name__} rejected by ret/abs "
            f"back {backing_side} {match.tostring(extended=True)}")
        return AllowBackResult(False, 'retired' if is_retired else 'absence')

    if backing_side.is_oppose(match.last_res_advantage_side()):
        log.info(
            "{} rejected by last_results_side {} {}".format(
                alert.__class__.__name__,
                backing_side.fliped(),
                match.tostring(extended=True),
            )
        )
        return AllowBackResult(False, 'last_results')

    if prob is not None and prob < 0.75:
        vs_incomer_adv_side = match.vs_incomer_advantage_side()
        if backing_side.is_oppose(vs_incomer_adv_side):
            log.info(
                "{} rejected by vs_incomer_adv_side {}".format(
                    alert.__class__.__name__, match.tostring(extended=True)
                )
            )
            return AllowBackResult(False, 'vs_incomer')

    if back_in_blacklist(match, backing_side, case_name):
        log.info(
            "{} rejected by blacklist {}".format(
                alert.__class__.__name__, match.tostring()
            )
        )
        return AllowBackResult(False, 'blacklist')

    return AllowBackResult(True, '')


def back_in_blacklist(match, backing_side: Side, case_name=None):
    backplr = match.first_player if backing_side.is_left() else match.second_player
    if backplr is None:
        return False
    return backplr.name in back_in_blacklist.blacknames_dct[(match.sex, case_name)]


back_in_blacklist.blacknames_dct = {
    ("wta", None): [],
    ("atp", None): [],
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
