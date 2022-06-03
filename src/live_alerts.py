r"""
module gives Alert with subclasses.
"""
from collections import namedtuple
from typing import Optional
from dataclasses import dataclass, field

import clf_decided_00dog_apply

try:
    import clf_decided_00nodog_apply
except ImportError:
    clf_decided_00nodog_apply = None

from side import Side
import common as co
from loguru import logger as log
import clf_common as cco
from score import setidx_by_name
from live import LiveMatch

import feature
import predicts_db


def remain_soft_main_skip_cond(match):
    return match.soft_level != "main"


def remain_ge_chal_skip_cond(match):
    return match.level not in ("chal", "main", "gs", "masters", "teamworld")


AlertMessage = namedtuple(
    "AlertMessage",
    [
        "hashid",
        "text",
        "fst_id",
        "snd_id",
        "fst_name",
        "snd_name",
        "back_side",
        "case_name",
        "sex",
        "summary_href",
        "fst_betfair_name",
        "snd_betfair_name",
        "prob",
        "book_prob",
        "comment",
        "level",
    ],
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

    def case_name(self, match: LiveMatch):
        return ""

    def make_message(self, match: LiveMatch, back_side: Side, prob=None, comment=""):
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
            text=f"{match.tostring()} {back_text} {prob_text} {comment} {self_text}",
            fst_id=match.first_player.ident,
            snd_id=match.second_player.ident,
            fst_name=match.first_player.name if match.first_player else "",
            snd_name=match.second_player.name if match.second_player else "",
            back_side=back_side,
            case_name=self.case_name(match),
            sex=match.sex,
            summary_href=match.summary_href(),
            fst_betfair_name=match.first_player.disp_name("betfair", ""),
            snd_betfair_name=match.second_player.disp_name("betfair", ""),
            prob=prob,
            book_prob=book_prob,
            comment=comment,
            level=str(match.level),
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
        result = self.condition_common(match)
        if result in (False, True, None):
            return result
        trg_srv_side = result
        result = self.condition_now(match, trg_srv_side)
        if isinstance(result, tuple):
            return self.make_message(match, *result)
        if isinstance(result, AlertData):
            return self.make_message(
                match, result.back_side, result.proba, result.comment)
        if result in (co.LEFT, co.RIGHT):
            return self.make_message(match, result)
        return result

    def condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)"""
        raise NotImplementedError()

    def condition_common(self, match: LiveMatch):
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


def is_edge_choke_side(match: LiveMatch, setnum: int, check_side: Side) -> Optional[bool]:
    """ проверяется: check_side в сете setnum подавал на партию >= 1 раза и проиграл"""
    set_scr = match.score[setnum - 1]
    if (
        (check_side.is_left() and set_scr[0] < set_scr[1])
        or
        (check_side.is_right() and set_scr[0] > set_scr[1])
    ):
        cnt = match.quad_stat.edge_scores.serveonset_count(setnum=setnum, side=check_side)
        return cnt is not None and cnt >= 1


def final_comment(*args) -> str:
    return ' '.join([a for a in args if a])


class AlertSetDecided(AlertSet):
    """уведомление о решающем сете"""

    def __init__(self, back_opener: Optional[bool]):
        super(AlertSetDecided, self).__init__(
            setname="decided", back_opener=back_opener
        )

    def condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """
        if match.score:
            inset = match.score[-1]
            if inset in ((0, 0), (0, 1), (1, 0), (1, 1)):
                return self.back_side(set_opener_side)
        return True


class AlertSetDecidedWinDogFavClf(AlertSetDecided):
    """
    увед-ние о реш. сете, где есть рекомендация о предпочтительном игроке от
    классификатора где есть разделение fav/dog
    """

    common_rejectables = [
        'contra-tie', 'toobigfav', 'edge_choke', 'favinrnd1', 'change_surf',
        'recently_won_h2h', 'absence', 'retired', 'blacklist', 'last_results',
        'pastyear_nmatches'
    ]

    # исп-ся по причинам: 1) выигрыш от сильного фаворита мал, если входить при равном счете
    #                     2) clf легко может переобучен и переоценивать победу сильн. фав-та
    max_book_start_chances = {'atp': 0.82, 'wta': 0.81}

    def __init__(self):
        super().__init__(back_opener=None)

    def condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if clf_decided_00dog_apply.skip_cond(match):
            return False
        if match.left_service is None:
            return None
        return super().condition_common(match)

    def case_name(self, match: LiveMatch):
        return "decided_00dog"

    def condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
        Return back side (if fired), False (if gone), True (if attention)
        """

        def is_max_start_chance_over():
            if back_dog is False:
                start_bet_chance = match.first_player_bet_chance()
                if back_side == co.RIGHT:
                    start_bet_chance = 1.0 - start_bet_chance
                return start_bet_chance >= self.max_book_start_chances[match.sex]

        def checked_return():
            allow_res = allow_back(match, back_side)
            if is_max_start_chance_over():
                allow_res.add_comment('toobigfav')
            if is_edge_choke_side(match, setnum=2, check_side=back_side):
                allow_res.add_comment('edge_choke')
            if not back_dog and match.rnd == 'First' and match.sex == 'atp':
                allow_res.add_comment('favinrnd1')
            reject = allow_res.exist_comment(self.common_rejectables)
            main_comment = 'backdog' if back_dog else 'backfav'
            if reject:
                predicts_db.write_rejected(
                    match, self.case_name(match), back_side, reason=str(allow_res))
                log.info(f"{self.__class__.__name__} {match.name}"
                         f" reject our {back_side} {main_comment} {str(allow_res)}")
                return False
            return back_side, proba, main_comment

        if match.score:
            inset = match.score[-1]
            if inset == (0, 0):
                back_side, proba, back_dog = clf_decided_00dog_apply.match_has_min_proba(
                    match, set_opener_side)
                if back_side is None or (match.sex == 'wta' and match.level == 'chal'):
                    return False
                return checked_return()
        return True


class AlertSetDecidedWinNodogClf(AlertSetDecided):
    """
    увед-ние о реш. сете, где есть рекомендация о предпочтительном игроке от
    классификатора где нет разделения fav/dog (примерно равные оценки букмекеров)
    """

    def __init__(self):
        super().__init__(back_opener=None)

    def condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        if clf_decided_00nodog_apply and clf_decided_00nodog_apply.skip_cond(match):
            return False
        if match.left_service is None:
            return None
        return super().condition_common(match)

    def case_name(self, match: LiveMatch):
        return "decided_00nodog"

    def condition_now(self, match: LiveMatch, set_opener_side: Side):
        """ Analyze current match state.
            Return back side (if fired), False (if gone), True (if attention) """

        def checked_return(back_side: Side, proba):
            allow_res = allow_back(match, back_side)
            if is_edge_choke_side(match, setnum=2, check_side=back_side):
                allow_res.add_comment('edge_choke')
            comment = str(allow_res)
            reject = allow_res.exist_comment(
                ('edge_choke', 'contra-tie', 'recently_won_h2h', 'absence',
                 'retired', 'blacklist', 'change_surf', 'last_results',
                 'pastyear_nmatches'))
            if reject:
                predicts_db.write_rejected(
                    match, self.case_name(match), back_side, reason=comment)
                log.info(f"{self.__class__.__name__} {match.name}"
                         f" reject our {back_side} {comment}")
                return False
            return back_side, proba, comment

        if match.score:
            inset = match.score[-1]
            if inset == (0, 0) and clf_decided_00nodog_apply:
                clf_back_side, prob = clf_decided_00nodog_apply.match_has_min_proba(
                    match, set_opener_side)
                if clf_back_side is None:
                    return False
                return checked_return(clf_back_side, prob)
        return True


class AlertSetDecidedWinDispatchClf(AlertSetDecided):
    """
    увед-ние о реш. сете, где есть рекомендация о предпочтительном игроке от
    классификатора
    """

    def __init__(self, use_nodog=False):
        super().__init__(back_opener=None)
        self.alert_dogfav = AlertSetDecidedWinDogFavClf()
        if use_nodog:
            self.alert_nodog = AlertSetDecidedWinNodogClf()
        else:
            self.alert_nodog = None

    def condition_common(self, match: LiveMatch):
        """co.LEFT, co.RIGHT (who will opener), False (gone), True (attention)"""
        is_dogfav = cco.exist_dogfav(match)
        if is_dogfav:
            return self.alert_dogfav.condition_common(match)
        elif is_dogfav is False and self.alert_nodog:
            return self.alert_nodog.condition_common(match)
        return False

    def case_name(self, match: LiveMatch):
        is_dogfav = cco.exist_dogfav(match)
        if is_dogfav:
            return self.alert_dogfav.case_name(match)
        elif is_dogfav is False and self.alert_nodog:
            return self.alert_nodog.case_name(match)
        else:
            return 'decided_00'

    def condition_now(self, match: LiveMatch, set_opener_side: Side):
        """Analyze current match state.
            Return back side (if fired), False (if gone), True (if attention)
        """
        is_dogfav = cco.exist_dogfav(match)
        if is_dogfav:
            return self.alert_dogfav.condition_now(match, set_opener_side)
        elif is_dogfav is False and self.alert_nodog:
            return self.alert_nodog.condition_now(match, set_opener_side)
        return False


@dataclass
class CacheCell:
    value: Optional[float] = field(default=None, init=False, repr=False)

    def is_skip(self):
        return self.value == float('inf')

    def set_skip(self):
        self.value = float('inf')

    def set_value(self, value: float):
        self.value = value


@dataclass
class CachePairCell:
    __slots__ = ['left', 'right']
    left: CacheCell
    right: CacheCell

    def __getitem__(self, index):
        if index == 0:
            return self.left
        elif index == 1:
            return self.right
        else:
            raise ValueError(f'invalid index {index} for CachePairCell')

    def set_skip_all(self):
        self.left.set_skip()
        self.right.set_skip()


class AllowBackResult:
    """ служит для аккумуляции возможных ошибок (рез-т ф-ии allow_back).
        __bool__ gives True if empty content (comment)
    """

    def __init__(self, comment: str = ''):
        self.comment = comment

    def add_comment(self, note: str):
        if self.comment:
            self.comment += f" {note}"
        else:
            self.comment = note

    def exist_comment(self, notes):
        for note in notes:
            if note in self.comment:
                return True

    def __bool__(self):
        return bool(self.comment) is False

    def __str__(self):
        return self.comment

    def __repr__(self):
        return f'{self.__class__.__name__}({self.comment})'


def allow_back(match: LiveMatch, back_side: Side, case_name=None) -> AllowBackResult:
    result = AllowBackResult()
    if back_side.is_oppose(match.recently_won_h2h_side()):
        result.add_comment('recently_won_h2h')

    if feature.is_player_feature(match.features, "retired", is_first=back_side.is_left()):
        result.add_comment('retired')
    if feature.is_player_feature(match.features, "absence", is_first=back_side.is_left()):
        result.add_comment('absence')

    if back_side.is_oppose(match.last_res_advantage_side()):
        result.add_comment('last_results')
    if back_side.is_oppose(match.change_surf_advantage_side()):
        result.add_comment('change_surf')

    if back_side.is_oppose(match.vs_incomer_advantage_side()):
        result.add_comment('vs_incomer')

    if back_in_blacklist(match, back_side, case_name):
        result.add_comment('blacklist')

    pastyear_nmatches_feat = feature.player_feature(
        match.features, name='pastyear_nmatches', is_first=back_side.is_left())
    if pastyear_nmatches_feat and pastyear_nmatches_feat.value < 10:
        result.add_comment('pastyear_nmatches')

    return result


def back_in_blacklist(match, backing_side: Side, case_name=None):
    """ return True if back_player in black list or opponent in strong_opponents """
    if backing_side.is_left():
        backplr, oppoplr = match.first_player, match.second_player
    else:
        backplr, oppoplr = match.second_player, match.first_player

    isinlist = backplr.name in back_in_blacklist.blacknames_dct[(match.sex, case_name)]
    if isinlist:
        return True

    return False


back_in_blacklist.blacknames_dct = {
    ("wta", None): [
        "Petra Kvitova",  # 3 retires in 2021, 1 retire in feb 2022
        "Lesya Tsurenko",  # 4 retires in [2021-11-24, 2022-04-21]
    ],
    ("atp", None): [
        "Bernard Tomic",  # low motivation
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
            return co.LEFT,
        is_right_adv_2points = is_left_advance_2points(ingame[1], ingame[0])
        if is_right_adv_2points:
            return co.RIGHT,
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
