
from collections import namedtuple, OrderedDict
import unittest
import copy

import log
import common as co
import score
from detailed_score import DetailedScore, DetailedGame


# разыгрывается геймбол (left_side - у левого ли геймбол,
#                        left_serve - левый ли подающий, multier - кратность)
GamePointEvent = namedtuple("GamePointEvent", "left_side left_serve tiebreak multier")


class MissPoint(object):
    def __init__(self, game_point_event, revenged, scope=co.GAME):
        self.scope = scope  # from co.Scope
        self.left_side = game_point_event.left_side
        self.left_serve = game_point_event.left_serve
        self.tiebreak = game_point_event.tiebreak
        self.multier = game_point_event.multier

        # Отыгрался ли позже этот мяч (None - нет, или отыгрался на уровне scope)
        self.revenged = revenged

    @property
    def side(self):
        return co.side(self.left_side)

    @property
    def on_serve(self):
        return self.left_side == self.left_serve

    def __eq__(self, other):
        return (
            self.scope == other.scope
            and self.left_side == other.left_side
            and self.left_serve == other.left_serve
            and self.tiebreak == other.tiebreak
            and self.multier == other.multier
            and self.revenged == other.revenged
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return (
            "{}(GamePointEvent(left_side={}, left_serve={}, tiebreak={}, "
            + "multier={}), revenged={}, scope={})"
        ).format(
            self.__class__.__name__,
            self.left_side,
            self.left_serve,
            self.tiebreak,
            self.multier,
            self.revenged,
            self.scope,
        )

    def __str__(self):
        return "scope:{} side:{} srv:{} tie:{} multier:{} revenged:{}".format(
            self.scope,
            "L" if self.left_side else "R",
            "l" if self.left_serve else "r",
            self.tiebreak,
            self.multier,
            self.revenged,
        )


def multier_sublists(misspoints):
    """вернем список списков.
    каждый подсписок из MissPoint относится к продвижению (desc) кратности для
    одного из игроков. Для обычного гейма maxlen(подсписок) = 3, для тайбрейка = 6.
    misspoints для мини куска игры должны идти в порядке убывания кратности.
    """

    def joinable(miss_point_prev, miss_point):
        return (
            miss_point_prev.left_side == miss_point.left_side
            and miss_point_prev.scope == miss_point.scope
            and miss_point_prev.multier == (miss_point.multier + 1)
        )

    result = []
    sublist = []
    for mp in misspoints:
        if not sublist:
            sublist.append(mp)
        else:
            if joinable(sublist[-1], mp):
                sublist.append(mp)
            else:
                result.append(copy.copy(sublist))
                sublist = [mp]
    if sublist:
        result.append(sublist)
    return result


def get_penalty(sex, misspoints, scope_predicate, max_penalty=None):
    """оценка штрафов за упущенные game/set/match misspoints
    вернем (left_penalty, right_penalty). Предполагаем, что
    пойнты идут в порядке убывания кратности (если был например тройной геймбол)
    """

    def toavg_coef(sublist_len):
        coef = 1.0
        if sublist_len > 1:
            coef += 0.2 * (sublist_len - 1)
        return coef

    def sublist_penalty(sublist):
        scope_srv_to_penalty = __penalty_miss_dict(sex)
        penalty = 0.0
        for mp in sublist:
            if not scope_predicate(mp.scope):
                continue
            if mp.left_side:
                penalty += scope_srv_to_penalty[(mp.scope, mp.left_serve)]
            else:
                penalty += scope_srv_to_penalty[(mp.scope, not mp.left_serve)]
        return penalty * toavg_coef(len(sublist))

    left_penalty, right_penalty = 0.0, 0.0
    for mult_sublist in multier_sublists(misspoints):
        penalty = sublist_penalty(mult_sublist)
        if mult_sublist:
            if mult_sublist[0].left_side:
                left_penalty += penalty
            else:
                right_penalty += penalty

    if max_penalty is not None:
        left_penalty = min(max_penalty, left_penalty)
        right_penalty = min(max_penalty, right_penalty)
    return (left_penalty, right_penalty)


def __penalty_miss_dict(sex):
    """# вернем dict{(scope, is_srv) : penalty} для miss point"""
    return (
        __penalty_miss_dict.wta_dict if sex == "wta" else __penalty_miss_dict.atp_dict
    )


GROW_COEF = 1.1

#   (scope,    is_srv): penalty
__penalty_miss_dict.wta_dict = {
    (co.GAME, True): 0.29328 * GROW_COEF,
    (co.GAME, False): 0.24 * GROW_COEF,
    (co.SET, True): 1.95552 * GROW_COEF,
    (co.SET, False): 1.6 * GROW_COEF,
    (co.MATCH, True): 5.37776,
    (co.MATCH, False): 4.4,
}

#   (scope,    is_srv): penalty
__penalty_miss_dict.atp_dict = {
    (co.GAME, True): 0.37384 * GROW_COEF,
    (co.GAME, False): 0.24 * GROW_COEF,
    (co.SET, True): 2.492 * GROW_COEF,
    (co.SET, False): 1.6 * GROW_COEF,
    (co.MATCH, True): 6.8532,
    (co.MATCH, False): 4.4,
}


def setball_chance_side(setscore_t, tiebreak):
    maxv = max(setscore_t[0], setscore_t[1])
    if maxv < 5:
        return None
    if setscore_t in ((6, 6), (12, 12), (0, 0)) and tiebreak:
        return co.ANY
    minv = min(setscore_t[0], setscore_t[1])
    if minv == maxv:
        return None
    return co.LEFT if maxv == setscore_t[0] else co.RIGHT


def matchball_chance_side(score_t, best_of_five, sball_chance_side):
    """sball_chance_side - who has setball chance in cur set (score_t[-1])"""
    sets_count = len(score_t)
    if sets_count <= 1 or sets_count > 5 or (sets_count > 3 and not best_of_five):
        return None
    w, l = 0, 0  # get previous sets score, and prevsets_lead_side for these sets.
    for setnum in range(1, sets_count):
        if score_t[setnum - 1][0] > score_t[setnum - 1][1]:
            w += 1
        else:
            l += 1
    if ((w, l) == (1, 0) and not best_of_five) or (w == 2 and l < 2 and best_of_five):
        prevsets_lead_side = co.LEFT
    elif ((w, l) == (0, 1) and not best_of_five) or (l == 2 and w < 2 and best_of_five):
        prevsets_lead_side = co.RIGHT
    elif ((w, l) == (1, 1) and not best_of_five) or ((w, l) == (2, 2) and best_of_five):
        prevsets_lead_side = co.ANY
    else:
        prevsets_lead_side = None
    if prevsets_lead_side == co.ANY:
        return sball_chance_side
    elif prevsets_lead_side is not None and sball_chance_side in (
        co.ANY,
        prevsets_lead_side,
    ):
        return prevsets_lead_side


def edit_miss_points(miss_points, predicate, attr, value):
    for mpoint in miss_points:
        if predicate(mpoint):
            setattr(mpoint, attr, value)


def edit_miss_dict(miss_dict, key_predicate, mp_predicate, attr, value):
    for key, mpoints in miss_dict.items():
        if key_predicate(key):
            for mpoint in mpoints:
                if mp_predicate(mpoint):
                    setattr(mpoint, attr, value)


def setballing(miss_dict, final_score_t):
    for score_t, mpoints in miss_dict.items():
        setnum = len(score_t)
        chance_side = setball_chance_side(score_t[setnum - 1], mpoints[0].tiebreak)
        if not chance_side:
            continue
        edit_miss_points(
            mpoints,
            predicate=(
                lambda m, cs=chance_side: True if co.ANY == cs else m.side == cs
            ),
            attr="scope",
            value=co.SET,
        )

    for setnum in range(1, len(final_score_t) + 1):
        final_setscore_t = final_score_t[setnum - 1]
        if score.is_full_set(final_setscore_t):
            winset_side = (
                co.LEFT if final_setscore_t[0] > final_setscore_t[1] else co.RIGHT
            )
            edit_miss_dict(
                miss_dict,
                key_predicate=lambda k: len(k) == setnum,
                mp_predicate=(
                    lambda m, ws=winset_side: m.revenged is None
                    and m.scope == co.SET
                    and m.side == ws
                ),
                attr="revenged",
                value=co.SET,
            )


def matchballing(miss_dict, best_of_five, final_score_t, decided_tie_info=None):
    was_matchballable_score = False
    for score_t, mpoints in miss_dict.items():
        sball_chance_side = setball_chance_side(score_t[-1], mpoints[0].tiebreak)
        if not sball_chance_side:
            continue
        mball_chance_side = matchball_chance_side(
            score_t, best_of_five, sball_chance_side
        )
        if not mball_chance_side:
            continue
        was_matchballable_score = True
        edit_miss_points(
            mpoints,
            predicate=(
                lambda m, cs=mball_chance_side: True if cs == co.ANY else m.side == cs
            ),
            attr="scope",
            value=co.MATCH,
        )

    final_setscore_t = final_score_t[-1]
    setnum = len(final_score_t)
    is_decided = setnum == (5 if best_of_five else 3)
    if (
        score.is_full_set(final_setscore_t, is_decided, decided_tie_info)
        and was_matchballable_score
    ):
        winmatch_side = (
            co.LEFT if final_setscore_t[0] > final_setscore_t[1] else co.RIGHT
        )
        edit_miss_dict(
            miss_dict,
            key_predicate=lambda _: True,
            mp_predicate=(
                lambda m: m.revenged is None
                and m.scope == co.MATCH
                and m.side == winmatch_side
            ),
            attr="revenged",
            value=co.MATCH,
        )


def detailed_game_miss_points(detailed_game):
    for point in detailed_game:
        if point.is_simulated:
            continue
        left_gp_cnt = point.game_point_score(left=True, before=True)
        if left_gp_cnt and not point.win(left=True):
            revenged = co.GAME if detailed_game.left_wingame else None
            yield MissPoint(
                GamePointEvent(
                    left_side=True,
                    left_serve=point.serve(left=True),
                    tiebreak=detailed_game.tiebreak,
                    multier=left_gp_cnt,
                ),
                revenged,
            )

        right_gp_cnt = point.game_point_score(left=False, before=True)
        if right_gp_cnt and not point.win(left=False):
            revenged = co.GAME if detailed_game.right_wingame else None
            yield MissPoint(
                GamePointEvent(
                    left_side=False,
                    left_serve=point.serve(left=True),
                    tiebreak=detailed_game.tiebreak,
                    multier=right_gp_cnt,
                ),
                revenged,
            )


# main api entry:
def detailed_score_miss_points(
    det_score, best_of_five, final_score=None, decided_tie_info=None
):
    miss_dict = OrderedDict()
    for score_t, det_game in det_score.items():
        mpoints = list(detailed_game_miss_points(det_game))
        if mpoints:
            miss_dict[score_t] = copy.copy(mpoints)
    setballing(
        miss_dict, det_score.final_score() if final_score is None else final_score
    )
    matchballing(
        miss_dict,
        best_of_five,
        det_score.final_score() if final_score is None else final_score,
        decided_tie_info=decided_tie_info,
    )
    return miss_dict


def miss_dict_count(miss_dict, key_predicate, point_predicate):
    cnt = 0
    for (key, mpoints) in miss_dict.items():
        if key_predicate(key):
            cnt += co.count(mpoints, point_predicate)
    return cnt


def miss_dict_count_at(miss_dict, score_key, predicate):
    if score_key in miss_dict:
        return co.count(miss_dict[score_key], predicate)
    return 0


def miss_dict_count_before(miss_dict, score_key, predicate, last_set_only):
    cnt = 0
    for (key, mpoints) in miss_dict.items():
        if key == score_key or len(key) > len(score_key):
            break
        if last_set_only and len(score_key) != len(key):
            continue
        if len(key) < len(score_key) or (
            len(key) == len(score_key) and key < score_key
        ):
            cnt += co.count(mpoints, predicate)
    return cnt


def failed_count_at(
    side, miss_dict, score_key, scope, revenged_predicate=lambda r: True
):
    predicate = (
        lambda p: p.side == side and p.scope == scope and revenged_predicate(p.revenged)
    )
    return miss_dict_count_at(miss_dict, score_key, predicate)


def failed_count_before(side, miss_dict, score_key, scope, last_set_only):
    return miss_dict_count_before(
        miss_dict,
        score_key,
        predicate=lambda p: p.side == side and p.scope == scope,
        last_set_only=last_set_only,
    )


class CountMissDictTest(unittest.TestCase):
    def test_failed_count_before(self):
        mdict = MakeMissDictTest.twosets_dscore_mdict()[1]
        # предполагаем что там 5 упущенных сетболов левого игрока и все в 1 сете:
        # два из них до ((6,5),) и три позже
        cnt_before = failed_count_before(
            co.LEFT, mdict, ((6, 5),), co.SET, last_set_only=True
        )
        self.assertEqual(2, cnt_before)

        cnt_before = failed_count_before(
            co.LEFT, mdict, ((7, 6), (6, 6)), co.SET, last_set_only=True
        )
        self.assertEqual(0, cnt_before)

        cnt_before = failed_count_before(
            co.LEFT, mdict, ((7, 6), (6, 6)), co.SET, last_set_only=False
        )
        self.assertEqual(5, cnt_before)


class MakeMissDictTest(unittest.TestCase):
    @staticmethod
    def oneset_dscore_mdict():
        ds = DetailedScore()
        mdict = OrderedDict()
        ds[((4, 4),)] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((5, 4),)] = DetailedGame(
            "10001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((5, 4),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
        ]
        ds[((5, 5),)] = DetailedGame(
            "11101", left_wingame=True, left_opener=True, tiebreak=False
        )
        mdict[((5, 5),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=True, tiebreak=False, multier=3
                ),
                revenged=co.GAME,
            )
        ]
        ds[((6, 5),)] = DetailedGame(
            "00011111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((6, 5),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=3
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
        ]
        ds[((6, 6),)] = DetailedGame(
            "1111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        return ds, mdict

    @staticmethod
    def twosets_dscore_mdict():
        ds, mdict = MakeMissDictTest.oneset_dscore_mdict()
        ds[((7, 6), (0, 0))] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (1, 0))] = DetailedGame(
            "1111", left_wingame=False, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (1, 1))] = DetailedGame(
            "01111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 1))] = DetailedGame(
            "001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (2, 2))] = DetailedGame(
            "110000", left_wingame=False, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 3))] = DetailedGame(
            "00011111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((7, 6), (2, 3))] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=3
                ),
                revenged=None,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=None,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=None,
            ),
        ]
        ds[((7, 6), (2, 4))] = DetailedGame(
            "110000", left_wingame=False, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 5))] = DetailedGame(
            "0011000", left_wingame=True, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (3, 5))] = DetailedGame(
            "11000111", left_wingame=True, left_opener=True, tiebreak=False
        )
        mdict[((7, 6), (3, 5))] = [
            MissPoint(
                GamePointEvent(
                    left_side=False, left_serve=True, tiebreak=False, multier=1
                ),
                revenged=None,
                scope=co.SET,
            )
        ]
        ds[((7, 6), (4, 5))] = DetailedGame(
            "10000", left_wingame=True, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (5, 5))] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (6, 5))] = DetailedGame(
            "10001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((7, 6), (6, 5))] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.MATCH,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.MATCH,
            ),
        ]
        ds[((7, 6), (6, 6))] = DetailedGame(
            "1111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        return ds, mdict

    def test_oneset_match(self):
        ds, mdict_wait = self.oneset_dscore_mdict()
        mdict = detailed_score_miss_points(ds, best_of_five=False)
        self.assertEqual(mdict_wait, mdict)

    def test_twosets_match(self):
        ds, mdict_wait = self.twosets_dscore_mdict()
        mdict = detailed_score_miss_points(ds, best_of_five=False)
        self.assertEqual(mdict_wait, mdict)


class FromDetailedGameTest(unittest.TestCase):
    def test_tiebreaks(self):
        dg = DetailedGame(
            "0000000", left_wingame=False, left_opener=True, tiebreak=True
        )
        self.assertEqual([], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "11111101", left_wingame=True, left_opener=True, tiebreak=True
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=False, tiebreak=True, multier=6),
            revenged=co.GAME,
        )
        self.assertEqual([mp], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "111111001", left_wingame=True, left_opener=True, tiebreak=True
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=False, tiebreak=True, multier=6),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=True, multier=5),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

    def test_1(self):
        dg = DetailedGame("1111", left_wingame=True, left_opener=True, tiebreak=False)
        self.assertEqual([], list(detailed_game_miss_points(dg)))

        dg = DetailedGame("11101", left_wingame=True, left_opener=True, tiebreak=False)
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=3),
            revenged=co.GAME,
        )
        self.assertEqual([mp], list(detailed_game_miss_points(dg)))

    def test_2(self):
        dg = DetailedGame(
            "11010011", left_wingame=True, left_opener=True, tiebreak=False
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=2),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=1),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "1101000111", left_wingame=True, left_opener=True, tiebreak=False
        )
        mp3r = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=1),
            revenged=None,
        )
        self.assertEqual([mp, mp2, mp3r], list(detailed_game_miss_points(dg)))

    def test_2_right(self):
        dg = DetailedGame(
            "00101100", left_wingame=False, left_opener=True, tiebreak=False
        )
        mp = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=2),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=1),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

        # -----------------------------------------
        dg = DetailedGame(
            "0010111000", left_wingame=False, left_opener=True, tiebreak=False
        )
        mp3L = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=1),
            revenged=None,
        )
        self.assertEqual([mp, mp2, mp3L], list(detailed_game_miss_points(dg)))


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", "debug")
    unittest.main()
