# -*- coding: utf-8 -*-
import sys
import unittest
from collections import OrderedDict, namedtuple
if sys.version_info >= (3, 9):
    from collections import abc
else:
    import abc
from functools import reduce
from typing import List, Dict

import log
import common as co
from stat_cont import WinLoss
from feature import Feature, RigidFeature, make_pair2, make_pair, FeatureList
from score import Scr, nil_scr, ingame_to_num, tie_normalized
from tennis import soft_level, Match
from detailed_score import SetItems, error_contains
import ratings_elo
from side import Side
from markov import set_win_prob2, tie_win_prob2


class Calc(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def result_features(self, prefix: str) -> List[Feature]:
        raise NotImplementedError()

    @abc.abstractmethod
    def proc_set(self, setitems: SetItems):
        raise NotImplementedError()


class CalcExistScr(Calc):
    """
    отвечает на вопросы о наличии счета из нижеприведенного списка _exist_scores.
    ориентация счетов СЕМАНТИЧЕСКИ (реально открыватель партии м.б. на люб. стороне)
    привязана к открывателю партии:
       при x == y открыватель партии на подаче;
       при (x + 1, x) открыватель партии лидирует гейм и принимает
       при (x, x + 1) открыватель партии отстает гейм и принимает (-1 брейк)
    если счет из _exist_scores найден, то его значение 1.
        если счет не найден, то его значение 0.
        если неизвестно / мало данных, то его значение -1.
    """

    _exist_scores = (
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),  # set-opener srv & equal
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),  # set-opener rcv & lead
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),  # set-opener rcv & trail
    )

    def __init__(self):
        self.result_dct = OrderedDict()

    def result_features(self, prefix: str) -> List[Feature]:
        """вернет List[RigidFeature] т.к. счета привязаны к семантич. ситуациям,
        не зависящим от swap игроков"""
        return [
            RigidFeature(name=f"{prefix}_is_{x}-{y}", value=v)
            for (x, y), v in self.result_dct.items()
        ]

    def proc_set(self, setitems: SetItems):
        set_opener_left = setitems.set_opener_side() == co.LEFT
        for x, y in self._exist_scores:
            if x == y:
                is_scr = setitems.exist_scr((x, y), left_opener=set_opener_left)
            elif abs(x - y) == 1:
                is_scr = setitems.exist_scr(
                    (x, y)
                    if set_opener_left
                    else (y, x),  # adapt to real set-opener side
                    left_opener=not set_opener_left,  # set-opener receive
                )
            else:
                raise co.TennisError("_score_exist_dict bad scr ex {}, {}".format(x, y))
            self.result_dct[(x, y)] = -1 if is_scr is None else int(is_scr)

    def score_exist_val(self, x, y):
        return self.result_dct[(x, y)]


class CalcSimulGames(Calc):
    def __init__(self):
        self.n_simul_games = 0

    def result_features(self, prefix: str) -> List[Feature]:
        return [RigidFeature(name=f"{prefix}_simul_games", value=self.n_simul_games)]

    def proc_set(self, setitems: SetItems):
        for _, dg in setitems:
            if error_contains(dg.error, "GAME_SCORE_SIMULATED"):
                self.n_simul_games += 1


class CalcEmptyGames(Calc):
    def __init__(self):
        self.n_empty_games = 0

    def result_features(self, prefix: str) -> List[Feature]:
        return [RigidFeature(name=f"{prefix}_empty_games", value=self.n_empty_games)]

    def proc_set(self, setitems: SetItems):
        for _, dg in setitems:
            if len(dg) == 0:
                self.n_empty_games += 1


class CalcAbsentGames(Calc):
    feat_corename = "absent_games"

    def __init__(self):
        self.n_absent_games = None

    def result_features(self, prefix: str) -> List[Feature]:
        return [RigidFeature(name=self.feature_name(prefix), value=self.n_absent_games)]

    def proc_set(self, setitems: SetItems):
        if setitems.fin_scr != nil_scr:
            self.n_absent_games = max(0, sum(setitems.fin_scr) - len(setitems))

    def feature_name(self, prefix: str) -> str:
        return f"{prefix}_{self.feat_corename}"

    def cumul_result_features(
        self, prefix: str, others: "List[CalcAbsentGames]"
    ) -> List[Feature]:
        if self.n_absent_games is None:
            return [RigidFeature(name=self.feature_name(prefix), value=None)]
        n_absent_games = self.n_absent_games
        for calc in others:
            if calc.n_absent_games is None:
                return [RigidFeature(name=self.feature_name(prefix), value=None)]
            n_absent_games += calc.n_absent_games
        return [RigidFeature(name=self.feature_name(prefix), value=n_absent_games)]


class CalcWinclose(Calc):
    """how close players was as set-leader"""

    feat_corename = "winclose"

    def __init__(
        self,
        sex: str,
        surface,
        level,
        is_qual: bool,
        min_prob: float,
        is_mean: bool = False,
        is_before66: bool = False,
    ):
        self.fst_measure = 0.0
        self.snd_measure = 0.0
        self.min_prob = min_prob
        self.sex = sex
        self.surface = surface
        self.level = level
        self.is_qual = is_qual
        self.fst_n = 0
        self.snd_n = 0
        self.is_mean = is_mean
        self.is_before66 = is_before66

    def _feature_name(self, prefix: str, is_fst: bool) -> str:
        return (
            f"{prefix}_{'fst' if is_fst else 'snd'}_"
            f"{self.feat_corename}_{'mean_' if self.is_mean else ''}"
            f"{'before66_' if self.is_before66 else ''}"
            f"{int(self.min_prob * 100)}"
        )

    def result_features(self, prefix: str) -> List[Feature]:
        if self.is_mean:
            val1 = (self.fst_measure / self.fst_n) if self.fst_n else 0.0
            val2 = (self.snd_measure / self.snd_n) if self.snd_n else 0.0
        else:
            val1 = self.fst_measure
            val2 = self.snd_measure
        f1, f2 = make_pair(
            fst_name=self._feature_name(prefix, is_fst=True),
            snd_name=self._feature_name(prefix, is_fst=False),
            fst_value=val1,
            snd_value=val2,
        )
        return [f1, f2]

    def proc_set(self, setitems: SetItems):
        result_loser, result_winner = 0.0, 0.0
        loser_n, winner_n = 0, 0
        is_fliped = False
        setopener = setitems.set_opener_side()
        if setopener == co.RIGHT:
            setitems.flip()
            is_fliped = True
        setwinner = setitems.set_winner_side()
        if setwinner is None:
            return
        setloser: Side = setwinner.fliped()
        for scr, dg in setitems:
            if self.is_before66 and (scr == (6, 6) or sum(scr) > 12):
                break
            for pnt in dg:
                p_scr_txt = pnt.text_score(before=True)
                p_scr = ingame_to_num(p_scr_txt, tiebreak=dg.tiebreak)
                if dg.tiebreak:
                    p_scr = tie_normalized(p_scr, is_super=dg.supertiebreak)
                    prob = tie_win_prob2(
                        self.sex,
                        self.surface,
                        self.level,
                        self.is_qual,
                        p_scr,
                        Side(dg.left_opener),
                        Side(pnt.serve()),
                        target_side=setloser,
                    )
                else:
                    prob = set_win_prob2(
                        self.sex,
                        self.surface,
                        self.level,
                        self.is_qual,
                        scr,
                        p_scr,
                        Side(dg.left_opener),
                        target_side=setloser,
                    )
                if prob is not None:
                    if prob >= self.min_prob:
                        result_loser += prob
                        loser_n += 1
                    if (1.0 - prob) >= self.min_prob:
                        result_winner += 1.0 - prob
                        winner_n += 1
                if pnt.win_game() or pnt.loss_game():
                    break
        if setloser.is_left():
            self.fst_measure = result_loser
            self.fst_n = loser_n
            self.snd_measure = result_winner
            self.snd_n = winner_n
        else:
            self.snd_measure = result_loser
            self.snd_n = loser_n
            self.fst_measure = result_winner
            self.fst_n = winner_n

        if is_fliped:
            self.fst_measure, self.snd_measure = self.snd_measure, self.fst_measure
            self.fst_n, self.snd_n = self.snd_n, self.fst_n
            setitems.flip()


class CalcLastinrow(Calc):
    GameWinPos = namedtuple("GameWinPos", "scr is_ok")

    def __init__(self):
        self.fst_lastrow_games = None

    def result_features(self, prefix: str) -> List[Feature]:
        name = f"{prefix}_fst_lastrow_games"
        if self.fst_lastrow_games is None:
            return [RigidFeature(name=name, value=None)]
        return [
            Feature(
                name=name,
                value=self.fst_lastrow_games,
                flip_value=-self.fst_lastrow_games,
            )
        ]

    def proc_set(self, setitems: SetItems):
        if setitems.fin_scr[1] == 0:
            self.fst_lastrow_games = setitems.fin_scr[0]
            return
        if setitems.fin_scr[0] == 0:
            self.fst_lastrow_games = -setitems.fin_scr[1]
            return

        setwinner = setitems.set_winner_side()
        if setwinner is None:
            return
        setloser = setwinner.fliped()
        loser_game_wins = []
        for scr, dg in setitems:
            if (dg.left_wingame and setloser == co.LEFT) or (
                dg.right_wingame and setloser == co.RIGHT
            ):
                game_ok = not (error_contains(dg.error, "GAME_SCORE_SIMULATED"))
                loser_game_wins.append(self.GameWinPos(scr=scr, is_ok=game_ok))
        n_loser_game_wins = len(loser_game_wins)
        if (
            n_loser_game_wins > 0
            and n_loser_game_wins == min(setitems.fin_scr)
            and loser_game_wins[-1].is_ok
        ):
            inrow = sum(setitems.fin_scr) - sum(loser_game_wins[-1].scr) - 1
            self.fst_lastrow_games = inrow if setwinner == co.LEFT else -inrow


class CalcSrvRatio(Calc):

    feat_corename = "srv_ratio"  # win ratio

    complete_usial_game: Dict[Scr, Scr] = {
        (0, 3): (8, 6),  # from 0:40
        (1, 3): (8, 6),  # from 15:40
        (2, 3): (6, 4),  # from 30:40
        (0, 2): (7, 5),  # from 0:30
        (1, 2): (5, 3),  # from 15:30
        (0, 1): (6, 4),  # from 0:15
        (0, 0): (4, 2),  # from 0:0
        (1, 0): (4, 1),  # from 15:0
        (1, 1): (4, 2),  # from 15:15
        (2, 0): (4, 1),  # from 30:0
        (2, 1): (4, 2),  # from 30:15
        (2, 2): (5, 3),  # from 30:30
        (3, 3): (6, 4),  # from 40:40
        (3, 2): (4, 2),  # from 40:30
        (3, 1): (4, 1),  # from 40:15
        (3, 0): (4, 0),  # from 40:0
    }

    @staticmethod
    def complete_usial_fin_score(fin_scr: Scr) -> Scr:
        """suppose: game winner is left in fin_scr"""
        if fin_scr[0] >= 4 and fin_scr[0] >= (fin_scr[1] + 2):
            return fin_scr  # ok need not complete
        res_fin_scr = CalcSrvRatio.complete_usial_game.get(fin_scr)
        if res_fin_scr is not None:
            return res_fin_scr
        if abs(fin_scr[0] - fin_scr[1]) <= 1:
            x = max(4, fin_scr[0] + 2)
            return x, fin_scr[1]
        log.warn(
            f"lose fin {fin_scr} game winner CalcSrvRatio::complete_usial_fin_score"
        )
        return 5, 4

    def __init__(self):
        self.fst_wl = WinLoss()
        self.snd_wl = WinLoss()

    def result_features(self, prefix: str) -> List[Feature]:
        f1, f2 = make_pair(
            fst_name=f"{prefix}_fst_{self.feat_corename}",
            snd_name=f"{prefix}_snd_{self.feat_corename}",
            fst_value=self.fst_wl.ratio,
            snd_value=self.snd_wl.ratio,
        )
        return [f1, f2]

    def cumul_result_features(
        self, prefix: str, others: "List[CalcSrvRatio]"
    ) -> List[Feature]:
        fst_wl = reduce(lambda x, y: x + y, [o.fst_wl for o in others], self.fst_wl)
        snd_wl = reduce(lambda x, y: x + y, [o.snd_wl for o in others], self.snd_wl)
        f1, f2 = make_pair(
            fst_name=f"{prefix}_fst_{self.feat_corename}",
            snd_name=f"{prefix}_snd_{self.feat_corename}",
            fst_value=fst_wl.ratio,
            snd_value=snd_wl.ratio,
        )
        return [f1, f2]

    def proc_set(self, setitems: SetItems):
        for _, dg in setitems:
            if not dg.tiebreak:
                self._proc_usial_game(dg)
            else:
                self._proc_tiebreak(setitems, dg)

    def _proc_tiebreak(self, setitems: SetItems, dg):
        scr = (0, 0)
        for pnt in dg:
            if pnt.serve(left=True):
                srv_win = pnt.win(left=True)
                self.fst_wl.hit(srv_win)
                scr = (scr[0] + 1, scr[1]) if srv_win else (scr[0], scr[1] + 1)
            else:
                srv_win = pnt.win(left=False)
                self.snd_wl.hit(srv_win)
                scr = (scr[0], scr[1] + 1) if srv_win else (scr[0] + 1, scr[1])

        if scr <= setitems.tie_scr and scr != setitems.tie_scr:
            # approximate finishing
            x_add, y_add = setitems.tie_scr[0] - scr[0], setitems.tie_scr[1] - scr[1]
            x_win, x_lose = x_add // 2, y_add // 2
            self.fst_wl.add_win(x_win)
            self.fst_wl.add_loss(x_lose)
            y_win, y_lose = y_add // 2, x_add // 2
            self.snd_wl.add_win(y_win)
            self.snd_wl.add_loss(y_lose)

    def _proc_usial_game(self, dg):
        fin_scr = dg.final_num_score(before=False) or (0, 0)
        wl = self.fst_wl if dg.left_opener else self.snd_wl
        if dg.hold:
            res_fin_scr = self.complete_usial_fin_score(fin_scr)
        else:
            # game winner is right in fin_scr
            res_fin_scr = self.complete_usial_fin_score(co.reversed_tuple(fin_scr))
            res_fin_scr = co.reversed_tuple(res_fin_scr)  # we got again right dominate
        wl.add_win(res_fin_scr[0])
        wl.add_loss(res_fin_scr[1])


class CalcSrvRatioBeforeTie(CalcSrvRatio):
    """
    for collect data before tiebreak (not including tiebreak).
    If tiebreak is not encountered (or encountered at improper places) than collect all data
    """

    feat_corename = "beftie_srv_ratio"  # win ratio

    def __init__(self):
        super().__init__()

    def proc_set(self, setitems: SetItems):
        for num, (scr, dg) in enumerate(setitems, start=1):
            if not dg.tiebreak:
                self._proc_usial_game(dg)
            else:
                if scr in ((0, 0), (6, 6), (12, 12)) and num == len(setitems):
                    break
                self._proc_tiebreak(setitems, dg)


def common_features(sex: str, tour, best_of_five, match: Match) -> FeatureList:
    result = FeatureList([RigidFeature("date", match.date)])
    co.add_pair(
        result,
        *make_pair2(
            "pid",
            fst_value=match.first_player.ident,
            snd_value=match.second_player.ident,
        ),
    )
    result.append(Feature("fst_win_match", value=1, flip_value=0))
    result.append(RigidFeature("level_text", soft_level(tour.level, match.rnd)))
    result.append(RigidFeature("raw_level_text", str(tour.level)))
    result.append(RigidFeature("tour_id", tour.ident))
    result.append(RigidFeature("tour_money", tour.money))
    result.append(RigidFeature("tour_rank", tour.rank))
    result.append(RigidFeature("rnd_text", match.rnd.value))
    result.append(RigidFeature("best_of_five", int(bool(best_of_five))))
    result.append(RigidFeature("decided_tiebreak", int(match.decided_tiebreak)))
    result.append(RigidFeature("surface_text", str(tour.surface)))
    result.extend(match.score.features())
    h2h_direct = match.head_to_head.direct()
    age1, age2 = match.first_player.age(match.date), match.second_player.age(match.date)
    co.add_pair(
        result, *make_pair("fst_age", "snd_age", fst_value=age1, snd_value=age2)
    )
    co.add_pair(
        result,
        *make_pair(
            "fst_lefty",
            "snd_lefty",
            fst_value=match.first_player.lefty,
            snd_value=match.second_player.lefty,
        ),
    )

    result.append(
        Feature(
            "h2h_direct",
            value=h2h_direct,
            flip_value=None if h2h_direct is None else -h2h_direct,
        )
    )
    fst_bet_chance = match.first_player_bet_chance()
    result.append(
        Feature(
            "fst_bet_chance",
            value=fst_bet_chance,
            flip_value=None if fst_bet_chance is None else 1.0 - fst_bet_chance,
        )
    )
    ratings_elo.add_features(sex, tour.surface, match, result)
    ratings_elo.add_features(sex, tour.surface, match, result, isalt=True)
    co.add_pair(
        result,
        *make_pair2(
            corename="std_rank",
            fst_value=match.first_player.rating.rank("std"),
            snd_value=match.second_player.rating.rank("std"),
        ),
    )

    result.extend(_simple_calc_features(sex, match, tour.level, tour.surface))
    return result


def _simple_calc_features(sex, match, level, surface) -> List[Feature]:
    simple_calcs_dict: Dict[int, List[Calc]] = {
        1: [
            CalcExistScr(),
            CalcSrvRatio(),
            CalcLastinrow(),
            CalcSimulGames(),
            CalcEmptyGames(),
            CalcAbsentGames(),
            CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.55),
            CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.58),
            CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.60),
            CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.62),
            CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.65),
            CalcWinclose(
                sex,
                surface,
                level,
                match.rnd.qualification(),
                min_prob=0.55,
                is_mean=True,
            ),
            CalcWinclose(
                sex,
                surface,
                level,
                match.rnd.qualification(),
                min_prob=0.58,
                is_mean=True,
            ),
            CalcWinclose(
                sex,
                surface,
                level,
                match.rnd.qualification(),
                min_prob=0.60,
                is_mean=True,
            ),
            CalcWinclose(
                sex,
                surface,
                level,
                match.rnd.qualification(),
                min_prob=0.62,
                is_mean=True,
            ),
            CalcWinclose(
                sex,
                surface,
                level,
                match.rnd.qualification(),
                min_prob=0.65,
                is_mean=True,
            ),
        ],
        2: [CalcExistScr()],
    }

    res_features = []
    for setnum, setitems in match.set_items.items():
        if setnum in simple_calcs_dict:
            for calc in simple_calcs_dict[setnum]:
                calc.proc_set(setitems)
                res_features.extend(calc.result_features(prefix=f"s{setnum}"))
    return res_features


def openspecific_calc_features(setitems_dict: Dict[int, SetItems]) -> FeatureList:
    res_features = FeatureList([])
    setitems = setitems_dict[1]
    calc = CalcSrvRatioBeforeTie()
    calc.proc_set(setitems)
    res_features.extend(calc.result_features(prefix="s1"))
    return res_features


def decspecific_calc_features(
    best_of_five: bool, sex: str, match: Match, level, surface
) -> FeatureList:
    res_features = FeatureList([])

    setitems_dict: Dict[int, SetItems] = match.set_items
    prev_decided(best_of_five, setitems_dict, CalcLastinrow(), res_features)
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.55),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.58),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.60),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.62),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(sex, surface, level, match.rnd.qualification(), min_prob=0.65),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(
            sex, surface, level, match.rnd.qualification(), min_prob=0.55, is_mean=True
        ),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(
            sex, surface, level, match.rnd.qualification(), min_prob=0.58, is_mean=True
        ),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(
            sex, surface, level, match.rnd.qualification(), min_prob=0.60, is_mean=True
        ),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(
            sex, surface, level, match.rnd.qualification(), min_prob=0.62, is_mean=True
        ),
        res_features,
    )
    prev_decided(
        best_of_five,
        setitems_dict,
        CalcWinclose(
            sex, surface, level, match.rnd.qualification(), min_prob=0.65, is_mean=True
        ),
        res_features,
    )

    cumulate_to_prev_decided(best_of_five, setitems_dict, CalcSrvRatio(), res_features)
    cumulate_to_prev_decided(
        best_of_five, setitems_dict, CalcSimulGames(), res_features
    )
    cumulate_to_prev_decided(
        best_of_five, setitems_dict, CalcEmptyGames(), res_features
    )
    cumulate_to_prev_decided(
        best_of_five, setitems_dict, CalcAbsentGames(), res_features
    )

    prefix = "sd"  # decided
    calc_scr = CalcExistScr()
    if best_of_five:
        calc_scr.proc_set(setitems_dict[5])
    else:
        calc_scr.proc_set(setitems_dict[3])
    res_features.extend(calc_scr.result_features(prefix))

    prefix = "ssd"  # summarize decided
    calc_srvratio = CalcSrvRatio()
    calc_srvratio_beftie = CalcSrvRatioBeforeTie()

    calc_srvratio.proc_set(setitems_dict[1])
    calc_srvratio.proc_set(setitems_dict[2])
    if best_of_five:
        calc_srvratio.proc_set(setitems_dict[3])
        calc_srvratio.proc_set(setitems_dict[4])
        calc_srvratio_beftie.proc_set(setitems_dict[5])
    else:
        calc_srvratio_beftie.proc_set(setitems_dict[3])
    res_features.extend(
        calc_srvratio_beftie.cumul_result_features(prefix, [calc_srvratio])
    )
    return res_features


def prev_decided(
    best_of_five: bool,
    setitems_dict: Dict[int, SetItems],
    calc: Calc,
    res_features: FeatureList,
):
    if best_of_five:
        calc.proc_set(setitems_dict[4])
    else:
        calc.proc_set(setitems_dict[2])
    res_features.extend(calc.result_features(prefix="spd"))


def cumulate_to_prev_decided(
    best_of_five: bool,
    setitems_dict: Dict[int, SetItems],
    calc: Calc,
    res_features: FeatureList,
):
    calc.proc_set(setitems_dict[1])
    calc.proc_set(setitems_dict[2])
    if best_of_five:
        calc.proc_set(setitems_dict[3])
        calc.proc_set(setitems_dict[4])
    res_features.extend(calc.result_features(prefix="sspd"))


if __name__ == "__main__":
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    unittest.main()
