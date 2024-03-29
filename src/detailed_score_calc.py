﻿import sys
import unittest
from collections import OrderedDict, namedtuple
if sys.version_info >= (3, 9):
    from collections import abc
else:
    import abc
from functools import reduce
from typing import List, Dict

from loguru import logger as log
import common as co
from stat_cont import WinLoss
from feature import Feature, RigidFeature, make_pair2, make_pair, FeatureList
from score import Scr, nil_scr
from tennis import Match
from lev import soft_level
from detailed_score import SetItems, error_contains
import ratings_elo
from stat_cont import EdgeScrTrack


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


class CalcOnsetsrvCount(Calc):

    feat_corename = "onsetsrvcount"

    def __init__(self):
        self.fst_cnt = 0
        self.snd_cnt = 0

    def result_features(self, prefix: str) -> List[Feature]:
        """ prefix задает номер сета """
        f1, f2 = make_pair(
            fst_name=f"{prefix}_fst_{self.feat_corename}",
            snd_name=f"{prefix}_snd_{self.feat_corename}",
            fst_value=self.fst_cnt,
            snd_value=self.snd_cnt,
        )
        return [f1, f2]

    def proc_set(self, setitems: SetItems):
        edge_scr_track = EdgeScrTrack()
        for scr, dg in setitems:
            if not dg.tiebreak:
                edge_scr_track.put(x=scr[0], y=scr[1], left_serve=dg.left_opener)
        self.fst_cnt = edge_scr_track.serveonset_count(co.LEFT)
        self.snd_cnt = edge_scr_track.serveonset_count(co.RIGHT)


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
        log.warning(
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
    result.append(RigidFeature("decided_tiebreak", int(bool(match.decided_tiebreak))))
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
    ratings_elo.add_features(
        sex, tour.surface, match, result,
        corenames=['elo_pts', 'elo_alt_pts', 'elo_rank', 'elo_alt_rank'])
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
            CalcOnsetsrvCount(),
        ],
        2: [
            CalcExistScr(),
            CalcOnsetsrvCount(),
        ],
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
    log.add('../log/detailed_score_calc.log', level='INFO',
            rotation='10:00', compression='zip')

    unittest.main()
