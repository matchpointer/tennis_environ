import os
import datetime
from collections import defaultdict, namedtuple
from operator import itemgetter
import copy
from contextlib import closing
from typing import Dict, Tuple, Optional, DefaultDict

import tkinter.ttk

import oncourt.sql
from oncourt import dbcon, extplayers
import common as co
import tennis_time as tt
from loguru import logger as log
from surf import make_surf
from stat_cont import create_summator_histogram, WinLoss, Sumator
import cfg_dir
from tennis import Round
from lev import soft_level
import dict_tools
from report_line import SizedValue
from bet import get_betcity_company, PickWin
import file_utils as fu


class StatSide(object):
    def __init__(
        self, srv1in, srv1win, srv2win, bpwin, aces, double_faults, tot_pnt_won
    ):
        """на входе пары (сколько выиграно, всего) -
        кроме aces, double_faults, tot_pnt_won"""

        def make_winloss(win_of):
            """Тут на входе пара (сколько выиграно, всего)"""
            if win_of[0] is not None and win_of[1] is not None:
                return WinLoss(win_of[0], win_of[1] - win_of[0])

        self.first_service_in = make_winloss(srv1in)
        self.first_service_win = make_winloss(srv1win)
        self.second_service_win = make_winloss(srv2win)
        self.bp_win = make_winloss(bpwin)
        self.aces = aces
        self.double_faults = double_faults
        self.tot_pnt_won = tot_pnt_won

    def service_win(self):
        if (
            self.first_service_win is not None
            and self.second_service_win is not None
            and self.first_service_in is not None
        ):
            in_ratio = self.first_service_in.ratio
            fst_ratio = self.first_service_win.ratio
            snd_ratio = self.second_service_win.ratio
            if in_ratio is not None and fst_ratio is not None and snd_ratio is not None:
                res_ratio = in_ratio * fst_ratio + (1.0 - in_ratio) * snd_ratio
                res_size = self.first_service_win.size + self.second_service_win.size
                return WinLoss.from_ratio(res_ratio, res_size)

    def other_receive_win(self):
        """общая результативность на приеме для оппонента этой стороны"""
        result = self.service_win()
        if result is not None:
            return result.reversed()

    def other_receive_first_win(self):
        """результативность на приеме первой подачи для оппонента этой стороны"""
        if self.first_service_win is not None:
            return self.first_service_win.reversed()

    def other_receive_second_win(self):
        """результативность на приеме второй подачи для оппонента этой стороны"""
        if self.second_service_win is not None:
            return self.second_service_win.reversed()


class MatchStat:
    SexSurf_OptFloat = Dict[
        Tuple[
            str,  # sex
            str  # surface
        ],
        Optional[float]  # value from [0...1]
    ]

    fst_srv_in: SexSurf_OptFloat = defaultdict(lambda: None)

    @staticmethod
    def generic_first_service_in(sex, surface) -> Optional[float]:
        """0 <= return <= 1"""
        result = MatchStat.fst_srv_in[(sex, surface)]
        if result is not None:
            return result
        structkey = co.StructKey(level="main", surface=str(surface))
        MatchStat.fst_srv_in[(sex, surface)] = generic_result_dict(
            sex, "first_service_in"
        )[structkey].value
        return MatchStat.fst_srv_in[(sex, surface)]

    def __init__(
        self,
        srv1in_1,
        srv1win_1,
        srv2win_1,
        bpwin_1,
        aces_1,
        df_1,
        tpw_1,
        srv1in_2,
        srv1win_2,
        srv2win_2,
        bpwin_2,
        aces_2,
        df_2,
        tpw_2,
    ):
        """Тут на входе пары (сколько выиграно, всего) - кроме aces, df, tpw"""
        self.left_side = StatSide(
            srv1in_1, srv1win_1, srv2win_1, bpwin_1, aces_1, df_1, tpw_1
        )
        self.right_side = StatSide(
            srv1in_2, srv1win_2, srv2win_2, bpwin_2, aces_2, df_2, tpw_2
        )

    def flip(self):
        self.left_side, self.right_side = (
            self.right_side,
            self.left_side,
        )

    def break_points_saved(self):
        left_result, right_result = None, None
        if self.right_side.bp_win is not None:
            left_result = self.right_side.bp_win.reversed()

        if self.left_side.bp_win is not None:
            right_result = self.left_side.bp_win.reversed()
        return left_result, right_result

    def break_points_converted(self):
        return self.left_side.bp_win, self.right_side.bp_win

    def total_points_won(self):
        """return (left_count, right_count)"""
        return self.left_side.tot_pnt_won, self.right_side.tot_pnt_won

    def is_total_points_won(self, min_size=60):
        lval, rval = self.total_points_won()
        if lval is None or rval is None:
            return False
        return (lval + rval) >= min_size

    def aces_pergame(self, score):
        left_srv_games, right_srv_games, ties = self.__service_games(score)
        if left_srv_games is None or right_srv_games is None:
            return None, None

        if (left_srv_games + ties) > 0 and self.left_side.aces is not None:
            left_result = float(self.left_side.aces) / float(left_srv_games + ties)
        else:
            left_result = None

        if (right_srv_games + ties) > 0 and self.right_side.aces is not None:
            right_result = float(self.right_side.aces) / float(right_srv_games + ties)
        else:
            right_result = None
        return left_result, right_result

    def double_faults_pergame(self, score):
        left_srv_games, right_srv_games, ties = self.__service_games(score)
        if left_srv_games is None or right_srv_games is None:
            return None, None

        if (left_srv_games + ties) > 0 and self.left_side.double_faults is not None:
            left_result = float(self.left_side.double_faults) / float(
                left_srv_games + ties
            )
        else:
            left_result = None

        if (right_srv_games + ties) > 0 and self.right_side.double_faults is not None:
            right_result = float(self.right_side.double_faults) / float(
                right_srv_games + ties
            )
        else:
            right_result = None
        return left_result, right_result

    def service_win(self):
        return self.left_side.service_win(), self.right_side.service_win()

    def receive_win(self):
        return self.right_side.other_receive_win(), self.left_side.other_receive_win()

    def first_service_in(self):
        return self.left_side.first_service_in, self.right_side.first_service_in

    def __service_games(self, score):
        """вернем (left_srv_games, right_srv_games)"""

        def games_layout():
            lw, rw, ties = 0, 0, 0
            for setscore in score:
                if setscore in ((7, 6), (6, 7)):
                    lw += 6
                    rw += 6
                    ties += 1
                else:
                    lw += setscore[0]
                    rw += setscore[1]
            return lw, rw, ties

        def complete_game(lhs_win_games, rhs_win_games):
            """определим кто имел больше геймов со своей подачей.
            вход: lhs_win_games - выиграно геймов левым по счету без тай-брейков
                  rhs_win_games - выиграно геймов правым по счету без тай-брейков
            выход: (1, 0) - больше у левого. (0, 1) - у правого. (0, 0) - не опред.
            """
            middle = (lhs_win_games + rhs_win_games) / 2
            lhs = (
                middle
                - self.right_side.bp_win.win_count
                + self.left_side.bp_win.win_count
            )
            rhs = (
                middle
                - self.left_side.bp_win.win_count
                + self.right_side.bp_win.win_count
            )
            if rhs >= rhs_win_games and lhs < lhs_win_games:
                return 1, 0  # complete to left
            if lhs >= lhs_win_games and rhs < rhs_win_games:
                return 0, 1  # complete to right
            return 0, 0

        left_win_games, right_win_games, ties = games_layout()
        games_count_half = (left_win_games + right_win_games) / 2
        if ((left_win_games + right_win_games) % 2) == 0:
            return games_count_half, games_count_half, ties
        if self.left_side.bp_win is None or self.right_side.bp_win is None:
            return None, None, None
        left_add_game, right_add_game = complete_game(left_win_games, right_win_games)
        if left_add_game == 0 and right_add_game == 0:
            log.debug("UNKNOWN matchstat complete game: {}".format(score))
            left_add_game = 1
        return (
            games_count_half + left_add_game,
            games_count_half + right_add_game,
            ties,
        )

    def service_hold(self, score):
        left_srv_games, right_srv_games, _ = self.__service_games(score)
        if left_srv_games is None or right_srv_games is None:
            return None, None
        return self.__service_hold_impl(left_srv_games, right_srv_games)

    def __service_hold_impl(self, left_srv_games, right_srv_games):
        """input: left_srv_games - всего геймов с подачей левого игрока
        right_srv_games - всего геймов с подачей правого игрока
        """
        left_result, right_result = None, None
        if self.right_side.bp_win is not None:
            win_srv_games = left_srv_games - self.right_side.bp_win.win_count
            left_result = WinLoss(win_srv_games, self.right_side.bp_win.win_count)
        if self.left_side.bp_win is not None:
            win_srv_games = right_srv_games - self.left_side.bp_win.win_count
            right_result = WinLoss(win_srv_games, self.left_side.bp_win.win_count)
        return left_result, right_result

    def receive_hold(self, score):
        left_result, right_result = self.service_hold(score)
        if left_result is not None:
            left_result = left_result.reversed()
        if right_result is not None:
            right_result = right_result.reversed()
        return left_result, right_result


#  player_min_id ~ MatchStat.left_side, player_max_id ~ MatchStat.right_side
StorageKey = namedtuple("StorageKey", "tour_id rnd player_min_id player_max_id")

Sex_YearWeeknum_Storkey_OptMstat = DefaultDict[
    str,  # sex
    DefaultDict[  # will ordered?
        Tuple[
            int,  # year
            int  # week num
        ],
        DefaultDict[
            StorageKey,
            Optional[MatchStat]
        ]
    ]
]

_sex_dict: Sex_YearWeeknum_Storkey_OptMstat = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: None))
)

Sex_Tourid_OptSurface = DefaultDict[
    str,  # sex
    DefaultDict[
        int,  # tour_id
        Optional[str]  # surface
    ]
]

_sex_surf_dict: Sex_Tourid_OptSurface = defaultdict(lambda: defaultdict(lambda: None))


def initialize(
    sex=None, min_date=None, max_date=None, time_reverse=False, tour_surface=False
):
    if sex in ("wta", None):
        if "wta" in _sex_dict:
            _sex_dict["wta"].clear()
        if "wta" in _sex_surf_dict:
            _sex_surf_dict["wta"].clear()
        __initialize_sex(
            "wta", min_date=min_date, max_date=max_date, time_reverse=time_reverse
        )
        if tour_surface:
            __initialize_sex_surface("wta", min_date=min_date, max_date=max_date)

    if sex in ("atp", None):
        if "atp" in _sex_dict:
            _sex_dict["atp"].clear()
        if "atp" in _sex_surf_dict:
            _sex_surf_dict["atp"].clear()
        __initialize_sex(
            "atp", min_date=min_date, max_date=max_date, time_reverse=time_reverse
        )
        if tour_surface:
            __initialize_sex_surface("atp", min_date=min_date, max_date=max_date)


def get(sex, tour_id, rnd, first_player_id, second_player_id, year_weeknum):
    """return mstat/none."""
    player_min_id = min(first_player_id, second_player_id)
    player_max_id = max(first_player_id, second_player_id)
    storkey = StorageKey(tour_id, rnd, player_min_id, player_max_id)
    mstat = _sex_dict[sex][year_weeknum][storkey]
    if mstat is not None:
        if player_min_id == second_player_id:
            mstat.flip()
    return mstat


def player_srvrcv_win(sex, player_id, year_weeknums, surface_to, stop_stat_size):
    """return (SizedValue, SizedValue) for values: (srv_win_ratio, rcv_win_ratio)"""

    def get_srv_rcv(mstat, side):
        """return (WinLoss, WinLoss). If [i] not supplied then None tuple member"""
        if side.is_left():
            return mstat.left_side.service_win(), mstat.right_side.other_receive_win()
        else:
            return mstat.right_side.service_win(), mstat.left_side.other_receive_win()

    srv_win, rcv_win = Sumator(), Sumator()
    ywn_to_dct = _sex_dict[sex]
    for ywn in year_weeknums:
        dct = ywn_to_dct[ywn]
        for storkey, mstat in dct.items():
            if mstat is None:
                continue
            tot_pnt_won = mstat.total_points_won()
            if (
                tot_pnt_won[0]
                and tot_pnt_won[1]
                and (tot_pnt_won[0] + tot_pnt_won[1]) <= 65
            ):
                continue  # too short match, may be retired.
            if storkey.player_min_id == player_id:
                srvwl, rcvwl = get_srv_rcv(mstat, co.LEFT)
            elif storkey.player_max_id == player_id:
                srvwl, rcvwl = get_srv_rcv(mstat, co.RIGHT)
            else:
                continue
            surface_from = _sex_surf_dict[sex][storkey.tour_id]
            if srvwl and srvwl.ratio < 1.01 and srv_win.size <= stop_stat_size:
                srv_coef = generic_surface_trans_coef(
                    sex, "service_win", surface_from, surface_to
                )
                srv_win += srvwl.ratio * srv_coef
            if rcvwl and rcvwl.ratio < 1.01 and rcv_win.size <= stop_stat_size:
                rcv_coef = generic_surface_trans_coef(
                    sex, "receive_win", surface_from, surface_to
                )
                rcv_win += rcvwl.ratio * rcv_coef
            if srv_win.size >= stop_stat_size and rcv_win.size >= stop_stat_size:
                return (
                    SizedValue.create_from_sumator(srv_win),
                    SizedValue.create_from_sumator(rcv_win),
                )
    return (
        SizedValue.create_from_sumator(srv_win),
        SizedValue.create_from_sumator(rcv_win),
    )


def _matchstates(sex):
    result = []
    for storkey_dct in _sex_dict[sex].values():
        for mstat in storkey_dct.values():
            result.append(mstat)
    return result


def __initialize_sex_surface(sex, min_date=None, max_date=None):
    query = """
          select tours.ID_T, Courts.NAME_C
          from Tours_{0} AS tours, Courts 
          where tours.ID_C_T = Courts.ID_C
          {1};
          """.format(
        sex, oncourt.sql.sql_dates_condition(min_date, max_date)
    )
    with closing(dbcon.get_connect().cursor()) as cursor:
        for (tour_id, surf_txt) in cursor.execute(query):
            _sex_surf_dict[sex][tour_id] = make_surf(surf_txt)


def __initialize_sex(sex, min_date=None, max_date=None, time_reverse=False):
    query = """
       select tours.DATE_T, st.ID_T, st.ID_R, st.ID1, st.ID2,
         FS_1, FSOF_1, W1S_1, W1SOF_1, W2S_1, W2SOF_1, BP_1, BPOF_1, ACES_1, DF_1, TPW_1,
         FS_2, FSOF_2, W1S_2, W1SOF_2, W2S_2, W2SOF_2, BP_2, BPOF_2, ACES_2, DF_2, TPW_2
       from Stat_{0} AS st, Tours_{0} AS tours 
       where st.ID_T = tours.ID_T {1}
       order by tours.DATE_T {2}, st.ID_T, st.ID_R {2};
       """.format(
        sex,
        oncourt.sql.sql_dates_condition(min_date, max_date),
        "desc" if time_reverse else "asc",
    )
    with closing(dbcon.get_connect().cursor()) as cursor:
        for (
            tour_dt,
            tour_id,
            rnd_id,
            lhs_plr_id,
            rhs_plr_id,
            lhs_srv1in,
            lhs_srv1in_of,
            lhs_srv1win,
            lhs_srv1win_of,
            lhs_srv2win,
            lhs_srv2win_of,
            lhs_bpwin,
            lhs_bpwin_of,
            lhs_aces,
            lhs_df,
            lhs_tpw,
            rhs_srv1in,
            rhs_srv1in_of,
            rhs_srv1win,
            rhs_srv1win_of,
            rhs_srv2win,
            rhs_srv2win_of,
            rhs_bpwin,
            rhs_bpwin_of,
            rhs_aces,
            rhs_df,
            rhs_tpw,
        ) in cursor.execute(query):
            tour_date = tour_dt.date() if tour_dt else None
            rnd = Round.from_oncourt_id(rnd_id)
            plr_min_id = min(lhs_plr_id, rhs_plr_id)
            plr_max_id = max(lhs_plr_id, rhs_plr_id)
            storkey = StorageKey(tour_id, rnd, plr_min_id, plr_max_id)
            srv1in_1 = (lhs_srv1in, lhs_srv1in_of)
            srv1win_1 = (lhs_srv1win, lhs_srv1win_of)
            srv2win_1 = (lhs_srv2win, lhs_srv2win_of)
            bpwin_1 = (lhs_bpwin, lhs_bpwin_of)
            srv1in_2 = (rhs_srv1in, rhs_srv1in_of)
            srv1win_2 = (rhs_srv1win, rhs_srv1win_of)
            srv2win_2 = (rhs_srv2win, rhs_srv2win_of)
            bpwin_2 = (rhs_bpwin, rhs_bpwin_of)
            mstat = MatchStat(
                srv1in_1,
                srv1win_1,
                srv2win_1,
                bpwin_1,
                lhs_aces,
                lhs_df,
                lhs_tpw,
                srv1in_2,
                srv1win_2,
                srv2win_2,
                bpwin_2,
                rhs_aces,
                rhs_df,
                rhs_tpw,
            )
            if plr_min_id == rhs_plr_id:
                mstat.flip()
            _sex_dict[sex][tt.get_year_weeknum(tour_date)][storkey] = mstat


def match_keys_combinations(level, surface, rnd=None):
    args = []
    qual_prefix = ""
    if rnd is not None and rnd.qualification():
        qual_prefix = "q-"
    if level:
        args.append({"level": qual_prefix + str(level)})
    if surface:
        args.append({"surface": surface})
    return sorted(co.keys_combinations(args), key=len)


def fun_names():
    return (
        "service_hold",
        "receive_hold",
        "service_win",
        "receive_win",
        "break_points_saved",
        "break_points_converted",
        "first_service_in",
        "double_faults_pergame",
        "aces_pergame",
    )


def extract_results(fun_name, match):
    fun = getattr(match.stat, fun_name)
    if "_hold" in fun_name or "_pergame" in fun_name:
        return fun(match.score)
    return fun()


class Generic:
    def __init__(self):
        self.dict_from_sexfun = {}
        for sex in ("wta", "atp"):
            dirname = os.path.join(cfg_dir.stat_misc_dir(sex), "matchstat")
            for filename in fu.find_files_in_folder(
                dirname, filemask="*.txt", recursive=False
            ):
                fun_name = os.path.basename(filename).replace(".txt", "")
                dct = dict_tools.load(
                    filename,
                    createfun=lambda: defaultdict(lambda: None),
                    keyfun=co.StructKey.create_from_text,
                    valuefun=SizedValue.create_from_text,
                )
                assert bool(dct), "can not load file {} for: {} {}".format(
                    filename, sex, fun_name
                )
                self.dict_from_sexfun[(sex, fun_name)] = copy.copy(dct)

    def result(self, sex, fun_name, key):
        """тут fun_name одно из имен в fun_names() выше, key д.б. co.StructKey"""
        dct = self.dict_from_sexfun[(sex, fun_name)]
        if dct:
            return dct[key]

    def result_dict(self, sex, fun_name):
        """тут fun_name одно из имен в fun_names() выше"""
        dct = self.dict_from_sexfun[(sex, fun_name)]
        if dct:
            return copy.copy(dct)


__generic = None


def generic_result(sex, fun_name, key):
    """тут fun_name одно из имен в fun_names() выше, key д.б. co.StructKey"""
    global __generic
    if __generic is None:
        obj = Generic()
        __generic = obj
    result = __generic.result(sex, fun_name, key)
    return result if result is not None else SizedValue()


def generic_result_value(sex, fun_name, level, surface):
    if level == "qual":  # for soft level style
        level = "q-main"
    key = co.StructKey(level=level, surface=surface)
    sval = generic_result(sex, fun_name, key)
    if sval is not None:
        return sval.value


def generic_result_dict(sex, fun_name):
    """тут fun_name одно из имен в fun_names() выше"""
    global __generic
    if __generic is None:
        obj = Generic()
        __generic = obj
    return __generic.result_dict(sex, fun_name)


def generic_surface_trans_coef(sex, fun_name, surface_from, surface_to):
    """get coef according: genval_surf_from * coef = genval_surf_to
    тут fun_name одно из имен в fun_names() выше"""
    if (
        surface_from == surface_to
        or surface_from in (None, "Acrylic")
        or surface_to in (None, "Acrylic")
    ):
        return 1.0
    gen_dct = generic_result_dict(sex, fun_name)
    sval_from = gen_dct[co.StructKey(surface=surface_from)]
    sval_to = gen_dct[co.StructKey(surface=surface_to)]
    if sval_from and sval_to and abs(sval_from.value) > 0.001:
        return sval_to.value / sval_from.value
    return 1.0


class Personal(object):
    def __init__(self):
        self.dict_from_sexplrfunshort = defaultdict(lambda: None)

    @staticmethod
    def _read_dict(sex, player_id, fun_name, short):
        filename = "{}/matchstat/{}/{}/{}.txt".format(
            cfg_dir.stat_players_dir(sex),
            fun_name,
            str(history_days_personal(short)),
            str(player_id),
        )
        dct = dict_tools.load(
            filename,
            createfun=lambda: defaultdict(lambda: None),
            keyfun=co.StructKey.create_from_text,
            valuefun=SizedValue.create_from_text,
        )
        return dct

    def result(self, sex, player_id, fun_name, short, key):
        dct = self.dict_from_sexplrfunshort[(sex, player_id, fun_name, short)]
        if dct is not None:
            return dct[key]
        else:
            dct = self._read_dict(sex, player_id, fun_name, short)
            self.dict_from_sexplrfunshort[(sex, player_id, fun_name, short)] = dct
            if dct is not None:
                return dct[key]

    def result_dict(self, sex, player_id, fun_name, short):
        dct = self.dict_from_sexplrfunshort[(sex, player_id, fun_name, short)]
        if dct is not None:
            return dct
        else:
            dct = self._read_dict(sex, player_id, fun_name, short)
            self.dict_from_sexplrfunshort[(sex, player_id, fun_name, short)] = dct
            if dct is not None:
                return dct


__personal = Personal()


def personal_result(sex, player_id, fun_name, short, key):
    result = __personal.result(sex, player_id, fun_name, short, key)
    return result if result is not None else SizedValue()


def personal_result_dict(sex, player_id, fun_name, short):
    result = __personal.result_dict(sex, player_id, fun_name, short)
    return result if result is not None else {}


def personal_keyall_adjust_result(sex, player, fun_name, short, level, surface):
    if player is None:
        return SizedValue()
    sval_from_key = personal_result_dict(sex, player.ident, fun_name, short)
    return adjust_keyall_for_dict(sex, fun_name, level, surface, sval_from_key)


def adjust_keyall_for_dict(sex, fun_name, level, surface, sval_from_key):
    """0) пусть key = StructKey(level, surface)
    1) получить целев generic зн-ние trg_gen_sval для sex, fun_name, key
    2) 'выровнять' зн-я 2-х k-len эл-тов sval_from_key[ki]
       исходя из отношения generic зн-ния для sex, fun_name, ki к trg_gen_sval
       и из выровн-х зн-й получить результ 'all' как среднее зн-е (SizedValue)
    """
    trg_gen_sval = generic_result(
        sex, fun_name, co.StructKey(level=level, surface=surface)
    )
    result_sum, result_count = 0.0, 0
    for ki, sval in sval_from_key.items():
        if len(ki) != 2:
            continue
        src_gen_sval = generic_result(sex, fun_name, ki)
        sval_adjust = copy.copy(sval)
        if trg_gen_sval and src_gen_sval and sval:
            if abs(src_gen_sval.value) > co.epsilon:
                sval_adjust.value = sval.value * (
                    trg_gen_sval.value / src_gen_sval.value
                )
            else:
                sval_adjust.value = sval.value
        if sval_adjust:
            result_sum += sval_adjust.value * sval_adjust.size
            result_count += sval_adjust.size
    return (
        SizedValue(float(result_sum) / result_count, result_count)
        if result_count > 0
        else SizedValue()
    )


def history_days_personal(short):
    return 91 if short else 364


class MatchStatGenericProcessor(object):
    def __init__(self, sex, fun_name, history_days, max_rating=500):
        """fun_name is one from fun_names()"""
        self.sex = sex
        self.fun_name = fun_name
        self.min_date = tt.now_minus_history_days(history_days)
        self.histo = create_summator_histogram()
        self.softmain_histo = create_summator_histogram()
        self.max_rating = max_rating

    def process(self, tour):
        if tour.level.junior or tour.date < self.min_date:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q":
                continue
            keys = match_keys_combinations(tour.level, tour.surface, rnd)
            for m in matches:
                if (
                    m.stat is not None
                    and not m.paired()
                    and m.score.valid()
                    and m.score.sets_count(full=True) >= 2
                ):
                    self.__process_match(m, keys, tour.level, tour.surface)

    def __process_match(self, match, keys, level, surface):
        def softmain_fill(result):
            value = result.ratio if hasattr(result, "ratio") else result
            if value is not None:
                self.softmain_histo.hit((str(surface),), value)

        if match.is_ranks_both_below(self.max_rating) is False:
            result_left, result_right = extract_results(self.fun_name, match)
            self.__generic(keys, result_left)
            self.__generic(keys, result_right)
            if (
                match.rnd is not None
                and match.date is not None
                and match.date >= datetime.date(2012, 1, 1)
                and soft_level(level, match.rnd) == "main"
            ):
                softmain_fill(result_left)
                softmain_fill(result_right)

    def __generic(self, keys, result):
        if result is not None:
            value = result.ratio if hasattr(result, "ratio") else result
            if value is not None:
                self.histo.hit(keys, value)

    def reporting(self):
        log.info(
            "{} {} matchstat generic output start...".format(self.sex, self.fun_name)
        )
        filename = "{}/matchstat/{}.txt".format(
            cfg_dir.stat_misc_dir(self.sex), self.fun_name
        )
        dict_tools.dump(
            self.histo,
            filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        sm_dir = "{}/matchstat/softmain".format(cfg_dir.stat_misc_dir(self.sex))
        fu.ensure_folder(sm_dir)
        sm_filename = "{}/softmain_{}.txt".format(sm_dir, self.fun_name)
        dict_tools.dump(
            self.softmain_histo,
            sm_filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        log.info(
            "{} {} matchstat generic output finish".format(self.sex, self.fun_name)
        )


class PlayersDirsBinaryOperation(object):
    def __init__(self, sex, fun_name_1, fun_name_2, oper_fun, short_history=False):
        self.sex = sex
        self.fun_name_1 = fun_name_1
        self.fun_name_2 = fun_name_2
        self.oper_fun = oper_fun
        self.history_days = history_days_personal(short=short_history)
        self.result_files_count = 0

    def oper_result_name(self):
        return co.binary_oper_result_name(
            self.oper_fun, self.fun_name_1, self.fun_name_2
        )

    def __run_generic(self):
        dirname = "{}/matchstat".format(cfg_dir.stat_misc_dir(self.sex))
        filename_1 = os.path.join(dirname, self.fun_name_1 + ".txt")
        assert os.path.isfile(filename_1), "can not find file '{}'".format(filename_1)
        filename_2 = os.path.join(dirname, self.fun_name_2 + ".txt")
        assert os.path.isfile(filename_2), "can not find file '{}'".format(filename_2)
        dct_1 = dict_tools.load(
            filename_1,
            keyfun=co.StructKey.create_from_text,
            valuefun=SizedValue.create_from_text,
        )
        dct_2 = dict_tools.load(
            filename_2,
            keyfun=co.StructKey.create_from_text,
            valuefun=SizedValue.create_from_text,
        )
        dct_result = dict_tools.binary_operation(self.oper_fun, dct_1, dct_2)
        if dct_result:
            dict_tools.dump(
                dct_result,
                os.path.join(dirname, self.oper_result_name() + ".txt"),
                keyfun=str,
                valuefun=str,
            )

    def run(self):
        self.result_files_count = 0
        self.__run_generic()
        result_dirname = self.__players_dirname(self.oper_result_name())
        for filename_1 in fu.find_files_in_folder(
            self.__players_dirname(self.fun_name_1), filemask="*.txt", recursive=False
        ):
            dct_1 = dict_tools.load(
                filename_1,
                keyfun=co.StructKey.create_from_text,
                valuefun=SizedValue.create_from_text,
            )
            if not dct_1:
                continue
            plr_basename = os.path.basename(filename_1)
            filename_2 = os.path.join(
                self.__players_dirname(self.fun_name_2), plr_basename
            )
            dct_2 = dict_tools.load(
                filename_2,
                keyfun=co.StructKey.create_from_text,
                valuefun=SizedValue.create_from_text,
            )
            if not dct_2:
                continue
            dct_result = dict_tools.binary_operation(self.oper_fun, dct_1, dct_2)
            dict_tools.dump(
                dct_result,
                os.path.join(result_dirname, plr_basename),
                keyfun=str,
                valuefun=str,
            )
            if dct_result:
                self.result_files_count += 1
        log.info(
            "finish {} {} result_files_count: {}".format(
                self.sex, self.oper_result_name(), self.result_files_count
            )
        )
        assert self.result_files_count > 0, "zero result_files_count"

    def __players_dirname(self, fun_name):
        return "{}/matchstat/{}/{}".format(
            cfg_dir.stat_players_dir(self.sex), fun_name, self.history_days
        )


class MatchStatPersonalProcessor(object):
    def __init__(self, sex, fun_name, short_history):
        """fun_name is one from fun_names()"""
        self.sex = sex
        self.fun_name = fun_name
        self.history_days = history_days_personal(short=short_history)
        self.min_date = tt.now_minus_history_days(self.history_days)
        self.histo_from_player = defaultdict(create_summator_histogram)
        self.actual_players = extplayers.get_players(sex)

    def process(self, tour):
        if tour.level.junior or tour.date < self.min_date:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q":
                continue
            keys = match_keys_combinations(tour.level, tour.surface, rnd)
            for m in matches:
                if (
                    m.stat is not None
                    and not m.paired()
                    and m.score.sets_count(full=True) >= 2
                ):
                    self.__process_match(m, keys)

    def __process_match(self, match, keys):
        histo_left, histo_right = None, None
        if match.first_player in self.actual_players:
            histo_left = self.histo_from_player[match.first_player.ident]
        if match.second_player in self.actual_players:
            histo_right = self.histo_from_player[match.second_player.ident]

        if histo_left is not None or histo_right is not None:
            result_left, result_right = extract_results(self.fun_name, match)
            self.__personal(keys, histo_left, result_left)
            self.__personal(keys, histo_right, result_right)

    @staticmethod
    def __personal(keys, histo, result):
        if result is not None and histo is not None:
            value = result.ratio if hasattr(result, "ratio") else result
            if value is not None:
                histo.hit(keys, value)

    def reporting(self):
        log.info(
            "{} {} matchstat personal output start...".format(self.sex, self.fun_name)
        )
        dirname = "{}/matchstat/{}/{}".format(
            cfg_dir.stat_players_dir(self.sex), self.fun_name, self.history_days
        )
        assert os.path.isdir(dirname), "dir not found: " + dirname
        fu.remove_files_in_folder(dirname)
        for playerid, histo in self.histo_from_player.items():
            filename = os.path.join(dirname, str(playerid) + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                alignvalues=False,
            )
        log.info(
            "{} {} matchstat personal output finish".format(self.sex, self.fun_name)
        )


def hold_estimator(sex):
    return MatchstatEstimator(
        sex,
        srv_fun_name="service_hold",
        rcv_fun_name="receive_hold",
        short=True,
        srv_min_diff=0.003,
        rcv_min_diff=0.003,
        min_size=9,
    )


def win_estimator(sex):
    return MatchstatEstimator(
        sex,
        srv_fun_name="service_win",
        rcv_fun_name="receive_win",
        short=True,
        srv_min_diff=0.003,
        rcv_min_diff=0.003,
        min_size=9,
    )


def hold_cmp_estimator(sex):
    return GoodVsBadEstimator(
        sex,
        srv_fun_name="service_hold",
        rcv_fun_name="receive_hold",
        short=True,
        min_size=9,
    )


def win_cmp_estimator(sex):
    return GoodVsBadEstimator(
        sex,
        srv_fun_name="service_win",
        rcv_fun_name="receive_win",
        short=True,
        min_size=9,
    )


class ValueTager(object):
    def __init__(self, under, neutral_begin, neutral_end, above):
        self.under = under
        self.neutral_begin = neutral_begin
        self.neutral_end = neutral_end
        self.above = above

    def get_tag(self, value):
        if value < self.under:
            return "very under"
        if self.under <= value < self.neutral_begin:
            return "under"
        if self.neutral_begin <= value < self.neutral_end:
            return ""
        if self.neutral_end <= value < self.above:
            return "above"
        if self.above <= value:
            return "very above"


def make_value_tager(sex, fun_name, level):
    if fun_name == "service_hold":
        if sex == "atp" and level == "main":
            return ValueTager(
                under=0.64, neutral_begin=0.75, neutral_end=0.80, above=0.85
            )
        elif sex == "atp" and level == "chal":
            return ValueTager(
                under=0.61, neutral_begin=0.72, neutral_end=0.76, above=0.81
            )
        elif sex == "wta" and level == "main":
            return ValueTager(
                under=0.50, neutral_begin=0.62, neutral_end=0.68, above=0.74
            )
        elif sex == "wta" and level == "chal":
            return ValueTager(
                under=0.48, neutral_begin=0.59, neutral_end=0.64, above=0.72
            )
    elif fun_name == "receive_hold":
        if sex == "atp" and level == "main":
            return ValueTager(
                under=0.10, neutral_begin=0.122, neutral_end=0.285, above=0.35
            )
        elif sex == "atp" and level == "chal":
            return ValueTager(
                under=0.10, neutral_begin=0.150, neutral_end=0.31, above=0.36
            )
        elif sex == "wta" and level == "main":
            return ValueTager(
                under=0.15, neutral_begin=0.33, neutral_end=0.37, above=0.44
            )
        elif sex == "wta" and level == "chal":
            return ValueTager(
                under=0.16, neutral_begin=0.33, neutral_end=0.38, above=0.45
            )


class GoodVsBadEstimator(object):
    def __init__(self, sex, srv_fun_name, rcv_fun_name, short, min_size):
        self.sex = sex
        self.srv_fun_name = srv_fun_name
        self.rcv_fun_name = rcv_fun_name
        self.short = short
        self.min_size = min_size  # min matches count

    def find_picks(self, today_match):
        result = self.find_picks_impl(
            today_match.offer,
            today_match,
            today_match.sex,
            today_match.level,
            today_match.surface,
            today_match.rnd,
        )
        return [] if result is None else result

    def find_picks_impl(self, offer, match, sex, level, surface, rnd):
        if (
            sex != self.sex
            or not offer
            or not offer.win_coefs
            or level == "future"
            or (rnd and rnd.qualification())
        ):
            return
        dominate_id, dominate_txt = self.__dominate_choice(match, level, surface)
        if dominate_id:
            if match.first_player.ident == dominate_id:
                coef = offer.win_coefs.first_coef
            else:
                coef = offer.win_coefs.second_coef
            return [
                PickWin(
                    get_betcity_company(),
                    coef,
                    dominate_id,
                    explain="matchstat " + dominate_txt,
                )
            ]
        return []

    def __dominate_choice(self, match, level, surface):
        dominate_id_1, dominate_txt_1 = self.__dominate(
            match.first_player, match.second_player, level, surface
        )
        dominate_id_2, dominate_txt_2 = self.__dominate(
            match.second_player, match.first_player, level, surface
        )
        if not dominate_id_1 and not dominate_id_2:
            return None, None  # никто не доминирует
        if dominate_id_1 and dominate_id_2 and dominate_id_1 != dominate_id_2:
            return None, None  # оба доминируют в чем-то

        dominate_id = dominate_id_1 if dominate_id_1 else dominate_id_2
        if dominate_txt_1 and dominate_txt_2:
            dominate_txt = dominate_txt_1 + "\n  AND " + dominate_txt_2
        else:
            dominate_txt = dominate_txt_1 if dominate_txt_1 else dominate_txt_2
        return dominate_id, dominate_txt

    def __dominate(self, left_player, right_player, level, surface):
        """проверяем ситуацию: left_player подает, right_player принимает.
        если есть доминация вернем (dominate_id, dominate_txt) иначе: (None, None)
        """
        if not left_player or not right_player:
            return None, None
        left_sval = personal_keyall_adjust_result(
            self.sex, left_player, self.srv_fun_name, self.short, level, surface
        )
        if left_sval.size < self.min_size:
            return None, None
        value_tager = make_value_tager(self.sex, self.srv_fun_name, level)
        left_tag = value_tager.get_tag(left_sval.value) if value_tager else ""
        if not left_tag:
            return None, None

        right_sval = personal_keyall_adjust_result(
            self.sex, right_player, self.rcv_fun_name, self.short, level, surface
        )
        if right_sval.size < self.min_size:
            return None, None
        value_tager = make_value_tager(self.sex, self.rcv_fun_name, level)
        right_tag = value_tager.get_tag(right_sval.value) if value_tager else ""
        if not right_tag:
            return None, None

        if "under" in left_tag and "above" in right_tag:
            txt = "{} with {}, opponent {} with {}".format(
                right_tag, self.rcv_fun_name, left_tag, self.srv_fun_name
            )
            return right_player.ident, txt
        return None, None


class MatchstatEstimator(object):
    def __init__(
        self,
        sex,
        srv_fun_name,
        rcv_fun_name,
        short,
        srv_min_diff,
        rcv_min_diff,
        min_size,
    ):
        self.sex = sex
        self.srv_fun_name = srv_fun_name
        self.rcv_fun_name = rcv_fun_name
        self.short = short
        self.srv_min_diff = srv_min_diff
        self.rcv_min_diff = rcv_min_diff
        self.min_size = min_size  # min matches count

    def find_picks(self, today_match):
        result = self.find_picks_impl(today_match.offer, today_match, today_match.sex)
        return [] if result is None else result

    def find_picks_impl(self, offer, match, sex):
        if sex != self.sex or not offer or not offer.win_coefs:
            return
        left_chance, right_chance = offer.win_coefs.chances()
        srv_cmp = self.__compare(
            self.srv_fun_name,
            self.srv_min_diff,
            match.first_player,
            match.second_player,
        )
        rcv_cmp = self.__compare(
            self.rcv_fun_name,
            self.rcv_min_diff,
            match.first_player,
            match.second_player,
        )
        if srv_cmp == co.GT and rcv_cmp == co.GT and left_chance <= 0.5:
            return [
                PickWin(
                    get_betcity_company(),
                    offer.win_coefs.first_coef,
                    match.first_player.ident,
                    explain="matchstat " + self.srv_fun_name,
                )
            ]
        elif srv_cmp == co.LT and rcv_cmp == co.LT and right_chance <= 0.5:
            return [
                PickWin(
                    get_betcity_company(),
                    offer.win_coefs.second_coef,
                    match.second_player.ident,
                    explain="matchstat " + self.srv_fun_name,
                )
            ]
        return []

    def __compare(self, fun_name, min_diff, left_player, right_player):
        if not left_player or not right_player:
            return
        key = co.StructKey()
        left_sval = personal_result(
            self.sex, left_player.ident, fun_name, self.short, key
        )
        right_sval = personal_result(
            self.sex, right_player.ident, fun_name, self.short, key
        )
        if left_sval.size < self.min_size or right_sval.size < self.min_size:
            return
        return co.value_compare(left_sval.value, right_sval.value, min_diff)


# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> visual str, where key is left-most entity
ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")


def make_player_columninfo(
    sex, player_id, fun_name, short, title, number, level, surface, value_default="-"
):
    def value_function(key):
        if key == co.StructKey():
            # prepare adjusted (to used target level, surface) value
            sval_from_key = personal_result_dict(sex, player_id, fun_name, short)
            result = adjust_keyall_for_dict(
                sex, fun_name, level, surface, sval_from_key
            )
        else:
            result = personal_result(sex, player_id, fun_name, short, key)
        return co.formated(result.value, result.size, round_digits=3).strip()

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if player_id is not None else empty_function,
        value_default=value_default,
    )


def make_generic_columninfo(sex, fun_name, title, number, value_default="-"):
    def value_function(key):
        result = generic_result(sex, fun_name, key)
        return co.float_to_str(result.value, round_digits=3, none_str=value_default)

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function,
        value_default=value_default,
    )


def make_mix_players_columninfo(
    sex,
    fst_player_id,
    snd_player_id,
    fst_fun_name,
    snd_fun_name,
    short,
    title,
    number,
    level,
    surface,
    value_default="-",
):
    def value_function(key):
        if key == co.StructKey():
            # prepare adjusted (to used target level, surface) value
            fst_sval_from_key = personal_result_dict(
                sex, fst_player_id, fst_fun_name, short
            )
            fst_result = adjust_keyall_for_dict(
                sex, fst_fun_name, level, surface, fst_sval_from_key
            )
            snd_sval_from_key = personal_result_dict(
                sex, snd_player_id, snd_fun_name, short
            )
            snd_result = adjust_keyall_for_dict(
                sex, snd_fun_name, level, surface, snd_sval_from_key
            )
        else:
            fst_result = personal_result(sex, fst_player_id, fst_fun_name, short, key)
            snd_result = personal_result(sex, snd_player_id, snd_fun_name, short, key)
        fst_chance, snd_chance = co.twoside_values(fst_result, snd_result)
        return co.formated(
            fst_chance, fst_result.size + snd_result.size, round_digits=3
        ).strip()

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if fst_player_id and snd_player_id else empty_function,
        value_default=value_default,
    )


class PageBuilder(object):
    def __init__(self):
        self.columns = []
        self.keys = []

    def create_widgets(self, obj):
        rownum = 1
        obj.estimate_btn = tkinter.ttk.Button(
            obj, text="Estimate", command=obj.estimate
        )
        obj.estimate_btn.grid(row=rownum, column=0)

        rownum += 1
        RowBuilder.create_title_row(obj, rownum, self.columns)

        rownum += 1
        for key_idx, key in enumerate(self.keys):
            RowBuilder.create_row(obj, rownum + key_idx, key, self.columns)

    def clear(self, obj):
        rownum = 3
        for key_idx in range(len(self.keys)):
            RowBuilder.clear_row(obj, rownum + key_idx, self.columns)

    def update(self, obj):
        rownum = 3
        for key_idx, key in enumerate(self.keys):
            RowBuilder.update_row(obj, rownum + key_idx, key, self.columns)


class PageBuilderLGR(PageBuilder):
    """шаблон для закладки c тремя колонками: left_player, generic, right_player"""

    def __init__(
        self, sex, level, surface, rnd, short, left_player, right_player, fun_name
    ):
        super(PageBuilderLGR, self).__init__()
        self.set_data(
            sex, level, surface, rnd, short, left_player, right_player, fun_name
        )

    def set_data(
        self, sex, level, surface, rnd, short, left_player, right_player, fun_name
    ):
        self.keys = match_keys_combinations(level, surface, rnd)
        self.columns = [
            make_player_columninfo(
                sex,
                left_player.ident if left_player else None,
                fun_name,
                short,
                "left",
                1,
                level,
                surface,
            ),
            make_generic_columninfo(sex, fun_name, title="avg", number=2),
            make_player_columninfo(
                sex,
                right_player.ident if right_player else None,
                fun_name,
                short,
                "right",
                3,
                level,
                surface,
            ),
        ]


class PageBuilderMutual(PageBuilder):
    """шаблон для закладки c одной смешанной колонкой для left_player, right_player"""

    def __init__(
        self,
        sex,
        level,
        surface,
        rnd,
        short,
        fst_player,
        snd_player,
        fst_fun_name,
        snd_fun_name,
    ):
        super(PageBuilderMutual, self).__init__()
        self.set_data(
            sex,
            level,
            surface,
            rnd,
            short,
            fst_player,
            snd_player,
            fst_fun_name,
            snd_fun_name,
        )

    def set_data(
        self,
        sex,
        level,
        surface,
        rnd,
        short,
        fst_player,
        snd_player,
        fst_fun_name,
        snd_fun_name,
    ):
        self.keys = match_keys_combinations(level, surface, rnd)
        self.columns = [
            make_mix_players_columninfo(
                sex,
                fst_player.ident if fst_player is not None else None,
                snd_player.ident if snd_player is not None else None,
                fst_fun_name,
                snd_fun_name,
                short,
                "fst_srv",
                1,
                level,
                surface,
            ),
            make_mix_players_columninfo(
                sex,
                snd_player.ident if snd_player is not None else None,
                fst_player.ident if fst_player is not None else None,
                fst_fun_name,
                snd_fun_name,
                short,
                "snd_srv",
                2,
                level,
                surface,
            ),
        ]


class RowBuilder(object):
    @staticmethod
    def widget_name(rownum, columninfo):
        return "var_" + str(rownum) + "_" + str(columninfo.number)

    @staticmethod
    def create_title_row(obj, rownum, columninfo_list):
        """create most up (first) widgets in obj (page) from columninfo_list[i].title
        first slot (above keys column) is skiped and leaves as empty (widget absent).
        """
        column_start = 1  # numerate from 0, skip absent key slot
        for column_idx, columninfo in enumerate(columninfo_list):
            widget_name = RowBuilder.widget_name(rownum, columninfo)
            setattr(obj, widget_name, tkinter.ttk.Label(obj, text=columninfo.title))
            getattr(obj, widget_name).grid(row=rownum, column=column_idx + column_start)

    @staticmethod
    def create_row(obj, rownum, key, columninfo_list):
        """создает виджеты в obj"""
        column_start = 0
        keyvarname = "varkey_" + str(rownum)
        setattr(obj, keyvarname, tkinter.ttk.Label(obj, text=str(key)))
        getattr(obj, keyvarname).grid(row=rownum, column=column_start)
        column_start += 1

        for column_idx, columninfo in enumerate(columninfo_list):
            widget_name = RowBuilder.widget_name(rownum, columninfo)
            setattr(
                obj, widget_name, tkinter.ttk.Label(obj, text=columninfo.value_default)
            )
            getattr(obj, widget_name).grid(row=rownum, column=column_idx + column_start)

    @staticmethod
    def clear_row(obj, rownum, columninfo_list):
        for columninfo in columninfo_list:
            widget_name = RowBuilder.widget_name(rownum, columninfo)
            getattr(obj, widget_name)["text"] = columninfo.value_default

    @staticmethod
    def update_row(obj, rownum, key, columninfo_list):
        # key update:
        keyvarname = "varkey_" + str(rownum)
        getattr(obj, keyvarname)["text"] = str(key)
        # values update:
        for columninfo in columninfo_list:
            widget_name = RowBuilder.widget_name(rownum, columninfo)
            getattr(obj, widget_name)["text"] = columninfo.value_fun(key)


HIST_WEEKS_NUM = 35  # pandemic correction +5 at begin 2021
