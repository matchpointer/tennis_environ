# -*- coding: utf-8 -*-
import sys
import unittest
import datetime
from collections import defaultdict, namedtuple
import argparse
from contextlib import closing

import log
import cfg_dir
import common as co
import file_utils as fu
import dict_tools
import oncourt_players
import report_line as rl
import stat_cont as st
from score import Score
import tennis
import tennis_time as tt
import dba
import feature
from detailed_score_dbsa import open_db


ASPECT_UNDER = "aftertie_under_perf"
ASPECT_PRESS = "aftertie_press_perf"
DIRNAME = "aftertie_perf_stat"

MIN_SIZE_DEFAULT = 8


class PlayerData:
    def __init__(self, under_sum=None, press_sum=None):
        self.under_sum = under_sum if under_sum is not None else st.Sumator()
        self.press_sum = press_sum if press_sum is not None else st.Sumator()

    def put(self, perf_value, is_press):
        if is_press:
            self.press_sum.hit(perf_value)
        else:
            self.under_sum.hit(perf_value)

    def __add__(self, other):
        return PlayerData(
            self.under_sum + other.under_sum, self.press_sum + other.press_sum
        )

    def __iadd__(self, other):
        self.under_sum += other.under_sum
        self.press_sum += other.press_sum
        return self

    def ok_min_size(self, aspects=(ASPECT_UNDER, ASPECT_PRESS)):
        for aspect in aspects:
            if aspect == ASPECT_UNDER and self.under_sum.size < MIN_SIZE_DEFAULT:
                return False
            if aspect == ASPECT_PRESS and self.press_sum.size < MIN_SIZE_DEFAULT:
                return False
        return True

    def sized_value(self, aspect):
        if aspect == ASPECT_UNDER:
            return rl.SizedValue(
                value=self.under_sum.average(), size=self.under_sum.size
            )
        if aspect == ASPECT_PRESS:
            return rl.SizedValue(
                value=self.press_sum.average(), size=self.press_sum.size
            )


# sex -> ASC ordered dict{(year, weeknum) -> dict{pid -> PlayerData}}
data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(PlayerData)))


SCR_ABS_PERF_DICT = {
    # min scr games -> abs perf value
    0: 1.0,
    1: 0.95,
    2: 0.7,
    3: 0.5,
    4: 0.2,
    5: 0.2,
    6: 0.1,
    7: 0.1,
}

SCR_SUPERTIE_ABS_PERF_DICT = {
    # min scr games -> abs perf value
    0: 1.0,
    1: 0.95,
    2: 0.85,
    3: 0.75,
    4: 0.7,
    5: 0.5,
    6: 0.35,
    7: 0.2,
    8: 0.1,
}


def _abs_perf_value(set_score, isdec):
    if (max(set_score) - min(set_score)) <= 2:
        return 0.1
    if isdec and max(set_score) == 10:
        return SCR_SUPERTIE_ABS_PERF_DICT.get(min(set_score), 0.0)
    return SCR_ABS_PERF_DICT.get(min(set_score), 0.0)


def add_features(features, sex, min_date, max_date, fst_id, snd_id):
    def work_pair(fst_sum, snd_sum, aspect):
        fst_val = fst_sum.value if fst_sum.size >= MIN_SIZE_DEFAULT else None
        snd_val = snd_sum.value if snd_sum.size >= MIN_SIZE_DEFAULT else None
        feature.add_pair(features, aspect, fst_value=fst_val, snd_value=snd_val)

    fst_pdata = get_player_data(sex, fst_id, min_date, max_date)
    snd_pdata = get_player_data(sex, snd_id, min_date, max_date)
    work_pair(fst_pdata.under_sum, snd_pdata.under_sum, ASPECT_UNDER)
    work_pair(fst_pdata.press_sum, snd_pdata.press_sum, ASPECT_PRESS)


def add_read_features_pair(features, sex, fst_id, snd_id, aspect):
    sv1 = _read_player_sized_value(sex, fst_id, aspect)
    sv2 = _read_player_sized_value(sex, snd_id, aspect)
    val1 = None if (sv1 is None or sv1.size < MIN_SIZE_DEFAULT) else sv1.value
    val2 = None if (sv2 is None or sv2.size < MIN_SIZE_DEFAULT) else sv2.value
    f1, f2 = feature.make_pair2(aspect, fst_value=val1, snd_value=val2)
    features.append(f1)
    features.append(f2)


def get_player_data(sex, player_id, min_date, max_date):
    min_ywn = tt.get_year_weeknum(min_date) if min_date else None
    max_ywn = tt.get_year_weeknum(max_date) if max_date else None
    res = PlayerData()
    dct = data_dict[sex]
    for ywn, plr_dct in dct.items():
        if min_ywn is not None and ywn < min_ywn:
            continue
        if max_ywn is not None and ywn >= max_ywn:
            break
        res += plr_dct[player_id]
    return res


def _read_player_sized_value(sex, player_id, aspect):
    cache_dct = _read_player_sized_value.cached_from_sex[(sex, aspect)]
    if cache_dct is not None:
        return cache_dct.get(player_id)

    dct = _read_players_dict(sex, aspect)
    _read_player_sized_value.cached_from_sex[(sex, aspect)] = dct
    return dct.get(player_id)


def _read_players_dict(sex, aspect):
    filename = players_filename(sex, aspect, suffix="_id")
    dct = dict_tools.load(
        filename,
        createfun=lambda: defaultdict(rl.SizedValue),
        keyfun=int,
        valuefun=rl.SizedValue.create_from_text,
    )
    return dct


_read_player_sized_value.cached_from_sex = {
    ("wta", ASPECT_UNDER): None,
    ("wta", ASPECT_PRESS): None,
    ("atp", ASPECT_UNDER): None,
    ("atp", ASPECT_PRESS): None,
}


# class ReadFilesTest(unittest.TestCase):
#     def test_read_players_dict(self):
#         dct = _read_players_dict(sex='wta')
#         self.assertTrue(len(dct) > 500)
#         if dct:
#             vals = [sv.value for sv in dct.values()]
#             self.assertTrue(all([(0 <= v <= 1) for v in vals]))


def initialize(sex, dbdet, min_date, max_date):
    # max_date = None  # datetime.date(2010, 1, 1)
    if not dba.initialized():
        dba.open_connect()
    # min_date = None  # tt.now_minus_history_days(get_history_days(sex))

    _initialize_results_sex(sex, dbdet, min_date=min_date, max_date=max_date)


def _initialize_results_sex(sex, dbdet, min_date=None, max_date=None):
    sql = """select tours.DATE_T, tours.NAME_T, tours.ID_T,  
                   games.ID_R_G, games.DATE_G, games.RESULT_G, games.ID1_G, games.ID2_G
             from Tours_{0} AS tours, games_{0} AS games, Players_{0} AS fst_plr
             where games.ID_T_G = tours.ID_T 
               and games.ID1_G = fst_plr.ID_P
               and (tours.NAME_T Not Like '%juniors%')
               and (fst_plr.NAME_P Not Like '%/%') """.format(
        sex
    )
    sql += dba.sql_dates_condition(min_date, max_date)
    sql += " order by tours.DATE_T;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (
            tour_dt,
            tour_name,
            tour_id,
            rnd_id,
            match_dt,
            score_txt,
            fst_id,
            snd_id,
        ) in cursor.execute(sql):
            if not score_txt:
                continue
            tour_date = tour_dt.date() if tour_dt else None
            match_date = match_dt.date() if match_dt else None
            date = match_date if match_date is not None else tour_date
            if date is None:
                raise co.TennisScoreError("none date {}".format(tour_name))
            score = Score(score_txt)
            sets_count = score.sets_count(full=True)
            if score.retired and sets_count < 2:
                continue
            rnd = tennis.Round.from_oncourt_id(rnd_id)
            fst_pdata, snd_pdata = _get_match_data(
                sex, dbdet, tour_id, rnd, fst_id, snd_id, score, sets_count
            )
            if fst_pdata or snd_pdata:
                past_monday = tt.past_monday_date(date)
                ywn = tt.get_year_weeknum(past_monday)
                if fst_pdata:
                    data_dict[sex][ywn][fst_id] += fst_pdata
                if snd_pdata:
                    data_dict[sex][ywn][snd_id] += snd_pdata


def _get_match_data(sex, dbdet, tour_id, rnd, fst_id, snd_id, score, sets_count):
    def is_nodecided_tie():
        return any(
            [
                (s in ((6, 7), (7, 6)))
                for n, s in enumerate(score, start=1)
                if n < sets_count
            ]
        )

    def get_detailed_affect(view_side):
        def view_side_loss_game(game_num):
            det_game = next_set_items[game_num - 1][1]
            game_win_side = co.side(det_game.left_wingame)
            return game_win_side.is_oppose(view_side)

        next_set_items = list(det_score.set_items(setnum=setnum + 1))
        if len(next_set_items) < 6:
            return None
        loss_g1 = view_side_loss_game(game_num=1)
        loss_g2 = view_side_loss_game(game_num=2)
        loss_g3 = view_side_loss_game(game_num=3)
        loss_g4 = view_side_loss_game(game_num=4)
        loss_g5 = view_side_loss_game(game_num=5)
        loss_g6 = view_side_loss_game(game_num=6)
        if loss_g1 and loss_g2 and loss_g3 and loss_g4 and loss_g5 and loss_g6:
            return 1.0  # 0:6
        if loss_g1 and loss_g2 and loss_g3 and loss_g4 and loss_g5:
            return 0.9  # 0:5
        if loss_g1 and loss_g2 and loss_g3 and loss_g4:
            return 0.8  # 0:4
        if sum((loss_g1, loss_g2, loss_g3, loss_g4, loss_g5, loss_g6)) >= 5:
            return 0.8  # 1:5 most probably
        if len(next_set_items) >= 7:
            loss_g7 = view_side_loss_game(game_num=7)
            if (
                sum((loss_g1, loss_g2, loss_g3, loss_g4, loss_g5, loss_g6, loss_g7))
                == 5
            ):
                return 0.7  # 2:5
        if sum((loss_g1, loss_g2, loss_g3, loss_g4, loss_g5)) >= 4:
            return 0.7  # 1:4 most probably
        if loss_g1 and loss_g2 and loss_g3:
            return 0.65  # 0:3
        if loss_g1 and loss_g2:
            return 0.4  # 0:2
        if sum((loss_g1, loss_g2, loss_g3, loss_g4)) >= 3:
            return 0.4  # 1:3
        if sum((loss_g1, loss_g2, loss_g3, loss_g4, loss_g5, loss_g6)) >= 4:
            return 0.4  # 2:4 most probably

    def get_left_perfomance():
        """:returns (left player perfomance value, is_left_press)"""

        def make_result(abs_perf_val):
            if is_left_nextwin:
                return abs_perf_val, is_left_press
            else:
                return -abs_perf_val, is_left_press

        is_left_press = set_scr[0] > set_scr[1]
        next_set_scr = score[setidx + 1]
        abs_perf_val_by_scr = _abs_perf_value(next_set_scr, isdec)
        is_left_nextwin = next_set_scr[0] > next_set_scr[1]
        if min(next_set_scr) in (0, 1) or det_score is None:
            return make_result(abs_perf_val_by_scr)
        nextwin_side = co.side(is_left_nextwin)
        perf_val = get_detailed_affect(view_side=nextwin_side.fliped())
        if perf_val is None:
            return make_result(abs_perf_val_by_scr)
        return make_result(max(perf_val, abs_perf_val_by_scr))

    if not is_nodecided_tie():
        return None, None
    det_score = dbdet.get_detailed_score(tour_id, rnd, fst_id, snd_id)
    decset_num = 5 if score.best_of_five() else 3
    fst_pdata, snd_pdata = PlayerData(), PlayerData()
    for setidx in range(sets_count - 1):
        setnum = setidx + 1
        set_scr = score[setidx]
        if set_scr not in ((6, 7), (7, 6)):
            continue
        isdec = setnum == decset_num
        fst_perf_val, is_fst_press = get_left_perfomance()
        fst_pdata.put(fst_perf_val, is_fst_press)
        snd_pdata.put(-fst_perf_val, not is_fst_press)
    return fst_pdata, snd_pdata


# --------------------- work with data files ----------------------------


def get_history_days(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365.25 * 7)
    elif sex == "wta":
        return int(365.25 * 7)
    raise co.TennisError("bad sex {}".format(sex))


PlayerStat = namedtuple("PlayerStat", "player data")


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex, min_date):
        self.sex = sex
        self.min_date = min_date
        self.actual_players = oncourt_players.players(sex)
        assert len(self.actual_players) > 0, "oncourt_players not inited"
        self.plr_stat_list = []

    def process_all(self):
        self.plr_stat_list = []
        self._build_players_list()
        self._output_avg_generic_value()
        self._output_players_values()

    def _build_players_list(self):
        self.plr_stat_list = []
        for plr in self.actual_players:
            plr_stat = PlayerStat(
                player=plr,
                data=get_player_data(self.sex, plr.ident, self.min_date, max_date=None),
            )
            if not plr_stat.data.ok_min_size(
                aspects=(ASPECT_PRESS,)
            ) and not plr_stat.data.ok_min_size(aspects=(ASPECT_UNDER,)):
                continue
            self.plr_stat_list.append(plr_stat)

    def _sort_players_data(self, aspect):
        self.plr_stat_list.sort(
            key=lambda i: i.data.sized_value(aspect).value, reverse=True
        )

    def _output_players_values(self):
        fu.ensure_folder(players_dirname(self.sex))

        for aspect in (ASPECT_PRESS, ASPECT_UNDER):
            filename = players_filename(self.sex, aspect)
            # self._sort_players_data(aspect)
            with open(filename, "w") as fh:
                for plr_stat in self.plr_stat_list:
                    sv = plr_stat.data.sized_value(aspect)
                    if sv.size >= MIN_SIZE_DEFAULT:
                        fh.write("{}__{}\n".format(plr_stat.player, sv))

            filename = players_filename(self.sex, aspect, suffix="_id")
            with open(filename, "w") as fh:
                for plr_stat in self.plr_stat_list:
                    sv = plr_stat.data.sized_value(aspect)
                    if sv.size >= MIN_SIZE_DEFAULT:
                        fh.write(
                            "{}__{:.4f} ({})\n".format(
                                plr_stat.player.ident, sv.value, sv.size
                            )
                        )

    def _output_avg_generic_value(self):
        fu.ensure_folder(generic_dirname(self.sex))
        for aspect in (ASPECT_PRESS, ASPECT_UNDER):
            avg_val = self._get_avg_value(aspect)
            filename = generic_filename(self.sex, aspect)
            with open(filename, "w") as fh:
                fh.write("avg__{:.4f}\n".format(avg_val))

    def _get_avg_value(self, aspect):
        res = st.Sumator()
        for plr_stat in self.plr_stat_list:
            sv = plr_stat.data.sized_value(aspect)
            if sv.size >= MIN_SIZE_DEFAULT:
                res.hit(sv.value)
        return res.average()


def generic_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_misc_dir(sex), DIRNAME)


def generic_filename(sex, aspect):
    return "{}/{}.txt".format(generic_dirname(sex), aspect)


def players_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_players_dir(sex), DIRNAME)


def players_filename(sex, aspect, suffix=""):
    dirname = players_dirname(sex)
    return "{}/players_{}{}.txt".format(dirname, aspect, suffix)


def read_generic_value(sex, aspect):
    filename = generic_filename(sex, aspect)
    with open(filename, "r") as fh:
        line = fh.readline().strip()
        head, valtxt = line.split("__")
        return float(valtxt)


def do_stat():
    try:
        msg = "sex: {}".format(args.sex)
        log.info(__file__ + " started {}".format(msg))
        dba.open_connect()
        oncourt_players.initialize(yearsnum=7)
        min_date = tt.now_minus_history_days(get_history_days(args.sex))
        max_date = tt.future_monday_date(datetime.date.today())
        dbdet.query_matches(min_date=min_date)
        initialize(args.sex, dbdet=dbdet, min_date=min_date, max_date=max_date)
        maker = StatMaker(args.sex, min_date=min_date)
        maker.process_all()
        log.info(__file__ + " done {}".format(ASPECT_UNDER))

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


# # -------------- for extern use
# avg_affect = {
#     'wta': read_generic_value('wta'),
#     'atp': read_generic_value('atp'),
# }
#
# above_avg_high_ratio = {
#     'wta': 0.34,
#     'atp': 0.28,
# }
#
# min_size = {
#     'wta': 10,
#     'atp': 14,
# }
#
#
# def is_player_affect_high(sex, player_id):
#     sv = read_sized_value_affect_ratio(sex, player_id)
#     if sv.size >= min_size[sex]:
#         return sv.value > high_affect_threshold(sex)
#
#
# def is_player_affect_above_avg(sex, player_id):
#     sv = read_sized_value_affect_ratio(sex, player_id)
#     if sv.size >= min_size[sex]:
#         return sv.value > avg_affect[sex]
#
#
# def high_affect_threshold(sex):
#     avg_val = avg_affect[sex]
#     return avg_val + avg_val * above_avg_high_ratio[sex]
#
#
# # ------------------------------ GUI ----------------------------------------
# # here value_fun is function(key) -> str, where key is left-most entity (aspect)
# ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")
#
#
# def make_player_columninfo(sex, player_id, title, number, value_default='-'):
#     def value_function(key):
#         def get_text_val(sized_value):
#             if not sized_value:
#                 return value_default
#             return co.formated(sized_value.value, sized_value.size, round_digits=2).strip()
#
#         sval = read_sized_value_affect_ratio(sex, player_id)
#         return get_text_val(sval)
#
#     def empty_function(key):
#         return value_default
#
#     return ColumnInfo(
#         title=title,  number=number,
#         value_fun=value_function if player_id else empty_function,
#         value_default=value_default
#     )
#
#
# def make_generic_columninfo(sex, title, number, value_default='-'):
#     def value_function(key):
#         result = read_generic_value(sex)
#         if result is None:
#             return value_default
#         return "{0:.2f}".format(result)
#
#     return ColumnInfo(title=title, number=number, value_fun=value_function,
#                       value_default=value_default)
#
#
# class PageBuilderLGR(matchstat.PageBuilder):
#     """ шаблон для закладки c тремя колонками: left_player, generic, right_player """
#     def __init__(self, sex, left_player, right_player):
#         super(PageBuilderLGR, self).__init__()
#         self.keys.append(ASPECT_UNDER)
#         self.set_data(sex, left_player, right_player)
#
#     def set_data(self, sex, left_player, right_player):
#         self.columns = [
#             make_player_columninfo(
#                 sex, left_player.ident if left_player else None, 'left', 1),
#             make_generic_columninfo(sex, title='avg', number=2),
#             make_player_columninfo(
#                 sex, right_player.ident if right_player else None, 'right', 3)
#         ]
#
#
# class TielossAffectPageLGR(ttk.Frame):
#     """ notebook's page (tab) """
#
#     def __init__(self, parent, application):
#         ttk.Frame.__init__(self, parent)  # parent is notebook obj
#         self.application = application
#         self.page_builder = PageBuilderLGR(
#             application.sex(), application.left_player(), application.right_player())
#         self.page_builder.create_widgets(self)  # fill self with widgets
#
#     def estimate(self):
#         self.page_builder.set_data(
#             self.application.sex(), self.application.left_player(),
#             self.application.right_player()
#         )
#         self.page_builder.clear(self)
#         self.page_builder.update(self)
#
#
def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--sex", choices=["wta", "atp"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        log.initialize(
            co.logname(__file__, instance=str(args.sex)),
            file_level="debug",
            console_level="info",
        )
        dbdet = open_db(sex=args.sex)
        sys.exit(do_stat())
    else:
        log.initialize(co.logname(__file__, test=True), "info", "info")
        dba.open_connect()
        unittest.main()
        dba.close_connect()
        sys.exit(0)
