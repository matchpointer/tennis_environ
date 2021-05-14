# -*- coding: utf-8 -*-
import sys
import datetime
import unittest
import copy
from collections import defaultdict, namedtuple
import argparse
from contextlib import closing

import tkinter.ttk

import log
import cfg_dir
import common as co
import file_utils as fu
import dict_tools
import oncourt_players
import report_line as rl
import stat_cont as st
import score as sc
import tennis
import tennis_time as tt
import dba
import feature
from detailed_score_dbsa import open_db
import matchstat
from detailed_score import error_contains

ASPECT = "tieloss_affect"

MIN_SIZE_DEFAULT = 9


# sex -> ASC ordered dict{(year, weeknum) -> dict{pid -> Sumator}}
# Sumator where n means n_tieloss_affects fails, value is sum of fail affects
data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(st.Sumator)))


NODETAILED_AFFECT_DICT = {(0, 6): 1.0, (1, 6): 0.95, (2, 6): 0.2}


def affect_value(std_set_score):
    return NODETAILED_AFFECT_DICT.get(std_set_score, 0.0)


def add_feature_for_tie_loser(
    features, sex, match, setnum, max_date, prefix="s1_loser_"
):
    scr = match.score[setnum - 1]
    if scr in ((6, 7), (7, 6)):
        player_id = (
            match.first_player.ident if scr == (6, 7) else match.second_player.ident
        )
        add_feature(features, sex, player_id, max_date, prefix)
        return
    features.append(feature.RigidFeature(name=prefix + ASPECT))


def add_feature(features, sex, player_id, max_date, prefix):
    sv = sized_value_affect_ratio(sex, player_id, max_date)
    if sv.size <= MIN_SIZE_DEFAULT:
        features.append(feature.RigidFeature(name=prefix + ASPECT))
        return False
    features.append(feature.RigidFeature(name=prefix + ASPECT, value=sv.value))
    return True


def add_read_feature(features, sex, player_id):
    sv = read_sized_value_affect_ratio(sex, player_id)
    if sv.size <= MIN_SIZE_DEFAULT:
        features.append(feature.RigidFeature(name=ASPECT))
        return False
    features.append(feature.RigidFeature(name=ASPECT, value=sv.value))
    return True


def add_empty_feature(features):
    """uses for best_of_five matches (not implemented)"""
    features.append(feature.RigidFeature(name=ASPECT))


def sized_value_affect_ratio(sex, player_id, max_date):
    max_ywn = tt.get_year_weeknum(max_date) if max_date else None
    res = st.Sumator()
    dct = data_dict[sex]
    for ywn, plr_dct in dct.items():
        if max_ywn is not None and ywn >= max_ywn:
            break
        res += plr_dct[player_id]
    return rl.SizedValue(res.average(), res.size)


def read_sized_value_affect_ratio(sex, player_id):
    cache_dct = read_sized_value_affect_ratio.cached_from_sex[sex]
    if cache_dct is not None:
        return cache_dct[player_id]

    dct = _read_players_dict(sex)
    read_sized_value_affect_ratio.cached_from_sex[sex] = dct
    return dct[player_id]


def _read_players_dict(sex):
    filename = players_filename(sex, suffix="_id")
    dct = dict_tools.load(
        filename,
        createfun=lambda: defaultdict(rl.SizedValue),
        keyfun=int,
        valuefun=rl.SizedValue.create_from_text,
    )
    return dct


read_sized_value_affect_ratio.cached_from_sex = {
    "wta": None,
    "atp": None,
}


class ReadFilesTest(unittest.TestCase):
    def test_read_players_dict(self):
        dct = _read_players_dict(sex="wta")
        self.assertTrue(len(dct) > 500)
        if dct:
            vals = [sv.value for sv in dct.values()]
            self.assertTrue(all([(0 <= v <= 1) for v in vals]))

    def test_high_threshol_player(self):
        goerges_id = 7102
        is_aff = is_player_affect_high("wta", goerges_id)
        self.assertTrue(is_aff)


def initialize(sex):
    max_date = None  # datetime.date(2010, 1, 1)
    if not dba.initialized():
        dba.open_connect()
    min_date = None  # tt.now_minus_history_days(get_history_days(sex))

    if sex in ("wta", None):
        _initialize_results_sex("wta", min_date=min_date, max_date=max_date)
    if sex in ("atp", None):
        _initialize_results_sex("atp", min_date=min_date, max_date=max_date)


def _initialize_results_sex(sex, min_date=None, max_date=None):
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
            scr = sc.Score(score_txt)
            sets_count = scr.sets_count(full=True)
            if scr.retired and sets_count < 2:
                continue
            rnd = tennis.Round.from_oncourt_id(rnd_id)
            fst_wl, snd_wl = _get_match_data(
                sex, date, tour_id, rnd, fst_id, snd_id, scr, sets_count
            )
            if fst_wl or snd_wl:
                past_monday = tt.past_monday_date(date)
                ywn = tt.get_year_weeknum(past_monday)
                if fst_wl:
                    data_dict[sex][ywn][fst_id] += fst_wl
                if snd_wl:
                    data_dict[sex][ywn][snd_id] += snd_wl


detscr_wl = st.WinLoss()


def _get_match_data(sex, date, tour_id, rnd, fst_id, snd_id, scr, sets_count):
    def is_nodecided_tie():
        return any(
            [
                (s in ((6, 7), (7, 6)))
                for n, s in enumerate(scr, start=1)
                if n < sets_count
            ]
        )

    def get_detailed_affect(tieloss_side):
        def tieloss_side_loss_game(game_num):
            det_game = next_set_items[game_num - 1][1]
            game_win_side = co.side(det_game.left_wingame)
            return game_win_side.is_oppose(tieloss_side)

        def is_game_simulated(game_num):
            det_game = next_set_items[game_num - 1][1]
            return error_contains(det_game.error, "GAME_SCORE_SIMULATED")

        def is_absent_game_in_first(n_games=5):
            for gidx in range(n_games):
                at_scr = next_set_items[gidx][0][-1]
                is_absent = sum(at_scr) > gidx
                if is_absent:
                    return True
            return False

        next_set_items = list(det_score.set_items(setnum=setnum + 1))
        if (
            len(next_set_items) < 5
            or is_game_simulated(game_num=1)
            or is_game_simulated(game_num=2)
            or is_game_simulated(game_num=3)
            or is_game_simulated(game_num=4)
            or is_game_simulated(game_num=5)
            or is_absent_game_in_first(n_games=5)
        ):
            return None
        loss_g1 = tieloss_side_loss_game(game_num=1)
        loss_g2 = tieloss_side_loss_game(game_num=2)
        loss_g3 = tieloss_side_loss_game(game_num=3)
        loss_g4 = tieloss_side_loss_game(game_num=4)
        loss_g5 = tieloss_side_loss_game(game_num=5)
        if loss_g1 and loss_g2 and loss_g3 and loss_g4 and loss_g5:
            return 0.8  # 0:5
        if loss_g1 and loss_g2 and loss_g3 and loss_g4:
            return 0.6  # 0:4
        if loss_g1 and sum((loss_g2, loss_g3, loss_g4, loss_g5)) == 3:
            return 0.5  # 1:4 with loss game1
        if loss_g1 and loss_g2 and loss_g3:
            return 0.35  # 0:3
        if loss_g1 and loss_g2:
            return 0.2  # 0:2
        if loss_g1 and sum((loss_g2, loss_g3, loss_g4)) == 2:
            return 0.16  # 1:3 with loss game1
        det_game1 = next_set_items[0][1]
        if loss_g1 and not det_game1.opener_wingame:
            return 0.1  # 0:1 broken

    def get_affect():
        tieloss_side = co.LEFT if set_scr[0] < set_scr[1] else co.RIGHT
        next_set_scr = scr[setidx + 1]
        nextwin_side = co.LEFT if next_set_scr[0] > next_set_scr[1] else co.RIGHT
        if nextwin_side == tieloss_side:
            aff_val = 0.0
        else:
            std_next_set_scr = next_set_scr
            if nextwin_side == co.LEFT:
                std_next_set_scr = co.reversed_tuple(next_set_scr)

            if std_next_set_scr in ((0, 6), (1, 6)):
                aff_val = affect_value(std_next_set_scr)
            elif det_score is not None:
                aff_val = get_detailed_affect(tieloss_side)
                if aff_val is None:
                    aff_val = affect_value(std_next_set_scr)
            else:
                aff_val = affect_value(std_next_set_scr)
        return tieloss_side, aff_val

    if not is_nodecided_tie():
        return None, None
    det_score = dbdet.get_detailed_score(tour_id, rnd, fst_id, snd_id)
    if date > datetime.date(2010, 1, 1):
        detscr_wl.hit(det_score is not None)

    fst_wl, snd_wl = st.Sumator(), st.Sumator()
    for setidx in range(sets_count - 1):
        setnum = setidx + 1
        set_scr = scr[setidx]
        if set_scr not in ((6, 7), (7, 6)):
            continue
        side, aff_value = get_affect()
        if side is None:
            continue
        if side.is_left():
            fst_wl.hit(aff_value)
        else:
            snd_wl.hit(aff_value)
    return fst_wl, snd_wl


# --------------------- work with data files ----------------------------


def get_history_days(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365 * 20)
    elif sex == "wta":
        return int(365 * 20)
    raise co.TennisError("bad sex {}".format(sex))


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex):
        self.sex = sex
        self.actual_players = oncourt_players.players(sex)
        assert len(self.actual_players) > 0, "oncourt_players not inited"
        self.plr_val = []

    def process_all(self):
        self.plr_val = []
        self._build_players_list()
        self.plr_val.sort(key=lambda i: i[1].value, reverse=True)
        self._output_avg_generic_value()
        self._output_players_values()

    def _build_players_list(self):
        self.plr_val = []
        for plr in self.actual_players:
            plr_sval = sized_value_affect_ratio(self.sex, plr.ident, max_date=None)
            if plr_sval.size < MIN_SIZE_DEFAULT:
                continue
            self.plr_val.append((copy.copy(plr), plr_sval))

    def _output_avg_generic_value(self):
        values = [sv.value for _, sv in self.plr_val]
        sum_val = sum(values[2:-2])  # with skip 2 min and 2 max
        n_players = len(values) - 4
        if n_players > 0:
            fu.ensure_folder(generic_dirname(self.sex))
            avg_txt = "{:.4f}".format(sum_val / float(n_players))
            print("mean value:", avg_txt)
            filename = generic_filename(self.sex)
            with open(filename, "w") as fh:
                fh.write("avg__{}\n".format(avg_txt))

    def _output_players_values(self):
        fu.ensure_folder(players_dirname(self.sex))

        filename = players_filename(self.sex)
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{}\n".format(plr, sv))

        filename = players_filename(self.sex, suffix="_id")
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{:.4f} ({})\n".format(plr.ident, sv.value, sv.size))


def generic_dirname(sex):
    return "{}/{}_stat".format(cfg_dir.stat_misc_dir(sex), ASPECT)


def generic_filename(sex):
    return "{}/avg.txt".format(generic_dirname(sex))


def players_dirname(sex):
    return "{}/{}_stat".format(cfg_dir.stat_players_dir(sex), ASPECT)


def players_filename(sex, suffix=""):
    dirname = players_dirname(sex)
    return "{}/players{}.txt".format(dirname, suffix)


def read_generic_value(sex):
    filename = generic_filename(sex)
    with open(filename, "r") as fh:
        line = fh.readline().strip()
        head, valtxt = line.split("__")
        return float(valtxt)


def do_stat():
    try:
        msg = "sex: {}".format(args.sex)
        log.info(__file__ + " started {}".format(msg))
        dba.open_connect()
        dbdet.query_matches()
        oncourt_players.initialize(yearsnum=15)
        initialize(args.sex)
        maker = StatMaker(args.sex)
        maker.process_all()
        log.info(__file__ + " done {}".format(ASPECT))
        log.info("detscr_wl: {}".format(detscr_wl))

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


# -------------- for extern use
avg_affect = {
    "wta": read_generic_value("wta"),
    "atp": read_generic_value("atp"),
}

above_avg_high_ratio = {
    "wta": 0.31,
    "atp": 0.27,
}

min_size = {
    "wta": 10,
    "atp": 14,
}


def is_player_affect_high(sex, player_id):
    sv = read_sized_value_affect_ratio(sex, player_id)
    if sv.size >= min_size[sex]:
        return sv.value > high_affect_threshold(sex)


def is_player_affect_above_avg(sex, player_id):
    sv = read_sized_value_affect_ratio(sex, player_id)
    if sv.size >= min_size[sex]:
        return sv.value > avg_affect[sex]


def high_affect_threshold(sex):
    avg_val = avg_affect[sex]
    return avg_val + avg_val * above_avg_high_ratio[sex]


# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> str, where key is left-most entity (aspect)
ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")


def make_player_columninfo(sex, player_id, title, number, value_default="-"):
    def value_function(key):
        def get_text_val(sized_value):
            if not sized_value:
                return value_default
            return co.formated(
                sized_value.value, sized_value.size, round_digits=2
            ).strip()

        sval = read_sized_value_affect_ratio(sex, player_id)
        return get_text_val(sval)

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if player_id else empty_function,
        value_default=value_default,
    )


def make_generic_columninfo(sex, title, number, value_default="-"):
    def value_function(key):
        result = read_generic_value(sex)
        if result is None:
            return value_default
        return "{0:.2f}".format(result)

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function,
        value_default=value_default,
    )


class PageBuilderLGR(matchstat.PageBuilder):
    """шаблон для закладки c тремя колонками: left_player, generic, right_player"""

    def __init__(self, sex, left_player, right_player):
        super(PageBuilderLGR, self).__init__()
        self.keys.append(ASPECT)
        self.set_data(sex, left_player, right_player)

    def set_data(self, sex, left_player, right_player):
        self.columns = [
            make_player_columninfo(
                sex, left_player.ident if left_player else None, "left", 1
            ),
            make_generic_columninfo(sex, title="avg", number=2),
            make_player_columninfo(
                sex, right_player.ident if right_player else None, "right", 3
            ),
        ]


class TielossAffectPageLGR(tkinter.ttk.Frame):
    """notebook's page (tab)"""

    def __init__(self, parent, application):
        tkinter.ttk.Frame.__init__(self, parent)  # parent is notebook obj
        self.application = application
        self.page_builder = PageBuilderLGR(
            application.sex(), application.left_player(), application.right_player()
        )
        self.page_builder.create_widgets(self)  # fill self with widgets

    def estimate(self):
        self.page_builder.set_data(
            self.application.sex(),
            self.application.left_player(),
            self.application.right_player(),
        )
        self.page_builder.clear(self)
        self.page_builder.update(self)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        log.initialize(
            co.logname(__file__, instance=str(args.sex)),
            file_level="debug",
            console_level="info",
        )
        dbdet = open_db(args.sex)
        sys.exit(do_stat())
    else:
        log.initialize(co.logname(__file__, test=True), "info", "info")
        dba.open_connect()
        unittest.main()
        dba.close_connect()
        sys.exit(0)
