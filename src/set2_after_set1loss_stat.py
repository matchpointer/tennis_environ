# -*- coding: utf-8 -*-
import sys
import os
import unittest
from collections import defaultdict, namedtuple
import copy
import argparse
from contextlib import closing
import datetime

import tkinter.ttk

import log
import cfg_dir
import common as co
import file_utils as fu
import report_line as rl
import dict_tools
import stat_cont as st
import score as sc
import oncourt_players
import tennis_time as tt
import dba
import ratings_std
import feature
import matchstat
from clf_common import RANK_STD_BOTH_ABOVE, RANK_STD_MAX_DIF

RECOVERY_ASPECT = "set2win_after_set1loss"
KEEP_ASPECT = "set2win_after_set1win"

MIN_SIZE_DEFAULT = 20  # pandemic correction -4
MAX_SIZE_DEFAULT = 50

# sex -> REVERSE(DESC) ordered dict{date -> list of (plr_id, set2_win)}
recovery_dict = defaultdict(lambda: defaultdict(list))

keep_dict = defaultdict(lambda: defaultdict(list))


def add_features(
    features,
    sex,
    aspect,
    fst_id,
    snd_id,
    min_date,
    max_date,
    min_size=MIN_SIZE_DEFAULT,
    max_size=MAX_SIZE_DEFAULT,
):
    f1, f2 = get_features_pair(
        sex,
        aspect,
        fst_id,
        snd_id,
        min_date,
        max_date,
        min_size=min_size,
        max_size=max_size,
    )
    features.append(f1)
    features.append(f2)


def get_features_pair(
    sex,
    aspect,
    fst_id,
    snd_id,
    min_date,
    max_date,
    min_size=MIN_SIZE_DEFAULT,
    max_size=MAX_SIZE_DEFAULT,
):
    fst_wl = player_winloss(
        sex, aspect, fst_id, max_size=max_size, min_date=min_date, max_date=max_date
    )
    snd_wl = player_winloss(
        sex, aspect, snd_id, max_size=max_size, min_date=min_date, max_date=max_date
    )
    return feature.make_pair(
        fst_name="fst_" + aspect,
        snd_name="snd_" + aspect,
        fst_value=fst_wl.ratio if fst_wl.size >= min_size else None,
        snd_value=snd_wl.ratio if snd_wl.size >= min_size else None,
    )


def read_features_pair(sex, aspect, fst_id, snd_id):
    """read data from already prepared files"""
    assert aspect in (RECOVERY_ASPECT, KEEP_ASPECT), "bad aspect {}".format(aspect)
    fst_sv = read_player_sized_value(sex, aspect, fst_id)
    snd_sv = read_player_sized_value(sex, aspect, snd_id)
    return feature.make_pair(
        fst_name="fst_" + aspect,
        snd_name="snd_" + aspect,
        fst_value=fst_sv.value if fst_sv.size >= MIN_SIZE_DEFAULT else None,
        snd_value=snd_sv.value if snd_sv.size >= MIN_SIZE_DEFAULT else None,
    )


def player_winloss(sex, aspect, ident, max_size, min_date=None, max_date=None):
    """return WinLoss. dates do as semi-closed range: [min_date,...,max_date)"""
    wl = st.WinLoss()
    dct = recovery_dict if aspect == RECOVERY_ASPECT else keep_dict
    for date, match_results_list in dct[sex].items():
        if max_date is not None and date >= max_date:
            continue
        if min_date is not None and date < min_date:
            break
        for plr_id, set2_win in match_results_list:
            if plr_id == ident:
                wl.hit(set2_win)
                if wl.size >= max_size:
                    return wl
    return wl


def read_player_sized_value(sex, aspect, player_id):
    cache_dct = read_player_sized_value.cached_from_sex[(sex, aspect)]
    if cache_dct is not None:
        return cache_dct[player_id]

    filename = players_filename(sex, aspect, suffix="_id")
    dct = dict_tools.load(
        filename,
        createfun=lambda: defaultdict(rl.SizedValue),
        keyfun=int,
        valuefun=rl.SizedValue.create_from_text,
    )
    read_player_sized_value.cached_from_sex[(sex, aspect)] = dct
    return dct[player_id]


read_player_sized_value.cached_from_sex = {
    ("wta", RECOVERY_ASPECT): None,
    ("wta", KEEP_ASPECT): None,
    ("atp", RECOVERY_ASPECT): None,
    ("atp", KEEP_ASPECT): None,
}


def initialize_results(
    sex,
    max_rating=RANK_STD_BOTH_ABOVE,
    max_rating_dif=RANK_STD_MAX_DIF,
    with_bo5=False,
    min_date=None,
    max_date=None,
):
    if sex in ("wta", None):
        if "wta" in recovery_dict:
            recovery_dict["wta"].clear()
        if "wta" in keep_dict:
            keep_dict["wta"].clear()
        _initialize_results_sex(
            "wta",
            max_rating=max_rating,
            max_rating_dif=max_rating_dif,
            with_bo5=False,
            min_date=min_date,
            max_date=max_date,
        )
    if sex in ("atp", None):
        if "atp" in recovery_dict:
            recovery_dict["atp"].clear()
        if "atp" in keep_dict:
            keep_dict["atp"].clear()
        _initialize_results_sex(
            "atp",
            max_rating=max_rating,
            max_rating_dif=max_rating_dif,
            with_bo5=with_bo5,
            min_date=min_date,
            max_date=max_date,
        )


def _initialize_results_sex(
    sex, max_rating, max_rating_dif, with_bo5=False, min_date=None, max_date=None
):
    tmp_recov_dct = defaultdict(list)  # date -> list of (plr_id, set2win)
    tmp_keep_dct = defaultdict(list)  # date -> list of (plr_id, set2win)
    sql = """select tours.DATE_T, games.DATE_G, games.RESULT_G, games.ID1_G, games.ID2_G
             from Tours_{0} AS tours, games_{0} AS games, Players_{0} AS fst_plr
             where games.ID_T_G = tours.ID_T 
               and games.ID1_G = fst_plr.ID_P
               and (tours.NAME_T Not Like '%juniors%')
               and (fst_plr.NAME_P Not Like '%/%')""".format(
        sex
    )
    sql += dba.sql_dates_condition(min_date, max_date)
    sql += " ;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (tour_dt, match_dt, score_txt, fst_id, snd_id) in cursor.execute(sql):
            tdate = tour_dt.date() if tour_dt else None
            mdate = match_dt.date() if match_dt else None
            if not score_txt:
                continue
            scr = sc.Score(score_txt)
            sets_count = scr.sets_count(full=True)
            if sets_count < 2:
                continue
            date = mdate if mdate else tdate
            if date is None:
                continue
            recov, keep = _get_match_data(
                sex, date, fst_id, snd_id, scr, max_rating, max_rating_dif, with_bo5
            )
            if recov is not None:
                tmp_recov_dct[date].append((recov.player_id, recov.set2_win))
            if keep is not None:
                tmp_keep_dct[date].append((keep.player_id, keep.set2_win))
        recov_dates = list(tmp_recov_dct.keys())
        recov_dates.sort(reverse=True)
        for date in recov_dates:
            recovery_dict[sex][date] = tmp_recov_dct[date]
        keep_dates = list(tmp_keep_dct.keys())
        keep_dates.sort(reverse=True)
        for date in keep_dates:
            keep_dict[sex][date] = tmp_keep_dct[date]


MatchData = namedtuple("MatchData", "player_id set2_win")


def _get_match_data(
    sex, date, fst_id, snd_id, scr, max_rating, max_rating_dif, with_bo5
):
    def admit_rtg_dif(match_data, recovery_mode):
        match_winer_strong_fav = snd_rtg > (fst_rtg + max_rating_dif)
        if not match_winer_strong_fav:
            return True
        if recovery_mode and match_data.set2_win and match_data.player_id == snd_id:
            return True  # match loser ok recovered in set2
        if (
            not recovery_mode
            and not match_data.set2_win
            and match_data.player_id == fst_id
        ):
            return True  # match winner failed keep in set2
        return False

    fst_rtg = ratings_std.get_rank(sex, fst_id, date)
    if fst_rtg is None or fst_rtg > max_rating:
        return None, None
    snd_rtg = ratings_std.get_rank(sex, snd_id, date)
    if snd_rtg is None or snd_rtg > max_rating:
        return None, None
    if sex == "atp" and not with_bo5 and scr.best_of_five():
        return None, None
    set1, set2 = scr[0], scr[1]
    if set1[0] < set1[1]:
        recov = MatchData(fst_id, set2[0] > set2[1])
        keep = MatchData(snd_id, set2[0] <= set2[1])
    else:
        recov = MatchData(snd_id, set2[0] <= set2[1])
        keep = MatchData(fst_id, set2[0] > set2[1])

    if not admit_rtg_dif(recov, recovery_mode=True):
        recov = None
    if not admit_rtg_dif(keep, recovery_mode=False):
        keep = None
    return recov, keep


# --------------------- work with data files ----------------------------


def history_days_personal(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365 * 6)
    elif sex == "wta":
        return int(365 * 6)
    raise co.TennisError("bad sex {}".format(sex))


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex, min_size, max_size):
        self.sex = sex
        self.min_size = min_size
        self.max_size = max_size
        self.actual_players = oncourt_players.players(sex)
        self.plr_val = []

    def process_all(self, aspect):
        self.plr_val = []
        for plr in self.actual_players:
            plr_wl = player_winloss(self.sex, aspect, plr.ident, max_size=self.max_size)
            if plr_wl.size < self.min_size:
                continue
            self.plr_val.append((copy.copy(plr), plr_wl))
        self.output(aspect)

    def output(self, aspect):
        self.plr_val.sort(key=lambda i: i[1].ratio, reverse=True)
        self.output_avg_value(aspect)
        self.output_players_values(aspect)

    def output_avg_value(self, aspect):
        fu.ensure_folder(generic_dirname(self.sex, aspect))
        avg_val = self.get_avg_value()
        if avg_val is not None:
            filename = generic_filename(self.sex, aspect)
            with open(filename, "w") as fh:
                fh.write("avg_set2win_ratio: {:.4f}".format(avg_val))

    def get_avg_value(self):
        values = [sv.value for _, sv in self.plr_val]
        sum_val = sum(values[2:-2])  # with skip 2 min and 2 max
        n_players = len(values) - 4
        if n_players > 0:
            return sum_val / float(n_players)

    def output_players_values(self, aspect):
        fu.ensure_folder(players_dirname(self.sex, aspect))

        filename = players_filename(self.sex, aspect)
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{}\n".format(plr, sv))

        filename = players_filename(self.sex, aspect, suffix="_id")
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{:.4f} ({})\n".format(plr.ident, sv.value, sv.size))


def generic_dirname(sex, aspect):
    return "{}/{}_stat".format(cfg_dir.stat_misc_dir(sex), aspect)


def generic_filename(sex, aspect):
    return "{}/avg.txt".format(generic_dirname(sex, aspect))


def players_dirname(sex, aspect):
    return "{}/{}_stat".format(cfg_dir.stat_players_dir(sex), aspect)


def players_filename(sex, aspect, suffix=""):
    dirname = players_dirname(sex, aspect)
    return os.path.join(dirname, "avg_values{}.txt".format(suffix))


def read_generic_value(sex, aspect):
    if (sex, aspect) in read_generic_value.cache:
        return read_generic_value.cache[(sex, aspect)]
    filename = generic_filename(sex, aspect)
    with open(filename, "r") as fh:
        line = fh.readline().strip()
        head, valtxt = line.split(" ")
        value = float(valtxt)
        read_generic_value.cache[(sex, aspect)] = value
        return value


read_generic_value.cache = dict()


def do_stat():
    log.initialize(
        co.logname(__file__, instance=str(args.sex)),
        file_level="debug",
        console_level="info",
    )
    try:
        msg = "sex: {} min_size: {} max_size: {} max_rtg: {}  max_rtg_dif: {}".format(
            args.sex, args.min_size, args.max_size, args.max_rating, args.max_rating_dif
        )
        log.info(__file__ + " started {}".format(msg))
        dba.open_connect()
        oncourt_players.initialize(yearsnum=8)
        min_date = tt.now_minus_history_days(history_days_personal(args.sex))
        ratings_std.initialize(
            sex=args.sex, min_date=min_date - datetime.timedelta(days=7)
        )
        initialize_results(
            args.sex,
            max_rating=args.max_rating,
            max_rating_dif=args.max_rating_dif,
            min_date=min_date,
        )

        maker = StatMaker(args.sex, min_size=args.min_size, max_size=args.max_size)
        maker.process_all(aspect=RECOVERY_ASPECT)
        log.info(__file__ + " done {}".format(RECOVERY_ASPECT))
        maker.process_all(aspect=KEEP_ASPECT)
        log.info(__file__ + " done {}".format(KEEP_ASPECT))

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


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

        sval = read_player_sized_value(sex, key, player_id)
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
        result = read_generic_value(sex, key)
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
        self.keys.append(RECOVERY_ASPECT)
        self.keys.append(KEEP_ASPECT)
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


class Set2RecoveryPageLGR(tkinter.ttk.Frame):
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
    parser.add_argument("--min_size", type=int, default=MIN_SIZE_DEFAULT)
    parser.add_argument("--max_size", type=int, default=MAX_SIZE_DEFAULT)
    parser.add_argument("--max_rating", type=int, default=RANK_STD_BOTH_ABOVE)
    parser.add_argument("--max_rating_dif", type=int, default=RANK_STD_MAX_DIF)
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        sys.exit(do_stat())
    else:
        log.initialize(co.logname(__file__, test=True), "info", "info")
        unittest.main()
        sys.exit(0)
