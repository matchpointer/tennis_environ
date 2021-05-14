# -*- coding: utf-8 -*-

""" cases: 
    A) player down 1 break after 2 games (0:2);
    B) first 2 games gives 1-1. player down 1 break (1:3) after 4 games;
    C) first 2 games gives 1-1. player down  (0:3) after 3 games;
    D) first 2 games gives 1-1. player down  (1:4) after 5 games;
       Result: If he achives equal score in next games.
    E)... see more in function down_then_equal
"""

import sys
import datetime
import copy
import unittest
from collections import defaultdict, namedtuple
import argparse

import tkinter.ttk

import log
import common as co
import cfg_dir
import file_utils as fu
import stat_cont as st
import report_line as rl
import dict_tools
import dba
import oncourt_players
import tennis_time as tt
from detailed_score import is_best_of_five, SetItems
from detailed_score_dbsa import open_db
import ratings_std
import feature
import matchstat
from clf_common import RANK_STD_BOTH_ABOVE, RANK_STD_MAX_DIF

ROOTNAME = "inset_keep_recovery_stat"

RECOVERY_ASPECT = "recovery"
KEEP_ASPECT = "keep"
BEGIN_ASPECT = "begin"
BEGIN_ASPECT_ENABLE = True

BEST_OF_FIVE_ENABLE = True
TREAT_DECIDED_01_AS_DN_UP = True

MAX_SIZE_DEFAULT = 46

# (sex, aspect, setname) -> asc ordered dict{date -> list of (plr_id, is_ok)}
all_dict = defaultdict(lambda: defaultdict(list))


minmax_size_default_dict = {
    ("wta", RECOVERY_ASPECT, "open"): (15, 45),
    ("wta", KEEP_ASPECT, "open"): (15, 45),
    ("wta", BEGIN_ASPECT, "open"): (20, 50),
    #
    ("wta", RECOVERY_ASPECT, "decided"): (10, 40),
    ("wta", KEEP_ASPECT, "decided"): (10, 40),
    ("wta", BEGIN_ASPECT, "decided"): (15, 45),
    #
    ("atp", RECOVERY_ASPECT, "open"): (21, 60),
    ("atp", KEEP_ASPECT, "open"): (21, 60),
    ("atp", BEGIN_ASPECT, "open"): (32, 70),
    #
    ("atp", RECOVERY_ASPECT, "decided"): (14, 60),
    ("atp", KEEP_ASPECT, "decided"): (14, 60),
    ("atp", BEGIN_ASPECT, "decided"): (22, 70),
}


def get_min_size_default(sex, aspect, setname, shift=0):
    minmax = minmax_size_default_dict.get((sex, aspect, setname))
    if minmax is not None:
        return minmax[0] + shift


def get_max_size_default(sex, aspect, setname, shift=0):
    minmax = minmax_size_default_dict.get((sex, aspect, setname))
    if minmax is not None:
        return minmax[1] + shift


def all_aspects():
    result = [RECOVERY_ASPECT, KEEP_ASPECT]
    if BEGIN_ASPECT_ENABLE:
        result.append(BEGIN_ASPECT)
    return result


def player_feature_name(is_fst, aspect, setname, feat_suffix=""):
    return "{}_{}_{}{}".format("fst" if is_fst else "snd", setname, aspect, feat_suffix)


def check_aspect(aspect):
    assert aspect in all_aspects(), "bad aspect {}".format(aspect)
    return True


def add_features(
    features,
    sex,
    setnames,
    fst_id,
    snd_id,
    min_date,
    max_date,
    sizes_shift=0,
    feat_suffix="",
):
    for aspect in all_aspects():
        for setname in setnames:
            min_size = get_min_size_default(sex, aspect, setname, sizes_shift)
            max_size = get_max_size_default(sex, aspect, setname, sizes_shift)
            f1, f2 = get_features_pair(
                sex,
                aspect,
                setname,
                fst_id,
                snd_id,
                min_date,
                max_date,
                min_size=min_size,
                max_size=max_size,
                feat_suffix=feat_suffix,
            )
            features.append(copy.copy(f1))
            features.append(copy.copy(f2))


def add_read_features(
    features, sex, setnames, fst_id, snd_id, sizes_shift=0, feat_suffix=""
):
    for aspect in all_aspects():
        for setname in setnames:
            min_size = get_min_size_default(sex, aspect, setname, sizes_shift)
            f1, f2 = read_features_pair(
                sex,
                aspect,
                setname,
                fst_id,
                snd_id,
                min_size=min_size,
                feat_suffix=feat_suffix,
            )
            features.append(copy.copy(f1))
            features.append(copy.copy(f2))


def get_features_pair(
    sex,
    aspect,
    setname,
    fst_id,
    snd_id,
    min_date,
    max_date,
    min_size=None,
    max_size=None,
    feat_suffix="",
):
    check_aspect(aspect)
    if max_size is None:
        max_size = get_max_size_default(sex, aspect, setname)
    fst_wl = player_winloss(
        sex,
        aspect,
        setname,
        fst_id,
        max_size=max_size,
        min_date=min_date,
        max_date=max_date,
    )
    snd_wl = player_winloss(
        sex,
        aspect,
        setname,
        snd_id,
        max_size=max_size,
        min_date=min_date,
        max_date=max_date,
    )
    if min_size is None:
        min_size = get_min_size_default(sex, aspect, setname)
    return feature.make_pair(
        fst_name=player_feature_name(True, aspect, setname, feat_suffix),
        snd_name=player_feature_name(False, aspect, setname, feat_suffix),
        fst_value=fst_wl.ratio if fst_wl.size >= min_size else None,
        snd_value=snd_wl.ratio if snd_wl.size >= min_size else None,
    )


def read_features_pair(
    sex, aspect, setname, fst_id, snd_id, min_size=None, feat_suffix=""
):
    """read data from already prepared files"""
    check_aspect(aspect)
    fst_sv = read_player_sized_value(sex, aspect, setname, fst_id)
    snd_sv = read_player_sized_value(sex, aspect, setname, snd_id)
    if min_size is None:
        min_size = get_min_size_default(sex, aspect, setname)
    return feature.make_pair(
        fst_name=player_feature_name(True, aspect, setname, feat_suffix),
        snd_name=player_feature_name(False, aspect, setname, feat_suffix),
        fst_value=fst_sv.value if fst_sv.size >= min_size else None,
        snd_value=snd_sv.value if snd_sv.size >= min_size else None,
    )


def player_winloss(
    sex, aspect, setname, ident, max_size=None, min_date=None, max_date=None
):
    """return WinLoss. dates do as semi-closed range: [min_date,...,max_date)"""
    if max_size is None:
        max_size = get_max_size_default(sex, aspect, setname)
    wl = st.WinLoss()
    for date, match_results_list in all_dict[(sex, aspect, setname)].items():
        if max_date is not None and date >= max_date:
            break
        if min_date is not None and date < min_date:
            continue
        for plr_id, is_ok in match_results_list:
            if plr_id == ident:
                wl.hit(is_ok)
                if wl.size >= max_size:
                    return wl
    return wl


def read_player_sized_value(sex, aspect, setname, player_id):
    cache_dct = read_player_sized_value.cached_from_sex[(sex, aspect, setname)]
    if cache_dct is not None:
        return cache_dct[player_id]
    filename = players_filename(sex, aspect, setname, suffix="_id")
    dct = dict_tools.load(
        filename,
        createfun=lambda: defaultdict(rl.SizedValue),
        keyfun=int,
        valuefun=rl.SizedValue.create_from_text,
    )
    read_player_sized_value.cached_from_sex[(sex, aspect, setname)] = dct
    return dct[player_id]


read_player_sized_value.cached_from_sex = defaultdict(lambda: None)


class SetInform(object):
    def __init__(self, sex, setnames):
        self.sex = sex
        self.setnames = setnames

    def num_name_pairs(self, det_score):
        for setname in self.setnames:
            if setname == "open":
                yield 1, "open"
            elif setname == "decided":
                if self.sex == "wta":
                    yield 3, "decided"
                else:
                    if is_best_of_five(det_score):
                        if BEST_OF_FIVE_ENABLE:
                            yield 5, "decided"
                    else:
                        yield 3, "decided"


def initialize_results(
    sex,
    dbdet,
    setnames,
    max_rating=RANK_STD_BOTH_ABOVE,
    max_rating_dif=RANK_STD_MAX_DIF,
):
    set_inform = SetInform(sex, setnames)
    _initialize_results_sex(
        sex, dbdet, set_inform, max_rating=max_rating, max_rating_dif=max_rating_dif
    )


MatchData = namedtuple("MatchData", "player_id is_ok")


def _initialize_results_sex(sex, dbdet, set_inform, max_rating, max_rating_dif):
    for rec in dbdet.records:
        date = rec.date
        score = rec.score
        det_score = rec.detailed_score
        for setnum, setname in set_inform.num_name_pairs(det_score):
            isdec = setname == "decided"
            trail_eq, lead_sus = _get_match_data(
                sex,
                setnum,
                isdec,
                date,
                det_score,
                score,
                rec.left_id,
                rec.right_id,
                max_rating,
                max_rating_dif,
            )
            if trail_eq is not None:
                all_dict[(sex, RECOVERY_ASPECT, setname)][date].append(
                    (trail_eq.player_id, trail_eq.is_ok)
                )
                if BEGIN_ASPECT_ENABLE:
                    all_dict[(sex, BEGIN_ASPECT, setname)][date].append(
                        (trail_eq.player_id, False)
                    )
            if lead_sus is not None:
                all_dict[(sex, KEEP_ASPECT, setname)][date].append(
                    (lead_sus.player_id, lead_sus.is_ok)
                )
                if BEGIN_ASPECT_ENABLE:
                    all_dict[(sex, BEGIN_ASPECT, setname)][date].append(
                        (lead_sus.player_id, True)
                    )


def _get_match_data(
    sex,
    setnum,
    isdec,
    date,
    det_score,
    score,
    left_id,
    right_id,
    max_rating,
    max_rating_dif,
):
    def admit_rtg_dif(match_data):
        if match_data.is_ok:
            win_rtg, loss_rtg = (
                (fst_rtg, snd_rtg)
                if match_data.player_id == fst_id
                else (snd_rtg, fst_rtg)
            )
        else:
            loss_rtg, win_rtg = (
                (fst_rtg, snd_rtg)
                if match_data.player_id == fst_id
                else (snd_rtg, fst_rtg)
            )

        if loss_rtg > (win_rtg + max_rating_dif):
            return False  # strong underdog fail or strong favorite prevail
        return True

    fst_id, snd_id = left_id, right_id
    fst_rtg = ratings_std.get_rank(sex, fst_id, date)
    if fst_rtg is None or fst_rtg > max_rating:
        return None, None
    snd_rtg = ratings_std.get_rank(sex, snd_id, date)
    if snd_rtg is None or snd_rtg > max_rating:
        return None, None

    set_items = SetItems.from_scores(setnum, det_score, score)
    val_fst = down_then_equal(set_items, co.LEFT, isdec)
    val_snd = down_then_equal(set_items, co.RIGHT, isdec)
    if val_fst is None and val_snd is None:
        return None, None
    if val_fst == 2:
        val_snd = None
    if val_snd == 2:
        val_fst = None
    if val_fst is not None and val_snd is not None:
        log.error(
            "SIMULT fst: {} snd: {}\nitems: {}".format(val_fst, val_snd, set_items)
        )
        return None, None

    if val_fst is not None:
        trail_eq = MatchData(player_id=fst_id, is_ok=bool(val_fst))
        lead_sus = MatchData(player_id=snd_id, is_ok=not bool(val_fst))
    else:
        trail_eq = MatchData(player_id=snd_id, is_ok=bool(val_snd))
        lead_sus = MatchData(player_id=fst_id, is_ok=not bool(val_snd))

    if not admit_rtg_dif(trail_eq):
        trail_eq = None
    if not admit_rtg_dif(lead_sus):
        lead_sus = None
    return trail_eq, lead_sus


def down_then_equal(set_items: SetItems, side, isdec: bool):
    """:return None if skip/unknown;
            1 if side down then equalOK;
            0 if side down then equalFail;
            2 if in decided set side broken in game1 and do break in game2;
    equalOK if equal score found in next games.
    """
    try:

        def get_decided_01_case():
            if isdec and TREAT_DECIDED_01_AS_DN_UP:
                g1, g2 = set_items[0][1], set_items[1][1]
                g1_break = not g1.opener_wingame
                if g1_break and g1.left_opener is side.is_left():
                    # side was broken in game1. get side result in game2:
                    return g2.left_wingame is side.is_left()

        def two_items_score(item1, item2):
            """return (is side win game 1, is side win game 2)"""
            is_win1 = item1[1].left_wingame is side.is_left()
            is_win2 = item2[1].left_wingame is side.is_left()
            return int(is_win1), int(is_win2)

        win1, win2 = two_items_score(set_items[0], set_items[1])
        if (win1, win2) == (0, 0):  # 0:2 in games 1, 2
            if set_items[2][1].left_wingame is not side.is_left():  # -3
                res = set_items.exist_equal_after(3, maxlen=6)
                return int(res) if res is not None else None
            # side have 1:2 here
            if set_items[3][1].left_wingame is side.is_left():  # equal
                return True
            # side have 1:3 here
            res = set_items.exist_equal_after(4, maxlen=5)
            return int(res) if res is not None else None
        elif (win1 + win2) == 1:  # 1:1 in games 1, 2
            dec_01_res = get_decided_01_case()
            if dec_01_res is not None:
                return dec_01_res * 2
            win3, win4 = two_items_score(set_items[2], set_items[3])
            if (win3, win4) == (0, 0):  # -2 after games 3, 4
                if set_items[4][1].left_wingame is not side.is_left():  # -3
                    res = set_items.exist_equal_after(5, maxlen=6)
                    return int(res) if res is not None else None
                # side have -1 here
                if set_items[5][1].left_wingame is side.is_left():  # equal
                    return True
                # side have 2:4 here
                res = set_items.exist_equal_after(6, maxlen=5)
                return int(res) if res is not None else None
            elif (win3 + win4) == 1:  # 2:2 in games 1, 2, 3, 4
                win5, win6 = two_items_score(set_items[4], set_items[5])
                if (win5, win6) == (0, 0):  # -2 after games 5, 6
                    if set_items[6][1].left_wingame is not side.is_left():  # -3
                        res = set_items.exist_equal_after(7, maxlen=6)
                        return int(res) if res is not None else None
                    # side have -1 here
                    if set_items[7][1].left_wingame is side.is_left():  # equal
                        return True
                    # side have 3:5 here
                    res = set_items.exist_equal_after(8, maxlen=5)
                    return int(res) if res is not None else None
    except co.TennisUnknownScoreError:
        return None


# --------------------- work with data files ----------------------------


def history_days_personal(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365 * 9)
    elif sex == "wta":
        return int(365 * 10)
    raise co.TennisError("bad sex {}".format(sex))


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex, setnames):
        self.sex = sex
        self.setnames = setnames
        self.actual_players = oncourt_players.players(sex)
        self.plr_val = []

    def process_all(self):
        for aspect in all_aspects():
            for setname in self.setnames:
                min_size = get_min_size_default(self.sex, aspect, setname, shift=-5)
                max_size = get_max_size_default(self.sex, aspect, setname, shift=0)
                self.plr_val = []
                for plr in self.actual_players:
                    plr_wl = player_winloss(
                        self.sex, aspect, setname, plr.ident, max_size=max_size
                    )
                    if plr_wl.size < min_size:
                        continue
                    self.plr_val.append((copy.copy(plr), plr_wl))
                self.output(aspect, setname)
            log.info(__file__ + " done {}".format(aspect))

    def output(self, aspect, setname):
        self.plr_val.sort(key=lambda i: i[1].ratio, reverse=True)
        self.output_avg_value(aspect, setname)
        self.output_players_values(aspect, setname)

    def output_avg_value(self, aspect, setname):
        fu.ensure_folder(generic_dirname(self.sex))
        avg_val = self.get_avg_value()
        if avg_val is not None:
            filename = generic_filename(self.sex, aspect, setname)
            with open(filename, "w") as fh:
                fh.write("avg_ok_ratio: {:.4f}".format(avg_val))

    def get_avg_value(self):
        values = [sv.value for _, sv in self.plr_val]
        sum_val = sum(values[2:-2])  # with skip 2 min and 2 max
        n_players = len(values) - 4
        if n_players > 0:
            return sum_val / float(n_players)

    def output_players_values(self, aspect, setname):
        fu.ensure_folder(players_dirname(self.sex))

        filename = players_filename(self.sex, aspect, setname)
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{}\n".format(plr, sv))

        filename = players_filename(self.sex, aspect, setname, suffix="_id")
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{:.4f} ({})\n".format(plr.ident, sv.value, sv.size))


def generic_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_misc_dir(sex), ROOTNAME)


def generic_filename(sex, aspect, setname):
    dirname = generic_dirname(sex)
    return "{}/{}_{}.txt".format(dirname, setname, aspect)


def players_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_players_dir(sex), ROOTNAME)


def players_filename(sex, aspect, setname, suffix=""):
    dirname = players_dirname(sex)
    return "{}/{}_{}{}.txt".format(dirname, setname, aspect, suffix)


def read_generic_value(sex, aspect, setname):
    filename = generic_filename(sex, aspect, setname)
    with open(filename, "r") as fh:
        line = fh.readline().strip()
        head, valtxt = line.split(" ")
        return float(valtxt)


def do_stat():
    try:
        msg = "sex: {} max_rtg: {}  max_rtg_dif: {}".format(
            args.sex, args.max_rating, args.max_rating_dif
        )
        log.info(__file__ + " started {}".format(msg))
        dba.open_connect()
        oncourt_players.initialize(yearsnum=8)
        min_date = tt.now_minus_history_days(history_days_personal(args.sex))
        dbdet.query_matches(min_date=min_date)
        ratings_std.initialize(
            sex=args.sex, min_date=min_date - datetime.timedelta(days=7)
        )
        initialize_results(
            sex=args.sex,
            dbdet=dbdet,
            setnames=("open", "decided"),
            max_rating=args.max_rating,
            max_rating_dif=args.max_rating_dif,
        )
        maker = StatMaker(args.sex, setnames=("open", "decided"))
        maker.process_all()

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> str, where key is left-most entity (aspect)
ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")


def make_player_columninfo(sex, setname, player_id, title, number, value_default="-"):
    def value_function(key):
        def get_text_val(sized_value):
            if not sized_value:
                return value_default
            return co.formated(
                sized_value.value, sized_value.size, round_digits=2
            ).strip()

        sval = read_player_sized_value(
            sex, aspect=key, setname=setname, player_id=player_id
        )
        return get_text_val(sval)

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if player_id else empty_function,
        value_default=value_default,
    )


def make_generic_columninfo(sex, setname, title, number, value_default="-"):
    def value_function(key):
        result = read_generic_value(sex, aspect=key, setname=setname)
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

    def __init__(self, sex, setname, left_player, right_player):
        super(PageBuilderLGR, self).__init__()
        self.setname = setname
        self.keys = all_aspects()
        self.set_data(sex, left_player, right_player)

    def set_data(self, sex, left_player, right_player):
        self.columns = [
            make_player_columninfo(
                sex, self.setname, left_player.ident if left_player else None, "left", 1
            ),
            make_generic_columninfo(sex, self.setname, title="avg", number=2),
            make_player_columninfo(
                sex,
                self.setname,
                right_player.ident if right_player else None,
                "right",
                3,
            ),
        ]


class SetKeepRecoveryPageLGR(tkinter.ttk.Frame):
    """notebook's page (tab)"""

    def __init__(self, parent, application, setname):
        tkinter.ttk.Frame.__init__(self, parent)  # parent is notebook obj
        self.application = application
        self.page_builder = PageBuilderLGR(
            application.sex(),
            setname,
            application.left_player(),
            application.right_player(),
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
    parser.add_argument("--max_rating", type=int, default=RANK_STD_BOTH_ABOVE)
    parser.add_argument("--max_rating_dif", type=int, default=RANK_STD_MAX_DIF)
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


class ReadFeaturesTest(unittest.TestCase):
    def test_wta_add_features(self):
        hercog_id = 7926
        golubic_id = 10749
        features = []
        setnames = ("open", "decided")
        add_read_features(
            features, "wta", setnames, hercog_id, golubic_id, feat_suffix="_sh0"
        )
        self.assertEqual(len(features), 2 * 3 * len(setnames))

        dif_decided_begin = feature.dif_player_features(features, "decided_begin_sh0")
        self.assertTrue(dif_decided_begin is not None)


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        log.initialize(co.logname(__file__), "debug", "debug")
        dbdet = open_db(sex=args.sex)
        sys.exit(do_stat())
    else:
        log.initialize(co.logname(__file__, test=True), "debug", "debug")
        unittest.main()
        sys.exit(0)
