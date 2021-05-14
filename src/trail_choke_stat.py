# -*- coding: utf-8 -*-

import sys
import datetime
import copy
import unittest
from collections import defaultdict, namedtuple, OrderedDict
import argparse
import functools

import tkinter
import tkinter.ttk


import log
import common as co
import cfg_dir
import file_utils as fu
import report_line as rl
import set_naming
import stat_cont as st
import dict_tools
import dba
import oncourt_players
import tennis_time as tt
from detailed_score import SetItems
from detailed_score_dbsa import open_db
import ratings_std
import feature
import matchstat
from clf_common import RANK_STD_BOTH_ABOVE, RANK_STD_MAX_DIF

ASPECT_TRAIL = "trail"  # it is bonus value for recovery and set win
ASPECT_CHOKE = "choke"  # it is penalty value for choke and set loss


setname_to_code = {
    "open": 1,
    "press": 2,
    "under": 3,
    "decided": 4,
    "open2": 5,
    "press2": 6,
    "under2": 7,
}


def get_setname_code(setname):
    return setname_to_code[setname]


def get_setname(code):
    for setname, set_code in setname_to_code.items():
        if code == set_code:
            return setname


def get_setnames(sex):
    if sex == "wta":
        return [n for n in setname_to_code if "2" not in n]
    return list(setname_to_code.keys())


def get_min_size(sex, setnames):
    n_setnames = len(get_setnames(sex)) if setnames is None else len(setnames)
    if n_setnames == 1:
        result = 15
    elif 2 <= n_setnames <= 3:
        result = 22
    else:
        result = 28
    return (result + 5) if sex == "atp" else result


def all_aspects():
    return ASPECT_TRAIL, ASPECT_CHOKE


def root_name():
    return "{}_{}_stat".format(ASPECT_TRAIL, ASPECT_CHOKE)


# (sex, aspect) -> asc ordered dict:
#  {date -> {(plr_id, setname_code) -> (n_sets_with_lead, penalty_value)}}
all_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: (0, 0.0))))


def player_feature_name(is_fst, aspect, setname, feat_suffix=""):
    return "{}_{}{}{}".format(
        "fst" if is_fst else "snd",
        aspect,
        "" if setname is None else "_" + setname,
        feat_suffix,
    )


def add_features(
    features,
    sex,
    aspect,
    setname,
    fst_id,
    snd_id,
    min_date,
    max_date,
    max_size=None,
    agregate_bo5=False,
    feat_suffix="",
):
    min_size = get_min_size(sex, (setname,))
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
        agregate_bo5=agregate_bo5,
        feat_suffix=feat_suffix,
    )
    features.append(copy.copy(f1))
    features.append(copy.copy(f2))


def add_features_agr(
    features,
    sex,
    aspect,
    fst_id,
    snd_id,
    min_date,
    max_date,
    max_size=None,
    feat_suffix="",
):
    min_size = get_min_size(sex, setnames=None)
    f1, f2 = get_features_pair(
        sex,
        aspect,
        None,
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
    features, sex, aspect, setname, fst_id, snd_id, agregate_bo5=False, feat_suffix=""
):
    min_size = get_min_size(sex, (setname,))
    f1, f2 = read_features_pair(
        sex,
        aspect,
        setname,
        fst_id,
        snd_id,
        min_size=min_size,
        agregate_bo5=agregate_bo5,
        feat_suffix=feat_suffix,
    )
    features.append(copy.copy(f1))
    features.append(copy.copy(f2))


def add_agr_read_features(
    features, sex, aspect, setnames, fst_id, snd_id, feat_suffix=""
):
    min_size = get_min_size(
        sex, setnames=get_setnames(sex) if setnames is None else setnames
    )
    f1, f2 = agr_read_features_pair(
        sex,
        aspect,
        setnames=get_setnames(sex) if setnames is None else setnames,
        fst_id=fst_id,
        snd_id=snd_id,
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
    min_size,
    max_size=None,
    agregate_bo5=False,
    feat_suffix="",
):
    fst_sv = player_sized_value(
        sex,
        aspect,
        setname,
        fst_id,
        max_size=max_size,
        min_date=min_date,
        max_date=max_date,
    )
    snd_sv = player_sized_value(
        sex,
        aspect,
        setname,
        snd_id,
        max_size=max_size,
        min_date=min_date,
        max_date=max_date,
        agregate_bo5=agregate_bo5,
    )
    return feature.make_pair(
        fst_name=player_feature_name(True, aspect, setname, feat_suffix),
        snd_name=player_feature_name(False, aspect, setname, feat_suffix),
        fst_value=fst_sv.value if fst_sv.size >= min_size else None,
        snd_value=snd_sv.value if snd_sv.size >= min_size else None,
    )


def read_features_pair(
    sex, aspect, setname, fst_id, snd_id, min_size, agregate_bo5=False, feat_suffix=""
):
    """read data from already prepared files"""
    if agregate_bo5:
        fst_sv = agr_read_player_sized_value(sex, aspect, setname, fst_id)
        snd_sv = agr_read_player_sized_value(sex, aspect, setname, snd_id)
    else:
        fst_sv = read_player_sized_value(sex, aspect, setname, fst_id)
        snd_sv = read_player_sized_value(sex, aspect, setname, snd_id)
    return feature.make_pair(
        fst_name=player_feature_name(True, aspect, setname, feat_suffix),
        snd_name=player_feature_name(False, aspect, setname, feat_suffix),
        fst_value=fst_sv.value if fst_sv.size >= min_size else None,
        snd_value=snd_sv.value if snd_sv.size >= min_size else None,
    )


def agr_read_features_pair(
    sex, aspect, setnames, fst_id, snd_id, min_size, feat_suffix=""
):
    """read data from already prepared files"""
    fst_sv = agr_read_player_sized_value(sex, aspect, setnames, fst_id)
    snd_sv = agr_read_player_sized_value(sex, aspect, setnames, snd_id)
    fst_name = player_feature_name(True, aspect, None, feat_suffix)
    snd_name = player_feature_name(False, aspect, None, feat_suffix)
    return feature.make_pair(
        fst_name=fst_name,
        snd_name=snd_name,
        fst_value=fst_sv.value if fst_sv.size >= min_size else None,
        snd_value=snd_sv.value if snd_sv.size >= min_size else None,
    )


def player_sized_value(
    sex,
    aspect,
    setname,
    ident,
    max_size=None,
    min_date=None,
    max_date=None,
    agregate_bo5=False,
):
    """return SizedValue. dates do as semi-closed range: [min_date,...,max_date)
    if setname is None then all possible setnames for given sex
    """

    def get_sized_value():
        if n_sets_out == 0:
            return rl.SizedValue()
        return rl.SizedValue(
            value=float(value_sum) / float(n_sets_out), size=n_sets_out
        )

    def get_set_codes():
        if setname is None:
            return [setname_to_code[n] for n in get_setnames(sex)]  # all possible codes

        if agregate_bo5 and sex == "atp":
            return [setname_to_code[n] for n in get_setnames(sex) if setname in n]

        return [get_setname_code(setname)]

    n_sets_out, value_sum = 0, 0
    set_codes = get_set_codes()
    for date in all_dict[(sex, aspect)]:
        if max_date is not None and date >= max_date:
            continue
        if min_date is not None and date < min_date:
            break

        for code in set_codes:
            n_sets, value = all_dict[(sex, aspect)][date][(ident, code)]
            if n_sets > 0:
                n_sets_out += n_sets
                value_sum += value

        if max_size is not None and n_sets_out >= max_size:
            return get_sized_value()
    return get_sized_value()


def agr_read_player_sized_value(sex, aspect, setnames, player_id):
    svalues = []
    for sname in setnames:
        sval = read_player_sized_value(sex, aspect, sname, player_id)
        if sval:
            svalues.append(sval)

    if svalues:
        return functools.reduce(lambda a, b: a.ballanced_with(b), svalues)
    return rl.SizedValue()


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


def initialize_results(
    sex, dbdet, max_rating=RANK_STD_BOTH_ABOVE, max_rating_dif=RANK_STD_MAX_DIF
):
    _initialize_results_sex(
        sex, dbdet, max_rating=max_rating, max_rating_dif=max_rating_dif
    )


IsActualPlayer = namedtuple("IsActualPlayer", "fst snd")


def _initialize_results_sex(sex, dbdet, max_rating, max_rating_dif):
    def actual_player_conditions():
        if __name__ == "__main__":
            fst_id, snd_id = rec.left_id, rec.right_id
            fst_plr = co.find_first(actual_players, lambda p: p.ident == fst_id)
            snd_plr = co.find_first(actual_players, lambda p: p.ident == snd_id)
            return IsActualPlayer(fst=fst_plr is not None, snd=snd_plr is not None)
        return IsActualPlayer(fst=True, snd=True)

    def edit_result(aspect, setname, plr_id, value):
        code = get_setname_code(setname)
        n_sets_prev, value_prev = all_dict[(sex, aspect)][date][(plr_id, code)]
        all_dict[(sex, aspect)][date][(plr_id, code)] = (
            n_sets_prev + 1,
            value_prev + value,
        )

    actual_players = oncourt_players.players(sex)
    for rec in dbdet.records:
        date = rec.date
        if not _is_match_admit(sex, date, rec, max_rating, max_rating_dif):
            continue
        is_act_plr = actual_player_conditions()
        if not is_act_plr.fst and not is_act_plr.snd:
            continue
        det_score = rec.detailed_score
        score = rec.score
        best_of_five = score.best_of_five()
        n_sets = score.sets_count(full=True)
        for setnum in range(1, n_sets + 1):
            setnames = set_naming.get_names(score, setnum, best_of_five)
            if setnames is None:
                continue  # may be invalid score
            set_items = SetItems.from_scores(setnum, det_score, score)
            if not set_items.ok_set():
                continue
            setname_by_fst, setname_by_snd = setnames
            set_scr = score[setnum - 1]
            set_winner = co.LEFT if set_scr[0] > set_scr[1] else co.RIGHT
            if set_winner.is_left():
                lead2 = _set_penalty(set_items, co.RIGHT)
                if lead2 > 0:
                    if is_act_plr.fst:
                        edit_result(ASPECT_TRAIL, setname_by_fst, rec.left_id, lead2)
                    if is_act_plr.snd:
                        edit_result(ASPECT_CHOKE, setname_by_snd, rec.right_id, lead2)
                else:
                    if is_act_plr.fst:
                        edit_result(ASPECT_CHOKE, setname_by_fst, rec.left_id, 0)
                    if is_act_plr.snd:
                        edit_result(ASPECT_TRAIL, setname_by_snd, rec.right_id, 0)
            else:  # right set winner
                lead1 = _set_penalty(set_items, co.LEFT)
                if lead1 > 0:
                    if is_act_plr.fst:
                        edit_result(ASPECT_CHOKE, setname_by_fst, rec.left_id, lead1)
                    if is_act_plr.snd:
                        edit_result(ASPECT_TRAIL, setname_by_snd, rec.right_id, lead1)
                else:
                    if is_act_plr.fst:
                        edit_result(ASPECT_TRAIL, setname_by_fst, rec.left_id, 0)
                    if is_act_plr.snd:
                        edit_result(ASPECT_CHOKE, setname_by_snd, rec.right_id, 0)


def _is_match_admit(sex, date, record, max_rating, max_rating_dif):
    def admit_rtg_dif():
        win_rtg, loss_rtg = fst_rtg, snd_rtg
        if loss_rtg > (win_rtg + max_rating_dif):
            return False  # strong underdog fail or strong favorite prevail
        return True

    fst_id, snd_id = record.left_id, record.right_id
    fst_rtg = ratings_std.get_rank(sex, fst_id, date)
    if fst_rtg is None or fst_rtg > max_rating:
        return False
    snd_rtg = ratings_std.get_rank(sex, snd_id, date)
    if snd_rtg is None or snd_rtg > max_rating:
        return False
    return admit_rtg_dif()


# serve_side, (left_games, right_games) -> penalty (leadership style)
penalty_dict = OrderedDict(
    [
        ((co.LEFT, (5, 0)), 13.0 / 13.0),
        ((co.RIGHT, (5, 0)), 12.0 / 13.0),
        ((co.LEFT, (5, 1)), 11.0 / 13.0),
        ((co.RIGHT, (5, 1)), 10.0 / 13.0),
        ((co.LEFT, (4, 0)), 9.0 / 13.0),
        ((co.LEFT, (5, 2)), 9.0 / 13.0),
        ((co.RIGHT, (4, 0)), 8.5 / 13.0),
        ((co.LEFT, (4, 1)), 8.0 / 13.0),
        ((co.RIGHT, (5, 2)), 7.0 / 13.0),
        ((co.LEFT, (3, 0)), 7.0 / 13.0),
        ((co.LEFT, (5, 3)), 6.0 / 13.0),
        ((co.RIGHT, (4, 1)), 6.0 / 13.0),
        ((co.RIGHT, (5, 3)), 5.5 / 13.0),
        #
        ((co.LEFT, (6, 5)), 5.0 / 13.0),
        ((co.LEFT, (5, 4)), 5.0 / 13.0),
        ((co.LEFT, (7, 6)), 5.0 / 13.0),
        ((co.LEFT, (8, 7)), 5.0 / 13.0),
        ((co.LEFT, (9, 8)), 5.0 / 13.0),
        ((co.LEFT, (11, 10)), 5.0 / 13.0),
        ((co.LEFT, (12, 11)), 5.0 / 13.0),
        #
        ((co.LEFT, (4, 2)), 5.0 / 13.0),
        ((co.RIGHT, (3, 0)), 5.0 / 13.0),
        ((co.RIGHT, (4, 2)), 4.5 / 13.0),
        ((co.LEFT, (4, 3)), 4.0 / 13.0),
        ((co.LEFT, (3, 1)), 4.0 / 13.0),
        ((co.RIGHT, (3, 1)), 3.5 / 13.0),
        ((co.LEFT, (3, 2)), 3.0 / 13.0),
        ((co.LEFT, (2, 0)), 3.0 / 13.0),
        ((co.RIGHT, (2, 0)), 2.5 / 13.0),
        ((co.LEFT, (2, 1)), 2.0 / 13.0),
        ((co.LEFT, (1, 0)), 1.0 / 13.0),
    ]
)


def _set_penalty(set_items: SetItems, side):
    need_flip = side == co.RIGHT
    penalties = [0.0]
    for scr, dg in set_items:
        serve_side = co.side(dg.left_opener)
        if need_flip:
            scr = co.reversed_tuple(scr)
            serve_side = serve_side.fliped()
        if (serve_side, scr) in penalty_dict:
            penalties.append(penalty_dict[(serve_side, scr)])
    return max(penalties)


# --------------------- work with data files ----------------------------


def history_days_personal(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365 * 8)
    elif sex == "wta":
        return int(365 * 10)
    raise co.TennisError("unexpected sex {}".format(sex))


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex):
        self.sex = sex
        self.actual_players = oncourt_players.players(sex)
        self.plr_val = []
        self.gen_sumer_dict = defaultdict(st.Sumator)  # (aspect, setname) -> Sumator

    def process_all(self):
        for aspect in all_aspects():
            for setname in get_setnames(self.sex):
                self.plr_val = []
                log.info(
                    "{} start {} {} {}".format(__file__, self.sex, aspect, setname)
                )
                for plr in self.actual_players:
                    plr_sv = player_sized_value(
                        self.sex,
                        aspect,
                        setname,
                        ident=plr.ident,
                        max_size=args.max_size,
                        agregate_bo5=False,
                    )
                    if plr_sv.size >= args.min_size_per_file:
                        self.gen_sumer_dict[(aspect, setname)].hit(plr_sv.value)
                        self.plr_val.append((copy.copy(plr), plr_sv))
                self.output(aspect, setname)
                log.info("{} done {} {} {}".format(__file__, self.sex, aspect, setname))
        self._output_generic()

    def _get_generic_all_setnames(self, aspect):
        setname_avg_svalues = [
            rl.SizedValue(value=sm.average(), size=sm.size)
            for (asp, sn), sm in self.gen_sumer_dict.items()
            if asp == aspect
        ]
        return functools.reduce(lambda a, b: a.ballanced_with(b), setname_avg_svalues)

    def _output_generic_value(self, aspect, setname, sized_value):
        fu.ensure_folder(generic_dirname(self.sex))
        if sized_value:
            filename = generic_filename(self.sex, aspect, setname)
            with open(filename, "w") as fh:
                fh.write("avg__{:.4f} ({})".format(sized_value.value, sized_value.size))

    def _output_generic(self):
        for aspect in all_aspects():
            all_sv = self._get_generic_all_setnames(aspect)
            self._output_generic_value(aspect, setname=None, sized_value=all_sv)

        for (aspect, setname), sumer in self.gen_sumer_dict.items():
            self._output_generic_value(
                aspect, setname, rl.SizedValue.create_from_sumator(sumer)
            )

    def output(self, aspect, setname):
        self.plr_val.sort(key=lambda i: i[1].value, reverse=True)
        self.output_players_values(aspect, setname)

    def output_players_values(self, aspect, setname):
        fu.ensure_folder(players_dirname(self.sex))

        filename = players_filename(self.sex, aspect, setname)
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write(
                    "{} ({})__{:.4f} ({})\n".format(
                        plr.name, plr.cou, sv.value, sv.size
                    )
                )

        filename = players_filename(self.sex, aspect, setname, suffix="_id")
        with open(filename, "w") as fh:
            for plr, sv in self.plr_val:
                fh.write("{}__{:.4f} ({})\n".format(plr.ident, sv.value, sv.size))


def generic_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_misc_dir(sex), root_name())


def generic_filename(sex, aspect, setname):
    dirname = generic_dirname(sex)
    if setname is None:
        return "{}/{}.txt".format(dirname, aspect)
    return "{}/{}_{}.txt".format(dirname, aspect, setname)


def players_dirname(sex):
    return "{}/{}".format(cfg_dir.stat_players_dir(sex), root_name())


def players_filename(sex, aspect, setname, suffix=""):
    dirname = players_dirname(sex)
    return "{}/{}_{}{}.txt".format(dirname, aspect, setname, suffix)


def read_generic_value(sex, aspect, setname):
    filename = generic_filename(sex, aspect, setname)
    with open(filename, "r") as fh:
        line = fh.readline().strip()
        head, valtxt = line.split("__")
        return rl.SizedValue.create_from_text(valtxt)


def do_stat():
    try:
        msg = "sex: {} max_rtg: {}  max_rtg_dif: {}".format(
            args.sex, args.max_rating, args.max_rating_dif
        )
        log.info(__file__ + " started {}".format(msg))
        dba.open_connect()
        oncourt_players.initialize(yearsnum=1)
        min_date = tt.now_minus_history_days(history_days_personal(args.sex))
        ratings_std.initialize(
            sex=args.sex, min_date=min_date - datetime.timedelta(days=7)
        )
        matchstat.initialize(sex=args.sex, min_date=min_date)
        dbdet.query_matches(min_date=min_date)
        initialize_results(
            sex=args.sex,
            dbdet=dbdet,
            max_rating=args.max_rating,
            max_rating_dif=args.max_rating_dif,
        )
        maker = StatMaker(args.sex)
        maker.process_all()

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> str, where key is left-most entity (aspect)
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
#         sval = read_player_sized_value(sex, setname=key, player_id=player_id)
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
#         result = read_generic_value(sex, aspect=key)
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
#         self.keys = all_aspects()
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
# class BreakUpLeadLosePageLGR(ttk.Frame):
#     """ notebook's page (tab) """
#
#     def __init__(self, parent, application):
#         ttk.Frame.__init__(self, parent)  # parent is notebook obj
#         self.application = application
#         self.page_builder = PageBuilderLGR(
#             application.sex(), application.left_player(),
#             application.right_player())
#         self.page_builder.create_widgets(self)  # fill self with widgets
#
#     def estimate(self):
#         self.page_builder.set_data(
#             self.application.sex(), self.application.left_player(),
#             self.application.right_player()
#         )
#         self.page_builder.clear(self)
#         self.page_builder.update(self)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--max_rating", type=int, default=RANK_STD_BOTH_ABOVE)
    parser.add_argument("--max_rating_dif", type=int, default=RANK_STD_MAX_DIF)
    parser.add_argument("--min_size_per_file", type=int, default=6)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


# class ReadFeaturesTest(unittest.TestCase):
#     def test_wta_add_features(self):
#         hercog_id = 7926
#         golubic_id = 10749
#         features = []
#         setnames = ('open', 'decided')
#         add_read_features(
#             features, 'wta', setnames, hercog_id, golubic_id, feat_suffix='_sh0')
#         self.assertEqual(len(features), 2 * 3 * len(setnames))
#
#         dif_decided_begin = feature.dif_player_features(features, "decided_begin_sh0")
#         self.assertTrue(dif_decided_begin is not None)

# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> (str, str), where key is left-most entity (aspect)
ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")


def _relative_dif_player_value(plr_svalue, gen_svalue):
    assert gen_svalue and gen_svalue.value > 0.001, "bad generic {}".format(gen_svalue)
    if plr_svalue:
        return (plr_svalue.value - gen_svalue.value) / gen_svalue.value


def make_player_columninfo(sex, is_lead, player_id, title, number, value_default="-"):
    def value_function(key):  # assume that key is (left_setname, right_setname)
        setname = key[0] if number == 1 else key[1]
        aspect = ASPECT_CHOKE if is_lead else ASPECT_TRAIL
        if setname is None:
            plr_sv = agr_read_player_sized_value(
                sex, aspect, get_setnames(sex), player_id
            )
        else:
            plr_sv = read_player_sized_value(sex, aspect, setname, player_id)
        gen_sval = read_generic_value(sex, aspect, setname)
        rel_dif_val = _relative_dif_player_value(plr_sv, gen_sval)
        if rel_dif_val is None:
            return value_default
        return co.formated(rel_dif_val, plr_sv.size, round_digits=3).strip()

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if player_id else empty_function,
        value_default=value_default,
    )


def make_difsum_columninfo(
    sex, fst_lead, fst_id, snd_id, title, number, value_default="-"
):
    def value_function(key):  # assume that key is (left_setname, right_setname)
        setname1 = key[0]
        aspect1 = ASPECT_CHOKE if fst_lead else ASPECT_TRAIL
        gen1_sv = read_generic_value(sex, aspect1, setname1)
        if setname1 is None:
            plr1_sv = agr_read_player_sized_value(
                sex, aspect1, get_setnames(sex), fst_id
            )
        else:
            plr1_sv = read_player_sized_value(sex, aspect1, setname1, fst_id)
        reldif1 = _relative_dif_player_value(plr1_sv, gen1_sv)

        setname2 = key[1]
        aspect2 = ASPECT_TRAIL if fst_lead else ASPECT_CHOKE
        gen2_sv = read_generic_value(sex, aspect2, setname2)
        if setname2 is None:
            plr2_sv = agr_read_player_sized_value(
                sex, aspect2, get_setnames(sex), snd_id
            )
        else:
            plr2_sv = read_player_sized_value(sex, aspect2, setname2, snd_id)
        reldif2 = _relative_dif_player_value(plr2_sv, gen2_sv)
        ratio_coef1 = gen1_sv.value / (gen1_sv.value + gen2_sv.value)
        ratio_coef2 = gen2_sv.value / (gen1_sv.value + gen2_sv.value)
        if reldif1 is not None and reldif2 is not None:
            return "{:+.3f}".format(reldif1 * ratio_coef2 + reldif2 * ratio_coef1)
        else:
            return value_default

    def empty_function(key):
        return value_default

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function,
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
        obj.fst_lead_var = tkinter.IntVar()
        obj.fst_lead_check = tkinter.Checkbutton(
            obj, text="fst_lead", variable=obj.fst_lead_var, onvalue=1, offvalue=0
        )
        obj.fst_lead_check.grid(row=rownum, column=0)
        obj.fst_lead_check.select()

        rownum += 1
        RowBuilder.create_title_row(obj, rownum, self.columns)

        rownum += 1
        for key_idx, key in enumerate(self.keys):
            RowBuilder.create_row(obj, rownum + key_idx, key, self.columns)

    def clear(self, obj):
        rownum = 4
        for key_idx in range(len(self.keys)):
            RowBuilder.clear_row(obj, rownum + key_idx, self.columns)

    def update(self, obj):
        rownum = 4
        for key_idx, key in enumerate(self.keys):
            RowBuilder.update_row(obj, rownum + key_idx, key, self.columns)


class MutualBuilder(PageBuilder):
    def __init__(self, sex, fst_player, snd_player):
        super(MutualBuilder, self).__init__()
        self.set_data(sex, fst_player, snd_player)

    def set_data(
        self,
        sex,
        fst_player,
        snd_player,
        fst_lead=True,
        left_setname="open",
        right_setname="open",
    ):
        self.keys = [(left_setname, right_setname), (None, None)]
        self.columns = [
            make_player_columninfo(
                sex,
                fst_lead,
                fst_player.ident if fst_player is not None else None,
                "P1reldif",
                1,
            ),
            make_difsum_columninfo(
                sex,
                fst_lead,
                fst_player.ident if fst_player is not None else None,
                snd_player.ident if snd_player is not None else None,
                "reldifsum",
                2,
            ),
            make_player_columninfo(
                sex,
                not fst_lead,
                snd_player.ident if snd_player is not None else None,
                "P2reldif",
                3,
            ),
        ]


class TrailChokeMutualPage(tkinter.ttk.Frame):
    """notebook's page (tab)"""

    def __init__(self, parent, application):
        tkinter.ttk.Frame.__init__(self, parent)  # parent is notebook obj
        self.application = application
        self.page_builder = MutualBuilder(
            application.sex(), application.left_player(), application.right_player()
        )
        self.page_builder.create_widgets(self)  # fill self with widgets

    def estimate(self):
        scr = self.application.score()
        set_names = set_naming.get_names(
            scr, setnum=len(scr), best_of_five=self.application.best_of_five()
        )
        print("set_names: '{}', '{}'".format(set_names[0], set_names[1]))
        if set_names == (None, None):
            print("BAD set_names: '{}', '{}'".format(set_names[0], set_names[1]))
            return
        self.page_builder.set_data(
            self.application.sex(),
            self.application.left_player(),
            self.application.right_player(),
            fst_lead=bool(self.fst_lead_var.get()),
            left_setname=set_names[0],
            right_setname=set_names[1],
        )
        self.page_builder.clear(self)
        self.page_builder.update(self)


class RowBuilder(object):
    @staticmethod
    def widget_name(rownum, columninfo):
        return "var_" + str(rownum) + "_" + str(columninfo.number)

    @staticmethod
    def create_title_row(obj, rownum, columninfo_list):
        """create most up (first) widgets in obj (page) from columninfo_list[i].title
        first slot (above keys column) is skiped and leaves as empty (widget absent).
        """
        column_start = 1  # numerate from 0, but skip absent key slot
        for column_idx, columninfo in enumerate(columninfo_list):
            widget_name = RowBuilder.widget_name(rownum, columninfo)
            setattr(obj, widget_name, tkinter.ttk.Label(obj, text=columninfo.title))
            getattr(obj, widget_name).grid(row=rownum, column=column_idx + column_start)

    @staticmethod
    def create_row(obj, rownum, key, columninfo_list):
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
