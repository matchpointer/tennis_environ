# -*- coding: utf-8 -*-
import functools
import sys
import datetime
import unittest
from collections import defaultdict, namedtuple
from contextlib import closing
import argparse

import tkinter.ttk

import cfg_dir
import log
import common as co
import file_utils as fu
import matchstat
import oncourt_players
import stat_cont as st
import dict_tools
import dba
import score as sc
import feature
from report_line import SizedValue
import tennis_time as tt
import tennis
from tournament import Tournament
import set_naming

ASPECT = "tie"

# TieType: (MID only for best_of_five when sets score is 1-1)
OPEN = 1
MID = 2
DECIDED = 3
FST_PRESS = 4
FST_UNDER = 5

# here left_result_code in [1.,...,2.] if left player win tie and negate if loss.
TieResult = namedtuple("TieResult", "left_setname_code left_result_code")

# here tie_results is tuple
MatchTieResults = namedtuple("MatchTieResults", "first_id second_id tie_results")

# sex -> ordereddict{date -> list of MatchTieResults}
results_dict = defaultdict(lambda: defaultdict(list))


TIMENAMES = ("deep",)  # ('med', 'deep')


class TimeSpan:
    names_days = {
        ("wta", "med"): 3653,  # 10 years
        ("wta", "deep"): 7310,  # 20 years
        ("atp", "med"): 3105,  # 8.5 years
        ("atp", "deep"): 6210,  # 17 years
    }

    def __init__(self, sex, name):
        assert sex in ("wta", "atp")
        assert name in TIMENAMES
        self.sex = sex
        self.name = name

    def __str__(self):
        return f"{self.sex}_{self.name}"

    def get_min_date(self, current_date=None):
        days = self.names_days[(self.sex, self.name)]
        return self._get_min_date_impl(days, current_date=current_date)

    @staticmethod
    def get_deep_min_date(sex):
        days = TimeSpan.names_days[(sex, "deep")]
        return TimeSpan._get_min_date_impl(days, current_date=None)

    @staticmethod
    def _get_min_date_impl(days, current_date=None):
        collect_timedelta = datetime.timedelta(days=days)
        if current_date:
            return tt.past_monday_date(current_date) - collect_timedelta
        return tt.past_monday_date(datetime.date.today()) - collect_timedelta


def read_player_sized_value(
    sex: str, timename: str, plr_id: int, set_names, is_ratio=True
) -> SizedValue:
    """if set_names is None then all possible set_names"""
    prefix = "plr_wl_" if is_ratio else "plr_val_"
    dct = read_dict(
        player_filename(sex, timename=timename, plr_id=plr_id, prefix=prefix)
    )
    sval = ballanced_value_from_dict(set_names, dct)
    return sval


def add_read_sv_features_pair(
    features, name, sex, fst_id, snd_id, set_names, min_size, is_ratio=True
):
    """if set_names is None then all possible set_names"""
    fst_sval = read_player_sized_value(
        sex, "deep", fst_id, set_names, is_ratio=is_ratio
    )
    snd_sval = read_player_sized_value(
        sex, "deep", snd_id, set_names, is_ratio=is_ratio
    )
    fst_value = None if fst_sval.size < min_size else fst_sval
    snd_value = None if snd_sval.size < min_size else snd_sval
    fst_feat, snd_feat = feature.make_pair(
        fst_name="fst_" + name,
        snd_name="snd_" + name,
        fst_value=fst_value,
        snd_value=snd_value,
    )
    features.append(fst_feat)
    features.append(snd_feat)


def add_read_sv_features_cross_pair(
    features,
    name,
    sex,
    fst_id,
    snd_id,
    fst_set_names,
    snd_set_names,
    min_size,
    is_ratio=True,
):
    """assume fst_set_names and snd_set_names are complement"""
    fst_sval = read_player_sized_value(
        sex, "deep", fst_id, fst_set_names, is_ratio=is_ratio
    )
    snd_sval = read_player_sized_value(
        sex, "deep", snd_id, snd_set_names, is_ratio=is_ratio
    )
    fst_value = None if fst_sval.size < min_size else fst_sval
    snd_value = None if snd_sval.size < min_size else snd_sval
    features.append(feature.RigidFeature(name, fst_value))
    features.append(feature.RigidFeature(name, snd_value))


class TestAddReadSvFeaturesPair(unittest.TestCase):
    def test_secondset_add_read_sv_features_pair(self):
        """atp Jack Sock - Facundo Bagnis"""
        features = []
        sock_id, bagnis_id = 17474, 11881
        add_read_sv_features_pair(
            features,
            name="s2_under_tie_ratio",
            sex="atp",
            fst_id=sock_id,
            snd_id=bagnis_id,
            set_names=("second",),
            min_size=14,
            is_ratio=True,
        )
        self.assertEqual(len(features), 2)
        if len(features) == 2:
            f1 = features[0]
            f2 = features[1]
            self.assertTrue(bool(f1))
            self.assertTrue(bool(f2))


def get_features_pair(
    name, sex, fst_id, snd_id, min_date, max_date, set_names, min_size
):
    """if set_names is None then all possible set_names"""
    fst_sval = player_sized_value(
        sex, fst_id, set_names=set_names, min_date=min_date, max_date=max_date
    )
    snd_sval = player_sized_value(
        sex, snd_id, set_names=set_names, min_date=min_date, max_date=max_date
    )
    fst_value = None if fst_sval.size < min_size else fst_sval.value
    snd_value = None if snd_sval.size < min_size else snd_sval.value
    return feature.make_pair(
        fst_name="fst_" + name,
        snd_name="snd_" + name,
        fst_value=fst_value,
        snd_value=snd_value,
    )


def player_sized_value(sex, ident, set_names, min_date=None, max_date=None):
    """return SizedValue. if set_names is None then all possible set_names"""

    def collect_result(tie_results, side):
        for tie_result in tie_results:
            if side == co.LEFT:
                if set_codes is None or tie_result.left_setname_code in set_codes:
                    sumer.hit(tie_result.left_result_code)
            else:
                right_setcode = set_naming.get_opponent_code(
                    tie_result.left_setname_code
                )
                if set_codes is None or right_setcode in set_codes:
                    sumer.hit(-tie_result.left_result_code)

    sumer = st.Sumator()
    set_codes = (
        None if set_names is None else [set_naming.get_code(sn) for sn in set_names]
    )
    for date, match_tie_results_list in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_tie_results in match_tie_results_list:
            if match_tie_results.first_id == ident:
                collect_result(match_tie_results.tie_results, side=co.LEFT)
            elif match_tie_results.second_id == ident:
                collect_result(match_tie_results.tie_results, side=co.RIGHT)
    return SizedValue.create_from_sumator(sumer)


def player_winloss(sex, ident, set_names, min_date=None, max_date=None):
    """return WinLoss. if set_names is None then all possible set_names"""

    def collect_result(tie_results, side):
        for tie_result in tie_results:
            if side == co.LEFT:
                if set_codes is None or tie_result.left_setname_code in set_codes:
                    winloss.hit(tie_result.left_result_code > 0)
            else:
                right_setcode = set_naming.get_opponent_code(
                    tie_result.left_setname_code
                )
                if set_codes is None or right_setcode in set_codes:
                    winloss.hit(-tie_result.left_result_code > 0)

    winloss = st.WinLoss()
    set_codes = (
        None if set_names is None else [set_naming.get_code(sn) for sn in set_names]
    )
    for date, match_tie_results_list in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_tie_results in match_tie_results_list:
            if match_tie_results.first_id == ident:
                collect_result(match_tie_results.tie_results, side=co.LEFT)
            elif match_tie_results.second_id == ident:
                collect_result(match_tie_results.tie_results, side=co.RIGHT)
    return winloss


def initialize_results(sex=None, min_date=None, max_date=None):
    if sex in ("wta", None):
        if "wta" in results_dict:
            results_dict["wta"].clear()
        _initialize_results_sex("wta", min_date=min_date, max_date=max_date)
    if sex in ("atp", None):
        if "atp" in results_dict:
            results_dict["atp"].clear()
        _initialize_results_sex("atp", min_date=min_date, max_date=max_date)


def players_dirname(sex: str, timename: str) -> str:
    return f"{cfg_dir.stat_players_dir(sex)}/{ASPECT}_stat/{timename}"


def player_filename(sex: str, timename: str, plr_id: int, prefix: str):
    dirname = players_dirname(sex, timename)
    return f"{dirname}/{prefix}{plr_id}.txt"


def generic_dirname(sex: str) -> str:
    return "{}/{}_stat".format(cfg_dir.stat_misc_dir(sex), ASPECT)


def generic_filename(sex: str) -> str:
    dirname = generic_dirname(sex)
    return "{}/avg.txt".format(dirname)


def ballanced_value_from_dict(set_names, dictionary: dict) -> SizedValue:
    svalues = [
        sv for sn, sv in dictionary.items() if set_names is None or sn in set_names
    ]
    if svalues:
        return functools.reduce(lambda a, b: a.ballanced_with(b), svalues)
    return SizedValue()


def read_dict(filename: str):
    return dict_tools.load(filename, keyfun=str, valuefun=SizedValue.create_from_text)


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex):
        self.sex = sex
        self.actual_players = oncourt_players.players(sex)
        assert len(self.actual_players) > 0, "oncourt_players not inited"
        self.min_size_for_gen = 8 if sex == "wta" else 10

        # for output sorted players (generic only)
        self.open_plr_sv = []
        self.dec_plr_sv = []
        self.open_plr_wl = []
        self.dec_plr_wl = []
        self.min_plr_size = 10 if sex == "wta" else 15

    @staticmethod
    def all_setnames(sex):
        result = ["open", "press", "under", "decided"]
        if sex == "atp":
            result.extend(["open2", "press2", "under2"])
        return result

    def process_all(self):
        timespans = [TimeSpan(self.sex, tn) for tn in TIMENAMES]
        for timespan in timespans:
            log.info(f"begin {timespan.name}")
            fu.ensure_folder(players_dirname(self.sex, timespan.name))
            fu.remove_files_in_folder(
                players_dirname(self.sex, timespan.name), filemask="*", recursive=False
            )
            fu.ensure_folder(generic_dirname(self.sex))
            gen_dct = defaultdict(st.Sumator)
            min_date = timespan.get_min_date()
            for plr in self.actual_players:
                plr_val_dct, plr_wl_dct = {}, {}
                for set_name in self.all_setnames(self.sex):
                    plr_sval = player_sized_value(
                        self.sex, plr.ident, set_names=(set_name,), min_date=min_date
                    )
                    if plr_sval:
                        plr_val_dct[set_name] = plr_sval
                    self._output_player_data(plr.ident, plr_val_dct, timespan.name)
                    if (
                        timespan.name == "deep"
                        and plr_sval.size >= self.min_size_for_gen
                    ):
                        gen_dct[set_name].hit(plr_sval.value)

                    plr_wl = player_winloss(
                        self.sex, plr.ident, set_names=(set_name,), min_date=min_date
                    )
                    if plr_wl:
                        plr_wl_dct[set_name] = plr_wl
                    self._output_player_data(plr.ident, plr_wl_dct, timespan.name)

                    if (
                        set_name == "open"
                        and timespan.name == "deep"
                        and plr_sval.size >= self.min_plr_size
                    ):
                        self.open_plr_sv.append((plr, plr_sval))
                        self.open_plr_wl.append((plr, plr_wl))
                    if (
                        set_name == "decided"
                        and timespan.name == "deep"
                        and plr_sval.size >= self.min_plr_size
                    ):
                        self.dec_plr_sv.append((plr, plr_sval))
                        self.dec_plr_wl.append((plr, plr_wl))
            if timespan.name == "deep":
                self._output_generic_data(gen_dct)

    def _output_generic_data(self, gen_dct):
        filename = generic_filename(self.sex)
        with open(filename, "w") as fh:
            for set_name, sumer in gen_dct.items():
                fh.write(
                    "{}__{:.4f} ({})\n".format(set_name, sumer.average(), sumer.size)
                )

        self._output_plr_sorted(self.open_plr_sv, "plr_open_sort.txt")
        self._output_plr_sorted(self.dec_plr_sv, "plr_dec_sort.txt")
        self._output_plr_sorted(
            [(p, SizedValue(wl.ratio, wl.size)) for p, wl in self.open_plr_wl],
            "plr_wl_open_sort.txt",
        )
        self._output_plr_sorted(
            [(p, SizedValue(wl.ratio, wl.size)) for p, wl in self.dec_plr_wl],
            "plr_wl_dec_sort.txt",
        )

    def _output_plr_sorted(self, plr_sv_list, shortname):
        plr_sv_list.sort(key=lambda i: i[1].value, reverse=True)
        filename = "{}/{}".format(generic_dirname(self.sex), shortname)
        with open(filename, "w") as fh:
            for plr, sv in plr_sv_list:
                fh.write("{}__{:.4f} ({})\n".format(str(plr), sv.value, sv.size))

    def _output_player_data(self, plr_id, plr_dct, timename: str):
        if len(plr_dct) == 0:
            return
        is_winloss = isinstance(list(plr_dct.values())[0], st.WinLoss)
        filename = player_filename(
            self.sex, timename, plr_id, prefix="plr_wl_" if is_winloss else "plr_val_"
        )
        with open(filename, "w") as fh:
            for set_name, val in plr_dct.items():
                if is_winloss:
                    fh.write("{}__{}\n".format(set_name, val.ratio_size_str()))
                else:
                    fh.write("{}__{}\n".format(set_name, val.tostring(precision=4)))


DEBUG_PLR_ID = None
DEBUG_SET_CODE = 6
DEBUG_OPP_SET_CODE = 4


def _debug_plr(fst_id, snd_id, tie_results, date, rnd, scr):
    if not DEBUG_PLR_ID:
        return
    our_tie_res = []
    our_res_val = None
    if DEBUG_PLR_ID == fst_id:
        our_tie_res = [
            tr for tr in tie_results if tr.left_setname_code == DEBUG_SET_CODE
        ]
        our_res_val = [tr.left_result_code for tr in our_tie_res]
    elif DEBUG_PLR_ID == snd_id:
        our_tie_res = [
            tr for tr in tie_results if tr.left_setname_code == DEBUG_OPP_SET_CODE
        ]
        our_res_val = [-tr.left_result_code for tr in our_tie_res]
    if not our_tie_res:
        return
    log.debug(
        "DEBUG_PLR_ID {} date: {} rnd: {} scr: {}".format(our_res_val, date, rnd, scr)
    )


tour_cache = dict()  # (sex, tour_id) -> Tournament


def _initialize_results_sex(sex, min_date=None, max_date=None):
    tmp_dct = defaultdict(list)  # date -> list of MatchTieResults
    sql = """select tours.ID_T, tours.NAME_T, tours.DATE_T, games.DATE_G, 
                    Rounds.NAME_R, Courts.NAME_C, tours.COUNTRY_T, tours.RANK_T, 
                    tours.PRIZE_T, games.RESULT_G, fst_plr.ID_P, fst_plr.NAME_P, 
                    snd_plr.ID_P, snd_plr.NAME_P
             from Tours_{0} AS tours, games_{0} AS games, Rounds, Players_{0} AS fst_plr,
                  Players_{0} AS snd_plr, Courts
             where games.ID_T_G = tours.ID_T
               and tours.ID_C_T = Courts.ID_C 
               and games.ID1_G = fst_plr.ID_P
               and games.ID2_G = snd_plr.ID_P
               and games.ID_R_G = Rounds.ID_R
               and (tours.NAME_T Not Like '%juniors%')
               and (fst_plr.NAME_P Not Like '%/%')""".format(
        sex
    )
    sql += dba.sql_dates_condition(min_date, max_date)
    sql += " ;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (
            tour_id,
            tour_name,
            tour_dt,
            match_dt,
            rnd_name,
            surf_txt,
            tour_cou,
            tour_rank,
            tour_money,
            score_txt,
            fst_id,
            fst_name,
            snd_id,
            snd_name,
        ) in cursor.execute(sql):
            tdate = tour_dt.date() if tour_dt else None
            mdate = match_dt.date() if match_dt else None
            if not score_txt:
                continue
            score = sc.Score(score_txt)
            if score.retired:
                continue
            if (sex, tour_id) in tour_cache:
                tour = tour_cache[(sex, tour_id)]
            else:
                tour = Tournament(
                    tour_id,
                    tour_name,
                    sex=sex,
                    surface=tennis.Surface(surf_txt),
                    rank=tour_rank,
                    date=tdate,
                    money=tour_money,
                    cou=tour_cou,
                )
                tour_cache[(sex, tour_id)] = tour
            rnd = tennis.Round(rnd_name)
            date = mdate if mdate else tdate
            try:
                score, dectieinfo = sc.get_decided_tiebreak_info_ext(tour, rnd, score)
            except co.TennisScoreSuperTieError as err:
                log.warn(f"{err} {fst_name} {snd_name}")
                continue
            tie_results = _get_tie_results(score, dectieinfo)
            tie_results_len = len(tie_results)
            if tie_results_len == 0:
                continue
            # _debug_plr(fst_id, snd_id, tie_results, date, rnd, scr)
            tmp_dct[date].append(MatchTieResults(fst_id, snd_id, tie_results))
        dates = list(tmp_dct.keys())
        dates.sort()
        for date in dates:
            results_dict[sex][date] = tmp_dct[date]


def _get_tie_results(score, dectieinfo):
    res_lst = []
    best_of_five = score.best_of_five()
    decsetnum = 5 if best_of_five else 3
    for set_num, inset_score in enumerate(score, start=1):
        isdec = set_num == decsetnum
        if (
            (not isdec and inset_score in ((7, 6), (6, 7)))
            or (
                isdec
                and inset_score in ((7, 6), (6, 7))
                and dectieinfo.beg_scr == (6, 6)
            )
            or (
                isdec
                and inset_score in ((13, 12), (12, 13))
                and dectieinfo.beg_scr == (12, 12)
            )
            or (
                isdec
                and inset_score in ((1, 0), (0, 1))
                and dectieinfo.beg_scr == (0, 0)
            )
        ):
            left_setname_code = set_naming.get_code(
                set_naming.get_names(score, set_num, best_of_five)
            )
            tie_loser_result = score.tie_loser_result(set_num)
            if left_setname_code is not None:
                left_win = inset_score[0] >= inset_score[1]
                left_res_code = _get_result_code(dectieinfo, isdec, tie_loser_result)
                if not left_win:
                    left_res_code = -left_res_code
                res_lst.append(TieResult(left_setname_code, left_res_code))
    return tuple(res_lst)


def _get_result_code(dec_tie_info, decided, tie_loser_result):
    """:return: float in [1,...2]"""
    if tie_loser_result is None:
        return 1.3  # near about mean
    is_super_tie = decided and dec_tie_info and dec_tie_info.is_super
    min_winner_pts = 10 if is_super_tie else 7
    winner_pts = max(min_winner_pts, tie_loser_result + 2)
    dif = winner_pts - tie_loser_result
    return 1.0 + (dif - 2.0) / (min_winner_pts - 2.0)


class PlayerAtpAO2019Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        min_date = datetime.date(2019, 1, 14)
        max_date = datetime.date(2019, 1, 28)
        initialize_results(sex="atp", min_date=min_date, max_date=max_date)

    def test_supertie_carreno_busta(self):
        carreno_busta = 14432
        sv = player_sized_value("atp", ident=carreno_busta, set_names=("decided",))
        self.assertEqual(sv.size, 1)
        self.assertEqual(sv.value, -1)  # lose super tiebreak 8:10 (vs Nishikori)


class PlayerAtp2017SummerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        min_date = datetime.date(2017, 5, 15)
        max_date = datetime.date(2017, 8, 7)  # not include this monday
        initialize_results(sex="atp", min_date=min_date, max_date=max_date)

    def test_player_anderson(self):
        anderson = 7459
        sv = player_sized_value("atp", ident=anderson, set_names=None)
        self.assertTrue(sv.value > 0)

        sv = player_sized_value("atp", ident=anderson, set_names=("open2",))
        self.assertTrue(sv.value > 0)

        sv = player_sized_value("atp", ident=anderson, set_names=("open",))
        self.assertTrue(sv.value > 0)

        sv = player_sized_value("atp", ident=anderson, set_names=("decided",))
        # 25.05 lose dec tie 7-6(6)
        # 03.08 win  dec tie 7-6(7)
        self.assertTrue(sv.value == 0)
        self.assertTrue(sv.size == 2)

        sv = player_sized_value("atp", ident=anderson, set_names=("press2",))
        # 30.05 win press2 tie 7-6(4) -> val=+1.2
        # 07.07 win press2 tie 7-6(3) -> val=+1.4
        self.assertTrue(abs(sv.value - 1.3) < 0.001)
        self.assertEqual(sv.size, 2)

    def test_player_jaziri(self):
        jaziri = 6002
        sv = player_sized_value("atp", ident=jaziri, set_names=("under2",))
        # 30.05 lose press2 tie 7-6(4) -> val=-1.2
        self.assertEqual(sv.size, 1)
        self.assertEqual(sv.value, -1.2)


def do_stat():
    try:
        msg = "sex: {}".format(args.sex)
        log.info(__file__ + " started {}".format(msg))
        oncourt_players.initialize(yearsnum=2)
        initialize_results(sex=args.sex, min_date=TimeSpan.get_deep_min_date(args.sex))
        maker = StatMaker(args.sex)
        maker.process_all()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


# ------------------------------ GUI ----------------------------------------
# here value_fun is function(key) -> str, where key is 3-tuple:
#    (left_setname, right_setname, 'winloss' or 'value')
ColumnInfo = namedtuple("ColumnInfo", "title number value_fun value_default")


def make_player_columninfo(sex, player_id, title, number, value_default="-"):
    def value_function(key):
        def get_text_val(sized_value):
            if not sized_value:
                return value_default
            return co.formated(
                sized_value.value, sized_value.size, round_digits=2
            ).strip()

        is_ratio = key[2] != "value"
        plr_setname = key[0] if number == 1 else key[1]
        sval = read_player_sized_value(
            sex,
            "deep",
            player_id,
            set_names=(plr_setname,) if plr_setname else None,
            is_ratio=is_ratio,
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


def make_ballance_columninfo(sex, fst_pid, snd_pid, title, number, value_default="-"):
    def value_function(key):
        is_ratio = key[2] != "value"
        if not is_ratio:
            return value_default
        plr_setname = key[0] if number in (1, 2) else key[1]
        sval1 = read_player_sized_value(
            sex,
            "deep",
            fst_pid,
            set_names=(plr_setname,) if plr_setname else None,
            is_ratio=is_ratio,
        )
        plr_setname = key[1] if number in (2, 3) else key[1]
        sval2 = read_player_sized_value(
            sex,
            "deep",
            snd_pid,
            set_names=(plr_setname,) if plr_setname else None,
            is_ratio=is_ratio,
        )
        min_size = 11 if sex == "wta" else 15
        if sval1.size < min_size or sval2.size < min_size:
            return value_default
        prob1, prob2 = co.twoside_values(sval1, sval2)
        return "{:.3f}/{:.3f}".format(prob1, prob2)

    return ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function,
        value_default=value_default,
    )


class PageBuilderLBR(matchstat.PageBuilder):
    """шаблон для закладки c тремя колонками: left_player, ballanced_val, right_player"""

    def __init__(self, sex, left_player, right_player):
        super(PageBuilderLBR, self).__init__()
        self.keys = []
        self.set_data(sex, left_player, right_player)

    def set_data(
        self, sex, fst_player, snd_player, left_setname="open", right_setname="open"
    ):
        self.keys = [
            (left_setname, right_setname, "winloss"),
            (left_setname, right_setname, "value"),
        ]
        self.columns = [
            make_player_columninfo(
                sex, fst_player.ident if fst_player is not None else None, "Plr1", 1
            ),
            make_ballance_columninfo(
                sex,
                fst_player.ident if fst_player is not None else None,
                snd_player.ident if snd_player is not None else None,
                "bln",
                2,
            ),
            make_player_columninfo(
                sex, snd_player.ident if snd_player is not None else None, "Plr2", 3
            ),
        ]


class TieStatPageLBR(tkinter.ttk.Frame):
    """notebook's page (tab)"""

    def __init__(self, parent, application):
        tkinter.ttk.Frame.__init__(self, parent)  # parent is notebook obj
        self.application = application
        self.page_builder = PageBuilderLBR(
            application.sex(), application.left_player(), application.right_player()
        )
        self.page_builder.create_widgets(self)  # fill self with widgets

    def estimate(self):
        scr = self.application.score()
        set_names = set_naming.get_names(
            scr, setnum=len(scr), best_of_five=self.application.best_of_five()
        )
        self.page_builder.set_data(
            self.application.sex(),
            self.application.left_player(),
            self.application.right_player(),
            left_setname=set_names[0],
            right_setname=set_names[1],
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
    if args.stat and args.sex:
        log.initialize(co.logname(__file__), "info", "info")
        dba.open_connect()
        res_code = do_stat()
        dba.close_connect()
        sys.exit(res_code)
    else:
        log.initialize(co.logname(__file__, test=True), "debug", "debug")
        dba.open_connect()
        unittest.main()
        dba.close_connect()
        sys.exit(0)
