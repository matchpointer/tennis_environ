# -*- coding: utf-8 -*-
import sys
import os
import datetime
import unittest
from collections import defaultdict, namedtuple
import argparse
import copy
from contextlib import closing

import log
import common as co
import cfg_dir
import file_utils as fu
import stat_cont as st
import report_line as rl
import dict_tools
import dba
import oncourt_players
import matchstat
import score as sc
import feature

# here first_id is decided set winner with games_dif (bonus_value) in decided
MatchResult = namedtuple("MatchResult", "first_id second_id games_dif")

# sex -> ordereddict{date -> list of MatchResult}
results_dict = defaultdict(lambda: defaultdict(list))

START_HISTORY_DATE = datetime.date(2014, 1, 1)


def winloss_to_float(winloss):
    if winloss.size <= 5:
        return 0.5
    ratio = winloss.ratio
    if winloss.size <= 15:
        if ratio > 0.5:
            return 0.5 + (ratio - 0.5) * 0.5
        if ratio < 0.5:
            return 0.5 - (0.5 - ratio) * 0.5
    return ratio


def sized_value_to_float(sized_value):
    if sized_value.size <= 5:
        return 0.0
    avg_value = sized_value.value
    if sized_value.size <= 15:
        return avg_value * 0.5
    return avg_value


def add_feature_dif_ratio(
    features, feature_name, sex, pid1, pid2, min_date=None, max_date=None
):
    value = get_dif_ratio(sex, pid1, pid2, min_date=min_date, max_date=max_date)
    features.append(feature.Feature(name=feature_name, value=value, flip_value=-value))


def get_dif_ratio(sex, pid1, pid2, min_date=None, max_date=None):
    result1, result2 = st.WinLoss(), st.WinLoss()
    for date, match_results in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_res in match_results:
            if match_res.first_id == pid1:
                result1.add_win(1)
            elif match_res.second_id == pid1:
                result1.add_loss(1)

            if match_res.first_id == pid2:
                result2.add_win(1)
            elif match_res.second_id == pid2:
                result2.add_loss(1)
    return winloss_to_float(result2) - winloss_to_float(result1)


def add_feature_dif_bonus(
    features, feature_name, sex, pid1, pid2, min_date=None, max_date=None
):
    value = get_dif_bonus(sex, pid1, pid2, min_date=min_date, max_date=max_date)
    features.append(feature.Feature(name=feature_name, value=value, flip_value=-value))


def get_dif_bonus(sex, pid1, pid2, min_date=None, max_date=None):
    result1, result2 = st.Sumator(), st.Sumator()
    for date, match_results in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_res in match_results:
            if match_res.first_id == pid1:
                result1.hit(match_res.games_dif)  # win value
            elif match_res.second_id == pid1:
                result1.hit(-match_res.games_dif)  # loss value

            if match_res.first_id == pid2:
                result2.hit(match_res.games_dif)  # win value
            elif match_res.second_id == pid2:
                result2.hit(-match_res.games_dif)  # loss value
    svalue1 = rl.SizedValue.create_from_sumator(result1)
    svalue2 = rl.SizedValue.create_from_sumator(result2)
    return sized_value_to_float(svalue2) - sized_value_to_float(svalue1)


def player_winloss(sex, ident, min_date=None, max_date=None, as_float=False):
    result = st.WinLoss()
    for date, match_results in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_res in match_results:
            if match_res.first_id == ident:
                result.add_win(1)
            elif match_res.second_id == ident:
                result.add_loss(1)
    return winloss_to_float(result) if as_float else result


def player_games_dif_average(sex, ident, min_date=None, max_date=None, as_float=False):
    """:returns SizedValue if as_float is False"""
    sumer = st.Sumator()
    for date, match_results in results_dict[sex].items():
        if min_date is not None and date < min_date:
            continue
        if max_date is not None and date > max_date:
            break
        for match_res in match_results:
            if match_res.first_id == ident:
                sumer.hit(match_res.games_dif)  # win value
            elif match_res.second_id == ident:
                sumer.hit(-match_res.games_dif)  # loss value
    svalue = rl.SizedValue.create_from_sumator(sumer)
    return sized_value_to_float(svalue) if as_float else svalue


def initialize_results(sex=None, min_date=None, max_date=None):
    if sex in ("wta", None):
        if "wta" in results_dict:
            results_dict["wta"].clear()
        _initialize_results_sex("wta", min_date=min_date, max_date=max_date)
    if sex in ("atp", None):
        if "atp" in results_dict:
            results_dict["atp"].clear()
        _initialize_results_sex("atp", min_date=min_date, max_date=max_date)


def _initialize_results_sex(sex, min_date=None, max_date=None):
    tmp_dct = defaultdict(list)  # date -> list of match_results
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
            if scr.retired:
                continue
            sets_cnt = scr.sets_count(full=True)
            if sets_cnt not in (3, 5):
                continue
            if sets_cnt == 3:
                sets_sc = scr.sets_score()
                if sets_sc != (2, 1):
                    continue
            indec_sc = scr[sets_cnt - 1]
            games_dif = indec_sc[0] - indec_sc[1]
            if games_dif <= 0:
                log.error(
                    "strange decided score {} in {} 1id {} 2id {}".format(
                        scr, sex, fst_id, snd_id
                    )
                )
                continue
            date = mdate if mdate else tdate
            tmp_dct[date].append(MatchResult(fst_id, snd_id, games_dif))
        dates = list(tmp_dct.keys())
        dates.sort()
        for date in dates:
            results_dict[sex][date] = tmp_dct[date]


class WinlossPlayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        max_date = datetime.date.today() - datetime.timedelta(days=7)
        initialize_results(
            sex="wta",
            min_date=datetime.date(year=2013, month=1, day=1),
            max_date=max_date,
        )

    def test_date_none(self):
        self.assertFalse(None in list(results_dict["wta"].keys()))

    def test_date_order(self):
        dates = copy.copy(list(results_dict["wta"].keys()))
        dates2 = copy.copy(dates)
        dates2.sort()
        self.assertEqual(dates2, dates)

    def test_player_winloss(self):
        sharapova = 503
        wl = player_winloss(sex="wta", ident=sharapova)
        self.assertTrue(wl.ratio > 0.65)
        self.assertTrue(wl.size > 10)

        serenawilliams = 241
        wl = player_winloss(sex="wta", ident=serenawilliams)
        self.assertTrue(wl.ratio > 0.65)
        self.assertTrue(wl.size > 8)


class DecidedSetPlayersProcessor(object):
    def __init__(self, sex):
        self.dec_sets_from_player = defaultdict(st.WinLoss)
        self.sex = sex
        self.actual_players = oncourt_players.players(sex)

    def process(self, tour):
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q":
                continue
            for match in matches:
                if not match.paired() and match.score.valid():
                    self.__process_match(match)

    def __process_match(self, match):
        setnum = match.score.sets_count(full=True)
        if setnum not in (3, 5):
            return
        decided = setnum == (5 if match.score.best_of_five() else 3)
        if not decided:
            return
        is_fst_plr = match.first_player in self.actual_players
        is_snd_plr = match.second_player in self.actual_players
        if is_fst_plr or is_snd_plr:
            setidx = setnum - 1
            left_winset = match.score[setidx][0] > match.score[setidx][1]
            if is_fst_plr:
                self.dec_sets_from_player[match.first_player.ident].hit(left_winset)

            if is_snd_plr:
                self.dec_sets_from_player[match.second_player.ident].hit(
                    not left_winset
                )

    def reporting(self):
        def report_dict(dictionary, dirname, subname):
            filename = "{}/{}.txt".format(dirname, subname)
            dict_tools.dump(
                dictionary, filename, keyfun=str, valuefun=lambda v: v.ratio_size_str()
            )

        log.info("{} decided_set output start...".format(self.sex))
        dirname = cfg_dir.stat_players_dir(self.sex) + "/decided_set"
        assert os.path.isdir(dirname), "dir not found: " + dirname
        fu.remove_files_in_folder(dirname)
        report_dict(self.dec_sets_from_player, dirname, "decided")
        log.info("{} decided_set output finish".format(self.sex))


def process_sex(sex):
    make_stat.process(
        [DecidedSetPlayersProcessor(sex)],
        sex,
        min_date=START_HISTORY_DATE,
        max_date=datetime.date.today(),
    )


def do_stat(sex=None):
    try:
        dba.open_connect()
        oncourt_players.initialize()

        log.info(__file__ + " begining for stat with sex: " + str(sex))
        start_datetime = datetime.datetime.now()

        if sex is None:
            process_sex("wta")
            process_sex("atp")
        else:
            process_sex(sex)

        dba.close_connect()
        log.info(
            "{} finished within {}".format(
                __file__, str(datetime.datetime.now() - start_datetime)
            )
        )
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exc_info=True)
        return 1


# ----------------- доступ к файловым данным (для GUI/today) ------------------
class Personal(object):
    def __init__(self):
        self.val_from_sexnameplr = defaultdict(rl.SizedValue)

    def __initialize(self):
        for sex in ("atp", "wta"):
            dirname = os.path.join(cfg_dir.stat_players_dir(sex), "decided_set")
            for filename in fu.find_files_in_folder(
                dirname, filemask="*", recursive=False
            ):
                subname = os.path.basename(filename).replace(".txt", "")  # 'decided'
                dct = dict_tools.load(
                    filename,
                    createfun=lambda: defaultdict(rl.SizedValue),
                    keyfun=int,
                    valuefun=rl.SizedValue.create_from_text,
                )
                for plrid, val in dct.items():
                    self.val_from_sexnameplr[(sex, subname, plrid)] = val

    def value(self, sex, subname, player_id):
        if not self.val_from_sexnameplr:
            self.__initialize()
        return self.val_from_sexnameplr[(sex, subname, player_id)]


def personal_value(sex, subname, player_id):
    """вернем SizedValue. subname тут не нужен (пусть будет 'decided') и
    только для совместимости с модулями mentality1b, tie_importance_stat"""
    if subname != "decided":
        return rl.SizedValue()
    return personal_value.obj.value(sex, subname, player_id)


personal_value.obj = Personal()


# ------------------ часть ниже относится к GUI -------------------------


def make_player_columninfo(sex, player_id, title, number, value_default="-"):
    def value_function(subname):
        result = personal_value(sex, subname, player_id)
        return co.formated(result.value, result.size, round_digits=3).strip()

    def empty_function(subname):
        return value_default

    return matchstat.ColumnInfo(
        title=title,
        number=number,
        value_fun=value_function if player_id is not None else empty_function,
        value_default=value_default,
    )


def make_generic_columninfo(sex, title, number, value_default="-"):
    return matchstat.ColumnInfo(
        title=title,
        number=number,
        value_fun=lambda k: "0.",
        value_default=value_default,
    )


class PageBuilderDecidedSetLGR(matchstat.PageBuilder):
    """шаблон для закладки c тремя колонками: left_player, generic, right_player"""

    def __init__(self, sex, left_player, right_player):
        super(PageBuilderDecidedSetLGR, self).__init__()
        self.set_data(sex, left_player, right_player)

    def set_data(self, sex, left_player, right_player):
        self.keys = ("decided",)
        self.columns = [
            make_player_columninfo(
                sex, left_player.ident if left_player else None, "left", 1
            ),
            make_generic_columninfo(sex, title="avg", number=2),
            make_player_columninfo(
                sex, right_player.ident if right_player else None, "right", 3
            ),
        ]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp", "both"])
    parser.add_argument("--stat", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    log.initialize(co.logname(__file__), "info", "info")
    args = parse_command_line_args()
    if args.stat:
        import make_stat

        sys.exit(do_stat(sex=None if args.sex == "both" else args.sex))
    else:
        dba.open_connect()
        unittest.main()
        dba.close_connect()
        sys.exit(0)
