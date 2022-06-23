# -*- coding=utf-8 -*-
"""
module for defining player features: atfirst_after_absence, atfirst_after_retired
"""
import os
import datetime
from collections import defaultdict, namedtuple
import argparse
from contextlib import closing
from typing import List, Set

from loguru import logger as log
import cfg_dir
import common as co
import file_utils as fu
import score as sc
from oncourt import dba
import feature


ROOT = "atfirst_after"
ASPECT_ABSENCE = "absence"
ASPECT_RETIRED = "retired"

# глубина отката в прошлое от сегодня для фиксации 1 стадии ОТСУТСТВИЯ
MIN_ABSENCE_DAYS = 90

# 2 стадия: те кто не был в 1 стадии, долж. БЫТЬ при откате от MIN_ABSENCE_DAYS еще на эти года
# это чтобы не рассматривать новобранцев, а только тех кто приходит сегодня после перерыва
PAST_YEARS = 4

# глубина отката в прошлое от сегодня для фиксации факта снятия
MAX_DAYS_FOR_RETIRED = 30 * 4


root_dirname = cfg_dir.pre_live_dir(ROOT)


def add_read_features(features, sex: str, pid1: int, pid2: int):
    date = datetime.date.today()
    add_read_aspect_feature(features, sex, ASPECT_RETIRED, date, pid1, "fst")
    add_read_aspect_feature(features, sex, ASPECT_RETIRED, date, pid2, "snd")
    add_read_aspect_feature(features, sex, ASPECT_ABSENCE, date, pid1, "fst")
    add_read_aspect_feature(features, sex, ASPECT_ABSENCE, date, pid2, "snd")


def add_read_aspect_feature(
    features, sex: str, aspect: str, date: datetime.date, pid: int, preffix: str
):
    featname = f"{preffix}_{aspect}"
    for plr_res in get_players_results(sex, aspect, date):
        if plr_res.id == pid:
            features.append(feature.RigidFeature(featname, plr_res.days_ago))
            return
    features.append(feature.RigidFeature(featname))


def get_players_results(sex: str, aspect: str, date: datetime.date):
    key = (sex, aspect, date)
    results = get_players_results.cache.get(key)
    if results is None:
        results = _read_players_results(sex, aspect, date)
        get_players_results.cache[key] = results
    return results


get_players_results.cache = dict()  # (sex, aspect, date) -> List[PlayerResult]


PlayerResult = namedtuple("PlayerResult", "id days_ago")


# API
def initialize_day(scheduled_events, date: datetime.date):
    def log_report():
        rpt = ''
        for sx in ('wta', 'atp'):
            rpt += (f'\nsex:{sx} n_plrs:{len(sex_to_idents[sx])}'
                    f' evnames:{sex_to_evnames[sx]}'
                    f' emptyevnames:{sex_to_emptyevnames[sx]}')
        log.info(f'atfirst_after report:\n{rpt}')

    fu.ensure_folder(root_dirname)
    sex_to_idents, sex_to_evnames, sex_to_emptyevnames = _get_events_players(
        scheduled_events)
    for sex, idents in sex_to_idents.items():
        absence_results: List[PlayerResult] = long_absence_results(idents, sex, date)
        _write_players_results(absence_results, sex, ASPECT_ABSENCE, date)

        retired_results: List[PlayerResult] = after_retired_results(idents, sex, date)
        _write_players_results(retired_results, sex, ASPECT_RETIRED, date)
    log_report()


def _get_events_players(scheduled_events):
    sex_to_idents = defaultdict(set)
    sex_to_evnames = defaultdict(set)
    sex_to_emptyevnames = defaultdict(set)
    for evt in scheduled_events:
        if evt.level in ("team", "future", "junior"):
            continue
        n_evt_players = 0
        for match in evt.matches:
            if match.first_player and match.first_player.ident:
                sex_to_idents[evt.sex].add(match.first_player.ident)
                n_evt_players += 1
            if match.second_player and match.second_player.ident:
                sex_to_idents[evt.sex].add(match.second_player.ident)
                n_evt_players += 1
        if n_evt_players > 0:
            sex_to_evnames[evt.sex].add(evt.tour_name.name)
        elif n_evt_players == 0:
            sex_to_emptyevnames[evt.sex].add(evt.tour_name.name)
    return sex_to_idents, sex_to_evnames, sex_to_emptyevnames


def long_absence_results(
    idents: Set[int], sex: str, date: datetime.date
) -> List[PlayerResult]:
    """
    fetch from db. Вернет тех, кто не играл в [date - MIN_ABSENCE_DAYS, date),
         и играл в [date - MIN_ABSENCE_DAYS - PAST_YEARS, date - MIN_ABSENCE_DAYS)
    """
    min_date = date - datetime.timedelta(MIN_ABSENCE_DAYS)
    presence_results = get_presence_results(
        idents, sex, min_date=min_date, max_date=date
    )
    nopresence_idents = idents - {pr.id for pr in presence_results}
    if not nopresence_idents:
        return []  # all idents are present

    past_min_date = min_date - datetime.timedelta(days=365 * PAST_YEARS)
    past_presence_results = get_presence_results(
        nopresence_idents, sex, min_date=past_min_date, max_date=min_date
    )
    return [pr for pr in past_presence_results if pr.id in nopresence_idents]


def after_retired_results(
    idents: Set[int], sex: str, date: datetime.date
) -> List[PlayerResult]:
    """
    fetch from db. Вернет тех, кто retired in [date - MAX_DAYS_FOR_RETIRED, date),
         и после этого не играл
    """
    min_date = date - datetime.timedelta(MAX_DAYS_FOR_RETIRED)
    ret_results = get_presence_results(
        idents, sex, min_date=min_date, max_date=date, with_retired=True
    )

    ret_idents = {r.id for r in ret_results}
    noret_results = get_presence_results(
        ret_idents, sex, min_date=min_date, max_date=date, with_retired=False
    )
    results = []
    for pr in ret_results:
        noret_res = co.find_first(noret_results, lambda r: r.id == pr.id)
        if noret_res.days_ago < pr.days_ago:
            continue  # after retired player was active
        results.append(pr)
    return results


def get_presence_results(
    idents: Set[int],
    sex: str,
    min_date: datetime.date,
    max_date: datetime.date,
    with_retired=False,
) -> List[PlayerResult]:
    """[min_date, max_date) is semiclose range
    :return list where date is latest for [min_date,...,max_date)
    """
    present_pids = set()
    present_results = []
    date_now = datetime.date.today()
    if not dba.initialized():
        dba.open_connect()
    sql = """select games.DATE_G, games.RESULT_G, games.ID1_G, games.ID2_G
             from Tours_{0} AS tours, games_{0} AS games, Players_{0} AS fst_plr
             where games.ID_T_G = tours.ID_T 
               and games.ID1_G = fst_plr.ID_P
               and (tours.NAME_T Not Like '%juniors%')
               and (fst_plr.NAME_P Not Like '%/%') """.format(sex)
    sql += dba.sql_dates_condition(min_date, max_date, dator="games.DATE_G")
    sql += " order by games.DATE_G desc;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (match_dt, score_txt, fst_id, snd_id) in cursor.execute(sql):
            match_date = match_dt.date() if match_dt else None
            if not score_txt:
                continue  # may be it is scheduled match
            date = match_date
            if date is None:
                log.error(
                    f"empty date in {sex} pid1{fst_id} pid2 {snd_id} score:{score_txt}"
                )
                continue
            if with_retired and not sc.Score(score_txt).retired:
                continue
            if not with_retired and fst_id in idents and fst_id not in present_pids:
                present_pids.add(fst_id)
                present_results.append(
                    PlayerResult(id=fst_id, days_ago=(date_now - date).days)
                )
            if snd_id in idents and snd_id not in present_pids:
                present_pids.add(snd_id)
                present_results.append(
                    PlayerResult(id=snd_id, days_ago=(date_now - date).days)
                )
    return present_results


def is_initialized_day(date):
    for sex in ('wta', 'atp'):
        for aspect in (ASPECT_ABSENCE, ASPECT_RETIRED):
            if not os.path.isfile(players_filename(sex, aspect, date)):
                return False
    return True


def _write_players_results(
    player_results: List[PlayerResult], sex: str, aspect: str, date: datetime.date
):
    filename = players_filename(sex, aspect, date)
    with open(filename, "w") as fh:
        for plr_res in player_results:
            fh.write(f"{plr_res.id},{plr_res.days_ago}\n")


def _read_players_results(
    sex: str, aspect: str, date: datetime.date
) -> List[PlayerResult]:
    result = []
    filename = players_filename(sex, aspect, date)
    if not os.path.isfile(filename):
        return result
    with open(filename, "r") as fh:
        for line in fh.readlines():
            if line:
                pid_text, days_ago_text = line.strip().split(",")
                result.append(PlayerResult(int(pid_text), int(days_ago_text)))
    return result


def players_filename(sex, aspect, date):
    return f"{root_dirname}/{co.date_to_str(date)}_{sex}_{aspect}.txt"


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()
