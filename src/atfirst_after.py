# -*- coding: utf-8 -*-
import os
import sys
import datetime
import unittest
from collections import defaultdict, namedtuple
import argparse
from contextlib import closing
from typing import List, Set

import log
import cfg_dir
import common as co
import file_utils as fu
import oncourt_players
import score as sc
import dba
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


# AspectType = Union[Literal[ASPECT_ABSENCE], Literal[ASPECT_RETIRED]]

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


class ReadFilesTest(unittest.TestCase):
    def test_read(self):
        pass


PlayerResult = namedtuple("PlayerResult", "id days_ago")


# API
def initialize_day(scheduled_events, date: datetime.date):
    fu.ensure_folder(root_dirname)
    sex_to_idents = _get_events_players(scheduled_events)
    for sex, idents in sex_to_idents.items():
        absence_results: List[PlayerResult] = long_absence_results(idents, sex, date)
        _write_players_results(absence_results, sex, ASPECT_ABSENCE, date)

        retired_results: List[PlayerResult] = after_retired_results(idents, sex, date)
        _write_players_results(retired_results, sex, ASPECT_RETIRED, date)


def _get_events_players(scheduled_events):
    sex_to_idents = defaultdict(set)
    for evt in scheduled_events:
        if evt.level in ("team", "future", "junior"):
            continue
        for match in evt.matches:
            if match.first_player and match.first_player.ident:
                sex_to_idents[evt.sex].add(match.first_player.ident)
            if match.second_player and match.second_player.ident:
                sex_to_idents[evt.sex].add(match.second_player.ident)
    return sex_to_idents


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
               and (fst_plr.NAME_P Not Like '%/%') """.format(
        sex
    )
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


def test_add_features():
    pid = 8696
    features = []
    add_read_aspect_feature(
        features, "wta", ASPECT_RETIRED, datetime.date.today(), pid, "fst"
    )
    print(f"feat len {len(features)}")
    if features:
        print(f"feat {features[0].name} {features[0].value}")


def test_init_day():
    import common_wdriver
    from live import get_events, MatchStatus, skip_levels_work

    oncourt_players.initialize("wta")
    oncourt_players.initialize("atp")

    drv = common_wdriver.wdriver(company_name="FS", headless=True)
    drv.start()
    drv.go_live_page()

    initialize_day(
        get_events(
            drv.page(),
            skip_levels=skip_levels_work(),
            match_status=MatchStatus.scheduled,
        ),
        datetime.date.today(),
    )


if __name__ == "__main__":
    args = parse_command_line_args()
    # if args.stat:
    #     log.initialize(co.logname(__file__, instance=str(args.sex)),
    #                    file_level='debug', console_level='info')
    #     # sys.exit(do_stat())
    # else:
    log.initialize(co.logname(__file__, test=True), "info", "info")
    dba.open_connect()
    test_init_day()
    # unittest.main()
    dba.close_connect()
    sys.exit(0)
