"""
    info about lefties taken from: http://tennisabstract.com/reports/leftyRankings.html
"""

from collections import defaultdict
import datetime
from contextlib import closing
import argparse

import csv

import log
import cfg_dir
import common as co
import tennis_time as tt
import dba
import tennis
import file_utils as fu

__players_from_sex = defaultdict(set)


def dict_filename(sex):
    return "{}/oncourt-{}.json".format(cfg_dir.oncourt_players_dir(), sex)


def initialize(sex=None, yearsnum=2):
    min_date = tt.past_monday_date(datetime.date.today()) - datetime.timedelta(
        days=int(365.25 * yearsnum)
    )

    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        dct = fu.json_load_intkey_dict(dict_filename(sex))
        read_players_from_db(sex, __players_from_sex[sex], min_date, dct)


def players(sex):
    """loaded players"""
    return __players_from_sex[sex]


def get_player(sex, pid):
    """convenient for debugs/tests/experiments"""
    return co.find_first(players(sex), lambda p: p.ident == pid)


def read_players_from_db(sex, players_set, min_date, players_ext_dct):
    """fill players_set"""

    def add_player(pid, name, country):
        ext_dct = players_ext_dct.get(pid)
        if ext_dct is not None:
            players_set.add(
                tennis.Player(
                    pid,
                    name,
                    cou=country,
                    lefty=ext_dct.get("lefty"),
                    disp_names=ext_dct.get("disp_names"),
                )
            )
        else:
            players_set.add(tennis.Player(pid, name, country))

    sql = """
    select plr_left.ID_P,  plr_left.NAME_P,  plr_left.COUNTRY_P,
           plr_right.ID_P, plr_right.NAME_P, plr_right.COUNTRY_P
    from Games_{0} AS games, Tours_{0} AS tours, 
         Players_{0} AS plr_left, Players_{0} AS plr_right
    where (games.ID_T_G = tours.ID_T)
      and games.ID1_G = plr_left.ID_P 
      and games.ID2_G = plr_right.ID_P
      and games.RESULT_G IS NOT NULL
      and (tours.NAME_T Not Like '%juniors%')
      and (tours.NAME_T Not Like '%Wildcard%')
      and (plr_left.NAME_P Not Like '%/%')
      and (plr_right.NAME_P Not Like '%/%')
      and tours.DATE_T >= {1};""".format(
        sex, dba.msaccess_date(min_date)
    )
    with closing(dba.get_connect().cursor()) as cursor:
        for (
            plr_left_id,
            plr_left_name,
            plr_left_country,
            plr_right_id,
            plr_right_name,
            plr_right_country,
        ) in cursor.execute(sql):
            add_player(plr_left_id, plr_left_name, plr_left_country)
            add_player(plr_right_id, plr_right_name, plr_right_country)


def actual_players_id(sex):
    if not actual_players_id.cache[sex]:
        actual_players_id.cache[sex] = _read_actual_players_id(sex)
    return actual_players_id.cache[sex]


actual_players_id.cache = defaultdict(list)


def write_actual_players(sex, yearsnum=1.8):
    dba.open_connect()
    initialize(sex, yearsnum=yearsnum)
    with open(_actual_players_filename(sex), "w") as fh:
        fh.write(",".join([str(p.ident) for p in players(sex)]))
    dba.close_connect()


def _actual_players_filename(sex):
    return "{}/actual_players-{}.csv".format(cfg_dir.oncourt_players_dir(), sex)


def _read_actual_players_id(sex):
    with open(_actual_players_filename(sex)) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            return [int(i) for i in row]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_actual_players", action="store_true")
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.make_actual_players:
        log.initialize(co.logname(__file__), "debug", "debug")
        if args.sex in ("wta", None):
            write_actual_players(sex="wta")
            log.info("wta actual players done")
        if args.sex in ("atp", None):
            write_actual_players(sex="atp")
            log.info("atp actual players done")
