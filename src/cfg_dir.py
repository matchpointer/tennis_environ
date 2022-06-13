# -*- coding=utf-8 -*-
import os

import config_file as cf


def oncourt_exe_dir():
    return os.environ.get("ONCOURTPATH")


# ----------------------------------- players ------------------------------------
def stat_players_dir(sex):
    return cf.getval("dirs", sex + "_stat_players")


def stat_players_total_dir(sex):
    return os.path.join(stat_players_dir(sex), "total")


def stat_players_handicap_dir(sex):
    return os.path.join(stat_players_dir(sex), "handicap")


# ----------------------------------- tours ------------------------------------
def stat_tours_root_dir(sex):
    return cf.getval("dirs", sex + "_stat_tours")


def stat_tours_total_dir(sex):
    return os.path.join(stat_tours_root_dir(sex), "total")


def stat_tours_sets_dir(sex, fun_name=None):
    result = os.path.join(stat_tours_root_dir(sex), "sets")
    if os.path.isdir(result) and fun_name:
        return os.path.join(result, fun_name)
    else:
        return result


# --------------------------------- misc -------------------------------
def log_dir():
    return cf.getval("dirs", "log")


def unit_test_data_dir():
    return cf.getval("dirs", "unit_test_data")


def stat_misc_dir(sex):
    return cf.getval("dirs", sex + "_stat_misc")


def bin_dir():
    return cf.getval("dirs", "bin")


def pre_live_dir(name):
    return os.path.join(bin_dir(), "pre_live", name)


def sounds_dir():
    return os.path.join(bin_dir(), "sounds")


def oncourt_players_dir():
    return cf.getval("dirs", "oncourt_players")


def ratings_dir():
    return cf.getval("dirs", "ratings")


def oncourt_sql_dir():
    return cf.getval("dirs", "oncourt_sql")


def interrupt_dir():
    return cf.getval("dirs", "interrupt")


def analysis_data_dir():
    return cf.getval("dirs", "analysis_data")


def analysis_data_file(sex, typename):
    return os.path.join(analysis_data_dir(), sex + "_" + typename + "_data.csv")


def detailed_score_dir():
    return os.path.join(bin_dir(), "detailed_score_db")


def predicts_db_dir():
    return os.path.join(bin_dir(), "predicts_db")
