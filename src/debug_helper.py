# -*- coding=utf-8 -*-
from collections import namedtuple
import copy
import datetime


_debug_match_name = "no debug"
DebugMatchData = namedtuple("DebugMatchData", "score ingame left_service")
debug_timed_data_seq = list()  # list of (time, DebugMatchData) # point scores seq
debug_match_feat_dicts = {}  # casename -> feat_dict


def get_debug_match_name():
    return _debug_match_name


def set_debug_match_name(name):
    global _debug_match_name
    _debug_match_name = name


def add_debug_match_feat_dict(casename, features):
    if debug_match_feat_dicts.get(casename) is None:
        debug_match_feat_dicts[casename] = copy.copy(features)


def add_debug_match_data(score, ingame, left_service):
    match_data = DebugMatchData(score, ingame, left_service)
    if not debug_timed_data_seq or (
        debug_timed_data_seq and (debug_timed_data_seq[-1][1] != match_data)
    ):
        debug_timed_data_seq.append((datetime.datetime.now(), match_data))


def debug_match_data_save():
    def get_line():
        if match_data.left_service is None:
            srv = "None"
        elif match_data.left_service:
            srv = "Left"
        else:
            srv = "Right"
        return "{:02d}:{:02d}:{:02d} {} {} {}\n".format(
            dtime.hour,
            dtime.minute,
            dtime.second,
            str(match_data.score),
            match_data.ingame,
            srv,
        )

    filename = "./debug_match_data.txt"
    with open(filename, "w") as fh:
        fh.write(_debug_match_name + "\n")
        for (dtime, match_data) in debug_timed_data_seq:
            fh.write(get_line())

    filename = "./debug_match_feat_dicts.txt"
    with open(filename, "w") as fh:
        fh.write(_debug_match_name + "\n")
        for casename, features in debug_match_feat_dicts.items():
            fh.write(" ----------- case {} -----------------\n".format(casename))
            for feat in features:
                fh.write("{}\n".format(feat.tostring(extended=True)))
