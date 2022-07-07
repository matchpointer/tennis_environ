# -*- coding=utf-8 -*-
from collections import defaultdict
from contextlib import closing
from typing import Tuple, Optional, DefaultDict
import datetime

import common as co
import oncourt.sql
from oncourt import dbcon

""" access to oncourt ratings (since 2003.01.06)
"""

SexDate_Pid_Rating = DefaultDict[
    str,  # sex
    DefaultDict[
        datetime.date,  # order by desc
        DefaultDict[
            int,  # player_id
            Tuple[
                Optional[int],  # rating_pos
                Optional[float]  # rating_pts
            ]
        ]
    ]
]

_sex_dict: SexDate_Pid_Rating = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: (None, None)))
)


def clear(sex):
    if sex is None:
        _sex_dict.clear()
    else:
        key_list = list(_sex_dict.keys())
        for key in key_list:
            if key == sex:
                _sex_dict[sex].clear()
                del _sex_dict[sex]


def initialize(sex=None, min_date=None, max_date=None):
    if sex in ("wta", None):
        if "wta" in _sex_dict:
            _sex_dict["wta"].clear()
        __initialize_sex("wta", min_date=min_date, max_date=max_date)

    if sex in ("atp", None):
        if "atp" in _sex_dict:
            _sex_dict["atp"].clear()
        __initialize_sex("atp", min_date=min_date, max_date=max_date)


def initialized():
    return len(_sex_dict) > 0


def get_rank(sex, player_id, date):
    return _get_pos_pts(sex, player_id, date)[0]


def get_pts(sex, player_id, date):
    return _get_pos_pts(sex, player_id, date)[1]


def _get_pos_pts(sex, player_id, date):
    valid_date, data_from_pid = __date_entry(sex, date)
    if valid_date is None:
        return None, None
    return data_from_pid[player_id]


default_min_diff = 260


def compare_rank(first_rating, second_rating, min_diff=default_min_diff):
    """
    Cравнение двух рейтинговых позиций с допуском min_diff.
    дает  LT если first позиция выше (значение first меньше),
          EQ если равны с допуском min_diff,
          GT если first ниже (значение first больше).
    """
    left_rating = first_rating if first_rating is not None else 950  # max_pos is 900
    right_rating = second_rating if second_rating is not None else 950  # max_pos is 900
    return co.value_compare(left_rating, right_rating, min_diff)


def __date_entry(sex, date):
    """return date, data_from_pid"""
    ratings_dct = _sex_dict[sex]
    data_from_pid = ratings_dct.get(date, None)
    if data_from_pid is not None:
        return date, data_from_pid
    for d in ratings_dct:  # dates are ordered by desc
        if d < date:
            return d, ratings_dct[d]
    return None, None


def __initialize_sex(sex, min_date=None, max_date=None):
    query = """select DATE_R, ID_P_R, POS_R, POINT_R 
             from Ratings_{} """.format(
        sex
    )
    dates_cond = oncourt.sql.sql_dates_condition(min_date, max_date=max_date, dator="DATE_R")
    if dates_cond:
        query += """
               where 1 = 1 {} """.format(
            dates_cond
        )
    query += """
           order by DATE_R desc;"""
    with closing(dbcon.get_connect().cursor()) as cursor:
        for (dtime, player_id, rating_pos, rating_pts) in cursor.execute(query):
            _sex_dict[sex][dtime.date()][player_id] = (rating_pos, rating_pts)


def top_players_id_list(sex, top):
    """at max date which initialized"""
    from operator import itemgetter

    maxdate = list(_sex_dict[sex].keys())[0]
    pos_from_pid = _sex_dict[sex][maxdate]
    pos_pid_lst = [(pos, pid) for pid, (pos, _) in pos_from_pid.items() if pos <= top]
    pos_pid_lst.sort(key=itemgetter(0))
    return [pid for (_, pid) in pos_pid_lst]


def top_players_pts_list(sex, date, top=500):
    """:returns list of pts"""
    pos_from_pid = _sex_dict[sex][date]
    pts_lst = [
        pts for _pid, (pos, pts) in pos_from_pid.items() if pos <= top and pts > 0
    ]
    pts_lst.sort(reverse=False)
    return pts_lst
