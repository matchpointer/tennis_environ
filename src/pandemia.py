# -*- coding=utf-8 -*-
from typing import Dict
from datetime import date

_exhibition_tour_id: Dict[str, int] = {"wta": 13233, "atp": 16726}


def exhib_tour_id(sex: str):
    return _exhibition_tour_id[sex]


def max_prev_rating_date():
    """ после этой даты начисление рейтингов приостановлено """
    return date(2020, 3, 16)


def min_date():
    return date(2020, 4, 1)


def max_date():
    return date(2020, 12, 26)

