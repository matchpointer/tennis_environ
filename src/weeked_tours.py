from collections import OrderedDict
import datetime
from typing import Tuple, List, Dict, OrderedDict as OrderedDictType

import cfg_dir
from loguru import logger as log
import tournament as trmt
import tennis_time as tt

Sex_YearWeeknum_ToursList = Dict[
    str,  # sex
    OrderedDictType[
        Tuple[
            int,  # year
            int  # weeknum
        ],
        List[trmt.Tournament]
    ]
]

__sex_dict: Sex_YearWeeknum_ToursList = {}


def _default_inited_dict(min_date, max_date):
    """return OrderedDict where values are inited with []"""
    dct = OrderedDict()
    first_monday = tt.past_monday_date(min_date)
    if max_date is None:
        max_date = tt.past_monday_date(datetime.date.today()) + datetime.timedelta(
            days=7
        )
    last_monday = tt.past_monday_date(max_date)
    monday = first_monday
    while monday <= last_monday:
        year_weeknum = tt.get_year_weeknum(monday)
        dct[year_weeknum] = []
        monday = monday + datetime.timedelta(days=7)
    return dct


def initialize_sex(
    sex,
    min_date,
    max_date,
    split=False,
    with_today=False,
    with_paired=False,
    with_mix=False,
    rnd_detailing=False,
    with_ratings=False,
    with_bets=False,
    with_stat=False,
    with_pers_det=False,
):
    if sex in __sex_dict:
        log.warning("reinit {} weeked tours".format(sex))
        del __sex_dict[sex]
    __sex_dict[sex] = _default_inited_dict(min_date, max_date)
    if split:
        assert not with_stat, "with_stat is not compatible with split"
        for tour in trmt.week_splited_tours(
            sex,
            todaymode=None if with_today else False,
            min_date=min_date,
            max_date=max_date,
            time_reverse=False,
            with_paired=with_paired,
            with_mix=with_mix,
            rnd_detailing=rnd_detailing,
            with_bets=with_bets,
            with_ratings=with_ratings,
            with_pers_det=with_pers_det,
        ):
            __sex_dict[sex][tour.year_weeknum].append(tour)
    else:
        if with_today:
            initialize_sex_with_today(
                sex,
                min_date,
                max_date,
                with_paired=with_paired,
                with_mix=with_mix,
                rnd_detailing=rnd_detailing,
                with_ratings=with_ratings,
                with_bets=with_bets,
                with_stat=with_stat,
                with_pers_det=with_pers_det,
            )
        else:
            for tour in trmt.tours_generator(
                sex,
                todaymode=False,
                min_date=min_date,
                max_date=max_date,
                time_reverse=False,
                with_paired=with_paired,
                with_mix=with_mix,
                rnd_detailing=rnd_detailing,
                with_bets=with_bets,
                with_stat=with_stat,
                with_ratings=with_ratings,
                with_pers_det=with_pers_det,
            ):
                __sex_dict[sex][tour.year_weeknum].append(tour)


def initialize_sex_with_today(
    sex,
    min_date,
    max_date,
    with_paired,
    with_mix,
    rnd_detailing,
    with_ratings,
    with_bets,
    with_stat,
    with_pers_det,
):
    full_tours = list(
        trmt.tours_generator(
            sex,
            todaymode=False,
            min_date=min_date,
            max_date=max_date,
            time_reverse=False,
            with_paired=with_paired,
            with_mix=with_mix,
            rnd_detailing=False,
            with_bets=with_bets,
            with_stat=with_stat,
            with_ratings=with_ratings,
            with_pers_det=with_pers_det,
        )
    )
    td_tours = list(
        trmt.tours_generator(
            sex,
            todaymode=True,
            min_date=min_date,
            max_date=max_date,
            time_reverse=False,
            with_paired=False,
            with_mix=False,
            rnd_detailing=False,
            with_bets=with_bets,
            with_stat=False,
            with_ratings=with_ratings,
            with_pers_det=with_pers_det,
        )
    )
    trmt.complete_with_today(full_tours, td_tours, with_unscored_matches=True)
    full_tours.sort(key=lambda t: t.date)
    for tour in full_tours:
        if rnd_detailing:
            tour.rounds_detailing()
        __sex_dict[sex].setdefault(tour.year_weeknum, []).append(tour)


def tours(sex, year_weeknum):
    return __sex_dict[sex].get(year_weeknum, [])


def year_weeknum_iter(sex):
    return iter(__sex_dict[sex].keys())


use_tail_tours_cache = False


def tail_tours(sex, tail_weeks=3):
    if use_tail_tours_cache and (sex, tail_weeks) in _tail_tours_cache:
        return _tail_tours_cache[(sex, tail_weeks)]
    result = []
    passed_weeks = 0
    for idx, ywn in enumerate(sorted(year_weeknum_iter(sex), reverse=True)):
        if idx == 0 and not __sex_dict[sex][ywn]:
            continue
        result.extend(__sex_dict[sex][ywn])
        passed_weeks += 1
        if passed_weeks >= tail_weeks:
            break
    if use_tail_tours_cache:
        _tail_tours_cache[(sex, tail_weeks)] = result
    return result


SexTailweeks_ToursList = Dict[
    Tuple[
        str,  # sex
        int  # tail weeks num
    ],
    List[trmt.Tournament]
]

_tail_tours_cache: SexTailweeks_ToursList = dict()


def all_tours(sex):
    result = []
    for ywn in year_weeknum_iter(sex):
        result.extend(__sex_dict[sex][ywn])
    return result


def dump_tail_tours(sex, tail_weeks, filename=None):
    tailtours = tail_tours(sex=sex, tail_weeks=tail_weeks)
    if not tailtours:
        log.error("empty {} tailtours, tailweeks: {}".format(sex, tail_weeks))
    else:
        if filename is None:
            filename = f"{cfg_dir.log_dir()}/dump_tailtours_{sex}_{tail_weeks}.txt"
        trmt.tours_write_file(tailtours, filename)
