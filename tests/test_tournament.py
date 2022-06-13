# -*- coding=utf-8 -*-
import os
import datetime

import cfg_dir
import tennis_time as tt
import tennis
import tour_name
from tournament import best_of_five, tours_generator, tours_write_file


def test_best_of_five():
    isbo5 = best_of_five(
        date=datetime.date(2020, 9, 3),
        sex="atp",
        tourname=tour_name.TourName("U.S. Open"),
        level="gs",
        rnd=tennis.Round("Third"),
    )
    assert isbo5

    isbo5 = best_of_five(
        date=datetime.date(2018, 9, 3),
        sex="atp",
        tourname=tour_name.TourName("Davis Cup"),
        level="teamworld",
        rnd=tennis.Round("Robin"),
    )
    assert isbo5

    isbo5 = best_of_five(
        date=datetime.date(2020, 9, 3),
        sex="atp",
        tourname=tour_name.TourName("ATP Cup"),
        level="teamworld",
        rnd=tennis.Round("Robin"),
    )
    assert not isbo5

    isbo5 = best_of_five(
        date=datetime.date(2020, 9, 3),
        sex="atp",
        tourname=tour_name.TourName("Davis Cup"),
        level="teamworld",
        rnd=tennis.Round("Robin"),
    )
    assert not isbo5

    isbo5 = best_of_five(
        date=datetime.date(2020, 9, 3),
        sex="atp",
        tourname="Davis Cup",
        level="teamworld",
        rnd=tennis.Round("Robin"),
    )
    assert not isbo5

    isbo5 = best_of_five(
        date=datetime.date(2018, 9, 3),
        sex="atp",
        tourname="Davis Cup",
        level="team",
        rnd=tennis.Round("Robin"),
    )
    assert not isbo5


def test_write_tours(get_dba):
    sex = "atp"
    # few weeks in past:
    min_date = (tt.past_monday_date(datetime.date.today()) -
                datetime.timedelta(days=7*4))
    max_date = min_date + datetime.timedelta(days=7*2)

    tours = list(
        tours_generator(
            sex,
            todaymode=False,
            min_date=min_date,
            max_date=max_date,
            with_paired=False,
            with_mix=False,
            rnd_detailing=True,
        )
    )
    assert tours is not None and len(tours) > 0
    if tours:
        dirname = cfg_dir.log_dir()
        filename = os.path.join(dirname, f'test_{sex}_tours.txt')
        tours_matches_ordering(tours)
        tours_write_file(tours, filename)


def tours_matches_ordering(tours):
    for tour in tours:
        matches_ordering(tour)


def matches_ordering(tour):
    for rnd in tour.matches_from_rnd.keys():
        tour.matches_from_rnd[rnd].sort(
            key=lambda m: m.first_player.name + m.second_player.name
        )