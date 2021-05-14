# -*- coding: utf-8 -*-
import datetime
import doctest
import unittest

import common as co
import log


def date_generator(start, end):
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def week_date_generator(start_date, end_date):
    assert start_date <= end_date, "bad init weeks {} > {}".format(start_date, end_date)
    current = past_monday_date(start_date)
    finish = past_monday_date(end_date)
    while current <= finish:
        yield current
        current += datetime.timedelta(days=7)


def future_monday_date(in_date):
    """
    Возвращает дату ближайшего к in_date грядущего понедельника.
    Если in_date понедельник, то возвращается in_date.
    """
    weekday = in_date.isoweekday()
    if weekday == 1:
        result_date = in_date
    elif weekday in (2, 3, 4, 5, 6, 7):
        result_date = in_date + datetime.timedelta(days=8 - weekday)
    return result_date


def past_monday_date(in_date):
    """
    Возвращает дату ближайшего к in_date прошедшего понедельника.
    Если in_date понедельник, то возвращается in_date.
    """
    weekday = in_date.isoweekday()
    if weekday == 1:
        result_date = in_date
    elif weekday in (2, 3, 4, 5, 6, 7):
        result_date = in_date - datetime.timedelta(days=weekday - 1)
    return result_date


def current_week_date(in_date):
    now_date = datetime.datetime.now().date()
    cur_week_monday = past_monday_date(now_date)
    next_week_monday = cur_week_monday + datetime.timedelta(days=7)
    return cur_week_monday <= in_date < next_week_monday


def now_minus_history_days(history_days):
    collect_timedelta = datetime.timedelta(days=history_days)
    return past_monday_date(datetime.date.today()) - collect_timedelta


def formated_datetime(dt):
    return "%02d.%02d.%04d %02d:%02d:%02d" % (
        dt.day,
        dt.month,
        dt.year,
        dt.hour,
        dt.minute,
        dt.second,
    )


class DateTest(unittest.TestCase):
    def test_future_monday_date(self):
        monday_date = datetime.date(year=2013, month=3, day=4)
        self.assertEqual(future_monday_date(monday_date), monday_date)

        past_friday_date = datetime.date(year=2013, month=3, day=1)
        self.assertEqual(future_monday_date(past_friday_date), monday_date)

    def test_past_monday_date(self):
        monday_date = datetime.date(year=2013, month=3, day=4)
        self.assertEqual(past_monday_date(monday_date), monday_date)

        wednesday_date = datetime.date(year=2013, month=3, day=7)
        self.assertEqual(past_monday_date(wednesday_date), monday_date)


def get_monday(year, weeknum):
    """return begin date for given year, weeknum

    >>> get_monday(year=2019, weeknum=1)
    datetime.date(2018, 12, 31)
    >>> get_monday(year=2019, weeknum=52)
    datetime.date(2019, 12, 23)
    >>> get_monday(year=1998, weeknum=53)
    datetime.date(1998, 12, 28)
    >>> get_monday(year=1999, weeknum=1)
    datetime.date(1999, 1, 4)
    """
    beg_date = datetime.date(year, 1, 1)
    if beg_date.weekday() > 3:
        beg_date = beg_date + datetime.timedelta(7 - beg_date.weekday())
    else:
        beg_date = beg_date - datetime.timedelta(beg_date.weekday())
    return beg_date + datetime.timedelta(days=(weeknum - 1) * 7)


def get_year_weeknum(tour_date):
    """return (year, weeknum)

    >>> get_year_weeknum(datetime.date(2014,9,8)) # monday
    (2014, 37)
    >>> get_year_weeknum(datetime.date(2014,9,14)) # sunday
    (2014, 37)
    >>> get_year_weeknum(datetime.date(2014,9,7)) # prev sunday
    (2014, 36)
    >>> get_year_weeknum(datetime.date(2017,1,2)) # monday
    (2017, 1)
    >>> get_year_weeknum(datetime.date(2017,1,8)) # sunday
    (2017, 1)
    >>> get_year_weeknum(datetime.date(2017,1,9)) # next monday
    (2017, 2)
    """
    return tour_date.isocalendar()[0], tour_date.isocalendar()[1]


def year_weeknum_diff(start, finish):
    """
    start, finish - входные пары (year, weeknum).
    Возврашается разность в неделях (возможно отрицательная).
    """
    if start > finish:
        return -year_weeknum_diff(finish, start)
    year, wn_sum = start[0], 0
    while year < finish[0]:
        wn_sum += weeknums_in_year(year)
        year += 1
    return wn_sum + finish[1] - start[1]


def year_weeknum_prev(year_weeknum):
    """
    Возврашается предыдущее по хронологии значение для year_weeknum.
    """
    res_year, res_weeknum = year_weeknum
    if res_weeknum <= 1:
        res_year = res_year - 1
        res_weeknum = weeknums_in_year(res_year)
    else:
        res_weeknum = res_weeknum - 1
    return res_year, res_weeknum


def year_weeknum_reversed(start_year_weeknum, max_weeknum_dist):
    """generator begins from start_year_weeknum, and goes to backward"""
    year_weeknum = start_year_weeknum
    while year_weeknum_diff(year_weeknum, start_year_weeknum) <= max_weeknum_dist:
        yield year_weeknum
        year_weeknum = year_weeknum_prev(year_weeknum)


def weeknums_in_year(year):
    last_days = (
        datetime.date(year, 12, 28),
        datetime.date(year, 12, 29),
        datetime.date(year, 12, 30),
        datetime.date(year, 12, 31),
    )
    max_weeknum = 52
    for d in last_days:
        cal = d.isocalendar()
        if cal[0] == year and cal[1] > max_weeknum:
            max_weeknum = cal[1]
    return max_weeknum


class YearWeeknumTest(unittest.TestCase):
    def test_year_weeknum_rev(self):
        lst = list(
            year_weeknum_reversed(start_year_weeknum=(2010, 30), max_weeknum_dist=14)
        )
        self.assertEqual(len(lst), 15)

    def test_year_weeknum_diff(self):
        self.assertEqual(year_weeknum_diff(start=(2012, 52), finish=(2013, 1)), 1)
        self.assertEqual(year_weeknum_diff(start=(2012, 51), finish=(2012, 52)), 1)
        self.assertEqual(year_weeknum_diff(start=(2012, 1), finish=(2012, 52)), 51)
        self.assertEqual(year_weeknum_diff(start=(2012, 1), finish=(2013, 1)), 52)
        self.assertEqual(year_weeknum_diff(start=(2012, 2), finish=(2013, 3)), 53)
        self.assertEqual(year_weeknum_diff(start=(2013, 2), finish=(2012, 1)), -53)

        # 53 in 1998
        self.assertEqual(year_weeknum_diff(start=(1998, 1), finish=(1999, 1)), 53)

        # 53 in 1998
        self.assertEqual(year_weeknum_diff(start=(1997, 52), finish=(1999, 1)), 54)

    def test_year_weeknum(self):
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1997, month=1, day=1)), (1997, 1)
        )
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1997, month=12, day=30)), (1998, 1)
        )

        # thuesday
        self.assertEqual(
            get_year_weeknum(datetime.date(year=2013, month=1, day=1)), (2013, 1)
        )

        # such years was with Adelaide started at wednesday
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1992, month=1, day=1)), (1992, 1)
        )
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1997, month=1, day=1)), (1997, 1)
        )

        # monday
        self.assertEqual(
            get_year_weeknum(datetime.date(year=2008, month=12, day=29)), (2009, 1)
        )
        self.assertEqual(
            get_year_weeknum(datetime.date(year=2012, month=12, day=31)), (2013, 1)
        )

        # monday. At this date first atp tours started from weeknum=2
        # (before this there are no real atp events).
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1998, month=1, day=5)), (1998, 2)
        )

        # monday. ISO gives: 1998, 53. There are no real tours started at this date.
        self.assertEqual(
            get_year_weeknum(datetime.date(year=1998, month=12, day=28)), (1998, 53)
        )


class YearWeekProgress(object):
    def __init__(self, head=""):
        self.last_datetime = None  # for pass week process time
        self.year_weeknum = None, None
        self.head = head
        self.new_week_begining = False
        self.passed_weeks = 0

    def put_tour(self, tour):
        year_week = tour.year_weeknum
        if year_week != self.year_weeknum:
            if self.year_weeknum != (None, None):
                self.passed_weeks += 1
            now = datetime.datetime.now()
            if self.last_datetime is None:
                timedelta_str = ""
            else:
                timedelta_str = str(now - self.last_datetime)
            self.year_weeknum = year_week
            self.last_datetime = now
            log.info(
                "{} {} {} time: {}".format(
                    self.head, tour.sex, year_week, timedelta_str
                )
            )
            self.new_week_begining = True
        else:
            self.new_week_begining = False


class RaceTime(object):
    def __init__(self, collect_timedelta, min_date, max_date=None):
        self.collect_timedelta = collect_timedelta
        self.min_date = min_date
        self.max_date = max_date
        self.min_active_date = min_date + collect_timedelta
        if max_date is not None:
            assert self.min_active_date < max_date, "min_active_date >= max_date"

    def is_active_date(self, date):
        if date is None:
            return False
        if self.max_date is None:
            return self.min_active_date <= date
        return self.min_active_date <= date < self.max_date


class MatchBeforeCondition(object):
    def __init__(self, before_date, before_rnd):
        self.before_date = before_date
        self.before_rnd = before_rnd

    def admit(self, match):
        if match.date is not None:
            return match.date < self.before_date
        else:
            return match.rnd < self.before_rnd


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", None)
    doctest.testmod()
    unittest.main()
