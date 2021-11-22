import datetime
import unittest

from tennis_time import (
    future_monday_date,
    past_monday_date,
    year_weeknum_reversed,
    get_year_weeknum,
    year_weeknum_diff,
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

        tuesday_date = datetime.date(year=2013, month=3, day=6)
        self.assertEqual(past_monday_date(tuesday_date), monday_date)

        sunday_date = datetime.date(year=2013, month=3, day=10)
        self.assertEqual(past_monday_date(sunday_date), monday_date)


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


if __name__ == "__main__":
    unittest.main()
