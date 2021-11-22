import unittest
import datetime

from head2head import Calc, Item


class CalcTest(unittest.TestCase):
    time_discount = 0.95

    def test_1_to_0(self):
        calc = Calc(
            date=None,
            items=[Item(is_left_win=True, date=None)],
            time_discount=self.time_discount,
        )
        direct = calc.direct(small_size_adjust=False)
        self.assertTrue(direct == 1.0, "direct: {}".format(direct))

        direct_adj = calc.direct(small_size_adjust=True)
        print("1-0 direct_adj", direct_adj)
        self.assertTrue(0.75 < direct_adj <= 0.8, "direct: {}".format(direct_adj))

    def test_1_to_0_dated(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[Item(is_left_win=True, date=datetime.date(2020, 1, 6))],
            time_discount=self.time_discount,
        )
        direct = calc.direct(small_size_adjust=False)
        print("1-0 dated direct", direct)
        self.assertTrue(0.97 < direct < 1.0, "direct: {}".format(direct))

        direct_adj = calc.direct(small_size_adjust=True)
        print("1-0 dated direct_adj", direct_adj)
        self.assertTrue(0.75 < direct_adj <= 0.8, "direct_adj: {}".format(direct_adj))

    def test_0_to_5_dated(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[
                Item(is_left_win=False, date=datetime.date(2016, 1, 6)),
                Item(is_left_win=False, date=datetime.date(2017, 1, 4)),
                Item(is_left_win=False, date=datetime.date(2018, 1, 7)),
                Item(is_left_win=False, date=datetime.date(2019, 1, 5)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
            ],
            time_discount=self.time_discount,
        )
        direct = calc.direct(small_size_adjust=False)
        print("0-5 dated direct", direct)
        self.assertTrue(0.0 < direct < 0.4, "direct: {}".format(direct))

        direct_adj = calc.direct(small_size_adjust=True)
        print("0-5 dated direct_adj", direct_adj)
        self.assertTrue(0.0 < direct < 0.4, "direct: {}".format(direct_adj))

    def test_10_to_1_dated(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[
                Item(is_left_win=True, date=datetime.date(2011, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2012, 1, 4)),
                Item(is_left_win=True, date=datetime.date(2013, 1, 7)),
                Item(is_left_win=True, date=datetime.date(2014, 1, 5)),
                Item(is_left_win=True, date=datetime.date(2015, 1, 9)),
                Item(is_left_win=True, date=datetime.date(2016, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2017, 1, 4)),
                Item(is_left_win=True, date=datetime.date(2018, 1, 7)),
                Item(is_left_win=True, date=datetime.date(2019, 1, 5)),
                Item(is_left_win=True, date=datetime.date(2020, 1, 1)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
            ],
            time_discount=self.time_discount,
        )
        direct = calc.direct(small_size_adjust=False)
        print("10-1 dated direct", direct)
        self.assertTrue(0.88 < direct < 1.0, "direct: {}".format(direct))

    def test_10old_to_3now_dated(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[
                Item(is_left_win=True, date=datetime.date(2010, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 4)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 7)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 5)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 9)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 4)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 7)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 5)),
                Item(is_left_win=True, date=datetime.date(2010, 1, 1)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
            ],
            time_discount=self.time_discount,
        )
        direct = calc.direct()
        print("10old-3now direct", direct)
        self.assertTrue(0.6 < direct < 1.0, "direct: {}".format(direct))

    def test_4_to_3_dated_asc(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[
                Item(is_left_win=True, date=datetime.date(2015, 1, 9)),
                Item(is_left_win=True, date=datetime.date(2016, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2017, 1, 4)),
                Item(is_left_win=True, date=datetime.date(2018, 1, 7)),
                Item(is_left_win=False, date=datetime.date(2019, 1, 5)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 1)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
            ],
            time_discount=self.time_discount,
        )
        direct = calc.direct()
        print("4-3 dated asc direct", direct)
        self.assertTrue(0.5 < direct < 0.7, "direct: {}".format(direct))

    def test_3_to_4_dated_asc(self):
        calc = Calc(
            date=datetime.date(2020, 1, 23),
            items=[
                Item(is_left_win=True, date=datetime.date(2015, 1, 9)),
                Item(is_left_win=True, date=datetime.date(2016, 1, 6)),
                Item(is_left_win=True, date=datetime.date(2017, 1, 4)),
                Item(is_left_win=False, date=datetime.date(2018, 1, 7)),
                Item(is_left_win=False, date=datetime.date(2019, 1, 5)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 1)),
                Item(is_left_win=False, date=datetime.date(2020, 1, 9)),
            ],
            time_discount=self.time_discount,
        )
        direct = calc.direct()
        print("3-4 dated asc direct", direct)
        self.assertTrue(0.35 < direct < 0.5, "direct: {}".format(direct))


if __name__ == '__main__':
    unittest.main()
