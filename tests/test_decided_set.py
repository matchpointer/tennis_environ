# -*- coding=utf-8 -*-
import unittest
import datetime
import copy

from oncourt import dba
from decided_set import player_winloss, initialize_results, results_dict


class WtaWinlossPlayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        max_date = datetime.date.today() - datetime.timedelta(days=7)
        initialize_results(
            sex="wta",
            min_date=datetime.date(year=2013, month=1, day=1),
            max_date=max_date,
        )

    def test_date_none(self):
        self.assertFalse(None in list(results_dict["wta"].keys()))

    def test_date_order(self):
        dates = copy.copy(list(results_dict["wta"].keys()))
        dates2 = copy.copy(dates)
        dates2.sort()
        self.assertEqual(dates2, dates)

    def test_player_winloss(self):
        sharapova = 503
        wl = player_winloss(sex="wta", ident=sharapova)
        self.assertTrue(wl.ratio > 0.65)
        self.assertTrue(wl.size > 10)

        serenawilliams = 241
        wl = player_winloss(sex="wta", ident=serenawilliams)
        self.assertTrue(wl.ratio > 0.65)
        self.assertTrue(wl.size > 8)


if __name__ == '__main__':
    dba.open_connect()
    unittest.main()
    dba.close_connect()
