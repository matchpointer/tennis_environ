# -*- coding=utf-8 -*-
import unittest
import datetime

from loguru import logger as log
import common as co
import dba
from ratings_std import top_players_pts_list, _sex_dict, get_rank, get_pts, initialize
from tennis import Player


class RatingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dba.open_connect()
        initialize(sex=None, min_date=datetime.date(2003, 1, 6))

    @staticmethod
    def is_all_handred_even(iterable):
        return all(((n % 100) == 0 for n in iterable))

    @staticmethod
    def write_pts_csv(sex, date, pts_list):
        filename = "./{}_{:4d}_{:02d}_{:02d}_pts.csv".format(
            sex, date.year, date.month, date.day
        )
        with open(filename, "w") as fh:
            line = ",".join([str(i) for i in pts_list])
            fh.write(line + "\n")

    def test_write_pts_csv(self):
        dates = [
            datetime.date(2010, 6, 7),
            datetime.date(2014, 1, 13),
            datetime.date(2015, 5, 18),
            datetime.date(2019, 9, 30),
        ]
        for date in dates:
            for sex in ("wta", "atp"):
                pts_list = top_players_pts_list(sex, date, top=500)
                is_100_even = self.is_all_handred_even(pts_list)
                if sex == "wta":
                    self.assertTrue(is_100_even)
                    self.write_pts_csv(sex, date, [n // 100 for n in pts_list])
                else:
                    print("atp is_100_even", is_100_even)
                    self.write_pts_csv(sex, date, pts_list)
                self.assertTrue(len(pts_list) >= 500)

    def test_common_wta(self):
        sex = "wta"
        sex_dict = _sex_dict[sex]
        dates = list(sex_dict.keys())
        n_rtg = 0
        for date in dates:
            n_rtg += len(sex_dict[date])
        self.assertTrue(len(dates) >= 40)

    def test_get_rank_wta(self):
        plr = Player(ident=14364, name="Angeliki Kairi", cou="GRE")
        pos = get_rank("wta", plr.ident, datetime.date(2014, 7, 21))
        self.assertEqual(pos, None)

        plr = Player(ident=14010, name="Beatrice Cedermark", cou="SWE")
        pos = get_rank("wta", plr.ident, datetime.date(2014, 7, 21))
        self.assertEqual(pos, 723)

        plr = Player(ident=431, name="Vera Zvonareva", cou="RUS")
        pos = get_rank("wta", plr.ident, datetime.date(2012, 5, 28))
        self.assertEqual(pos, 11)
        pts = get_pts("wta", plr.ident, datetime.date(2012, 5, 28))
        self.assertEqual(pts, 344000)

        pos = get_rank("wta", plr.ident, datetime.date(2003, 1, 13))
        self.assertEqual(pos, 43)

        plr = Player(ident=7574, name="Petra Martic", cou="CRO")
        pos = get_rank("wta", plr.ident, datetime.date(2020, 8, 3))
        self.assertEqual(pos, 15)

    def test_get_rank_atp(self):
        date = datetime.date(2014, 7, 21)

        plr = Player(ident=21812, name="Rodrigo Arus", cou="URU")
        pos = get_rank("atp", plr.ident, date)
        self.assertEqual(pos, None)

        plr = Player(ident=13962, name="Valentin Florez", cou="ARG")
        pos = get_rank("atp", plr.ident, date)
        self.assertEqual(pos, 689)


if __name__ == "__main__":
    unittest.main()
