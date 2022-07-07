# -*- coding=utf-8 -*-
import unittest
import datetime

import common as co
from oncourt import dbcon
from bet_coefs import MARATHON_ID, initialize, db_offers


class SimpleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dbcon.open_connect()
        initialize(sex="wta", bettor_id=MARATHON_ID, min_date=datetime.date(2017, 1, 1))
        initialize(sex="atp", bettor_id=MARATHON_ID, min_date=datetime.date(2018, 1, 1))

    def test_tsitsipas_dzumhur_toronto_2018(self):
        db_offer = self.find_dboffer(
            sex="atp",
            tour_id=14752,
            fst_plr_id=30470,
            snd_plr_id=13447,
            bettor_id=MARATHON_ID,
        )
        self.assertTrue(db_offer is not None)

    def test_diyas_kvitova_newhaven_2018(self):
        db_offer = self.find_dboffer(
            sex="wta",
            tour_id=11794,
            fst_plr_id=10394,
            snd_plr_id=8234,
            bettor_id=MARATHON_ID,
        )
        self.assertTrue(db_offer is not None)

    def test_wang_muguruza_hongkong_2018(self):
        db_offer = self.find_dboffer(
            sex="wta",
            tour_id=11805,
            fst_plr_id=8531,
            snd_plr_id=10633,
            bettor_id=MARATHON_ID,
        )
        self.assertTrue(db_offer is not None)
        self.assertTrue(db_offer.win_coefs)

    @staticmethod
    def find_dboffer(sex, tour_id, fst_plr_id, snd_plr_id, bettor_id):
        dboffers = db_offers(sex, bettor_id=bettor_id)
        predicate = lambda o: o.tour_id == tour_id and (
            (o.first_player_id == fst_plr_id and o.second_player_id == snd_plr_id)
            or (o.first_player_id == snd_plr_id and o.second_player_id == fst_plr_id)
        )
        db_offer = co.find_first(dboffers, predicate)
        if db_offer is not None:
            if db_offer.win_coefs:
                if (
                    db_offer.first_player_id == snd_plr_id
                    and db_offer.second_player_id == fst_plr_id
                ):
                    db_offer.flip()
            return db_offer


if __name__ == '__main__':
    unittest.main()
    dbcon.close_connect()
