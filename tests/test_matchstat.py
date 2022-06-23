# -*- coding=utf-8 -*-
import unittest
import datetime

from oncourt import dba
import tennis_time as tt
from stat_cont import WinLoss
from tennis import Player, Round
from matchstat import (
    get,
    initialize,
    generic_surface_trans_coef,
    generic_result_value,
    generic_result_dict,
    _matchstates,
)


class MatchStatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize(
            sex="wta",
            min_date=datetime.date(2010, 1, 1),
            max_date=datetime.date(2016, 12, 30),
        )

    def test_matchstates(self):
        def valid_stat(mstat, fun_name):
            method = getattr(mstat, fun_name)
            ret = method()
            if ret is None:
                return False
            fst_obj, snd_obj = ret
            return fst_obj is not None and snd_obj is not None

        sex = "wta"
        mstates = _matchstates(sex)
        self.assertTrue(len(mstates) > 100)

        full_mstates = [ms for ms in mstates if ms.is_total_points_won()]

        ok_mstates_fsi = [
            ms for ms in full_mstates if valid_stat(ms, "first_service_in")
        ]
        ok_mstates_sw = [ms for ms in full_mstates if valid_stat(ms, "service_win")]
        ok_mstates_rw = [ms for ms in full_mstates if valid_stat(ms, "receive_win")]

    def test_get_puig_sharapova(self):
        plr1 = Player(ident=10222, name="Monica Puig", cou="PUR")
        plr2 = Player(ident=503, name="Maria Sharapova", cou="RUS")
        mstat = get(
            sex="wta",
            tour_id=9125,
            rnd=Round("Second"),
            first_player_id=plr1.ident,
            second_player_id=plr2.ident,
            year_weeknum=tt.get_year_weeknum(datetime.date(2014, 5, 14)),
        )
        self.assertNotEqual(mstat, None)
        self.assertEqual(mstat.left_side.aces, 3)
        self.assertEqual(mstat.right_side.aces, 5)
        self.assertEqual(mstat.left_side.first_service_in, WinLoss(32, 24))
        self.assertEqual(mstat.right_side.first_service_in, WinLoss(38, 25))
        self.assertEqual(mstat.left_side.bp_win, WinLoss(2, 5))
        self.assertEqual(mstat.right_side.bp_win, WinLoss(4, 3))
        self.assertEqual(mstat.total_points_won(), (51, 68))

    def test_get_baroni_williams(self):
        """baroni win 6-4 6-3"""
        plr1 = Player(ident=180, name="Mirjana Lucic-Baroni", cou="CRO")
        plr2 = Player(ident=151, name="Venus Williams", cou="USA")
        mstat = get(
            sex="wta",
            tour_id=9148,
            rnd=Round("Final"),
            first_player_id=plr1.ident,
            second_player_id=plr2.ident,
            year_weeknum=tt.get_year_weeknum(datetime.date(2014, 9, 14)),
        )
        self.assertNotEqual(mstat, None)
        self.assertEqual(mstat.left_side.aces, 6)
        self.assertEqual(mstat.right_side.aces, 5)
        self.assertEqual(mstat.left_side.bp_win, WinLoss(4, 7))
        self.assertEqual(mstat.right_side.bp_win, WinLoss(2, 2))
        self.assertEqual(mstat.left_side.first_service_win, WinLoss(31, 9))
        self.assertEqual(mstat.right_side.first_service_win, WinLoss(23, 14))
        self.assertEqual(mstat.left_side.second_service_win, WinLoss(10, 13))
        self.assertEqual(mstat.right_side.second_service_win, WinLoss(11, 21))
        self.assertEqual(mstat.left_side.double_faults, 4)
        self.assertEqual(mstat.right_side.double_faults, 2)
        self.assertEqual(mstat.total_points_won(), (76, 56))

    def test_surf_trans_coef(self):
        coef = generic_surface_trans_coef(
            sex="atp", fun_name="service_win", surface_from="Clay", surface_to="Carpet"
        )
        self.assertTrue(coef > 1.0)

    def test_generic_result_value(self):
        val = generic_result_value(
            sex="wta", fun_name="service_win", level="qual", surface="Carpet"
        )
        self.assertTrue(val is not None and val > 0.5)

    def test_generic_result_dict(self):
        dct = generic_result_dict(sex="wta", fun_name="service_win")
        self.assertTrue(len(dct) > 1)


if __name__ == "__main__":
    dba.open_connect()
    unittest.main()
    dba.close_connect()
