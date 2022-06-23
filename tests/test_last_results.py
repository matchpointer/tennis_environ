# -*- coding=utf-8 -*-
import unittest
import datetime

import common as co
import tennis_time as tt
import weeked_tours
from last_results import LastResults
from surf import Clay, Hard


class WinstreakAdvTest(unittest.TestCase):
    def test_long_poor_practice(self):
        # simulate Andeescu at 2022 Stutgart Second vs Alexandrova
        lr = LastResults.from_week_results(
            [
                [[True, Clay]],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ]
        )
        self.assertTrue(lr.long_poor_practice())

    def test_winstreak_adv(self):
        # simulate Yafan Wang - Lauren Davis 2019.09.23
        wang_lr = LastResults.from_week_results(
            [
                [],
                [[False, Hard], [True, Hard], [True, Hard], [True, Hard]],
                [[False, Hard]],
                [],
                [[False, Hard]],
                [[False, Hard]]
            ]
        )
        davis_lr = LastResults.from_week_results(
            [
                [[True, Hard], [True, Hard]],
                [],
                [],
                [[False, Hard], [True, Hard], [True, Hard], [True, Hard]],
                [[False, Hard], [True, Hard]],
                []
            ]
        )
        self.assertFalse(wang_lr.poor_practice())
        self.assertFalse(wang_lr.last_weeks_empty(weeks_num=3))

        self.assertFalse(davis_lr.poor_practice())
        self.assertFalse(davis_lr.last_weeks_empty(weeks_num=3))

    def test_winstreak_adv2(self):
        # simulate Caroline Garcia - Anna Blinkova 2019.05.30
        garcia_lr = LastResults.from_week_results(
            [
                [(True, Clay)],
                [(False, Clay), (True, Clay), (True, Clay), (True, Clay), (True, Clay)],
                [(False, Clay)],
                [(False, Clay), (True, Clay), (True, Clay)],
                [],
                [(False, Clay)],
            ]
        )
        blinkova_lr = LastResults.from_week_results(
            [
                [(True, Clay), (True, Clay), (True, Clay), (True, Clay)],
                [],
                [(False, Clay), (True, Clay), (True, Clay), (True, Clay), (True, Clay)],
                [(False, Clay), (True, Clay)],
                [(False, Clay), (True, Clay), (True, Clay), (True, Clay)],
                [(False, Clay), (True, Clay)],
            ]
        )

        fst_ratio = garcia_lr.best_win_streak_ratio(min_ratio=0.777, min_size=8)
        snd_ratio = blinkova_lr.best_win_streak_ratio(min_ratio=0.777, min_size=8)
        adv_side = None
        if fst_ratio and fst_ratio > (snd_ratio + 0.2):
            adv_side = co.LEFT
        if not fst_ratio and snd_ratio > (fst_ratio + 0.2):
            adv_side = co.RIGHT

        self.assertTrue(adv_side is not None)


# @unittest.skip("actual only was at concrete date")
class PlayerTodayLastresCaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from oncourt import extplayers
        from oncourt import dba

        dba.open_connect()
        extplayers.initialize(yearsnum=1.2)

        date = tt.past_monday_date(datetime.date.today())

        sex = "atp"
        hist_weeks_len = 6
        weeked_tours.initialize_sex(
            sex,
            min_date=date - datetime.timedelta(days=7 * hist_weeks_len),
            max_date=date + datetime.timedelta(days=11),
            with_today=True,
            with_paired=True,
            with_bets=True,
            with_stat=True,
            rnd_detailing=False,
        )

    # @unittest.skip("actual only was at concrete date: 2022-04-16")
    def test_show_player_today_lastres(self):
        # self.show_player_today_lastres(
        #     weeks_ago=4, sex='atp', player_id=34233, player_name='popyrin')
        self.show_player_today_lastres(
            weeks_ago=4, sex='atp', player_id=29457, player_name='pellegrino')
        self.assertTrue(1 == 1)

    @staticmethod
    def show_player_today_lastres(weeks_ago, sex, player_id, player_name):
        last_res = LastResults(sex=sex, player_id=player_id, weeks_ago=weeks_ago)
        print(f"{player_name} last_res:\n{last_res}")


if __name__ == '__main__':
    unittest.main()
