import unittest

import common as co
from stat_cont import (
    Histogram,
    WinLoss,
    HoldStat,
    AllowBreakPointStat,
    SetOpenerMatchTracker,
    QuadServiceStat,
)
import score as sc


class WinLossTest(unittest.TestCase):
    def test_compare(self):
        wl1 = WinLoss.create_from_text("41.2% (17)")
        wl2 = WinLoss.create_from_text("37.8% (339)")
        self.assertTrue(wl1 > wl2)
        self.assertTrue(wl2 < wl1)
        self.assertEqual([wl2, wl1], sorted([wl1, wl2]))

    def test_sum(self):
        wl1 = WinLoss(7, 3)
        wl2 = WinLoss(6, 4)
        wl3 = WinLoss(0, 0)
        dct = {1: wl1, 2: wl2, 3: wl3}
        s = sum((dct[i] for i in range(1, 4)), WinLoss())
        self.assertEqual(wl1 + wl2 + wl3, s)


class SetsScoreHistogramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.histos = [
            Histogram(int, {(2, 0): 5, (2, 1): 3, (1, 2): 1, (0, 2): 2}),
            Histogram(int, {(2, 0): 6, (2, 1): 4, (1, 2): 2, (0, 2): 3}),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.histos = []

    def test_init_and_sum(self):
        self.assertEqual(self.histos[0][(2, 0)], 5)
        self.assertEqual(self.histos[0][(2, 1)], 3)
        self.assertEqual(self.histos[0][(1, 2)], 1)
        self.assertEqual(self.histos[0][(0, 2)], 2)

        self.assertEqual(self.histos[1][(2, 0)], 6)
        self.assertEqual(self.histos[1][(2, 1)], 4)
        self.assertEqual(self.histos[1][(1, 2)], 2)
        self.assertEqual(self.histos[1][(0, 2)], 3)

        start_histo = Histogram()
        sc.complete_sets_score_keys(start_histo)
        histo = sum(self.histos, start_histo)
        self.assertEqual(type(histo), Histogram)
        self.assertEqual(histo[(2, 0)], 11)
        self.assertEqual(histo[(2, 1)], 7)
        self.assertEqual(histo[(1, 2)], 3)
        self.assertEqual(histo[(0, 2)], 5)


class HoldStatTest(unittest.TestCase):
    def test_close_previous(self):
        stat1 = HoldStat()
        stat1.close_previous(
            sc.Score("1-1"), prev_left_service=True, prev_left_wingame=True
        )  # 1/1, 0/0
        stat1.close_previous(
            sc.Score("2-1"), prev_left_service=False, prev_left_wingame=False
        )  # 1/1, 1/1
        stat1.close_previous(
            sc.Score("2-2"), prev_left_service=True, prev_left_wingame=False
        )  # 1/2, 1/1
        stat1.close_previous(
            sc.Score("2-3"), prev_left_service=False, prev_left_wingame=False
        )  # 1/2, 2/2
        stat1.close_previous(
            sc.Score("2-4"), prev_left_service=True, prev_left_wingame=True
        )  # 2/3, 2/2
        stat1.close_previous(
            sc.Score("6-6"), prev_left_service=True, prev_left_wingame=False
        )  # skip tie
        res_pair = stat1.result_pair(setnum=1, as_float=False)
        self.assertEqual(res_pair, (WinLoss(2, 1), WinLoss(2, 0)))


class AllowBreakPointStatTest(unittest.TestCase):
    def test_continue_bp(self):
        stat1 = AllowBreakPointStat()

        prev_score = sc.Score("5-2")
        prev_left_service = False
        prev_ingame = ("30", "30")
        ingame = ("40", "30")
        stat1.continue_current(prev_score, prev_ingame, prev_left_service, ingame)
        self.assertEqual((0, 1), stat1.get_counts())


class TestSetOpener(unittest.TestCase):
    def test_match_track(self):
        trk = SetOpenerMatchTracker()
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, None)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, None)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)
        r = trk.put(setnum=1, scr=(5, 3), is_left_service=True)
        self.assertEqual(r, True)

        r = trk.put(setnum=2, scr=(0, 0), is_left_service=False)
        self.assertEqual(r, None)

        r = trk.get_opener_side(setnum=1, scr=(5, 3), at=True)
        self.assertEqual(r, co.LEFT)

        r = trk.get_opener_side(setnum=1, scr=(5, 3), at=False)
        self.assertEqual(r, co.RIGHT)

        r = trk.get_opener_side(setnum=1, scr=(5, 4), at=True)
        self.assertEqual(r, co.RIGHT)

        r = trk.get_opener_side(setnum=1, scr=(0, 0), at=True)
        self.assertEqual(r, co.LEFT)


class QuadServiceStatTest(unittest.TestCase):
    def test_close_by_right(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("5-1"),
            prev_ingame=("30", "40"),
            prev_left_service=False,
            score=sc.Score("5-2"),
            ingame=("0", "0"),
            left_service=True,
        )

        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))

        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.LEFT))

    def test_close_by_left_then_fresh_right(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("5-1"),
            prev_ingame=("30", "40"),
            prev_left_service=False,
            score=sc.Score("6-1 0-0"),
            ingame=("0", "40"),
            left_service=True,
        )
        # closing: right broken by 3 points
        self.assertEqual(WinLoss(0, 2), qstat1.srv_win_loss(co.RIGHT, co.ADV))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))

        # freshing: left loss 3 points
        self.assertEqual(WinLoss(0, 2), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.LEFT, co.DEUCE))

    def test_close_by_left_then_fresh_right_tiebreak(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("6-4 5-6"),
            prev_ingame=("30", "40"),
            prev_left_service=True,
            score=sc.Score("6-4 6-6"),
            ingame=("0", "2"),
            left_service=True,
        )
        # closing: left hold by win 3 points: ADV(2), DEUCE(1)
        # freshing: right by win 1 point: DEUCE(1)
        # freshing: left by loss 1 point: ADV(0)
        self.assertEqual(WinLoss(2, 1), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.LEFT, co.DEUCE))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))

    def test_close_by_left_then_fresh_left_tiebreak(self):
        qstat1 = QuadServiceStat()
        qstat1.update_with(
            prev_score=sc.Score("6-4 5-6"),
            prev_ingame=("30", "40"),
            prev_left_service=True,
            score=sc.Score("6-4 6-6"),
            ingame=("2", "0"),
            left_service=True,
        )
        # closing: left hold by win 3 points: ADV(2), DEUCE(1)
        # freshing: right opener by loss 1 point: DEUCE(0)
        # freshing: left by win 1 point: ADV(1)
        self.assertEqual(WinLoss(3, 0), qstat1.srv_win_loss(co.LEFT, co.ADV))
        self.assertEqual(WinLoss(1, 0), qstat1.srv_win_loss(co.LEFT, co.DEUCE))
        self.assertEqual(WinLoss(0, 1), qstat1.srv_win_loss(co.RIGHT, co.DEUCE))
        self.assertEqual(WinLoss(0, 0), qstat1.srv_win_loss(co.RIGHT, co.ADV))


if __name__ == "__main__":
    unittest.main()
