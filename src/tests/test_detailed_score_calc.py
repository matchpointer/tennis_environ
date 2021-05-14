# -*- coding: utf-8 -*-
import unittest
import copy

import common as co
import log
from score import Score
from stat_cont import WinLoss
from detailed_score import (
    DetailedGame,
    error_code,
    DetailedScore,
    SetItems,
)
from detailed_score_dbsa import open_db
from detailed_score_calc import (
    CalcSrvRatio,
    CalcSrvRatioBeforeTie,
    CalcLastinrow,
    CalcExistScr,
    CalcWinclose,
)
import markov


def make_1set_det_score():
    """:return DetailedScore for 6-7(3) where left_opener, 5-5 & 5-6 simulated"""
    items = [
        (
            ((0, 0),),
            DetailedGame("1111", left_wingame=True, left_opener=True, tiebreak=False),
        ),
        (
            ((1, 0),),
            DetailedGame("0000", left_wingame=True, left_opener=False, tiebreak=False),
        ),
        (
            ((2, 0),),
            DetailedGame("1111", left_wingame=True, left_opener=True, tiebreak=False),
        ),
        (
            ((3, 0),),
            DetailedGame("0000", left_wingame=True, left_opener=False, tiebreak=False),
        ),
        (
            ((4, 0),),
            DetailedGame("0000", left_wingame=False, left_opener=True, tiebreak=False),
        ),
        (
            ((4, 1),),
            DetailedGame("1111", left_wingame=False, left_opener=False, tiebreak=False),
        ),
        (
            ((4, 2),),
            DetailedGame("0000", left_wingame=False, left_opener=True, tiebreak=False),
        ),
        (
            ((4, 3),),
            DetailedGame("1111", left_wingame=False, left_opener=False, tiebreak=False),
        ),
        (
            ((4, 4),),
            DetailedGame("0000", left_wingame=False, left_opener=True, tiebreak=False),
        ),
        (
            ((4, 5),),
            DetailedGame("0000", left_wingame=True, left_opener=False, tiebreak=False),
        ),
        (
            ((5, 5),),
            DetailedGame(
                "0000",
                left_wingame=False,
                left_opener=True,
                tiebreak=False,
                error=error_code("GAME_SCORE_SIMULATED"),
            ),
        ),
        (
            ((5, 6),),
            DetailedGame(
                "0000",
                left_wingame=True,
                left_opener=False,
                tiebreak=False,
                error=error_code("GAME_SCORE_SIMULATED"),
            ),
        ),
        (
            ((6, 6),),
            DetailedGame(
                "1110000000", left_wingame=False, left_opener=True, tiebreak=True
            ),
        ),
    ]
    return DetailedScore(items)


def make_setitems_with_simulated():
    setitems = SetItems.from_scores(
        setnum=1, detailed_score=make_1set_det_score(), score=Score("6-7(3)")
    )
    return setitems


class TestCalcWinclose(unittest.TestCase):
    def test_set1_with_tie(self):
        s1items = make_setitems_with_simulated()  # '6-7(3)' left_opener=True
        calc55 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.55
        )
        calc60 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.60
        )
        calc55.proc_set(s1items)
        calc60.proc_set(s1items)
        self.assertTrue(calc55.fst_measure > 0.0)
        self.assertTrue(calc60.fst_measure > 0.0)
        self.assertTrue(calc55.fst_measure > calc60.fst_measure)

    def test_set1flip_with_tie(self):
        s1items = copy.deepcopy(
            make_setitems_with_simulated()
        )  # '6-7(3)' left_opener=True
        s1items.flip()  # '7-6(3)' left_opener=False
        calc55 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.55
        )
        calc60 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.60
        )
        calc55.proc_set(s1items)
        calc60.proc_set(s1items)
        self.assertTrue(calc55.snd_measure > 0.0)
        self.assertTrue(calc60.snd_measure > 0.0)
        self.assertTrue(calc55.snd_measure > calc60.snd_measure)


class TestCalcExistScr(unittest.TestCase):
    def test_with_simulated(self):
        def impl_with_simulated():
            calc = CalcExistScr()
            calc.proc_set(setitems)
            is54 = calc.score_exist_val(5, 4)
            self.assertEqual(is54, 0)
            is45 = calc.score_exist_val(4, 5)
            self.assertEqual(is45, 1)
            is55 = calc.score_exist_val(5, 5)
            self.assertEqual(is55, -1)
            is66 = calc.score_exist_val(6, 6)
            self.assertEqual(is66, 1)

        setitems = make_setitems_with_simulated()
        impl_with_simulated()
        setitems.flip()
        impl_with_simulated()


class TestWtaDbMatch(unittest.TestCase):
    dbdet = open_db(sex="wta")

    @classmethod
    def setUpClass(cls):
        cls.dbdet.query_matches()

        # 6-0 4-6 7-6 (supertie: 10-6, max dif was 9-6). AO 2019.
        cls.boulter_makarova_det_score = cls.dbdet.get_detailed_score(
            tour_id=12357, rnd="First", left_id=14175, right_id=4159
        )
        cls.boulter_makarova_score = cls.dbdet.get_score(
            tour_id=12357, rnd="First", left_id=14175, right_id=4159
        )

        # 1-6 7-6(2) 6-0
        cls.gracheva_mladenovic_det_score = cls.dbdet.get_detailed_score(
            tour_id=13059, rnd="Second", left_id=44510, right_id=9458
        )
        cls.gracheva_mladenovic_score = cls.dbdet.get_score(
            tour_id=13059, rnd="Second", left_id=44510, right_id=9458
        )

        # 6-0 6-7(3) 6-4
        cls.koukalova_safarova_det_score = cls.dbdet.get_detailed_score(
            tour_id=7233, rnd="Second", left_id=454, right_id=4089
        )
        cls.koukalova_safarova_score = cls.dbdet.get_score(
            tour_id=7233, rnd="Second", left_id=454, right_id=4089
        )

    def test_boulter_makarova_existscr(self):
        def get_10_01(setitems, prefix):
            calc = CalcExistScr()
            calc.proc_set(setitems)
            feats = calc.result_features(prefix)
            fis10 = co.find_first(feats, lambda f: f.name == f"{prefix}_is_1-0")
            fis01 = co.find_first(feats, lambda f: f.name == f"{prefix}_is_0-1")
            self.assertIsNotNone(fis10)
            self.assertIsNotNone(fis01)
            return fis10.value, fis01.value

        det_score, score = self.boulter_makarova_det_score, self.boulter_makarova_score
        s1items = SetItems.from_scores(1, det_score, score)
        val10, val01 = get_10_01(s1items, "s1")
        self.assertEqual(val10, 0)
        self.assertEqual(val01, 1)

        s1items.flip()
        val10, val01 = get_10_01(s1items, "s1")
        self.assertEqual(val10, 0)
        self.assertEqual(val01, 1)

    def test_boulter_makarova_lastinrow(self):
        det_score, score = self.boulter_makarova_det_score, self.boulter_makarova_score
        s1items = SetItems.from_scores(1, det_score, score)
        s2items = SetItems.from_scores(2, det_score, score)
        s3items = SetItems.from_scores(3, det_score, score)
        calc1 = CalcLastinrow()
        calc1.proc_set(s1items)
        self.assertEqual(calc1.fst_lastrow_games, 6)
        calc2 = CalcLastinrow()
        calc2.proc_set(s2items)
        self.assertEqual(calc2.fst_lastrow_games, -2)
        calc3 = CalcLastinrow()
        calc3.proc_set(s3items)
        self.assertEqual(calc3.fst_lastrow_games, 2)

    def test_koukalova_safarova_calcwinclose(self):
        # Koukalova lose set2
        # Koukalova leads in set2: 6-5 0-0 rcv; tie 1-0
        det_score, score = (
            self.koukalova_safarova_det_score,
            self.koukalova_safarova_score,
        )
        self.assertEqual(score, Score("6-0 6-7(3) 6-4"))
        s2items = SetItems.from_scores(2, det_score, score)
        self.assertTrue(s2items.ok_set())
        calc65 = CalcWinclose(
            sex="wta",
            surface="Grass",
            level="gs",
            is_qual=False,
            min_prob=0.65,
            is_mean=True,
        )
        calc65.proc_set(s2items)
        self.assertTrue((calc65.fst_measure / calc65.fst_n) < 0.66)

    def test_gracheva_mladenovic_calcwinclose(self):
        det_score, score = (
            self.gracheva_mladenovic_det_score,
            self.gracheva_mladenovic_score,
        )
        self.assertEqual(score, Score("1-6 7-6(2) 6-0"))
        s2items = SetItems.from_scores(2, det_score, score)
        self.assertTrue(s2items.ok_set())

        calc60 = CalcWinclose(
            sex="wta",
            surface="Hard",
            level="gs",
            is_qual=False,
            min_prob=0.60,
            is_mean=True,
        )
        calc60.proc_set(s2items)
        avg60L = calc60.snd_measure / calc60.snd_n
        avg60W = calc60.fst_measure / calc60.fst_n
        print(
            f"60 avgL: {avg60L} nL: {calc60.snd_n} vL: {calc60.snd_measure} "
            f"avgW: {avg60W} nW: {calc60.fst_n} vW: {calc60.fst_measure}"
        )
        self.assertTrue(avg60L > 0.8)
        self.assertTrue(calc60.snd_measure > calc60.fst_measure)

        calc65 = CalcWinclose(
            sex="wta",
            surface="Hard",
            level="gs",
            is_qual=False,
            min_prob=0.65,
            is_mean=True,
        )
        calc65.proc_set(s2items)
        avg65L = calc65.snd_measure / calc65.snd_n
        avg65W = calc65.fst_measure / calc65.fst_n
        print(
            f"65 avgL: {avg65L} nL: {calc65.snd_n} vL: {calc65.snd_measure} "
            f"avgW: {avg65W} nW: {calc65.fst_n} vW: {calc65.fst_measure}"
        )
        self.assertTrue(avg65L > 0.8)
        self.assertTrue(calc65.snd_measure > calc65.fst_measure)

        calc70 = CalcWinclose(
            sex="wta",
            surface="Hard",
            level="gs",
            is_qual=False,
            min_prob=0.70,
            is_mean=True,
        )
        calc70.proc_set(s2items)
        avg70L = calc70.snd_measure / calc70.snd_n
        avg70W = calc70.fst_measure / calc70.fst_n
        print(
            f"70 avgL: {avg70L} nL: {calc70.snd_n} vL: {calc70.snd_measure} "
            f"avgW: {avg70W} nW: {calc70.fst_n} vW: {calc70.fst_measure}"
        )
        self.assertTrue(avg70L > 0.8)
        self.assertTrue(calc70.snd_measure > calc70.fst_measure)

    def test_boulter_makarova_calcwinclose(self):
        det_score, score = self.boulter_makarova_det_score, self.boulter_makarova_score
        self.assertEqual(score, Score("6-0 4-6 7-6(6)"))
        # s3 is super tie (not implied in markov.py)
        s2items = SetItems.from_scores(2, det_score, score)
        self.assertTrue(s2items.ok_set())
        calc55 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.55
        )
        calc60 = CalcWinclose(
            sex="wta", surface="Hard", level="gs", is_qual=False, min_prob=0.60
        )
        calc55.proc_set(s2items)
        calc60.proc_set(s2items)
        self.assertTrue(calc55.fst_measure > 0.0)
        self.assertTrue(calc60.fst_measure > 0.0)
        self.assertTrue(calc55.fst_measure > calc60.fst_measure)

    def test_boulter_makarova_calcsrvratio(self):
        det_score, score = self.boulter_makarova_det_score, self.boulter_makarova_score
        self.assertEqual(score, Score("6-0 4-6 7-6(6)"))
        s1items = SetItems.from_scores(1, det_score, score)
        s2items = SetItems.from_scores(2, det_score, score)
        s3items = SetItems.from_scores(3, det_score, score)
        self.assertTrue(s1items.ok_set() and s2items.ok_set() and s3items.ok_set())

        calc1 = CalcSrvRatio()
        calc1.proc_set(s1items)
        self.assertEqual(calc1.fst_wl, WinLoss(12, 3))
        self.assertEqual(calc1.snd_wl, WinLoss(14, 21))

        calc2 = CalcSrvRatio()
        calc2.proc_set(s2items)
        self.assertTrue(calc2.fst_wl.ratio < calc2.snd_wl.ratio)

        calc3 = CalcSrvRatio()
        calc3.proc_set(s3items)
        self.assertTrue(calc3.fst_wl.ratio > calc3.snd_wl.ratio)

        calc3bt = CalcSrvRatioBeforeTie()
        calc3bt.proc_set(s3items)
        self.assertTrue(0 < calc3bt.fst_wl.size < calc3.fst_wl.size)
        self.assertTrue(0 < calc3bt.snd_wl.size < calc3.snd_wl.size)
        feats = calc3bt.result_features("pref")
        self.assertEqual(feats[0].name, "pref_fst_beftie_srv_ratio")
        self.assertEqual(feats[1].name, "pref_snd_beftie_srv_ratio")

        feats = calc3.cumul_result_features("pref", [calc1, calc2])
        self.assertTrue(len(feats) == 2)
        self.assertEqual(feats[0].name, "pref_fst_srv_ratio")
        self.assertTrue(feats[0].value > feats[1].value)


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", "debug")
    markov.initialize_gen()
    unittest.main()
