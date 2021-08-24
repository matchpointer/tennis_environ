# -*- coding: utf-8 -*-
import unittest

import common as co

# import log
from score import Score
from detailed_score import (
    DetailedGame,
    tie_leadchanges_count,
    tie_minibreaks_count,
    error_code,
    match_win_side,
    DetailedScore,
    DetailedScoreError,
    BreaksMove,
    is_full,
    SetItems,
)
from detailed_score_dbsa import open_db


class TiebreakEnumPointsTest(unittest.TestCase):
    def test_enumerate(self):
        det_game = DetailedGame(
            "001101010111", left_wingame=True, left_opener=True, tiebreak=False
        )
        pnts = list(iter(det_game))
        self.assertEqual(pnts[0].num_score(before=False), (0, 1))
        self.assertEqual(pnts[1].num_score(before=False), (0, 2))
        self.assertEqual(pnts[2].num_score(before=False), (1, 2))
        self.assertEqual(pnts[3].num_score(before=False), (2, 2))
        self.assertEqual(pnts[4].num_score(before=False), (2, 3))
        self.assertEqual(pnts[5].num_score(before=False), (3, 3))
        self.assertEqual(pnts[6].num_score(before=False), (3, 4))
        self.assertEqual(pnts[7].num_score(before=False), (4, 4))
        self.assertEqual(pnts[8].num_score(before=False), (4, 5))
        self.assertEqual(pnts[9].num_score(before=False), (5, 5))
        self.assertEqual(pnts[10].num_score(before=False), (6, 5))
        self.assertEqual(pnts[11].num_score(before=False), (7, 5))

    def test_enumerate_tie(self):
        det_game = DetailedGame(
            "0001111111", left_wingame=False, left_opener=False, tiebreak=True
        )
        pnts = list(iter(det_game))
        self.assertEqual(pnts[0].num_score(before=False), (0, 1))
        self.assertEqual(pnts[1].num_score(before=False), (0, 2))
        self.assertEqual(pnts[2].num_score(before=False), (0, 3))
        self.assertEqual(pnts[3].num_score(before=False), (1, 3))
        self.assertEqual(pnts[-1].num_score(before=False), (7, 3))

    def test_enumerate_tie_2(self):
        det_game = DetailedGame(
            "1111111", left_wingame=False, left_opener=False, tiebreak=True
        )
        pnts = list(iter(det_game))
        self.assertEqual(pnts[0].num_score(before=False), (1, 0))
        self.assertEqual(pnts[1].num_score(before=False), (2, 0))
        self.assertEqual(pnts[-1].num_score(before=False), (7, 0))


class TiebreakCountsTest(unittest.TestCase):
    def test_leadchanges_count(self):
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        self.assertEqual(leadchanges_count, 3)

        det_game = DetailedGame(
            "01100110111", left_wingame=False, left_opener=False, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        self.assertEqual(leadchanges_count, 3)

        det_game = DetailedGame(
            "0000000", left_wingame=False, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        self.assertEqual(leadchanges_count, 0)

        det_game = DetailedGame(
            "01111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        self.assertEqual(leadchanges_count, 1)

    def test_minibreaks_count(self):
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        minibreaks_count = tie_minibreaks_count(det_game)
        self.assertEqual(minibreaks_count, 10)

    def test_minibreaks_diff(self):
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        self.assertEqual(points[0].minibreaks_diff(), -1)
        self.assertEqual(points[0].minibreaks_diff(left=False), 1)

        self.assertEqual(points[1].minibreaks_diff(), 0)
        self.assertEqual(points[1].minibreaks_diff(left=False), 0)

        self.assertEqual(points[2].minibreaks_diff(), 1)
        self.assertEqual(points[2].minibreaks_diff(left=False), -1)


class DetailedGameTest(unittest.TestCase):
    def test_final_num_score(self):
        dg = DetailedGame.from_str("wot_E32768PLWLWL11011")
        fin_num_scr = dg.final_num_score(before=False)
        self.assertEqual(fin_num_scr, (6, 4))
        fin_num_scr = dg.final_num_score(before=True)
        self.assertEqual(fin_num_scr, (5, 4))

    def test_game_orient_error(self):
        """д.б. ошибка"""
        det_game = DetailedGame(
            "010110110000", left_wingame=True, left_opener=True, tiebreak=True
        )
        self.assertTrue(det_game.error)
        self.assertTrue(det_game.orientation_error())

        det_game = DetailedGame(
            "010110", left_wingame=True, left_opener=True, tiebreak=True
        )
        self.assertTrue(det_game.error)
        self.assertTrue(det_game.orientation_error())

        det_game = DetailedGame(
            "001111", left_wingame=False, left_opener=True, tiebreak=False
        )
        self.assertTrue(det_game.error)
        self.assertTrue(det_game.orientation_error())

        det_game = DetailedGame(
            "110000", left_wingame=True, left_opener=True, tiebreak=False
        )
        self.assertTrue(det_game.error)
        self.assertTrue(det_game.orientation_error())

    def test_save_restore(self):
        det_game = DetailedGame(
            "0110011011",
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("SPANING"),
        )
        text = str(det_game)
        det_game2 = DetailedGame.from_str(text)
        self.assertEqual(det_game, det_game2)

    def test_save_restore2(self):
        det_game = DetailedGame(
            "000001111111111",
            left_wingame=True,
            left_opener=True,
            tiebreak=True,
            supertiebreak=True,
            error=0,
        )
        text = str(det_game)
        det_game2 = DetailedGame.from_str(text)
        self.assertEqual(det_game, det_game2)
        self.assertEqual(15, len(list(iter(det_game2))))

    def test_state(self):
        """here error_code value is not meaning and used for test get/set"""
        det_game = DetailedGame(
            "0110011011",
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("GAME_ORIENT"),
        )
        self.assertTrue(det_game.error)
        self.assertTrue(det_game.left_wingame)
        self.assertTrue(det_game.left_opener)
        self.assertFalse(det_game.tiebreak)
        self.assertEqual(det_game.error, error_code("GAME_ORIENT"))
        det_game.error = 0
        det_game.left_wingame = False
        det_game.left_opener = False
        det_game.tiebreak = True
        self.assertFalse(det_game.error)
        self.assertFalse(det_game.left_wingame)
        self.assertFalse(det_game.left_opener)
        self.assertTrue(det_game.tiebreak)
        det_game.error = error_code("GAME_ORIENT")
        self.assertTrue(det_game.error)
        self.assertEqual(det_game.error, error_code("GAME_ORIENT"))
        self.assertFalse(det_game.left_wingame)
        self.assertFalse(det_game.left_opener)
        self.assertTrue(det_game.tiebreak)

    def test_point_text_score(self):
        det_game = DetailedGame(
            "01100110100111", left_wingame=True, left_opener=True, tiebreak=False
        )
        points = [p for p in det_game]
        self.assertEqual(points[0].text_score(), ("0", "0"))
        self.assertEqual(points[1].text_score(), ("0", "15"))
        self.assertEqual(points[2].text_score(), ("15", "15"))
        self.assertEqual(points[3].text_score(), ("30", "15"))
        self.assertEqual(points[4].text_score(), ("30", "30"))
        self.assertEqual(points[5].text_score(), ("30", "40"))
        self.assertEqual(points[6].text_score(), ("40", "40"))
        self.assertEqual(points[7].text_score(), ("A", "40"))
        self.assertEqual(points[8].text_score(), ("40", "40"))
        self.assertEqual(points[9].text_score(), ("A", "40"))
        self.assertEqual(points[10].text_score(), ("40", "40"))
        self.assertEqual(points[11].text_score(), ("40", "A"))
        self.assertEqual(points[12].text_score(), ("40", "40"))
        self.assertEqual(points[13].text_score(), ("A", "40"))
        self.assertEqual(points[13].text_score(left=True), ("A", "40"))
        self.assertEqual(points[13].text_score(left=False), ("40", "A"))

    def test_point_text_score_tie(self):
        pts = "100110011100100111"
        det_game = DetailedGame(
            points=pts, left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        self.assertEqual(points[0].text_score(), ("0", "0"))
        self.assertEqual(points[1].text_score(), ("1", "0"))
        self.assertEqual(points[2].text_score(), ("1", "1"))
        self.assertEqual(points[3].text_score(), ("1", "2"))
        self.assertEqual(points[4].text_score(), ("2", "2"))
        self.assertEqual(points[5].text_score(), ("3", "2"))
        self.assertEqual(points[6].text_score(), ("3", "3"))
        self.assertEqual(points[7].text_score(), ("3", "4"))
        self.assertEqual(points[8].text_score(), ("4", "4"))
        self.assertEqual(points[9].text_score(), ("5", "4"))
        self.assertEqual(points[10].text_score(), ("6", "4"))
        self.assertEqual(points[11].text_score(), ("6", "5"))
        self.assertEqual(points[12].text_score(), ("6", "6"))
        self.assertEqual(points[13].text_score(), ("7", "6"))
        self.assertEqual(points[14].text_score(), ("7", "7"))
        self.assertEqual(points[15].text_score(), ("7", "8"))
        self.assertEqual(points[16].text_score(), ("8", "8"))
        self.assertEqual(points[17].text_score(), ("9", "8"))
        self.assertEqual(points[17].text_score(left=True), ("9", "8"))
        self.assertEqual(points[17].text_score(left=False), ("8", "9"))
        self.assertEqual(len(points), 18)

    def test_left_break_point_score(self):
        det_game = DetailedGame(
            "00011100", left_wingame=False, left_opener=True, tiebreak=False
        )
        points = [p for p in det_game]
        self.assertEqual(points[0].break_point_score(left=True), 0)
        self.assertEqual(points[1].break_point_score(left=True), 0)
        self.assertEqual(points[2].break_point_score(left=True), 0)
        self.assertEqual(points[3].break_point_score(left=True), 3)
        self.assertEqual(points[4].break_point_score(left=True), 2)
        self.assertEqual(points[5].break_point_score(left=True), 1)
        self.assertEqual(points[6].break_point_score(left=True), 0)
        self.assertEqual(points[7].break_point_score(left=True), 1)

    def test_right_break_point_score(self):
        det_game = DetailedGame(
            "0001110111", left_wingame=False, left_opener=False, tiebreak=False
        )
        points = [p for p in det_game]
        self.assertEqual(points[0].break_point_score(left=False), 0)
        self.assertEqual(points[1].break_point_score(left=False), 0)
        self.assertEqual(points[2].break_point_score(left=False), 0)
        self.assertEqual(points[3].break_point_score(left=False), 3)
        self.assertEqual(points[4].break_point_score(left=False), 2)
        self.assertEqual(points[5].break_point_score(left=False), 1)
        self.assertEqual(points[6].break_point_score(left=False), 0)
        self.assertEqual(points[7].break_point_score(left=False), 1)
        self.assertEqual(points[8].break_point_score(left=False), 0)
        self.assertEqual(points[9].break_point_score(left=False), 0)

    def test_few_points(self):
        pts = "1"
        det_game = DetailedGame(
            points=pts,
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("POINT_SCORE_FEW_DATA"),
        )
        points = [p for p in det_game]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].win(), True)

        pts = ""
        det_game = DetailedGame(
            points=pts,
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("POINT_SCORE_FEW_DATA"),
        )
        points = [p for p in det_game]
        self.assertEqual(len(points), 0)

    def test_tie_points_iter(self):
        pts = "100110011100100111"
        det_game = DetailedGame(
            points=pts, left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        self.assertEqual(len(points), len(pts))
        self.assertTrue(points[-1].win_game() and points[-1].win())
        self.assertTrue(points[-2].game_point_score(before=False) and points[-2].win())
        self.assertTrue(points[-3].equal_score(before=False) and points[-3].win())
        self.assertTrue(points[-4].equal_score(before=True))
        self.assertTrue(
            points[-4].break_point_score(before=False) and points[-4].loss()
        )
        self.assertTrue(points[-5].equal_score(before=False) and points[-5].loss())
        self.assertTrue(points[-6].game_point_score(before=False))
        self.assertTrue(points[-7].equal_score(before=False))
        self.assertEqual(len([p for p in points if p.equal_score(before=False)]), 3)
        self.assertEqual(len([p for p in points if p.game_point_score()]), 4)
        self.assertEqual(
            len([p for p in points if p.break_point_score(before=False)]), 1
        )
        # win_minibreak():
        self.assertTrue(
            points[9].win()
            and not points[9].serve()
            and points[9].win_minibreak(True)
            and points[9].loss_minibreak(False)
        )
        # serve:
        self.assertTrue(
            points[0].serve(True)
            and points[0].win(True)
            and not points[0].win_minibreak(True)
        )
        self.assertTrue(
            not points[1].serve(True)
            and points[1].loss(True)
            and not points[1].win_minibreak(True)
        )
        self.assertTrue(
            not points[2].serve(True)
            and points[2].loss(True)
            and not points[2].win_minibreak(True)
        )
        self.assertTrue(
            points[3].serve() and points[3].serve(True) and not points[3].serve(False)
        )
        self.assertTrue(
            points[4].serve() and points[4].serve(True) and not points[4].serve(False)
        )
        self.assertTrue(
            not points[5].serve()
            and not points[5].serve(True)
            and points[5].serve(False)
        )

    def test_points_text_score(self):
        det_game = DetailedGame(
            "01100110100111", left_wingame=True, left_opener=True, tiebreak=False
        )
        points_0_0 = list(det_game.points_text_score(("0", "0")))
        self.assertEqual(points_0_0[0].text_score(), ("0", "0"))
        self.assertEqual(1, len(points_0_0))

        points_40_40 = list(det_game.points_text_score(("40", "40")))
        self.assertEqual(4, len(points_40_40))
        for pnt in points_40_40:
            self.assertEqual(pnt.text_score(), ("40", "40"))

        points_a_40 = list(det_game.points_text_score(("A", "40")))
        self.assertEqual(3, len(points_a_40))
        for pnt in points_a_40:
            self.assertEqual(pnt.text_score(), ("A", "40"))


def beforekeys_test(afterkeys):
    """сделано для тестирования аналогичной схемы внедренной в старый класс"""

    def get_before_key_test(key):
        if key[-1] in ((7, 6), (6, 7)):
            lstset_sc = (6, 6)
        elif key[-1] in ((1, 0), (0, 1)):
            lstset_sc = (0, 0)
        else:
            raise DetailedScoreError("get_before_key_test bad key: {}".format(key))
        return key[0 : len(key) - 1] + (lstset_sc,)

    if bool(afterkeys):
        sets_num = 0
        beforekey = get_before_key_test(afterkeys[0])
        for afterkey in afterkeys:
            if len(afterkey) > sets_num:
                sets_num = len(afterkey)
                beforekey = get_before_key_test(afterkey)
            yield beforekey
            beforekey = afterkey


class BeforeKeysTest(unittest.TestCase):
    afterkeys = None
    beforekeys = None

    @classmethod
    def setUpClass(cls):
        # after style
        cls.afterkeys = [
            ((1, 0),),
            ((1, 1),),
            ((1, 2),),
            ((2, 2),),
            ((3, 2),),
            ((4, 2),),
            ((5, 2),),
            ((6, 2),),
            ((6, 2), (0, 1)),
        ]

        # before style
        cls.beforekeys = [
            ((0, 0),),
            ((1, 0),),
            ((1, 1),),
            ((1, 2),),
            ((2, 2),),
            ((3, 2),),
            ((4, 2),),
            ((5, 2),),
            ((6, 2), (0, 0)),
        ]

    def test_beforekeys(self):
        self.assertEqual(self.beforekeys, list(beforekeys_test(self.afterkeys)))


class TestWtaDbRead(unittest.TestCase):
    dbdet = open_db(sex="wta")

    @classmethod
    def setUpClass(cls):
        cls.dbdet.query_matches()

    def test_read_navarro_floris_fes2010(self):
        """6-1 5-7 6-1"""
        score = Score("6-1 5-7 6-1")
        det_score = self.dbdet.get_detailed_score(
            tour_id=6602, rnd="First", left_id=5873, right_id=661
        )
        for setnum in (1, 2, 3):
            set_items = SetItems.from_scores(setnum, det_score, score)
            if setnum in (1, 3):
                self.assertTrue(set_items.ok_set())
            n_steps = 0
            for idx, (it, dg) in enumerate(set_items):
                if setnum in (1, 3):
                    self.assertTrue(set_items.ok_idx(idx))
                self.assertTrue(type(it) == tuple and len(it) == 2)
                self.assertTrue(type(dg) == DetailedGame and not dg.tiebreak)
                n_steps += 1
            self.assertEqual(n_steps, len(set_items))

    def test_read_boulter_makarova_ao2019_supertie(self):
        """6-0 4-6 7-6 (tie: 10-6, max dif was 9-6).
        it is supertie: in dec set >= 2019 AO(when 6-6) and Wim(when 12-12)"""
        det_score = self.dbdet.get_detailed_score(
            tour_id=12357, rnd="First", left_id=14175, right_id=4159
        )
        self.assertEqual(match_win_side(det_score), co.LEFT)
        self.assertTrue(det_score is not None)
        dec_tie = list(det_score.values())[-1]
        self.assertTrue(dec_tie.tiebreak)
        self.assertTrue(dec_tie.supertiebreak)
        self.assertTrue(dec_tie.valid)
        self.assertEqual(dec_tie.final_num_score(left=True, before=False), (10, 6))
        points = list(iter(dec_tie))
        self.assertEqual(len(points), 16)
        max_left_gp = 0
        for pnt in points:
            left_gp = pnt.game_point_score(left=True)
            if left_gp > max_left_gp:
                max_left_gp = left_gp
        self.assertEqual(max_left_gp, 3)  # at moment when 9-6
        self.check_is_full(det_score, Score("6-0 4-6 7-6(6)"))

    def test_read_lepchenko_tomova_ao2019_supertie(self):
        """4-6 6-2 7-6 (tie: 10-6, max dif was 9-4).
        it is supertie: in dec set >= 2019 AO(when 6-6) and Wim(when 12-12)"""
        det_score = self.dbdet.get_detailed_score(
            tour_id=12357, rnd="q-First", left_id=461, right_id=12432
        )
        self.assertEqual(match_win_side(det_score), co.LEFT)
        self.assertTrue(det_score is not None)
        dec_tie = list(det_score.values())[-1]
        self.assertTrue(dec_tie.tiebreak)
        self.assertTrue(dec_tie.supertiebreak)
        self.assertTrue(dec_tie.valid)
        self.assertEqual(dec_tie.final_num_score(left=True, before=False), (10, 6))
        points = list(iter(dec_tie))
        self.assertEqual(len(points), 16)
        max_left_gp = 0
        for pnt in points:
            left_gp = pnt.game_point_score(left=True)
            if left_gp > max_left_gp:
                max_left_gp = left_gp
        self.assertEqual(max_left_gp, 5)  # at moment when 9-4
        self.check_is_full(det_score, Score("4-6 6-2 7-6(6)"))

    def test_read_kvitova_goerges_spb2018(self):
        """7-5 4-6 6-2 Kvitova opens set1, at 5-5 she holds, then at 6-5 break"""
        det_score = self.dbdet.get_detailed_score(
            tour_id=11757, rnd="1/2", left_id=8234, right_id=7102
        )
        self.assertEqual(match_win_side(det_score), co.LEFT)
        self.assertTrue(det_score is not None)
        if det_score is not None:
            self.assertTrue(det_score.set_valid(setnum=1))
            self.assertTrue(det_score.set_valid(setnum=2))
            self.assertTrue(det_score.set_valid(setnum=3))
            s1_55hold = det_score.item(
                (
                    lambda k, dg: len(k) == 1
                    and k[0] == (5, 5)
                    and dg.opener_wingame
                    and dg.left_opener
                )
            )
            self.assertTrue(s1_55hold is not None)
            s1_66 = det_score.item(lambda k, dg: len(k) == 1 and k[0] == (6, 6))
            self.assertTrue(s1_66 is None)
            self.check_is_full(det_score, Score("7-5 4-6 6-2"))

    def check_is_full(self, det_score, score):
        full = is_full(det_score, score)
        self.assertEqual(full, True)


class DetailedScoreTest(unittest.TestCase):
    def test_empty(self):
        ds = DetailedScore()
        self.assertEqual(bool(ds), False)

    def test_set_items(self):
        ds = DetailedScoreTest.make_2set_det_score()
        self.assertEqual(match_win_side(ds), co.RIGHT)
        set_items = list(ds.set_items(setnum=2))
        self.assertEqual(len(set_items), 6)

        beg_item = set_items[0]
        self.assertEqual(
            beg_item[0],
            (
                (6, 7),
                (0, 0),
            ),
        )
        self.assertEqual(
            beg_item[1],
            DetailedGame("1111", left_wingame=False, left_opener=False, tiebreak=False),
        )

        end_item = set_items[-1]
        self.assertEqual(
            end_item[0],
            (
                (6, 7),
                (0, 5),
            ),
        )
        self.assertEqual(
            end_item[1],
            DetailedGame("01000", left_wingame=False, left_opener=True, tiebreak=False),
        )

    def test_save_restore(self):
        ds = DetailedScoreTest.make_2set_det_score()
        text = ds.tostring()
        ds2 = DetailedScore.from_str(text)
        self.assertEqual(ds, ds2)

    def test_save_restore2(self):
        items = [
            (
                ((0, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((1, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
        ]
        ds = DetailedScore(items, retired=True)
        ds.error = error_code("SET5_SCORE_PROBLEMS") | error_code("SET4_SCORE_PROBLEMS")
        text = ds.tostring()
        ds2 = DetailedScore.from_str(text)
        self.assertEqual(ds, ds2)

    def test_set_valid(self):
        dsbad = DetailedScoreTest.make_2set_det_score_badend()
        self.assertTrue(dsbad.set_valid(setnum=1))
        self.assertFalse(dsbad.set_valid(setnum=2))

        ds = DetailedScoreTest.make_2set_det_score()
        self.assertTrue(ds.set_valid(setnum=1))
        self.assertTrue(ds.set_valid(setnum=2))

    def test_decsupertie_is_full(self):
        det_score = self.make_3set_det_score_decsupertie()
        last_item = det_score.last_item()
        self.assertEqual(last_item[0], ((7, 6), (0, 6), (0, 0)))
        self.assertTrue(last_item[1].tiebreak)
        self.assertTrue(last_item[1].supertiebreak)
        full = is_full(det_score, Score("7-6 0-6 10-3", decsupertie=True))
        self.assertEqual(full, True)

    @staticmethod
    def make_3set_det_score_decsupertie():
        """return DetailedScore variant for Score("7-6 0-6 10-3", decsupertie=True)"""
        items = [
            (
                ((0, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((1, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((2, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((3, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 0),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 1),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 2),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 3),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 4),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 5),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((5, 5),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((5, 6),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((6, 6),),
                DetailedGame(
                    "1111111", left_wingame=True, left_opener=True, tiebreak=True
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 0),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 1),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 2),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 3),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 4),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 5),
                ),
                DetailedGame(
                    "01000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (7, 6),
                    (0, 6),
                    (0, 0),
                ),
                DetailedGame(
                    "1110000000000",
                    left_wingame=True,
                    left_opener=False,
                    tiebreak=True,
                    supertiebreak=True,
                ),
            ),
        ]
        return DetailedScore(items)

    @staticmethod
    def make_2set_det_score():
        """:return DetailedScore for 6-7(0) 0-6 where match left_opener"""
        items = [
            (
                ((0, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((1, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((2, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((3, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 0),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 1),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 2),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 3),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 4),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 5),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((5, 5),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((5, 6),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((6, 6),),
                DetailedGame(
                    "0000000", left_wingame=False, left_opener=True, tiebreak=True
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 0),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 1),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 2),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 3),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 4),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 5),
                ),
                DetailedGame(
                    "01000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
        ]
        return DetailedScore(items)

    @staticmethod
    def make_2set_det_score_badend():
        items = [
            (
                ((0, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((1, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((2, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((3, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 0),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 1),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 2),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 3),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 4),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 5),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((5, 5),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((5, 6),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((6, 6),),
                DetailedGame(
                    "0000000", left_wingame=False, left_opener=True, tiebreak=True
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 0),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 1),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 2),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 3),
                ),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                (
                    (6, 7),
                    (0, 4),
                ),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
        ]
        return DetailedScore(items)


class SetItemsTest(unittest.TestCase):
    def test_make_empty(self):
        ds = DetailedScoreTest.make_2set_det_score()
        score = Score("6-7(0) 0-6")
        s3_items = SetItems.from_scores(setnum=3, detailed_score=ds, score=score)
        self.assertEqual(bool(s3_items), False)

    def test_set_items(self):
        ds = DetailedScoreTest.make_2set_det_score()
        score = Score("6-7(6) 0-6")
        s1_items = SetItems.from_scores(setnum=1, detailed_score=ds, score=score)
        self.assertEqual(s1_items.set_opener_side(), co.LEFT)
        self.assertEqual(s1_items.ok_set(), True)
        self.assertEqual(s1_items.fin_scr, (6, 7))
        self.assertEqual(s1_items.tie_scr, (6, 8))
        self.assertEqual(s1_items.exist_scr((0, 0), left_opener=True), True)
        self.assertEqual(s1_items.exist_scr((0, 0), left_opener=False), False)
        self.assertEqual(s1_items.exist_scr((6, 6), left_opener=False), False)

        s3_items = SetItems.from_scores(setnum=3, detailed_score=ds, score=score)
        self.assertEqual(s3_items.set_opener_side(), None)
        self.assertEqual(s3_items.ok_set(), False)
        self.assertEqual(bool(s3_items), False)
        self.assertEqual(s3_items.exist_scr((0, 0), left_opener=True), None)
        self.assertEqual(s3_items.exist_scr((0, 0), left_opener=False), None)

        s1_items.flip()
        self.assertEqual(s1_items.set_opener_side(), co.RIGHT)
        self.assertEqual(s1_items.exist_scr((0, 0), left_opener=True), False)


class BreaksMovesTest(unittest.TestCase):
    def test_break_moves(self):
        ds = BreaksMovesTest.make_1set_det_score()
        brk_moves = list(ds.breaks_moves(lambda k, v: len(k) == 1 and v.valid))
        waited = [
            BreaksMove(initor=True, count=2, back_count=2),
            BreaksMove(initor=False, count=1, back_count=1),
            BreaksMove(initor=False, count=1, back_count=1),
        ]
        self.assertEqual(brk_moves, waited)

    @staticmethod
    def make_1set_det_score():
        items = [
            (
                ((0, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((1, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((2, 0),),
                DetailedGame(
                    "1111", left_wingame=True, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((3, 0),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 0),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 1),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 2),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 3),),
                DetailedGame(
                    "1111", left_wingame=False, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((4, 4),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((4, 5),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((5, 5),),
                DetailedGame(
                    "0000", left_wingame=False, left_opener=True, tiebreak=False
                ),
            ),
            (
                ((5, 6),),
                DetailedGame(
                    "0000", left_wingame=True, left_opener=False, tiebreak=False
                ),
            ),
            (
                ((6, 6),),
                DetailedGame(
                    "0000000", left_wingame=False, left_opener=True, tiebreak=True
                ),
            ),
        ]
        return DetailedScore(items)


if __name__ == "__main__":
    # log.initialize(co.logname(__file__, test=True), 'debug', 'debug')
    unittest.main()
