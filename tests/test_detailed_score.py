import pytest

import common as co

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
import get_det_scores


class TestTiebreakEnumPoints:
    @staticmethod
    def test_enumerate():
        det_game = DetailedGame(
            "001101010111", left_wingame=True, left_opener=True, tiebreak=False
        )
        pnts = list(iter(det_game))
        assert pnts[0].num_score(before=False) == (0, 1)
        assert pnts[1].num_score(before=False) == (0, 2)
        assert pnts[2].num_score(before=False) == (1, 2)
        assert pnts[3].num_score(before=False) == (2, 2)
        assert pnts[4].num_score(before=False) == (2, 3)
        assert pnts[5].num_score(before=False) == (3, 3)
        assert pnts[6].num_score(before=False) == (3, 4)
        assert pnts[7].num_score(before=False) == (4, 4)
        assert pnts[8].num_score(before=False) == (4, 5)
        assert pnts[9].num_score(before=False) == (5, 5)
        assert pnts[10].num_score(before=False) == (6, 5)
        assert pnts[11].num_score(before=False) == (7, 5)

    @staticmethod
    def test_enumerate_tie():
        det_game = DetailedGame(
            "0001111111", left_wingame=False, left_opener=False, tiebreak=True
        )
        pnts = list(iter(det_game))
        assert pnts[0].num_score(before=False) == (0, 1)
        assert pnts[1].num_score(before=False) == (0, 2)
        assert pnts[2].num_score(before=False) == (0, 3)
        assert pnts[3].num_score(before=False) == (1, 3)
        assert pnts[-1].num_score(before=False) == (7, 3)

    @staticmethod
    def test_enumerate_tie_2():
        det_game = DetailedGame(
            "1111111", left_wingame=False, left_opener=False, tiebreak=True
        )
        pnts = list(iter(det_game))
        assert pnts[0].num_score(before=False) == (1, 0)
        assert pnts[1].num_score(before=False) == (2, 0)
        assert pnts[-1].num_score(before=False) == (7, 0)


class TestTiebreakCounts:
    @staticmethod
    def test_leadchanges_count():
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        assert leadchanges_count == 3

        det_game = DetailedGame(
            "01100110111", left_wingame=False, left_opener=False, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        assert leadchanges_count == 3

        det_game = DetailedGame(
            "0000000", left_wingame=False, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        assert leadchanges_count == 0

        det_game = DetailedGame(
            "01111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        leadchanges_count = tie_leadchanges_count(det_game)
        assert leadchanges_count == 1

    @staticmethod
    def test_minibreaks_count():
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        minibreaks_count = tie_minibreaks_count(det_game)
        assert minibreaks_count == 10

    @staticmethod
    def test_minibreaks_diff():
        det_game = DetailedGame(
            "01100110111", left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        assert points[0].minibreaks_diff() == -1
        assert points[0].minibreaks_diff(left=False) == 1

        assert points[1].minibreaks_diff() == 0
        assert points[1].minibreaks_diff(left=False) == 0

        assert points[2].minibreaks_diff() == 1
        assert points[2].minibreaks_diff(left=False) == -1


class TestDetailedGame:
    def test_final_num_score(self):
        dg = DetailedGame.from_str("wot_E32768PLWLWL11011")
        fin_num_scr = dg.final_num_score(before=False)
        assert fin_num_scr == (6, 4)
        fin_num_scr = dg.final_num_score(before=True)
        assert fin_num_scr == (5, 4)

    @staticmethod
    def test_game_orient_error():
        """д.б. ошибка"""
        det_game = DetailedGame(
            "010110110000", left_wingame=True, left_opener=True, tiebreak=True
        )
        assert det_game.error
        assert det_game.orientation_error()

        det_game = DetailedGame(
            "010110", left_wingame=True, left_opener=True, tiebreak=True
        )
        assert det_game.error
        assert det_game.orientation_error()

        det_game = DetailedGame(
            "001111", left_wingame=False, left_opener=True, tiebreak=False
        )
        assert det_game.error
        assert det_game.orientation_error()

        det_game = DetailedGame(
            "110000", left_wingame=True, left_opener=True, tiebreak=False
        )
        assert det_game.error
        assert det_game.orientation_error()

    @staticmethod
    def test_save_restore():
        det_game = DetailedGame(
            "0110011011",
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("SPANING"),
        )
        text = str(det_game)
        det_game2 = DetailedGame.from_str(text)
        assert det_game == det_game2

    @staticmethod
    def test_save_restore2():
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
        assert det_game == det_game2
        assert 15 == len(list(iter(det_game2)))

    @staticmethod
    def test_state():
        """here error_code value is not meaning and used for test get/set"""
        det_game = DetailedGame(
            "0110011011",
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("GAME_ORIENT"),
        )
        assert det_game.error
        assert det_game.left_wingame
        assert det_game.left_opener
        assert not det_game.tiebreak
        assert det_game.error == error_code("GAME_ORIENT")
        det_game.error = 0
        det_game.left_wingame = False
        det_game.left_opener = False
        det_game.tiebreak = True
        assert not det_game.error
        assert not det_game.left_wingame
        assert not det_game.left_opener
        assert det_game.tiebreak
        det_game.error = error_code("GAME_ORIENT")
        assert det_game.error
        assert det_game.error == error_code("GAME_ORIENT")
        assert not det_game.left_wingame
        assert not det_game.left_opener
        assert det_game.tiebreak

    @staticmethod
    def test_point_text_score():
        det_game = DetailedGame(
            "01100110100111", left_wingame=True, left_opener=True, tiebreak=False
        )
        points = [p for p in det_game]
        assert points[0].text_score() == ("0", "0")
        assert points[1].text_score() == ("0", "15")
        assert points[2].text_score() == ("15", "15")
        assert points[3].text_score() == ("30", "15")
        assert points[4].text_score() == ("30", "30")
        assert points[5].text_score() == ("30", "40")
        assert points[6].text_score() == ("40", "40")
        assert points[7].text_score() == ("A", "40")
        assert points[8].text_score() == ("40", "40")
        assert points[9].text_score() == ("A", "40")
        assert points[10].text_score() == ("40", "40")
        assert points[11].text_score() == ("40", "A")
        assert points[12].text_score() == ("40", "40")
        assert points[13].text_score() == ("A", "40")
        assert points[13].text_score(left=True) == ("A", "40")
        assert points[13].text_score(left=False) == ("40", "A")

    @staticmethod
    def test_point_text_score_tie():
        pts = "100110011100100111"
        det_game = DetailedGame(
            points=pts, left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        assert points[0].text_score() == ("0", "0")
        assert points[1].text_score() == ("1", "0")
        assert points[2].text_score() == ("1", "1")
        assert points[3].text_score() == ("1", "2")
        assert points[4].text_score() == ("2", "2")
        assert points[5].text_score() == ("3", "2")
        assert points[6].text_score() == ("3", "3")
        assert points[7].text_score() == ("3", "4")
        assert points[8].text_score() == ("4", "4")
        assert points[9].text_score() == ("5", "4")
        assert points[10].text_score() == ("6", "4")
        assert points[11].text_score() == ("6", "5")
        assert points[12].text_score() == ("6", "6")
        assert points[13].text_score() == ("7", "6")
        assert points[14].text_score() == ("7", "7")
        assert points[15].text_score() == ("7", "8")
        assert points[16].text_score() == ("8", "8")
        assert points[17].text_score() == ("9", "8")
        assert points[17].text_score(left=True) == ("9", "8")
        assert points[17].text_score(left=False) == ("8", "9")
        assert len(points) == 18

    @staticmethod
    def test_left_break_point_score():
        det_game = DetailedGame(
            "00011100", left_wingame=False, left_opener=True, tiebreak=False
        )
        points = [p for p in det_game]
        assert points[0].break_point_score(left=True) == 0
        assert points[1].break_point_score(left=True) == 0
        assert points[2].break_point_score(left=True) == 0
        assert points[3].break_point_score(left=True) == 3
        assert points[4].break_point_score(left=True) == 2
        assert points[5].break_point_score(left=True) == 1
        assert points[6].break_point_score(left=True) == 0
        assert points[7].break_point_score(left=True) == 1

    @staticmethod
    def test_right_break_point_score():
        det_game = DetailedGame(
            "0001110111", left_wingame=False, left_opener=False, tiebreak=False
        )
        points = [p for p in det_game]
        assert points[0].break_point_score(left=False) == 0
        assert points[1].break_point_score(left=False) == 0
        assert points[2].break_point_score(left=False) == 0
        assert points[3].break_point_score(left=False) == 3
        assert points[4].break_point_score(left=False) == 2
        assert points[5].break_point_score(left=False) == 1
        assert points[6].break_point_score(left=False) == 0
        assert points[7].break_point_score(left=False) == 1
        assert points[8].break_point_score(left=False) == 0
        assert points[9].break_point_score(left=False) == 0

    @staticmethod
    def test_few_points():
        pts = "1"
        det_game = DetailedGame(
            points=pts,
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("POINT_SCORE_FEW_DATA"),
        )
        points = [p for p in det_game]
        assert len(points) == 1
        assert points[0].win() == True

        pts = ""
        det_game = DetailedGame(
            points=pts,
            left_wingame=True,
            left_opener=True,
            tiebreak=False,
            error=error_code("POINT_SCORE_FEW_DATA"),
        )
        points = [p for p in det_game]
        assert len(points) == 0

    @staticmethod
    def test_tie_points_iter():
        pts = "100110011100100111"
        det_game = DetailedGame(
            points=pts, left_wingame=True, left_opener=True, tiebreak=True
        )
        points = [p for p in det_game]
        assert len(points) == len(pts)
        assert points[-1].win_game() and points[-1].win()
        assert points[-2].game_point_score(before=False) and points[-2].win()
        assert points[-3].equal_score(before=False) and points[-3].win()
        assert points[-4].equal_score(before=True)
        assert points[-4].break_point_score(before=False) and points[-4].loss()        
        assert points[-5].equal_score(before=False) and points[-5].loss()
        assert points[-6].game_point_score(before=False)
        assert points[-7].equal_score(before=False)
        assert len([p for p in points if p.equal_score(before=False)]) == 3
        assert len([p for p in points if p.game_point_score()]) == 4
        assert len([p for p in points if p.break_point_score(before=False)]) == 1
        
        # win_minibreak():
        assert (
            points[9].win()
            and not points[9].serve()
            and points[9].win_minibreak(True)
            and points[9].loss_minibreak(False)
        )
        # serve:
        assert (
            points[0].serve(True)
            and points[0].win(True)
            and not points[0].win_minibreak(True)
        )
        assert (
            not points[1].serve(True)
            and points[1].loss(True)
            and not points[1].win_minibreak(True)
        )
        assert (
            not points[2].serve(True)
            and points[2].loss(True)
            and not points[2].win_minibreak(True)
        )
        assert (
            points[3].serve() and points[3].serve(True) and not points[3].serve(False)
        )
        assert (
            points[4].serve() and points[4].serve(True) and not points[4].serve(False)
        )
        assert (
            not points[5].serve()
            and not points[5].serve(True)
            and points[5].serve(False)
        )

    @staticmethod
    def test_points_text_score():
        det_game = DetailedGame(
            "01100110100111", left_wingame=True, left_opener=True, tiebreak=False
        )
        points_0_0 = list(det_game.points_text_score(("0", "0")))
        assert points_0_0[0].text_score() == ("0", "0")
        assert 1 == len(points_0_0)

        points_40_40 = list(det_game.points_text_score(("40", "40")))
        assert 4 == len(points_40_40)
        for pnt in points_40_40:
            assert pnt.text_score() == ("40", "40")

        points_a_40 = list(det_game.points_text_score(("A", "40")))
        assert 3 == len(points_a_40)
        for pnt in points_a_40:
            assert pnt.text_score() == ("A", "40")


class TestBeforeKeys:

    @staticmethod
    def get_beforekeys(afterkeys):
        """сделано для тестирования аналогичной схемы внедренной в старый класс"""

        def get_before_key_impl(key):
            if key[-1] in ((7, 6), (6, 7)):
                lstset_sc = (6, 6)
            elif key[-1] in ((1, 0), (0, 1)):
                lstset_sc = (0, 0)
            else:
                raise DetailedScoreError("get_before_key_test bad key: {}".format(key))
            return key[0 : len(key) - 1] + (lstset_sc,)

        if bool(afterkeys):
            sets_num = 0
            beforekey = get_before_key_impl(afterkeys[0])
            for afterkey in afterkeys:
                if len(afterkey) > sets_num:
                    sets_num = len(afterkey)
                    beforekey = get_before_key_impl(afterkey)
                yield beforekey
                beforekey = afterkey

    @staticmethod
    def test_beforekeys():
        # after style
        afterkeys = [
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
        beforekeys = [
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
        assert beforekeys == list(TestBeforeKeys.get_beforekeys(afterkeys))


@pytest.mark.detscoredb
def test_dbread_navarro_floris_fes2010(all_dbdet_wta):
    det_score, scor = get_det_scores.scores_navarro_floris_fes2010(all_dbdet_wta)
    assert scor == Score('6-1 5-7 6-1')
    for setnum in (1, 2, 3):
        set_items = SetItems.from_scores(setnum, det_score, scor)
        if setnum in (1, 3):
            assert set_items.ok_set()
        n_steps = 0
        for idx, (it, dg) in enumerate(set_items):
            if setnum in (1, 3):
                assert set_items.ok_idx(idx)
            assert type(it) == tuple and len(it) == 2
            assert type(dg) == DetailedGame and not dg.tiebreak
            n_steps += 1
        assert n_steps == len(set_items)


@pytest.mark.detscoredb
def test_dbread_boulter_makarova_ao2019_supertie(all_dbdet_wta):
    """ (supertie: 10-6, max dif was 9-6).
        it is supertie: in dec set >= 2019 AO(when 6-6) and Wim(when 12-12)"""
    det_score, scor = get_det_scores.scores_boulter_makarova_2019_ao(all_dbdet_wta)
    assert scor == Score("6-0 4-6 7-6(6)")
    assert match_win_side(det_score) == co.LEFT
    assert det_score is not None
    dec_tie = list(det_score.values())[-1]
    assert dec_tie.tiebreak
    assert dec_tie.supertiebreak
    assert dec_tie.valid
    assert dec_tie.final_num_score(left=True, before=False) == (10, 6)
    points = list(iter(dec_tie))
    assert len(points) == 16
    max_left_gp = 0
    for pnt in points:
        left_gp = pnt.game_point_score(left=True)
        if left_gp > max_left_gp:
            max_left_gp = left_gp
    assert max_left_gp == 3  # at moment when 9-6
    assert is_full(det_score, scor)


@pytest.mark.detscoredb
def test_dbread_lepchenko_tomova_ao2019_supertie(all_dbdet_wta):
    """ (supertie: 10-6, max dif was 9-4).
        it is supertie: in dec set >= 2019 AO(when 6-6) and Wim(when 12-12)"""
    det_score, scor = get_det_scores.scores_lepchenko_tomova_ao2019(all_dbdet_wta)
    assert scor == Score("4-6 6-2 7-6(6)")
    assert match_win_side(det_score) == co.LEFT
    assert det_score is not None
    dec_tie = list(det_score.values())[-1]
    assert dec_tie.tiebreak
    assert dec_tie.supertiebreak
    assert dec_tie.valid
    assert dec_tie.final_num_score(left=True, before=False) == (10, 6)
    points = list(iter(dec_tie))
    assert len(points) == 16
    max_left_gp = 0
    for pnt in points:
        left_gp = pnt.game_point_score(left=True)
        if left_gp > max_left_gp:
            max_left_gp = left_gp
    assert max_left_gp == 5  # at moment when 9-4
    assert is_full(det_score, scor)


@pytest.mark.detscoredb
def test_dbread_kvitova_goerges_spb2018(all_dbdet_wta):
    """ Kvitova opens set1, at 5-5 she holds, then at 6-5 break """
    det_score, scor = get_det_scores.scores_kvitova_goerges_spb2018(all_dbdet_wta)
    assert scor == Score("7-5 4-6 6-2")
    assert match_win_side(det_score) == co.LEFT
    assert det_score is not None
    if det_score is not None:
        assert det_score.set_valid(setnum=1)
        assert det_score.set_valid(setnum=2)
        assert det_score.set_valid(setnum=3)
        s1_55hold = det_score.item(
            (
                lambda k, dg: len(k) == 1
                and k[0] == (5, 5)
                and dg.opener_wingame
                and dg.left_opener
            )
        )
        assert s1_55hold is not None
        s1_66 = det_score.item(lambda k, dg: len(k) == 1 and k[0] == (6, 6))
        assert s1_66 is None
        assert is_full(det_score, scor)


class TestDetailedScore:
    @staticmethod
    def test_empty():
        ds = DetailedScore()
        assert not bool(ds)

    @staticmethod
    def test_set_items():
        ds = TestDetailedScore.make_2set_det_score()
        assert match_win_side(ds) == co.RIGHT
        set_items = list(ds.set_items(setnum=2))
        assert len(set_items) == 6

        beg_item = set_items[0]
        assert (
            beg_item[0] == (
                (6, 7),
                (0, 0),
            )
        )
        assert (
            beg_item[1] == DetailedGame(
                "1111", left_wingame=False, left_opener=False, tiebreak=False)
        )

        end_item = set_items[-1]
        assert (
            end_item[0] == (
                (6, 7),
                (0, 5),
            )
        )
        assert (
            end_item[1] == DetailedGame(
                "01000", left_wingame=False, left_opener=True, tiebreak=False),
        )

    @staticmethod
    def test_save_restore():
        ds = TestDetailedScore.make_2set_det_score()
        text = ds.tostring()
        ds2 = DetailedScore.from_str(text)
        assert ds == ds2

    @staticmethod
    def test_save_restore2():
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
        assert ds == ds2

    @staticmethod
    def test_set_valid():
        dsbad = TestDetailedScore.make_2set_det_score_badend()
        assert dsbad.set_valid(setnum=1)
        assert not dsbad.set_valid(setnum=2)

        ds = TestDetailedScore.make_2set_det_score()
        assert ds.set_valid(setnum=1)
        assert ds.set_valid(setnum=2)

    @staticmethod
    def test_decsupertie_is_full():
        det_score = TestDetailedScore.make_3set_det_score_decsupertie()
        last_item = det_score.last_item()
        assert last_item[0] == ((7, 6), (0, 6), (0, 0))
        assert last_item[1].tiebreak
        assert last_item[1].supertiebreak
        assert is_full(det_score, Score("7-6 0-6 10-3", decsupertie=True))

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


class TestSetItems:
    @staticmethod
    def test_make_empty():
        ds = TestDetailedScore.make_2set_det_score()
        scor = Score("6-7(0) 0-6")
        s3_items = SetItems.from_scores(setnum=3, detailed_score=ds, score=scor)
        assert not bool(s3_items)

    @staticmethod
    def test_set_items():
        ds = TestDetailedScore.make_2set_det_score()
        scor = Score("6-7(6) 0-6")
        s1_items = SetItems.from_scores(setnum=1, detailed_score=ds, score=scor)
        assert s1_items.set_opener_side() == co.LEFT
        assert s1_items.ok_set()
        assert s1_items.fin_scr == (6, 7)
        assert s1_items.tie_scr == (6, 8)
        assert s1_items.exist_scr((0, 0), left_opener=True)
        assert not s1_items.exist_scr((0, 0), left_opener=False)
        assert not s1_items.exist_scr((6, 6), left_opener=False)

        s3_items = SetItems.from_scores(setnum=3, detailed_score=ds, score=scor)
        assert s3_items.set_opener_side() is None
        assert not s3_items.ok_set()
        assert not bool(s3_items)
        assert s3_items.exist_scr((0, 0), left_opener=True) is None
        assert s3_items.exist_scr((0, 0), left_opener=False) is None

        s1_items.flip()
        assert s1_items.set_opener_side() == co.RIGHT
        assert not s1_items.exist_scr((0, 0), left_opener=True)


class TestBreaksMoves:
    @staticmethod
    def test_break_moves():
        ds = TestBreaksMoves.make_1set_det_score()
        brk_moves = list(ds.breaks_moves(lambda k, v: len(k) == 1 and v.valid))
        waited = [
            BreaksMove(initor=True, count=2, back_count=2),
            BreaksMove(initor=False, count=1, back_count=1),
            BreaksMove(initor=False, count=1, back_count=1),
        ]
        assert brk_moves == waited

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
