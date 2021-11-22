import unittest
from collections import OrderedDict

import common as co
from detailed_score import DetailedGame, DetailedScore
from miss_points import (
    MissPoint,
    detailed_game_miss_points,
    GamePointEvent,
    detailed_score_miss_points,
    failed_count_before,
)


class FromDetailedGameTest(unittest.TestCase):
    def test_tiebreaks(self):
        dg = DetailedGame(
            "0000000", left_wingame=False, left_opener=True, tiebreak=True
        )
        self.assertEqual([], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "11111101", left_wingame=True, left_opener=True, tiebreak=True
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=False, tiebreak=True, multier=6),
            revenged=co.GAME,
        )
        self.assertEqual([mp], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "111111001", left_wingame=True, left_opener=True, tiebreak=True
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=False, tiebreak=True, multier=6),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=True, multier=5),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

    def test_1(self):
        dg = DetailedGame("1111", left_wingame=True, left_opener=True, tiebreak=False)
        self.assertEqual([], list(detailed_game_miss_points(dg)))

        dg = DetailedGame("11101", left_wingame=True, left_opener=True, tiebreak=False)
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=3),
            revenged=co.GAME,
        )
        self.assertEqual([mp], list(detailed_game_miss_points(dg)))

    def test_2(self):
        dg = DetailedGame(
            "11010011", left_wingame=True, left_opener=True, tiebreak=False
        )
        mp = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=2),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=1),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

        dg = DetailedGame(
            "1101000111", left_wingame=True, left_opener=True, tiebreak=False
        )
        mp3r = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=1),
            revenged=None,
        )
        self.assertEqual([mp, mp2, mp3r], list(detailed_game_miss_points(dg)))

    def test_2_right(self):
        dg = DetailedGame(
            "00101100", left_wingame=False, left_opener=True, tiebreak=False
        )
        mp = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=2),
            revenged=co.GAME,
        )
        mp2 = MissPoint(
            GamePointEvent(left_side=False, left_serve=True, tiebreak=False, multier=1),
            revenged=co.GAME,
        )
        self.assertEqual([mp, mp2], list(detailed_game_miss_points(dg)))

        # -----------------------------------------
        dg = DetailedGame(
            "0010111000", left_wingame=False, left_opener=True, tiebreak=False
        )
        mp3L = MissPoint(
            GamePointEvent(left_side=True, left_serve=True, tiebreak=False, multier=1),
            revenged=None,
        )
        self.assertEqual([mp, mp2, mp3L], list(detailed_game_miss_points(dg)))


class CountMissDictTest(unittest.TestCase):
    def test_failed_count_before(self):
        mdict = MakeMissDictTest.twosets_dscore_mdict()[1]
        # предполагаем что там 5 упущенных сетболов левого игрока и все в 1 сете:
        # два из них до ((6,5),) и три позже
        cnt_before = failed_count_before(
            co.LEFT, mdict, ((6, 5),), co.SET, last_set_only=True
        )
        self.assertEqual(2, cnt_before)

        cnt_before = failed_count_before(
            co.LEFT, mdict, ((7, 6), (6, 6)), co.SET, last_set_only=True
        )
        self.assertEqual(0, cnt_before)

        cnt_before = failed_count_before(
            co.LEFT, mdict, ((7, 6), (6, 6)), co.SET, last_set_only=False
        )
        self.assertEqual(5, cnt_before)


class MakeMissDictTest(unittest.TestCase):
    @staticmethod
    def oneset_dscore_mdict():
        ds = DetailedScore()
        mdict = OrderedDict()
        ds[((4, 4),)] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((5, 4),)] = DetailedGame(
            "10001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((5, 4),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
        ]
        ds[((5, 5),)] = DetailedGame(
            "11101", left_wingame=True, left_opener=True, tiebreak=False
        )
        mdict[((5, 5),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=True, tiebreak=False, multier=3
                ),
                revenged=co.GAME,
            )
        ]
        ds[((6, 5),)] = DetailedGame(
            "00011111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((6, 5),)] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=3
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.SET,
            ),
        ]
        ds[((6, 6),)] = DetailedGame(
            "1111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        return ds, mdict

    @staticmethod
    def twosets_dscore_mdict():
        ds, mdict = MakeMissDictTest.oneset_dscore_mdict()
        ds[((7, 6), (0, 0))] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (1, 0))] = DetailedGame(
            "1111", left_wingame=False, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (1, 1))] = DetailedGame(
            "01111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 1))] = DetailedGame(
            "001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (2, 2))] = DetailedGame(
            "110000", left_wingame=False, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 3))] = DetailedGame(
            "00011111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((7, 6), (2, 3))] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=3
                ),
                revenged=None,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=None,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=None,
            ),
        ]
        ds[((7, 6), (2, 4))] = DetailedGame(
            "110000", left_wingame=False, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (2, 5))] = DetailedGame(
            "0011000", left_wingame=True, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (3, 5))] = DetailedGame(
            "11000111", left_wingame=True, left_opener=True, tiebreak=False
        )
        mdict[((7, 6), (3, 5))] = [
            MissPoint(
                GamePointEvent(
                    left_side=False, left_serve=True, tiebreak=False, multier=1
                ),
                revenged=None,
                scope=co.SET,
            )
        ]
        ds[((7, 6), (4, 5))] = DetailedGame(
            "10000", left_wingame=True, left_opener=False, tiebreak=False
        )
        ds[((7, 6), (5, 5))] = DetailedGame(
            "1111", left_wingame=True, left_opener=True, tiebreak=False
        )
        ds[((7, 6), (6, 5))] = DetailedGame(
            "10001111", left_wingame=False, left_opener=False, tiebreak=False
        )
        mdict[((7, 6), (6, 5))] = [
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=2
                ),
                revenged=co.SET,
                scope=co.MATCH,
            ),
            MissPoint(
                GamePointEvent(
                    left_side=True, left_serve=False, tiebreak=False, multier=1
                ),
                revenged=co.SET,
                scope=co.MATCH,
            ),
        ]
        ds[((7, 6), (6, 6))] = DetailedGame(
            "1111111", left_wingame=True, left_opener=True, tiebreak=True
        )
        return ds, mdict

    def test_oneset_match(self):
        ds, mdict_wait = self.oneset_dscore_mdict()
        mdict = detailed_score_miss_points(ds, best_of_five=False)
        self.assertEqual(mdict_wait, mdict)

    def test_twosets_match(self):
        ds, mdict_wait = self.twosets_dscore_mdict()
        mdict = detailed_score_miss_points(ds, best_of_five=False)
        self.assertEqual(mdict_wait, mdict)


if __name__ == "__main__":
    unittest.main()
