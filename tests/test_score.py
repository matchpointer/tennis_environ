import unittest
import doctest

from score import (
    tie_opener_serve_at,
    Score,
    completed_score,
    prev_game_score,
    score_tail_gap_iter,
    set_tail_gaps,
)


class TieWhoServeTest(unittest.TestCase):
    def test_opener_serve_at(self):
        self.assertTrue(tie_opener_serve_at((0, 0)))

        self.assertFalse(tie_opener_serve_at((0, 1)))
        self.assertFalse(tie_opener_serve_at((1, 0)))
        self.assertFalse(tie_opener_serve_at((1, 1)))
        self.assertFalse(tie_opener_serve_at((0, 2)))

        self.assertTrue(tie_opener_serve_at((2, 1)))
        self.assertTrue(tie_opener_serve_at((0, 3)))
        self.assertTrue(tie_opener_serve_at((4, 0)))

        self.assertFalse(tie_opener_serve_at((5, 0)))
        self.assertFalse(tie_opener_serve_at((0, 6)))


class ScoreTest(unittest.TestCase):
    def test_empty(self):
        empty_sc = Score("")
        self.assertTrue(len(empty_sc) == 0)
        self.assertTrue(str(empty_sc) == "")
        self.assertEqual(bool(empty_sc), False)

    def test_withdraw_empty(self):
        empty_sc = Score("w/o")
        self.assertTrue(len(empty_sc) == 0)
        self.assertTrue(str(empty_sc) == "ret.")
        self.assertEqual(bool(empty_sc), False)

    def test_valid(self):
        self.assertTrue(Score("6-4 7-6(5)").valid())
        self.assertTrue(Score("6-4 5-7 7-6(8)").valid())
        self.assertTrue(Score("6-4 3-6 2-6 7-6(12) 22-20").valid())

        self.assertFalse(Score("6-4 3-6 2-6 7-6(12) 2-6 22-20").valid())
        self.assertFalse(Score("6-4 6-7(4) 3-6 6-4").valid())
        self.assertFalse(Score("6-4 6-7(4) 3-6").valid())
        self.assertFalse(Score("6-4 6-7(3)").valid())
        self.assertFalse(Score("6-4").valid())
        self.assertFalse(Score("7-5 3-6 10-4").valid())
        self.assertFalse(Score("").valid())

    def test_slice(self):
        scr = Score("6-4 6-7(7) 7-6")
        self.assertEqual(scr[1:], [(6, 7), (7, 6)])

    def test_sets_count(self):
        self.assertEqual(3, Score("6-4 6-7(7) 7-6").sets_count(full=True))
        self.assertEqual(2, Score("6-4 6-7(7) 6-6 ret.").sets_count(full=True))
        self.assertEqual(3, Score("6-4 6-7(7) 6-6 ret.").sets_count(full=False))
        self.assertEqual(2, Score("6-4 6-7(7) 6-5 ret.").sets_count(full=True))

    def test_retired(self):
        self.assertFalse(Score("6-4 7-6(7)").retired)
        self.assertFalse(Score("6-4 5-7 7-6(2)").retired)
        self.assertFalse(Score("6-4 3-6 2-6 7-6(12) 22-20").retired)

        self.assertTrue(Score("6-4 2-6 ret.").retired)
        self.assertTrue(Score("6-4 5-7 1-1 n/p").retired)
        self.assertTrue(Score("6-4 3-6 2-6 7-5(12) def.").retired)

    def test_equal(self):
        self.assertEqual(Score("6-4 7-6(3)"), Score("6-4 7-6(4)"))
        self.assertEqual(Score("6-4 7-6(3)"), Score("6-4 7-6"))
        # self.assertEqual(Score('6-4 6-7(3)'), Score('6-4 6-7 0-0'))

    def test_prev_game(self):
        self.assertEqual(prev_game_score(Score("6-4 7-6")), Score("6-4 6-6"))
        self.assertEqual(prev_game_score(Score("6-4 7-6(4)")), Score("6-4 6-6(4)"))
        self.assertEqual(prev_game_score(Score("6-7 7-5")), Score("6-7 6-5"))

    def test_reversed(self):
        self.assertEqual(Score("6-4 7-6").fliped(), Score("4-6 6-7"))
        self.assertEqual(Score("6-4 0-6 6-6").fliped(), Score("4-6 6-0 6-6"))

    def test_completed_score(self):
        completed = completed_score(Score("6-4 3-3"))
        self.assertEqual(completed, Score("6-4 6-3"))

        self.assertEqual(completed_score(Score("6-4 2-3")), Score("6-4 6-3"))
        self.assertEqual(completed_score(Score("6-4 4-5")), Score("6-4 7-5"))
        self.assertEqual(completed_score(Score("4-6 2-5")), Score("4-6 7-5 6-0"))

        self.assertEqual(completed_score(Score("6-5")), Score("7-5 6-0"))

        self.assertEqual(completed_score(Score("0-6")), Score("0-6 6-0 6-0"))
        self.assertEqual(completed_score(Score("0-5")), Score("7-5 6-0"))

        self.assertEqual(
            completed_score(Score("6-4 2-3"), best_of_five=True), Score("6-4 6-3 6-0")
        )
        self.assertEqual(
            completed_score(Score("6-2 1-6 2-3"), best_of_five=True),
            Score("6-2 1-6 6-3 6-0"),
        )
        self.assertEqual(
            completed_score(Score("2-6 1-6 2-3"), best_of_five=True),
            Score("2-6 1-6 6-3 6-0 6-0"),
        )

    def test_tie_loser_result(self):
        scr = Score("6-4 7-6(5)")
        self.assertTrue(scr.tie_loser_result(setnum=2) == 5)

        scr = Score("6-4 3-6 2-6 7-6(12) 22-20")
        self.assertTrue(scr.tie_loser_result(setnum=4) == 12)
        self.assertTrue(scr.tie_loser_result(setnum=1) is None)

    def test_decsupertie(self):
        scr = Score("6-4 6-7(5) 12-10", decsupertie=True)
        self.assertTrue(scr.tie_loser_result(setnum=2) == 5)
        self.assertTrue(scr.tie_loser_result(setnum=3) == 10)
        self.assertTrue(scr.valid())
        self.assertEqual(scr.tupled(), ((6, 4), (6, 7), (1, 0)))
        self.assertEqual(str(scr), "6-4 6-7(5) 12-10 decsupertie")

    def test_decsupertie_retired(self):
        scr = Score("4-6 6-2 6-3 ret.", decsupertie=True)
        self.assertTrue(scr.tie_loser_result(setnum=3) == 3)
        self.assertTrue(scr.valid())
        self.assertEqual(scr.tupled(), ((4, 6), (6, 2), (1, 0)))
        self.assertEqual(str(scr), "4-6 6-2 6-3 ret. decsupertie")


class ScoreTailGapIteratorTest(unittest.TestCase):
    def test_set_tail_gaps(self):
        self.assertEqual(list(set_tail_gaps(start=(4, 2), end=(4, 2))), [])

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 3))), [(5, 3), (6, 3)]
        )

        self.assertEqual(list(set_tail_gaps(start=(5, 4), end=(6, 4))), [(6, 4)])

        self.assertEqual(list(set_tail_gaps(start=(4, 5), end=(4, 6))), [(4, 6)])

        self.assertEqual(
            list(set_tail_gaps(start=(0, 0), end=(2, 1))), [(1, 0), (1, 1), (2, 1)]
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 5))),
            [(5, 3), (5, 4), (5, 5), (6, 5)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 6))),
            [(5, 3), (5, 4), (5, 5), (6, 5), (6, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(7, 6))),
            [(5, 3), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(7, 5))),
            [(5, 3), (5, 4), (5, 5), (6, 5), (7, 5)],
        )

        self.assertEqual(list(set_tail_gaps(start=(6, 5), end=(7, 5))), [(7, 5)])

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 7))),
            [(5, 3), (5, 4), (5, 5), (6, 5), (6, 6), (6, 7)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(8, 6))),
            [(5, 3), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (8, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(2, 4), end=(8, 6))),
            [(3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (8, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 6), end=(8, 6))), [(6, 6), (7, 6), (8, 6)]
        )

        self.assertEqual(list(set_tail_gaps(start=(6, 5), end=(6, 6))), [(6, 6)])

    def test_set_tail_gaps_left_semiopen(self):
        self.assertEqual(
            list(set_tail_gaps(start=(4, 2), end=(4, 2), left_semiopen=True)), []
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 3), left_semiopen=True)),
            [(5, 2), (5, 3)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 4), end=(6, 4), left_semiopen=True)), [(5, 4)]
        )

        self.assertEqual(
            list(set_tail_gaps(start=(4, 5), end=(4, 6), left_semiopen=True)), [(4, 5)]
        )

        self.assertEqual(
            list(set_tail_gaps(start=(0, 0), end=(2, 1), left_semiopen=True)),
            [(0, 0), (1, 0), (1, 1)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 5), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 6), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5), (6, 5)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(7, 6), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5), (6, 5), (6, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(7, 5), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5), (6, 5)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(6, 5), end=(7, 5), left_semiopen=True)), [(6, 5)]
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(6, 7), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5), (6, 5), (6, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 2), end=(8, 6), left_semiopen=True)),
            [(5, 2), (5, 3), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(2, 4), end=(8, 6), left_semiopen=True)),
            [(2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(5, 6), end=(8, 6), left_semiopen=True)),
            [(5, 6), (6, 6), (7, 6)],
        )

        self.assertEqual(
            list(set_tail_gaps(start=(6, 5), end=(6, 6), left_semiopen=True)), [(6, 5)]
        )

    def test_score_tail_gap_iter(self):
        beg = ((6, 4), (5, 3))
        end = ((6, 4), (6, 3))
        self.assertEqual([((6, 4), (6, 3))], list(score_tail_gap_iter(beg, end)))

        beg = ((6, 4), (5, 2))
        end = ((6, 4), (6, 3))
        self.assertEqual(
            [((6, 4), (5, 3)), ((6, 4), (6, 3))], list(score_tail_gap_iter(beg, end))
        )

        beg = ((6, 4), (5, 2))
        end = ((6, 4), (7, 5))
        self.assertEqual(
            [
                ((6, 4), (5, 3)),
                ((6, 4), (5, 4)),
                ((6, 4), (5, 5)),
                ((6, 4), (6, 5)),
                ((6, 4), (7, 5)),
            ],
            list(score_tail_gap_iter(beg, end)),
        )

        beg = ((6, 4), (5, 2))
        end = ((6, 4), (7, 6))
        self.assertEqual(
            [
                ((6, 4), (5, 3)),
                ((6, 4), (5, 4)),
                ((6, 4), (5, 5)),
                ((6, 4), (6, 5)),
                ((6, 4), (6, 6)),
                ((6, 4), (7, 6)),
            ],
            list(score_tail_gap_iter(beg, end)),
        )

    def test_score_tail_gap_iter_left_semiopen(self):
        beg = ((6, 4), (0, 0))
        end = ((6, 4), (1, 0))
        self.assertEqual(
            [((6, 4), (0, 0))], list(score_tail_gap_iter(beg, end, left_semiopen=True))
        )

        beg = ((6, 4), (5, 2))
        end = ((6, 4), (6, 3))
        self.assertEqual(
            [((6, 4), (5, 2)), ((6, 4), (5, 3))],
            list(score_tail_gap_iter(beg, end, left_semiopen=True)),
        )

        # TODO error ends with ((5,4),(0,0))

    #       beg = ((5,4),)
    #       end = ((6,4), (1,0))
    #       self.assertEqual([((5,4),), ((6,4),(0,0))],
    #                        list(score_tail_gap_iter(beg, end, left_semiopen=True)))

    def test_score_tail_gap_iter_trans2(self):
        beg = ((6, 4),)
        end = ((6, 4), (6, 3))
        full = list(score_tail_gap_iter(beg, end))
        len_wait = 9
        self.assertEqual(len(full), len_wait)
        s1full_wait = [f for (f, s) in full if f == (6, 4)]
        self.assertEqual(len(s1full_wait), len_wait)
        s2full = [s for (f, s) in full]
        self.assertEqual(len(s2full), len_wait)
        s2full_lt = [
            s
            for (i, s) in enumerate(s2full)
            if i < len(s2full) and s[0] < 6 and s[1] <= 4
        ]
        self.assertEqual(len(s2full_lt), len(s2full) - 1)

    def test_score_tail_gap_iter_trans2b(self):
        beg = ((4, 4),)
        end = ((6, 4), (6, 3))
        full = list(score_tail_gap_iter(beg, end))
        len_wait_1 = 2
        len_wait_2 = 9
        len_wait = len_wait_1 + len_wait_2
        self.assertEqual(len(full), len_wait)
        s1full = [s[0] for s in full]
        s1grow = [s[0] for (n, s) in enumerate(full, start=1) if n <= len_wait_1]
        self.assertEqual(self.is_set_grow(s1full, strict=False), True)
        self.assertEqual(self.is_set_grow(s1grow, strict=True), True)
        s2grow = [s[1] for (n, s) in enumerate(full, start=1) if n > len_wait_1]
        self.assertEqual(len(s2grow), len_wait_2)
        self.assertEqual(self.is_set_grow(s2grow, strict=True), True)

    def is_set_grow(self, seq, strict):
        prev = (0, 0)
        for s in seq:
            if prev[0] > s[0] or prev[1] > s[1]:
                return False
            if strict and not (s[0] > prev[0] or s[1] > prev[1]):
                return False
            prev = s
        return True

    def test_score_tail_gap_iter_trans3(self):
        beg = ((6, 4),)
        end = ((6, 4), (6, 3), (7, 5))
        full = list(score_tail_gap_iter(beg, end))
        len_wait_2 = 9
        len_wait_3 = 12
        len_wait = len_wait_2 + len_wait_3
        self.assertEqual(len(full), len_wait)
        s1full_wait = [s[0] for s in full if s[0] == (6, 4)]
        self.assertEqual(len(s1full_wait), len_wait)
        s2full = [s[1] for s in full]
        self.assertEqual(len(s2full), len_wait)
        self.assertEqual(self.is_set_grow(s2full, strict=False), True)
        s3full = [s[2] for (i, s) in enumerate(full, start=1) if i > len_wait_2]
        self.assertEqual(len(s3full), len_wait_3)
        self.assertEqual(self.is_set_grow(s3full, strict=True), True)

    def test_score_tail_gap_iter_except(self):
        beg = ((4, 6),)
        end = ((6, 4), (6, 3))
        try:
            full = list(score_tail_gap_iter(beg, end))
            print(full)
            self.assertTrue(False)
        except NotImplementedError:
            pass


if __name__ == "__main__":
    doctest.testmod()
    unittest.main()
