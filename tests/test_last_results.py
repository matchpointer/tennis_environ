import unittest

import log
import common as co
from last_results import LastResults


class WinstreakAdvTest(unittest.TestCase):
    def test_winstreak_adv(self):
        # simulate Wang - Davis 2019.09.22
        wang_lr = LastResults.from_week_results(
            [[], [False, True, True, True], [False], [], [False], [False]]
        )
        davis_lr = LastResults.from_week_results(
            [[True, True], [], [], [False, True, True, True], [False, True], []]
        )
        self.assertFalse(wang_lr.poor_practice())
        self.assertFalse(wang_lr.last_weeks_empty(weeks_num=3))

        self.assertFalse(davis_lr.poor_practice())
        self.assertFalse(davis_lr.last_weeks_empty(weeks_num=3))

    def test_winstreak_adv2(self):
        # simulate Caroline Garcia - Anna Blinkova 2019.05.30
        garcia_lr = LastResults.from_week_results(
            [
                [True],
                [False, True, True, True, True],
                [False],
                [False, True, True],
                [],
                [False],
            ]
        )
        blinkova_lr = LastResults.from_week_results(
            [
                [True, True, True, True],
                [],
                [False, True, True, True, True],
                [False, True],
                [False, True, True, True],
                [False, True],
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


if __name__ == '__main__':
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    unittest.main()
