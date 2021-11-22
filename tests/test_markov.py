import unittest

from side import Side


from markov import initialize, initialize_gen, Prob, tie_win_prob, set_win_prob


class TieWinProbTest(unittest.TestCase):
    def test_tie_win_prob_02(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(0, 2),
            game_opener=Side("RIGHT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 02 res_prob for R {res_pr}")
        self.assertTrue(0.1 < res_pr < 0.5)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(0, 2),
            game_opener=Side("RIGHT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 02 res_prob2 for L {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)

    def test_tie_win_prob_30(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 30 res_prob {res_pr}")
        self.assertTrue(0.6 < res_pr < 1)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 30 res_prob2 for R {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)

    def test_tie_win_prob_00(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = tie_win_prob(
            prob=prob,
            game_scr=(0, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"tie 00 res_prob {res_pr}")
        self.assertTrue(abs(res_pr - 0.5) < 0.001)

        res_pr2 = tie_win_prob(
            prob=prob,
            game_scr=(0, 0),
            game_opener=Side("LEFT"),
            srv_side=Side("LEFT"),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        print(f"tie 00 res_prob2 for R {res_pr2}")
        self.assertTrue(abs((1.0 - res_pr) - res_pr2) < 0.001)


class SetWinLeftProbTest(unittest.TestCase):
    def test_set_win_prob_54(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(4, 5),
            game_opener=Side("RIGHT"),
            game_scr=(3, 0),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr is not None)
        self.assertTrue(0.7 < res_pr < 1, f"{res_pr} <= 0.7")

        res_pr2 = set_win_prob(
            prob=prob,
            set_scr=(4, 5),
            game_opener=Side("RIGHT"),
            game_scr=(0, 3),
            target_side=Side("RIGHT"),
        )
        self.assertTrue(res_pr2 is not None)
        self.assertTrue(0.45 < res_pr2 < 0.6)

    def test_set_win_prob_53(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(5, 3),
            game_scr=(3, 0),
            game_opener=Side("LEFT"),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        self.assertTrue(0.7 < res_pr < 1)

    def test_set_win_prob_00(self):
        prob = Prob(win_point=0.63, hold=0.65)
        res_pr = set_win_prob(
            prob=prob,
            set_scr=(0, 0),
            game_opener=Side("LEFT"),
            game_scr=(0, 0),
            target_side=Side("LEFT"),
        )
        self.assertTrue(res_pr is not None)
        print(f"res_pr_00: {res_pr}")
        self.assertTrue(0.51 < res_pr < 1)


if __name__ == "__main__":
    import doctest

    initialize_gen()
    initialize(probs=[Prob(win_point=0.63, hold=0.65)])
    doctest.testmod()
    unittest.main()
