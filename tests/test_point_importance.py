# -*- coding=utf-8 -*-
import unittest

from point_importance import (
    initialize,
    prepare,
    point_importance_by_name,
    _probdict_from_name,
    _impdict_from_name,
    game_srv_win_prob,
    EQUAL,
    ADS,
    ADR,
)
import point_importance


class PrepareTest(unittest.TestCase):
    def test_65_percent_matrix(self):
        name, srvwin_prob = "test_65", 0.65
        prepare(name, srvwin_prob)

        prob_len = len(_probdict_from_name[name])
        imp_len = len(_impdict_from_name[name])
        self.assertEqual(imp_len, prob_len)

        eps = 0.01
        prob = game_srv_win_prob(name, ADS)
        self.assertTrue(abs(prob - 0.92) < eps)

        prob = game_srv_win_prob(name, ADR)
        self.assertTrue(abs(prob - 0.5) < eps)

        prob = game_srv_win_prob(name, EQUAL)
        self.assertTrue(abs(prob - 0.78) < eps)
        theory_prob = srvwin_prob ** 2 / (srvwin_prob ** 2 + (1.0 - srvwin_prob) ** 2)
        self.assertTrue(abs(prob - theory_prob) < eps)

        prob = game_srv_win_prob(name, ("0", "0"))
        self.assertTrue(abs(prob - 0.83) < eps)

        prob = game_srv_win_prob(name, ("40", "0"))
        self.assertTrue(abs(prob - 0.99) < eps)

        prob = game_srv_win_prob(name, ("0", "40"))
        self.assertTrue(abs(prob - 0.21) < eps)

        # --------------------------------------
        imp = point_importance_by_name(name, EQUAL)
        self.assertTrue(abs(imp - 0.42) < eps)

        imp = point_importance_by_name(name, ADS)
        self.assertTrue(abs(imp - 0.23) < eps)

        imp = point_importance_by_name(name, ADR)
        self.assertTrue(abs(imp - 0.78) < eps)

        imp = point_importance_by_name(name, ("40", "0"))
        self.assertTrue(abs(imp - 0.03) < eps)

        imp = point_importance_by_name(name, ("0", "0"))
        self.assertTrue(abs(imp - 0.22) < eps)

        imp = point_importance_by_name(name, ("15", "15"))
        self.assertTrue(abs(imp - 0.29) < eps)

    def test_initialize(self):
        initialize()
        name = "sex=wta,level=main,surface=Hard"

        prob_len = len(_probdict_from_name[name])
        imp_len = len(_impdict_from_name[name])
        self.assertEqual(imp_len, prob_len)
        self.assertEqual(imp_len, 18)


if __name__ == "__main__":
    point_importance.test_mode = True
    unittest.main()
