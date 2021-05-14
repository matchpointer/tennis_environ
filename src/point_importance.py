# -*- coding: utf-8 -*-

r""" module for calculate point importance.
     www.princeton.edu/~dixitak/Teaching/IntroductoryGameTheory/Precepts/Prcpt03.pdf
     where article 'Illustration of Rollback in a Decision Problem,
                    and Dynamic Games of Competition'
"""

import unittest
from collections import defaultdict

import matchstat

test_mode = False

EQUAL, ADS, ADR = (("40", "40"), ("A", "40"), ("40", "A"))

# name -> dict(ingame -> srv_win_game_probability)
_probdict_from_name = defaultdict(dict)

# name -> dict(ingame -> importance)
_impdict_from_name = defaultdict(dict)


def point_importance(sex, level, surface, qualification, ingame):
    """вернем важность (float) розыгрыша при счете ingame.
    Считаем, что это одинаковая важность и для подающего и для принимающего.
    """
    if level == "teamworld":
        level = "main"
    if qualification:
        name = "sex=%s,level=q-%s,surface=%s" % (sex, level, surface)
        if name in _impdict_from_name:
            return _impdict_from_name[name][ingame]
    name = "sex=%s,level=%s,surface=%s" % (sex, level, surface)
    return _impdict_from_name[name][ingame]


def point_importance_by_name(name, ingame):
    """вернем важность розыгрыша при счете ingame.
    name - имя схемы с заданной вероятностью выигрыша
           розыгрыша на своей подаче. Предвар-но
           нужно готовить схему вызовом prepare (ниже)
            или initialize
    ingame - текстовый 2-tuple, например больше это: ('A', '40')
    """
    return _impdict_from_name[name][ingame]


def game_srv_win_prob(name, ingame):
    """вернем вероятность выигрыша сервером гейма на своей подаче со счета ingame.
    name - имя схемы с заданной вероятностью выигрыша розыгрыша на своей подаче.
    ingame - текстовый 2-tuple, например больше это: ('A', '40')
    """
    return _probdict_from_name[name][ingame]


def initialize():
    for sex in ("wta", "atp"):
        dct = matchstat.generic_result_dict(sex, "service_win")
        # here dct: StructKey -> probability as SizedValue
        assert bool(dct), "bad initialize in point_importance_by_name %s" % sex
        for key, svalue in dct.items():
            if len(key) == 2:
                name = ("sex=%s," % sex) + str(key)
                prepare(name, svalue.value)


def prepare(name, srv_win_point_prob):
    __prepare_probability(name, srv_win_point_prob)
    __prepare_importance(name)
    if not test_mode:
        __prepare_importance_normalize(name)


def __prepare_probability(name, srv_win_point_prob):
    """построить вероятности выигрыша сервером гейма на своей подаче с любого
    заданного счета.
    srv_win_point_prob - вероятность выигрыша розыгрыша сервером на своей подаче.
    """
    if name in _probdict_from_name:
        return
    dct = {}
    x = srv_win_point_prob
    delim = 1.0 - 2.0 * x + 2.0 * x * x
    dct[EQUAL] = (x * x) / delim
    dct[ADR] = (x * x * x) / delim
    dct[ADS] = x * (1.0 - x + x * x) / delim
    dct[("40", "30")] = dct[ADS]
    dct[("40", "15")] = x + (1.0 - x) * dct[ADS]
    dct[("40", "0")] = x + (1.0 - x) * dct[("40", "15")]
    dct[("30", "40")] = dct[ADR]
    dct[("30", "30")] = dct[EQUAL]
    dct[("30", "15")] = x * dct[("40", "15")] + (1.0 - x) * dct[EQUAL]
    dct[("30", "0")] = x * dct[("40", "0")] + (1.0 - x) * dct[("30", "15")]
    dct[("15", "40")] = x * dct[ADR]
    dct[("15", "30")] = x * dct[EQUAL] + (1.0 - x) * dct[("15", "40")]
    dct[("15", "15")] = x * dct[("30", "15")] + (1.0 - x) * dct[("15", "30")]
    dct[("15", "0")] = x * dct[("30", "0")] + (1.0 - x) * dct[("15", "15")]
    dct[("0", "40")] = x * dct[("15", "40")]
    dct[("0", "30")] = x * dct[("15", "30")] + (1.0 - x) * dct[("0", "40")]
    dct[("0", "15")] = x * dct[("15", "15")] + (1.0 - x) * dct[("0", "30")]
    dct[("0", "0")] = x * dct[("15", "0")] + (1.0 - x) * dct[("0", "15")]
    _probdict_from_name[name] = dct


def __prepare_importance(name):
    if name in _impdict_from_name:
        return
    probdct = _probdict_from_name[name]
    dct = {}
    dct[EQUAL] = probdct[ADS] - probdct[ADR]
    dct[ADR] = probdct[EQUAL]
    dct[ADS] = 1.0 - probdct[EQUAL]
    dct[("40", "30")] = dct[ADS]
    dct[("40", "15")] = 1.0 - probdct[ADS]
    dct[("40", "0")] = 1.0 - probdct[("40", "15")]
    dct[("30", "40")] = dct[ADR]
    dct[("30", "30")] = dct[EQUAL]
    dct[("30", "15")] = probdct[("40", "15")] - probdct[EQUAL]
    dct[("30", "0")] = probdct[("40", "0")] - probdct[("30", "15")]
    dct[("15", "40")] = probdct[ADR]
    dct[("15", "30")] = probdct[EQUAL] - probdct[("15", "40")]
    dct[("15", "15")] = probdct[("30", "15")] - probdct[("15", "30")]
    dct[("15", "0")] = probdct[("30", "0")] - probdct[("15", "15")]
    dct[("0", "40")] = probdct[("15", "40")]
    dct[("0", "30")] = probdct[("15", "30")] - probdct[("0", "40")]
    dct[("0", "15")] = probdct[("15", "15")] - probdct[("0", "30")]
    dct[("0", "0")] = probdct[("15", "0")] - probdct[("0", "15")]
    _impdict_from_name[name] = dct


def __prepare_importance_normalize(name):
    """normalize imp values so max_imp_value = 1"""
    dct = _impdict_from_name[name]
    if dct:
        max_imp = max(dct.values())
        if max_imp > 0.001:
            norm_coef = 1.0 / max_imp
            for key in dct:
                dct[key] *= norm_coef


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
    import doctest

    test_mode = True
    doctest.testmod()
    unittest.main()
