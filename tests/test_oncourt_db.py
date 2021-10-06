import unittest

from oncourt_db import (
    remove_middle_cap_item_len2,
    _atp_name_future_level,
    ATP_COU_FUTURE_PREF_RE,
    ATP_FUTURE_INBRACKETS_TOURNAME_RE,
    ATP_MONEY_TOURNAME_RE,
    WTA_MONEY_TOURNAME_RE,
    get_money,
)


class RemoveMidItemTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(remove_middle_cap_item_len2("be, PO, fin1."), "be, fin1.")

    def test_not_change(self):
        self.assertEqual(remove_middle_cap_item_len2("be, Pi, fin1."), "be, Pi, fin1.")


class RegexTest(unittest.TestCase):
    def test_atp_name_future_level_simple(self):
        name, level = _atp_name_future_level("Bulgaria-w1")
        self.assertEqual(level, "future")
        self.assertEqual(name, "Bulgaria")

    def test_atp_cou_future_pref(self):
        mch = ATP_COU_FUTURE_PREF_RE.match("Turkey F1 (Antalya)")
        self.assertTrue(mch is not None)
        self.assertEqual("Turkey", mch.group("country"))
        self.assertEqual("F1", mch.group("future_level"))

    def test_atp_inbrackets_tourname(self):
        mch = ATP_FUTURE_INBRACKETS_TOURNAME_RE.match("Turkey F1 (Antalya)")
        self.assertTrue(mch is not None)
        self.assertEqual("Antalya", mch.group("name"))

    def test_atp_money_tourname(self):
        mch = ATP_MONEY_TOURNAME_RE.match("M25 Antalya")
        self.assertTrue(mch is not None)
        self.assertEqual("Antalya", mch.group("name"))
        self.assertEqual("25", mch.group("money"))

        mch = ATP_MONEY_TOURNAME_RE.match("M30 Hong Kong")
        self.assertTrue(mch is not None)
        self.assertEqual("Hong Kong", mch.group("name"))
        self.assertEqual("30", mch.group("money"))

        mch = ATP_MONEY_TOURNAME_RE.match("M15+H Antalya")
        self.assertTrue(mch is not None)
        self.assertEqual("Antalya", mch.group("name"))
        self.assertEqual("15", mch.group("money"))

    def test_wta_tourname(self):
        mch = WTA_MONEY_TOURNAME_RE.match("W25 Antalya")
        self.assertTrue(mch is not None)
        self.assertEqual("Antalya", mch.group("name"))
        self.assertEqual("25", mch.group("money"))


class ParseMoneyTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(get_money("$10K"), 10000.0)
        self.assertEqual(get_money("$10M"), 10000000.0)
        self.assertEqual(get_money("$10K+H"), 10000.0)
        self.assertEqual(get_money("$1,34M"), 1340000.0)
        self.assertEqual(get_money("$1.34M"), 1340000.0)
        self.assertEqual(get_money("L4.946M"), 4946000.0)
        self.assertEqual(get_money("16M"), 16000000.0)


if __name__ == "__main__":
    unittest.main()

