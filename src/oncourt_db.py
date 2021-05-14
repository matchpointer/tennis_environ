# -*- coding: utf-8 -*-
import re
import unittest
import datetime

import common as co
import log

MIN_WTA_TOUR_DATE = datetime.date(1997, 1, 1)
MIN_ATP_TOUR_DATE = datetime.date(1990, 1, 1)

MAX_WTA_FUTURE_MONEY = 19000.0
MIN_WTA_MASTERS_MONEY = 1000000.0


def set_wta_future_max_money(money):
    global MAX_WTA_FUTURE_MONEY
    MAX_WTA_FUTURE_MONEY = money


# in db small (features) 2017, 2018? sample: "Turkey F1 (Antalya)"
ATP_COU_FUTURE_PREF_RE = re.compile(
    r"(?P<country>[a-zA-Z].*) (?P<future_level>F\d\d?).*"
)

ATP_FUTURE_INBRACKETS_TOURNAME_RE = re.compile(r".* \((?P<name>[a-zA-Z].*)\)")

# in db small (features) tour_names since >=2019.01? sample: "M15+H Antalya"
ATP_MONEY_TOURNAME_RE = re.compile(r"M(?P<money>\d\d\d?)(\+H)? (?P<name>[a-zA-Z].*)")

# in db small (features) tour_names since >=2019.01? sample: "W15 Antalya"
WTA_MONEY_TOURNAME_RE = re.compile(r"W(?P<money>\d\d\d?)(\+H)? (?P<name>[a-zA-Z].*)")


def remove_middle_cap_item_len2(text):
    m_is_mid = remove_middle_cap_item_len2.BEGIN_MID_END_RE.match(text)
    if m_is_mid:
        return m_is_mid.group("begin") + " " + m_is_mid.group("end")
    return text


remove_middle_cap_item_len2.BEGIN_MID_END_RE = re.compile(
    r"^(?P<begin>[a-zA-Z].*)(?P<mid> [A-Z][A-Z], )(?P<end>[a-zA-Z].*)$"
)


class RemoveMidItemTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(remove_middle_cap_item_len2("be, PO, fin1."), "be, fin1.")

    def test_not_change(self):
        self.assertEqual(remove_middle_cap_item_len2("be, Pi, fin1."), "be, Pi, fin1.")


def _atp_name_future_level(raw_name):
    """if oncourt feature template then -> name, Level('future')"""
    from tennis import Level

    level = None
    m_fut_pref = ATP_COU_FUTURE_PREF_RE.match(raw_name)
    if m_fut_pref:
        level = Level("future")
        m_brackets_name = ATP_FUTURE_INBRACKETS_TOURNAME_RE.match(raw_name)
        if m_brackets_name:
            return m_brackets_name.group("name"), level
    m_money_name = ATP_MONEY_TOURNAME_RE.match(raw_name)
    if m_money_name:
        return m_money_name.group("name"), Level("future")
    if " #" in raw_name:
        suff_idx = raw_name.index(" #")
        return raw_name[:suff_idx], Level("future")
    if (
        raw_name.endswith("-w1")
        or raw_name.endswith("-w2")
        or raw_name.endswith("-w3")
        or raw_name.endswith("-w4")
    ):
        suff_idx = raw_name.index("-w")
        return raw_name[:suff_idx], Level("future")
    return raw_name, level


def _wta_name_future_level(raw_name, db_money=None):
    """if oncourt feature template then -> name, Level('future')"""
    from tennis import Level

    name, level = raw_name, None
    money_from_name = 0
    m_itf_money_name = WTA_MONEY_TOURNAME_RE.match(raw_name)
    if m_itf_money_name:
        name = m_itf_money_name.group("name")
        money_from_name = int(m_itf_money_name.group("money")) * 1000
    money = max(0 if db_money is None else db_money, money_from_name)
    if 0 < money < MAX_WTA_FUTURE_MONEY:
        level = Level("future")
    elif MAX_WTA_FUTURE_MONEY <= money < MIN_WTA_MASTERS_MONEY and m_itf_money_name:
        level = Level("chal")
    return name, level


def _is_masters_cup(sex, raw_name, rank, date):
    if (
        sex == "atp"
        and date is not None
        and date.month in (10, 11)
        and rank == 3
        and (
            "Championship" in raw_name
            or "Masters Cup" in raw_name
            or "World Tour Finals" in raw_name
        )
    ):
        return True
    if (
        sex == "wta"
        and date is not None
        and date.month in (10, 11)
        and ((rank == 2 and date.year <= 1998) or (rank == 3 and date.year > 1998))
        and "Championships" in raw_name
        and "Advanta Championships" not in raw_name
        and "European Championships" not in raw_name
    ):
        return True


def get_name_level(sex, raw_name, rank, money, date=None):
    from tennis import Level

    if sex == "atp":
        name, level = _atp_name_future_level(raw_name)
    else:
        name, level = _wta_name_future_level(raw_name, money)
    if level is not None:
        return name, level

    if "(juniors)" in name:
        return name, Level("junior")
    name_low = name.lower()
    if "atp cup" in name_low:
        return "ATP Cup", Level("teamworld")
    if "olympics" in name_low:
        return "Olympics", Level("main")
    if "davis cup" in name_low or "fed cup" in name_low:
        if "world group" in name_low:
            return remove_middle_cap_item_len2(name), Level("teamworld")
        else:
            return name, Level("team")
    if raw_name.endswith(" Challenger"):
        return name[0 : -len(" Challenger")], Level("chal")
    if (
        sex == "wta"
        and rank == 0
        and money is not None
        and money < MAX_WTA_FUTURE_MONEY
    ):
        return name, Level("future")
    if (
        sex == "wta"
        and rank == 2
        and money is not None
        and money >= MIN_WTA_MASTERS_MONEY
    ):
        return name, Level("masters")
    if (
        "Australian Open" in raw_name
        or "U.S. Open" in raw_name
        or "French Open" in raw_name
        or "Wimbledon" in raw_name
    ) and "Wildcard" not in raw_name:
        # here we use only name part before ' - New York/Melbourne/Paris/London'
        return name.split(" - ")[0], Level("gs")
    if rank == 3:
        return name, Level("masters")
    if _is_masters_cup(sex, raw_name, rank, date):
        return "Masters Cup", Level("masters")
    if rank in (2, 4, 5) or "Hopman Cup" in raw_name:
        return name, Level("main")
    return name, Level("chal")


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


MONEY_RE = re.compile(
    r"(?P<currency>(A\$|\$|L|в‚¬))?(?P<value>([1-9][.0-9]*))(?P<multier>[KM])(\+H)?"
)


def get_money(text):
    if not text:
        return None
    raw_text = co.to_ascii(text.replace(",", ".").strip())
    match = MONEY_RE.match(raw_text)
    if match:
        value = float(match.group("value"))
        if match.group("multier") == "K":
            return value * 1000
        elif match.group("multier") == "M":
            return value * 1000000
    return None


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
    log.initialize(
        co.logname(__file__, test=True), file_level="debug", console_level="debug"
    )
    unittest.main()
