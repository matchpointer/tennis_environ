"""
модуль для облегчения работы с базой oncourt:
дает константы минимальных дат турниров.
get_name_level(...) - вернет (имя турнира, уровень турнира)
                    например ('Rome', lev.masters)
get_money(...) - извлечение денежной суммы (с помощью regexpr)
"""

import re
import datetime
from typing import Optional, Tuple

import common as co
import lev

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
    m_is_mid = _BEGIN_MID_END_RE.match(text)
    if m_is_mid:
        return m_is_mid.group("begin") + " " + m_is_mid.group("end")
    return text


_BEGIN_MID_END_RE = re.compile(
    r"^(?P<begin>[a-zA-Z].*)(?P<mid> [A-Z][A-Z], )(?P<end>[a-zA-Z].*)$"
)


def _atp_name_future_level(raw_name) -> Tuple[str, str]:
    """if oncourt feature template then -> (name, lev.future)"""
    level = ''
    m_fut_pref = ATP_COU_FUTURE_PREF_RE.match(raw_name)
    if m_fut_pref:
        level = lev.future
        m_brackets_name = ATP_FUTURE_INBRACKETS_TOURNAME_RE.match(raw_name)
        if m_brackets_name:
            return m_brackets_name.group("name"), level
    m_money_name = ATP_MONEY_TOURNAME_RE.match(raw_name)
    if m_money_name:
        return m_money_name.group("name"), lev.future
    if " #" in raw_name:
        suff_idx = raw_name.index(" #")
        return raw_name[:suff_idx], lev.future
    if (
        raw_name.endswith("-w1")
        or raw_name.endswith("-w2")
        or raw_name.endswith("-w3")
        or raw_name.endswith("-w4")
    ):
        suff_idx = raw_name.index("-w")
        return raw_name[:suff_idx], lev.future
    return raw_name, level


def _wta_name_future_level(raw_name, db_money=None) -> Tuple[str, str]:
    """if oncourt feature template then -> (name, lev.future) """
    name, level = raw_name, ''
    money_from_name = 0
    m_itf_money_name = WTA_MONEY_TOURNAME_RE.match(raw_name)
    if m_itf_money_name:
        name = m_itf_money_name.group("name")
        money_from_name = int(m_itf_money_name.group("money")) * 1000
    money = max(0 if db_money is None else db_money, money_from_name)
    if 0 < money < MAX_WTA_FUTURE_MONEY:
        level = lev.future
    elif MAX_WTA_FUTURE_MONEY <= money < MIN_WTA_MASTERS_MONEY and m_itf_money_name:
        level = lev.chal
    return name, level


def _is_masters_cup(sex, raw_name, rank, tour_date: Optional[datetime.date]):
    if (
        sex == "atp"
        and tour_date is not None
        and tour_date.month in (10, 11)
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
        and tour_date is not None
        and tour_date.month in (10, 11)
        and (
            (rank == 2 and tour_date.year <= 1998)
            or (rank == 3 and tour_date.year > 1998)
        )
        and (
            "WTA Finals" in raw_name
            or ("Championships" in raw_name
                and "Advanta Championships" not in raw_name
                and "European Championships" not in raw_name)
        )
    ):
        return True


def get_name_level(sex, raw_name, rank, money: Optional[float], date=None) -> Tuple[str, str]:
    """ вернет (имя турнира, уровень турнира) например ('Rome', lev.masters)) """
    if sex == "atp":
        name, level = _atp_name_future_level(raw_name)
    else:
        name, level = _wta_name_future_level(raw_name, money)
    if level:
        return name, level

    if "(juniors)" in name:
        return name, lev.junior
    name_low = name.lower()
    if "atp cup" in name_low:
        return "ATP Cup", lev.teamworld
    if "olympics" in name_low:
        return "Olympics", lev.main
    if "davis cup" in name_low or "fed cup" in name_low:
        if "world group" in name_low:
            return remove_middle_cap_item_len2(name), lev.teamworld
        else:
            return name, lev.team
    if raw_name.endswith(" Challenger"):
        return name[0 : -len(" Challenger")], lev.chal
    if (
        sex == "wta"
        and rank == 0
        and money is not None
        and money < MAX_WTA_FUTURE_MONEY
    ):
        return name, lev.future
    if (
        sex == "wta"
        and rank == 2
        and money is not None
        and money >= MIN_WTA_MASTERS_MONEY
    ):
        return name, lev.masters
    if (
        "Australian Open" in raw_name
        or "U.S. Open" in raw_name
        or "French Open" in raw_name
        or "Wimbledon" in raw_name
    ) and "Wildcard" not in raw_name:
        # here we use only name part before ' - New York/Melbourne/Paris/London'
        return name.split(" - ")[0], lev.gs
    if rank == 3:
        return name, lev.masters
    if _is_masters_cup(sex, raw_name, rank, date):
        return "Masters Cup", lev.masters
    if rank in (2, 4, 5) or "Hopman Cup" in raw_name:
        return name, lev.main
    return name, lev.chal


MONEY_RE = re.compile(
    r"(?P<currency>(A\$|\$|L|в‚¬))?(?P<value>([1-9][.0-9]*))(?P<multier>[KM])(\+H)?"
)


def get_money(text) -> Optional[float]:
    """ извлечение денежной суммы (с помощью regexpr) """
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
