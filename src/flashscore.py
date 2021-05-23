# -*- coding: utf-8 -*-
import unittest
import datetime
import copy
from collections import defaultdict
from typing import Optional, Tuple
import re

from lxml import etree
import lxml.html

import dba
import common as co
import log
import oncourt_players
import tennis
import tennis_time as tt
import tennis_parse
import weeked_tours
import score as sc
from live import (
    LiveEventToskipError,
    MatchStatus,
    MatchSubStatus,
    LiveMatch,
    TourInfo,
    LiveTourEvent,
    skip_levels_default,
)
from oncourt_db import MAX_WTA_FUTURE_MONEY
from geo import city_in_europe
import tour_name

SKIP_SEX = None
SKIP_FINAL_RESULT_ONLY = True
FAKE_TOUR_NAME = "Others matches"


def make_events_impl(cur_date, tree, match_status, skip_levels):
    tour_events = []
    # div(tour_info), div(match), div(match)..., div(tour_info), div(match)...
    all_divs = list(
        tree.xpath(
            ("//div[@class='leagues--live ']/" + "div[@class='sportName tennis']/div")
        )
    )
    is_events = bool(all_divs)
    start_idx = 0
    today_date = datetime.date.today()
    while is_events:
        try:
            tennis_event = make_live_tour_event(
                all_divs[start_idx:], match_status, skip_levels, cur_date, start_idx
            )
            if tennis_event and not tennis_event.doubles:
                if "fed-cup" in str(tennis_event.tour_name) or (
                    ("davis-cup" in str(tennis_event.tour_name))
                    and not (
                        today_date >= datetime.date(2019, 11, 1)
                        and tennis_event.level == "teamworld"
                    )
                ):
                    # this was ok, BUT since 2019.11 atp has new format (Finals), where
                    # there are groups stage (each group has unique ID in oncourt db)
                    # flashscore.com does not display group name info.
                    # So now problem to dispatch group matches and get correct ID.
                    for evt in _split_team_event_by_countries(tennis_event):
                        tour_events.append(evt)
                else:
                    tour_events.append(tennis_event)
        except LiveEventToskipError:
            pass
        except co.TennisError as err:
            log.error("{} [{}]".format(err, err.__class__.__name__))
        if (start_idx + 1) >= (len(all_divs) - 1):
            break
        idx, head_el = co.find_indexed_first(
            all_divs[start_idx + 1 :], lambda e: "event__header" in e.get("class")
        )
        is_events = head_el is not None
        start_idx += 1 + idx
    return [t for t in tour_events if t.matches]


def _split_team_event_by_countries(team_event):
    """motiv: Fed/Davis Cup flashcore event contains matches where surfaces differs.
    Also oncourt tour id is unique (mostly) for pair (country1, country2)"""

    def _make_event(coupair, matches):
        live_event = LiveTourEvent()
        live_event.tour_info = copy.deepcopy(team_event.tour_info)
        live_event.tour_info.tour_name.add_part(coupair[0])
        live_event.tour_info.tour_name.add_part(coupair[1])
        for mch in matches:
            mch.live_event = live_event  # new owner
            live_event.matches.append(mch)
        find_surface(live_event)
        return live_event

    def find_surface(live_event):
        if live_event.surface is None:
            surface = sex_tourname_surf_map.get(
                (live_event.sex, live_event.tour_name, None)
            )
            if surface is not None:
                live_event.tour_info.surface = surface

    coupair_to_matches = defaultdict(list)
    for m in team_event.matches:
        if (
            m.first_player
            and m.first_player.cou
            and m.second_player
            and m.second_player.cou
        ):
            coupair_to_matches[(m.first_player.cou, m.second_player.cou)].append(m)

    return [
        _make_event(coupair, matches) for coupair, matches in coupair_to_matches.items()
    ]


def make_events(webpage, skip_levels, match_status=MatchStatus.live, target_date=None):
    """Return events list. skip_levels is dict: sex -> levels"""
    parser = lxml.html.HTMLParser(encoding="utf8")
    tree = lxml.html.document_fromstring(webpage, parser)
    if target_date is None:
        target_date = datetime.date.today()  # also possible make_current_date(tree)
    result = make_events_impl(target_date, tree, match_status, skip_levels)
    return result


def make_live_tour_event(elements, match_status, skip_levels, current_date, start_idx):
    tour_info = make_tourinfo(elements[0], start_idx)
    if (
        tour_info.doubles
        or tour_info.level in skip_levels[tour_info.sex]
        or tour_info.exhibition
        or (tour_info.tour_name and "Next Gen Finals" in tour_info.tour_name)
    ):
        raise LiveEventToskipError()

    live_event = LiveTourEvent()
    live_event.tour_info = tour_info
    for elem in elements[1:]:
        if "event__match" not in elem.get("class"):
            break  # next tour event
        live_match = make_live_match(elem, live_event, current_date, match_status)
        if live_match is None:
            continue
        if live_match.paired():
            continue
        live_event.matches.append(live_match)
    return live_event


def make_live_match(element, live_event, current_date, match_status: MatchStatus):
    def is_final_result_only():
        # score (not point by point) will be after finish.
        # after finish we can not determinate that is was state FRO.
        el = co.find_first_xpath(
            element, "child::div[@class='event__time']/div[@title='Final result only.']"
        )
        return el is not None

    def get_time():
        time_el = co.find_first_xpath(element, "child::div[@class='event__time']")
        if time_el is not None:
            time_txt = time_el.text
            if time_txt:
                hour_txt, minute_txt = time_txt.split(":")
                return datetime.time(
                    hour=int(hour_txt), minute=int(minute_txt), second=0
                )

    def current_status() -> Tuple[Optional[MatchStatus], Optional[MatchSubStatus]]:
        status, sub_status = None, None
        cls_text = element.get("class")
        if "event__match--scheduled" in cls_text:
            status = MatchStatus.scheduled
        elif "event__match--live" in cls_text:
            status = MatchStatus.live
        stage_el = co.find_first_xpath(
            element, "child::div[contains(@class,'event__stage')]"
        )
        if stage_el is not None:
            text = stage_el.text_content().strip().lower()
            if "finished" in text:
                status = MatchStatus.finished

            if "retired" in text or "walkover" in text:
                sub_status = MatchSubStatus.retired
            elif "interrupted" in text:
                sub_status = MatchSubStatus.interrupted
            elif "cancelled" in text:
                sub_status = MatchSubStatus.cancelled
        return status, sub_status

    def get_name_cou(is_left):
        inclass_txt = "event__participant--{}".format("home" if is_left else "away")
        xpath_txt = "child::div[contains(@class, '{}')]".format(inclass_txt)
        try:
            plr_el = co.find_first_xpath(element, xpath_txt)
            if plr_el is None:
                raise co.TennisParseError(
                    "nofound plr_el in \n{}".format(
                        etree.tostring(element, encoding="utf8")
                    )
                )
            txt = co.to_ascii(plr_el.text).strip()
            bracket_idx = txt.index("(")
            name = txt[0:bracket_idx].strip()
            cou = txt[bracket_idx + 1 : bracket_idx + 4].strip().upper()
            return name, cou
        except ValueError:
            raise LiveEventToskipError()  # maybe it is team's result div without '(cou)'

    def is_left_service():
        if (
            co.find_first_xpath(element, "child::svg[contains(@class, '--serveHome')]")
            is not None
        ):
            return True
        if (
            co.find_first_xpath(element, "child::svg[contains(@class, '--serveAway')]")
            is not None
        ):
            return False

    def ingame_score():
        def get_xpath(is_left):
            return (
                "child::div[contains(@class,'event__part--{}') and "
                + "contains(@class,'event__part--6')]"
            ).format("home" if is_left else "away")

        fst_el = co.find_first_xpath(element, get_xpath(is_left=True))
        snd_el = co.find_first_xpath(element, get_xpath(is_left=False))
        if fst_el is not None and snd_el is not None:
            result = fst_el.text, snd_el.text
            if result != ("", ""):
                return result

    def make_score(retired):
        def make_side_scr_elements(is_left):
            inclass_txt = "event__part--{}".format("home" if is_left else "away")
            xpath_txt = (
                "child::div[contains(@class, '{}') and "
                + "not(contains(@class, 'event__part--6'))]"
            ).format(inclass_txt)
            return element.xpath(xpath_txt)  # elements

        pairs = zip(
            make_side_scr_elements(is_left=True), make_side_scr_elements(is_left=False)
        )
        wl_games = [(int(e1.text), int(e2.text)) for e1, e2 in pairs]
        return sc.Score.from_pairs(wl_games, retired=retired)

    def make_href():
        matchid = element.get("id")
        if matchid:
            return "https://www.flashscore.com/match/{}".format(matchid[4:])

    cur_st, cur_sub_st = current_status()
    if SKIP_FINAL_RESULT_ONLY and is_final_result_only():
        raise LiveEventToskipError()
    if cur_st != match_status:
        return None
    obj = LiveMatch(live_event)
    obj.href = make_href()
    if cur_st == MatchStatus.scheduled:
        obj.time = get_time()
    obj.date = current_date
    left_name, left_cou = get_name_cou(is_left=True)
    right_name, right_cou = get_name_cou(is_left=False)
    obj.name = left_name + " - " + right_name
    obj.first_player = find_player(live_event.sex, left_name, left_cou)
    obj.second_player = find_player(live_event.sex, right_name, right_cou)
    if cur_st == MatchStatus.live:
        obj.left_service = is_left_service()
        obj.ingame = ingame_score()
    obj.score = make_score(retired=cur_sub_st == MatchSubStatus.retired)
    return obj


wta_today_players_cache = defaultdict(lambda: None)  # (disp_name, cou) -> Player
atp_today_players_cache = defaultdict(lambda: None)

deep_find_player = False

init_cache_mode = False


def initialize_players_cache(webpage, match_status=MatchStatus.scheduled):
    global deep_find_player, init_cache_mode
    mem_deep_find_player = deep_find_player
    deep_find_player = True
    init_cache_mode = True
    try:
        make_events(
            webpage,
            skip_levels=skip_levels_default(),
            match_status=match_status,
            target_date=datetime.date.today(),
        )
    finally:
        deep_find_player = mem_deep_find_player
        init_cache_mode = False


def find_player(sex, disp_name, cou):
    cache = wta_today_players_cache if sex == "wta" else atp_today_players_cache
    if cache:
        return cache[(disp_name, cou)]
    if deep_find_player:
        plr = tennis_parse.identify_player("FS", sex, disp_name, cou)
        if init_cache_mode:
            if plr is not None:
                cache[(disp_name, cou)] = plr
            else:
                log.warn("fail player ident {} '{}' {}".format(sex, disp_name, cou))
        return plr
    else:
        return co.find_first(
            oncourt_players.players(sex),
            lambda p: p.cou == cou and p.disp_name("flashscore") == disp_name,
        )


def make_tourinfo(element, start_idx):
    def split_before_after_text(text, part):
        if part in text:
            before_including = co.strip_after_find(text, part)
            after_not_including = text[text.index(part) + len(part) :]
            return before_including, after_not_including

    def split_sex_doubles_text(text):
        if "DOUBLES" in text or "EXHIBITION" in text or "Laver Cup" in text:
            raise LiveEventToskipError("not interested event")
        # now ':' is not in text_content() and we need split text by key words:
        result = split_before_after_text(text, "SINGLES")
        if result is None:
            result = split_before_after_text(text, "TEAMS - WOMEN")
            if result is None:
                result = split_before_after_text(text, "TEAMS - MEN")
        return result

    if "event__header" not in element.get("class"):
        log.error(
            "class != event__header got class: '{}' start_idx {}\n{}".format(
                element.get("class"),
                start_idx,
                etree.tostring(element, encoding="utf8"),
            )
        )
        raise Exception(
            "class != event__header got class: '{}' start_idx {}".format(
                element.get("class"), start_idx
            )
        )

    evt_title_el = co.find_first_xpath(
        element, "child::div[contains(@class,'event__title ')]"
    )
    if evt_title_el is None:
        raise co.TennisParseError(
            "not parsed sex_doubles el FS TourInfo from\n{}".format(
                etree.tostring(element, encoding="utf8")
            )
        )
    evt_title_text = evt_title_el.text_content().strip()
    if not evt_title_text:
        raise co.TennisParseError("empty sex_doubles txt FS TourInfo")
    evt_title_parts = split_sex_doubles_text(evt_title_text)
    if evt_title_parts is None:
        raise co.TennisParseError(
            "fail split sex_doubles: '{}'\n{}".format(
                evt_title_text, etree.tostring(element, encoding="utf8")
            )
        )
    return TourInfoFlashscore(evt_title_parts[0], evt_title_parts[1])


# sample: "W15 Antalya (Turkey)"
WTA_MONEY_TOURNAME_RE = re.compile(
    r"W(?P<money>\d\d\d?)(\+H)? (?P<name>[A-Z][0-9a-zA-Z '-]+)"
)

ATP_MONEY_TOURNAME_RE = re.compile(
    r"M(?P<money>\d\d\d?)(\+H)? (?P<name>[A-Z][0-9a-zA-Z '-]+)"
)


def itf_wta_money_tourname(text):
    """:returns (money_K, tourname) or None if fail.
    Details started with open bracket (country) are ignored"""
    m_money_tourname = WTA_MONEY_TOURNAME_RE.match(text)
    if m_money_tourname:
        return (
            int(m_money_tourname.group("money")) * 1000,
            m_money_tourname.group("name").rstrip(),
        )


def itf_atp_money_tourname(text):
    """:returns (money_K, tourname) or None if fail.
    Details started with open bracket (country) are ignored"""
    m_money_tourname = ATP_MONEY_TOURNAME_RE.match(text)
    if m_money_tourname:
        return (
            int(m_money_tourname.group("money")) * 1000,
            m_money_tourname.group("name").rstrip(),
        )


def split_ontwo_enclosed(text: str, delim_op: str, delim_cl: str):
    if text.endswith(delim_cl):
        pos = text.rfind(delim_op)
        if pos >= 0:
            return (
                text[:pos].strip(),
                text[pos + len(delim_op) : len(text) - len(delim_cl)].strip(),
            )
    return text.strip(), ""


class TourInfoFlashscore(TourInfo):
    def __init__(self, part_one, part_two):
        super(TourInfoFlashscore, self).__init__()

        body_part, surf_part = co.split_ontwo(part_two, delim=", ")
        body_part, qual_part = co.split_ontwo(body_part, delim=" - ")
        body_part, country_part = split_ontwo_enclosed(
            body_part, delim_op="(", delim_cl=")"
        )
        body_part, desc_part = split_ontwo_enclosed(
            body_part, delim_op="(", delim_cl=")"
        )
        number = None
        if body_part and body_part[-1].isdigit() and body_part[-2] == " ":
            number = int(body_part[-1])
            body_part = body_part[:-2]

        self.teams = (
            "FED Cup" in body_part
            or "Davis Cup" in body_part
            or "ATP Cup" in body_part
            or "Billie Jean King Cup" in body_part
        )
        if self.teams and "World Group" not in qual_part and "ATP Cup" not in body_part:
            raise LiveEventToskipError("it is not top leage of Davis Cup")
        if "WOMEN" in part_one or "WTA" in part_one or "GIRLS" in part_one:
            self.sex = "wta"
        else:
            self.sex = "atp"
        if self.sex == SKIP_SEX:
            raise LiveEventToskipError()
        self.doubles = "DOUBLES" in body_part
        is_grand_slam = (
            "JUNIORS" not in body_part
            and "BOYS" not in part_one
            and "GIRLS" not in part_one
            and (
                "French Open" in body_part
                or "Wimbledon" in body_part
                or "U.S. Open" in body_part
                or "Australian Open" in body_part
            )
        )
        self.qualification = "Qualification" in qual_part and not self.teams
        self.exhibition = "EXHIBITION" in part_one
        self.itf = part_one.startswith("ITF")
        self.surface = tennis.Surface(surf_part) if surf_part else None
        self.tour_name, money = self.init_tourname(body_part, desc_part, number)
        self.level = self.__init_level(is_grand_slam, part_one, qual_part, money)
        self.country = country_part
        self.corrections()

    def corrections(self):
        if (
            self.surface == "Hard"
            and (
                datetime.date.today().month in (12, 1, 2)
                or datetime.date.today().month == 3
                and datetime.date.today().day < 20
            )
            and city_in_europe(self.tour_name.name)  # not play hard in winter europe
        ):
            self.surface = tennis.Surface("Carpet")

    def init_tourname(self, body_part, desc_part, number):
        def map_to_oncourt():
            if number:
                in_name = f"{body_part} {number}"
            else:
                in_name = body_part
            out_name = self.tour_name_map_to_oncourt(self.sex, in_name)
            if out_name != in_name:
                return out_name, None  # number is replaced
            else:
                return body_part, number

        money = None
        if self.itf:
            if self.sex == "wta":
                money_tourname = itf_wta_money_tourname(body_part)
            else:
                money_tourname = itf_atp_money_tourname(body_part)
            if money_tourname:
                money = money_tourname[0]
                body_part = money_tourname[1]

        body_part, num = map_to_oncourt()
        # body_part = trmt.Tournament.name_correction(self.sex, body_part)
        return (
            tour_name.TourName(name=body_part, desc_name=desc_part, number=num),
            money,
        )

    def __eq__(self, other):
        return (
            self.tour_name == other.tour_name
            and self.sex == other.sex
            and self.teams == other.teams
            and self.qualification == other.qualification
            and self.doubles == other.doubles
            and self.itf == other.itf
        )

    def __init_level(self, is_grand_slam, part_one, qual_part, money):
        if is_grand_slam:
            result = tennis.Level("gs")
        elif "CHALLENGER" in part_one:
            result = tennis.Level("chal")
        elif "BOYS" in part_one or "GIRLS" in part_one:
            result = tennis.Level("junior")
        elif self.teams:
            if "World Group" in qual_part or "ATP Cup" in self.tour_name.name:
                result = tennis.Level("teamworld")
            else:
                result = tennis.Level("team")
        elif self.itf:
            if self.sex == "wta" and (
                (money is not None and money > MAX_WTA_FUTURE_MONEY)
                or (self.tour_name, self.surface) in wta_chal_tour_surf
            ):
                result = tennis.Level("chal")
            else:
                result = tennis.Level("future")
        else:
            result = tennis.Level("main")
        return result


wta_chal_tour_surf = set()  # set of (tour_name, surface)


def initialize(prev_week=False):
    """must be called AFTER weeked_tours::initialize.
    init wta_chal_tour_surf, and add to sex_tourname_surf_map"""
    wta_chal_tour_surf.clear()
    monday = tt.past_monday_date(datetime.date.today())
    if prev_week:
        monday = monday - datetime.timedelta(days=7)
    year_weeknum = tt.get_year_weeknum(monday)
    for tour in weeked_tours.tours("wta", year_weeknum):
        if tour.level == "teamworld":
            sex_tourname_surf_map[("wta", tour.name, None)] = tour.surface
        if tour.level == "chal":
            wta_chal_tour_surf.add((tour.name, tour.surface))
            log.info(
                "FS::init prev_week {} wta_chal {} {}".format(
                    prev_week, tour.name, tour.surface
                )
            )


sex_tourname_surf_map = {  # her in values are more correct surface
    ("wta", tour_name.TourName("Fukuoka"), tennis.Surface("Hard")): tennis.Surface(
        "Carpet"
    ),
    ("wta", tour_name.TourName("Kurume"), tennis.Surface("Hard")): tennis.Surface(
        "Carpet"
    ),
    ("atp", tour_name.TourName("ATP Cup"), tennis.Surface("Carpet")): tennis.Surface(
        "Hard"
    ),
}


if __name__ == "__main__":
    log.initialize(
        co.logname(__file__, test=True), file_level="info", console_level="info"
    )
    dba.open_connect()
    oncourt_players.initialize(yearsnum=1.2)

    min_date = tt.past_monday_date(datetime.date.today()) - datetime.timedelta(days=7)
    weeked_tours.initialize_sex(
        "wta", min_date=min_date, max_date=None, with_today=True, with_bets=True
    )
    weeked_tours.initialize_sex(
        "atp", min_date=min_date, max_date=None, with_today=True, with_bets=True
    )
    initialize()

    unittest.main()
    dba.close_connect()
