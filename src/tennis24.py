r"""
module gives possibility (make_events) for parsing live matches at flashscore site
including points score in game (for example 40:30)
"""
import datetime
import time
from collections import defaultdict, Counter
from typing import Optional, Tuple, List
import re

from lxml import etree
import lxml.html

from country_code import alpha_3_code
from loguru import logger as log

import common as co
import lev
import oncourt_players
from surf import make_surf, Hard, Carpet
import tennis_time as tt
import player_name_ident
import weeked_tours
import score as sc
from live import (
    LiveEventToskipError,
    MatchStatus,
    MatchSubStatus,
    LiveMatch,
    LiveTourEvent,
    skip_levels_work,
)
from live_tourinfo import TourInfo, TourInfoCache
from oncourt_db import MAX_WTA_FUTURE_MONEY
from geo import city_in_europe
from tour_name import TourName, tourname_number_split

SKIP_SEX = None
SKIP_FINAL_RESULT_ONLY = True  # events marked as FRO are without point by point


def make_events(
    webpage, skip_levels, match_status=MatchStatus.live, target_date=None
) -> List[LiveTourEvent]:
    """Return events list. skip_levels is dict: sex -> levels"""
    parser = lxml.html.HTMLParser(encoding="utf8")
    tree = lxml.html.document_fromstring(webpage, parser)
    if target_date is None:
        target_date = datetime.date.today()  # also possible _make_current_date(tree)
    result = _make_events_impl(target_date, tree, match_status, skip_levels)
    return result


_err_counter = Counter()
MAX_LOG_RECORDS_PER_ERROR = 100


def _make_events_impl(cur_date, tree, match_status, skip_levels) -> List[LiveTourEvent]:
    tour_events = []
    # div(tour_info), div(match), div(match)..., div(tour_info), div(match)...
    all_divs = list(
        tree.xpath("//div[@class='leagues--live ']/div[@class='sportName tennis']/div")
    )
    is_events = bool(all_divs)
    start_idx = 0
    while is_events:
        try:
            tennis_event = _make_live_tour_event(
                all_divs[start_idx:], match_status, skip_levels, cur_date, start_idx
            )
            if tennis_event:
                tour_events.append(tennis_event)
        except LiveEventToskipError:
            pass
        except co.TennisError as err:
            msg = "{} [{}]".format(err, err.__class__.__name__)
            _err_counter[msg] += 1
            if _err_counter[msg] < MAX_LOG_RECORDS_PER_ERROR:
                log.error(msg)
        if (start_idx + 1) >= (len(all_divs) - 1):
            break
        idx, head_el = co.find_indexed_first(
            all_divs[start_idx + 1:], lambda e: "event__header" in e.get("class")
        )
        is_events = head_el is not None
        start_idx += 1 + idx
    return [t for t in tour_events if t.matches]


def _make_live_tour_event(elements, match_status, skip_levels, current_date, start_idx):
    tour_info = _make_tourinfo(elements[0], start_idx, skip_levels)
    live_event = LiveTourEvent()
    live_event.tour_info = tour_info
    for elem in elements[1:]:
        if "event__match" not in elem.get("class"):
            break  # next tour event
        live_match = _make_live_match(elem, live_event, current_date, match_status)
        if live_match is None:
            continue
        if live_match.paired():
            continue
        live_event.matches.append(live_match)
    return live_event


def _make_live_match(element, live_event, current_date, match_status: MatchStatus):
    def is_final_result_only():
        # score (not point by point) will be after match finish.
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

    def get_name(is_left):
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
            name = co.to_ascii(plr_el.text).strip()
            return name
        except ValueError:
            # maybe it is team's result div without '(cou)':
            tourinfo_cache.edit_skip(live_event.tour_info)

    def get_cou(is_left):
        """ style with plr_country_long=True, national flag image """
        inclass_txt = "event__logo--{}".format("home" if is_left else "away")
        xpath_txt = "child::span[contains(@class, '{}')]".format(inclass_txt)
        try:
            cou_el = co.find_first_xpath(element, xpath_txt)
            if cou_el is None:
                raise co.TennisParseError(
                    "nofound cou_el in \n{}".format(
                        etree.tostring(element, encoding="utf8")
                    )
                )
            cou = co.to_ascii(cou_el.get('title')).strip()
            return cou
        except ValueError:
            # maybe it is team's result div without '(cou)':
            tourinfo_cache.edit_skip(live_event.tour_info)

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
                f"child::div[contains(@class,'event__part--"
                f"{'home' if is_left else 'away'}') and "
                f"contains(@class,'event__part--6')]"
            )

        fst_el = co.find_first_xpath(element, get_xpath(is_left=True))
        snd_el = co.find_first_xpath(element, get_xpath(is_left=False))
        if fst_el is not None and snd_el is not None:
            result = fst_el.text, snd_el.text
            if result != ("", ""):
                return result

    def make_score(retired):
        def make_side_scr_elements(is_left):
            inclass_txt = f"event__part--{'home' if is_left else 'away'}"
            xpath_txt = (
                f"child::div[contains(@class, '{inclass_txt}') "
                f"and not(contains(@class, 'event__part--6'))]"
            )
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
        tourinfo_cache.edit_skip(live_event.tour_info)
    if cur_st != match_status:
        return None
    obj = LiveMatch(live_event)
    obj.href = make_href()
    if cur_st == MatchStatus.scheduled:
        obj.time = get_time()
    obj.date = current_date
    left_name = get_name(is_left=True)
    left_cou = alpha_3_code(get_cou(is_left=True))
    right_name = get_name(is_left=False)
    right_cou = alpha_3_code(get_cou(is_left=False))
    obj.name = f"{left_name} - {right_name}"
    obj.first_player = _find_player(live_event.sex, left_name, left_cou)
    obj.second_player = _find_player(live_event.sex, right_name, right_cou)
    if cur_st == MatchStatus.live:
        obj.left_service = is_left_service()
        obj.ingame = ingame_score()
    obj.score = make_score(retired=cur_sub_st == MatchSubStatus.retired)
    return obj


wta_today_players_cache = defaultdict(lambda: None)  # (disp_name, cou) -> Player
atp_today_players_cache = defaultdict(lambda: None)
init_players_cache_mode = False

deep_find_player_mode = False


def initialize_players_cache(webpage, match_status=MatchStatus.scheduled):
    global deep_find_player_mode, init_players_cache_mode
    mem_deep_find_player_mode = deep_find_player_mode
    deep_find_player_mode = True
    init_players_cache_mode = True
    try:
        make_events(
            webpage,
            skip_levels=skip_levels_work(),
            match_status=match_status,
            target_date=datetime.date.today(),
        )
    finally:
        deep_find_player_mode = mem_deep_find_player_mode
        init_players_cache_mode = False


def _find_player(sex: str, disp_name: str, cou: str):
    cache = wta_today_players_cache if sex == "wta" else atp_today_players_cache
    if cache and not init_players_cache_mode:
        return cache[(disp_name, cou)]
    if deep_find_player_mode:
        plr = player_name_ident.identify_player(sex, disp_name, cou)
        if init_players_cache_mode:
            if plr is not None:
                cache[(disp_name, cou)] = plr
            else:
                log.warning("fail player ident {} '{}' {}".format(sex, disp_name, cou))
        return plr
    else:
        return co.find_first(
            oncourt_players.players(sex),
            lambda p: p.cou == cou and p.disp_name("flashscore") == disp_name,
        )


def _make_tourinfo(element, start_idx, skip_levels):
    def is_skip_key():
        return (
            "DOUBLES" in evt_title_text
            or "EXHIBITION" in evt_title_text
            or "Next Gen Finals" in evt_title_text
            or "Laver Cup" in evt_title_text
        )

    def is_skip_obj(tourinfo_fs: TourInfoFlashscore):
        return (
            tourinfo_fs.level in skip_levels[tourinfo_fs.sex]
            or tourinfo_fs.sex == SKIP_SEX
            or tourinfo_fs.teams
            or (tourinfo_fs.qualification and tourinfo_fs.level in ('chal', 'future'))
        )

    def split_before_after_text(text, part):
        if part in text:
            before_including = co.strip_after_find(text, part)
            after_not_including = text[text.index(part) + len(part):]
            return before_including, after_not_including

    def split_sex_doubles_text(text):
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
                etree.tostring(element, encoding="utf8"))
        )
        raise Exception(
            "class != event__header got class: '{}' start_idx {}".format(
                element.get("class"), start_idx))

    evt_title_el = co.find_first_xpath(
        element, "child::div[contains(@class,'event__title ')]")
    if evt_title_el is None:
        raise co.TennisParseError(
            "not parsed sex_doubles el T24 TourInfo from\n{}".format(
                etree.tostring(element, encoding="utf8")))
    evt_title_text = evt_title_el.text_content().strip()
    if not evt_title_text:
        raise co.TennisParseError("empty sex_doubles txt T24 TourInfo")

    if is_skip_key():
        raise LiveEventToskipError()
    obj = tourinfo_cache.get(evt_title_text)
    if obj is not None:
        return obj

    evt_title_parts = split_sex_doubles_text(evt_title_text)
    if evt_title_parts is None:
        raise co.TennisParseError(
            "fail split sex_doubles: '{}'\n{}".format(
                evt_title_text, etree.tostring(element, encoding="utf8"))
        )
    obj = TourInfoFlashscore(evt_title_parts[0], evt_title_parts[1])
    if is_skip_obj(obj):
        tourinfo_cache.put_skip(evt_title_text)
    else:
        tourinfo_cache.put(evt_title_text, obj)
        return obj


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
    """ sample: text 'ab(c)-d(ef)' delim_op='(', delim_cl=')' return ('ab(c)-d', 'ef') """
    if text.endswith(delim_cl):
        pos = text.rfind(delim_op)
        if pos >= 0:
            return (
                text[:pos].strip(),
                text[pos + len(delim_op): len(text) - len(delim_cl)].strip(),
            )
    return text.strip(), ""


class TourInfoFlashscore(TourInfo):
    def __init__(self, part_one, part_two):
        super().__init__()

        body_part, surf_part = co.split_ontwo(part_two, delim=', ')
        body_part, qual_part = co.split_ontwo(body_part, delim=' - ')
        body_part, country_part = split_ontwo_enclosed(
            body_part, delim_op='(', delim_cl=')')

        if '(' in body_part:
            body_part, desc_part = split_ontwo_enclosed(
                body_part, delim_op='(', delim_cl=')')
        elif ' - ' in body_part:
            body_part, desc_part = co.split_ontwo(body_part, delim=' - ')
        else:
            desc_part = ''
        body_part = body_part.split(',')[0]

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
        if "WOMEN" in part_one or "WTA" in part_one or "GIRLS" in part_one:
            self.sex = "wta"
        else:
            self.sex = "atp"
        self.doubles = "DOUBLES" in body_part
        is_grand_slam = (
            "JUNIORS" not in body_part
            and "BOYS" not in part_one
            and "GIRLS" not in part_one
            and (
                "French Open" in body_part
                or "Wimbledon" in body_part
                or "U.S. Open" in body_part
                or "US Open" in body_part
                or "Australian Open" in body_part
            )
        )
        self.qualification = "Qualification" in qual_part and not self.teams
        self.exhibition = "EXHIBITION" in part_one
        self.itf = part_one.startswith("ITF")
        self.surface = make_surf(surf_part) if surf_part else None
        self.tour_name, money = self.init_tourname(body_part, desc_part, number)
        # if end season Finals (rnd 1/2 or Final) -> qual_part='Play Offs' country='World'
        self.level = self.__init_level(is_grand_slam, part_one, qual_part, money)
        self.country = country_part
        self.corrections()

    def corrections(self):
        surface = sex_tourname_surf_map.get((self.sex, self.tour_name, self.surface))
        if surface is not None:
            self.surface = surface
        elif (
            self.surface == Hard
            and (
                datetime.date.today().month in (12, 1, 2)
                or (
                    datetime.date.today().month == 3
                    and datetime.date.today().day < 20
                )
            )
            and city_in_europe(self.tour_name.name)  # not play hard in winter europe
        ):
            self.surface = Carpet

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
        if num is None:
            desc_part, num = tourname_number_split(desc_part)
        return (
            TourName(name=body_part, desc_name=desc_part, number=num),
            money,
        )

    def __init_level(self, is_grand_slam, part_one, qual_part, money):
        if is_grand_slam:
            result = lev.gs
        elif "CHALLENGER" in part_one:
            result = lev.chal
        elif "BOYS" in part_one or "GIRLS" in part_one:
            result = lev.junior
        elif self.teams:
            if "World Group" in qual_part or "ATP Cup" in self.tour_name.name:
                result = lev.teamworld
            else:
                result = lev.team
        elif self.itf:
            if self.sex == "wta" and (
                (money is not None and money > MAX_WTA_FUTURE_MONEY)
                or (self.tour_name, self.surface) in wta_chal_tour_surf
            ):
                result = lev.chal
            else:
                result = lev.future
        else:
            result = lev.main
        return result


tourinfo_cache = TourInfoCache(
    skip_exc_cls=LiveEventToskipError,
    skip_keys=('ITF WOMEN - SINGLESW80 Macon, GA (USA)',)
)


def _make_current_date(root_elem):
    i_el = co.find_first_xpath(root_elem, "//svg[@class='calendar__icon']")
    if i_el is not None:
        date_txt = i_el.tail
        if date_txt is not None:
            date_txt = date_txt.strip()
            if len(date_txt) >= 5:
                day_txt, month_txt = date_txt[:5].split(r"/")
                return datetime.date(
                    year=datetime.date.today().year,
                    month=int(month_txt),
                    day=int(day_txt),
                )


def goto_date(fsdrv, days_ago, start_date, wait_sec=5):
    """ goto days_ago into past from start_date (today if start_date is None).
        if daysago > 0 then go to backward, if daysago=-1 then go to forward (+1 day)
        :returns target_date if ok, or raise TennisError
    """

    def prev_day_button_coords():
        # y=695 with advertise. handy measure at Gennady notebook. y=585 without advertise
        return 1235, 585

    def next_day_button_coords():
        return 1235 + 184, 585

    def neighbour_day_click(is_backward):
        import automate

        if is_backward:
            x, y = prev_day_button_coords()
        else:
            x, y = next_day_button_coords()
        automate.mouse_click((x, y))
        fsdrv.implicitly_wait(wait_sec)
        time.sleep(5)

    target_date = start_date - datetime.timedelta(days=days_ago)
    for _ in range(abs(days_ago)):
        if days_ago >= 0:
            neighbour_day_click(is_backward=True)
        else:
            neighbour_day_click(is_backward=False)
    fsdrv.implicitly_wait(wait_sec)
    parser = lxml.html.HTMLParser(encoding="utf8")
    tree = lxml.html.document_fromstring(fsdrv.page(), parser)
    cur_date = _make_current_date(tree)
    if cur_date != target_date:
        raise co.TennisError(
            "target_date {} != cur_date {} days_ago: {}".format(
                target_date, cur_date, days_ago)
        )
    return cur_date


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
                "T24::init prev_week {} wta_chal {} {}".format(
                    prev_week, tour.name, tour.surface)
            )


sex_tourname_surf_map = {  # here in values are more correct surface then flashscore's
    ("wta", TourName("Chicago"), Carpet): Hard,
    ("wta", TourName("Fukuoka"), Hard): Carpet,
    ("wta", TourName("Kurume"), Hard): Carpet,
    ("atp", TourName("ATP Cup"), Carpet): Hard,
}
