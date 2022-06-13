# -*- coding=utf-8 -*-
from typing import List

from pages.base import WebPage
from live import MatchStatus, LiveTourEvent
import flashscore
import tennis24


class ScoreCompany:
    def __init__(self, abbrname):
        assert abbrname in _abbrnames
        self.abbrname = abbrname

    def __repr__(self):
        return self.abbrname

    def start_url(self):
        raise NotImplementedError()

    def players_key(self):
        raise NotImplementedError()

    def fetch_events(
            self, page_source: str, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        """ target_date used for construct datetime attr for matches.
            If target_date is None then assume today date """
        raise NotImplementedError()

    def initialize(self, prev_week=False):
        """ (must be called AFTER weeked_tours::initialize)
            does init wta_chal_tour_surf, and add to sex_tourname_surf_map """
        raise NotImplementedError()

    def initialize_players_cache(self, page_source: str,
                                 match_status=MatchStatus.scheduled):
        raise NotImplementedError()

    def goto_date(self, wpage: WebPage, days_ago: int, start_date):
        """ goto days_ago into past from start_date (today if start_date is None).
            if daysago > 0 then go to backward, if daysago=-1 then go to forward (+1 day)
            :returns target_date if ok, or raise TennisError
        """
        for _ in range(int(abs(days_ago))):
            if days_ago < 0:
                wpage.prev_day_button.click()
            else:
                wpage.next_day_button.click()
            wpage.wait_page_loaded()
        return wpage.parse_date()


class FlashscoreCompany(ScoreCompany):
    def start_url(self):
        return "http://www.flashscore.com/tennis/"

    def players_key(self):
        return 'flashscore'

    def fetch_events(
            self, page_source: str, skip_levels, match_status=MatchStatus.live,
            target_date=None
    ) -> List[LiveTourEvent]:
        return flashscore.make_events(
            page_source, skip_levels, match_status,
            target_date=target_date, plr_country_long=True)

    def initialize(self, prev_week=False):
        """
        must be called AFTER weeked_tours::initialize.
        init wta_chal_tour_surf, and add to sex_tourname_surf_map
        """
        flashscore.initialize(prev_week=prev_week)

    def initialize_players_cache(self, page_source: str,
                                 match_status=MatchStatus.scheduled):
        flashscore.initialize_players_cache(page_source, match_status=match_status)


class Tennis24Company(ScoreCompany):
    def start_url(self):
        return "http://www.tennis24.com/"

    def players_key(self):
        return 'flashscore'  # remain super name

    def fetch_events(
            self, page_source: str, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        return tennis24.make_events(
            page_source, skip_levels, match_status, target_date=target_date)

    def initialize(self, prev_week=False):
        """
        must be called AFTER weeked_tours::initialize.
        init wta_chal_tour_surf, and add to sex_tourname_surf_map
        """
        tennis24.initialize(prev_week=prev_week)

    def initialize_players_cache(self, page_source: str,
                                 match_status=MatchStatus.scheduled):
        tennis24.initialize_players_cache(page_source, match_status=match_status)


class TennisbetsiteCompany(ScoreCompany):
    def start_url(self):
        return "http://www.tennisbetsite.com/"

    def players_key(self):
        return 'flashscore'  # remain super name

    def fetch_events(
            self, page_source: str, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        raise NotImplementedError()

    def initialize(self, prev_week=False):
        raise NotImplementedError()

    def initialize_players_cache(self, page_source: str,
                                 match_status=MatchStatus.scheduled):
        raise NotImplementedError()

    def goto_date(self, wpage: WebPage, days_ago: int, start_date):
        raise NotImplementedError()


_abbrnames = ('FS', 'T24', 'TBS')

FS = FlashscoreCompany('FS')
T24 = Tennis24Company('T24')
TBS = TennisbetsiteCompany('TBS')


def get_company(abbrname: str):
    if abbrname == 'T24':
        return T24
    if abbrname == 'FS':
        return FS
    if abbrname == 'TBS':
        return TBS
