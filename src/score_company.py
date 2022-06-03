from typing import List

from live import MatchStatus, LiveTourEvent
import flashscore
import tennis24


class ScoreCompany:
    abbrnames = ('FS', 'T24')

    def __init__(self, abbrname):
        assert abbrname in self.abbrnames
        self.abbrname = abbrname

    def __repr__(self):
        return self.abbrname

    def start_url(self):
        raise NotImplementedError()

    def players_key(self):
        raise NotImplementedError()

    def fetch_events(
            self, webpage, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        """ target_date used for construct datetime attr for matches.
            If target_date is None then assume today date """
        raise NotImplementedError()

    def initialize(self, prev_week=False):
        """ (must be called AFTER weeked_tours::initialize)
            does init wta_chal_tour_surf, and add to sex_tourname_surf_map """
        raise NotImplementedError()

    def initialize_players_cache(self, webpage, match_status=MatchStatus.scheduled):
        raise NotImplementedError()

    def goto_date(self, drv, days_ago: int, start_date, wait_sec=5):
        """ goto days_ago into past from start_date (today if start_date is None).
            if daysago > 0 then go to backward, if daysago=-1 then go to forward (+1 day)
            :returns target_date if ok, or raise TennisError
        """
        raise NotImplementedError()


class FlashscoreCompany(ScoreCompany):
    def start_url(self):
        return "http://www.flashscore.com/tennis/"

    def players_key(self):
        return 'flashscore'

    def fetch_events(
            self, webpage, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        return flashscore.make_events(
            webpage, skip_levels, match_status,
            target_date=target_date, plr_country_long=True)

    def initialize(self, prev_week=False):
        """
        must be called AFTER weeked_tours::initialize.
        init wta_chal_tour_surf, and add to sex_tourname_surf_map
        """
        flashscore.initialize(prev_week=prev_week)

    def initialize_players_cache(self, webpage, match_status=MatchStatus.scheduled):
        flashscore.initialize_players_cache(webpage, match_status=match_status)

    def goto_date(self, drv, days_ago: int, start_date, wait_sec=5):
        flashscore.goto_date(drv, days_ago, start_date, wait_sec=wait_sec)


class Tennis24Company(ScoreCompany):
    def start_url(self):
        return "http://www.tennis24.com/"

    def players_key(self):
        return 'flashscore'  # remain super name

    def fetch_events(
            self, webpage, skip_levels, match_status=MatchStatus.live, target_date=None
    ) -> List[LiveTourEvent]:
        return tennis24.make_events(
            webpage, skip_levels, match_status, target_date=target_date)

    def initialize(self, prev_week=False):
        """
        must be called AFTER weeked_tours::initialize.
        init wta_chal_tour_surf, and add to sex_tourname_surf_map
        """
        tennis24.initialize(prev_week=prev_week)

    def initialize_players_cache(self, webpage, match_status=MatchStatus.scheduled):
        tennis24.initialize_players_cache(webpage, match_status=match_status)

    def goto_date(self, drv, days_ago: int, start_date, wait_sec=5):
        tennis24.goto_date(drv, days_ago, start_date, wait_sec=wait_sec)


FS = FlashscoreCompany('FS')
T24 = Tennis24Company('T24')


def get_company(abbrname: str):
    if abbrname == 'T24':
        return T24
    if abbrname == 'FS':
        return FS
