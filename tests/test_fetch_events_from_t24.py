import datetime
import os
from collections import defaultdict

import pytest

from loguru import logger as log
import oncourt_players
import tennis_time as tt
import tournament_misc as trmt_misc
import weeked_tours
from live import MatchStatus, skip_levels_work, skip_levels_default


match_status = MatchStatus.scheduled


def get_skip_levels(restrict):
    if restrict:
        return skip_levels_work()
    elif restrict is False:
        return skip_levels_default()
    elif restrict is None:
        return defaultdict(list)


skip_levels = get_skip_levels(restrict=False)


@pytest.fixture(scope="module")
def prep_t24_company_page(get_dba, get_t24_comp, t24_driver):
    log.add(f"../log/{os.path.basename(__file__).replace('.py', '.log')}",
            level='INFO')
    oncourt_players.initialize(yearsnum=1.2)
    min_date = tt.past_monday_date(datetime.date.today()) - datetime.timedelta(days=7)
    weeked_tours.initialize_sex(
        "wta", min_date=min_date, max_date=None, with_today=True, with_bets=True)
    weeked_tours.initialize_sex(
        "atp", min_date=min_date, max_date=None, with_today=True, with_bets=True)

    # here we potentially corrupt get_t24_comp if use in other module of current session?
    get_t24_comp.initialize()

    yield get_t24_comp, t24_driver.page()  # this is where the testing happens


def _count_defined_matches(evt):
    num_ok, num_fail = 0, 0
    for match in evt.matches:
        if (
            match.first_player
            and match.first_player.ident
            and match.second_player
            and match.second_player.ident
        ):
            num_ok += 1
        else:
            num_fail += 1
    return num_ok, num_fail


@pytest.mark.webtest
def test_make_and_log_events(prep_t24_company_page):
    scr_company, page = prep_t24_company_page

    events = scr_company.fetch_events(
        webpage=page,
        skip_levels=skip_levels,
        match_status=match_status,
    )
    for event in events:
        event.define_players()
        event.define_level()

        n_ok, n_fail = _count_defined_matches(event)
        log.info(
            f"{event.tour_name} tour_id: {event.tour_id} n_ok: {n_ok} n_fail: {n_fail}")
        if (n_ok + n_fail) > 0:
            assert n_ok > 0

    trmt_misc.log_events(
        events,
        head=f"test_make_and_log_events {match_status}",
        extended=True
    )


