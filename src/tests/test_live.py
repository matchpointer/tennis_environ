import datetime
import pytest

import common as co
import feature
from live import LiveMatch, LiveTourEvent, TourInfo
import ratings


def wta_abudhabi_tourinfo():
    return TourInfo(
        sex="wta",
        tour_name="abu-dhabi",
        surface="Hard",
        level="main",
        qualification=False,
        cou="UAE",
    )


def wta_abudhabi_evt():
    return LiveTourEvent(tour_info=wta_abudhabi_tourinfo())


def raw_match_siegemund_flipkens():
    match = LiveMatch(live_event=wta_abudhabi_evt())
    match.name = "Siegemund L. - Flipkens K."
    return match


@pytest.fixture()
def ini_rtg_wta_since_pandemia(get_dba):
    sex = "wta"
    ratings.initialize(
        sex=sex,
        rtg_names=("std", "elo"),
        min_date=datetime.date(2020, 3, 16),  # last prev pandemia date
    )
    ratings.Rating.register_rtg_name("elo")

    yield

    ratings.clear(sex=sex, rtg_names=("std", "elo"))


def test_wta_load_pre_live_data():
    m = raw_match_siegemund_flipkens()
    m.load_pre_live_data()
    assert len(m.features) > 0
    f1 = co.find_first(m.features, lambda f: f.name == "fst_plr_tour_adapt")
    f2 = co.find_first(m.features, lambda f: f.name == "snd_plr_tour_adapt")
    assert f1 is not None
    assert f2 is not None
    assert f1.value == -60
    assert f2.value == -28
    dif = feature.dif_player_features(m.features, "plr_tour_adapt")
    assert dif == 32
