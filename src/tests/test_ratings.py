import datetime
import pytest

import ratings
from tennis import Player


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


def test_wta_plr_read_rank(ini_rtg_wta_since_pandemia):
    sex = "wta"
    plr = Player(ident=7574, name="Petra Martic", cou="CRO")
    date = datetime.date(2020, 8, 3)
    plr.read_rating(sex=sex, date=date, surfaces=("all",))
    rank = plr.rating.rank(rtg_name="std")
    assert rank == 15
