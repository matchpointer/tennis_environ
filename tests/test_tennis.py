# -*- coding=utf-8 -*-
import datetime
import pytest

import lev
import score as sc
from tennis import Round, Player, Match, HeadToHead


class TestSoftLevel:
    @staticmethod
    def test_masters_q():
        val = lev.soft_level(level=lev.masters, rnd=None, qualification=True)
        assert val == lev.main

        val = lev.soft_level(level=lev.masters, rnd=Round('q-First'), qualification=True)
        assert val == lev.main

        val = lev.soft_level(level=lev.masters, rnd=Round('q-First'))
        assert val == lev.main

    @staticmethod
    def test_main_q():
        val = lev.soft_level(level=lev.main, rnd=Round('q-First'), qualification=True)
        assert val == 'qual'

        val = lev.soft_level(level=lev.main, rnd=Round('q-First'))
        assert val == 'qual'

        val = lev.soft_level(level=lev.main, rnd=None, qualification=True)
        assert val == 'qual'

    @staticmethod
    def test_gs_q():
        val = lev.soft_level(level=lev.gs, rnd=Round('q-First'), qualification=True)
        assert val == 'qual'

        val = lev.soft_level(level=lev.gs, rnd=Round('q-First'))
        assert val == 'qual'

        val = lev.soft_level(level=lev.gs, rnd=None, qualification=True)
        assert val == 'qual'

    @staticmethod
    def test_teamworld_q():
        val = lev.soft_level(level=lev.teamworld, rnd=Round('q-First'), qualification=True)
        assert val == 'qual'

        val = lev.soft_level(level=lev.teamworld, rnd=Round('q-First'))
        assert val == 'qual'

        val = lev.soft_level(level=lev.teamworld, rnd=None, qualification=True)
        assert val == 'qual'

    @staticmethod
    def test_team_q():
        val = lev.soft_level(level=lev.team, rnd=Round('q-First'), qualification=True)
        assert val == 'qual'

        val = lev.soft_level(level=lev.team, rnd=Round('q-First'))
        assert val == 'qual'

        val = lev.soft_level(level=lev.team, rnd=None, qualification=True)
        assert val == 'qual'

    @staticmethod
    def test_chal_q():
        val = lev.soft_level(level=lev.chal, rnd=Round('q-First'), qualification=True)
        assert val == lev.chal

        val = lev.soft_level(level=lev.chal, rnd=Round('q-First'))
        assert val == lev.chal

        val = lev.soft_level(level=lev.chal, rnd=None, qualification=True)
        assert val == lev.chal


def test_round_cmp():
    assert Round("Rubber 3") < "Rubber 5"
    assert Round("Rubber 3") > "Rubber 2"
    assert "Rubber 3" < Round("Rubber 5")
    assert "Rubber 3" > Round("Rubber 2")

    assert "Rubber 3" == Round("Rubber 3")
    assert Round("Rubber 3") == "Rubber 3"

    assert not Round("Rubber 2").qualification()

    assert Round("First") == "First"


@pytest.mark.skip("it was actual when match was in database")
def test_h2h_ito_kukushkin(get_dba):
    plr1 = Player(ident=8313, name="Tatsuma Ito", cou="JPN")
    plr2 = Player(ident=9043, name="Mikhail Kukushkin", cou="KAZ")
    match = Match(
        first_player=plr1,
        second_player=plr2,
        score=sc.Score("4-6 7-6(3) 7-6(6)"),
        rnd=Round("Second"),
        date=datetime.date(2017, 2, 1),
    )
    h2hdir = HeadToHead(sex="atp", match=match, completed_only=True).direct()
    assert h2hdir is None


@pytest.mark.skip("actual only was 2019,10,18 and recent h2h match was 2019,9,26")
def test_h2h_recent_winner(get_dba):
    plr1 = Player(ident=6590, name="Cagla Buyukakcay", cou="TUR")
    plr2 = Player(ident=8951, name="Veronica Cepede Royg", cou="PAR")
    match = Match(first_player=plr1, second_player=plr2,
                  score=None, rnd=Round('Second'),
                  date=datetime.date(2019, 10, 18))
    h2h = HeadToHead(sex='wta', match=match, completed_only=True)
    recent_winner_id = h2h.recently_won_player_id()
    assert recent_winner_id == plr1.ident

