# -*- coding: utf-8 -*-
import datetime
import pytest

from tour_name import TourName
import common as co
from tennis import Player, Round, Level
from score import Score

from flashscore_match import (
    parse_title_score,
    TitleScore,
    parse_odds,
    build_root,
    parse_match_detailed_score,
)
from detailed_score_misc import missed_match_points, MatchPointCounts


class FakeMatch:
    def __init__(
        self,
        date,
        sex: str,
        tour_name: TourName,
        level: Level,
        rnd: Round,
        fst_name: str,
        snd_name: str,
        std_score: Score,
        href: str,
        best_of_five=None,
    ):
        self.date = date
        self.sex = sex
        self.tour_name = tour_name
        self.first_player = Player(ident=None, name=fst_name)
        self.second_player = Player(ident=None, name=snd_name)
        self.level = level
        self.rnd = rnd
        self.score = std_score
        self.href = href
        self.best_of_five = best_of_five

    @property
    def qualification(self):
        return self.rnd.qualification()

    def pointbypoint_href(self, setnum=1):
        if self.href:
            return f"{self.href}/#match-summary/point-by-point/{setnum - 1}"

    def summary_href(self):
        if self.href:
            return "{}/#match-summary/match-summary".format(self.href)


def test_parse_det_score_gasanova_pavly(fscore_driver):
    def make_fake_match():
        return FakeMatch(
            date=datetime.date(2021, 3, 18),
            sex="wta",
            tour_name=TourName("St. Petersburg"),
            level=Level("main"),
            rnd=Round("Second"),
            fst_name="Anastasia Gasanova",
            snd_name="Anastasia Pavlyuchenkova",
            std_score=Score("1-6 7-6(8) 7-6 (4)"),
            href="https://www.flashscore.com/match/6NvqT9V8",
        )

    match = make_fake_match()
    verbose = True
    res = parse_match_detailed_score(match, fsdriver=fscore_driver, verbose=verbose)
    is_attr = hasattr(match, "detailed_score")
    print(f"res={res} is_attr:{is_attr}")
    assert res is True
    assert is_attr is True
    if is_attr:
        print(str(match.detailed_score))
        assert match.detailed_score.final_score() == ((1, 6), (7, 6), (7, 6))

        lose_side = co.RIGHT
        saved_mp = missed_match_points(
            match.detailed_score, lose_side, best_of_five=False
        )
        assert saved_mp == MatchPointCounts(mp_on_srv=1, mp_on_rcv=2)


def test_parse_det_score_popko_rola(fscore_driver):
    def make_fake_match():
        return FakeMatch(
            date=datetime.date(2021, 3, 13),
            sex="atp",
            tour_name=TourName("Dubai"),
            level=Level("main"),
            rnd=Round("q-Second"),
            fst_name="Dmitry Popko",
            snd_name="Blaz Rola",
            std_score=Score("3-6 6-4 6-7 (12)"),
            href="https://www.flashscore.com/match/6XLRjIHk",
        )

    match = make_fake_match()
    verbose = True
    res = parse_match_detailed_score(match, fsdriver=fscore_driver, verbose=verbose)
    is_attr = hasattr(match, "detailed_score")
    # print(f"res={res} is_attr:{is_attr}")
    assert res is True
    assert is_attr is True
    if is_attr:
        # print(str(match.detailed_score))
        assert match.detailed_score.final_score() == ((3, 6), (6, 4), (6, 7))


def test_parse_title_score_finished(fscore_driver):
    match_href = (
        "https://www.flashscore.com/match/4K1Qa9M9/#match-summary/match-summary"
    )
    res = parse_title_score(match_href, fscore_driver)
    assert res is not None
    if res is not None:
        assert res.inmatch == (2, 1)
        assert res.inset is None
        assert res.finished is True


def test_parse_prematch_odds(fscore_driver):
    match_href = (
        "https://www.flashscore.com/match/4K1Qa9M9/#match-summary/match-summary"
    )
    r = parse_odds(match_href, fscore_driver)
    assert r is not None
    if r is not None:
        odds_1, odds_2, is_live = r
        # print(f'\nodds: {odds_1}, {odds_2} live:{is_live}')
        assert odds_1 == 1.74
        assert odds_2 == 2.1
        assert not is_live


@pytest.mark.skip(reason="requires adhoc live")
def test_parse_live_odds(fscore_driver):
    match_href = (
        "https://www.flashscore.com/match/UBfHNuei/#match-summary/match-summary"
    )
    r = parse_odds(match_href, fscore_driver)
    assert r is not None
    if r is not None:
        # for my region will be answered 1xbet data:
        odds_1, odds_2, is_live = r
        # print(f'\nodds: {odds_1}, {odds_2} live:{is_live}')
        assert (1 <= odds_1 <= 1.5) and (10 <= odds_2 <= 15)
        assert is_live


@pytest.mark.skip(reason="requires adhoc live")
def test_parse_live_title_score_stages(fscore_driver):
    match_href = (
        "https://www.flashscore.com/match/SzjH9kVe/#match-summary/match-summary"
    )
    res = parse_title_score(summary_href=match_href, fsdriver=fscore_driver)
    assert res is not None
    if res is not None:
        assert res.is_left_srv is False
        assert res.inmatch == (0, 1)
        assert res.inset == (2, 3)
        assert not res.finished
        assert res.setnum == 2


# here odds values: (1.57, 2.25)
pre_match_odds_parent = """
<div class="oddsWrapper___3ShkMCE">
    <div class="oddsRow___Ik2HKHW">
        <div class="oddsRowContent___gXFchnM">
            <div class="odds___2aJyDCz live-odds ">
                <div class="bookmaker___2U_yeE5 "><a
                        href="/bookmaker/16/?sport=2&amp;from=detail&amp;gicc=MD&amp;gisc=MD-CU" title="bet365"
                        target="_blank" class="link___2cfGV84"><img class="logo___1VsHmha"
                                                                    src="/res/image/data/bookmakers/30-16.png"
                                                                    title="bet365" alt="bet365"></img></a></div>
                <div class="liveBetIcon___3AjBj-V" title="This match will be available for LIVE betting!">
                </div>
                <div class="cellWrapper___2KG4ULl" title="1.53[u]1.57[br]Add this match to bet slip on bet365!"><span
                        class="cell___2ejh55S o_1  "><span class="oddsType___3n54QLL">1</span><span
                        class="oddsValue___1euLZeq odds-wrap up">1.57</span></span></div>
                <div class="cellWrapper___2KG4ULl" title="2.37 » 2.25
Add this match to bet slip on bet365!"><span class="cell___2ejh55S o_2  "><span class="oddsType___3n54QLL">2</span><span
                        class="oddsValue___1euLZeq odds-wrap down">2.25</span></span></div>
            </div>
        </div>
    </div>
    <div class="bonus___SZCjQPv" style="background-color: rgb(1, 123, 91);"><a class="link___1tMC5YI"
                                                                               href="/bookmaker/16/?from=detail-bonus&amp;sport=2&amp;gicc=MD&amp;gisc=MDCU&amp;bonusId=27"
                                                                               target="_blank"
                                                                               style="color: rgb(255, 255, 255);">100%
        bonus up to €50 / €100</a>
        <div class="description___26h6Yk5" style="color: rgb(255, 255, 255);">Check bet365.com for latest offers and
            details. Geo-variations and T&amp;Cs apply. 18+
        </div>
    </div>
</div>"""
