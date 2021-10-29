import datetime
import pytest

from tour_name import TourName
from tennis import Player, Round, Level
from score import Score

from flashscore_match import (
    parse_title_score,
    parse_odds,
    parse_match_detailed_score,
)


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
