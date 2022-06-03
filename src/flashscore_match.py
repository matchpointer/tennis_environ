""" work with flashscore point by point page(s); also work with match summary page.
    goes on LiveMatch.pointbypoint_href() for match point-by-point info.
    TODO parse player full name lowercased from tag a:
    <div class="team-primary-content">
      ..<div class="team-text tname-home">
      ..<div class="team-text tname-away">
        ..<div class="tname__text">
          <a href="#" class="participant-imglink"
          onclick="window.open('/player/lu-jia-jing/0OZYPwg9'); return false;">Lu J.</a>
"""
import copy
import lxml.html

# from lxml.etree import tostring
import random
from typing import Optional, Tuple, NamedTuple

import common as co
import wdriver
from detailed_score import DetailedScore, DetailedGame
import score as sc
from side import Side


class TitleScore:
    def __init__(
        self,
        inmatch=None,
        setnum=None,
        inset=None,
        ingame=None,
        is_left_srv=None,
        finished=None,
    ):
        self.inmatch: Optional[Tuple[int, int]] = inmatch
        self.setnum: Optional[int] = setnum
        self.inset: Optional[Tuple[int, int]] = inset
        self.ingame: Optional[Tuple[str, str]] = ingame
        self.is_left_srv: Optional[bool] = is_left_srv
        self.finished: Optional[bool] = finished

    def __repr__(self):
        res = ""
        if self.inmatch:
            res = f"inm:{self.inmatch}"
        if self.setnum:
            res += f" sn:{self.setnum}"
        if self.inset:
            res += f" ins:{self.inset}"
        if self.is_left_srv is not None:
            res += f" srv_left:{self.is_left_srv}"
        if self.finished:
            res += " finished"
        return (res if res else "<empty title scr>") + " " + hex(id(self))

    def inset_games_count(self):
        if self.inset is not None:
            return sum(self.inset)

    def srv_side_at_begin(self, setnum: int):
        if (
            self.setnum is not None
            and self.inset is not None
            and self.is_left_srv is not None
        ):
            if self.setnum == setnum:
                return (
                    Side(self.is_left_srv)
                    if co.is_even(sum(self.inset))
                    else Side(not self.is_left_srv)
                )
            if (self.setnum + 1) == setnum and max(self.inset) >= 5:
                prob_set_winner = Side(self.inset[0] > self.inset[1])
                if self.is_left_srv is prob_set_winner.is_left():
                    return prob_set_winner.fliped()
                return prob_set_winner


def build_root(url, fsdriver, verbose: bool = False, print_err=print):
    wdriver.load_url(fsdriver.drv, url)
    fsdriver.implicitly_wait(3)
    page = fsdriver.page()
    if page is None:
        if verbose:
            print_err(f"not fetched summary page in {url}")
        return None
    parser = lxml.html.HTMLParser(encoding="utf8")
    tree = lxml.html.document_fromstring(page, parser)  # lxml.html.HtmlElement
    if tree is None:
        if verbose:
            print_err(f"fail build root url:{url}")
        return None
    content_el = co.find_first_xpath(tree, '//*[@id="detail"]')
    if content_el is None:
        if verbose:
            print_err(f"fail tennis/detail url:{url}")
        return None
    return content_el


def parse_odds(
    summary_href, fsdriver, verbose: bool = False, print_err=print
) -> Optional[Tuple[float, float, bool]]:
    def is_live_active():
        qtxt = 'descendant::div[@class="tabs__tab selected" and text() = "Live odds"]'
        return co.find_first_xpath(root, qtxt) is not None

    def get_odds():
        elements = root.xpath("descendant::span[starts-with(@class, 'oddsValue___')]")
        result_odds = []
        if elements:
            for elem in elements:
                txt = elem.text_content()
                if txt:
                    try:
                        result_odds.append(float(txt.strip()))
                    except ValueError:
                        result_odds.append(None)  # '-' coef is not presented
        return result_odds

    root = build_root(summary_href, fsdriver, verbose=verbose)
    odds = get_odds()
    if len(odds) != 2:
        if verbose:
            print_err(f"fail odds: {odds}")
        return None
    return odds[0], odds[1], is_live_active()


def parse_title_score(summary_href, fsdriver,
                      verbose: bool = False, print_err=print) -> TitleScore:
    """return TitleScore or None."""

    def is_left_serve():
        xpath_txt = (
            "child::div[@class='duelParticipant__home']/"
            "div[@class='participant__participantServe']/"
            "div[@title='Serving player']"
        )
        return co.find_first_xpath(entry_elem0, xpath_txt) is not None

    def is_right_serve():
        xpath_txt = (
            "child::div[@class='duelParticipant__away']/"
            "div[@class='participant__participantServe']/"
            "div[@title='Serving player']"
        )
        return co.find_first_xpath(entry_elem0, xpath_txt) is not None

    def fill_inmatch():
        xpath_txt = "child::div[starts-with(@class, 'detailScore__wrapper')]"
        inmatch_el = co.find_first_xpath(entry_elem, xpath_txt)
        if inmatch_el is not None:
            txt = inmatch_el.text_content().strip()
            if txt:
                fst, snd = txt.split("-")
                if fst and snd:
                    result.inmatch = int(fst), int(snd)

    def fill_details():
        status_el = co.find_first_xpath(
            entry_elem, "child::div[starts-with(@class, 'detailScore__status')]"
        )
        if status_el is not None:
            detstatus_el = co.find_first_xpath(
                status_el, "child::span[contains(@class, 'detailStatus')]"
            )
            if detstatus_el is not None:
                text = detstatus_el.text
                if text:
                    if "Finished" in text:
                        result.finished = True
                    elif "Set " in text:
                        result.setnum = int(text.strip()[4])

            # games tuple-2 score
            scr_el = co.find_first_xpath(
                status_el, "child::span[@class='detailScore__detailScoreServe']"
            )
            if scr_el is not None:
                text = scr_el.text
                if text and ":" in text:
                    fst, snd = text.strip().split(":")
                    result.inset = int(fst), int(snd)

    result = TitleScore()
    root = build_root(summary_href, fsdriver, verbose=verbose, print_err=print_err)
    if root is None:
        if verbose:
            print_err(f"root build failed, title score in {summary_href}")
        return result
    entry_elem0 = co.find_first_xpath(
        root,
        "descendant::div[@class='duelParticipant']"
    )
    if entry_elem0 is None:
        if verbose:
            print_err(f"not found entry0 title score in {summary_href}")
        return result
    if is_left_serve():
        result.is_left_srv = True
    elif is_right_serve():
        result.is_left_srv = False
    path = (
        "child::div[@class='duelParticipant__score']/"
        "div[starts-with(@class, 'detailScore__matchInfo')]"
    )
    entry_elem = co.find_first_xpath(entry_elem0, path)
    if entry_elem is None:
        if verbose:
            print_err(f"not found entry title score in {summary_href}")
        return result
    fill_inmatch()
    fill_details()
    return result


def make_pair(item):
    pair = item.split(":")
    return int(pair[0]), int(pair[1])


def make_num_pairs(point_scores_txt):
    txtscore_items = (
        point_scores_txt.strip().replace("40", "45").replace("A", "60").split(", ")
    )
    return [make_pair(i) for i in txtscore_items]


def make_detailed_game(opener_side, game_winside, num_pairs, tiebreak, is_super_tie):
    """opener_side - who serve at usial game, or who begin in tiebreak"""

    def distances(prev_pair, pair, unit=1 if tiebreak else 15):
        left_dist = (pair[0] - prev_pair[0]) // unit
        right_dist = (pair[1] - prev_pair[1]) // unit
        return left_dist, right_dist

    def is_opener_win_point(prev_pair, pair):
        left_dist, right_dist = distances(prev_pair, pair)
        if opener_side.is_left():
            return left_dist > right_dist
        elif opener_side.is_right():
            return right_dist > left_dist
        raise co.TennisScoreError(
            "bad pts seq in make_detailed_game {} {}".format(prev_pair, pair)
        )

    prev_pair = (0, 0)
    points = ""
    for pair in num_pairs:
        points += str(int(is_opener_win_point(prev_pair, pair)))
        prev_pair = pair
    if not tiebreak:
        if opener_side == game_winside:
            points += "1"
        else:
            points += "0"
    return DetailedGame(
        points,
        left_wingame=game_winside.is_left(),
        left_opener=opener_side.is_left(),
        tiebreak=tiebreak,
        supertiebreak=is_super_tie,
    )


def _make_next_key(det_score, setnum, cur_score):
    """makes next key for det_score dict. setnum and cur_score are new data.
    cur_score (is 2-tuple). det_score is prev data."""
    keys = list(det_score.keys())
    if len(keys) == 0:
        return (cur_score,)
    last_key = keys[-1]
    last_setnum = len(last_key)
    if last_setnum == setnum:
        return last_key[0: setnum - 1] + (cur_score,)
    elif (last_setnum + 1) == setnum:
        if det_score[last_key].left_wingame:
            last_set_sc = (last_key[-1][0] + 1, last_key[-1][1])
        else:
            last_set_sc = (last_key[-1][0], last_key[-1][1] + 1)
        return last_key[0: last_setnum - 1] + (last_set_sc,) + (cur_score,)
    else:
        raise co.TennisScoreError(
            "bad sets seq in _make_next_key {} {} in ".format(last_setnum, setnum)
        )


class DetGameResult(NamedTuple):
    x: int
    y: int
    srv_side: Side
    is_lost_srv: bool

    def winner(self):
        return self.srv_side.fliped() if self.is_lost_srv else self.srv_side


def get_tie_info(match, is_decided_set):
    """:return score.TieInfo"""
    if not is_decided_set:
        return sc.default_tie_info
    return sc.get_decided_tiebreak_info(
        match.sex, match.date.year, match.tour_name, match.qualification
    )


def parse_match_detailed_score(match, fsdriver, verbose: bool = False, print_err=print):
    """return True if ok, False if logged error. type(match) == LiveMatch"""
    if not match.href or match.score is None or match.score.retired:
        return
    sets_count = match.score.sets_count()
    best_of_five = match.best_of_five
    det_score = DetailedScore()
    try:
        for setnum in range(1, sets_count + 1):
            det_set_el = build_root(
                match.pointbypoint_href(setnum), fsdriver, verbose=verbose
            )
            fsdriver.implicitly_wait(random.randint(5, 7))
            entry_elem = co.find_first_xpath(
                det_set_el, "descendant::div[contains(@class,'matchHistoryWrapper')]"
            )
            if entry_elem is None:
                if verbose:
                    print_err(f"not found matchHistoryWrapper in set={setnum} {match}")
                return False

            set_scr = match.score[setnum - 1]
            is_decided_set = setnum == (5 if best_of_five else 3)
            tie_inf = get_tie_info(match, is_decided_set)
            is_tie = tie_inf.beg_scr is not None and set_scr in (
                (tie_inf.beg_scr[0] + 1, tie_inf.beg_scr[1]),
                (tie_inf.beg_scr[0], tie_inf.beg_scr[1] + 1),
            )
            mhistos = list(
                entry_elem.xpath("child::div[contains(@class, 'matchHistory')]")
            )
            fifteens = list(entry_elem.xpath("child::div[contains(@class, 'fifteen')]"))
            n_usial_games = sum(set_scr) - 1 if is_tie else sum(set_scr)
            for game_idx in range(min(n_usial_games, len(fifteens))):
                mhisto = mhistos[game_idx]
                fifteen = fifteens[game_idx]
                dg_res = _parse_out_result(mhisto)
                game_winside = dg_res.winner()
                if game_winside.is_left():
                    src_score = (dg_res.x - 1, dg_res.y)
                else:
                    src_score = (dg_res.x, dg_res.y - 1)
                point_scores_txt = _parse_point_scores(fifteen)
                nkey = _make_next_key(det_score, setnum, src_score)
                det_game = make_detailed_game(
                    dg_res.srv_side,
                    game_winside,
                    make_num_pairs(point_scores_txt),
                    tiebreak=False,
                    is_super_tie=False,
                )
                det_score[nkey] = copy.copy(det_game)

            if is_tie:
                mhisto = mhistos[n_usial_games]
                x_sup, y_sup = _parse_detgame_tiesup_score(mhisto)
                at_xy = (n_usial_games // 2, n_usial_games // 2)
                nkey = _make_next_key(det_score, setnum, (min(at_xy), min(at_xy)))
                det_game = _parse_tie_scores(
                    mhistos[n_usial_games + 1:],
                    setnum,
                    (x_sup, y_sup),
                    tie_inf.is_super,
                )
                det_score[nkey] = copy.copy(det_game)
    except co.TennisScoreError as err:
        if verbose:
            print_err("{} in {}".format(err, match))
        return False
    if len(det_score) >= 12:
        match.detailed_score = det_score
        return True


def _parse_detgame_lost_serve_side(parent_elem):
    def is_lost_serve(lostserve_elem):
        span_elem = co.find_first_xpath(
            lostserve_elem, "child::span[contains(text(), 'LOST SERVE')]"
        )
        return span_elem is not None

    lostsrv_elems = list(parent_elem.xpath("child::div[contains(@class,'lostServe')]"))
    if len(lostsrv_elems) == 2:
        if is_lost_serve(lostsrv_elems[0]):
            return co.LEFT
        if is_lost_serve(lostsrv_elems[1]):
            return co.RIGHT
        return None
    else:
        raise co.TennisScoreError(
            "can not find two lostServe elems in _parse_detgame_lost_serve_side"
        )


def _parse_detgame_serve_side(parent_elem):
    def is_serve(serve_elem):
        elem = co.find_first_xpath(
            serve_elem, "child::div[contains(@title, 'Serving player')]"
        )
        return elem is not None

    srv_elems = list(parent_elem.xpath("child::div[contains(@class,'servis')]"))
    if len(srv_elems) == 2:
        if is_serve(srv_elems[0]):
            return co.LEFT
        if is_serve(srv_elems[1]):
            return co.RIGHT
        raise co.TennisScoreError("nobody does servis in _parse_detgame_serve_side")
    else:
        raise co.TennisScoreError(
            "can not find two servis in _parse_detgame_serve_side"
        )


def _parse_detgame_inset_score(parent_elem) -> Tuple[Optional[int], Optional[int]]:
    def get_num(scr_elem):
        txt_num = scr_elem.text
        if txt_num:
            return int(txt_num.strip())

    scr_elems = list(
        parent_elem.xpath(
            "child::div[contains(@class,'scoreBox')]/div[contains(@class,'score')]"
        )
    )
    if len(scr_elems) == 2:
        fst = get_num(scr_elems[0])
        snd = get_num(scr_elems[1])
        return fst, snd
    raise co.TennisScoreError("can not find two scores in _parse_detgame_inset_score")


def _parse_detgame_tiesup_score(parent_elem) -> Tuple[Optional[int], Optional[int]]:
    def get_num(scr_elem):
        txt_num = scr_elem.text
        if txt_num:
            return int(txt_num.strip())

    scr_elems = list(
        parent_elem.xpath(
            "child::div[contains(@class,'scoreBox')]/div[contains(@class,'score')]/sup"
        )
    )
    if len(scr_elems) == 2:
        fst = get_num(scr_elems[0])
        snd = get_num(scr_elems[1])
        return fst, snd
    raise co.TennisScoreError("not found two tiesup in _parse_detgame_tiesup_score")


def _parse_out_result(mhisto_elem):
    srv_side = _parse_detgame_serve_side(mhisto_elem)
    lost_srv_side = _parse_detgame_lost_serve_side(mhisto_elem)
    x, y = _parse_detgame_inset_score(mhisto_elem)
    return DetGameResult(
        x=x, y=y, srv_side=srv_side, is_lost_srv=srv_side == lost_srv_side
    )


def _parse_point_scores(fifteen_elem):
    return (
        fifteen_elem.text_content()
        .replace("MP", "")
        .replace("SP", "")
        .replace("BP", "")
    )


def _parse_tie_scores(mhisto_elems, setnum, fin_score, is_super_tie):
    """fin_score (tie inner sup) must be equal last result_score"""
    num_pairs = []
    opener_side = None
    prev_result_score = None  # for site's errors handle
    for num, mhisto_el in enumerate(mhisto_elems, start=1):
        srvside = _parse_detgame_serve_side(mhisto_el)
        if num == 1:
            opener_side = srvside
        result_score = None
        try:
            pnt_res = _parse_out_result(mhisto_el)
            result_score = pnt_res.x, pnt_res.y
        except co.TennisScoreError as err:
            if prev_result_score is not None and prev_result_score == result_score:
                continue  # not meaning error score, skip it
            else:
                print("{} setnum: {} tiefinscr: {}".format(err, setnum, fin_score))
        num_pairs.append(result_score)
        if num == len(mhisto_elems) and result_score != fin_score:
            raise co.TennisScoreError(
                "tie fin_scr {} != result_scr {} open {}".format(
                    fin_score, result_score, opener_side
                )
            )
        prev_result_score = result_score
    wintie_side = co.LEFT if fin_score[0] > fin_score[1] else co.RIGHT
    return make_detailed_game(
        opener_side, wintie_side, num_pairs, tiebreak=True, is_super_tie=is_super_tie
    )
