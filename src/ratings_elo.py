# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import datetime
import argparse
from typing import List

from recordclass import recordclass

import file_utils as fu
import cfg_dir
import dba
import log
import common as co
import oncourt_players
import tennis_time as tt
from score import is_dec_supertie_scr
import feature

_surface_keys = ("all", "Clay", "Hard", "Carpet", "Grass")

Cell = recordclass("Cell", ("elo_pts", "elo_alt_pts", "n_matches"))


# (sex, surface_as_text)-> default dict{player_id -> (elo_pts, n_matches)}
# for universal (all surfaces) use surface_as_text='all'
_sex_surf_dict = defaultdict(lambda: defaultdict(lambda: Cell(1500.0, 1500.0, 0)))


# (sex, surface_as_text)-> default dict{player_id -> rank} for actual players only
# for universal (all surfaces) use surface_as_text='all'
_rank_dict = defaultdict(lambda: defaultdict(lambda: None))
_rank_alt_dict = defaultdict(lambda: defaultdict(lambda: None))


def rank_dict(isalt: bool):
    return _rank_alt_dict if isalt else _rank_dict


def initialize(sex=None, date=None):
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        for surface in _surface_keys:
            if date is not None:
                file_date = date
            else:
                file_date = _last_file_date(sex, surface)
                if file_date is None:
                    raise Exception(
                        "not found last file date for {} surf: {}".format(sex, surface)
                    )
            filename = dict_filename(sex, file_date, surface)
            load_json(sex, filename, surface)

    _init_ranks(sex, isalt=False)
    _init_ranks(sex, isalt=True)


def _init_ranks(sex=None, isalt=False):
    rank_dct = rank_dict(isalt)
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        actual_pids = oncourt_players.actual_players_id(sex)
        for surface in _surface_keys:
            if isalt:
                pid_pts_list = [
                    (pid, va)
                    for pid, (v, va, _n) in _sex_surf_dict[(sex, surface)].items()
                    if pid in actual_pids
                ]
            else:
                pid_pts_list = [
                    (pid, v)
                    for pid, (v, va, _n) in _sex_surf_dict[(sex, surface)].items()
                    if pid in actual_pids
                ]
            pid_pts_list.sort(key=lambda i: i[1], reverse=True)
            for rank, (pid, _) in enumerate(pid_pts_list, start=1):
                rank_dct[(sex, surface)][pid] = rank


def make_files(sex, max_date=None, is_actual_players=True):
    import oncourt_db

    clear(sex)
    min_date = (
        oncourt_db.MIN_WTA_TOUR_DATE if sex == "wta" else oncourt_db.MIN_ATP_TOUR_DATE
    )
    if max_date is None:
        max_date = datetime.date.today()
    process(sex, min_date, max_date)
    for surface in _surface_keys:
        filename = dict_filename(sex, max_date, surface)
        dump_json(sex, filename, surface, is_actual_players=is_actual_players)


def make_reper_files(max_date=datetime.date(2009, 12, 27)):
    """utility for initial creating file on concrete date"""
    for sex in ("wta", "atp"):
        clear(sex)
        make_files(sex=sex, max_date=max_date, is_actual_players=False)


def remove_files(sex, date):
    for surface in _surface_keys:
        filename = dict_filename(sex, date, surface)
        fu.remove_file(filename)


def alt_win_result(match_score):
    n_sets = match_score.sets_count()
    is_dec = n_sets in (5, 3)
    if is_dec:
        dscr = match_score[n_sets - 1]
        is_dec_tie = (
            dscr
            in (
                (7, 6),
                (6, 7),
                (13, 12),
                (12, 13),
            )
            or is_dec_supertie_scr(dscr, retired=False, unique_mode=True)
        )
    else:
        is_dec_tie = False
    win_scr = 1.0
    if is_dec:
        win_scr = 0.8 if is_dec_tie else 0.9
    return win_scr


def put_match_data(sex: str, tour, rnd, match, is_pandemia_tour=False):
    """assumed: matches incomes in time ascending order"""

    def new_elo_pts_pair(win_cell, lose_cell):
        winner_pts, winner_n_matches = win_cell.elo_pts, win_cell.n_matches
        loser_pts, loser_n_matches = lose_cell.elo_pts, lose_cell.n_matches
        winner_exp_scr, loser_exp_scr = get_expected_scores(winner_pts, loser_pts)
        winner_new_elo = get_new_elo(
            winner_pts,
            k_factor(winner_n_matches) * k_coef,
            actual_score=1.0,
            expected_score=winner_exp_scr,
        )
        loser_new_elo = get_new_elo(
            loser_pts,
            k_factor(loser_n_matches) * k_coef,
            actual_score=0.0,
            expected_score=loser_exp_scr,
        )
        return winner_new_elo, loser_new_elo

    def new_elo_alt_pts_pair(win_cell, lose_cell):
        win_scr = alt_win_result(match.score)
        winner_pts, winner_n_matches = win_cell.elo_alt_pts, win_cell.n_matches
        loser_pts, loser_n_matches = lose_cell.elo_alt_pts, lose_cell.n_matches
        winner_exp_scr, loser_exp_scr = get_expected_scores(winner_pts, loser_pts)
        winner_new_elo = get_new_elo(
            winner_pts,
            k_factor(winner_n_matches),
            actual_score=win_scr,
            expected_score=winner_exp_scr,
        )
        loser_new_elo = get_new_elo(
            loser_pts,
            k_factor(loser_n_matches),
            actual_score=1.0 - win_scr,
            expected_score=loser_exp_scr,
        )
        return winner_new_elo, loser_new_elo

    def edit(dictionary):
        win_cell = dictionary[winner_id]
        lose_cell = dictionary[loser_id]
        winner_new_elo, loser_new_elo = new_elo_pts_pair(win_cell, lose_cell)
        winner_new_elo_alt, loser_new_elo_alt = new_elo_alt_pts_pair(
            win_cell, lose_cell
        )
        dictionary[winner_id] = Cell(
            elo_pts=winner_new_elo,
            elo_alt_pts=winner_new_elo_alt,
            n_matches=win_cell.n_matches + 1,
        )
        dictionary[loser_id] = Cell(
            elo_pts=loser_new_elo,
            elo_alt_pts=loser_new_elo_alt,
            n_matches=lose_cell.n_matches + 1,
        )

    if is_pandemia_tour:
        return
    winner_id, loser_id = match.first_player.ident, match.second_player.ident
    k_coef = 0.8 if is_pandemia_tour else 1.0  # 1.1 if is_best_of_five() else 1.0
    surface = str(tour.surface)

    edit(_sex_surf_dict[(sex, "all")])
    if surface in _surface_keys:
        edit(_sex_surf_dict[(sex, surface)])


def put_match(sex: str, tour, rnd, match):
    from tournament import PandemiaTours

    """ assumed: matches incomes in time ascending order """
    if (
        tour.level in ("junior",)
        or (match.score and match.score.retired)
        or rnd == "Pre-q"
    ):
        return
    if tour.level == "future" and (rnd == "First" or rnd.qualification()):
        return
    put_match_data(
        sex,
        tour,
        rnd,
        match,
        is_pandemia_tour=PandemiaTours.is_pandemia_tour(sex, tour.ident),
    )


def get_pts(sex: str, player_id: int, surface="all", isalt=False):
    if isalt:
        return _sex_surf_dict[(sex, str(surface))][player_id].elo_alt_pts
    else:
        return _sex_surf_dict[(sex, str(surface))][player_id].elo_pts


def get_rank(sex: str, player_id: int, surface="all", isalt=False):
    rank_dct = rank_dict(isalt)
    return rank_dct[(sex, str(surface))][player_id]


def get_matches_played(sex: str, player_id: int, surface="all"):
    return _sex_surf_dict[(sex, str(surface))][player_id].n_matches


def add_features(
    sex: str, surface, match, features: List[feature.Feature], isalt=False
):
    fst_pts = get_pts(sex, match.first_player.ident, isalt=isalt)
    snd_pts = get_pts(sex, match.second_player.ident, isalt=isalt)
    core = "elo_alt" if isalt else "elo"
    f1, f2 = feature.make_pair(
        f"fst_{core}_pts", f"snd_{core}_pts", fst_value=fst_pts, snd_value=snd_pts
    )
    features.append(f1)
    features.append(f2)

    if surface in ("Clay", "Hard", "Carpet", "Grass"):
        fst_surf_pts = get_pts(sex, match.first_player.ident, surface, isalt=isalt)
        snd_surf_pts = get_pts(sex, match.second_player.ident, surface, isalt=isalt)
        fs1, fs2 = feature.make_pair(
            f"fst_surf_{core}_pts",
            f"snd_surf_{core}_pts",
            fst_value=fst_surf_pts,
            snd_value=snd_surf_pts,
        )
        features.append(fs1)
        features.append(fs2)
    else:
        raise Exception("empty/none surface in {} {}".format(sex, match))


def clear(sex: str):
    """convenient for debugs/tests/experiments"""
    for (k1, k2) in _sex_surf_dict.keys():
        if k1 == sex:
            _sex_surf_dict[(k1, k2)].clear()


def get_top_ordered(sex: str, surface="all", top=100, isalt=False):
    """:returns [(player_id, elo_points, n_matches_played)]"""
    act_plr_ids = oncourt_players.actual_players_id(sex)
    if isalt:
        lst = [
            (pid, alt_pts, n_m)
            for pid, (pts, alt_pts, n_m) in _sex_surf_dict[(sex, str(surface))].items()
            if pid in act_plr_ids
        ]
    else:
        lst = [
            (pid, pts, n_m)
            for pid, (pts, alt_pts, n_m) in _sex_surf_dict[(sex, str(surface))].items()
            if pid in act_plr_ids
        ]
    lst.sort(key=lambda i: i[1], reverse=True)
    return lst[:top]


def dict_filename(sex: str, date: datetime.date, surface="all"):
    date_txt = "%04d.%02d.%02d" % (date.year, date.month, date.day)
    return "{}/{}_{}_elo_{}.json".format(
        cfg_dir.ratings_dir(), sex, str(surface), date_txt
    )


def _last_file_date(sex: str, surface="all"):
    def filename_mask():
        return "{}_{}_elo_*.json".format(sex, str(surface))

    dates = []
    for filename in fu.find_files_in_folder(
        cfg_dir.ratings_dir(), filemask=filename_mask(), recursive=False
    ):
        if len(filename) >= 15:
            end_idx = -len(".json")
            beg_idx = end_idx - len("yyyy.mm.dd")
            yyyy_mm_dd = filename[beg_idx:end_idx]
            year = int(yyyy_mm_dd[0:4])
            month = int(yyyy_mm_dd[5:7])
            day = int(yyyy_mm_dd[8:10])
            dates.append(datetime.date(year, month, day))
    if dates:
        return max(dates)


def dump_json(sex: str, filename: str, surface="all", is_actual_players=False):
    indct = _sex_surf_dict[(sex, str(surface))]
    if is_actual_players:
        act_pids = oncourt_players.actual_players_id(sex)
        dct = {
            k: (v.elo_pts, v.elo_alt_pts, v.n_matches)
            for k, v in indct.items()
            if k in act_pids
        }
    else:
        dct = {k: (v.elo_pts, v.elo_alt_pts, v.n_matches) for k, v in indct.items()}
    fu.json_dump(dct, filename, indent=None)


def load_json(sex: str, filename: str, surface="all"):
    if not os.path.isfile(filename):
        raise Exception("not found json file {}".format(filename))
    dct = fu.json_load_intkey_dict(filename)
    for key, (pts, alt_pts, nm) in dct.items():
        _sex_surf_dict[(sex, str(surface))][key] = Cell(
            elo_pts=pts, elo_alt_pts=alt_pts, n_matches=nm
        )


def process(sex: str, min_date, max_date):
    from tournament import tours_generator

    if not dba.initialized():
        dba.open_connect()
    if max_date.isoweekday() == 1:
        # for including qualification matches with match.date < max_date
        max_tour_date = max_date + datetime.timedelta(days=7)
    else:
        max_tour_date = max_date
    for tour in tours_generator(sex, min_date=min_date, max_date=max_tour_date):
        for match, rnd in tour.match_rnd_list():
            if match.date and match.date > max_date:
                continue
            if match.date is None and tour.date >= tt.past_monday_date(max_date):
                continue
            put_match(sex, tour, rnd, match)


K_BASE = 250.0
OFFSET = 5
SHAPE = 0.4


# www.betfair.com.au/hub/tennis-elo-modelling/
# fivethirtyeight.com/features/
# serena-williams-and-the-difference-between-all-time-great-and-greatest-of-all-time/
def k_factor(n_matches):
    """K-factor is the maximum Elo update for any given match"""
    return K_BASE / ((n_matches + OFFSET) ** SHAPE)


def get_expected_scores(player_rating, opponent_rating):
    """classical elo estimate of win prob (also for other sports)"""
    plr_scr = 1.0 / (1.0 + (10 ** ((opponent_rating - player_rating) / 400.0)))
    opp_scr = 1.0 - plr_scr
    return plr_scr, opp_scr


def get_new_elo(old_elo, k_fact, actual_score, expected_score):
    return old_elo + float(k_fact) * (actual_score - expected_score)


def bo5_win_proba(bo3_win_proba):
    """from https://github.com/JeffSackmann/tennis_misc/blob/master/fiveSetProb.py"""
    from numpy import roots

    p1 = roots([-2, 3, 0, -1 * bo3_win_proba])[1]
    p5 = (p1 ** 3) * (4 - 3 * p1 + (6 * (1 - p1) * (1 - p1)))
    return p5


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp", "both"])
    parser.add_argument("--make_files", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    log.info("started with {} make_files: {}".format(args.sex, args.make_files))
    if args.make_files:
        for sex in ("wta", "atp"):
            if sex == args.sex or args.sex == "both":
                log.info("starting {}".format(sex))
                make_files(sex)
                log.info("done {}".format(sex))
