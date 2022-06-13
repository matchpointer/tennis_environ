# -*- coding=utf-8 -*-
import pytest

from ratings_elo import get_expected_scores, get_new_elo, k_factor


def get_delta_pts(
    winner_prev_pts,
    winner_n_matches,
    loser_prev_pts,
    loser_n_matches,
    actual_winner_scr=1.0,
    actual_loser_scr=0.0,
):
    winner_expected_prob, loser_expected_prob = get_expected_scores(
        winner_prev_pts, loser_prev_pts
    )
    print(f"actual_winner_scr={actual_winner_scr} expected={winner_expected_prob}")
    winner_new_pts = get_new_elo(
        winner_prev_pts,
        k_factor(winner_n_matches),
        actual_score=actual_winner_scr,
        expected_score=winner_expected_prob,
    )
    loser_new_pts = get_new_elo(
        loser_prev_pts,
        k_factor(loser_n_matches),
        actual_score=actual_loser_scr,
        expected_score=loser_expected_prob,
    )
    return winner_new_pts - winner_prev_pts, loser_new_pts - loser_prev_pts


@pytest.mark.parametrize(
    "actual_winner_scr, winner_prev_pts, winner_n_matches, loser_prev_pts, loser_n_matches",
    [
        (0.76, 1600.0, 150, 1400, 150),
        # (0.7, 1600., 150, 1400, 150),
        # (1.00, 1600., 150, 1400, 150),
    ],
)
def test_delta_pts(
    actual_winner_scr,
    winner_prev_pts,
    winner_n_matches,
    loser_prev_pts,
    loser_n_matches,
):
    delta = get_delta_pts(
        winner_prev_pts=winner_prev_pts,
        winner_n_matches=winner_n_matches,
        loser_prev_pts=loser_prev_pts,
        loser_n_matches=loser_n_matches,
        actual_winner_scr=actual_winner_scr,
        actual_loser_scr=1 - actual_winner_scr,
    )
    print(f"delta={delta}")
    assert delta[0] > 0
    assert delta[1] < 0
