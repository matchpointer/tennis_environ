"""
module gives service of classification (match_has_min_proba)
for already learned and saved to file model.
Positive predict is means: decided set opener will lose match
"""
import copy
import datetime
from typing import List, Optional


import numpy as np

from clf_decided_00_variants import (
    var_atp_main_clr_1f,
    var_wta_main_clr_rtg350_250_1
)
from side import Side
import log
import common as co
import clf_common as cco
import decided_win_by_two_sets_stat
import predicts_db

MODEL = "decided_00"
MIN_QUAD_SIZE = 14  # for one player per all history
MIN_HIST_DATA_SIZE = 5

apply_variants = [
    var_wta_main_clr_rtg350_250_1,
    var_atp_main_clr_1f,
]


def load_variants(variants: Optional[List[cco.Variant]] = None,
                  metric_name: Optional[str] = "Accuracy"):
    for variant in (apply_variants if variants is None else variants):
        variant.load_clf(MODEL, metric_name=metric_name)
        if variant.clf is None:
            raise co.TennisError(
                f"fail cb model load {variant.sex} key {variant.key} "
                f"metric_name {metric_name}"
            )


def skip_cond(match):
    if match.rnd == 'Final':
        return True
    if match.sex == "wta":
        return (
            match.level in ("teamworld", "team", "chal")
            or not match.is_ranks_both_above(350, rtg_name="std")
            or match.is_ranks_dif_wide(max_dif=250, rtg_name="std")
            or (
                match.level != "masters"
                and match.rnd is not None
                and match.rnd.qualification()
            )
        )
    elif match.sex == "atp":
        return (
            match.best_of_five
            or match.soft_level not in ("main",)
            or
            (
                match.level != "masters"
                and match.rnd is not None
                and match.rnd.qualification()
            )
        )


def match_has_min_proba(match, decset_open_side: Side):
    def log_estimate():
        if positive_prob > negative_prob:
            max_prob = round(positive_prob, 3)
            max_prob_plr = (
                match.second_player
                if decset_open_side.is_left()
                else match.first_player
            )
            back_beg_txt = "CLOSER"
        else:
            max_prob = round(negative_prob, 3)
            max_prob_plr = (
                match.first_player
                if decset_open_side.is_left()
                else match.second_player
            )
            back_beg_txt = "OPENER"
        log.info(
            f"clf {MODEL} try {match} \nopen_side:{decset_open_side} "
            f"PROB: {max_prob} {back_beg_txt} PLR {max_prob_plr}"
        )
        predicts_db.write_predict(
            match, case_name='decided_00',
            back_side=co.LEFT if max_prob_plr is match.first_player else co.RIGHT,
            proba=max_prob, comments=back_beg_txt)

    try:
        variant = cco.Variant.find_match_variant(match, apply_variants)
        if variant is None:
            log.error("clf {} novarianted for: {}".format(MODEL, match.tostring()))
            return None, None
        X_test = decset_match_features(match, decset_open_side, variant)
        clf = variant.clf
        proba = clf.predict_proba(X_test)
        positive_prob = float(proba[0, 1])
        negative_prob = float(proba[0, 0])
        log_estimate()
        if positive_prob >= variant.min_probas.pos:
            return decset_open_side.fliped(), positive_prob
        elif negative_prob >= variant.min_probas.neg:
            return decset_open_side, negative_prob
    except cco.FeatureError as err:
        log.warn("clf {} prep err: {} {}".format(MODEL, err, match.tostring()))
        return None, None
    return None, None


def decset_match_features(match, decset_open_side, variant: cco.Variant):
    def opener_s2choke_adv():
        s2loser = Side(match.score[1][0] < match.score[1][1])
        if s2loser.is_left():
            was_breakup = match.quad_stat.breakup_tracker.is_fst_breakup(setnum=2)
        else:
            was_breakup = match.quad_stat.breakup_tracker.is_snd_breakup(setnum=2)
        if not was_breakup:
            return 0
        return -1 if decset_open_side == s2loser else 1

    fst_bet_chance = match.first_player_bet_chance()
    if fst_bet_chance is None:
        raise cco.FeatureError("none bet_chance")
    age1 = match.first_player.age(datetime.date.today())
    age2 = match.second_player.age(datetime.date.today())
    if not match.check_hist_min_size(MIN_HIST_DATA_SIZE):
        raise cco.FeatureError("poor hist_srv-rcv size < {}".format(MIN_HIST_DATA_SIZE))
    if match.quad_stat.game_tense_stat.is_match_begin_empty():
        raise cco.FeatureError("not full data (begin_empty)")
    h2h_direct = 0.0 if match.h2h_direct is None else match.h2h_direct
    elo_pts_dif = match.elo_pts_dif_mixed(gen_part=1.0, isalt=True)
    elo_pts_surf_dif = match.elo_pts_dif_mixed(gen_part=0.0, isalt=True)
    if decset_open_side.is_left():
        features = match.features
    else:
        fst_bet_chance = 1.0 - fst_bet_chance
        age1, age2 = age2, age1
        h2h_direct = -h2h_direct
        elo_pts_dif = -elo_pts_dif
        elo_pts_surf_dif = -elo_pts_surf_dif
        features = copy.deepcopy(match.features)
        for feat in features:
            feat.flip_side()

    fst_deu_wl, fst_adv_wl, snd_deu_wl, snd_adv_wl = match.quad_stat.get_all()
    if (
        fst_deu_wl.size < MIN_QUAD_SIZE
        or fst_adv_wl.size < MIN_QUAD_SIZE
        or snd_deu_wl.size < MIN_QUAD_SIZE
        or snd_adv_wl.size < MIN_QUAD_SIZE
    ):
        raise cco.FeatureError("poor quad_stat size < {}".format(MIN_QUAD_SIZE))

    fst_srv_p = (fst_adv_wl + fst_deu_wl).ratio
    snd_srv_p = (snd_adv_wl + snd_deu_wl).ratio
    if decset_open_side.is_right():
        fst_srv_p, snd_srv_p = snd_srv_p, fst_srv_p
    dif_srv_p = snd_srv_p - fst_srv_p

    lastinrow = match.quad_stat.lastinrow.fst_count(setnum=2)
    if lastinrow in (0, None):
        raise cco.FeatureError("bad lastinrow: {}".format(lastinrow))

    closer_dec_win_val = decided_win_by_two_sets_stat.read_decided_win_ratio(
        match, side=decset_open_side.fliped()
    )
    if closer_dec_win_val is None:
        raise cco.FeatureError(
            "none closer_dec_win_val: {} {}".format(match.sex, match.soft_level)
        )

    soft_level = match.soft_level
    if soft_level not in ("main", "chal", "qual"):
        raise cco.FeatureError("other soft level: {}".format(soft_level))
    feat_dct = {
        "rnd_text": cco.extract_rnd_text(match.rnd),
        "surface_text": str(match.surface),
        "dif_age": age2 - age1,
        "avg_age": (age2 + age1) * 0.5,
        "level_text": soft_level,
        "h2h_direct": h2h_direct,
        "dif_srv_ratio": dif_srv_p,
        # here some features temporary droped
    }
    basevals = [feat_dct[fn] for fn in variant.feature_names.get_list()]
    return np.array([basevals])