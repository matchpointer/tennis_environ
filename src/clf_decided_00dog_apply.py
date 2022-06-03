"""
module gives service of classification (match_has_min_proba)
for already learned and saved to file model.
Positive predict is means: decided set opener will lose match
"""
import datetime
import copy

import numpy as np

from clf_decided_00dog_variants import (
    var_wta_main_clr_rtg550_350_1Npm04,
    var_atp_main_clr_rtg500_300_1ffrs,
)
from side import Side
import surf
from loguru import logger as log
import clf_common as cco
import feature
import predicts_db


MIN_QUAD_SIZE = 14  # for one player per all history
MIN_HIST_DATA_SIZE = 5


apply_variants = [
    var_wta_main_clr_rtg550_350_1Npm04,
    var_atp_main_clr_rtg500_300_1ffrs,
]


def skip_cond(match):
    if match.rnd is None:
        return True
    if match.sex == 'wta':
        if match.level == 'chal':
            return (
                match.rnd in ('Final',)
                or not match.is_ranks_both_above(550, rtg_name='std')
                or match.is_ranks_dif_wide(max_dif=350, rtg_name='std')
            )
        elif match.level in ('main', 'gs', 'masters'):
            return (
                # match.rnd == 'Robin'
                match.rnd in ('Final',)
                or 'Rubber' in str(match.rnd)
                or not match.is_ranks_both_above(550, rtg_name='std')
                or match.is_ranks_dif_wide(max_dif=350, rtg_name='std')
                or (match.rnd == 'Qualifying' and match.level == 'main')
                or (match.rnd in ('q-First', 'q-Second') and match.level != 'masters')
            )
        else:
            return True  # not our level
    elif match.sex == 'atp':
        if match.level == 'chal':
            return (
                match.rnd in ('Final',)
                or not match.is_ranks_both_above(500, rtg_name='std')
                or match.is_ranks_dif_wide(max_dif=300, rtg_name='std')
            )
        if match.level in ('main', 'gs', 'masters'):
            return (
                match.best_of_five
                # or match.rnd == 'Robin'
                or match.rnd in ('Final',)
                or 'Rubber' in str(match.rnd)
                or not match.is_ranks_both_above(500, rtg_name='std')
                or match.is_ranks_dif_wide(max_dif=300, rtg_name='std')
                or (match.rnd == 'Qualifying' and match.level == 'main')
                or (match.rnd in ('q-First', 'q-Second') and match.level != 'masters')
            )
        else:
            return True  # not our level


def match_has_min_proba(match, decset_open_side: Side):
    """ return back_side, prob, back_dog: bool, weak_fav: bool """
    def log_thresholded(proba):
        comment = "backdog" if back_dog else "backfav"
        log.info(
            f"clf {variant.name} try {match} \nback_side:{back_side} "
            f"PROB: {round(proba, 3)} thresholded {comment}")
        predicts_db.write_predict(
            match,
            case_name='decided_00dog',
            back_side=back_side,
            proba=proba,
            comments=comment,
            clf_hash=variant.name,
        )

    def log_not_thresholded():
        proba = negative_prob
        if proba < 0.5:
            log.info(
                f"clf {variant.name} try {match} \n "
                f"strange neg PROB: {round(proba, 3)} not thresholded")
            return
        fav_side = Side(match.first_player_bet_chance() >= 0.5)
        if abs(variant.min_probas.neg - 1) < 0.001:
            comment = "fav commented"
        else:
            comment = "fav not thresholded"
        log.info(
            f"clf {variant.name} try {match} \nfav_side:{fav_side} "
            f"PROB: {round(proba, 3)} {comment}")
        predicts_db.write_predict(
            match,
            case_name='decided_00dog',
            back_side=fav_side,
            proba=proba,
            comments=comment,
            clf_hash=variant.name,
        )

    empty_result = None, None, None
    try:
        variant = cco.Variant.find_match_variant(match, apply_variants)
        if variant is None:
            log.error("clf {} novarianted for: {}".format(__file__, match.tostring()))
            return empty_result
        X_test = decset_match_features(match, decset_open_side, variant)
        clf = variant.clf
        probas = clf.predict_proba(X_test)
        positive_prob = float(probas[0, 1])
        negative_prob = float(probas[0, 0])
        if positive_prob >= variant.min_probas.pos:
            # should to back dog
            back_side = Side(match.first_player_bet_chance() < 0.5)
            back_dog = True
            log_thresholded(proba=positive_prob)
            return back_side, positive_prob, back_dog
        elif negative_prob >= variant.min_probas.neg:
            # should to back fav
            back_side = Side(match.first_player_bet_chance() > 0.5)
            back_dog = False
            log_thresholded(proba=negative_prob)
            return back_side, negative_prob, back_dog
        else:
            log_not_thresholded()
    except cco.FeatureError as err:
        log.warning("clf {} prep err: {} {}".format(__file__, err, match.tostring()))
        return empty_result
    return empty_result


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

    if match.quad_stat.startgame_points_count == 0:
        raise cco.FeatureError("not full data (begin_empty)")
    fst_bet_chance = match.first_player_bet_chance()
    if fst_bet_chance is None:
        raise cco.FeatureError("none bet_chance")
    age1 = match.first_player.age(datetime.date.today())
    age2 = match.second_player.age(datetime.date.today())
    if not match.check_hist_min_size(MIN_HIST_DATA_SIZE):
        raise cco.FeatureError("poor hist_srv-rcv size < {}".format(MIN_HIST_DATA_SIZE))
    h2h_direct = 0.0 if match.h2h_direct is None else match.h2h_direct
    elo_pts_dif = match.elo_pts_dif_mixed(gen_part=1.0, isalt=True)
    elo_pts_surf_dif = match.elo_pts_dif_mixed(gen_part=0.0, isalt=True)
    if decset_open_side.is_left():
        fst_srv_win = match.hist_fst_srv_win.value
        fst_rcv_win = match.hist_fst_rcv_win.value
        snd_srv_win = match.hist_snd_srv_win.value
        snd_rcv_win = match.hist_snd_rcv_win.value
        features = match.features
    else:
        fst_bet_chance = 1.0 - fst_bet_chance
        age1, age2 = age2, age1
        h2h_direct = -h2h_direct
        elo_pts_dif = -elo_pts_dif
        elo_pts_surf_dif = -elo_pts_surf_dif
        fst_srv_win = match.hist_snd_srv_win.value
        fst_rcv_win = match.hist_snd_rcv_win.value
        snd_srv_win = match.hist_fst_srv_win.value
        snd_rcv_win = match.hist_fst_rcv_win.value
        features = copy.deepcopy(match.features)
        for feat in features:
            feat.flip_side()

    feat_dct = {
        "dif_elo_alt_pts": elo_pts_dif,
        "dif_service_win": snd_srv_win - fst_srv_win,
        "dif_receive_win": snd_rcv_win - fst_rcv_win,
        "rnd_text": cco.extract_rnd_text(match.rnd),
        "surface_text": str(match.surface),
        "dif_fatigue": feature.dif_player_features(features, "fatigue"),
        "rnd_stage": cco.rnd_stage(cco.extract_rnd_text(match.rnd), str(match.level)),
        "surface": surf.get_code(match.surface),
    }
    basevals = [feat_dct[fn] for fn in variant.feature_names.get_list()]
    if variant.is_cb_native() and variant.exist_cat():
        return [basevals]
    return np.array([basevals])
