import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import pprint

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from side import Side
import cfg_dir
import log
import common as co
import clf_common as cco
import stat_cont as st
import clf_cat_tools
import clf_columns

"""
initialy in openset_data.csv data prepared so that fst player is set1 opener,
We want data preparing (orienting) so that fst player is set1 loser,
    and after that orienting label is about set1_loser_win_set2 
"""

MODEL = "secondset_00"
LABEL_NAME = "s2_fst_win"

DEFAULT_RESERVE_SIZE = 0.10
RESERVE_RANDOM_STATE = 466
DEFAULT_EVAL_SIZE = 0.17
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CLASSIFIER = CatBoostClassifier


def initialize(variants=None):
    cco.set_seed(random_seed)

    for variant in work_variants if variants is None else variants:
        variant.load_clf(MODEL, metric_name=metric_name)
        if variant.clf is None:
            raise co.TennisError(
                f"fail cb model load {variant.sex} key {variant.key} "
                f"metric_name {metric_name}"
            )


def skip_cond(match):
    if match.sex == "wta":
        return (
            match.level not in ("gs", "masters", "main")
            or (
                match.level in ("gs", "main")
                and match.rnd is not None
                and match.rnd.qualification()
            )
            or not match.is_ranks_both_above(200, rtg_name="std")
            or match.is_ranks_dif_wide(max_dif=150, rtg_name="std")
        )
    elif match.sex == "atp":
        return (
            match.best_of_five
            or match.level not in ("gs", "masters", "main")
            or (
                match.level in ("gs", "main")
                and match.rnd is not None
                and match.rnd.qualification()
            )
            or not match.is_ranks_both_above(200, rtg_name="std")
            or match.is_ranks_dif_wide(max_dif=150, rtg_name="std")
        )


default_feat_names = cco.FeatureNames(
    names=[
        "fst_bet_chance",
        "dif_srv_ratio",
        "s1_dif_games",
        "dif_mix_elo_pts_055_045",
        "snd_set2win_after_set1win_minus_fst_set2win_after_set1loss",
        "snd_choke_plus_fst_trail",
        "dif_service_win_plus_dif_receive_win",
        "dif_age",
        "avg_age",
        "dif_plr_tour_adapt",
        "dif_fatigue",
        "h2h_direct",
        # 'dif_prevyear_tour_rnd',  # removed, not actual in 2020?
        # 'rnd_text',
        # 'surface_text',
        # 'level_text',
    ]
)

default_wta_feat_names = default_feat_names

default_atp_feat_names = default_feat_names


metric_name = "Precision"  # 'AUC'  # 'Accuracy'
random_seed = 44
early_stopping_rounds = None


def add_columns(sex, tbl):
    if sex == "wta":
        clf_columns.edit_column_value(tbl, "tour_money", lambda m: m / 100000)
    clf_columns.add_column(tbl, "rnd", lambda r: cco.rnd_code(r["rnd_text"]))
    clf_columns.add_column(tbl, "level", lambda r: cco.level_code(r["level_text"]))
    clf_columns.add_column(
        tbl, "surface", lambda r: cco.surface_code(r["surface_text"])
    )

    tbl["dif_srv_ratio"] = tbl["s1_snd_srv_ratio"] - tbl["s1_fst_srv_ratio"]
    tbl["s1_dif_games"] = tbl["s1_snd_games"] - tbl["s1_fst_games"]

    tbl["dif_service_win"] = tbl["snd_service_win"] - tbl["fst_service_win"]
    tbl["dif_receive_win"] = tbl["snd_receive_win"] - tbl["fst_receive_win"]

    tbl["dif_elo_pts"] = tbl["snd_elo_pts"] - tbl["fst_elo_pts"]
    tbl["dif_surf_elo_pts"] = tbl["snd_surf_elo_pts"] - tbl["fst_surf_elo_pts"]

    tbl["dif_elo_alt_pts"] = tbl["snd_elo_alt_pts"] - tbl["fst_elo_alt_pts"]
    tbl["dif_surf_elo_alt_pts"] = (
        tbl["snd_surf_elo_alt_pts"] - tbl["fst_surf_elo_alt_pts"]
    )


def make_orientation(sex, tbl):
    """after call fst, snd orient: fst is loser first set"""

    def flip_bool(idx, name):
        tbl.at[idx, name] = 1 - tbl.at[idx, name]

    def flip_sign(idx, name):
        tbl.at[idx, name] = -tbl.at[idx, name]

    def flip_row_chance(idx):
        tbl.at[idx, "fst_bet_chance"] = 1.0 - tbl.at[idx, "fst_bet_chance"]

    def flip_fst_snd_row(idx, suffix):
        tmp = tbl.at[idx, "fst_" + suffix]
        tbl.at[idx, "fst_" + suffix] = tbl.at[idx, "snd_" + suffix]
        tbl.at[idx, "snd_" + suffix] = tmp

    def flip_row_by_names(idx, fst_name, snd_name):
        tmp = tbl.at[idx, fst_name]
        tbl.at[idx, fst_name] = tbl.at[idx, snd_name]
        tbl.at[idx, snd_name] = tmp

    for i in tbl.index:
        s1_fst_win = tbl.at[i, "s1_fst_games"] > tbl.at[i, "s1_snd_games"]
        flip_need = s1_fst_win
        if not flip_need:
            continue
        flip_bool(i, "s2_fst_opener")  # already created before this call
        flip_bool(i, "fst_win_match")
        flip_row_chance(i)
        flip_sign(i, "h2h_direct")
        flip_fst_snd_row(i, suffix="pid")
        flip_fst_snd_row(i, suffix="elo_pts")
        flip_fst_snd_row(i, suffix="surf_elo_pts")
        flip_fst_snd_row(i, suffix="elo_alt_pts")
        flip_fst_snd_row(i, suffix="surf_elo_alt_pts")
        flip_fst_snd_row(i, suffix="service_win")
        flip_fst_snd_row(i, suffix="receive_win")
        flip_row_by_names(i, fst_name="s1_fst_games", snd_name="s1_snd_games")
        flip_row_by_names(i, fst_name="s2_fst_games", snd_name="s2_snd_games")
        flip_row_by_names(i, fst_name="s1_fst_srv_ratio", snd_name="s1_snd_srv_ratio")


def make_secondset_opener(data):
    def process_row(row):
        if co.is_odd(row["s1_fst_games"] + row["s1_snd_games"]):
            return 0
        return 1

    data["s2_fst_opener"] = data.apply(process_row, axis=1)


def drop_nan_labels(sex, data):
    todrop = [
        "s1_fst_games",
        "s1_snd_games",
        "s1_absent_games",
        "fst_bet_chance",
        "fst_service_win",
        "snd_service_win",
        "fst_receive_win",
        "snd_receive_win",
    ]
    if sex == "wta":
        todrop.append("tour_money")
    data.dropna(subset=todrop, inplace=True)


def match_has_min_proba(match, set_open_side: Side):
    def log_estimate():
        if positive_prob >= 0.5:
            max_prob = positive_prob
            max_prob_plr_name = (
                match.second_player if win_set1_side.is_left() else match.first_player
            )
            log.info(
                f"clf {MODEL} try {match} \nopen_side:{set_open_side} "
                f"PROB: {round(max_prob, 3)} PLR {max_prob_plr_name}"
            )

    try:
        variant = cco.Variant.find_match_variant(match, work_variants)
        if variant is None:
            log.error("clf {} novarianted for: {}".format(MODEL, match.tostring()))
            return None, None
        X_test, win_set1_side = get_match_features(match, variant)
        clf = variant.clf
        proba = clf.predict_proba(X_test)
        positive_prob = float(proba[0, 1])
        log_estimate()
        if positive_prob >= variant.min_proba:
            return win_set1_side.fliped(), positive_prob
    except cco.FeatureError as err:
        log.warn("clf {} prep err: {} {}".format(MODEL, err, match.tostring()))
        return None, None
    return None, None


# for one player per one set in history.
MIN_SET_SIZE = 11  # if 6-0 we have min = 3 games * 4 points
MIN_HIST_DATA_SIZE = 5


def get_match_features(match, variant: cco.Variant):
    def s1choke_adv():
        s1loser = win_set1_side.fliped()
        if s1loser.is_left():
            was_breakup = match.quad_stat.breakup_tracker.is_fst_breakup(setnum=1)
        else:
            was_breakup = match.quad_stat.breakup_tracker.is_snd_breakup(setnum=1)
        return -1 if was_breakup else 0

    def get_dif_srv_ratio():
        fst_deu_wl, fst_adv_wl, snd_deu_wl, snd_adv_wl = match.quad_stat.get_all()
        fst_wl = fst_adv_wl + fst_deu_wl
        snd_wl = snd_adv_wl + snd_deu_wl
        if fst_wl.size < MIN_SET_SIZE or snd_wl.size < MIN_SET_SIZE:
            raise cco.FeatureError(
                f"poor srv size < {MIN_SET_SIZE} "
                f"fst {fst_wl.size} snd {snd_wl.size}"
            )
        fst_srv_p = (fst_adv_wl + fst_deu_wl).ratio
        snd_srv_p = (snd_adv_wl + snd_deu_wl).ratio
        if is_flip_need:
            fst_srv_p, snd_srv_p = snd_srv_p, fst_srv_p
        return snd_srv_p - fst_srv_p

    win_set1_side = co.side(match.score[0][0] > match.score[0][1])
    fst_bet_chance = match.first_player_bet_chance()
    if fst_bet_chance is None:
        raise cco.FeatureError("none bet_chance")
    if not match.check_hist_min_size(min_size=MIN_HIST_DATA_SIZE):
        raise cco.FeatureError("poor hist_srv-rcv size < 5 or empty")
    elo_pts_dif = match.elo_pts_dif_mixed(gen_part=1.0, isalt=True)
    elo_pts_surf_dif = match.elo_pts_dif_mixed(gen_part=0.0, isalt=True)
    is_flip_need = win_set1_side == co.LEFT
    if not is_flip_need:
        dif_set1_games = match.score[0][1] - match.score[0][0]
        fst_srv_win = match.hist_fst_srv_win.value
        fst_rcv_win = match.hist_fst_rcv_win.value
        snd_srv_win = match.hist_snd_srv_win.value
        snd_rcv_win = match.hist_snd_rcv_win.value
    else:
        dif_set1_games = match.score[0][0] - match.score[0][1]
        fst_bet_chance = 1.0 - fst_bet_chance
        elo_pts_dif = -elo_pts_dif
        elo_pts_surf_dif = -elo_pts_surf_dif
        fst_srv_win = match.hist_snd_srv_win.value
        fst_rcv_win = match.hist_snd_rcv_win.value
        snd_srv_win = match.hist_fst_srv_win.value
        snd_rcv_win = match.hist_fst_rcv_win.value

    feat_dct = {
        "dif_elo_alt_pts": elo_pts_dif,
        "dif_surf_elo_alt_pts": elo_pts_surf_dif,
        "dif_service_win": snd_srv_win - fst_srv_win,
        "dif_receive_win": snd_rcv_win - fst_rcv_win,
        "dif_srv_ratio": get_dif_srv_ratio(),
        "fst_bet_chance": fst_bet_chance,
        "s1_dif_games": dif_set1_games,
    }
    basevals = [feat_dct[n] for n in variant.feature_names.get_list()]
    return np.array([basevals]), win_set1_side


def drop_by_condition(variant: cco.Variant, df: pd.DataFrame) -> None:
    cco.drop_by_condition(
        df,
        lambda d: (
            (d["s1_absent_games"] >= 1)
            | (d["s1_simul_games"] >= 1)
            | (d["s1_empty_games"] >= 1)
            | (d["best_of_five"] == 1)
        ),
    )
    variant.drop_by_condition(df)


def primary_edit(variant: cco.Variant, df: pd.DataFrame):
    drop_nan_labels(variant.sex, df)
    cco.drop_rank_std_any_below_or_wide_dif(
        df, cco.RANK_STD_BOTH_ABOVE, cco.RANK_STD_MAX_DIF
    )
    clf_columns.edit_column_value(df, "rnd_text", cco.unified_rnd_value)
    drop_by_condition(variant, df)
    make_secondset_opener(df)
    clf_columns.replace_column_empty_value(
        df, column_name="h2h_direct", replace_value=0.0
    )
    make_orientation(variant.sex, df)
    clf_columns.add_column(
        df, "s2_fst_win", lambda r: int(r["s2_fst_games"] > r["s2_snd_games"])
    )
    assert LABEL_NAME in df.columns, "no label after primary edit"


def write_note(variant: cco.Variant, subname: str, text: str):
    dirname = variant.persist_dirname(MODEL)
    filename = os.path.join(dirname, subname, "note.txt")
    with open(filename, "w") as f:
        f.write(f"{text}\n")


def write_df(variant: cco.Variant, df: pd.DataFrame, subname: str):
    dirname = variant.persist_dirname(MODEL)
    cco.save_df(df, dirname, subname=subname)


def read_df(variant: cco.Variant, subname: str) -> pd.DataFrame:
    dirname = variant.persist_dirname(MODEL)
    df = cco.load_df(dirname, subname=subname)
    if df is None:
        raise ValueError(f"no dataset {variant.sex} dirname: {dirname}")
    if df.shape[0] < cco.DATASET_MIN_LEN:
        raise ValueError(f"few dataset {variant.sex} dirname: {dirname}")
    return df


def make_main_reserve(
    variant: cco.Variant,
    reserve_size: float,
    is_shuffle: bool = False,
    random_state: int = RESERVE_RANDOM_STATE,
):
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="openset")
    data0 = pd.read_csv(filename, sep=",")
    primary_edit(variant, data0)
    if data0.shape[0] < cco.DATASET_MIN_LEN:
        msg = f"after drop nan poor dataset ({data0.shape[0]} rows) sex:{sex}"
        print(msg)
        raise ValueError(msg)
    df = cco.stage_shuffle(data0, is_shuffle, random_state=random_state)
    target_names = [LABEL_NAME] + variant.key.get_stratify_names(is_text_style=True)
    df_main, df_reserve = cco.split2_stratified(
        df, target_names=target_names, test_size=reserve_size, random_state=random_state
    )
    assert df_main.shape[0] > 0 and df_reserve.shape[0] > 0
    if variant.weight_mode != cco.WeightMode.NO:
        clf_columns.add_weight_column(variant.weight_mode, df_reserve, LABEL_NAME)
        # df_main will weighted during fill_data_ending_stratify_n
    write_df(variant, df=df_main, subname="main")
    write_df(variant, df=df_reserve, subname="reserve")

    msg = (
        f"shuffle: {is_shuffle} random_seed: {random_seed} "
        f"random_state: {random_state}\n"
    )
    msg_m = f"reserving size {1 - reserve_size} in {df.shape}, main {df_main.shape}\n"
    msg_r = f"reserving size {reserve_size} in {df.shape}, reserve {df_reserve.shape}\n"
    write_note(variant, subname="main", text=f"{msg}{msg_m}")
    write_note(variant, subname="reserve", text=f"{msg}{msg_r}")
    cco.out(msg_m)


def fill_data(
    variant: cco.Variant,
    split: Optional[bool],
    is_shuffle: bool = False,
    random_state: int = 0,
    use_storage: bool = False,
):
    vrnt_key = variant.key
    df0 = read_df(variant, subname="main")
    df = cco.stage_shuffle(df0, is_shuffle, random_state=random_state)
    add_columns(variant.sex, df)

    clf_columns.with_nan_columns(
        df, columns=variant.feature_names.get_list(), raise_ifnan=True
    )
    cat_feats_idx = (
        None if not variant.is_cb_native() else variant.feature_names.cat_indexes()
    )
    vrnt_data, df_spl = cco.fill_data_ending_stratify_n(
        df,
        split,
        test_size=DEFAULT_TEST_SIZE,
        eval_size=DEFAULT_EVAL_SIZE,
        storage_dir=cco.persist_dirname(MODEL, vrnt_key) if use_storage else "",
        feature_names=variant.feature_names.get_list(),
        label_name=LABEL_NAME,
        other_names=(vrnt_key.get_stratify_names(is_text_style=False)),
        cat_features_idx=cat_feats_idx,
        weight_mode=variant.weight_mode,
        random_state=random_state,
    )
    return vrnt_data, df_spl


var_atp_main_clr_rtg200_150_1spvns = cco.Variant(
    sex="atp",
    suffix="1spvns",
    key=cco.key_main_clr_rtg200_150,
    cls=CatBoostClassifier,
    clf_pars={
        "boosting_type": "Ordered",
        "depth": 4,
        "early_stopping_rounds": 80,
        "eval_metric": "AUC",
        "logging_level": "Silent",
        "per_float_feature_quantization": ["0:border_count=512"],
    },
    min_proba=0.54,
    min_neg_proba=1.0,
    profit_ratios=(0.5, 0.72),
    feature_names=cco.FeatureNames(
        names=[
            "fst_bet_chance",
            "dif_srv_ratio",
            "s1_dif_games",
            "dif_elo_alt_pts",
            "dif_surf_elo_alt_pts",
            # next features temporary removed
        ]
    ),
    calibrate="",
)

var_wta_main_clr_rtg200_150_0sp = cco.Variant(
    sex="wta",
    suffix="0sp",
    key=cco.key_main_clr_rtg200_150,
    cls=CatBoostClassifier,
    clf_pars={
        "boosting_type": "Ordered",
        "depth": 4,
        "iterations": 207,
        "learning_rate": 0.03,
        "logging_level": "Silent",
    },
    min_proba=0.53,
    min_neg_proba=1.0,
    profit_ratios=(0.55, 0.7),
    feature_names=cco.FeatureNames(
        names=[
            "fst_bet_chance",
            "dif_srv_ratio",
            "s1_dif_games",
            "dif_elo_alt_pts",
            "dif_surf_elo_alt_pts",
            # next features temporary removed
        ]
    ),
    calibrate="",
)

work_variants = [
    var_wta_main_clr_rtg200_150_0sp,
    var_atp_main_clr_rtg200_150_1spvns,
]
trial_variants = []

# # ---------------------- experiments ---------------------------
from hyperopt import hp

import stopwatch
import clf_hp
import clf_hp_gb
import clf_hp_cb
import clf_hp_rf


def put_seed(seed: int):
    global random_seed
    random_seed = seed
    cco.set_seed(seed)
    cco.out(f"set random_seed {random_seed}")


def scores_reserve(variant: cco.Variant, head: str = ""):
    sex = variant.sex
    if variant.feature_names.exist_cat():
        raise NotImplementedError(f"need pool from reserve, sex: {sex}")
    df = read_df(variant, subname="reserve")
    add_columns(variant.sex, df)
    X, y = cco.get_xy(df, variant.feature_names.get_list(), LABEL_NAME)
    y_pred = variant.clf.predict(X)
    if variant.weight_mode != cco.WeightMode.NO:
        prec = precision_score(y, y_pred, sample_weight=df["weight"].values)
        acc = accuracy_score(y, y_pred, sample_weight=df["weight"].values)
        auc = roc_auc_score(
            y, variant.clf.predict_proba(X)[:, 1], sample_weight=df["weight"].values
        )
    else:
        prec = precision_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, variant.clf.predict_proba(X)[:, 1])
    cco.out(f"{head} prc: {prec} acc: {acc} auc: {auc}")

    res = cco.get_result(variant, variant.clf, X_test=X, y_test=y)
    wl = res.negwl + res.poswl
    cco.out(
        f"wl {wl.ratio_size_str(precision=3)} " f"rat {round(wl.size / y.shape[0], 3)}"
    )


def final_train_save(variant: cco.Variant, seed: int, random_state: int):
    put_seed(seed)
    variant.set_random_state(seed)
    data, _ = fill_data(
        variant,
        split=None,
        is_shuffle=args.shuffle,
        random_state=random_state,
        use_storage=False,
    )
    variant.make_clf_fit(data, metric_name, random_seed=seed)
    variant.save_clf(model_name=MODEL, metric_name=metric_name)
    msg = (
        f"final_train_save done var: {variant} "
        f"seed: {seed} random_state={random_state}"
    )
    write_note(variant, subname="clf", text=msg)
    cco.out(msg)


def random_train(variant, msg="", split=True, plot=False):
    all_name_imp = defaultdict(lambda: 0.0)
    prc_list, acc_list, auc_list, treecnt_list, lrate_list = [], [], [], [], []
    all_wl = st.WinLoss()
    all_test_size = 0

    # log.info(f'====winclose_suffix: {winclose_suffix} params {params} ====')
    for seed in random_args.iter_seeds():
        put_seed(seed)
        variant.set_random_state(seed)

        for random_state in random_args.iter_states():
            log.info(f"random_state={random_state} start learning")
            data, _ = fill_data(
                variant,
                split=split,
                is_shuffle=args.shuffle,
                random_state=random_state,
                use_storage=False,
            )
            clf = variant.make_clf_fit(data, metric_name, random_seed=seed, plot=plot)
            name_imp = variant.get_features_importance(variant.feature_names.get_list())
            for name, imp in name_imp.items():
                all_name_imp[name] += imp
            if variant.weight_mode != cco.WeightMode.NO:
                prec = precision_score(
                    data.test.y, clf.predict(data.test.X), sample_weight=data.test.w
                )
                acc = accuracy_score(
                    data.test.y, clf.predict(data.test.X), sample_weight=data.test.w
                )
                auc = roc_auc_score(
                    data.test.y,
                    clf.predict_proba(data.test.X)[:, 1],
                    sample_weight=data.test.w,
                )
            else:
                prec = precision_score(data.test.y, clf.predict(data.test.X))
                acc = accuracy_score(data.test.y, clf.predict(data.test.X))
                auc = roc_auc_score(data.test.y, clf.predict_proba(data.test.X)[:, 1])
            prc_list.append(prec)
            acc_list.append(acc)
            auc_list.append(auc)
            if variant.is_cb_native():
                treecnt_list.append(clf.tree_count_)
                lrate_list.append(clf.learning_rate_)
            log.info(f"gomean prec {sum(prc_list) / len(prc_list)}")
            res = variant.make_test_result(data)
            # all_wl += (res.poswl + res.negwl)
            all_wl += res.poswl
            all_test_size += data.test.X.shape[0]

    log.info(
        f"******************************************\n"
        f"*****{msg}*** {variant.name} results******\n"
    )
    log.info(f"mean_prc {sum(prc_list) / random_args.space_size()}")
    log.info(f"mean_acc {sum(acc_list) / random_args.space_size()}")
    log.info(f"mean_auc {sum(auc_list) / random_args.space_size()}")
    if variant.is_cb_native():
        log.info(f"treecnt {sum(treecnt_list) / random_args.space_size()}")
        log.info(f"lrate {sum(lrate_list) / random_args.space_size()}")
    log.info(
        f"all_wl {all_wl.ratio_size_str(precision=4)} "
        f"ratio {round(all_wl.size / all_test_size, 3)}"
    )
    log.info("all_name_imp:")
    all_name_imp_list = [
        (k, v / random_args.space_size()) for k, v in all_name_imp.items()
    ]
    all_name_imp_list.sort(key=lambda it: it[1], reverse=True)
    log.info("\n" + pprint.pformat(all_name_imp_list))


def hp_search(
    variant: cco.Variant, data, max_evals, mix_algo_ratio=None, random_state=None
):
    timer = stopwatch.Timer()
    if variant.is_gb():
        clf_hp_gb.do_fmin(
            data,
            max_evals,
            mix_algo_ratio=mix_algo_ratio,
            max_depth_space=hp.pchoice("max_depth", [(0.40, 1), (0.60, 2)]),
            random_state=random_state,
        )
    elif variant.is_rf():
        clf_hp_rf.do_fmin(
            data,
            max_evals,
            mix_algo_ratio=mix_algo_ratio,
            max_depth_space=hp.pchoice("max_depth", [(0.45, 1), (0.55, 2)]),
            random_state=random_state,
        )
    elif variant.is_cb_native():
        pools = clf_cat_tools.make_pools(data)
        clf_hp_cb.do_fmin(
            pools,
            max_evals,
            mix_algo_ratio=mix_algo_ratio,
            random_state=random_state,
            how="native",
        )
    sec_elapsed = timer.elapsed
    cco.out("{} done for {}".format(variant.name, cco.fmt(sec_elapsed)))


if __name__ == "__main__":
    import argparse

    def parse_command_line_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--sex", choices=["wta", "atp"])
        parser.add_argument("--variant", type=str)
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument("--search", action="store_true")
        parser.add_argument("--scores_reserve", action="store_true")
        parser.add_argument("--final_train_save", action="store_true")
        parser.add_argument("--reserve", action="store_true")
        parser.add_argument("--random_train", action="store_true")
        parser.add_argument("--log_instance", type=int)
        parser.add_argument("--random_train_suffix", action="store_true")
        co.RandomArguments.prepare_parser(parser)
        return parser.parse_args()

    def go_search(variant: cco.Variant):
        put_seed(random_args.get_any_seed())
        random_state = random_args.get_any_state()
        data, _ = fill_data(
            variant,
            split=variant.is_cb_native(),
            is_shuffle=args.shuffle,
            random_state=random_state,
            use_storage=False,
        )
        hp_search(
            variant,
            data,
            max_evals=100,
            mix_algo_ratio=clf_hp.MixAlgoRatio(tpe=0.6, anneal=0.4, rand=0.00),
            random_state=random_state,
        )

    args = parse_command_line_args()
    log.initialize(
        co.logname(__file__, instance=args.log_instance),
        file_level="info",
        console_level="info",
    )
    co.log_command_line_args()
    random_args = co.RandomArguments(args, default_seed=random_seed)
    if args.variant:
        if args.sex:
            in_variant = cco.Variant.find_variant_by(
                variants=trial_variants,
                predicate=lambda v: v.name == args.variant and v.sex == args.sex,
            )
        else:
            in_variant = cco.Variant.find_variant_by(
                variants=trial_variants, predicate=lambda v: v.name == args.variant
            )
        assert in_variant is not None, f"not found variant '{args.variant}'"
    else:
        in_variant = None
    log.info(f"==== var {in_variant}")
    if args.search:
        go_search(in_variant)
    elif args.random_train:
        random_train(variant=in_variant)
    elif args.reserve:
        make_main_reserve(
            in_variant, reserve_size=DEFAULT_RESERVE_SIZE, is_shuffle=args.shuffle
        )
    elif args.scores_reserve:
        initialize(variants=[in_variant])
        scores_reserve(variant=in_variant)
    elif args.final_train_save:
        final_train_save(
            variant=in_variant,
            seed=random_args.get_any_seed(),
            random_state=random_args.get_any_state(),
        )
    # sys.exit(0)
