"""
module gives service of classification (match_has_min_proba).
also it for making classifiers and saving them to files.

initialy in openset_data.csv data prepared so that fst player is set1 opener,
We want data preparing (orienting) so that fst player is set1 loser,
    and after that orienting label is about set1_loser_win_set2
"""
import os
from typing import Optional

CMD_ENVIRON = os.environ
CMD_ENVIRON["OMP_NUM_THREADS"] = "1"

import pandas as pd

from catboost import CatBoostClassifier

from clf_secondset_00_variants import var_wta_main_clr_rtg200_150_0sp
from side import Side
import cfg_dir
import log
import common as co
import clf_common as cco
import clf_columns
import predicts_db

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
                match.level != "masters"
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
                match.level != "masters"
                and match.rnd is not None
                and match.rnd.qualification()
            )
            or not match.is_ranks_both_above(200, rtg_name="std")
            or match.is_ranks_dif_wide(max_dif=150, rtg_name="std")
        )


preanalyze_fnames = [
    "fst_bet_chance",
    "dif_srv_ratio",  # replace dif_avg_srv_code
    "s1_dif_games",
    # other features temporary removed
]

winclose_suffix = "_mean_57"
metric_name = "Precision"  # 'AUC'  # 'Accuracy'
random_seed = 44
early_stopping_rounds = None


def add_dif_winclose(sex, tbl):
    if sex == "wta":
        tbl["dif_winclose" + winclose_suffix] = (
            tbl["s1_snd_winclose" + winclose_suffix]
            - tbl["s1_fst_winclose" + winclose_suffix]
        )


def add_winclose_set2loser(sex, tbl):
    if sex == "wta":
        for i in tbl.index:
            s2_fst_win = tbl.at[i, "s2_fst_games"] > tbl.at[i, "s2_snd_games"]
            if s2_fst_win:
                val = tbl.at[i, "spd_snd_winclose" + winclose_suffix]
            else:
                val = tbl.at[i, "spd_fst_winclose" + winclose_suffix]
            tbl.at[i, "winclose_set2loser" + winclose_suffix] = val


def add_winclose_set2loser_signed(sex, tbl):
    if sex == "wta":
        for i in tbl.index:
            s2_fst_win = tbl.at[i, "s2_fst_games"] > tbl.at[i, "s2_snd_games"]
            if s2_fst_win:
                val = -tbl.at[i, "spd_snd_winclose" + winclose_suffix]
            else:
                val = tbl.at[i, "spd_fst_winclose" + winclose_suffix]
            tbl.at[i, "winclose_set2loser_signed" + winclose_suffix] = val


def add_columns(sex, tbl):
    if sex == "wta":
        clf_columns.edit_column_value(tbl, "tour_money", lambda m: m / 100000)
    clf_columns.add_column(tbl, "rnd", lambda r: cco.rnd_code(r["rnd_text"]))
    clf_columns.add_column(tbl, "level", lambda r: cco.level_code(r["level_text"]))
    clf_columns.add_column(
        tbl, "surface", lambda r: cco.surface_code(r["surface_text"])
    )

    add_dif_winclose(sex, tbl)
    add_winclose_set2loser_signed(sex, tbl)
    tbl["dif_srv_ratio"] = tbl["s1_snd_srv_ratio"] - tbl["s1_fst_srv_ratio"]
    tbl["s1_dif_games"] = tbl["s1_snd_games"] - tbl["s1_fst_games"]
    # other features temporary removed


def make_orientation(sex, tbl):
    """after call fst, snd orient: fst is loser first set"""

    def flip_bool(idx, name):
        tbl.at[idx, name] = 1 - tbl.at[idx, name]

    def flip_sign(idx, name):
        tbl.at[idx, name] = -tbl.at[idx, name]

    def flip_sign_nonzero(idx, name):
        if tbl.at[idx, name] != 0:
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
        # other features temporary removed


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
    ]
    if sex == "wta":
        todrop.append("tour_money")
    data.dropna(subset=todrop, inplace=True)


def match_has_min_proba(match, set_open_side: Side):
    def log_estimate():
        if positive_prob > 0.5:
            max_prob = round(positive_prob, 3)
            max_prob_plr = (
                match.second_player if win_set1_side.is_left() else match.first_player
            )
            log.info(
                f"clf {MODEL} try {match} \nopen_side:{set_open_side} "
                f"PROB: {max_prob} PLR {max_prob_plr}"
            )
            predicts_db.write_predict(
                match, case_name='secondset_00',
                back_side=win_set1_side.fliped(),
                proba=max_prob, comments='')

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
        if positive_prob >= variant.min_probas.pos:
            return win_set1_side.fliped(), positive_prob
    except cco.FeatureError as err:
        log.warn("clf {} prep err: {} {}".format(MODEL, err, match.tostring()))
        return None, None
    return None, None


# for one player per one set in history.
MIN_SET_SIZE = 11  # if 6-0 we have min = 3 games * 4 points
MIN_HIST_DATA_SIZE = 5


def get_match_features(match, variant: cco.Variant):
    """ return match features (X), win_set1_side """
    return None, None  # body temporary removed


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


def make_preanalyze_df(variant: cco.Variant):
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="openset")
    data0 = pd.read_csv(filename, sep=",")
    primary_edit(variant, data0)
    add_columns(variant.sex, data0)
    if data0.shape[0] < cco.DATASET_MIN_LEN:
        msg = f"after drop nan poor dataset ({data0.shape[0]} rows) sex:{sex}"
        print(msg)
        raise ValueError(msg)
    df_out = data0[preanalyze_fnames + [LABEL_NAME]]
    dirname = variant.persist_dirname(MODEL)
    filename = os.path.join(dirname, "preanalyze", "df.csv")
    df_out.to_csv(filename, index=False)
    write_note(variant, subname="preanalyze", text=f"size {df_out.shape}")


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


work_variants = [
    var_wta_main_clr_rtg200_150_0sp,
]


