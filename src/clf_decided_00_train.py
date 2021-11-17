"""
module for training classifiers and saving them to files.

data prepared so that:
    одна sample строка соответсвует одному матчу где есть решающий сет
    fst player is opener on serve in the decided set,
and snd player starts decided set with receive.
It remains for all sets data (s1_, sp_, sd_) // so s1_fst_... is opener in decset
So, snd is who serve last in set2.
if set2.total is odd -> snd is set2 opener;
if set2.total is even -> fst is set2 opener.

Исходные файлы csv доступный для экспериментов/тренировок/построения моделей:
    bin/analysis_data/wta_decidedset_data.csv
    bin/analysis_data/atp_decidedset_data.csv

Для каждого Variant автом-ки созд-ся папка
напр: decided_00_wta_main_clr_rtg350_250_1 для var_wta_main_clr_rtg350_250_1
Подпапки:
    clf для сохранения моделей clf_*.model
    preanalyze для сохранения набора данных X c колонками (features)
               которые хочется покрутить/посмотреть (корреляции и подобное)
               в предварительном режиме
    main
    reserve

some notations:
    srv_ratio - частота выигрыша розыгрыша при своей подаче в первых 2 партиях
    hold_s12 - частота выигрышаг ейма при своей подаче в первых 2 партиях
    split (как параметр): None нет никакого разбинения
                          False разбиение на 2 части (train, test)
                          True разбиение на 3 части (train, eval, test)
"""
import os
from collections import defaultdict
from typing import Optional

from clf_decided_00_apply import load_variants

CMD_ENVIRON = os.environ
CMD_ENVIRON["OMP_NUM_THREADS"] = "1"

import pandas as pd
import pprint

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from clf_decided_00_variants import (
    var_atp_main_clr_1f,
)
import cfg_dir
import log
import common as co
import clf_common as cco
import stat_cont as st
import clf_cat_tools
import clf_columns

MODEL = "decided_00"
LABEL_NAME = "opener_lose_match"

DEFAULT_RESERVE_SIZE = 0.10
RESERVE_RANDOM_STATE = 466
DEFAULT_EVAL_SIZE = 0.17
DEFAULT_TEST_SIZE = 0.20
CHRONO_SPLIT = True
DEFAULT_CLASSIFIER = CatBoostClassifier

metric_name = "Accuracy"
random_seed = 44
early_stopping_rounds = None


def add_fst_s2choke_adv(tbl: pd.DataFrame):
    """ добавляет в tbl колонку:
        fst_s2choke_adv = 1 if snd s2lose when brkup;
        fst_s2choke_adv = -1 if fst s2lose when brkup;
        fst_s2choke_adv = 0 if s2loser had no brkup;
    """

    def is_pos(fld):
        return fld > 0.5

    def is_zero(fld):
        return abs(fld) < 0.0001

    for i in tbl.index:
        fst_s2lose = tbl.at[i, "s2_fst_games"] < tbl.at[i, "s2_snd_games"]
        s2_total = tbl.at[i, "s2_fst_games"] + tbl.at[i, "s2_snd_games"]
        fst_s2opener = co.is_even(s2_total)  # snd serve last in s2
        s2opener_had_brkup = (
            (is_pos(tbl.at[i, "s2_is_1-0"]) and is_zero(tbl.at[i, "s2_is_1-1"]))
            or (is_pos(tbl.at[i, "s2_is_2-1"]) and is_zero(tbl.at[i, "s2_is_2-2"]))
            or (is_pos(tbl.at[i, "s2_is_3-2"]) and is_zero(tbl.at[i, "s2_is_3-3"]))
            or (is_pos(tbl.at[i, "s2_is_4-3"]) and is_zero(tbl.at[i, "s2_is_4-4"]))
            or (is_pos(tbl.at[i, "s2_is_5-4"]) and is_zero(tbl.at[i, "s2_is_5-5"]))
            or (is_pos(tbl.at[i, "s2_is_6-5"]) and is_zero(tbl.at[i, "s2_is_6-6"]))
        )
        s2closer_had_brkup = (
            is_pos(tbl.at[i, "s2_is_0-1"])
            or is_pos(tbl.at[i, "s2_is_1-2"])
            or is_pos(tbl.at[i, "s2_is_2-3"])
            or is_pos(tbl.at[i, "s2_is_3-4"])
            or is_pos(tbl.at[i, "s2_is_4-5"])
            or is_pos(tbl.at[i, "s2_is_5-6"])
        )
        if not s2opener_had_brkup and not s2closer_had_brkup:
            result = 0
        elif fst_s2lose:
            if fst_s2opener:
                result = -1 if s2opener_had_brkup else 0
            else:
                result = -1 if s2closer_had_brkup else 0
        else:  # snd_s2lose
            if fst_s2opener:
                result = int(s2closer_had_brkup)
            else:
                result = int(s2opener_had_brkup)
        tbl.at[i, "fst_s2choke_adv"] = result


def add_columns(sex: str, tbl: pd.DataFrame):
    clf_columns.replace_column_empty_value(
        tbl, column_name="h2h_direct", replace_value=0.0
    )
    clf_columns.add_column(tbl, "rnd", lambda r: cco.rnd_code(r["rnd_text"]))
    clf_columns.add_column(
        tbl, "rnd_stage", lambda r: cco.rnd_stage(r["rnd_text"], r["raw_level_text"])
    )
    clf_columns.add_column(tbl, "level", lambda r: cco.level_code(r["level_text"]))
    clf_columns.add_column(
        tbl, "surface", lambda r: cco.surface_code(r["surface_text"])
    )
    clf_columns.make_decided_win_side_by_score(
        tbl, side=co.RIGHT, new_name="decided_win_snd_plr_by_score"
    )
    add_fst_s2choke_adv(tbl)
    tbl["dif_srv_ratio"] = tbl["sspd_snd_srv_ratio"] - tbl["sspd_fst_srv_ratio"]

    tbl["dif_service_win"] = tbl["snd_service_win"] - tbl["fst_service_win"]
    tbl["dif_receive_win"] = tbl["snd_receive_win"] - tbl["fst_receive_win"]

    tbl["dif_age"] = tbl["snd_age"] - tbl["fst_age"]
    clf_columns.add_column(
        tbl, "avg_age", lambda r: (r["fst_age"] + r["snd_age"]) * 0.5
    )
    tbl["dif_fatigue"] = tbl["snd_fatigue"] - tbl["fst_fatigue"]

    tbl["dif_elo_pts"] = tbl["snd_elo_pts"] - tbl["fst_elo_pts"]
    tbl["dif_surf_elo_pts"] = tbl["snd_surf_elo_pts"] - tbl["fst_surf_elo_pts"]

    tbl["dif_elo_alt_pts"] = tbl["snd_elo_alt_pts"] - tbl["fst_elo_alt_pts"]
    tbl["dif_surf_elo_alt_pts"] = (
        tbl["snd_surf_elo_alt_pts"] - tbl["fst_surf_elo_alt_pts"]
    )


def drop_nan_labels(sex: str, data: pd.DataFrame):
    todrop = [
        "sspd_absent_games",
        "fst_bet_chance",
        "spd_fst_lastrow_games",
        "sspd_fst_srv_ratio",
        "sspd_snd_srv_ratio",
        "fst_service_win",
        "snd_service_win",
        "fst_receive_win",
        "snd_receive_win",
        "decided_win_by_set2_winner",
        "fst_age",
        "snd_age",
        "fst_fatigue",
        "snd_fatigue",
    ]
    data.dropna(subset=todrop, inplace=True)


def drop_seldom_score(df: pd.DataFrame) -> None:
    cco.drop_by_condition(
        df,
        lambda d: (
            (  # seldom scores ala 6-0(1), 0-6
                ((d["s1_fst_games"] == 6) & (d["s1_snd_games"].isin([0, 1])))
                & ((d["s2_fst_games"] == 0) & (d["s2_snd_games"] == 6))
            )
            | (  # seldom scores ala 6-0, 1-6
                ((d["s1_fst_games"] == 6) & (d["s1_snd_games"] == 0))
                & ((d["s2_fst_games"] == 1) & (d["s2_snd_games"] == 6))
            )
            | (  # seldom scores ala 0(1)-6, 6-0
                ((d["s1_fst_games"].isin([0, 1])) & (d["s1_snd_games"] == 6))
                & ((d["s2_fst_games"] == 6) & (d["s2_snd_games"] == 0))
            )
            | (  # seldom scores ala 0-6, 6-1
                ((d["s1_fst_games"] == 0) & (d["s1_snd_games"] == 6))
                & ((d["s2_fst_games"] == 6) & (d["s2_snd_games"] == 1))
            )
        ),
    )


def drop_by_condition(variant: cco.Variant, df: pd.DataFrame) -> None:
    cco.drop_by_condition(
        df,
        lambda d: (
            (d["sspd_absent_games"] >= 3)
            | (d["sspd_empty_games"] >= 4)
            # drop tough decset (exist 7-5, 7-6, 8-6 and so on) as noisy:
            | ((d["s3_fst_games"] + d["s3_snd_games"]) >= 12)
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
    if args.drop_seldom:
        drop_seldom_score(df)

    clf_columns.add_column(df, "year", lambda r: int(r["date"][:4]))
    df["opener_lose_match"] = 1 - df["fst_win_match"]  # So our target positive now
    assert LABEL_NAME in df.columns, "no label after primary edit"

    if variant.weight_mode == cco.WeightMode.BY_TOTAL_COLUMN:
        clf_columns.add_column(
            df, "total_dec", lambda r: min(12, (r["s3_fst_games"] + r["s3_snd_games"]))
        )
        clf_columns.add_weight_column(variant.weight_mode, df, LABEL_NAME)
        # df_main will weighted during fill_data_ending_stratify_n


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
    """ читает исходный файл csv,
       очищает (удаляет строки где важные колонки пустые(NaN)
                или где не вып-ны ограничения по рейтингам,
                или где не вып-ны ограничения по плотности решающ. партии),
       перемешивает строки (опционально, если задан is_shuffle)
       добавляет лищь некоторые клонки ('year', 'opener_lose_match'),
       разбивает на main/reserve части в соответствии с reserve_size, CHRONO_SPLIT
       сохраняет наборы в папки main/reserve для входного варанта variant
   """
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="decidedset")
    data0 = pd.read_csv(filename, sep=",")
    primary_edit(variant, data0)
    if data0.shape[0] < cco.DATASET_MIN_LEN:
        msg = f"after drop nan poor dataset ({data0.shape[0]} rows) sex:{sex}"
        print(msg)
        raise ValueError(msg)
    df = cco.stage_shuffle(data0, is_shuffle, random_state=random_state)
    target_names = [LABEL_NAME] + variant.key.get_stratify_names(is_text_style=True)
    if CHRONO_SPLIT:
        df_main, df_reserve = cco.split2_by_year(df, test_years=cco.CHRONO_TEST_YEARS)
    else:
        df_main, df_reserve = cco.split2_stratified(
            df,
            target_names=target_names,
            test_size=reserve_size,
            random_state=random_state,
        )
    assert df_main.shape[0] > 0 and df_reserve.shape[0] > 0

    if variant.weight_mode == cco.WeightMode.BALANCED:
        clf_columns.add_weight_column(variant.weight_mode, df, LABEL_NAME)
        # df_main will weighted during fill_data_ending_stratify_n

    write_df(variant, df=df_main, subname="main")
    write_df(variant, df=df_reserve, subname="reserve")

    msg = (f"shuffle: {is_shuffle} chrono_split: {CHRONO_SPLIT} "
           f"random_seed: {random_seed} random_state: {random_state}\n")
    msg_m = f"reserving size {1 - reserve_size} in {df.shape}, main {df_main.shape}\n"
    msg_r = f"reserving size {reserve_size} in {df.shape}, reserve {df_reserve.shape}\n"
    write_note(variant, subname="main", text=f"{msg}{msg_m}")
    write_note(variant, subname="reserve", text=f"{msg}{msg_r}")
    cco.out(msg_m)


def make_preanalyze_df(variant: cco.Variant):
    preanalyze_fnames = [
        "fst_bet_chance",
        "dset_ratio_dif",
        "decided_win_snd_plr_by_score",
        "dif_elo_alt_pts",
        "dif_surf_elo_alt_pts",
        "dif_srv_ratio",
        "spd_fst_lastrow_games",
        "fst_s2choke_adv",
        "dif_service_win",
        "dif_receive_win",
        "dif_age",
        "avg_age",
        "dif_plr_tour_adapt",
        "dif_fatigue",
        "h2h_direct",
    ]
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="decidedset")
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
    if CHRONO_SPLIT:
        vrnt_data, df_spl = cco.fill_data_ending_chrono(
            df,
            split,
            feature_names=variant.feature_names.get_list(),
            label_name=LABEL_NAME,
            other_names=(vrnt_key.get_stratify_names(is_text_style=False)),
            cat_features_idx=cat_feats_idx,
            weight_mode=variant.weight_mode,
        )
    else:
        vrnt_data, df_spl = cco.fill_data_ending_stratify_n(
            df,
            split,
            test_size=DEFAULT_TEST_SIZE,
            eval_size=DEFAULT_EVAL_SIZE,
            feature_names=variant.feature_names.get_list(),
            label_name=LABEL_NAME,
            other_names=(vrnt_key.get_stratify_names(is_text_style=False)),
            cat_features_idx=cat_feats_idx,
            weight_mode=variant.weight_mode,
            random_state=random_state,
        )
    return vrnt_data, df_spl


train_variants = [var_atp_main_clr_1f]

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
    # prec = precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, variant.clf.predict_proba(X)[:, 1])
    cco.out(f"{head} acc: {acc} auc: {auc}")
    cco.out(f"treecnt: {variant.clf.tree_count_}")
    cco.out(f"lrate: {variant.clf.learning_rate_}")

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
        # split=False,  # it supplies eval=None
        split=None,
        is_shuffle=args.shuffle,
        random_state=random_state,
    )
    variant.make_clf_fit(data, metric_name, random_seed=seed)
    variant.save_clf(model_name=MODEL, metric_name=metric_name)
    msg = (
        f"final_train_save done var: {variant} "
        f"seed: {seed} random_state={random_state}"
    )
    write_note(variant, subname="clf", text=msg)
    cco.out(msg)


def random_train(variant: cco.Variant, msg="", split=True, plot=False):
    all_name_imp = defaultdict(lambda: 0.0)
    prc_list, acc_list, auc_list, treecnt_list, lrate_list = [], [], [], [], []
    all_wl = st.WinLoss()
    all_test_size = 0

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
            )
            clf = variant.make_clf_fit(data, metric_name, random_seed=seed, plot=plot)
            name_imp = variant.get_features_importance(variant.feature_names.get_list())
            for name, imp in name_imp.items():
                all_name_imp[name] += imp
            prec = precision_score(data.test.y, clf.predict(data.test.X))
            acc = accuracy_score(data.test.y, clf.predict(data.test.X))
            auc = roc_auc_score(data.test.y, clf.predict_proba(data.test.X)[:, 1])
            prc_list.append(prec)
            acc_list.append(acc)
            auc_list.append(auc)
            if variant.is_cb_native():
                treecnt_list.append(clf.tree_count_)
                lrate_list.append(clf.learning_rate_)
            log.info(f"gomean acc {sum(acc_list) / len(acc_list)}")
            res = variant.make_test_result(data)
            all_wl += res.poswl + res.negwl
            all_test_size += data.test.X.shape[0]

    log.info(f"******************************************\n"
             f"*****{msg}*** {variant.name} results******\n")
    log.info(f"mean_prc {sum(prc_list) / random_args.space_size()}")
    log.info(f"mean_acc {sum(acc_list) / random_args.space_size()}")
    log.info(f"mean_auc {sum(auc_list) / random_args.space_size()}")
    if variant.is_cb_native():
        log.info(f"treecnt {sum(treecnt_list) / random_args.space_size()}")
        log.info(f"lrate {sum(lrate_list) / random_args.space_size()}")
    log.info(f"all_wl {all_wl.ratio_size_str(precision=4)} "
             f"ratio {round(all_wl.size / all_test_size, 3)}")
    log.info("all_name_imp:")
    all_name_imp_list = [
        (k, v / random_args.space_size()) for k, v in all_name_imp.items()
    ]
    all_name_imp_list.sort(key=lambda it: it[1], reverse=True)
    log.info("\n" + pprint.pformat(all_name_imp_list))


def hyperopt_search(
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
        parser.add_argument("--drop_seldom", action="store_true")
        parser.add_argument("--hyperopt_search", action="store_true")
        parser.add_argument("--scores_reserve", action="store_true")
        parser.add_argument("--final_train_save", action="store_true")
        parser.add_argument("--reserve", action="store_true")
        parser.add_argument("--random_train", action="store_true")
        parser.add_argument("--log_instance", type=int)
        parser.add_argument("--preanalyze", action="store_true")
        co.RandomArguments.prepare_parser(parser)
        return parser.parse_args()

    def go_hyperopt_search(variant: cco.Variant):
        put_seed(random_args.get_any_seed())
        random_state = random_args.get_any_state()
        data, _ = fill_data(
            variant,
            split=variant.is_cb_native(),
            is_shuffle=args.shuffle,
            random_state=random_state,
        )
        hyperopt_search(
            variant,
            data,
            max_evals=100,
            mix_algo_ratio=clf_hp.MixAlgoRatio(tpe=0.50, anneal=0.50, rand=0.00),
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
                variants=train_variants,
                predicate=lambda v: v.name == args.variant and v.sex == args.sex,
            )
        else:
            in_variant = cco.Variant.find_variant_by(
                variants=train_variants, predicate=lambda v: v.name == args.variant
            )
        assert in_variant is not None, f"not found variant '{args.variant}'"
    else:
        in_variant = None
    log.info(f"==== var {in_variant}")
    if args.hyperopt_search:
        go_hyperopt_search(in_variant)
    elif args.preanalyze:
        make_preanalyze_df(variant=in_variant)
    elif args.random_train:
        random_train(variant=in_variant)
    elif args.reserve:
        make_main_reserve(
            in_variant, reserve_size=DEFAULT_RESERVE_SIZE, is_shuffle=args.shuffle
        )
    elif args.scores_reserve:
        cco.set_seed(random_seed)
        load_variants(variants=[in_variant], metric_name=metric_name)
        scores_reserve(variant=in_variant)
    elif args.final_train_save:
        final_train_save(
            variant=in_variant,
            seed=random_args.get_any_seed(),
            random_state=random_args.get_any_state(),
        )
