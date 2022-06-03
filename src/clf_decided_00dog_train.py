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
from typing import Optional, Union

import df_util

CMD_ENVIRON = os.environ
CMD_ENVIRON["OMP_NUM_THREADS"] = "1"

import pandas as pd
import pprint

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score, average_precision_score
)

from clf_decided_00dog_variants import (
    var_atp_main_clr_rtg500_300_1ffrs,
)
import cfg_dir
from loguru import logger as log
import lev
import common as co
import surf
import clf_common as cco
import stat_cont as st
import clf_cat_tools
import clf_columns

LABEL_NAME = "dog_win_match"

DEFAULT_RESERVE_SIZE = 0.085  # for wta chal, usial ->   # 0.08
RESERVE_RANDOM_STATE = 466
DEFAULT_EVAL_SIZE = 0.17
DEFAULT_TEST_SIZE = 0.17  # for wta chal 0.20, usial ->  # 0.17
# если CHRONO_SPLIT задан то все наборы упорядочены по росту 'date'
# и разбиения тоже (на 3: train, eval, test), или (на 2: train, test)
CHRONO_SPLIT = False
DEFAULT_CLASSIFIER = CatBoostClassifier

random_seed = 44
early_stopping_rounds = None


train_variants = [
    var_atp_main_clr_rtg500_300_1ffrs,
]


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
    clf_columns.make_decided_win_side_by_score(
        tbl, side=co.RIGHT, new_name="decided_win_snd_plr_by_score"
    )
    add_fst_s2choke_adv(tbl)


def drop_nan_labels(sex: str, data: pd.DataFrame):
    todrop = [
        "dset_ratio_dif",
        "fst_bet_chance",
    ]
    data.dropna(subset=todrop, inplace=True)


def drop_seldom_score(df: pd.DataFrame) -> None:
    df_util.drop_by_condition(
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
    max_dog_chance, min_fav_chance = cco.dogfav_limit_chances(
        variant.sex, variant.exist_chal())
    df_util.drop_by_condition(
        df,
        lambda d: (
            (d["sspd_absent_games"] >= 3)
            | (d["sspd_empty_games"] >= 4)
            # drop tough decset (exist 7-5, 7-6, 8-6 and so on) as noisy:
            # | ((d["s3_fst_games"] + d["s3_snd_games"]) >= 12)
            # drop only tie decset with 7-6:
            # | ((d["s3_fst_games"] == 7) & (d["s3_snd_games"] == 6))
            # | ((d["s3_fst_games"] == 6) & (d["s3_snd_games"] == 7))
            | (d["best_of_five"] == 1)
            | (d["rnd_text"] == "Final")
            | ((d["fst_bet_chance"] >= max_dog_chance)
               & (d["fst_bet_chance"] <= min_fav_chance))
        ),
    )
    variant.drop_by_condition(df)


def primary_edit(variant: cco.Variant, df: pd.DataFrame):
    drop_nan_labels(variant.sex, df)
    df_util.drop_rank_std_any_below_or_wide_dif(
        df, cco.RANK_STD_BOTH_ABOVE, cco.RANK_STD_MAX_DIF
    )
    clf_columns.edit_column_value(df, "rnd_text", cco.unified_rnd_value)
    clf_columns.add_column(df, "rnd", lambda r: cco.rnd_code(r["rnd_text"]))
    # clf_columns.add_column(df, 'rnd_norm_value', lambda r: cco.rnd_norm_value(r['rnd_text']))
    # clf_columns.add_column(df, 'rnd_money_stage',
    #                        lambda r: cco.rnd_money_stage(
    #                            r['rnd_text'], r['raw_level_text'], r['tour_money']))
    clf_columns.add_column(
        df, "rnd_stage",
        lambda r: cco.rnd_stage(r["rnd_text"], r["raw_level_text"], qual_union=True)
    )
    clf_columns.add_column(df, "level", lambda r: lev.level_code(r["level_text"]))
    clf_columns.add_column(
        df, "surface", lambda r: surf.get_code(r["surface_text"]))

    drop_by_condition(variant, df)
    if args.drop_seldom:
        drop_seldom_score(df)

    clf_columns.add_column(
        df, "raw_level_code",
        lambda r: lev.raw_level_code(r["rnd_text"], r["raw_level_text"])
    )

    clf_columns.replace_column_empty_value(
        df, column_name="h2h_direct", replace_value=0.0)

    clf_columns.add_column(df, "year", lambda r: int(r["date"][:4]))
    # df["opener_lose_match"] = 1 - df["fst_win_match"]
    df["fst_dog"] = (df["fst_bet_chance"] < 0.5).astype(int)
    # our positive target:
    clf_columns.add_column(
        df, "dog_win_match", lambda r: int((r["fst_dog"] + r["fst_win_match"]) != 1)
    )
    assert LABEL_NAME in df.columns, "no label after primary edit"


def make_main_reserve(
    variant: cco.Variant,
    is_shuffle: bool = False,
    random_state: int = RESERVE_RANDOM_STATE,
):
    """ читает исходный файл csv,
       очищает (удаляет строки где важные колонки пустые(NaN)
                или где не вып-ны ограничения по рейтингам,
                или где не вып-ны ограничения по плотности решающ. партии),
       перемешивает строки (опционально, если задан is_shuffle)
       добавляет лищь некоторые клонки ('year', LABEL_NAME),
       разбивает на main/reserve части в соответствии с reserve_size, CHRONO_SPLIT
       сохраняет наборы в папки main/reserve для входного варанта variant
   """
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="decidedset")
    data0 = pd.read_csv(filename, sep=",")
    primary_edit(variant, data0)
    if data0.shape[0] < cco.DATASET_MIN_LEN:
        msg = f"after primary_edit poor dataset ({data0.shape[0]} rows) sex:{sex}"
        cco.out(msg)
        raise ValueError(msg)
    df = df_util.stage_shuffle(data0, is_shuffle, random_state=random_state)
    if DEFAULT_RESERVE_SIZE < 0.01:
        df_main = df
        reserve_size = 0.0
        reserve_shape = (0, 0)
    else:
        if CHRONO_SPLIT:
            df_main, df_reserve = cco.split_all_by_two_chunks(
                df, remain_size=DEFAULT_RESERVE_SIZE)
        else:
            target_names = [LABEL_NAME] + variant.key.get_stratify_names()
            df_main, df_reserve = cco.split2_stratified(
                df,
                target_names=target_names,
                test_size=DEFAULT_RESERVE_SIZE,
                random_state=random_state,
            )
        assert df_main.shape[0] > 0 and df_reserve.shape[0] > 0
        reserve_size = round(df_reserve.shape[0] / df.shape[0], 1)
        reserve_shape = df_reserve.shape
        variant.write_df(df=df_reserve, subname="reserve")

    variant.write_df(df=df_main, subname="main")

    msg = (f"shuffle: {is_shuffle} chrono_split: {CHRONO_SPLIT} "
           f"random_seed: {random_seed} random_state: {random_state}\n")
    msg_m = f"reserving size {1 - reserve_size} in {df.shape}, main: {df_main.shape}\n"
    msg_r = f"reserving size {reserve_size} in {df.shape}, reserve: {reserve_shape}\n"
    variant.write_note(subname="main", text=f"{msg}{msg_m}")
    variant.write_note(subname="reserve", text=f"{msg}{msg_r}")
    cco.out(msg_m)


def make_preanalyze_df(variant: cco.Variant):
    preanalyze_fnames = [
        "fst_bet_chance",
        "dif_elo_alt_pts",
        # and some others
    ]
    sex = variant.sex
    filename = cfg_dir.analysis_data_file(sex, typename="decidedset")
    data0 = pd.read_csv(filename, sep=",")
    primary_edit(variant, data0)
    add_columns(variant.sex, data0)
    if data0.shape[0] < cco.DATASET_MIN_LEN:
        msg = f"after drop nan poor dataset ({data0.shape[0]} rows) sex:{sex}"
        cco.out(msg)
        raise ValueError(msg)
    df_out = data0[preanalyze_fnames + [LABEL_NAME]]
    dirname = variant.persist_dirname()
    filename = os.path.join(dirname, "preanalyze", "df.csv")
    df_out.to_csv(filename, index=False)
    variant.write_note(subname="preanalyze", text=f"size {df_out.shape}")


def fill_data(
    variant: cco.Variant,
    split: Optional[bool],
    is_shuffle: bool = False,
    random_state: int = 0,
):
    vrnt_key = variant.key
    df0 = variant.read_df(subname="main")
    df = df_util.stage_shuffle(df0, is_shuffle, random_state=random_state)
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
            test_size=DEFAULT_TEST_SIZE,
            eval_size=DEFAULT_EVAL_SIZE,
            feature_names=variant.feature_names.get_list(),
            label_name=LABEL_NAME,
            other_names=vrnt_key.get_stratify_names(),
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
            other_names=vrnt_key.get_stratify_names(),
            cat_features_idx=cat_feats_idx,
            weight_mode=variant.weight_mode,
            random_state=random_state,
        )
    return vrnt_data, df_spl


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
    """ input: variant with fited clf, reserved csv file.
        out: predict precision, accuracy, auc """
    assert variant.clf is not None, "variant without clf"
    assert variant.clf.is_fitted(), "not fitted clf"
    df = variant.read_df(subname="reserve")
    add_columns(variant.sex, df)
    X, y = cco.get_xy(df, variant.feature_names.get_list(), LABEL_NAME)
    if variant.is_cb_native() and variant.exist_cat():
        X = cco.x_tolist(X, variant.feature_names.cat_indexes())
        y = y.tolist()
    y_pred = variant.clf.predict(X)
    prec = precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, variant.clf.predict_proba(X)[:, 1])
    cco.out(f"{head} prc: {prec} acc: {acc} auc: {auc}")

    res = cco.get_result(variant, variant.clf, X_test=X, y_test=y)
    wl = res.negwl + res.poswl
    cco.out(f"wl {wl.ratio_size_str(precision=3)} " 
            f"rat {round(wl.size / len(y), 3)}")
    cco.out(f"pwl {res.poswl.ratio_size_str(precision=3)} " 
            f"rat {round(res.poswl.size / len(y), 3)}")
    cco.out(f"nwl {res.negwl.ratio_size_str(precision=3)} " 
            f"rat {round(res.negwl.size / len(y), 3)}")


def final_train_save(variant: cco.Variant, seed: int, random_state: int):
    put_seed(seed)
    variant.set_random_state(seed)
    data, _ = fill_data(
        variant,
        split=False,  # it supplies eval=None
        # split=None,
        is_shuffle=args.shuffle,
        random_state=random_state,
    )
    variant.make_clf_fit(data, random_seed=seed)
    variant.save_clf()
    msg = (
        f"final_train_save done var: {variant} "
        f"seed: {seed} random_state={random_state}"
    )
    variant.write_note(subname="clf", text=msg)
    cco.out(msg)


def random_train(variant: cco.Variant, msg="", split=True, plot=False, verbose=False):
    all_name_imp = defaultdict(lambda: 0.0)
    aprc_list, prc_list, acc_list, auc_list, treecnt_list, lrate_list = (
        [], [], [], [], [], []
    )
    pos_wl = st.WinLoss()
    neg_wl = st.WinLoss()
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
            if verbose:
                cco.out(data.size_report())
            clf = variant.make_clf_fit(data, random_seed=seed, plot=plot)
            name_imp = variant.get_features_importance()
            for name, imp in name_imp.items():
                all_name_imp[name] += imp
            aprec = average_precision_score(data.test.y,
                                            clf.predict_proba(data.test.X)[:, 1])
            prec = precision_score(data.test.y, clf.predict(data.test.X))
            acc = accuracy_score(data.test.y, clf.predict(data.test.X))
            auc = roc_auc_score(data.test.y, clf.predict_proba(data.test.X)[:, 1])
            aprc_list.append(aprec)
            prc_list.append(prec)
            acc_list.append(acc)
            auc_list.append(auc)
            if variant.is_cb_native():
                treecnt_list.append(clf.tree_count_)
                lrate_list.append(clf.learning_rate_)
            log.info(f"gomean acc {sum(acc_list) / len(acc_list)}")
            res = variant.make_test_result(data)
            pos_wl += res.poswl
            neg_wl += res.negwl
            all_test_size += len(data.test.X)  # was shape[0]

    log.info(f"******************************************\n"
             f"*****{msg}*** {variant.name} results******")
    log.info(f"mean_aprc {sum(aprc_list) / random_args.space_size()}")
    log.info(f"mean_prc {sum(prc_list) / random_args.space_size()}")
    log.info(f"mean_acc {sum(acc_list) / random_args.space_size()}")
    log.info(f"mean_auc {sum(auc_list) / random_args.space_size()}")
    if variant.is_cb_native():
        log.info(f"treecnt {sum(treecnt_list) / random_args.space_size()}")
        log.info(f"lrate {sum(lrate_list) / random_args.space_size()}")
    all_wl = pos_wl + neg_wl
    log.info(f"all_wl {all_wl.ratio_size_str(precision=4)} "
             f"ratio {round(all_wl.size / all_test_size, 3)}")
    log.info(f"pos_wl {pos_wl.ratio_size_str(precision=4)} "
             f"ratio {round(pos_wl.size / all_test_size, 3)}")
    log.info(f"neg_wl {neg_wl.ratio_size_str(precision=4)} "
             f"ratio {round(neg_wl.size / all_test_size, 3)}")
    log.info("all_name_imp:")
    all_name_imp_list = [
        (k, v / random_args.space_size()) for k, v in all_name_imp.items()
    ]
    all_name_imp_list.sort(key=lambda it: it[1], reverse=True)
    log.info("\n" + pprint.pformat(all_name_imp_list))


def hyperopt_search(
    variant: cco.Variant,
    data: Union[cco.Data, clf_cat_tools.DataCatBoost],
    max_evals,
    mix_algo_ratio=None,
    random_state=None
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
        if data.exist_cat():
            assert variant.exist_cat(), (
                f"data with cat, variant without cat:"
                f"\n{variant.feature_names}")
            data.tolist()
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
        parser.add_argument("--search", action="store_true")
        parser.add_argument("--scores_reserve", action="store_true")
        parser.add_argument("--final_train_save", action="store_true")
        parser.add_argument("--reserve", action="store_true")
        parser.add_argument("--random_train", action="store_true")
        parser.add_argument("--log_instance", type=int, default=3)
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
            mix_algo_ratio=clf_hp.MixAlgoRatio(tpe=0.00, anneal=1.00, rand=0.00),
            random_state=random_state,
        )

    args = parse_command_line_args()
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
    if args.search:
        go_hyperopt_search(in_variant)
    elif args.preanalyze:
        make_preanalyze_df(variant=in_variant)
    elif args.random_train:
        random_train(variant=in_variant)
    elif args.reserve:
        put_seed(random_args.get_any_seed())
        make_main_reserve(in_variant, is_shuffle=args.shuffle,
                          random_state=random_args.get_any_state())
    elif args.scores_reserve:
        cco.set_seed(random_seed)
        cco.load_variants(variants=[in_variant])
        scores_reserve(variant=in_variant)
    elif args.final_train_save:
        final_train_save(
            variant=in_variant,
            seed=random_args.get_any_seed(),
            random_state=random_args.get_any_state(),
        )
    # sys.exit(0)
