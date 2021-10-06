from collections import defaultdict

import pprint
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import log
import common as co
import clf_common as cco
import stat_cont as st
import clf_cat_tools

from clf_secondset_00 import (
    LABEL_NAME,
    read_df,
    fill_data,
    add_columns,
    metric_name,
    random_seed,
    MODEL,
    initialize,
    make_preanalyze_df,
    make_main_reserve,
    DEFAULT_RESERVE_SIZE,
    write_note
)
from clf_secondset_00_variants import var_wta_main_clr_rtg200_150_0sp


from hyperopt import hp
import stopwatch
import clf_hp
import clf_hp_gb
import clf_hp_cb
import clf_hp_rf


trial_variants = [var_wta_main_clr_rtg200_150_0sp]


def put_seed(seed: int):
    cco.set_seed(seed)
    cco.out(f"set random_seed {seed}")


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
        parser.add_argument("--preanalyze", action="store_true")
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
    elif args.preanalyze:
        make_preanalyze_df(variant=in_variant)
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
