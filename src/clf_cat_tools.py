from collections import namedtuple

import numpy as np

from catboost import CatBoostClassifier, Pool, cv

import stat_cont as st
import clf_common as cco

"""
# base pipeline for module using (already defined random_seed, metric_name):
import pprint
variant = var_wta_cb_main
data, _ = fill_data(variant=variant, split=True)
pools = clf_cat_tools.make_pools(data)
clf = clf_cat_tools.train_model(pools, metric_name, early_stopping_rounds=early_stopping_rounds,
                                plot=True, logging_level=None, random_seed=random_seed)
print("tree_count: {} learning_rate {} depth {} use_best_model {}".format(
    clf.tree_count_, clf.learning_rate_, clf.get_all_params()['depth'], 
    clf.get_all_params()['use_best_model']))
name_imp = zip(variant.feature_names.get_list(), clf.get_feature_importance())
print("feat_importance:")
pprint.pprint(name_imp)
# pools = eval_into_train_pools(key)
# clf_cat_tools.cross_validation(clf, pools.train, metric_name, fold_count=5, stratified=True,
                                 early_stopping_rounds=early_stopping_rounds, plot=True)
# learn_utils.print_most_used_scores(clf, data.test.X, data.test.y)
# print str(clf_cat_tools.get_test_result(variant, clf, pools.test))
# clf_cat_tools.save_clf(clf, MODEL, key, metric_name, random_seed)
"""


def model_filename(model_name, key, metric_name, random_seed=None):
    dirname = cco.persist_dirname(model_name, key, "clf")
    suffix = "" if random_seed is None else "_" + str(random_seed)
    filename = "{}/clf_{}{}.model".format(dirname, metric_name, suffix)
    return filename


def load_clf(filename: str):
    clf = CatBoostClassifier()
    clf.load_model(filename)
    return clf


def save_clf(clf, filename: str):
    clf.save_model(filename)


class DataCatBoost(cco.Data):
    def __init__(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        X_test,
        y_test,
        cat_features_idx=None,
        w_train=None,
        w_eval=None,
        w_test=None,
    ):
        super().__init__(
            X_train, y_train, X_eval, y_eval, X_test, y_test, w_train, w_eval, w_test
        )
        self.cat_features = None if not cat_features_idx else cat_features_idx

    def exist_cat(self):
        return bool(self.cat_features)

    def tolist(self):
        cat_indices = [] if not self.cat_features else self.cat_features
        if self.train:
            self.train.tolist(cat_indices)
        if self.eval:
            self.eval.tolist(cat_indices)
        if self.test:
            self.test.tolist(cat_indices)


Pools = namedtuple("Pools", "train eval test")


def make_pools(data):
    """if exist cat_features then list it in cat_features_idx"""
    if data.train:
        train_pool = Pool(
            data.train.X,
            data.train.y,
            cat_features=data.cat_features,
            weight=data.train.w,
        )
    else:
        train_pool = None

    if data.eval:
        eval_pool = Pool(
            data.eval.X, data.eval.y, cat_features=data.cat_features, weight=data.eval.w
        )
    else:
        eval_pool = None

    if data.test:
        test_pool = Pool(
            data.test.X, data.test.y, cat_features=data.cat_features, weight=data.test.w
        )
    else:
        test_pool = None
    return Pools(train=train_pool, eval=eval_pool, test=test_pool)


def make_test_pool(data):
    """if exist cat_features then list it in cat_features_idx"""
    if data.test:
        return Pool(
            data.test.X, data.test.y, cat_features=data.cat_features, weight=data.test.w
        )


def grid_search(model, grid, pool: Pool):
    grid_search_result = model.grid_search(grid, pool)
    return grid_search_result


def train_model(
    pools: Pools, eval_set_admit: bool, metric_name: str, plot: bool, **kwargs
):
    """using train and eval pools.
    logging_level: 'Verbose'   metric_name: 'AUC', 'Precision', 'Accuracy', 'Logloss',...
    """
    assert isinstance(pools, Pools)
    print("train_model start with metric {},  {}".format(metric_name, kwargs))

    if metric_name == 'Logloss':
        params = {}
    elif metric_name != "AUC":
        params = {"custom_loss": [metric_name, "AUC"]}
    else:
        params = {"custom_loss": [metric_name]}
    params.update(kwargs)
    model = CatBoostClassifier(**params)
    if pools.eval and eval_set_admit:
        eval_set = pools.eval
    elif pools.test and eval_set_admit:
        eval_set = pools.test
    else:
        eval_set = None
    model.fit(
        pools.train, eval_set=eval_set, logging_level=params["logging_level"], plot=plot
    )
    return model


def cross_validation(
    clf,
    pool: Pool,
    metric_name,
    fold_count=5,
    stratified=True,
    early_stopping_rounds=None,
    plot=False,
):
    """metric_name: 'AUC', 'Precision', 'Accuracy'...  Output as sample:
    Best test Precision score: 0.72+-0.25 on step 22
    Best test AUC score: 0.57+-0.01 on step 23
    Best validation Logloss score: 0.69+-0.00 on step 0
    """
    assert isinstance(pool, Pool)
    print(f"cross_validation start with metric {metric_name} "
          f"fold_count {fold_count} stratified {stratified} "
          f"early_stopping {early_stopping_rounds}")
    cv_params = clf.get_params()
    cv_params.update({"loss_function": "Logloss"})
    cv_data = cv(
        pool,
        cv_params,
        fold_count=fold_count,
        stratified=stratified,
        plot=plot,
        early_stopping_rounds=early_stopping_rounds,
    )
    test_metric_pref = "test-" + metric_name
    print(
        "Best test {} score: {:.2f}+-{:.2f} on step {}".format(
            metric_name,
            np.max(cv_data[test_metric_pref + "-mean"]),
            cv_data[test_metric_pref + "-std"][
                np.argmax(cv_data[test_metric_pref + "-mean"])
            ],
            np.argmax(cv_data[test_metric_pref + "-mean"]),
        )
    )

    print(
        "Best validation Logloss score: {:.2f}+-{:.2f} on step {}".format(
            np.max(cv_data["test-Logloss-mean"]),
            cv_data["test-Logloss-std"][np.argmax(cv_data["test-Logloss-mean"])],
            np.argmax(cv_data["test-Logloss-mean"]),
        )
    )


def get_test_result(variant, clf, pool: Pool):
    probs = clf.predict_proba(pool)
    poswl, negwl = st.WinLoss(), st.WinLoss()
    min_pos_proba = variant.min_probas.pos
    min_neg_proba = variant.min_probas.neg
    for prob01, lab in zip(probs, pool.get_label()):
        if min_pos_proba is not None and prob01[1] >= min_pos_proba:
            poswl.hit(lab == 1)
        elif min_neg_proba is not None and prob01[0] >= min_neg_proba:
            negwl.hit(lab == 0)
    profit, pos_profit, neg_profit = 0.0, 0.0, 0.0
    profit_ratios = variant.profit_ratios
    if poswl:
        pos_profit = round(poswl.size * (poswl.ratio - profit_ratios.pos_ratio), 3)
    if negwl:
        neg_profit = round(negwl.size * (negwl.ratio - profit_ratios.neg_ratio), 3)
    profit = pos_profit + neg_profit
    return cco.Result(
        name=variant.name,
        mean=cco.fmt((poswl + negwl).ratio),
        leny=len(pool.get_label()),
        scr=cco.fmt(clf.score(pool)),
        poswl=poswl,
        negwl=negwl,
        profit=profit,
        pos_profit=pos_profit,
    )


def get_test_result_3class(variant, clf, pool: Pool):
    probs = clf.predict_proba(pool)
    poswl, negwl = st.WinLoss(), st.WinLoss()
    min_pos_proba = variant.min_probas.pos
    min_neg_proba = variant.min_probas.neg
    for prob0z1, lab in zip(probs, pool.get_label()):
        if min_pos_proba is not None and prob0z1[2] >= min_pos_proba:
            poswl.hit(lab == 1)
        elif min_neg_proba is not None and prob0z1[0] >= min_neg_proba:
            negwl.hit(lab == -1)
    profit, pos_profit, neg_profit = 0.0, 0.0, 0.0
    profit_ratios = variant.profit_ratios
    if poswl:
        pos_profit = round(poswl.size * (poswl.ratio - profit_ratios.pos_ratio), 3)
    if negwl:
        neg_profit = round(negwl.size * (negwl.ratio - profit_ratios.neg_ratio), 3)
    profit = pos_profit + neg_profit
    return cco.Result(
        name=variant.name,
        mean=cco.fmt((poswl + negwl).ratio),
        leny=len(pool.get_label()),
        scr=cco.fmt(clf.score(pool)),
        poswl=poswl,
        negwl=negwl,
        profit=profit,
        pos_profit=pos_profit,
    )

