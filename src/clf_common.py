import os
from collections import namedtuple, defaultdict
import copy
import random
import pickle
import unittest
from typing import Optional, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    recall_score,
)

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import common as co
import log
import file_utils as fu
import cfg_dir
import surf
import stat_cont as st


RANK_STD_BOTH_ABOVE = 400
RANK_STD_MAX_DIF = 280
DATASET_MIN_LEN = 300
NAN = "NaN"

CHRONO_TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
CHRONO_EVAL_YEARS = [2018, 2019]
CHRONO_TEST_YEARS = [2020, 2021]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


class IntRandomGen(object):
    """use IntRandomGen(None) or IntRandomGen(-1) for dummy stub"""

    def __init__(self, range_limit=10000000):
        self.range_limit = range_limit

    def __next__(self):
        if self.range_limit is None or self.range_limit < 0:
            return -1
        return random.randint(0, self.range_limit)


class FeatureError(co.TennisError):
    pass


class WeightMode(Enum):
    NO = 1
    BALANCED = 2  # by target label
    BY_TOTAL_COLUMN = 3


def rfe_include_exclude(clf, X, y, feature_names, n_features_to_select):
    select = RFE(clf, n_features_to_select=n_features_to_select)
    select.fit(X, y)
    mask = select.get_support()
    assert len(mask) == len(feature_names), "len_mask: {} len_features: {}".format(
        len(mask), len(feature_names)
    )
    include_names = [name for (name, sel) in zip(feature_names, mask) if sel]
    exclude_names = [name for (name, sel) in zip(feature_names, mask) if not sel]
    return include_names, exclude_names


def colsample_ratios(n_features):
    """use for hyperopt/grid search"""
    result = []
    beg = int(round(np.sqrt(n_features)))
    for num in range(beg, n_features + 1):
        result.append(float(num) / n_features)
    return result


def surface_onehot_encode(df):
    """make columns 'surface_hard', 'surface_clay', 'surface_carpet', 'surface_grass'
    with values 0 or 1
    """

    def process_row_hard(row):
        surface = row["surface_text"]
        return 1 if surface == "Hard" else 0

    def process_row_clay(row):
        surface = row["surface_text"]
        return 1 if surface == "Clay" else 0

    def process_row_carpet(row):
        surface = row["surface_text"]
        return 1 if surface == "Carpet" else 0

    def process_row_grass(row):
        surface = row["surface_text"]
        return 1 if surface == "Grass" else 0

    df["surface_hard"] = df.apply(process_row_hard, axis=1)
    df["surface_clay"] = df.apply(process_row_clay, axis=1)
    df["surface_carpet"] = df.apply(process_row_carpet, axis=1)
    df["surface_grass"] = df.apply(process_row_grass, axis=1)


def out(arg):
    if isinstance(arg, str):
        text = arg
    elif isinstance(arg, (pd.DataFrame, pd.Series)):
        text = f"\n{arg}"
    else:
        text = str(arg)
    if log.initialized():
        log.info(text)
    else:
        print(text)


_code_level = {0: "future", 1: "qual", 2: "chal", 3: "main", 4: "team", 5: "teamworld"}


def level_code(level):
    for code, _level in _code_level.items():
        if str(level) == _level:
            return code
    raise co.TennisError("unexpected level: " + str(level))


def get_level(code):
    return _code_level[code]


def surface_code(surface):
    return surf.get_code(str(surface))


def get_surface(code):
    return surf.get_name(code)


def drop_by_condition(df, expr_fun):
    """expr_fun is fun(df) -> expression (use as: df[expr] )"""
    idx_todel = df[expr_fun(df)].index
    df.drop(idx_todel, inplace=True)


def drop_rank_std_any_below_or_wide_dif(df, rank_std_both_above, rank_std_max_dif):
    drop_by_condition(
        df,
        lambda d: (
            (d["fst_std_rank"] > rank_std_both_above)
            | (d["snd_std_rank"] > rank_std_both_above)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > rank_std_max_dif)
        ),
    )


def smooth_bet_chance_column(df: pd.DataFrame, featname: str, factor):
    def process_row(row):
        return co.smooth_proba(row[featname], factor=factor)

    new_featname = "{0}_{1:.2f}".format(featname, factor).replace(".", "")
    df[new_featname] = df.apply(process_row, axis=1)


def stage_shuffle(df: pd.DataFrame, is_shuffle: bool = True, random_state=None):
    if is_shuffle:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def get_hash(df, key_names):
    lst = []
    for i in df.index:
        arr = df.loc[i, key_names].values
        lst.append(hash(tuple(arr)))
    return hash(tuple(lst))


class FeatureNames:
    def __init__(self, names: List[str], cat_names: Optional[List[str]] = None):
        self.names = names
        self.cat_names = [] if cat_names is None else cat_names
        if self.cat_names:
            assert set(self.cat_names).issubset(set(self.names))

    def __len__(self):
        return len(self.names)

    def get_list(self):
        return self.names

    def get_cat_list(self):
        return self.cat_names

    def exist_cat(self):
        return len(self.cat_names) > 0

    def cat_indexes(self):
        result = []
        for cat_name in self.cat_names:
            if cat_name in self.names:
                result.append(self.names.index(cat_name))
        return result


class LevSurf:
    def __init__(
        self,
        name: str,
        levels=None,
        surfaces=None,
        drop_cond=None,
        skip_match_cond=None,
        stratify_names=None,
    ):
        self.name = name
        self.levels = levels if levels is not None else []
        self.surfaces = surfaces if surfaces is not None else []
        self.drop_cond = drop_cond
        self.stratify_names = stratify_names
        self.skip_match_cond = skip_match_cond

    def is_levels(self):
        return len(self.levels) > 0

    def is_surfaces(self):
        return len(self.surfaces) > 0

    def surface_codes(self):
        return [surface_code(s) for s in self.surfaces]

    def __hash__(self):
        return hash((tuple(self.levels), tuple(self.surfaces)))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.levels == other.levels and self.surfaces == other.surfaces

    def __ne__(self, other):
        return not self.__eq__(other)

    def drop_by_condition(self, df: pd.DataFrame) -> None:
        if self.drop_cond is None:
            return
        drop_by_condition(df, self.drop_cond)

    def is_skip_match(self, match):
        return self.skip_match_cond and self.skip_match_cond(match)

    def get_stratify_names(self, is_text_style: bool):
        if self.stratify_names is None:
            return []
        return [((n + "_text") if is_text_style else n) for n in self.stratify_names]


q_rounds = ["q-First", "q-Second", "Qualifying"]


key_main_clr = LevSurf(
    name="main_clr",
    levels=["main", "gs", "masters"],
    surfaces=None,
    drop_cond=lambda d: (
        (d["level_text"].isin(["qual", "chal", "team", "teamworld", "future"]))
        | (d["rnd_text"] == "Rubber")
    ),
    skip_match_cond=lambda m: (
        m.level in ["qual", "chal", "team", "teamworld", "future"]
        or (m.rnd is not None and m.rnd.rubber())
    ),
    stratify_names=["surface", "rnd"],
)

key_main_clr_rtg350_250 = LevSurf(
    name="main_clr_rtg350_250",
    levels=["main", "gs", "masters"],
    surfaces=None,
    drop_cond=lambda d: (
        (d["level_text"].isin(["qual", "chal", "team", "teamworld", "future"]))
        | (d["rnd_text"] == "Rubber")
        | (d["fst_std_rank"] > 350)
        | (d["snd_std_rank"] > 350)
        | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > 250)
    ),
    skip_match_cond=lambda m: (
        m.level in ["qual", "chal", "team", "teamworld", "future"]
        or (m.rnd is not None and m.rnd.rubber())
    ),
    stratify_names=["surface", "rnd"],
)

key_main_clr_rtg200_150 = LevSurf(
    name="main_clr_rtg200_150",
    levels=["main", "gs", "masters"],
    surfaces=None,
    drop_cond=lambda d: (
        (d["level_text"].isin(["qual", "chal", "team", "teamworld", "future"]))
        | (d["rnd_text"] == "Rubber")
        | (d["fst_std_rank"] > 200)
        | (d["snd_std_rank"] > 200)
        | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > 150)
    ),
    skip_match_cond=lambda m: (
        m.level in ["qual", "chal", "team", "teamworld", "future"]
        or (m.rnd is not None and m.rnd.rubber())
    ),
    stratify_names=["surface", "rnd"],
)

ProfitRatios = namedtuple("ProfitRatios", "pos_ratio neg_ratio")
PosNeg = namedtuple("PosNeg", "pos neg")


class Variant:
    def __init__(
        self,
        sex: str,
        suffix: str,
        key: LevSurf,
        cls,
        clf_pars,
        min_probas: PosNeg,
        profit_ratios: Tuple[float, float],
        feature_names: FeatureNames,
        weight_mode=WeightMode.NO,
        calibrate=None,
        scaler=None,
        pca=None,
    ):
        self._sex = sex
        self.suffix = suffix
        self.key = key
        self.cls = cls
        self.clf_pars = clf_pars  # dict: param_name -> param_value
        self.min_probas = min_probas
        self.profit_ratios = ProfitRatios(
            pos_ratio=profit_ratios[0], neg_ratio=profit_ratios[1]
        )
        self.feature_names = feature_names
        self.weight_mode = weight_mode
        self.calibrate = calibrate
        self.scaler = scaler
        self.clf = None
        self.pca = pca
        assert isinstance(self.calibrate, str), "calibrate must be string"

    @property
    def name(self):
        return f"{self.sex}_{self.key.name}_{self.suffix}"

    def __str__(self):
        return (
            f"{self.name} {self.clf_pars} " f"P:{self.min_probas}"
        )

    def is_cb_native(self):
        return self.cls == CatBoostClassifier

    def is_rf(self):
        return self.cls == RandomForestClassifier

    def is_gb(self):
        return self.cls == GradientBoostingClassifier

    @property
    def sex(self):
        return self._sex

    def get_profit_ratios(self):
        return self.profit_ratios

    def set_random_state(self, random_state):
        if self.is_cb_native():
            self.clf_pars["random_seed"] = random_state
        else:
            self.clf_pars["random_state"] = random_state

    def choice_random_state(self):
        self.set_random_state(random.randint(0, 10000000))

    def make_clf(self, random_state=-1):
        if random_state >= 0:
            # we got extern random_state which overwrite self random_state (if exist)
            pars = copy.copy(self.clf_pars)
            random_key = "random_seed" if self.is_cb_native() else "random_state"
            pars[random_key] = random_state
        else:
            pars = self.clf_pars
        if self.calibrate:
            clf = CalibratedClassifierCV(self.cls(**pars), method=self.calibrate, cv=5)
        else:
            clf = self.cls(**pars)
        if self.scaler is not None:
            if self.pca is not None:
                self.clf = make_pipeline(
                    copy.copy(self.scaler), copy.copy(self.pca), clf
                )
            else:
                self.clf = make_pipeline(copy.copy(self.scaler), clf)
        else:
            self.clf = clf
        return self.clf

    def make_clf_fit(self, data, metric_name: str, random_seed: int, plot=False):
        import clf_cat_tools

        self.clf = None
        if self.is_cb_native():
            pools = clf_cat_tools.make_pools(data)
            clf = clf_cat_tools.train_model(
                pools,
                eval_set_admit=True,
                metric_name=metric_name,
                plot=plot,
                **self.clf_pars,
            )
            self.clf = clf
        else:
            clf = self.make_clf(random_state=random_seed)
            clf.fit(data.train.X, data.train.y)
        return clf

    def predict_proba(self, X_test):
        if self.clf is not None:
            return self.clf.predict_proba(X_test)

    def persist_dirname(self, model_name, subname=None):
        pers_dirname = os.path.join(cfg_dir.analysis_data_dir(), "persist")
        if subname:
            return os.path.join(pers_dirname, f"{model_name}_{self.name}", subname)
        return os.path.join(pers_dirname, f"{model_name}_{self.name}")

    def save_clf(self, model_name, metric_name):
        if self.is_cb_native():
            dname = self.persist_dirname(model_name, subname="clf")
            fname = os.path.join(dname, f"clf_{metric_name}.model")
            self.clf.save_model(fname)
        else:
            raise NotImplementedError(f"save_clf {self}")

    def load_clf(self, model_name, metric_name):
        if self.is_cb_native():
            dname = self.persist_dirname(model_name, subname="clf")
            fname = os.path.join(dname, f"clf_{metric_name}.model")
            self.clf = CatBoostClassifier()
            self.clf.load_model(fname)
        else:
            raise NotImplementedError(f"save_clf {self}")

    def get_features_importance(self, feature_names: List[str], data=None):
        from clf_cat_tools import make_pools

        if self.is_cb_native():
            if data is not None:
                pools = make_pools(data)
                return dict(
                    zip(
                        feature_names,
                        self.clf.get_feature_importance(
                            data=pools.train, type="LossFunctionChange"
                        ),
                    )
                )
            else:
                return dict(zip(feature_names, self.clf.get_feature_importance()))
        return dict(zip(feature_names, self.clf.feature_importances_))

    def make_test_result(self, data):
        import clf_cat_tools

        if self.is_cb_native():
            return clf_cat_tools.get_test_result(
                self, self.clf, clf_cat_tools.make_test_pool(data)
            )
        return get_result(self, self.clf, X_test=data.test.X, y_test=data.test.y)

    def drop_by_condition(self, df: pd.DataFrame) -> None:
        self.key.drop_by_condition(df)

    def is_skip_match(self, match):
        return self.key.is_skip_match(match)

    @staticmethod
    def find_match_variant(match, variants: List):
        results = []
        for variant in variants:
            if variant.sex != match.sex:
                continue
            if variant.key.levels and match.level not in variant.key.levels:
                continue
            if variant.key.surfaces and match.surface not in variant.key.surfaces:
                continue
            if variant.is_skip_match(match):
                continue
            results.append(variant)
        return results[0] if len(results) == 1 else None

    @staticmethod
    def find_variant_by(variants: List, predicate):
        results = [v for v in variants if predicate(v)]
        return results[0] if len(results) == 1 else None


_sex_surface_service_win = {
    # mean srvwin for softkey == 'main' plrs rating 250 and higher
    ("wta", "Clay"): 0.555,
    ("wta", "Hard"): 0.561,
    ("wta", "Carpet"): 0.572,
    ("wta", "Grass"): 0.591,
    ("atp", "Clay"): 0.627,
    ("atp", "Hard"): 0.638,
    ("atp", "Carpet"): 0.643,
    ("atp", "Grass"): 0.662,
}


def mean_service_win(sex, surface):
    r"""from stat\{sex}\misc\matchstat\service_win_softmain.txt"""
    result = _sex_surface_service_win.get((sex, surface))
    if result is None:
        raise FeatureError(
            "fail mean_service_win sex: {} surf: {}".format(sex, surface)
        )
    return result


rnd_name_norm_value = {
    "Pre-q": 0.0,
    "q-First": 0.05,
    "q-Second": 0.12,
    "Qualifying": 0.2,
    "First": 0.3,
    "Second": 0.4,
    "Third": 0.5,
    "Robin": 0.5,
    "Fourth": 0.56,
    "1/4": 0.62,
    "1/2": 0.77,
    "Bronze": 0.8,
    "Final": 1.0,
    "Rubber 1": 0.31,
    "Rubber 2": 0.32,
    "Rubber 3": 0.33,
    "Rubber 4": 0.34,
    "Rubber 5": 0.35,
}


_rnd_code = {
    "Pre-q": 0,
    "q-First": 1,
    "q-Second": 2,
    "Qualifying": 3,
    "First": 4,
    "Second": 5,
    "Robin": 7,
    "Third": 8,
    "Fourth": 9,
    "1/4": 10,
    "1/2": 11,
    "Bronze": 11,
    "Final": 12,
    "Rubber": 6,
    "Rubber 1": 6,
    "Rubber 2": 6,
    "Rubber 3": 6,
    "Rubber 4": 6,
    "Rubber 5": 6,
}


def rnd_norm_value(rnd):
    if isinstance(rnd, str):
        return rnd_name_norm_value[rnd]
    else:
        return rnd_name_norm_value[rnd.value]


def rnd_code(rnd_text):
    return _rnd_code[rnd_text]


def rnd_stage(rnd_text, raw_level_text):
    if rnd_text == "q-First":
        return -2
    if rnd_text == "q-Second":
        return -1
    if rnd_text == "Qualifying":
        return 0
    if rnd_text == "First":
        return 1
    if rnd_text == "Second":
        return 2
    if rnd_text == "Third":
        return 3
    if rnd_text == "Robin":
        return 3
    if rnd_text == "Fourth":
        return 4
    if rnd_text == "1/4":
        if raw_level_text == "main":
            return 3
        if raw_level_text == "masters":
            return 4
        if raw_level_text == "gs":
            return 5
        return 4
    if rnd_text == "1/2":
        if raw_level_text == "main":
            return 4
        if raw_level_text == "masters":
            return 5
        if raw_level_text == "gs":
            return 6
        return 5
    if rnd_text == "Final":
        if raw_level_text == "main":
            return 5
        if raw_level_text == "masters":
            return 6
        if raw_level_text == "gs":
            return 7
        return 6
    raise ValueError(f"invalid rnd_text:'{rnd_text}' raw_level_text:'{raw_level_text}'")


def rnd_money_stage(rnd_text, raw_level_text, tour_money):
    def stages_count():
        if raw_level_text == "gs":
            return 7
        if raw_level_text == "masters":
            return 6
        return 5

    if rnd_text == "q-First":
        return 0.5
    if rnd_text == "q-Second":
        return 0.7
    if rnd_text == "Qualifying":
        return 1.0

    n_stages = stages_count()
    money_per_stage = int(round((2 * tour_money) / (n_stages * (n_stages + 1))))
    i_stage = rnd_stage(rnd_text, raw_level_text)
    return i_stage * money_per_stage


def extract_rnd_text(rnd):
    if isinstance(rnd, str):
        value = rnd
    else:
        value = rnd.value
    return unified_rnd_value(value)


def unified_rnd_value(value):
    if value.startswith("Rubber"):
        return "Rubber"
    elif value == "Bronze":
        return "Final"
    return value


MATCH_SCOPE_COEF = 2.2  # comparing with set scope miss points
SETSRV_NOHOLD_COEF = 3.0


def miss_point_weight(sex, scope, on_serve, level, surface):
    from matchstat import generic_result_value

    if scope == co.MATCH:
        return MATCH_SCOPE_COEF * miss_point_weight(
            sex, co.SET, on_serve, level, surface
        )
    if not on_serve:
        return 1.0  # as base
    else:
        srv_win_ratio = generic_result_value(sex, "service_win", level, surface)
        assert (
            srv_win_ratio is not None
        ), "none service_win for {} lev:{} surf:{}".format(sex, level, surface)
        srv_weight = srv_win_ratio / (1.0 - srv_win_ratio)  # prob_srv / prob_rcv
        return srv_weight


def persist_dirname(model_name, key, subname=None):
    pers_dirname = os.path.join(cfg_dir.analysis_data_dir(), "persist")
    if subname:  # 'test' or 'eval'
        return os.path.join(pers_dirname, "{}_{}".format(key, model_name), subname)
    return os.path.join(pers_dirname, "{}_{}".format(key, model_name))


def save_df(df, storage_dir, subname):
    filename = os.path.join(storage_dir, subname, "df.csv")
    df.to_csv(filename, index=False)


def load_df(storage_dir, subname):
    if storage_dir:
        filename = os.path.join(storage_dir, subname, "df.csv")
        if os.path.isfile(filename):
            return pd.read_csv(filename, sep=",")


def load_clf(model_name, key):
    dirname = persist_dirname(model_name, key, "clf")
    filename = os.path.join(dirname, "clf.pkl")
    if os.path.isfile(filename):
        with open(filename, "r") as fh:
            return pickle.load(fh)


def save_clf(clf, model_name, key):
    dirname = persist_dirname(model_name, key, "clf")
    filename = os.path.join(dirname, "clf.pkl")
    if os.path.isfile(filename):
        fu.remove_file(filename)
    with open(filename, "w") as fh:
        pickle.dump(clf, fh)


def substract_df(df, df_substr, inplace=True):
    """:return df where removed row (matches) from df_substr"""
    key_names = ["date", "fst_pid", "snd_pid", "rnd_text"]
    substr_keys = list(map(tuple, df_substr[key_names].values))
    del_idxes = [
        i
        for i in df.index
        if (
            df.loc[i, "date"],
            df.loc[i, "fst_pid"],
            df.loc[i, "snd_pid"],
            df.loc[i, "rnd_text"],
        )
        in substr_keys
    ]
    if inplace:
        df.drop(del_idxes, inplace=True)
    else:
        return df.drop(del_idxes, inplace=False)


class DataSet(object):
    def __init__(self, X, y, w=None):
        self.X = X
        self.y = y
        self.w = w

    def __bool__(self):
        return self.X is not None and self.y is not None

    __nonzero__ = __bool__


class Data(object):
    def __init__(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        X_test,
        y_test,
        w_train=None,
        w_eval=None,
        w_test=None,
    ):
        self.train = DataSet(X=X_train, y=y_train, w=w_train)
        self.eval = DataSet(
            X=X_eval, y=y_eval, w=w_eval
        )  # for model choice and model tuning
        self.test = DataSet(
            X=X_test, y=y_test, w=w_test
        )  # final model testing (after fit train)

    def num_features(self):
        if self.train.X is not None:
            return self.train.X.shape[1]
        if self.eval.X is not None:
            return self.eval.X.shape[1]
        if self.test.X is not None:
            return self.test.X.shape[1]

    def is_eval(self):
        return bool(self.eval)

    def is_test(self):
        return bool(self.test)

    @property
    def X_train(self):
        return self.train.X

    @property
    def y_train(self):
        return self.train.y

    @property
    def X_test2(self):
        """for obsolete use"""
        return self.test.X

    @property
    def y_test2(self):
        """for obsolete use"""
        return self.test.y

    @property
    def X_test(self):
        """for obsolete use"""
        return self.eval.X

    @property
    def y_test(self):
        """for obsolete use"""
        return self.eval.y


class OutResult(object):
    def __init__(self, side, proba, note=""):
        self.side = side
        self.proba = proba
        self.note = note


class Result(object):
    """for tests and experiments"""

    def __init__(self, name, mean, leny, scr, poswl, negwl, profit, pos_profit=None):
        self.name = name
        self.mean = mean
        self.leny = leny
        self.scr = scr
        self.poswl = poswl
        self.negwl = negwl
        self.profit = profit
        self.pos_profit = pos_profit

    def hit_ratio(self, winloss):
        if self.leny:
            return float(winloss.size) / float(self.leny)

    def __str__(self):
        def hit_ratio_text(winloss):
            hit_ratio = self.hit_ratio(winloss)
            return "" if hit_ratio is None else "{:.2f}".format(hit_ratio)

        def pos_profit_text():
            if self.pos_profit is None:
                return ""
            return " pos_prof {:.3f}".format(self.pos_profit)

        profit_text = "" if self.profit is None else " prof {:.3f}".format(self.profit)
        result = "{} mean: {} scr: {} pos: {}{} phit: {}\n\tneg: {} nhit: {}{}".format(
            self.name,
            self.mean,
            self.scr if self.scr else "",
            self.poswl,
            pos_profit_text(),
            hit_ratio_text(self.poswl),
            self.negwl,
            hit_ratio_text(self.negwl),
            profit_text,
        )
        return result


def get_result(variant, clf, X_test, y_test):
    probs = clf.predict_proba(X_test)
    poswl, negwl = st.WinLoss(), st.WinLoss()
    min_pos_proba = variant.min_probas.pos
    min_neg_proba = variant.min_probas.neg
    for prob01, lab in zip(probs, y_test):
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
    return Result(
        name=variant.name,
        mean=fmt((poswl + negwl).ratio),
        leny=len(y_test),
        scr=fmt(clf.score(X_test, y_test)),
        poswl=poswl,
        negwl=negwl,
        profit=profit,
        pos_profit=pos_profit,
    )


def get_test_with_groupby(
    variant, clf, df, fnames, label_name, groupby_name, groupby_key_fun=lambda v: v
):
    def result_list(dct):
        key_wl_lst = list(dct.items())
        key_wl_lst.sort(key=lambda item: item[1].size, reverse=True)
        return [(k, str(wl)) for k, wl in key_wl_lst]

    pos_dct, neg_dct = defaultdict(st.WinLoss), defaultdict(st.WinLoss)
    for i in df.index:
        X_test = df.loc[i, fnames].values  # ndarray
        proba = clf.predict_proba(X_test)
        y_test = df.loc[i, label_name]
        groupby_name_val = df.loc[i, groupby_name]
        groupby_key = groupby_key_fun(groupby_name_val)
        positive_prob = proba[1]
        negative_prob = proba[0]
        if positive_prob >= variant.min_probas.pos:
            pos_dct[groupby_key].hit(y_test > 0.5)  # win if label 1
        elif negative_prob >= variant.min_probas.neg:
            neg_dct[groupby_key].hit(y_test < 0.5)  # win if label 0
    return result_list(pos_dct), result_list(neg_dct)


def series_bin_ratio(series, pos_value=1, neg_value=0):
    """if pandas series has 0, 1 values, then return n1 / (n1 + n0)"""
    if isinstance(series, pd.Series):
        val_counts = series.value_counts()
    else:
        val_counts = pd.Series(series).value_counts()
    if pos_value in val_counts and neg_value in val_counts:
        wl = st.WinLoss(val_counts[pos_value], val_counts[neg_value])
        return wl.ratio
    elif pos_value in val_counts:
        wl = st.WinLoss(val_counts[pos_value], 0)
        return wl.ratio
    elif neg_value in val_counts:
        wl = st.WinLoss(0, val_counts[neg_value])
        return wl.ratio


def series_three_ratio(series, pos_value=1, zero_value=0, neg_value=-1):
    if isinstance(series, pd.Series):
        val_counts = series.value_counts()
    else:
        val_counts = pd.Series(series).value_counts()

    pos_cnt = val_counts[pos_value] if pos_value in val_counts else 0
    zero_cnt = val_counts[zero_value] if zero_value in val_counts else 0
    neg_cnt = val_counts[neg_value] if neg_value in val_counts else 0
    all_size = pos_cnt + zero_cnt + neg_cnt
    if all_size > 0:
        return (
            float(val_counts[pos_value]) / all_size,
            float(val_counts[zero_value]) / all_size,
            float(val_counts[neg_value]) / all_size,
        )


def fmt(val):
    if pd.isnull(val):
        return ""
    if isinstance(val, float):
        return "{:.3f}".format(val)


def fmt_list(val):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return [fmt(v) for v in val]


def scores_text(y_test, y_pred):
    results = []
    for name, scr in (
        ("accuracy", accuracy_score),
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
    ):
        results.append("{}: {}".format(name, fmt(scr(y_test, y_pred))))
    return ",".join(results)


def features_imp_mix_corr(fnames, imp_values, corr_values, imp_coef=0.6):
    assert 0 <= imp_coef <= 1, "bad imp_coef"
    return [
        (n, abs(i) * imp_coef + abs(c) * (1.0 - imp_coef))
        for n, i, c in zip(fnames, imp_values, corr_values)
    ]


def out_corr_fnames_label(
    df: pd.DataFrame,
    feat_names: List[str],
    label_name: str,
    df_test: Optional[pd.DataFrame] = None,
):
    if df_test is None:
        out(df[feat_names].corrwith(df[label_name]))
    df_minus_test = substract_df(df, df_test, inplace=False)
    out(df_minus_test[feat_names].corrwith(df_minus_test[label_name]))


def plot_prec_recall_check(
    variant, data, metric_name: str, pos_proba=None, random_seed: int = 0
):
    """from mueller"""
    X_test, y_test = data.eval.X, data.eval.y
    if X_test is None or y_test is None:
        X_test, y_test = data.test.X, data.test.y
        if X_test is None or y_test is None:
            print("X_eval y_eval (or X_test, y_test) must be prepared")
            return
    clf = variant.make_clf_fit(
        data=data, metric_name=metric_name, random_seed=random_seed
    )
    pred_pos_proba = clf.predict_proba(X_test)[:, 1]

    avgprec = average_precision_score(y_test, pred_pos_proba)
    print("Average precision: {:.3f}".format(avgprec))

    precision_fr, recall_fr, thresholds_fr = precision_recall_curve(
        y_test, pred_pos_proba, pos_label=1
    )
    plt.plot(precision_fr, recall_fr, label="fr")
    if pos_proba is None:
        pos_proba = variant.min_probas.pos
    close_default_fr = np.argmin(np.abs(thresholds_fr - pos_proba))
    plt.plot(
        precision_fr[close_default_fr],
        recall_fr[close_default_fr],
        "^",
        c="k",
        markersize=10,
        label="threshold {:.2f} fr".format(pos_proba),
        fillstyle="none",
        mew=2,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc=2)


if __name__ == "__main__":
    log.initialize(co.logname(__file__), file_level="debug", console_level="debug")
    unittest.main()


def get_xy(df, feature_names, label_name):
    x = df[feature_names].values
    y = df[label_name].values
    return x, y


def get_frame_columns(feature_names, svc_names, label_name, stratify2_name):
    columns = svc_names + feature_names + [label_name]
    if stratify2_name and stratify2_name not in columns:
        columns.append(stratify2_name)
    return columns


def split2_stratified(
    df: pd.DataFrame, target_names: List[str], test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    multi_lbl_arr = df[target_names]
    i_train, i_test = next(msss.split(np.zeros(df.shape[0]), multi_lbl_arr))
    idx_train = [df.index[i] for i in i_train]
    idx_test = [df.index[i] for i in i_test]
    return (copy.deepcopy(df.loc[idx_train, :]), copy.deepcopy(df.loc[idx_test, :]))


def split2_by_year(df: pd.DataFrame, test_years) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[~(df["year"].isin(test_years))]
    df_test = df[df["year"].isin(test_years)]
    return df_train, df_test


Splited = namedtuple("Splited", "train eval test")


def splited_by_year(df: pd.DataFrame, split) -> Splited:
    if split:
        df_train = df[df["year"].isin(CHRONO_TRAIN_YEARS)]
        df_test = df[df["year"].isin(CHRONO_TEST_YEARS)]
        df_eval = df[df["year"].isin(CHRONO_EVAL_YEARS)]
    elif split is False:
        df_train = df[df["year"].isin(CHRONO_TRAIN_YEARS)]
        df_test = df[df["year"].isin(CHRONO_EVAL_YEARS)]
        df_eval = None
    elif split is None:
        df_train = df
        df_test = None
        df_eval = None
    else:
        raise ValueError(f"invalid split: {split}")
    return Splited(train=df_train, eval=df_eval, test=df_test)


def split_dataframe_stratify_n(
    df, split, test_size, eval_size, target_names, random_state=None
):
    assert isinstance(target_names, list) and len(target_names) >= 1
    if split is None:
        return Splited(train=df, eval=None, test=None)
    elif split is False:
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        multi_lbl_arr = df[target_names].values
        i_train, i_test = next(msss.split(np.zeros(df.shape[0]), multi_lbl_arr))
        idx_train = [df.index[i] for i in i_train]
        idx_test = [df.index[i] for i in i_test]
        df_test = df.loc[idx_test, :]
        df_train = df.loc[idx_train, :]
        return Splited(train=df_train, eval=None, test=df_test)
    elif split is True:
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        multi_lbl_arr = df[target_names].values
        i_train_eval, i_test = next(
            msss.split(np.zeros(df.shape[0]), multi_lbl_arr)
        )
        idx_train_eval = [df.index[i] for i in i_train_eval]
        idx_test = [df.index[i] for i in i_test]
        df_test = df.loc[idx_test, :]

        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=eval_size, random_state=random_state
        )
        multi_lbl_arr2 = df.loc[idx_train_eval, target_names].values
        i_train, i_eval = next(
            msss2.split(np.zeros(df.shape[0] - len(i_test)), multi_lbl_arr2)
        )
        idx_train = [idx_train_eval[i] for i in i_train]
        idx_eval = [idx_train_eval[i] for i in i_eval]
        df_eval = df.loc[idx_eval, :]
        df_train = df.loc[idx_train, :]
        return Splited(train=df_train, eval=df_eval, test=df_test)


def make_data(
    cat_features_idx,
    X_train,
    X_eval,
    X_test,
    y_train,
    y_eval,
    y_test,
    w_train=None,
    w_eval=None,
    w_test=None,
):
    from clf_cat_tools import DataCatBoost

    if cat_features_idx is not None:
        return DataCatBoost(
            X_train=X_train,
            X_eval=X_eval,
            X_test=X_test,
            y_train=y_train,
            y_eval=y_eval,
            y_test=y_test,
            cat_features_idx=cat_features_idx,
            w_train=w_train,
            w_eval=w_eval,
            w_test=w_test,
        )
    return Data(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        X_test=X_test,
        y_test=y_test,
        w_train=w_train,
        w_eval=w_eval,
        w_test=w_test,
    )


def weighted_splited(spl: Splited, weight_mode: WeightMode, label_name: str):
    from clf_columns import add_weight_column

    if spl.train is not None:
        add_weight_column(weight_mode, spl.train, label_name)
    if spl.eval is not None:
        add_weight_column(weight_mode, spl.eval, label_name)
    if spl.test is not None:
        add_weight_column(weight_mode, spl.test, label_name)


def fill_data_ending_stratify_n(
    df,
    split,
    test_size,
    eval_size,
    feature_names,
    label_name,
    other_names,
    cat_features_idx,
    weight_mode=WeightMode.NO,
    random_state=None,
):
    assert isinstance(other_names, list)
    df_spl = split_dataframe_stratify_n(
        df,
        split,
        test_size,
        eval_size,
        [label_name] + other_names,
        random_state=random_state,
    )
    if weight_mode != WeightMode.NO:
        weighted_splited(df_spl, weight_mode, label_name)
    if split is True:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        X_test, y_test = get_xy(df_spl.test, feature_names, label_name)
        X_eval, y_eval = get_xy(df_spl.eval, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train,
                    X_eval,
                    X_test,
                    y_train,
                    y_eval,
                    y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=df_spl.eval["weight"].values,
                    w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx, X_train, X_eval, X_test, y_train, y_eval, y_test
            ),
            df_spl,
        )
    elif split is False:  # eval is empty
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        X_test, y_test = get_xy(df_spl.test, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train,
                    X_eval=None,
                    X_test=X_test,
                    y_train=y_train,
                    y_eval=None,
                    y_test=y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=None,
                    w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train,
                X_eval=None,
                X_test=X_test,
                y_train=y_train,
                y_eval=None,
                y_test=y_test,
            ),
            df_spl,
        )
    elif split is None:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train,
                    X_eval=None,
                    X_test=None,
                    y_train=y_train,
                    y_eval=None,
                    y_test=None,
                    w_train=df_spl.train["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train,
                X_eval=None,
                X_test=None,
                y_train=y_train,
                y_eval=None,
                y_test=None,
            ),
            df_spl,
        )
    else:
        raise co.TennisError("invalid split value {}".format(split))


def fill_data_ending_chrono(
    df,
    split,
    feature_names,
    label_name,
    other_names,
    cat_features_idx,
    weight_mode=WeightMode.NO,
):
    assert isinstance(other_names, list)
    df_spl = splited_by_year(df, split)
    if weight_mode != WeightMode.NO:
        weighted_splited(df_spl, weight_mode, label_name)
    if split is True:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        X_test, y_test = get_xy(df_spl.test, feature_names, label_name)
        X_eval, y_eval = get_xy(df_spl.eval, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train,
                    X_eval,
                    X_test,
                    y_train,
                    y_eval,
                    y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=df_spl.eval["weight"].values,
                    w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx, X_train, X_eval, X_test, y_train, y_eval, y_test
            ),
            df_spl,
        )
    elif split is False:  # eval is empty
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        X_test, y_test = get_xy(df_spl.test, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train,
                    X_eval=None,
                    X_test=X_test,
                    y_train=y_train,
                    y_eval=None,
                    y_test=y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=None,
                    w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train,
                X_eval=None,
                X_test=X_test,
                y_train=y_train,
                y_eval=None,
                y_test=y_test,
            ),
            df_spl,
        )
    elif split is None:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train,
                    X_eval=None,
                    X_test=None,
                    y_train=y_train,
                    y_eval=None,
                    y_test=None,
                    w_train=df_spl.train["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train,
                X_eval=None,
                X_test=None,
                y_train=y_train,
                y_eval=None,
                y_test=None,
            ),
            df_spl,
        )
    else:
        raise co.TennisError("invalid split value {}".format(split))


def make_clf(cls, pars_dict, scaler=None, pca=None):
    clf = cls(**pars_dict)
    if scaler is not None:
        if pca is not None:
            return make_pipeline(copy.copy(scaler), copy.copy(pca), clf)
        return make_pipeline(copy.copy(scaler), clf)
    return clf
