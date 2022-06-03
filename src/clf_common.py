import os
from collections import namedtuple, defaultdict
import copy
import random
import pickle
from typing import Optional, List, Tuple, Union, Dict
from enum import Enum

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

from sklearn.pipeline import make_pipeline
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
from loguru import logger as log
import lev
import file_utils as fu
import cfg_dir
import surf
import stat_cont as st
from df_util import drop_by_condition, save_df, load_df, substract_df

RANK_STD_BOTH_ABOVE = 500
RANK_STD_MAX_DIF = 300
DATASET_MIN_LEN = 25
NAN = "NaN"


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


class IntRandomGen:
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


def colsample_ratios(n_features):
    """use for hyperopt/grid search"""
    result = []
    beg = int(round(np.sqrt(n_features)))
    for num in range(beg, n_features + 1):
        result.append(float(num) / n_features)
    return result


def out(arg):
    if isinstance(arg, str):
        text = arg
    elif isinstance(arg, (pd.DataFrame, pd.Series)):
        text = f"\n{arg}"
    else:
        text = str(arg)
    log.info(text)


_dogfav_limit_chances_dict = {
    # (sex, exist_chal):  (max_dog_chance, min_fav_chance)
    ('wta', False): (0.46, 0.54),
    ('wta', True): (0.47, 0.53),
    #
    ('atp', False): (0.46, 0.54),
    ('atp', True): (0.46, 0.54),
}


def exist_dogfav(match):
    max_dog_chance, min_fav_chance = dogfav_limit_chances(match.sex, match.level == 'chal')
    fst_plr_bet_chance = match.first_player_bet_chance()
    if fst_plr_bet_chance is None:
        return None
    return not (max_dog_chance <= fst_plr_bet_chance <= min_fav_chance)


def dogfav_limit_chances(sex: str, exist_chal: bool):
    """ return (max_dog_chance, min_fav_chance) """
    return _dogfav_limit_chances_dict[(sex, exist_chal)]


def smooth_bet_chance_column(df: pd.DataFrame, featname: str, factor):
    def process_row(row):
        return co.smooth_proba(row[featname], factor=factor)

    new_featname = "{0}_{1:.2f}".format(featname, factor).replace(".", "")
    df[new_featname] = df.apply(process_row, axis=1)


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
    def __init__(self, name: str, levels=None, surfaces=None, drop_cond=None,
                 skip_match_cond=None, stratify_names=None):
        self.name = name
        self.levels = levels if levels is not None else []
        self.surfaces = surfaces if surfaces is not None else []
        self.drop_cond = drop_cond
        self.stratify_names = stratify_names
        self.skip_match_cond = skip_match_cond

    def is_levels(self):
        return len(self.levels) > 0

    def exist_chal(self):
        return self.levels is not None and lev.chal in self.levels

    def is_surfaces(self):
        return len(self.surfaces) > 0

    def surface_codes(self):
        return [surf.get_code(s) for s in self.surfaces]

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

    def get_stratify_names(self):
        if self.stratify_names is None:
            return []
        return self.stratify_names
        # return [((n + "_text") if is_text_style else n) for n in self.stratify_names]


q_rounds = ["q-First", "q-Second", "Qualifying"]
q_beg_rounds = ["q-First", "q-Second"]


def make_key_chal(max_std_rank, max_dif_std_rank, surfaces=None):
    def drop_condition(d):
        return (
            (d["level_text"].isin(["qual", "main", "team", "teamworld", "future"]))
            | d["rnd_text"].isin(q_rounds)
            | (d["fst_std_rank"] > max_std_rank)
            | (d["snd_std_rank"] > max_std_rank)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > max_dif_std_rank)
        )

    def drop_condition_surf(d):
        return (
            (d["level_text"].isin(["qual", "main", "team", "teamworld", "future"]))
            | d["rnd_text"].isin(q_rounds)
            | (d["fst_std_rank"] > max_std_rank)
            | (d["snd_std_rank"] > max_std_rank)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > max_dif_std_rank)
            | (~(d["surface_text"].isin(surf_list)))
        )

    surf_list = [] if not surfaces else [str(s) for s in surfaces]
    surf_str = '_'.join(surf_list)
    return LevSurf(
        name=f"chal_{(surf_str + '_') if surf_str else ''}"
             f"rtg{max_std_rank}_{max_dif_std_rank}",
        levels=["chal"],
        surfaces=None if not surf_list else surf_list,
        drop_cond=drop_condition_surf if surf_list else drop_condition,
        skip_match_cond=(
            lambda m: m.level != "chal"
            or (m.rnd is not None and m.rnd.qualification())
            or (m.rnd is not None and m.rnd.rubber())
            or (surf_list and m.surface not in surf_list)
        ),
        stratify_names=["rnd"] if len(surf_list) == 1 else ["surface", "rnd"],
    )


# main_clr имелось ввиду 'очищенный' main
def make_key_main_clr(max_std_rank, max_dif_std_rank, surfaces=None,
                      rnd_stratify_name="rnd"):
    def drop_condition(d):
        return (
            (~(d["raw_level_text"].isin(["main", "gs", "masters"])))
            | (d["rnd_text"] == "Rubber")
            | ((d["rnd_text"] == "Qualifying") & (d["raw_level_text"] == "main"))
            | ((d["rnd_text"].isin(q_beg_rounds)) & (d["raw_level_text"] != "masters"))
            | (d["fst_std_rank"] > max_std_rank)
            | (d["snd_std_rank"] > max_std_rank)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > max_dif_std_rank)
        )

    def drop_condition_surf(d):
        return (
            (~(d["raw_level_text"].isin(["main", "gs", "masters"])))
            | (d["rnd_text"] == "Rubber")
            | ((d["rnd_text"] == "Qualifying") & (d["raw_level_text"] == "main"))
            | ((d["rnd_text"].isin(q_beg_rounds)) & (d["raw_level_text"] != "masters"))
            | (d["fst_std_rank"] > max_std_rank)
            | (d["snd_std_rank"] > max_std_rank)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > max_dif_std_rank)
            | (~(d["surface_text"].isin(surf_list)))
        )

    surf_list = [] if not surfaces else [str(s) for s in surfaces]
    surf_str = '_'.join(surf_list)
    return LevSurf(
        name=f"main_clr_{(surf_str + '_') if surf_str else ''}"
             f"rtg{max_std_rank}_{max_dif_std_rank}",
        levels=['main', 'gs', 'masters'],
        surfaces=None if not surf_list else surf_list,
        drop_cond=drop_condition_surf if surf_list else drop_condition,
        skip_match_cond=(
            lambda m: m.level not in ('main', 'gs', 'masters')
            or (m.rnd == 'Qualifying' and m.level == 'main')
            or (m.rnd in ('q-First', 'q-Second') and m.level != 'masters')
            or (m.rnd is not None and m.rnd.rubber())
            or (surf_list and m.surface not in surf_list)
        ),
        stratify_names=([rnd_stratify_name] if len(surf_list) == 1
                        else ['surface_text', rnd_stratify_name]),
    )


ProfitRatios = namedtuple("ProfitRatios", "pos_ratio neg_ratio")
PosNeg = namedtuple("PosNeg", "pos neg")
PosNeg2 = namedtuple("PosNeg2", "pos neg neg2")


class Variant:
    """ сущность с выбранными фичами, классиф-ром с выбранными настройками.
        также выбраны ограничения примененные к исходному набору данных
    """
    def __init__(
        self,
        sex: str,
        model_name: str,
        suffix: str,
        key: LevSurf,
        metric_name: str,
        cls,
        clf_pars,
        min_probas: Union[PosNeg, PosNeg2],
        profit_ratios: Tuple[float, float],
        feature_names: FeatureNames,
        weight_mode=WeightMode.NO,
        drop_cond=None,
    ):
        self._sex = sex
        self.model_name = model_name
        self.suffix = suffix
        self.key = key
        self.metric_name = metric_name
        self.cls = cls
        self.clf_pars = clf_pars  # dict: param_name -> param_value
        self.min_probas = min_probas
        self.profit_ratios = ProfitRatios(
            pos_ratio=profit_ratios[0], neg_ratio=profit_ratios[1]
        )
        self.feature_names = feature_names
        self.weight_mode = weight_mode
        self.clf = None
        self.drop_cond = drop_cond

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

    def exist_cat(self):
        return self.feature_names.exist_cat()

    @property
    def sex(self):
        return self._sex

    def exist_chal(self):
        return self.key.exist_chal()

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
        self.clf = self.cls(**pars)
        return self.clf

    def make_clf_fit(self, data, random_seed: int, plot=False):
        import clf_cat_tools

        self.clf = None
        if self.is_cb_native():
            if data.exist_cat():
                assert self.exist_cat(), (
                    f"data with cat, variant.feature_names without cat:"
                    f"\n{self.feature_names}")
                data.tolist()
            pools = clf_cat_tools.make_pools(data)
            clf = clf_cat_tools.train_model(
                pools,
                eval_set_admit=True,
                metric_name=self.metric_name,
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

    def persist_dirname(self, subname=None):
        pers_dirname = os.path.join(cfg_dir.analysis_data_dir(), "persist")
        if subname:
            return os.path.join(pers_dirname, f"{self.model_name}_{self.name}", subname)
        return os.path.join(pers_dirname, f"{self.model_name}_{self.name}")

    def save_clf(self):
        if self.is_cb_native():
            dname = self.persist_dirname(subname="clf")
            fname = os.path.join(dname, f"clf_{self.metric_name}.model")
            self.clf.save_model(fname)
        else:
            raise NotImplementedError(f"save_clf {self}")

    def load_clf(self):
        if self.is_cb_native():
            dname = self.persist_dirname(subname="clf")
            fname = os.path.join(dname, f"clf_{self.metric_name}.model")
            self.clf = CatBoostClassifier()
            self.clf.load_model(fname)
        else:
            raise NotImplementedError(f"save_clf {self}")

    def write_df(self, df: pd.DataFrame, subname: str):
        dirname = self.persist_dirname()
        save_df(df, dirname, subname=subname)

    def read_df(self, subname: str) -> pd.DataFrame:
        dirname = self.persist_dirname()
        df = load_df(dirname, subname=subname)
        if df is None:
            raise ValueError(f"no dataset {self.sex} dirname: {dirname} {subname}")
        if df.shape[0] < DATASET_MIN_LEN:
            raise ValueError(f"few dataset {self.sex} dirname: {dirname} {subname}")
        return df

    def write_note(self, subname: str, text: str):
        dirname = self.persist_dirname()
        filename = os.path.join(dirname, subname, "note.txt")
        with open(filename, "w") as f:
            f.write(f"{text}\n")

    def get_features_importance(self, data=None):
        from clf_cat_tools import make_pools

        feat_names = self.feature_names.get_list()
        if self.is_cb_native():
            if data is not None:
                if data.exist_cat():
                    assert self.exist_cat(), (
                        f"data with cat, variant.feature_names without cat:"
                        f"\n{self.feature_names}")
                    data.tolist()
                pools = make_pools(data)
                return dict(
                    zip(
                        feat_names,
                        self.clf.get_feature_importance(
                            data=pools.train, type="LossFunctionChange"
                        ),
                    )
                )
            else:
                return dict(zip(feat_names, self.clf.get_feature_importance()))
        return dict(zip(feat_names, self.clf.feature_importances_))

    def make_test_result(self, data):
        import clf_cat_tools

        if self.is_cb_native():
            return clf_cat_tools.get_test_result(
                self, self.clf, clf_cat_tools.make_test_pool(data)
            )
        return get_result(self, self.clf, X_test=data.test.X, y_test=data.test.y)

    def drop_by_condition(self, df: pd.DataFrame) -> None:
        self.key.drop_by_condition(df)
        if self.drop_cond is not None:
            drop_by_condition(df, self.drop_cond)

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


def load_variants(variants: Optional[List[Variant]] = None):
    if variants:
        for variant in variants:
            variant.load_clf()
            if variant.clf is None:
                raise co.TennisError(f"fail cb model load {variant.name}")


rnd_name_norm_value: Dict[str, float] = {
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


_rnd_code: Dict[str, int] = {
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


def rnd_code(rnd_text: str) -> int:
    return _rnd_code[rnd_text]


def rnd_stage(rnd_text: str, raw_level_text: str, qual_union=False) -> int:
    if rnd_text == "q-First":
        return 0 if qual_union else -2
    if rnd_text == "q-Second":
        return 0 if qual_union else -1
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
    if "Rubber" in rnd_text:
        return -1
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


def extract_rnd_text(rnd) -> str:
    if isinstance(rnd, str):
        value = rnd
    else:
        value = rnd.value
    return unified_rnd_value(value)


def unified_rnd_value(value: str) -> str:
    if value.startswith("Rubber"):
        return "Rubber"
    elif value == "Bronze":
        return "Final"
    return value


def miss_point_weight(sex, scope, on_serve, level, surface):
    from matchstat import generic_result_value

    MATCH_SCOPE_COEF = 2.2  # comparing with set scope miss points
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
        ), "none service_win for {} lev:{} srf:{}".format(sex, level, surface)
        srv_weight = srv_win_ratio / (1.0 - srv_win_ratio)  # prob_srv / prob_rcv
        return srv_weight


def persist_dirname(model_name, key, subname=None):
    pers_dirname = os.path.join(cfg_dir.analysis_data_dir(), "persist")
    if subname:  # 'test' or 'eval'
        return os.path.join(pers_dirname, "{}_{}".format(key, model_name), subname)
    return os.path.join(pers_dirname, "{}_{}".format(key, model_name))


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


class DataSet:
    def __init__(self, X, y, w=None):
        self.X = X
        self.y = y
        self.w = w

    def __bool__(self):
        return self.X is not None and self.y is not None

    __nonzero__ = __bool__

    def tolist(self, cat_indices):
        self.y = self.y.tolist()
        if self.w is not None:
            self.w = self.w.tolist()
        self.X = x_tolist(self.X, cat_indices)

    def rows_num(self):
        if self.X is not None:
            return len(self.X)


class Data:
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

    def size_report(self):
        train_size, eval_size, test_size = 0, 0, 0
        if self.train.X is not None and isinstance(self.train.X, np.ndarray):
            train_size = self.train.X.shape[0]
        if self.eval.X is not None and isinstance(self.eval.X, np.ndarray):
            eval_size = self.eval.X.shape[0]
        if self.test.X is not None and isinstance(self.test.X, np.ndarray):
            test_size = self.test.X.shape[0]
        tot_size = train_size + eval_size + test_size
        if tot_size:
            tran_rat_txt = f"{train_size/tot_size:.2f}"
            eval_rat_txt = f"{eval_size/tot_size:.2f}"
            test_rat_txt = f"{test_size/tot_size:.2f}"
            return (f"all={tot_size} train:{tran_rat_txt}"
                    f" eval:{eval_rat_txt} test: {test_rat_txt}")
        return 'empty Data (or not ndarray)'

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
        X_test = df.loc[i, fnames]
        y_test = df.loc[i, label_name]
        if variant.is_cb_native() and variant.exist_cat():
            X_tst = x_tolist(X_test.values, variant.feature_names.cat_indexes())
        else:
            X_tst = X_test.values
        proba = clf.predict_proba(X_tst)
        groupby_name_val = df.loc[i, groupby_name]
        groupby_key = groupby_key_fun(groupby_name_val)
        positive_prob = proba[1]
        negative_prob = proba[0]
        if positive_prob >= variant.min_probas.pos:
            pos_dct[groupby_key].hit(y_test >= 0.5)  # win if label 1
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


def plot_precision_recall(clf, X_test, y_test, pos_proba=0.5):
    """from mueller."""
    assert clf.is_fitted(), "not fitted clf"
    pred_pos_proba = clf.predict_proba(X_test)[:, 1]

    avgprec = average_precision_score(y_test, pred_pos_proba)
    print("Average precision: {:.3f}".format(avgprec))

    precision_fr, recall_fr, thresholds_fr = precision_recall_curve(
        y_test, pred_pos_proba, pos_label=1)
    plt.plot(precision_fr, recall_fr, label="fr")
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


def get_xy(df: pd.DataFrame, feature_names, label_name: str):
    x = df[feature_names].values
    y = df[label_name].values
    return x, y


def x_tolist(X: np.ndarray, cat_indices):
    dim = len(X.shape)
    result_x = X.tolist()
    for cat_idx in cat_indices:
        if dim >= 2:
            for row in result_x:
                assert 0 <= cat_idx < len(row)
                row[cat_idx] = int(round(row[cat_idx]))
        elif dim == 1:
            assert 0 <= cat_idx < len(result_x)
            result_x[cat_idx] = int(round(result_x[cat_idx]))
    return result_x


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
    return copy.deepcopy(df.loc[idx_train, :]), copy.deepcopy(df.loc[idx_test, :])


def split_all_by_two_chunks(df: pd.DataFrame, remain_size: float
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = df.shape[0]
    end_first_idx = max(0, int(round(n_rows * (1. - remain_size))))
    # splitting by row index (end_first_idx):
    df_first = df.iloc[:end_first_idx, :]
    df_remain = df.iloc[end_first_idx:, :]
    return df_first, df_remain


Splited = namedtuple("Splited", "train eval test")


def splited_by_chunks(df: pd.DataFrame, split,
                      test_size: float, eval_size: float) -> Splited:
    n_rows = df.shape[0]
    if split:
        # seq: train, eval, test
        n_eval_rows = int(round(n_rows * eval_size))
        n_test_rows = int(round(n_rows * test_size))
        beg_test_idx = max(0, n_rows - n_test_rows)
        beg_eval_idx = max(0, n_rows - n_test_rows - n_eval_rows)

        df_train = df.iloc[:beg_test_idx, :]
        df_eval = df.iloc[beg_eval_idx:beg_test_idx, :]
        df_test = df.iloc[beg_test_idx:, :]
    elif split is False:
        df_eval = None
        df_train, df_test = split_all_by_two_chunks(df, remain_size=test_size)
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
    X_train, X_eval, X_test,
    y_train, y_eval, y_test,
    w_train=None, w_eval=None, w_test=None,
):
    from clf_cat_tools import DataCatBoost

    if cat_features_idx is not None:
        return DataCatBoost(
            X_train=X_train, X_eval=X_eval, X_test=X_test,
            y_train=y_train, y_eval=y_eval, y_test=y_test,
            cat_features_idx=cat_features_idx,
            w_train=w_train, w_eval=w_eval, w_test=w_test,
        )
    return Data(
        X_train=X_train, y_train=y_train, X_eval=X_eval,
        y_eval=y_eval, X_test=X_test, y_test=y_test,
        w_train=w_train, w_eval=w_eval, w_test=w_test,
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
                    X_train, X_eval, X_test,
                    y_train, y_eval, y_test,
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
                    X_train=X_train, X_eval=None, X_test=X_test,
                    y_train=y_train, y_eval=None, y_test=y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=None, w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train, X_eval=None, X_test=X_test,
                y_train=y_train, y_eval=None, y_test=y_test,
            ),
            df_spl,
        )
    elif split is None:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train, X_eval=None, X_test=None,
                    y_train=y_train, y_eval=None, y_test=None,
                    w_train=df_spl.train["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train, X_eval=None, X_test=None,
                y_train=y_train, y_eval=None, y_test=None,
            ),
            df_spl,
        )
    else:
        raise co.TennisError("invalid split value {}".format(split))


def fill_data_ending_chrono(
    df,
    split,
    test_size: float,
    eval_size: float,
    feature_names,
    label_name,
    other_names,
    cat_features_idx,
    weight_mode=WeightMode.NO,
):
    assert isinstance(other_names, list)
    df_spl = splited_by_chunks(df, split, test_size=test_size, eval_size=eval_size)
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
                    X_train, X_eval, X_test,
                    y_train, y_eval, y_test,
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
                    X_train=X_train, X_eval=None, X_test=X_test,
                    y_train=y_train, y_eval=None, y_test=y_test,
                    w_train=df_spl.train["weight"].values,
                    w_eval=None,
                    w_test=df_spl.test["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train, X_eval=None, X_test=X_test,
                y_train=y_train, y_eval=None, y_test=y_test,
            ),
            df_spl,
        )
    elif split is None:
        X_train, y_train = get_xy(df_spl.train, feature_names, label_name)
        if weight_mode != WeightMode.NO:
            return (
                make_data(
                    cat_features_idx,
                    X_train=X_train, X_eval=None, X_test=None,
                    y_train=y_train, y_eval=None, y_test=None,
                    w_train=df_spl.train["weight"].values,
                ),
                df_spl,
            )
        return (
            make_data(
                cat_features_idx,
                X_train=X_train, X_eval=None, X_test=None,
                y_train=y_train, y_eval=None, y_test=None,
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
