
import msvcrt
import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from catboost import CatBoost, CatBoostClassifier, cv

import common as co
import clf_common as cco
import clf_hp
import clf_cat_tools

best_score = 0.0
trials_count = 0


def is_quit_pressed():
    if msvcrt.kbhit():
        return msvcrt.getch().decode("utf-8") == "q"
    return False


def get_space():
    params_space = {
        # 'learning_rate': hp.choice('learning_rate', [0.03]),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1.9, 5.2),
        # 'random_strength': hp.uniform('random_strength', 1., 2.),
        # 'bagging_temperature': hp.uniform('bagging_temperature', 0., 1.4),
        "depth": hp.choice("depth", [3]),
        # 'per_float_feature_quantization': hp.choice(
        #     'per_float_feature_quantization',
        #     [['0:border_count=255'], ['0:border_count=512'], ['0:border_count=720']]),
        "iterations": hp.choice("iterations", [130]),
    }
    return params_space


def do_fmin(pools, max_evals, mix_algo_ratio=None, random_state=None, how="cv"):
    assert isinstance(pools, clf_cat_tools.Pools)
    space = get_space()
    objective = get_objective(pools, random_state, how=how)
    return do_fmin_impl(
        objective,
        space,
        max_evals,
        mix_algo_ratio=mix_algo_ratio,
        random_state=random_state,
    )


def get_objective(pools, random_state, how="sklearn", metric_name="AUC"):
    def objective(space):
        global best_score, trials_count
        #       if os.path.isdir('./catboost_info'):
        #           shutil.rmtree('./catboost_info', ignore_errors=True)
        trials_count += 1
        if (trials_count % 5) == 0 and is_quit_pressed():
            raise co.TennisAbortError
        args_dct = dict(**space)
        params = {
            "eval_metric": metric_name,
            # 'eval_metric': 'Logloss',
            "random_seed": random_state,
            "logging_level": "Silent",
        }
        params.update(args_dct)
        if how == "cv":
            cv_data = cv(pools.train, params, stratified=True)
            scr_val = np.max(cv_data[f"test-{metric_name}-mean"])
        elif how == "sklearn":
            mdl = CatBoostClassifier(**params)
            mdl.fit(pools.train)
            pred = mdl.predict_proba(pools.eval)[:, 1]
            scr_val = roc_auc_score(pools.eval.y, pred)
        elif how == "native":
            mdl = CatBoost(params)
            mdl.fit(
                pools.train,
                eval_set=None,  # pools.eval if pools.eval else None,
                silent=True,
            )  # eval_set=pools.eval
            pred = mdl.predict(pools.eval, prediction_type="Probability")[:, 1]
            scr_val = roc_auc_score(pools.eval.get_label(), pred)
        else:
            raise Exception("bad how arg {}".format(how))

        #       pred = mdl.predict(data.X_test)
        #       scr_val = precision_score(data.y_test, pred)

        if scr_val > best_score:
            if how == "cv":
                cco.out("achieved best {} at {}".format(scr_val, params))
            else:
                cco.out(
                    "achieved best {} at {} lrate: {} ntrees: {}".format(
                        scr_val, mdl.get_params(), mdl.learning_rate_, mdl.tree_count_
                    )
                )
            best_score = scr_val
        return {"loss": 1.0 - scr_val, "status": STATUS_OK}

    return objective


def do_fmin_impl(objective, space, max_evals, mix_algo_ratio=None, random_state=None):
    # trials, max_evals = clf_hp.load_trials(max_evals)
    trials = Trials()
    best = None
    try:
        best = fmin(
            fn=objective,
            space=space,
            algo=(
                tpe.suggest
                if mix_algo_ratio is None
                else clf_hp.get_mix_algo(mix_algo_ratio)
            ),
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.RandomState(random_state),
        )
        print("fmin done")
    except co.TennisAbortError:
        cco.out("user quit event")
    if best is not None:
        cco.out("best: {}".format(best))
    cco.out("best_trial result: {}".format(trials.best_trial.get("result")))
    # clf_hp.save_trials(trials)
    return trials
