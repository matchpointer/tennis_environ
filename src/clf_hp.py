from collections import namedtuple
from functools import partial
import pickle
import msvcrt

from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, anneal, rand, mix, Trials, STATUS_OK, hp

import common as co
import clf_common as cco


best_score = 0.0
trials_count = 0


def is_quit_pressed():
    if msvcrt.kbhit():
        return msvcrt.getch().decode("utf-8") == "q"
    return False


def choice_space(name, choices):
    return hp.choice(name, list(choices))


def get_objective(data, classifier, cv, scoring="roc_auc", scaler=None):
    assert cv is not None, "unexpected cv {}".format(cv)

    def objective(space):
        global best_score, trials_count
        trials_count += 1
        if (trials_count % 10) == 0 and is_quit_pressed():
            raise co.TennisAbortError
        args_dct = dict(**space)
        clf = cco.make_clf(classifier, args_dct, scaler=scaler)
        scores = cross_val_score(
            clf, data.train.X, data.train.y, scoring=scoring, cv=cv
        )
        scr_val = scores.mean()
        if scr_val > best_score:
            cco.out("achieved best score {} at {}".format(scr_val, args_dct))
            best_score = scr_val
        return {"loss": 1.0 - scr_val, "status": STATUS_OK}

    return objective


def load_trials(max_evals):
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./trials.pkl", "rb"))
        cco.out("Found saved Trials! Loading...")
        new_max_evals = len(trials.trials) + max_evals
        cco.out(
            "Rerunning from {} trials to {} (+{}) trials".format(
                len(trials.trials), new_max_evals, max_evals
            )
        )
    except Exception as err:  # create a new trials object and start searching
        cco.out("load_trials failed: {}".format(err))
        trials = Trials()
        new_max_evals = max_evals
    return trials, new_max_evals


def save_trials(trials):
    with open("./trials.pkl", "wb") as fh:
        pickle.dump(trials, fh)
    cco.out("saved trials")


def do_fmin(objective, space, max_evals, mix_algo_ratio=None):
    trials, max_evals = load_trials(max_evals)
    best = None
    try:
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest
            if mix_algo_ratio is None
            else get_mix_algo(mix_algo_ratio),
            max_evals=max_evals,
            trials=trials,
        )
    except co.TennisAbortError:
        cco.out("user quit event")
    if best is not None:
        cco.out("best: {}".format(best))
    cco.out("best_trial result: {}".format(trials.best_trial.get("result")))
    save_trials(trials)
    return trials


MixAlgoRatio = namedtuple("MixAlgoRatio", "tpe anneal rand")


def get_mix_algo(mix_algo_ratio):
    return partial(
        mix.suggest,
        p_suggest=[
            (mix_algo_ratio.rand, rand.suggest),
            (mix_algo_ratio.tpe, tpe.suggest),
            (mix_algo_ratio.anneal, anneal.suggest),
        ],
    )
