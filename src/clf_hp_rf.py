from collections import namedtuple

from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp

import clf_common as cco
import clf_hp


NestimPart = namedtuple("NestimPart", "pratio n_estim1 n_estim2 n_estimq")


def get_parts():
    return [
        NestimPart(pratio=0.75, n_estim1=25, n_estim2=75, n_estimq=2),
        NestimPart(pratio=0.25, n_estim1=75, n_estim2=500, n_estimq=10),
    ]


def space_rf(parts, n_features, max_depth_space, random_state):
    if max_depth_space is not None:
        max_depth = max_depth_space
    else:
        max_depth = hp.pchoice(
            "max_depth",
            [
                (0.22, 1),
                (0.22, 2),
                (0.24, 3),
                (0.22, 4),
                (0.10, 5),
            ],
        )

    criterion = hp.pchoice("criterion", [(0.5, "gini"), (0.5, "entropy")])
    max_features = hp.choice("max_features", cco.colsample_ratios(n_features))
    min_samples_leaf = hp.choice("min_samples_leaf", [15, 20, 30, 40])

    space = hp.pchoice(
        "split_parts",
        [
            (
                part.pratio,
                {
                    "random_state": random_state,
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "n_estimators": hp.choice(
                        "estimators_" + str(idx),
                        list(range(part.n_estim1, part.n_estim2, part.n_estimq)),
                    ),
                },
            )
            for idx, part in enumerate(parts)
        ],
    )
    return space


def do_fmin(
    data, max_evals, mix_algo_ratio=None, max_depth_space=None, random_state=None
):
    parts = get_parts()
    space = space_rf(
        parts,
        n_features=data.X_train.shape[1],
        max_depth_space=max_depth_space,
        random_state=random_state,
    )
    objective = clf_hp.get_objective(
        data, classifier=RandomForestClassifier, cv=5, scoring="roc_auc"
    )
    return clf_hp.do_fmin(
        objective=objective,
        space=space,
        max_evals=max_evals,
        mix_algo_ratio=mix_algo_ratio,
    )
