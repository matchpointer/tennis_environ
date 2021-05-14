from collections import namedtuple

from sklearn.ensemble import GradientBoostingClassifier
from hyperopt import hp

import clf_common as cco
import clf_hp


GboostPart = namedtuple(
    "GboostPart", "pratio n_estim1 n_estim2 n_estimq learn_rate1 learn_rate2"
)


def get_parts():
    return [
        #       GboostPart(pratio=0.5, n_estim1=2000, n_estim2=4800, n_estimq=15,
        #                  learn_rate1=0.00001, learn_rate2=0.0007),
        #       GboostPart(pratio=0.5, n_estim1=430, n_estim2=2010, n_estimq=10, # was 15
        #                  learn_rate1=0.0005, learn_rate2=0.009),
        GboostPart(
            pratio=0.4,
            n_estim1=200,
            n_estim2=460,
            n_estimq=3,  # was 5
            learn_rate1=0.009,
            learn_rate2=0.017,
        ),
        GboostPart(
            pratio=0.6,
            n_estim1=30,
            n_estim2=250,
            n_estimq=3,  # was 5
            learn_rate1=0.016,
            learn_rate2=0.37,
        ),
    ]


def space_gboost(parts, n_features, max_depth_space=None, random_state=None):
    # n_estimators = hp.choice('n_estimators', range(100, 1150, 10))
    if max_depth_space is not None:
        max_depth = max_depth_space
    else:
        max_depth = hp.pchoice(
            "max_depth",
            [
                (0.29, 2),
                (0.27, 3),
                (0.24, 4),
                (0.20, 5),
                # (0.20, 6),
            ],
        )
    loss = hp.choice("loss", ["exponential", "deviance"])
    max_features = hp.choice("max_features", cco.colsample_ratios(n_features))

    space = hp.pchoice(
        "split_parts",
        [
            (
                part.pratio,
                {
                    "random_state": random_state,
                    "max_depth": max_depth,
                    "loss": loss,
                    "max_features": max_features,
                    "n_estimators": hp.choice(
                        "estimators_" + str(idx),
                        list(range(part.n_estim1, part.n_estim2, part.n_estimq)),
                    ),
                    "learning_rate": hp.uniform(
                        "learning_rate_" + str(idx), part.learn_rate1, part.learn_rate2
                    ),
                },
            )
            for idx, part in enumerate(parts)
        ],
    )
    return space


def do_fmin(
    data,
    max_evals,
    mix_algo_ratio=None,
    max_depth_space=None,
    random_state=None,
    scaler=None,
):
    space = space_gboost(
        get_parts(),
        n_features=data.X_train.shape[1],
        max_depth_space=max_depth_space,
        random_state=random_state,
    )
    objective = clf_hp.get_objective(
        data,
        classifier=GradientBoostingClassifier,
        cv=5,
        scoring="roc_auc",
        scaler=scaler,
    )
    return clf_hp.do_fmin(
        objective=objective,
        space=space,
        max_evals=max_evals,
        mix_algo_ratio=mix_algo_ratio,
    )
