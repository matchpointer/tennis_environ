from catboost import CatBoostClassifier

import clf_common as cco

default_feat_names = cco.FeatureNames(
    names=[
        "fst_bet_chance",
        "dif_srv_ratio",  # replace dif_avg_srv_code
        "s1_dif_games",
        # other features temporary removed
    ]
)

default_wta_feat_names = default_feat_names

default_atp_feat_names = default_feat_names


var_wta_main_clr_rtg200_150_0sp = cco.Variant(
    sex="wta",
    suffix="0sp",
    key=cco.key_main_clr_rtg200_150,
    cls=CatBoostClassifier,
    clf_pars={
        "boosting_type": "Ordered",
        "depth": 4,
        "iterations": 207,
        "learning_rate": 0.03,
        "logging_level": "Silent",
    },
    min_probas=cco.PosNeg(0.5, 1.0),
    profit_ratios=(0.55, 0.7),
    feature_names=cco.FeatureNames(
        names=[
            "fst_bet_chance",
            "dif_srv_ratio",
            "s1_dif_games",
            # other features temporary removed
        ]
    ),
    calibrate="",
)

