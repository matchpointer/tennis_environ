from catboost import CatBoostClassifier

import clf_common as cco

default_wta_feat_names = cco.FeatureNames(
    names=[
        "dif_age",
        "avg_age",
        "dif_plr_tour_adapt",
        "dif_fatigue",
        "h2h_direct",
        "fst_bet_chance",
        # here other features temporary droped
    ]
)

default_atp_feat_names = cco.FeatureNames(
    names=[
        "dif_age",
        "avg_age",
        "dif_plr_tour_adapt",
        "dif_fatigue",
        "h2h_direct",
        "fst_bet_chance",
        # here other features temporary droped
    ]
)



var_atp_main_clr_1f = cco.Variant(
    sex="atp",
    suffix="1f",
    key=cco.key_main_clr,
    cls=CatBoostClassifier,
    clf_pars={
        'boosting_type': 'Ordered',
        'depth': 5,
        'early_stopping_rounds': 80,
        'eval_metric': 'AUC',
        'learning_rate': 0.025,
        'logging_level': 'Silent',
        'per_float_feature_quantization': ['0:border_count=512'],
    },
    min_probas=cco.PosNeg(0.65, 0.65),
    profit_ratios=(0.6, 0.6),
    feature_names=cco.FeatureNames(
        names=[
            "dif_age",
            "avg_age",
            "dif_plr_tour_adapt",
            "dif_fatigue",
            "h2h_direct",
            "fst_bet_chance",
            # here other features temporary droped
        ]
    ),
    calibrate="",
)


var_wta_main_clr_rtg350_250_1 = cco.Variant(
    sex="wta",
    suffix="1",
    key=cco.key_main_clr_rtg350_250,
    cls=CatBoostClassifier,
    clf_pars={
        'depth': 4,
        'early_stopping_rounds': 100,
        'eval_metric': 'AUC',
        'learning_rate': 0.025,
        'logging_level': 'Silent'
    },
    min_probas=cco.PosNeg(0.65, 0.65),
    profit_ratios=(0.64, 0.64),
    feature_names=cco.FeatureNames(
        names=[
            "dif_age",
            "avg_age",
            "dif_plr_tour_adapt",
            "dif_fatigue",
            "h2h_direct",
            "fst_bet_chance",
            # here other features temporary droped
        ],
    ),
    calibrate="",
)

