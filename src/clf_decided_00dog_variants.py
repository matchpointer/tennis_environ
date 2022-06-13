from catboost import CatBoostClassifier

import clf_common as cco

default_model_name = "decided_00dog"
default_metric_name = "Logloss"  # 'AUC' 'Precision'  'Accuracy'

default_pos_proba = {
    'wta': 0.485,
    'atp': 0.485,
}

default_neg_proba = {
    'wta': 0.695,
    'atp': 0.68,
}


def get_default_min_probas(sex: str):
    return cco.PosNeg(pos=default_pos_proba[sex], neg=default_neg_proba[sex])


default_wta_feat_names = cco.FeatureNames(
    names=[
        "fst_bet_chance",
        "dset_ratio_dif",
        'dif_elo_pts',
        # and some others
    ]
)

default_atp_feat_names = cco.FeatureNames(
    names=[
        "dset_ratio_dif",
        "fst_bet_chance",
        "dif_srv_ratio",  # live by set1, set2
        "spd_fst_lastrow_games",
        # and some others
    ]
)

var_atp_main_clr_rtg500_300_1ffrs = cco.Variant(
    sex="atp",
    model_name=default_model_name,
    suffix="1ffrs",
    key=cco.make_key_main_clr(max_std_rank=500, max_dif_std_rank=300,
                              rnd_stratify_name="rnd_stage"),
    metric_name=default_metric_name,
    cls=CatBoostClassifier,
    clf_pars={
        'boosting_type': 'Ordered',
        'depth': 5,
        'leaf_estimation_method': 'Gradient',
        'learning_rate': 0.023,
        'logging_level': 'Silent',
        'per_float_feature_quantization': ['0:border_count=512']
    },
    min_probas=get_default_min_probas('atp'),
    feature_names=default_atp_feat_names
)

var_wta_main_clr_rtg550_350_1Npm04 = cco.Variant(
    sex="wta",
    model_name=default_model_name,
    suffix="1Npm04",  # trained with: book_chance NOT in [0.5-0.04, 0.5+0.04]
    key=cco.make_key_main_clr(max_std_rank=550, max_dif_std_rank=350,
                              rnd_stratify_name="rnd_stage"),
    metric_name=default_metric_name,
    cls=CatBoostClassifier,
    clf_pars={
        'boosting_type': 'Ordered',
        'depth': 5,
        'eval_metric': 'Logloss',
        'leaf_estimation_method': 'Gradient',
        'logging_level': 'Silent',
        'per_float_feature_quantization': ['0:border_count=512']
    },
    min_probas=cco.PosNeg(pos=0.48, neg=default_neg_proba['wta']),
    feature_names=default_wta_feat_names
)
