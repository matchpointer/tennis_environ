# -*- coding=utf-8 -*-
from typing import Optional

from loguru import logger as log

import common as co
from side import Side
from report_line import make_get_adv_side, SizedValue

ADV_PREF = 'advleft_'

# here
# 'min_adv_value':
#   if sizes are equal -> min for mixed value (balanced adv_side value and oppo_side value)
#   if sizes are no eq -> min for adv_side value
# 'min_adv_size' min size for advantage side
# 'min_oppo_size' min size for disadvantage side
# 'max_oppo_value' max value for disadvantage side
params = {
    "wta": {
        "sd_": {
            "fun_args": [
                {
                    "min_adv_size": 10,
                    "min_adv_value": 0.575,
                    "min_oppo_size": 10,
                    "max_oppo_value": 0.50,
                },
                {
                    "min_adv_size": 12,
                    "min_adv_value": 0.61,
                    "min_oppo_size": 12,
                    "max_oppo_value": 0.53,
                },
                {
                    "min_adv_size": 11,
                    "min_adv_value": 0.60,
                    "min_oppo_size": 2,
                    "max_oppo_value": 0.40,
                },
                {
                    "min_adv_size": 6,
                    "min_adv_value": 0.67,
                    "min_oppo_size": 12,
                    "max_oppo_value": 0.36,
                },
            ]
        },
        "s1_": {
            "fun_args": [
                {
                    "min_adv_size": 14,
                    "min_adv_value": 0.56,
                    "min_oppo_size": 14,
                    "max_oppo_value": 0.48,
                },
                {
                    "min_adv_size": 12,
                    "min_adv_value": 0.60,
                    "min_oppo_size": 12,
                    "max_oppo_value": 0.50,
                },
            ]
        },
    },
    "atp": {
        "sd_": {
            "fun_args": [
                {
                    "min_adv_size": 12,
                    "min_adv_value": 0.58,
                    "min_oppo_size": 12,
                    "max_oppo_value": 0.52,
                },
                {
                    "min_adv_size": 14,
                    "min_adv_value": 0.63,
                    "min_oppo_size": 14,
                    "max_oppo_value": 0.56,
                },
                {
                    "min_adv_size": 14,
                    "min_adv_value": 0.65,
                    "min_oppo_size": 3,
                    "max_oppo_value": 0.40,
                },
                {
                    "min_adv_size": 8,
                    "min_adv_value": 0.77,
                    "min_oppo_size": 12,
                    "max_oppo_value": 0.36,
                },
            ]
        },
        "s1_": {
            "fun_args": [
                {
                    "min_adv_size": 20,
                    "min_adv_value": 0.59,
                    "min_oppo_size": 20,
                    "max_oppo_value": 0.48,
                },
                {
                    "min_adv_size": 15,
                    "min_adv_value": 0.64,
                    "min_oppo_size": 15,
                    "max_oppo_value": 0.52,
                },
            ]
        },
    },
}


def get_adv_side(
    sex: str, setpref: str, fst_sv: SizedValue, snd_sv: SizedValue
) -> Optional[Side]:
    """ API. return Side if exist advantage, return None otherwise """
    par_dct = params[sex].get(setpref)
    if par_dct is None:
        log.error(f'Not found params for sex:{sex} setpref:{setpref}')
        return None
    fun_args = par_dct["fun_args"]
    adv_side_funs = [make_get_adv_side(**fun_arg) for fun_arg in fun_args]
    adv_side = _get_adv_side_impl(adv_side_funs, fst_sv, snd_sv)
    return adv_side


def _get_adv_side_impl(
    adv_side_funs, fst_sv: SizedValue, snd_sv: SizedValue
) -> Optional[Side]:
    """ adv_side_funs:
        callable objects полученные как рез-т вызова make_get_adv_side(**fun_args) """
    for adv_side_f in adv_side_funs:
        r = adv_side_f(fst_sv, snd_sv)
        if r in (co.LEFT, co.RIGHT):
            return r
