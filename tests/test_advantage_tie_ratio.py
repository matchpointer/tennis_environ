# -*- coding=utf-8 -*-
import pytest

from side import Side
from report_line import SizedValue
from advantage_tie_ratio import get_adv_side


@pytest.mark.parametrize(
    "sex, setpref, fst_sv, snd_sv, waited",
    [
        ('wta', 'sd_', SizedValue(0.32, 34), SizedValue(0.56, 18), Side('RIGHT')),
        ('wta', 'sd_', SizedValue(0.32, 4), SizedValue(0.56, 4), None),
        ('wta', 'sd_', SizedValue(0.61, 11), SizedValue(0.33, 3), Side('LEFT')),
        ('wta', 's1_', SizedValue(0.63, 80), SizedValue(0.50, 12), Side('LEFT')),
        #
        ('atp', 's1_', SizedValue(0.72, 35), SizedValue(0.51, 15), Side('LEFT')),
        ('atp', 'sd_', SizedValue(0.67, 12), SizedValue(0.50, 12), Side('LEFT')),
        #
        ('atp', 'sd_', SizedValue(0.36, 30), SizedValue(9/10, 10), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.36, 12), SizedValue(9/10, 10), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.36, 12), SizedValue(8/9, 9), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.36, 12), SizedValue(7/9, 9), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.3, 24), SizedValue(7/9, 9), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.36, 12), SizedValue(8/10, 8), Side('RIGHT')),
        ('atp', 'sd_', SizedValue(0.36, 12), SizedValue(7/8, 8), Side('RIGHT')),
    ],
)
def test_get_adv_side(
    sex, setpref, fst_sv, snd_sv, waited
):
    result = get_adv_side(sex, setpref, fst_sv, snd_sv)
    assert result == waited

