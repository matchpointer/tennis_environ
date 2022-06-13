# -*- coding=utf-8 -*-
import copy
from collections import namedtuple

import feature

Item = namedtuple("Item", "is_left_win date")

DEFAULT_TIME_DISCOUNT_FACTOR = 0.95


def add_features(features, date, h2hobj, time_discounts=(0.95, 0.90, 0.85, 0.80, 0.75)):
    assert h2hobj is not None
    assert date is not None
    items = [
        Item(
            is_left_win=h2hobj.fst_player == mch.first_player,
            date=mch.date if mch.date else tour.date,
        )
        for tour, mch, _ in h2hobj.tour_match_aset
    ]
    calc = Calc(date, items)
    for time_discount in time_discounts:
        feat_name = "h2h_{:.2f}".format(time_discount).replace(".", "")
        calc.time_discount = time_discount
        value = calc.direct()
        features.append(
            feature.Feature(
                name=feat_name,
                value=value,
                flip_value=None if value is None else 1.0 - value,
            )
        )


class Calc(object):
    def __init__(self, date, items, time_discount=DEFAULT_TIME_DISCOUNT_FACTOR):
        self.items = copy.copy(items)
        self.date = date
        self.time_discount = time_discount

    def direct(self, small_size_adjust=True, default_value=None):
        """returned value in [0...0.5) in favor left player,
        (0.5...1] in favor right player"""
        fst_cnt, fst_sum, snd_cnt, snd_sum = self._win_counts()
        if (fst_cnt, snd_cnt) == (0, 0):
            return default_value
        if (fst_cnt + snd_cnt) > 5 or not small_size_adjust:
            return self._direct_core(fst_cnt, fst_sum, snd_cnt, snd_sum)
        return self._small_size_adjust_direct(fst_cnt, fst_sum, snd_cnt, snd_sum)

    def _time_discount_coef(self, match_date):
        if match_date is None or self.date is None:
            return 1.0  # no discount
        delta = self.date - match_date
        delta_years = float(delta.days) / 365.0
        return self.time_discount ** delta_years

    def _win_counts(self):
        fst_cnt, snd_cnt = 0, 0
        fst_sum, snd_sum = 0.0, 0.0
        for item in self.items:
            disc_coef = self._time_discount_coef(item.date)
            if item.is_left_win:
                fst_cnt += 1
                fst_sum += disc_coef
            else:
                snd_cnt += 1
                snd_sum += disc_coef
        return fst_cnt, fst_sum, snd_cnt, snd_sum

    @staticmethod
    def _direct_core(fst_cnt, fst_sum, snd_cnt, snd_sum):
        if snd_cnt == 0:
            fst_adv_ratio = max(0.5, fst_sum / fst_cnt)
            return fst_adv_ratio
        elif fst_cnt == 0:
            snd_adv_ratio = max(0.5, snd_sum / snd_cnt)
            return 1.0 - snd_adv_ratio
        else:
            return float(fst_sum) / float(fst_sum + snd_sum)

    @staticmethod
    def _small_size_adjust_direct(fst_cnt, fst_sum, snd_cnt, snd_sum):
        """assumed (fst_sum + snd_sum) <= 5"""

        def get_adjust_value():
            all_cnt = fst_cnt + snd_cnt
            if all_cnt == 5:
                return 0.04
            if all_cnt == 4:
                return 0.08
            if all_cnt == 3:
                return 0.12
            if all_cnt == 2:
                return 0.17
            if all_cnt == 1:
                return 0.24
            return 0.0

        core_val = Calc._direct_core(fst_cnt, fst_sum, snd_cnt, snd_sum)
        adjust_value = get_adjust_value()
        if core_val > 0.5:
            result = max(0.5, core_val - adjust_value)
        elif core_val < 0.5:
            result = min(0.5, core_val + adjust_value)
        else:
            result = core_val
        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
