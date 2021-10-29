from collections import defaultdict, OrderedDict, Counter
from operator import itemgetter

import common as co
import report_line as rl
import dict_tools


def cntr_to_percent_dict(cntr: Counter, precision: int = 2) -> OrderedDict:
    return OrderedDict(
        [
            (
                key,
                str(round(count / sum(cntr.values()) * 100.0, precision))
                + f"% ({count})",
            )
            for key, count in cntr.most_common()
        ]
    )


def write_cntr_percent_format(cntr: Counter, filename: str, precision: int = 2):
    dct = cntr_to_percent_dict(cntr, precision=precision)
    with open(filename, mode="w") as f:
        for key, value in dct.items():
            f.write(f"{key}___{value}\n")


def less_greater_counts(cntr, front):
    less_cnt, greater_cnt = 0, 0
    if cntr:
        for key, cnt in cntr.items():
            if key < front:
                less_cnt += cnt
            elif key > front:
                greater_cnt += cnt
    return less_cnt, greater_cnt


def less_sized_chance(cntr, front):
    less_count, greater_count = less_greater_counts(cntr, front)
    all_count = less_count + greater_count
    if all_count > 0:
        return rl.SizedValue(float(less_count) / float(all_count), all_count)
    else:
        return rl.SizedValue()


def greater_sized_chance(cntr, front):
    less_count, greater_count = less_greater_counts(cntr, front)
    all_count = less_count + greater_count
    if all_count > 0:
        return rl.SizedValue(float(greater_count) / float(all_count), all_count)
    else:
        return rl.SizedValue()


def sized_avg(cntr):
    """Вернем среднее значение ключей в виде SizedValue"""
    all_count, keys_sum = 0, 0
    if cntr is not None:
        for k, c in cntr.items():
            all_count += c
            keys_sum += k * c
    if all_count > 0:
        return rl.SizedValue(float(keys_sum) / float(all_count), all_count)
    else:
        return rl.SizedValue()


# dict: keyi -> Counter
#               Counter: valuei -> count
def kcntr_dict_inc(dictionary, keys, value, cnt_value=1):
    if dictionary is not None:
        for key in keys:
            dictionary[key][value] += cnt_value


def kcntr_dict_load(
    filename,
    createfun=lambda: defaultdict(lambda: None),
    keyfun=co.StructKey.create_from_text,
    valuefun=None,
    filterfun=None,
):
    return dict_tools.load(
        filename,
        createfun=createfun,
        keyfun=keyfun,
        valuefun=valuefun,
        filterfun=filterfun,
    )


def kcntr_dict_dump(
    dictionary,
    filename,
    keyfun=co.identity,
    valuefun=None,
    sortfun=itemgetter(0),
    filterfun=None,
):
    dict_tools.dump(
        dictionary,
        filename,
        keyfun=keyfun,
        valuefun=valuefun,
        sortfun=sortfun,
        filterfun=filterfun,
    )


def kcntr_dict_values_sum(dictionary, keyselector=None):
    if dictionary is not None:
        values_sum = 0
        for key, cntr in dictionary.items():
            if keyselector is None or keyselector(key):
                values_sum += sum(cntr.values())
        return values_sum
