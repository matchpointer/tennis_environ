# -*- coding=utf-8 -*-
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

import common as co
from clf_common import out, WeightMode


def edit_column_value(df, column_name, value_fun):
    def process_row(row):
        value = row[column_name]
        return value_fun(value)

    df[column_name] = df.apply(process_row, axis=1)


def replace_column_empty_value(df, column_name, replace_value=0):
    def process_row(row):
        value = row[column_name]
        return replace_value if pd.isnull(value) else value

    df[column_name] = df.apply(process_row, axis=1)


def add_column(df, new_column_name, fun_by_row):
    def process_row(row):
        return fun_by_row(row)

    df[new_column_name] = df.apply(process_row, axis=1)


def edit_column(df, column_name, fun_by_fld):
    def process_field(fld):
        return fun_by_fld(fld)

    df[column_name] = df[column_name].apply(process_field)


def round_column(df, column_name, ndigits=0):
    def process_field(fld):
        return round(fld, ndigits)

    df[column_name] = df[column_name].apply(process_field)


def add_weight_column(weight_mode: WeightMode, df, label_name):
    if "weight" in df.columns:
        return
    if weight_mode == WeightMode.BALANCED:
        df["weight"] = compute_sample_weight("balanced", df[label_name])
    elif weight_mode == WeightMode.BY_TOTAL_COLUMN:
        dct = {6: 1.3, 7: 1.2, 8: 1.1, 9: 0.9, 10: 0.8, 12: 0.7}
        df["weight"] = compute_sample_weight(dct, df["total_dec"])


def add_weighted_column(
    df, new_column_name, fst_column_name, fst_weight, snd_column_name, snd_weight
):
    def process_row(row):
        return row[fst_column_name] * fst_weight + row[snd_column_name] * snd_weight

    new_column_name = "{0}_{1:.2f}_{2:.2f}".format(
        new_column_name, fst_weight, snd_weight
    ).replace(".", "")
    df[new_column_name] = df.apply(process_row, axis=1)


def with_nan_columns(df, columns=None, raise_ifnan=False):
    """return list of column names with nan"""
    result = []
    check_columns = columns if columns else df.columns
    for col in check_columns:
        if df[col].isnull().any():
            result.append(str(col))
    if result:
        err_text = "detected columns with nan: {}".format(result)
        out(err_text)
        if raise_ifnan:
            raise co.TennisError(err_text)
    return result


def add_year_column(df, new_name="year"):
    df[new_name] = pd.DatetimeIndex(df["date"]).year


def make_decided_win_side_by_score(df, side, new_name):
    """make sense decided_win_by side_player"""

    def process_row(row):
        value = row["decided_win_by_set2_winner"]
        if pd.isnull(value):
            return value

        if side.is_left():
            is_set2win_side = row["s2_fst_games"] > row["s2_snd_games"]
        else:
            is_set2win_side = row["s2_snd_games"] > row["s2_fst_games"]
        if is_set2win_side:
            return value
        return 1.0 - value

    df[new_name] = df.apply(process_row, axis=1)


def make_lead2_three_class_label(
    data, set_prefix, column_name, zero_label_pass_games=8
):
    """generate 1 if set opener trailed 2 games first,
    -1 if set closer trailed 2 games first
     0 if 4-4 achieved without anybody leadership with 2 games"""

    def process_row(row):
        if row[set_prefix + "is_1-0"] and row[set_prefix + "is_1-1"] == 0:
            return -1  # 2-0
        if row[set_prefix + "is_1-0"] == 0 and row[set_prefix + "is_1-1"] == 0:
            return 1  # 0-2

        if row[set_prefix + "is_2-1"] and row[set_prefix + "is_2-2"] == 0:
            return -1  # 3-1
        if row[set_prefix + "is_2-1"] == 0 and row[set_prefix + "is_2-2"] == 0:
            return 1  # 1-3

        if row[set_prefix + "is_3-2"] and row[set_prefix + "is_3-3"] == 0:
            return -1  # 4-2
        if row[set_prefix + "is_3-2"] == 0 and row[set_prefix + "is_3-3"] == 0:
            return 1  # 2-4

        if zero_label_pass_games == 6:
            return 0  # here will ratios (0.4303825, 0.14459, 0.4250)
        if row[set_prefix + "is_4-3"] and row[set_prefix + "is_4-4"] == 0:
            return -1  # 5-3
        if row[set_prefix + "is_4-3"] == 0 and row[set_prefix + "is_4-4"] == 0:
            return 1  # 3-5

        if zero_label_pass_games == 8:
            return 0  # here will ratios (0.4651366, 0.07409836, 0.460765)
        if row[set_prefix + "is_5-4"] and row[set_prefix + "is_5-5"] == 0:
            return -1  # 6-4
        if row[set_prefix + "is_5-4"] == 0 and row[set_prefix + "is_5-5"] == 0:
            return 1  # 4-6

        if zero_label_pass_games == 10:
            return 0
        if row[set_prefix + "is_6-5"] and row[set_prefix + "is_6-6"] == 0:
            return -1  # 7-5
        if row[set_prefix + "is_6-5"] == 0 and row[set_prefix + "is_6-6"] == 0:
            return 1  # 5-7
        return 0  # here will ratios (0.4900, 0.020874, 0.489071)

    data[column_name] = data.apply(process_row, axis=1)


def make_two_class_label(data, set_prefix, column_name):
    """generate 1 if set opener trailed 2 games first,
    0 otherwise"""

    def process_row(row):
        if row[set_prefix + "is_1-0"] and row[set_prefix + "is_1-1"] == 0:
            return 0  # 2-0, opener leads
        if row[set_prefix + "is_1-0"] == 0 and row[set_prefix + "is_1-1"] == 0:
            return 1  # 0-2

        if row[set_prefix + "is_2-1"] and row[set_prefix + "is_2-2"] == 0:
            return 0  # 3-1
        if row[set_prefix + "is_2-1"] == 0 and row[set_prefix + "is_2-2"] == 0:
            return 1  # 1-3

        if row[set_prefix + "is_3-2"] and row[set_prefix + "is_3-3"] == 0:
            return 0  # 4-2
        if row[set_prefix + "is_3-2"] == 0 and row[set_prefix + "is_3-3"] == 0:
            return 1  # 2-4

        if row[set_prefix + "is_4-3"] and row[set_prefix + "is_4-4"] == 0:
            return 0  # 5-3
        if row[set_prefix + "is_4-3"] == 0 and row[set_prefix + "is_4-4"] == 0:
            return 1  # 3-5

        if row[set_prefix + "is_5-4"] and row[set_prefix + "is_5-5"] == 0:
            return 0  # 6-4
        if row[set_prefix + "is_5-4"] == 0 and row[set_prefix + "is_5-5"] == 0:
            return 1  # 4-6

        if row[set_prefix + "is_6-5"] and row[set_prefix + "is_6-6"] == 0:
            return 0  # 7-5
        if row[set_prefix + "is_6-5"] == 0 and row[set_prefix + "is_6-6"] == 0:
            return 1  # 5-7
        return 0

    data[column_name] = data.apply(process_row, axis=1)


def make_first_lead2_label(data, set_prefix, is_set_opener_fun, column_name):
    """generate 1 if player lead 2 games first, 0 otherwise.
    whould our player be set opener shows is_set_opener_fun(row).
    """

    def process_row(row):
        result = process_row_tmp(row)
        return result if is_set_opener_fun(row) else 1 - result

    def process_row_tmp(row):
        if not row[set_prefix + "is_1-1"]:
            return row[set_prefix + "is_1-0"]  # if true: 2-0 opener leads
        if not row[set_prefix + "is_2-2"]:
            return row[set_prefix + "is_2-1"]  # if true: 3-1 opener leads
        if not row[set_prefix + "is_3-3"]:
            return row[set_prefix + "is_3-2"]  # if true: 4-2 opener leads
        if not row[set_prefix + "is_4-4"]:
            return row[set_prefix + "is_4-3"]  # if true: 5-3 opener leads
        if not row[set_prefix + "is_5-5"]:
            return row[set_prefix + "is_5-4"]  # if true: 6-4 opener leads
        if not row[set_prefix + "is_6-6"]:
            return row[set_prefix + "is_6-5"]  # if true: 7-5 opener leads
        return 0

    data[column_name] = data.apply(process_row, axis=1)


def make_lead2_label(data, set_prefix, is_set_opener_fun, column_name):
    """generate 1 if player lead 2 games existance, 0 otherwise
    whould our player be set opener shows is_set_opener_fun(row).
    """

    def process_row(row):
        is_opener = is_set_opener_fun(row)
        if not row[set_prefix + "is_1-1"]:
            if (is_opener and row[set_prefix + "is_1-0"]) or (
                not is_opener and row[set_prefix + "is_0-1"]
            ):
                return 1
        if not row[set_prefix + "is_2-2"]:
            if (is_opener and row[set_prefix + "is_2-1"]) or (
                not is_opener and row[set_prefix + "is_1-2"]
            ):
                return 1
        if not row[set_prefix + "is_3-3"]:
            if (is_opener and row[set_prefix + "is_3-2"]) or (
                not is_opener and row[set_prefix + "is_2-3"]
            ):
                return 1
        if not row[set_prefix + "is_4-4"]:
            if (is_opener and row[set_prefix + "is_4-3"]) or (
                not is_opener and row[set_prefix + "is_3-4"]
            ):
                return 1
        if not row[set_prefix + "is_5-5"]:
            if (is_opener and row[set_prefix + "is_5-4"]) or (
                not is_opener and row[set_prefix + "is_4-5"]
            ):
                return 1
        if not row[set_prefix + "is_6-6"]:
            if (is_opener and row[set_prefix + "is_6-5"]) or (
                not is_opener and row[set_prefix + "is_5-6"]
            ):
                return 1
        return 0

    data[column_name] = data.apply(process_row, axis=1)
