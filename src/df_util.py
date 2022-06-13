# -*- coding=utf-8 -*-
import os
from typing import Optional

import pandas as pd

import pandemia


def stage_shuffle(df: pd.DataFrame, is_shuffle: bool = True, random_state=None):
    if is_shuffle:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def get_hash(df: pd.DataFrame, key_names):
    lst = []
    for i in df.index:
        arr = df.loc[i, key_names].values
        lst.append(hash(tuple(arr)))
    return hash(tuple(lst))


def drop_by_condition(df: pd.DataFrame, expr_fun):
    """expr_fun is fun(df) -> expression (use as: df[expr] )"""
    idx_todel = df[expr_fun(df)].index
    df.drop(idx_todel, inplace=True)


def drop_rank_std_any_below_or_wide_dif(
        df: pd.DataFrame, rank_std_both_above: int, rank_std_max_dif: int):
    drop_by_condition(
        df,
        lambda d: (
            (d["fst_std_rank"] > rank_std_both_above)
            | (d["snd_std_rank"] > rank_std_both_above)
            | ((d["fst_std_rank"] - d["snd_std_rank"]).abs() > rank_std_max_dif)
        ),
    )


def save_df(df: pd.DataFrame, storage_dir: str, subname: str) -> None:
    filename = os.path.join(storage_dir, subname, "df.csv")
    df.to_csv(filename, index=False)


def load_df(storage_dir: str, subname: str) -> Optional[pd.DataFrame]:
    if storage_dir:
        filename = os.path.join(storage_dir, subname, "df.csv")
        if os.path.isfile(filename):
            return pd.read_csv(filename, sep=",")
    return None


def substract_df(df: pd.DataFrame, df_substr: pd.DataFrame, inplace=True):
    """:return df where removed row (matches) from df_substr"""
    key_names = ["date", "fst_pid", "snd_pid", "rnd_text"]
    substr_keys = list(map(tuple, df_substr[key_names].values))
    del_idxes = [
        i
        for i in df.index
        if (
            df.loc[i, "date"],
            df.loc[i, "fst_pid"],
            df.loc[i, "snd_pid"],
            df.loc[i, "rnd_text"],
        )
        in substr_keys
    ]
    if inplace:
        df.drop(del_idxes, inplace=True)
    else:
        return df.drop(del_idxes, inplace=False)


def substract_pandemia(df: pd.DataFrame, inplace=True):
    """ :return df where removed pandemia date records """
    min_date_str = str(pandemia.min_date())
    max_date_str = str(pandemia.max_date())
    del_idxes = [
        i
        for i in df.index
        if min_date_str <= df.loc[i, "date"] <= max_date_str
    ]
    if inplace:
        df.drop(del_idxes, inplace=True)
    else:
        return df.drop(del_idxes, inplace=False)
