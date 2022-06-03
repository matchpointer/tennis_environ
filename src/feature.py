from typing import Tuple, List
from collections import UserList

import common as co
import clf_common as cco


class Feature(object):
    def __init__(self, name, value, flip_value):
        self.name = name
        self._value = value
        self._flip_value = flip_value
        self.fliped = False

    @property
    def value(self):
        return self._flip_value if self.fliped else self._value

    def set_values(self, value, flip_value):
        self._value = value
        self._flip_value = flip_value

    def __str__(self):
        if self.value is None:
            return cco.NAN
        return str(self.value)

    def tostring(self, extended=False):
        if extended:
            return "{} is_fl={} val={} fl_val={}".format(
                self.name, self.fliped, self._value, self._flip_value
            )
        return "{}={}".format(self.name, str(self))

    def csv_value(self) -> str:
        val = self.value
        if val is None:
            return "NaN"
        if isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, bool):
            return f"{int(val)}"
        return f"{val}"

    def flip_side(self):
        self.fliped = not self.fliped

    def __bool__(self):
        return not self.empty()

    __nonzero__ = __bool__

    def empty(self):
        return self.value is None


class RigidFeature(Feature):
    """single value feature without flip"""

    def __init__(self, name, value=None):
        super().__init__(name, value, flip_value=None)

    def flip_side(self):
        pass


def make_pair(
    fst_name: str, snd_name: str, fst_value=None, snd_value=None
) -> Tuple[Feature, Feature]:
    fst_feat = Feature(name=fst_name, value=fst_value, flip_value=snd_value)
    snd_feat = Feature(name=snd_name, value=snd_value, flip_value=fst_value)
    return fst_feat, snd_feat


def make_pair2(
    corename: str, fst_value=None, snd_value=None
) -> Tuple[Feature, Feature]:
    fst_feat = Feature(name=f"fst_{corename}", value=fst_value, flip_value=snd_value)
    snd_feat = Feature(name=f"snd_{corename}", value=snd_value, flip_value=fst_value)
    return fst_feat, snd_feat


def make_empty_pair(corename: str) -> Tuple[Feature, Feature]:
    fst_feat = RigidFeature(name=f"fst_{corename}", value=None)
    snd_feat = RigidFeature(name=f"snd_{corename}", value=None)
    return fst_feat, snd_feat


def add_pair(features: List[Feature], corename: str, fst_value=None, snd_value=None):
    f1, f2 = make_pair(
        fst_name=f"fst_{corename}",
        snd_name=f"snd_{corename}",
        fst_value=fst_value,
        snd_value=snd_value,
    )
    features.append(f1)
    features.append(f2)


def all_valued(features: List[Feature]) -> bool:
    return all([not f.empty() for f in features])


def all_empty(features: List[Feature]) -> bool:
    return all([f.empty() for f in features])


def player_features(features: List[Feature], name: str) -> Tuple[Feature, Feature]:
    f1 = co.find_first(features, lambda f: f.name == f"fst_{name}")
    if not f1:
        raise cco.FeatureError("not found/empty {}".format(f"fst_{name}"))
    f2 = co.find_first(features, lambda f: f.name == f"snd_{name}")
    if not f2:
        raise cco.FeatureError("not found/empty {}".format(f"snd_{name}"))
    return f1, f2


def dif_player_features(features: List[Feature], name: str):
    fst_feat, snd_feat = player_features(features, name)
    if fst_feat and snd_feat:
        return snd_feat.value - fst_feat.value
    raise cco.FeatureError("bad dif features {}".format(name))


def player_feature(features: List[Feature], name: str, is_first: bool) -> Feature:
    prefix = "fst_" if is_first else "snd_"
    feat = co.find_first(features, lambda f: f.name == f"{prefix}{name}")
    if feat is None or feat.value is None:
        raise cco.FeatureError(f"not found/empty {prefix}{name}")
    return feat


def is_player_feature(features: List[Feature], name: str, is_first: bool) -> bool:
    prefix = "fst_" if is_first else "snd_"
    feat = co.find_first(features, lambda f: f.name == f"{prefix}{name}")
    return feat is not None and not feat.empty()


# Creating a List where deletion is not allowed
class FeatureList(UserList):
    def __init__(self, feat_list: List[Feature]):
        super().__init__(feat_list)
        self.is_fliped = False

    def flip_side(self):
        for feat in self:
            feat.flip_side()
        self.is_fliped = not self.is_fliped

    # Function to stop deletion from List
    def remove(self, s=None):
        raise RuntimeError("Deletion not allowed")

    # Function to stop pop from List
    def pop(self, s=None):
        raise RuntimeError("Deletion not allowed")


class CsvWriter:
    sep = ","

    def __init__(self, filename):
        self.filename = filename
        self.heads = []
        self.n_columns = 0
        self.rows: List[str] = []

    def put_row(self, features: FeatureList):
        # n_columns = len(features)
        # if self.n_columns > 0 and self.n_columns != n_columns:
        #     raise ValueError(f"err csvwriter {self.filename} 1st n_col {self.n_columns}"
        #                      f" != cur n_col {n_columns}")
        row: str = self.sep.join([f.csv_value() for f in features])
        self.rows.append(row + "\n")
        if not self.heads:
            self.heads = [f.name for f in features]

    def put_row_parts(self, features: FeatureList, features_part2: FeatureList):
        row: str = self.sep.join([f.csv_value() for f in features])
        row_part2: str = self.sep.join([f.csv_value() for f in features_part2])
        self.rows.append(row + "," + row_part2 + "\n")
        if not self.heads:
            self.heads = [f.name for f in features]
            for feat in features_part2:
                self.heads.append(feat.name)

    def write(self):
        with open(self.filename, mode="w") as f:
            f.write(self.sep.join(self.heads))
            f.write("\n")
            f.writelines(self.rows)
