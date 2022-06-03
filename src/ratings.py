from dataclasses import dataclass
from typing import Optional, Union

import common as co
import ratings_elo
import ratings_std
from side import Side


@dataclass
class CompareResult:
    """ рез-т сравнения r1, r2 игроков матча, или ri с value.
        Для ответов на запросы типа 'имеет ли side преимущество dif или более'.
        если у одной стороны было какое-то значение, а у другой стороны был None,
        то в этом объекте отмечается факт преимущества (prefer) а dif тогда None
    """
    left_prefer: Optional[bool]  # None means nobody prefer
    dif: Optional[Union[float, int]]  # мера преимущества prefer (in ranks or in points)

    @property
    def right_prefer(self):
        if self.left_prefer is not None:
            return not self.left_prefer

    def side_prefer(self, side: Side, dif: Union[float, int]) -> Optional[bool]:
        if self.left_prefer is None:
            return None  # ни у кого нет преимущества
        if side.is_left() != self.left_prefer:
            return False
        if self.dif is None:
            return True
        return self.dif >= dif


def compare_result(left_value, right_value, inranks=True) -> CompareResult:
    """ inranks == True: prefer smaller value. Otherwise prefer bigger value """
    if left_value is None and right_value is None:
        return CompareResult(None, None)
    if left_value is None:
        return CompareResult(left_prefer=False, dif=None)
    if right_value is None:
        return CompareResult(left_prefer=True, dif=None)

    if left_value < right_value:
        return CompareResult(left_prefer=inranks, dif=right_value - left_value)
    elif left_value > right_value:
        return CompareResult(left_prefer=not inranks, dif=left_value - right_value)
    else:
        return CompareResult(left_prefer=None, dif=0)


# for one Player rtg_name + "_" + surface -> (pts, rank)
class Rating(dict):
    rtg_names = {"std"}  # registered for read operation below

    @staticmethod
    def register_rtg_name(rtg_name):
        Rating.rtg_names.add(rtg_name)

    @staticmethod
    def _make_key(rtg_name, surface="all"):
        return f"{rtg_name}_{surface}"

    def __init__(self, items=None):
        super(Rating, self).__init__(items if items else [])

    def read(self, sex, player_id, surfaces=("all",), date=None):
        for rtg_name in self.rtg_names:
            for surface in surfaces:
                key = self._make_key(rtg_name, surface)
                if rtg_name == "elo":
                    pts = ratings_elo.get_pts(
                        sex, player_id, surface=surface, isalt=False
                    )
                    rank = ratings_elo.get_rank(
                        sex, player_id, surface=surface, isalt=False
                    )
                    self[key] = (pts, rank)
                elif rtg_name == "elo_alt":
                    pts = ratings_elo.get_pts(
                        sex, player_id, surface=surface, isalt=True
                    )
                    rank = ratings_elo.get_rank(
                        sex, player_id, surface=surface, isalt=True
                    )
                    self[key] = (pts, rank)
                elif rtg_name == "std":
                    pts = ratings_std.get_pts(sex, player_id, date)
                    rank = ratings_std.get_rank(sex, player_id, date)
                    self[key] = (pts, rank)

    def pts(self, rtg_name, surface="all"):
        pts_rank = self.get(self._make_key(rtg_name, surface))
        if pts_rank is None:
            return None
        return pts_rank[0]

    def rank(self, rtg_name, surface="all"):
        pts_rank = self.get(self._make_key(rtg_name, surface))
        if pts_rank is None:
            return None
        return pts_rank[1]

    def cmp_rank(self, value, rtg_name, surface="all") -> CompareResult:
        rank = self.rank(rtg_name, surface=surface)
        return compare_result(rank, value, inranks=True)


def initialize(sex, rtg_names, min_date=None):
    """min_date for only 'std'"""
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        for rtg_name in rtg_names:
            if rtg_name == "elo":
                ratings_elo.initialize(sex)
            elif rtg_name == "std":
                ratings_std.initialize(sex=sex, min_date=min_date)


def clear(sex, rtg_names):
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        for rtg_name in rtg_names:
            if rtg_name == "elo":
                ratings_elo.clear(sex)
            elif rtg_name == "std":
                ratings_std.clear(sex)


class SideCondition:
    def __init__(self, max_rating, max_dif_rating, rtg_name="std"):
        self.max_rating = max_rating  # дальний порог, за которым неопределенность
        self.max_dif_rating = max_dif_rating
        self.rtg_name = rtg_name

    def condition(self, match, back_side):
        """return True if ok (back_side лучше чем оппонент);
                  None if uncertancy (нет ясности);
                  False (back_side хуже чем оппонент)
       """
        back_rank, opp_rank = self._get_back_oppose(match, back_side)
        if back_rank is None or opp_rank is None:
            return None
        if back_rank > self.max_rating:
            return False
        if back_rank > (opp_rank + self.max_dif_rating):
            return False
        return True

    def condition_both_below(self, match):
        return match.is_ranks_both_below(self.max_rating, rtg_name=self.rtg_name)

    def _get_back_oppose(self, match, back_side):
        if (
            match.first_player is None
            or match.first_player.rating.rank(self.rtg_name) is None
        ):
            return None, None
        if (
            match.second_player is None
            or match.second_player.rating.rank(self.rtg_name) is None
        ):
            return None, None

        fst_rank, snd_rank = (
            match.first_player.rating.rank(self.rtg_name),
            match.second_player.rating.rank(self.rtg_name),
        )
        back_rank, opp_rank = fst_rank, snd_rank
        if back_side == co.RIGHT:
            back_rank, opp_rank = snd_rank, fst_rank
        return back_rank, opp_rank
