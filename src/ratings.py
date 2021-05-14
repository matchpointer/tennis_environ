# -*- coding: utf-8 -*-

import common as co
import ratings_elo
import ratings_std

ABOVE = "ABOVE"
BELOW = "BELOW"


# for one Player rtg_name + "_" + surface -> (pts, rank)
class Rating(dict):
    rtg_names = {"std"}  # registered for read operation below

    @staticmethod
    def register_rtg_name(rtg_name):
        Rating.rtg_names.add(rtg_name)

    @staticmethod
    def _make_key(rtg_name, surface="all"):
        return rtg_name + "_" + str(surface)

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

    def cmp_rank(self, value, rtg_name, surface="all"):
        rank = self.rank(rtg_name, surface=surface)
        if rank is None or value is None:
            return None
        return ABOVE if rank <= value else BELOW


def initialize(sex, rtg_names, min_date=None):
    """min_date for only 'std'"""
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        for rtg_name in rtg_names:
            if rtg_name == "elo":
                ratings_elo.initialize(sex)
            elif rtg_name == "std":
                ratings_std.initialize(sex, min_date)


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
        self.max_rating = max_rating
        self.max_dif_rating = max_dif_rating
        self.rtg_name = rtg_name

    def condition(self, match, back_side):
        """:return True if ok. None if uncertancy. False if not applied"""
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
