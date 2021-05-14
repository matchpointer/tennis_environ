# -*- coding: utf-8 -*-
import datetime
import unittest
from collections import namedtuple

import log
import common as co
import score as sc
import stat_cont as st
from tennis import Level, get_rnd_metric
import weeked_tours
import matchstat
import feature
import tennis_time as tt


def add_prevyear_tour_features(features, sex, ywn_hist_lst, tour, match):
    """match - curent target match where players should be estimation.
    ywn_hist_lst - начиная с текущей недели match назад на нужную глубину.
      (ywn_hist_lst[0] supposed is current match yweek)
    Before call weeked_tours must be initialized from necessary past weeks.
    """

    def prev_year_tour():
        for ywn in ywn_slice:
            for tr in weeked_tours.tours(sex, ywn):
                if (
                    tr.name == tour.name
                    and tr.level == tour.level
                    and tr.surface == tour.surface
                ):
                    return tr

    def prev_year_rnd(pid):
        prev_year_t = prev_year_tour()
        if prev_year_t is not None:
            for mch, rn in prev_year_t.match_rnd_list(reverse=True):  # time desc sorted
                if mch.paired():
                    continue
                if pid in (mch.first_player.ident, mch.second_player.ident):
                    return rn

    weeksago1 = 49
    weeksago2 = 56
    if tour.level in ("team", "teamworld") or len(ywn_hist_lst) < (weeksago1 + 1):
        feature.add_pair(features, "prevyear_tour_rnd", fst_value=0.0, snd_value=0.0)
        return
    ywn_slice = ywn_hist_lst[weeksago1 : min(len(ywn_hist_lst), weeksago2)]
    fst_rnd = prev_year_rnd(match.first_player.ident)
    snd_rnd = prev_year_rnd(match.second_player.ident)

    if fst_rnd is None:
        fst_metric = 0.0
    else:
        fst_rnd.detailing = None
        fst_metric = get_rnd_metric(fst_rnd)

    if snd_rnd is None:
        snd_metric = 0.0
    else:
        snd_rnd.detailing = None
        snd_metric = get_rnd_metric(snd_rnd)
    feature.add_pair(
        features, "prevyear_tour_rnd", fst_value=fst_metric, snd_value=snd_metric
    )


def add_stat_features(
    features,
    sex,
    ywn_hist_lst,
    match,
    tour,
    rnd,
    stat_names,
    min_stat_size=5,
    stop_stat_size=12,
):
    """tour, rnd, match - curent match.
    ywn_hist_lst - начиная с текущей недели match назад на нужную глубину.
      (ywn_hist_lst used, ywn_hist_lst[0] supposed is current match yweek)
    Before call weeked_tours must be initialized from necessary past weeks.
    """
    feat_getter = StatFeaturesGetter(
        sex,
        stat_names=stat_names,
        min_stat_size=min_stat_size,
        stop_stat_size=stop_stat_size,
    )
    new_features = feat_getter.get_features(ywn_hist_lst, match, tour=tour, rnd=rnd)
    features.extend(new_features)


class StatFeaturesGetter(object):
    def __init__(
        self,
        sex,
        stat_names=("service_win", "receive_win"),
        min_stat_size=5,
        stop_stat_size=12,
    ):
        self.sex = sex
        self.stat_names = stat_names
        self.min_stat_size = min_stat_size
        self.stop_stat_size = stop_stat_size

    def get_features(self, ywn_hist_lst, match, tour=None, rnd=None):
        """return list of features"""

        def add_features(name, fst_sumator, snd_sumator):
            if (
                fst_sumator.size >= self.min_stat_size
                and snd_sumator.size >= self.min_stat_size
            ):
                feature.add_pair(
                    features,
                    name,
                    fst_value=fst_sumator.average(),
                    snd_value=snd_sumator.average(),
                )
            else:
                feature.add_pair(features, name)

        def player_stat(mstat, stat_name, side):
            method = getattr(mstat, stat_name)
            fst_snd = method()
            if fst_snd is None:
                return None
            fst_obj, snd_obj = fst_snd
            return fst_obj if side.is_left() else snd_obj

        def visit_stat_match(mch, coef, stat_name, fst_sum, snd_sum):
            if (
                mch.stat is None
                or mch.paired()
                or mch.score.sets_count(full=True) < 2
                or not mch.stat.is_total_points_won()
            ):
                return
            if match.first_player.ident in (
                mch.first_player.ident,
                mch.second_player.ident,
            ):
                side = co.side(match.first_player.ident == mch.first_player.ident)
                fst_obj = player_stat(mch.stat, stat_name, side)
                if (
                    fst_obj
                    and fst_obj.value < 1.01
                    and fst_sum.size < self.stop_stat_size
                ):
                    fst_sum += fst_obj.value * coef
            if match.second_player.ident in (
                mch.first_player.ident,
                mch.second_player.ident,
            ):
                side = co.side(match.second_player.ident == mch.first_player.ident)
                snd_obj = player_stat(mch.stat, stat_name, side)
                if (
                    snd_obj
                    and snd_obj.value < 1.01
                    and snd_sum.size < self.stop_stat_size
                ):
                    snd_sum += snd_obj.value * coef

        features = []
        for stat_name in self.stat_names:
            fst_sum, snd_sum = st.Sumator(), st.Sumator()
            for tr in weeked_tours.tours(self.sex, ywn_hist_lst[0]):  # cur match week
                coef = matchstat.generic_surface_trans_coef(
                    self.sex, stat_name, str(tr.surface), str(tour.surface)
                )
                for mch, rn in tr.match_rnd_list():
                    if mch.date is not None and match.date is not None:
                        isvisitable = mch.date < match.date
                    else:
                        isvisitable = rn < rnd
                    if isvisitable:
                        visit_stat_match(mch, coef, stat_name, fst_sum, snd_sum)

            for ywn in ywn_hist_lst[1:]:  # strong before current match weeks
                if (
                    fst_sum.size >= self.stop_stat_size
                    and snd_sum.size >= self.stop_stat_size
                ):
                    break
                for tr in weeked_tours.tours(self.sex, ywn):
                    coef = matchstat.generic_surface_trans_coef(
                        self.sex, stat_name, str(tr.surface), str(tour.surface)
                    )
                    for mch in tr.matches(reverse=True):
                        visit_stat_match(mch, coef, stat_name, fst_sum, snd_sum)

            add_features(stat_name, fst_sum, snd_sum)
        return features


def fatig_rnd_coef(sex, rnd):
    if rnd in ("Final", "Bronze"):
        return 1.2 if sex == "wta" else 1.15
    elif rnd == "1/2":
        return 1.1 if sex == "wta" else 1.09
    else:
        return 1.0


def fatig_lsr_coef(sex, tour, rnd):
    weight = 1.0
    grand_slam_mdraw = (
        tour.grand_slam() and not rnd.qualification() and not rnd.pre_qualification()
    )
    if tour.level in (Level("team"), Level("teamworld")):
        weight = 1.4 if tour.surface == "Clay" else 1.3
    elif grand_slam_mdraw:
        weight = 1.25 if tour.surface == "Clay" else 1.18
    elif not grand_slam_mdraw and tour.surface == "Clay":
        weight = 1.1

    # season weighting
    weeknum = tour.year_weeknum[1]
    weeknums_in_year = tt.weeknums_in_year(tour.date.year)
    season_weight = 1.0 + 0.2 * (float(weeknum) / float(weeknums_in_year))
    weight *= season_weight

    weight *= fatig_rnd_coef(sex, rnd)
    return weight


def fatig_summary_coef(params_key, sex, daysago, tour, rnd, paired):
    sex_params = fatig_params[params_key][sex]
    discount_coef = sex_params["discount_coef_fun"](daysago)
    if paired:
        paired_coef = sex_params["paired_coef"]
    else:
        paired_coef = 1.0
    lsr_coef = sex_params["lsr_coef_fun"](sex, tour, rnd)
    return discount_coef * paired_coef * lsr_coef


fatig_params = {
    "default": {
        "feature_suffix": "",
        "wta": {
            "discount_coef_fun": lambda d: 0.75 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=3),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.75 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=3),
        },
    },
    "weeksago1": {
        "feature_suffix": "_w1",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
    },
    "weeksago2": {
        "feature_suffix": "_w2",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
    },
    "daysago11": {
        "feature_suffix": "_d11",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
    },
    "weeksago1_disc": {
        "feature_suffix": "_w1_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
    },
    "weeksago2_disc": {
        "feature_suffix": "_w2_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
    },
    "daysago11_disc": {
        "feature_suffix": "_d11_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": lambda s, t, r: 1.0,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
    },
    "weeksago1_weight": {
        "feature_suffix": "_w1_weight",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
    },
    "weeksago2_weight": {
        "feature_suffix": "_w2_weight",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
    },
    "daysago11_weight": {
        "feature_suffix": "_d11_weight",
        "wta": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
        "atp": {
            "discount_coef_fun": lambda d: 1.0,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
    },
    "weeksago1_weight_disc": {
        "feature_suffix": "_w1_weight_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[1]),
        },
    },
    "weeksago2_weight_disc": {
        "feature_suffix": "_w2_weight_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: tt.get_monday(*ywns[2]),
        },
    },
    "daysago11_weight_disc": {
        "feature_suffix": "_d11_weight_disc",
        "wta": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
        "atp": {
            "discount_coef_fun": lambda d: 0.82 ** d,
            "paired_coef": 0.57,
            "lsr_coef_fun": fatig_lsr_coef,
            "mindate_fun": lambda md, ywns: md - datetime.timedelta(days=11),
        },
    },
}


def add_fatigue_features(features, sex, ywn_hist_lst, match, rnd, params_key="default"):
    """match, rnd - curent target match where players should be fatigue estimation.
    If rnd is None then live mode (without concurrent matches in tour/week).
    ywn_hist_lst - начиная с текущей недели match назад на нужную глубину.
      (ywn_hist_lst used, ywn_hist_lst[0] supposed is current match yweek)
    Before call weeked_tours must be initialized from necessary past weeks.
    """
    if match.date is None:
        feature.add_pair(
            features, "fatigue" + fatig_params[params_key]["feature_suffix"]
        )
        return
    min_match_date = fatig_params[params_key][sex]["mindate_fun"](
        match.date, ywn_hist_lst
    )
    fst_games, snd_games = 0.0, 0.0
    for ywn in ywn_hist_lst[
        0:3
    ]:  # begin from current week ([0]), and 2 weeks to the past
        for tr in weeked_tours.tours(sex, ywn):
            for mch, rn in tr.match_rnd_list(reverse=True):  # time desc sorted
                if mch.date is None or mch.date > match.date:
                    continue
                elif mch.date == match.date:
                    if rnd is None:
                        continue
                    if mch.paired():
                        continue  # doubles usially play after target singles
                    if (
                        mch.first_player.ident == match.first_player.ident
                        and mch.second_player.ident == match.second_player.ident
                    ):
                        continue  # avoid the 'self' match
                    if (
                        mch.first_player.ident == match.second_player.ident
                        and mch.second_player.ident == match.first_player.ident
                    ):
                        continue  # avoid the 'self' fliped match
                    if rn > rnd:
                        continue
                elif mch.date < min_match_date:
                    break
                fst_games, snd_games = _visit_fatigue_match(
                    sex, params_key, tr, rn, mch, match, fst_games, snd_games
                )
    feature.add_pair(
        features,
        "fatigue" + fatig_params[params_key]["feature_suffix"],
        fst_value=fst_games,
        snd_value=snd_games,
    )


def _visit_fatigue_match(
    sex, params_key, tour, rnd, match, trg_match, fst_games, snd_games
):
    """tour, rnd, match - played for visiting. trg_match - for fatigue estimate"""
    side_left, side_right, paired = False, False, False
    if match.paired():
        paired = True
        if match.is_paired_participant(trg_match.first_player):
            side_left = True
        if match.is_paired_participant(trg_match.second_player):
            side_right = True
    else:
        if trg_match.first_player.ident in (
            match.first_player.ident,
            match.second_player.ident,
        ):
            side_left = True
        if trg_match.second_player.ident in (
            match.first_player.ident,
            match.second_player.ident,
        ):
            side_right = True
    if not side_left and not side_right:
        return fst_games, snd_games
    daysago = (trg_match.date - match.date).days
    if paired:
        total = sc.paired_match_games_count(tour, match.score, load_mode=True)
    else:
        total = match.score.games_count(load_mode=True)
    coef = fatig_summary_coef(params_key, sex, daysago, tour, rnd, paired)
    value = total * coef
    if side_left:
        fst_games += value
    if side_right:
        snd_games += value
    return fst_games, snd_games


# -------------- tour adapt feature part: ---------------------

tail_tours_from_sex = {}  # sex -> [tail_tours]


def init_tail_tours(sex):
    tail_tours_from_sex[sex] = weeked_tours.tail_tours(sex, tail_weeks=2)
    if not tail_tours_from_sex[sex]:
        log.error("fail init_tail_tours sex: {}".format(sex))


def find_tail_tour_by_id(sex, tour_id):
    if sex not in tail_tours_from_sex:
        init_tail_tours(sex)

    return co.find_first(tail_tours_from_sex[sex], lambda t: t.ident == tour_id)


def find_tail_tour_by_attr(sex, tour_name, level, surface):
    if sex not in tail_tours_from_sex:
        init_tail_tours(sex)

    if not sex or not tour_name or not level or not surface:
        msg = (
            "nofull attr find_tail_tour_by_attr sex: {} tn: {} lev: {} surf: {}".format(
                sex, tour_name, level, surface
            )
        )
        log.error(msg)
    if level == "main":
        eq_cond = (
            lambda t: t.name == tour_name
            and t.level in ("main", "masters")
            and t.surface == surface
        )
    else:
        eq_cond = (
            lambda t: t.name == tour_name and t.level == level and t.surface == surface
        )
    tailtours = co.find_all(tail_tours_from_sex[sex], eq_cond)
    if len(tailtours) == 1:
        return tailtours[0]
    elif len(tailtours) == 2:
        t1, t2 = tailtours[0], tailtours[1]
        if t1.date != t2.date:
            return t1 if t1.date > t2.date else t2


class WeekedToursTest(unittest.TestCase):
    def test_tail_tours_write(self):
        weeked_tours.dump_tail_tours(sex="wta", tail_weeks=2)
        weeked_tours.dump_tail_tours(sex="atp", tail_weeks=2)
        self.assertTrue(1)

    def test_tours_write(self):
        from tournament import tours_write_file

        wta_tours = weeked_tours.all_tours(sex="wta")
        atp_tours = weeked_tours.all_tours(sex="atp")
        tours_write_file(wta_tours, filename="./get_features_test_wta_tours.txt")
        tours_write_file(atp_tours, filename="./get_features_test_atp_tours.txt")
        self.assertTrue(1)

    def test_prevyear_tour_rnd_for_2020_begin_adhoc_only(self):
        from live import back_year_weeknums

        today_date = datetime.date.today()
        if (today_date.year, today_date.month) == (2020, 1) and today_date.day <= 15:
            FakedTour = namedtuple("FakedTour", "name level surface")
            FakedPlayer = namedtuple("FakedPlayer", "ident name")
            FakedMatch = namedtuple("FakedMatch", "first_player second_player")
            vekic = FakedPlayer(ident=13498, name="Donna Vekic")
            putintseva = FakedPlayer(ident=12390, name="Yulia Putintseva")
            match = FakedMatch(first_player=vekic, second_player=putintseva)
            tour = FakedTour(name="Brisbane", level="main", surface="Hard")
            features = []
            add_prevyear_tour_features(
                features, "wta", back_year_weeknums(max_weeknum_dist=56), tour, match
            )
            self.assertTrue(features)
            if features:
                f1 = feature.player_feature(
                    features, "prevyear_tour_rnd", is_first=True
                )
                f2 = feature.player_feature(
                    features, "prevyear_tour_rnd", is_first=False
                )
                self.assertTrue(
                    abs(f1.value - 4.2) < 0.001
                )  # vekic in 1/2 prevyear brisbane
                self.assertTrue(
                    abs(f2.value - 0.9) < 0.001
                )  # putintseva in First prevyear brisbane
        else:
            self.assertTrue(True)


def add_tour_adapt_features(features, trg_tour, trg_match, min_value=-60):
    f1, f2 = tour_adapt_features(trg_tour, trg_match, min_value)
    features.append(f1)
    features.append(f2)


def tour_adapt_features(trg_tour, trg_match, min_value=-60):
    """fst_ply, snd_ply get +n if ply played n single matches in tour before trg_match.
    If no single matches in tour before trg_match,
                                     then get -m passed days after last single match
    return (fst_plr_feat, snd_plr_feat).
    trg_tour is None -> try find tour from tail_tours by trg_match.live_event.tour_id
    """

    def trg_tour_ply_value(player):
        n_matches = 0
        for mch, rnd in trg_tour.match_rnd_list(
            predicate=lambda m: m.rnd is not None and m.date, reverse=False
        ):
            if (mch.date >= trg_match.date and trg_match.date) or (
                rnd >= trg_match.rnd and trg_match.rnd is not None
            ):
                break
            if (
                mch.score is not None
                and mch.score.sets_count(full=True) >= 1
                and player in (mch.first_player, mch.second_player)
                and not mch.paired()
            ):
                n_matches += 1
        return n_matches

    def before_trg_tour_ply_value(player):
        prev_year_weeknum = tt.year_weeknum_prev(trg_tour.year_weeknum)
        for year_weeknum in tt.year_weeknum_reversed(
            prev_year_weeknum, max_weeknum_dist=9
        ):
            for tour in weeked_tours.tours(trg_tour.sex, year_weeknum):
                for mch, _ in tour.match_rnd_list(
                    predicate=lambda m: m.rnd is not None and m.date, reverse=True
                ):
                    if (
                        mch.score is not None
                        and mch.score.sets_count(full=True) >= 1
                        and player in (mch.first_player, mch.second_player)
                        and not mch.paired()
                    ):
                        value = -(trg_match.date - mch.date).days
                        return max(value, min_value)
        return min_value

    if trg_tour is None:
        # try find tour from tail_tours by trg_match.live_event.tour_id
        if trg_match.live_event.tour_id is not None:
            trg_tour = find_tail_tour_by_id(trg_match.sex, trg_match.live_event.tour_id)
        else:
            trg_tour = find_tail_tour_by_attr(
                trg_match.sex,
                trg_match.tour_name,
                str(trg_match.level),
                str(trg_match.surface),
            )
        if trg_tour is None:
            msg = (
                "fail find_tail_tour id: {} for match {}\n\t"
                + "sex: {} tourname: {} level: {} surf: {} qual: {}"
            ).format(
                trg_match.live_event.tour_id,
                trg_match.tostring(extended=True),
                trg_match.sex,
                trg_match.tour_name,
                trg_match.level,
                trg_match.surface,
                trg_match.qualification,
            )
            log.error(msg)
            return feature.make_pair(
                "fst_plr_tour_adapt", "snd_plr_tour_adapt", None, None
            )
        return tour_adapt_features(trg_tour, trg_match, min_value)

    fst_plr_val = trg_tour_ply_value(trg_match.first_player)
    snd_plr_val = trg_tour_ply_value(trg_match.second_player)
    if fst_plr_val == 0:
        fst_plr_val = before_trg_tour_ply_value(trg_match.first_player)
    if snd_plr_val == 0:
        snd_plr_val = before_trg_tour_ply_value(trg_match.second_player)
    return feature.make_pair(
        "fst_plr_tour_adapt",
        "snd_plr_tour_adapt",
        fst_value=fst_plr_val,
        snd_value=snd_plr_val,
    )


if __name__ == "__main__":
    import dba
    import ratings_std
    import oncourt_players
    import bet_coefs

    log.initialize(
        co.logname(__file__, test=True), file_level="debug", console_level="debug"
    )
    dba.open_connect()
    date = tt.past_monday_date(datetime.date.today())
    ratings_std.initialize(min_date=datetime.date.today() - datetime.timedelta(days=21))
    oncourt_players.initialize(yearsnum=1.2)
    matchstat.initialize(
        min_date=date - datetime.timedelta(days=7 * matchstat.HIST_WEEKS_NUM),
        time_reverse=True,
        tour_surface=True,
    )

    bet_coefs.initialize(sex="wta", min_date=date - datetime.timedelta(days=7))
    bet_coefs.initialize(sex="atp", min_date=date - datetime.timedelta(days=7))
    # for fatigue, live-matches details(rnd-details, tour.level, tour.surface):
    weeked_tours.initialize_sex(
        "wta",
        min_date=date - datetime.timedelta(days=7 * 55),
        max_date=date + datetime.timedelta(days=11),
        with_today=True,
        with_paired=True,
        with_ratings=True,
        with_bets=True,
        with_stat=True,
        rnd_detailing=True,
    )
    weeked_tours.initialize_sex(
        "atp",
        min_date=date - datetime.timedelta(days=7 * 55),
        max_date=date + datetime.timedelta(days=11),
        with_today=True,
        with_paired=True,
        with_ratings=True,
        with_bets=True,
        with_stat=True,
        rnd_detailing=True,
    )
    unittest.main()
    dba.close_connect()
