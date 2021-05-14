# -*- coding: utf-8 -*-
import os
import sys
import datetime
from collections import defaultdict
import argparse
import psutil

import log
import cfg_dir
import common as co
import score as sc
import tennis_time as tt
import dba
import oncourt_players
import ratings_std
import ratings_elo
import tie_spec_view
import weeked_tours
from detailed_score import find_detailed_score2, SetItems, error_contains
from detailed_score_dbsa import open_db
import decided_set
import matchstat
from feature import Feature, RigidFeature, CsvWriter
import get_features
import tie_stat
import bet_coefs
import detailed_score_misc as dsm
import total
import tennis
from tournament import best_of_five2, PandemiaTours
import set2_after_set1loss_stat
import inset_keep_recovery
import trail_choke_stat
import decided_win_by_two_sets_stat
import after_tie_perf_stat
from detailed_score_calc import (
    common_features,
    openspecific_calc_features,
    decspecific_calc_features,
)
import markov
from clf_common import RANK_STD_BOTH_ABOVE, RANK_STD_MAX_DIF


def print_memory_usage(comment=""):
    """Prints current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    total, available, usage_perc, used, free = psutil.virtual_memory()
    total, available, used, free = (
        total / MEGA,
        available / MEGA,
        used / MEGA,
        free / MEGA,
    )
    proc = PROCESS.memory_info()[1] / MEGA
    print(
        f"{comment} process = {round(proc)} total = {round(total)} "
        f"available = {round(available)} used = {round(used)} "
        f"free = {round(free)} usage_perc = {usage_perc}"
    )


def add_total_h2h_feature(features, tour, match, tour_avgset):
    if tour_avgset is None:
        h2h_val, h2h_sets = None, None
    else:
        h2h_val, h2h_sets = match.head_to_head.total_target_evaled(
            tour.best_of_five(match.rnd), tour_avgset
        )
    features.append(RigidFeature("h2h_total", value=h2h_val))
    features.append(RigidFeature("h2h_sets", value=h2h_sets))


def add_total_plr_features(features, tour, match, tour_avgset):
    trg_best_of_five = tour.best_of_five(match.rnd)
    feat1est, feat2est, feat1sets, feat2sets = total.total_plr_features(
        tour.sex, trg_best_of_five, match, tour, tour_avgset
    )
    features.append(feat1est)
    features.append(feat2est)
    features.append(feat1sets)
    features.append(feat2sets)


def fav_fst_orient(match, features):  # using for total data working
    if match.is_second_favorite(min_chance=0.51, min_diff=1):
        match.flip()
        for feat in features:
            feat.flip_side()


class DetScoreStatProcessor(object):
    def __init__(self, sex):
        self.sex = sex
        analyze_draft_dir = f"{cfg_dir.analysis_data_dir()}/draft/{sex}"
        self.open_csv_writer = CsvWriter(
            filename=f"{analyze_draft_dir}/{sex}_openset_data.csv"
        )
        self.dec_csv_writer = CsvWriter(
            filename=f"{analyze_draft_dir}/{sex}_decidedset_data.csv"
        )

    def best_of_five(self, tour, rnd, match):
        if self.sex == "wta":
            return False
        if not match.score.retired:
            return match.score.best_of_five()
        return tour.best_of_five(rnd)

    @staticmethod
    def add_h2h_len_feature(match, features):
        h2h_count = 0
        if match.head_to_head:
            h2h_count = len(match.head_to_head)
        features.append(RigidFeature("h2h_len", h2h_count))

    @staticmethod
    def add_tie33_features(tie_item, features):
        if tie_item is None:
            return
        points = list(iter(tie_item[1]))
        before_pts, after_pts = dsm.split_by_score(points, num_score=(3, 3))
        feat_is = RigidFeature(name="s1_tie33_is", value=int(len(before_pts) > 0))
        features.append(feat_is)
        if feat_is.value == 0:
            features.append(RigidFeature(name="s1_tie33_dif_before"))
            features.append(RigidFeature(name="s1_tie33_fst_more_lead_before"))
            features.append(RigidFeature(name="s1_tie33_fst_firstly_lead2_after"))
            return
        dif, more_lead_side = dsm.get_max_dif(before_pts)
        features.append(RigidFeature(name="s1_tie33_dif_before", value=dif))
        features.append(
            Feature(
                name="s1_tie33_fst_more_lead_before",
                value=int(more_lead_side.is_left()),
                flip_value=int(more_lead_side.is_right()),
            )
        )
        firstly_lead_side = tie_spec_view.firstly_lead_side(after_pts, dif=2)
        features.append(
            Feature(
                name="s1_tie33_fst_firstly_lead2_after",
                value=int(firstly_lead_side.is_left()),
                flip_value=int(firstly_lead_side.is_right()),
            )
        )

    @staticmethod
    def add_tiebeg6_features(tie_item, features):
        feat_name = "s1_fst_tiebeg6_win"
        if tie_item is not None:
            left_right_cnt = sc.tie_begin6_score(tie_item[1])
            if left_right_cnt is not None:
                left_cnt, right_cnt = left_right_cnt
                if left_cnt != right_cnt:
                    fst_win = left_cnt > right_cnt
                    feat = Feature(
                        name=feat_name, value=int(fst_win), flip_value=int(not fst_win)
                    )
                    features.append(feat)
                    return
        features.append(Feature(name=feat_name, value=None, flip_value=None))

    def add_common_tie_feature(self, match, features):
        max_date = match.date - datetime.timedelta(days=1)
        min_date = max_date - datetime.timedelta(days=365 * 5)
        sall_feat1_5y, sall_feat2_5y = tie_stat.get_features_pair(
            "tie_val_5y",
            self.sex,
            match.first_player.ident,
            match.second_player.ident,
            min_date,
            max_date,
            set_names=None,
            min_size=10 if self.sex == "wta" else 15,
        )
        features.append(sall_feat1_5y)
        features.append(sall_feat2_5y)

    def add_open_tie_feature(self, match, features):
        max_date = match.date - datetime.timedelta(days=1)
        min_date = max_date - datetime.timedelta(days=365 * 5)
        s1_feat1_5y, s1_feat2_5y = tie_stat.get_features_pair(
            "s1_tie_val_5y",
            self.sex,
            match.first_player.ident,
            match.second_player.ident,
            min_date,
            max_date,
            set_names=("open",) if self.sex == "wta" else ("open", "open2"),
            min_size=9 if self.sex == "wta" else 14,
        )
        features.append(s1_feat1_5y)
        features.append(s1_feat2_5y)

    def add_decided_tie_feature(self, match, features):
        max_date = match.date - datetime.timedelta(days=1)
        min_date = max_date - datetime.timedelta(days=365 * 6)
        sd_feat1, sd_feat2 = tie_stat.get_features_pair(
            "sd_tie_val_6y",
            self.sex,
            match.first_player.ident,
            match.second_player.ident,
            min_date,
            max_date,
            set_names=("decided",),
            min_size=7 if self.sex == "wta" else 9,
        )
        features.append(sd_feat1)
        features.append(sd_feat2)

    @staticmethod
    def add_total_features(tour, tour_avgset_dir, match, features):
        tour_avgset = tour_avgset_dir[match.rnd.qualification()]
        if tour_avgset is None:
            return False
        features.append(RigidFeature(name="tour_avgset", value=tour_avgset))
        features.append(RigidFeature(name="sets", value=match.score.sets_count()))
        add_total_h2h_feature(features, tour, match, tour_avgset)
        add_total_plr_features(features, tour, match, tour_avgset)
        return True

    @staticmethod
    def get_tour_avgset_dir(tour, min_size=17):
        result = defaultdict(lambda: None)
        for isqual in (True, False):
            if "Wimbledon" in tour.name and isqual is False:
                continue
            avgset_sval = total.get_tour_avgset(
                tour.sex,
                ischal=tour.level == "chal",
                best_of_five=best_of_five2(tour.sex, tour.level, isqual),
                surface=tour.surface,
                tourname=tour.name,
                isqual=isqual,
                max_year=tour.date.year - 1,
            )
            if avgset_sval.size >= min_size:
                result[isqual] = avgset_sval.value
        return result

    def process(self, year_weeknum, stat_week_history, progress):
        """do week work"""
        ywn_hist_lst = list(tt.year_weeknum_reversed(year_weeknum, max_weeknum_dist=56))
        for tour in weeked_tours.tours(self.sex, year_weeknum):
            if tour.level in ("junior",):
                continue
            if "," in str(tour.name):
                msg = f"{self.sex} coma in tour name '{str(tour.name)}' id={tour.ident}"
                log.warn(msg)
            # tour_avgset_dir = self.get_tour_avgset_dir(tour)
            for match, rnd in tour.match_rnd_list():
                if rnd == "Pre-q" or match.paired() or not match.score:
                    continue
                if (
                    match.score is None
                    or match.date is None
                    or tour.level == ("future", "team")
                    or tour.surface == "Acrylic"
                    or PandemiaTours.is_pandemia_tour(self.sex, tour.ident)
                ):
                    ratings_elo.put_match(self.sex, tour, rnd, match)
                    continue
                if not match.set_decided_tiebreak_info(tour, rnd):
                    ratings_elo.put_match(self.sex, tour, rnd, match)
                    continue
                best_of_five = self.best_of_five(tour, rnd, match)
                soft_level = tennis.soft_level(tour.level, rnd)
                sets_count = match.score.sets_count(full=True)
                if (
                    sets_count < 2
                    or not match.offer
                    or not match.offer.win_coefs
                    or not match.is_ranks_both_above(
                        RANK_STD_BOTH_ABOVE, rtg_name="std"
                    )
                    or match.is_ranks_dif_wide(RANK_STD_MAX_DIF, rtg_name="std")
                    or not find_detailed_score2(
                        dbdet.records,
                        tour,
                        match.rnd,
                        match,
                        predicate=(
                            lambda ds: len(ds) >= 12
                            and not error_contains(ds.error, "SERVER_CHAIN")
                        ),
                    )
                ):
                    ratings_elo.put_match(self.sex, tour, rnd, match)
                    continue
                match.set_items = {
                    1: SetItems.from_scores(1, match.detailed_score, match.score)
                }
                if not match.set_items[1].ok_set(strict=False):
                    ratings_elo.put_match(self.sex, tour, rnd, match)
                    continue
                match.set_items[2] = SetItems.from_scores(
                    2, match.detailed_score, match.score
                )
                match.set_items[3] = SetItems.from_scores(
                    3, match.detailed_score, match.score
                )
                if best_of_five:
                    match.set_items[4] = SetItems.from_scores(
                        4, match.detailed_score, match.score
                    )
                    match.set_items[5] = SetItems.from_scores(
                        5, match.detailed_score, match.score
                    )
                match.read_birth_date(self.sex)
                match.head_to_head = tennis.HeadToHead(
                    self.sex, match, completed_only=True
                )
                com_feats = common_features(self.sex, tour, best_of_five, match)
                get_features.add_stat_features(
                    com_feats,
                    self.sex,
                    ywn_hist_lst[0:stat_week_history],
                    match,
                    tour,
                    rnd,
                    stat_names=("service_win", "receive_win"),
                )
                get_features.add_fatigue_features(
                    com_feats, self.sex, ywn_hist_lst, match, rnd
                )
                get_features.add_tour_adapt_features(com_feats, tour, match)
                self.add_common_tie_feature(match, com_feats)
                set2_after_set1loss_stat.add_features(
                    com_feats,
                    tour.sex,
                    "set2win_after_set1loss",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=match.date - datetime.timedelta(days=365 * 6),
                    max_date=match.date,
                    min_size=20,
                )
                set2_after_set1loss_stat.add_features(
                    com_feats,
                    tour.sex,
                    "set2win_after_set1win",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=match.date - datetime.timedelta(days=365 * 6),
                    max_date=match.date,
                    min_size=20,
                )
                trail_choke_stat.add_features_agr(
                    com_feats,
                    tour.sex,
                    "trail",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=420,
                )
                trail_choke_stat.add_features_agr(
                    com_feats,
                    tour.sex,
                    "choke",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=420,
                )

                op_feats = openspecific_calc_features(match.set_items)
                after_tie_perf_stat.add_features(
                    op_feats,
                    self.sex,
                    match.date - datetime.timedelta(days=365 * 7),
                    match.date,
                    match.first_player.ident,
                    match.second_player.ident,
                )
                self.add_open_tie_feature(match, op_feats)
                trail_choke_stat.add_features(
                    op_feats,
                    tour.sex,
                    "trail",
                    "open",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=220,
                )
                trail_choke_stat.add_features(
                    op_feats,
                    tour.sex,
                    "choke",
                    "open",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=220,
                )
                inset_keep_recovery.add_features(
                    op_feats,
                    tour.sex,
                    ("open",),
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    sizes_shift=-3,
                    feat_suffix="_sh-3",
                )
                if match.set_items[1].set_opener_side() == co.RIGHT:
                    com_feats.flip_side()
                    op_feats.flip_side()
                self.open_csv_writer.put_row_parts(com_feats, op_feats)

                is_dec = sets_count == (5 if best_of_five else 3)
                if (
                    not is_dec
                    or match.score.retired
                    or error_contains(match.detailed_score.error, "RAIN_INTERRUPT")
                    or not match.set_items[sets_count - 1].ok_set(strict=False)
                    or not match.set_items[sets_count].ok_set(strict=False)
                ):
                    ratings_elo.put_match(self.sex, tour, rnd, match)
                    continue

                if com_feats.is_fliped:
                    com_feats.flip_side()  # undo flip (when open writer prepare)
                dec_feats = decspecific_calc_features(
                    best_of_five, tour.sex, match, tour.level, tour.surface
                )
                self.add_decided_tie_feature(match, dec_feats)
                if best_of_five:
                    decided_win_by_two_sets_stat.add_empty_feature(dec_feats)
                else:
                    decided_win_by_two_sets_stat.add_feature(
                        dec_feats,
                        tour.sex,
                        soft_level,
                        tour.date,
                        match.score[0],
                        match.score[1],
                    )
                decided_set.add_feature_dif_ratio(
                    dec_feats,
                    "dset_ratio_dif",
                    self.sex,
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=match.date - datetime.timedelta(days=365 * 3),
                    max_date=match.date,
                )
                decided_set.add_feature_dif_bonus(
                    dec_feats,
                    "dset_bonus_dif",
                    self.sex,
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=match.date - datetime.timedelta(days=365 * 3),
                    max_date=match.date,
                )
                trail_choke_stat.add_features(
                    dec_feats,
                    tour.sex,
                    "trail",
                    "decided",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=220,
                )
                trail_choke_stat.add_features(
                    dec_feats,
                    tour.sex,
                    "choke",
                    "decided",
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    max_size=220,
                )
                inset_keep_recovery.add_features(
                    dec_feats,
                    tour.sex,
                    ("decided",),
                    match.first_player.ident,
                    match.second_player.ident,
                    min_date=None,
                    max_date=match.date,
                    sizes_shift=-3,
                    feat_suffix="_sh-3",
                )

                if (
                    best_of_five and match.set_items[5].set_opener_side() == co.RIGHT
                ) or (
                    not best_of_five
                    and match.set_items[3].set_opener_side() == co.RIGHT
                ):
                    com_feats.flip_side()
                    dec_feats.flip_side()
                self.dec_csv_writer.put_row_parts(com_feats, dec_feats)
                ratings_elo.put_match(self.sex, tour, rnd, match)
            if progress is not None:
                progress.put_tour(tour)

    def reporting(self):
        log.info("{} detscore with feat output start...".format(self.sex))
        self.open_csv_writer.write()
        self.dec_csv_writer.write()
        log.info("{} detscore with feat output finish".format(self.sex))


def learndata_preparing_sex(sex):
    def from_date():
        # next line is variant only for detscores:
        return tt.past_monday_date(dbdet.get_min_date())
        # return datetime.date(2010, 4, 26)  # for debug
        # return datetime.date(2000, 1, 1)  # 2004 because we have avgtoursets from 1997

    def to_date():
        return tt.past_monday_date(dbdet.get_max_date()) + datetime.timedelta(days=7)
        # return (PandemiaTours.max_prev_pandemia_rtg_date() +
        #         datetime.timedelta(days=7))

    stat_week_history = 30  # weeks number BEFORE current tour week
    dbdet.query_matches()
    min_date = from_date()
    max_date = to_date()
    log.info("min_date: {}".format(min_date))
    log.info("max_date: {}".format(max_date))
    markov.initialize_gen()
    decided_win_by_two_sets_stat.initialize(sex=sex)
    ratings_std.initialize(
        sex=sex,
        min_date=min_date
        - datetime.timedelta(days=365 * 5),  # set2_after_set1loss_stat
        # min_date=min_date - datetime.timedelta(days=7 * stat_week_history),
    )
    ratings_elo.initialize(sex, date=datetime.date(2009, 12, 27))
    bet_coefs.initialize(
        sex=sex,
        min_date=min_date - datetime.timedelta(days=7 * stat_week_history),
        max_date=max_date + datetime.timedelta(days=7),
    )
    matchstat.initialize(
        sex=sex,
        min_date=min_date - datetime.timedelta(days=7 * stat_week_history),
        max_date=max_date + datetime.timedelta(days=7),
    )
    print_memory_usage("matchstat")
    decided_set.initialize_results(
        sex=sex,
        min_date=min_date - datetime.timedelta(days=365 * 3),
        max_date=max_date + datetime.timedelta(days=7),
    )
    print_memory_usage("decided_set")
    tie_stat.initialize_results(
        sex=sex,
        min_date=min_date - datetime.timedelta(days=365 * 6),
        max_date=max_date + datetime.timedelta(days=7),
    )
    print_memory_usage("tie_stat")
    span_days = (max_date - min_date).days + stat_week_history * 7
    oncourt_players.initialize(sex=sex, yearsnum=span_days / 365.25)  # for pers details
    weeked_tours.initialize_sex(
        sex,
        min_date=min_date - datetime.timedelta(days=7 * stat_week_history),
        max_date=max_date,
        with_ratings=True,
        with_bets=True,
        with_stat=True,
        with_pers_det=True,
    )
    print_memory_usage("weeked_tours")
    set2_after_set1loss_stat.initialize_results(
        sex,
        min_date=min_date - datetime.timedelta(days=365 * 5),
        max_date=max_date + datetime.timedelta(days=7),
    )
    print_memory_usage("set2_after_set1loss_stat")

    inset_keep_recovery.initialize_results(sex, dbdet, setnames=("open", "decided"))
    print_memory_usage("inset_keep_recovery")
    trail_choke_stat.initialize_results(sex, dbdet)
    print_memory_usage("trail_choke_stat")
    after_tie_perf_stat.initialize(
        sex,
        dbdet,
        min_date=min_date - datetime.timedelta(days=365 * 7),
        max_date=max_date + datetime.timedelta(days=7),
    )
    log.info("data initialized, prepared")
    proc = DetScoreStatProcessor(sex)
    progress = tt.YearWeekProgress("dscore dan clf " + sex)
    for num, year_weeknum in enumerate(weeked_tours.year_weeknum_iter(sex), start=1):
        if num > stat_week_history:
            proc.process(year_weeknum, stat_week_history, progress=progress)
    proc.reporting()


def do_main():
    sex = args.sex
    try:
        log.info("started {}".format(sex))
        start_datetime = datetime.datetime.now()
        dba.open_connect()
        learndata_preparing_sex(sex)
        dba.close_connect()
        end_datetime = datetime.datetime.now()
        log.info(
            "{} {} finished with {}".format(
                __file__, sex, str(end_datetime - start_datetime)
            )
        )
        return 0
    except Exception as err:
        log.error("{} {} [{}]".format(sex, err, err.__class__.__name__), exception=True)
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    PROCESS = psutil.Process(os.getpid())
    MEGA = 10 ** 6
    args = parse_command_line_args()
    dbdet = open_db(args.sex)
    sys.exit(do_main())
