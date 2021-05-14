# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict
import datetime
from operator import itemgetter
from operator import sub
import argparse
import unittest

import file_utils as fu
import common as co
import log
import tennis
import tournament as trmt
import tennis_time as tt
import dba
import stat_cont as st
import cfg_dir
import ratings_std

# import rematches as rm
# import total
import qual_seeds
import dict_tools
import oncourt_players

# import handicap
import matchstat


class WinFavoriteProcessor(object):
    def __init__(self, sex, chance_thresholds=(0.55, 0.60, 0.65, 0.70)):
        #  (chance_threshold, pid) -> WinLoss
        self.data_dict = defaultdict(st.WinLoss)
        self.sex = sex
        self.actual_players = oncourt_players.players(sex)
        self.chance_thresholds = chance_thresholds
        self.min_size = 15 if sex == "wta" else 20

    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q":
                continue
            for m in matches:
                if not m.paired() and m.score.valid() and m.offer and m.offer.win_coefs:
                    self.__process_match(m)

    def __process_match(self, match):
        chances = match.offer.win_coefs.chances()
        if chances[0] > chances[1]:
            fav_chance = chances[0]
            fav_pid = match.first_player.ident
            fav_win = True
        else:
            fav_chance = chances[1]
            fav_pid = match.second_player.ident
            fav_win = False

        for chance_threshold in self.chance_thresholds:
            if fav_chance >= chance_threshold:
                self.data_dict[(chance_threshold, fav_pid)].hit(fav_win)

    def reporting(self):
        log.info("{} win favorite output start...".format(self.sex))
        dirname = cfg_dir.stat_players_dir(self.sex) + "/win_favorite"
        fu.ensure_folder(dirname)
        for threshold in self.chance_thresholds:
            self.report_threshold(dirname, threshold)
        log.info("{} win favorite output finish".format(self.sex))

    def report_threshold(self, dirname, threshold):
        pid_wl_list = [
            (pid, wl)
            for (thr, pid), wl in self.data_dict.items()
            if thr == threshold and wl.size >= self.min_size
        ]
        pid_wl_list.sort(key=lambda i: i[1].ratio, reverse=False)
        shortname = "fav_{:.2f}".format(threshold).replace(".", "")
        filename = os.path.join(dirname, "{}.txt".format(shortname))
        with open(filename, "w") as fh:
            for pid, wl in pid_wl_list:
                plr = co.find_first(
                    self.actual_players, predicate=lambda p: p.ident == pid
                )
                if plr:
                    fh.write("{}__{}\n".format(plr, wl))


class Cmp75Vs64Processor(object):
    def __init__(self, sex):
        self.sex = sex
        self.name = "cmp-75-vs-64"
        self.dirname = os.path.join(cfg_dir.stat_misc_dir(sex), self.name)
        self.s2_wl4 = st.create_winloss_histogram()
        self.s2_wl5 = st.create_winloss_histogram()
        self.s3_wl4 = st.create_winloss_histogram()
        self.s3_wl5 = st.create_winloss_histogram()
        self.s3_mutual_wl4 = st.create_winloss_histogram()
        self.s3_mutual_wl5 = st.create_winloss_histogram()
        self.s2_s3_wl4 = st.create_winloss_histogram()
        self.s2_s3_wl5 = st.create_winloss_histogram()

    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd.pre_qualification():
                continue
            for m in matches:
                sets_count = m.score.sets_count(full=True)
                if (
                    m.paired()
                    or not m.score.valid()
                    or m.score.retired
                    or m.score.best_of_five()
                    or sets_count < 2
                ):
                    continue
                self.__process_match(tour, m.score, sets_count)

    def __process_match(self, tour, score, sets_count):
        keys = tennis.match_keys_combinations(tour.level, tour.surface, rnd=None)
        if sets_count == 2:
            result4, result5 = self.__two_sets(keys, score)
            if result4 is not None:
                self.s2_s3_wl4.hit(keys, result4)
            if result5 is not None:
                self.s2_s3_wl5.hit(keys, result5)
        elif sets_count == 3:
            result4, result5 = self.__three_sets(keys, score)
            if result4 is not None:
                self.s2_s3_wl4.hit(keys, result4)
            if result5 is not None:
                self.s2_s3_wl5.hit(keys, result5)

    def __two_sets(self, keys, score):
        result4, result5 = None, None
        fst_set, snd_set = score[0], score[1]
        assert (
            fst_set[0] > fst_set[1] and snd_set[0] > snd_set[1]
        ), "bad straight score {} keys: {}".format(score, keys)
        if self.exist_score(fst_set, [(6, 4), (7, 5)]):
            if fst_set == (6, 4):
                result4 = True
                self.s2_wl4.hit(keys, result4)
            elif fst_set == (7, 5):
                result5 = True
                self.s2_wl5.hit(keys, result5)
        return result4, result5

    def __three_sets(self, keys, score):
        result4, result5 = None, None
        fst_set, snd_set, thd_set = score[0], score[1], score[2]
        assert thd_set[0] > thd_set[1], "bad 3set score {} keys: {}".format(score, keys)
        mutual = self.__three_sets_mutual(keys, fst_set, snd_set)
        if not mutual:
            result4, result5 = self.__get_results(fst_set, snd_set)
            if result4 is not None:
                self.s3_wl4.hit(keys, result4)
            if result5 is not None:
                self.s3_wl5.hit(keys, result5)
        return result4, result5

    def __get_results(self, fst_set, snd_set):
        result4, result5 = None, None
        if fst_set == (6, 4) or snd_set == (6, 4):
            result4 = True
        if fst_set == (4, 6) or snd_set == (4, 6):
            result4 = False
        if fst_set == (7, 5) or snd_set == (7, 5):
            result5 = True
        if fst_set == (5, 7) or snd_set == (5, 7):
            result5 = False
        return result4, result5

    def __three_sets_mutual(self, keys, fst_set, snd_set):
        result = False
        if (
            (fst_set == (6, 4) and snd_set == (5, 7))
            or (fst_set == (4, 6) and snd_set == (7, 5))
            or (fst_set == (7, 5) and snd_set == (4, 6))
            or (fst_set == (5, 7) and snd_set == (6, 4))
        ):
            result = True
            if (fst_set == (6, 4) and snd_set == (5, 7)) or (
                fst_set == (5, 7) and snd_set == (6, 4)
            ):
                self.s3_mutual_wl4.hit(keys, True)
                self.s3_mutual_wl5.hit(keys, False)
            if (fst_set == (4, 6) and snd_set == (7, 5)) or (
                fst_set == (7, 5) and snd_set == (4, 6)
            ):
                self.s3_mutual_wl4.hit(keys, False)
                self.s3_mutual_wl5.hit(keys, True)
        return result

    @staticmethod
    def exist_score(the_set, scores):
        for s in scores:
            if the_set == s:
                return True
        return False

    def reporting(self):
        def write_histogram(filename, histogram):
            dict_tools.dump(
                histogram,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=lambda kv: len(kv[0]),
                alignvalues=True,
            )

        log.info("{} {} output start...".format(self.sex, self.name))
        assert os.path.isdir(self.dirname), "not found dir: " + self.dirname
        write_histogram(os.path.join(self.dirname, "s2_wl4.txt"), self.s2_wl4)
        write_histogram(os.path.join(self.dirname, "s2_wl5.txt"), self.s2_wl5)
        write_histogram(os.path.join(self.dirname, "s3_wl4.txt"), self.s3_wl4)
        write_histogram(os.path.join(self.dirname, "s3_wl5.txt"), self.s3_wl5)
        write_histogram(
            os.path.join(self.dirname, "s3_mutual_wl4.txt"), self.s3_mutual_wl4
        )
        write_histogram(
            os.path.join(self.dirname, "s3_mutual_wl5.txt"), self.s3_mutual_wl5
        )
        write_histogram(os.path.join(self.dirname, "s2_s3_wl4.txt"), self.s2_s3_wl4)
        write_histogram(os.path.join(self.dirname, "s2_s3_wl5.txt"), self.s2_s3_wl5)
        log.info("{} {} output finish".format(self.sex, self.name))


class WinAfterWin2ndSetProcessor(object):
    """
    case: when win match after lose 1st set and win 2nd set.
    """

    def __init__(self, sex):
        self.sex = sex
        self.dirname = os.path.join(
            cfg_dir.stat_misc_dir(sex), "winmatch-after-winset2"
        )
        self.histo_from_file = defaultdict(st.create_winloss_histogram)
        self.clr_histo_from_file = defaultdict(st.create_winloss_histogram)
        self.dclr_histo_from_file = defaultdict(st.create_winloss_histogram)
        self.g_histo = st.create_winloss_histogram()
        self.losstie_then_winusial_histo = st.create_winloss_histogram()

    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if (
                rnd.qualification() and tour.level != "masters"
            ) or rnd.pre_qualification():
                continue
            for m in matches:
                if (
                    m.paired()
                    or not m.score.valid()
                    or m.score.retired
                    or m.score.best_of_five()
                    or m.score.sets_count(full=True) != 3
                ):
                    continue
                self.__process_match(tour, m.score)

    def __process_match(self, tour, score):
        first_set, second_set, third_set = score[0], score[1], score[2]
        assert third_set[0] > third_set[1], "left must win, but: {}".format(score)
        winner_win_2nd_set = second_set[0] > second_set[1]
        level = tour.level
        if level in ("gs", "masters"):
            level = tennis.Level("main")
        # ----- simple scores ---
        filename = "{}/{}.txt".format(self.dirname, level)
        if winner_win_2nd_set:
            sc_key = "{}-{} {}-{}".format(
                first_set[0], first_set[1], second_set[0], second_set[1]
            )
        else:
            sc_key = "{}-{} {}-{}".format(
                first_set[1], first_set[0], second_set[1], second_set[0]
            )
        self.histo_from_file[filename].hit([sc_key], winner_win_2nd_set)

        if (
            winner_win_2nd_set
            and first_set == (6, 7)
            and second_set not in ((7, 6), (6, 0), (6, 1))
        ) or (
            not winner_win_2nd_set
            and first_set == (7, 6)
            and second_set not in ((6, 7), (0, 6), (1, 6))
        ):
            self.losstie_then_winusial_histo.hit(
                [str(tour.surface), str(level), str(tour.surface) + "-" + str(level)],
                winner_win_2nd_set,
            )

        # ------ colors -------
        clr_sc_key = "{}-{}".format(self.set_desc(first_set), self.set_desc(second_set))
        clr_filename = "{}/color-{}.txt".format(self.dirname, level)
        self.clr_histo_from_file[clr_filename].hit([clr_sc_key], winner_win_2nd_set)

        # ---detailed colors ---
        dclr_sc_key = "{}-{}".format(
            self.set_desc(first_set, True), self.set_desc(second_set, True)
        )
        dclr_filename = "{}/detcolor-{}.txt".format(self.dirname, level)
        self.dclr_histo_from_file[dclr_filename].hit([dclr_sc_key], winner_win_2nd_set)

        self.g_histo.hit([str(level)], winner_win_2nd_set)

    def reporting(self):
        log.info("{} after win 2nd set output start...".format(self.sex))
        for filename, histo in self.histo_from_file.items():
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(1),
                sortreverse=True,
                alignvalues=True,
            )

        for filename, histo in self.clr_histo_from_file.items():
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(1),
                sortreverse=True,
                alignvalues=True,
            )

        for filename, histo in self.dclr_histo_from_file.items():
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(1),
                sortreverse=True,
                alignvalues=True,
            )

        dict_tools.dump(
            self.g_histo,
            "{}/g.txt".format(self.dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
        )
        dict_tools.dump(
            self.losstie_then_winusial_histo,
            "{}/losstie_then_winusial.txt".format(self.dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
        )
        log.info("{} after win 2nd set output finish".format(self.sex))

    @staticmethod
    def set_desc(set_score, detailed=False):
        games_sum = set_score[0] + set_score[1]
        if 6 <= games_sum <= 7:
            result = "Low"
        elif 8 <= games_sum <= 9:
            result = "Med"
        else:
            result = "High"
        if detailed and games_sum == 10:
            result = "Ten"
        return result


class WinShortSet2AfterWinSet1Processor(object):
    """
    case: победитель первого сета выигрывает второй сет уверенно (<=61, <=62, <=63)
    """

    def __init__(self, sex):
        self.winloss_histo_le61 = st.create_winloss_histogram()
        self.winloss_histo_le62 = st.create_winloss_histogram()
        self.winloss_histo_le63 = st.create_winloss_histogram()
        self.winloss_histo_md_le61 = st.create_winloss_histogram()
        self.winloss_histo_md_le62 = st.create_winloss_histogram()
        self.winloss_histo_md_le63 = st.create_winloss_histogram()
        self.sex = sex

    def process(self, tour):
        if tour.grand_slam():
            return  # grand_slams should be excluded (special motivation) !?
        if tour.surface == "Acrylic" or tour.level.junior:
            return
        for rnd in tour.matches_from_rnd.keys():
            if rnd.pre_qualification():
                continue
            keys = tennis.match_keys_combinations(
                level=tour.level, surface=tour.surface, rnd=rnd
            )
            for m in tour.matches_from_rnd[rnd]:
                if (
                    not m.score.valid()
                    or m.score.retired
                    or m.score.best_of_five()
                    or m.score.sets_count() < 2
                ):
                    continue
                self.__process_match(m, rnd, keys)

    def __process_match(self, match, rnd, keys):
        first_set = match.score[0]
        second_set = match.score[1]
        second_set_short_61 = (second_set[0] + second_set[1]) <= 7
        second_set_short_62 = (second_set[0] + second_set[1]) <= 8
        second_set_short_63 = (second_set[0] + second_set[1]) <= 9
        after_win_s1_win_s2 = (
            first_set[0] > first_set[1] and second_set[0] > second_set[1]
        ) or (first_set[0] < first_set[1] and second_set[0] < second_set[1])
        self.winloss_histo_le61.hit(keys, after_win_s1_win_s2 and second_set_short_61)
        self.winloss_histo_le62.hit(keys, after_win_s1_win_s2 and second_set_short_62)
        self.winloss_histo_le63.hit(keys, after_win_s1_win_s2 and second_set_short_63)
        if rnd.main_draw():
            self.winloss_histo_md_le61.hit(
                keys, after_win_s1_win_s2 and second_set_short_61
            )
            self.winloss_histo_md_le62.hit(
                keys, after_win_s1_win_s2 and second_set_short_62
            )
            self.winloss_histo_md_le63.hit(
                keys, after_win_s1_win_s2 and second_set_short_63
            )

    def reporting(self):
        name = "win-short-set2-after-win-set1"
        log.info("{} {} output start...".format(self.sex, name))
        dirname = os.path.join(cfg_dir.stat_misc_dir(self.sex), name)
        dict_tools.dump(
            self.winloss_histo_le61,
            "{}/le61-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )
        dict_tools.dump(
            self.winloss_histo_le62,
            "{}/le62-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )
        dict_tools.dump(
            self.winloss_histo_le63,
            "{}/le62-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )

        dict_tools.dump(
            self.winloss_histo_md_le61,
            "{}/md_le61-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )
        dict_tools.dump(
            self.winloss_histo_md_le62,
            "{}/md_le62-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )
        dict_tools.dump(
            self.winloss_histo_md_le63,
            "{}/md_le63-winloss.txt".format(dirname),
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
            filterfun=dict_tools.filter_value_size_fun(450),
        )
        log.info("{} {} output finish".format(self.sex, name))


class CountryMatesProcessor(object):
    """
    case: разбиваем на факторы:
          home_both - соплеменники играют дома
          away_both - соплеменники играют в другой стране
          home_one - один домашний игрок играет дома с иностранцем
          neutral - в некой стране играют играют не дома не соплеменники
    """

    def __init__(self, sex, fun_names):
        self.sex = sex
        self.fun_names = fun_names
        self.home_both_from_country = defaultdict(st.TotalSlices)
        self.home_one_from_country = defaultdict(st.TotalSlices)
        self.away_both_from_country = defaultdict(st.TotalSlices)
        self.home_both = st.TotalSlices()
        self.away_both = st.TotalSlices()
        self.home_one = st.TotalSlices()
        self.neutral = st.TotalSlices()
        self.home_one_wlpair_histo = st.create_winloss_pair_value_histogram()

    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd in (
                "Pre-q",
                "Rubber 1",
                "Rubber 2",
                "Rubber 3",
                "Rubber 4",
                "Rubber 5",
                "Bronze",
            ):
                continue
            for m in matches:
                self.__process_match(
                    tour, m.rnd, m.score, m.first_player, m.second_player
                )

    def __process_match(self, tour, rnd, score, first_player, second_player):
        level = tour.level
        if level == "gs":
            return  # grand_slams should be excluded (special motivation) !?
        if level == "masters":
            level = tennis.Level("main")
        if tour.surface in ("Grass", "Acrylic") or (
            not level.main and not level.chal and not level.future
        ):
            return
        if (
            not score.valid()
            or score.retired
            or score.best_of_five()
            or rnd.pre_qualification()
        ):
            return
        if tour.cou is None or first_player.cou is None or second_player.cou is None:
            return
        keys = tennis.match_keys_combinations(level, tour.surface, rnd)

        if (
            first_player.cou != second_player.cou
            and first_player.cou != tour.cou
            and second_player.cou != tour.cou
        ):
            self.neutral.addition(score, keys)
        elif first_player.cou != second_player.cou and (
            first_player.cou == tour.cou or second_player.cou == tour.cou
        ):
            self.home_one.addition(score, keys)
            cou = (
                first_player.cou if first_player.cou == tour.cou else second_player.cou
            )
            self.home_one_from_country[cou].addition(score, keys)
            homer_is_win = cou == first_player.cou  # first_player is winner
            self.home_one_wlpair_histo.hit(
                keys, (score.normalized_total(), homer_is_win)
            )
        elif first_player.cou == second_player.cou and first_player.cou == tour.cou:
            self.home_both.addition(score, keys)
            self.home_both_from_country[first_player.cou].addition(score, keys)
        elif first_player.cou == second_player.cou and first_player.cou != tour.cou:
            self.away_both.addition(score, keys)
            self.away_both_from_country[first_player.cou].addition(score, keys)

    def reporting(self):
        log.info("{} country mates output start...".format(self.sex))
        for fun_name in self.fun_names:
            dirname = os.path.join(
                cfg_dir.stat_misc_dir(self.sex), "country-mates", "totals", fun_name
            )
            fu.ensure_folder(dirname)
            self.home_both.write_report(
                dirname + "/home-both.txt",
                fun_name,
                sort_asc=True,
                sort_keyfun=itemgetter(0),
                threshold=0,
            )
            self.away_both.write_report(
                dirname + "/away-both.txt",
                fun_name,
                sort_asc=True,
                sort_keyfun=itemgetter(0),
                threshold=0,
            )
            self.home_one.write_report(
                dirname + "/home-one.txt",
                fun_name,
                sort_asc=True,
                sort_keyfun=itemgetter(0),
                threshold=0,
            )
            self.neutral.write_report(
                dirname + "/neutral.txt",
                fun_name,
                sort_asc=True,
                sort_keyfun=itemgetter(0),
                threshold=0,
            )
            st.slices_table_report(
                fun_name,
                slices=(self.home_both, self.home_one, self.neutral, self.away_both),
                titles=("HOME_BOTH", "HOME_ONE", "NEUTRAL", "AWAY_BOTH"),
                idx1=0,
                idx2=2,
                min_diff=0.9,
                min_size=200,
                filter_keys=None,
                filename=os.path.join(
                    dirname, "table_report_{}_home_both_dif09.txt".format(self.sex)
                ),
            )
            st.slices_table_report(
                fun_name,
                slices=(self.home_both, self.home_one, self.neutral, self.away_both),
                titles=("HOME_BOTH", "HOME_ONE", "NEUTRAL", "AWAY_BOTH"),
                idx1=1,
                idx2=2,
                min_diff=0.9,
                min_size=200,
                filter_keys=None,
                filename=os.path.join(
                    dirname, "table_report_{}_home_one_dif09.txt".format(self.sex)
                ),
            )
            st.slices_table_report(
                fun_name,
                slices=(self.home_both, self.home_one, self.neutral, self.away_both),
                titles=("HOME_BOTH", "HOME_ONE", "NEUTRAL", "AWAY_BOTH"),
                idx1=2,
                idx2=3,
                min_diff=0.5,
                min_size=100,
                filter_keys=None,
                filename=os.path.join(
                    dirname, "table_report_{}_away_both_dif05.txt".format(self.sex)
                ),
            )
            self.__all_countries_report(
                fun_name,
                self.home_both_from_country,
                os.path.join(dirname, "countries-home-both.txt"),
                threshold=80,
            )
            self.__all_countries_report(
                fun_name,
                self.home_one_from_country,
                os.path.join(dirname, "countries-home-one.txt"),
                threshold=80,
            )
            self.__all_countries_report(
                fun_name,
                self.away_both_from_country,
                os.path.join(dirname, "countries-away-both.txt"),
                threshold=65,
            )

            self.__countries_dirs_reports(
                fun_name, dirname, "home-both.txt", self.home_both_from_country
            )
            self.__countries_dirs_reports(
                fun_name, dirname, "home-one.txt", self.home_one_from_country
            )
            self.__countries_dirs_reports(
                fun_name, dirname, "away-both.txt", self.away_both_from_country
            )
            dict_tools.dump(
                self.home_one_wlpair_histo,
                dirname + "/" + "home-one-winloss.txt",
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                alignvalues=True,
                filterfun=dict_tools.filter_value_size_fun(40),
            )
        log.info("{} country mates output finish".format(self.sex))

    def __countries_dirs_reports(
        self, fun_name, dirname, filename, slices_from_country
    ):
        for cou in slices_from_country.keys():
            cou_dir = os.path.join(dirname, cou)
            fu.ensure_folder(cou_dir)
            slices = slices_from_country[cou]
            slices.write_report(
                cou_dir + "/" + filename,
                fun_name,
                sort_asc=True,
                sort_keyfun=itemgetter(0),
                threshold=0,
            )

    def __all_countries_report(self, fun_name, slice_from_country, filename, threshold):
        cou_val_size = []
        for cou, totslice in slice_from_country.items():
            cou_val_size.append(
                (
                    cou,
                    totslice.value(co.StructKey(), fun_name),
                    totslice.get_collector(co.StructKey()).hits_count(),
                )
            )
        with open(filename, "w") as fh:
            for cou, val, size in sorted(cou_val_size, key=itemgetter(1)):
                if size >= threshold:
                    fh.write("{} {} ({})\n".format(cou, val, size))


class Set12TotalProcessor(object):
    """
    тотал 1-го, 2-го сета
    """

    SKIP_MIN_CHANCE = 0.73
    SKIP_MIN_DIFF = 330

    def __init__(self, sex):
        self.sex = sex
        self.firstset_under_over_95 = st.create_winloss_histogram()
        self.secondset_under_over_95 = st.create_winloss_histogram()

        self.firstsetcare_under_over_95 = st.create_winloss_histogram()
        self.secondsetcare_under_over_95 = st.create_winloss_histogram()

    def process(self, tour):
        if tour.level in ("team", "teamworld", "junior", "future"):
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd in (
                "Pre-q",
                "Rubber 1",
                "Rubber 2",
                "Rubber 3",
                "Rubber 4",
                "Rubber 5",
                "Bronze",
            ):
                continue
            if rnd.qualification():
                continue
            keys = tennis.match_keys_combinations(
                level=tour.level, surface=tour.surface, rnd=rnd
            )
            for m in matches:
                self.__process_match(m, keys)

    def __process_match(self, match, keys):
        if not match.score.valid() or match.score.best_of_five():
            return
        sets_count = match.score.sets_count(full=True)
        exist_favorite = (
            match.favorite_player(self.SKIP_MIN_CHANCE, self.SKIP_MIN_DIFF) is not None
        )
        if sets_count >= 1:
            total_set1 = match.score[0][0] + match.score[0][1]
            self.firstset_under_over_95.hit(keys, total_set1 < 9.5)
            if not exist_favorite:
                self.firstsetcare_under_over_95.hit(keys, total_set1 < 9.5)
        if sets_count >= 2:
            total_set2 = match.score[1][0] + match.score[1][1]
            self.secondset_under_over_95.hit(keys, total_set2 < 9.5)
            if not exist_favorite:
                self.secondsetcare_under_over_95.hit(keys, total_set2 < 9.5)

    def reporting(self):
        log.info("{} set12 output start...".format(self.sex))
        dirname = cfg_dir.stat_misc_dir(self.sex) + "/set1"
        assert os.path.isdir(dirname), "dir not found: " + dirname
        filename = os.path.join(dirname, "set1.txt")
        dict_tools.dump(
            self.firstset_under_over_95,
            filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        dict_tools.dump(
            self.firstsetcare_under_over_95,
            dirname + "/set1care.txt",
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )

        dirname = cfg_dir.stat_misc_dir(self.sex) + "/set2"
        assert os.path.isdir(dirname), "dir not found: " + dirname
        filename = os.path.join(dirname, "set2.txt")
        dict_tools.dump(
            self.secondset_under_over_95,
            filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        dict_tools.dump(
            self.secondsetcare_under_over_95,
            dirname + "/set2care.txt",
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        log.info("{} set12 output finish".format(self.sex))


class TourAverageSetProcessor(object):
    """
    Ищем средний сет турнира.
    assert: time reverse going (order of process(tour))
    """

    def __init__(
        self,
        sex,
        max_tours_count=4,
        max_matches_count=254,
        max_years_count=3,
        max_diff_years=4,
        min_year=2000,
    ):
        self.sex = sex
        # condition for inner collected stuf:
        self.max_tours_count = max_tours_count
        self.max_matches_count = max_matches_count
        self.max_years_count = max_years_count
        self.max_diff_years = max_diff_years
        self.min_year = min_year
        self.as_years_nt_nm_from_surf_lev_cou_tname = defaultdict(
            lambda: [st.Sumator(), set(), 0, 0]
        )

    def process(self, tour):
        if tour.level in ("team", "teamworld", "junior", "future"):
            return
        if tour.surface not in ("Hard", "Clay", "Carpet", "Grass"):
            return
        tour_name = tour.name
        if not tour_name:
            log.error("empty tour name str for tour: {}".format(tour))
            return
        avgs_years_nt_nm = self.as_years_nt_nm_from_surf_lev_cou_tname[
            (tour.surface, tour.level, tour.cou, tour_name)
        ]
        if avgs_years_nt_nm[2] >= self.max_tours_count:
            return
        if (
            len(avgs_years_nt_nm[1]) >= self.max_years_count
            and tour.date.year not in avgs_years_nt_nm[1]
        ):
            return
        if tour.date.year < self.min_year or (
            len(avgs_years_nt_nm[1]) > 0
            and abs(tour.date.year - min(avgs_years_nt_nm[1])) >= self.max_diff_years
        ):
            return
        init_matches_count = avgs_years_nt_nm[0].count
        for rnd, matches in tour.matches_from_rnd.items():
            if (rnd.qualification() and tour.level != "masters") or rnd in (
                "Final",
                "Pre-q",
                "Rubber 1",
                "Rubber 2",
                "Rubber 3",
                "Rubber 4",
                "Rubber 5",
                "Bronze",
            ):
                continue
            for m in matches:
                if m.paired() or not m.score.valid() or m.score.retired:
                    continue
                if avgs_years_nt_nm[3] >= self.max_matches_count:
                    break
                for idx in range(len(m.score)):
                    wl = m.score[idx]
                    set_total = min(13, wl[0] + wl[1])
                    if set_total >= 6:
                        avgs_years_nt_nm[0] += set_total
                avgs_years_nt_nm[3] += 1
        if init_matches_count < avgs_years_nt_nm[0].count:
            avgs_years_nt_nm[1].add(tour.date.year)
            avgs_years_nt_nm[2] += 1

    def reporting(self):
        log.info("{} tour avg set output start...".format(self.sex))
        dirname = cfg_dir.stat_tours_sets_dir(self.sex, "mean_value")
        for level in (
            tennis.Level("masters"),
            tennis.Level("gs"),
            tennis.Level("main"),
            tennis.Level("chal"),
        ):
            for surface in (
                tennis.Surface("Hard"),
                tennis.Surface("Clay"),
                tennis.Surface("Carpet"),
                tennis.Surface("Grass"),
            ):
                res_list = []
                for (
                    surf,
                    lev,
                    cou,
                    tour_name,
                ) in self.as_years_nt_nm_from_surf_lev_cou_tname.keys():
                    if (surf != surface) or (lev != level):
                        continue
                    avgs_years_nt_nm = self.as_years_nt_nm_from_surf_lev_cou_tname[
                        (surf, lev, cou, tour_name)
                    ]
                    if avgs_years_nt_nm[0].count > 0:
                        res_list.append((cou, tour_name, avgs_years_nt_nm[0]))
                if res_list:
                    filename = "{}/level={},surface={}#cou,tour.txt".format(
                        dirname, level, surface
                    )
                    with open(filename, "w") as fhandle:
                        for cou, tname, avgs in sorted(
                            res_list, key=lambda rec: rec[2].average()
                        ):
                            fhandle.write(
                                "cou={},tour={}___{}\n".format(cou, tname, avgs)
                            )
        log.info("{} tour avg set output finish".format(self.sex))


class DrawStatusProcessor(object):
    """
    Cases: - частота выигрыша квалифаера в первом круге,
           - частота выигрыша byer во втором круге
    """

    def __init__(self, sex):
        self.sex = sex
        self.byeR2_sets_histo_from_lspairs = defaultdict(st.create_histogram)
        self.qualR1_sets_histo_from_lspairs = defaultdict(st.create_histogram)
        self.qualLL_winloss_histo = st.create_winloss_histogram()

        self.byeR2_wlpair_histo = st.create_winloss_pair_value_histogram()
        self.qualR1_wlpair_histo = st.create_winloss_pair_value_histogram()

    def process(self, tour):
        if tour.level.junior:
            return
        qual_rounds = [r for r in tour.matches_from_rnd.keys() if r.qualification()]
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd in (
                "Pre-q",
                "Rubber 1",
                "Rubber 2",
                "Rubber 3",
                "Rubber 4",
                "Rubber 5",
                "Bronze",
            ):
                continue
            for m in matches:
                if not m.score.valid() or m.score.retired or m.rnd.pre_qualification():
                    continue
                self.__process_match(tour, m, qual_rounds)

    def __lucky_looser_process(self, tour, match, qual_rounds):
        def is_lucky_looser(player, tour, qual_rounds):
            for qrnd in qual_rounds:
                for qm in tour.matches_from_rnd[qrnd]:
                    if player == qm.second_player:
                        return True
            return False

        if not qual_rounds or match.rnd not in (
            tennis.Round("First", "vs qual"),
            tennis.Round("First", "qual vs qual"),
        ):
            return
        is_lucky_looser_first = match.first_draw_status == "qual" and is_lucky_looser(
            match.first_player, tour, qual_rounds
        )
        is_lucky_looser_second = match.second_draw_status == "qual" and is_lucky_looser(
            match.second_player, tour, qual_rounds
        )
        if (not is_lucky_looser_first and not is_lucky_looser_second) or (
            is_lucky_looser_first and is_lucky_looser_second
        ):
            return
        keys = tennis.match_keys_combinations(
            level=tour.level, surface=tour.surface, rnd=None
        )
        self.qualLL_winloss_histo.hit(keys, is_lucky_looser_first)

    def __sets_process(self, tour, match):
        if match.score.best_of_five():
            return
        scoresets = match.score.sets_score()
        if match.rnd == tennis.Round("First", "vs qual"):
            assert "qual" in (
                match.first_draw_status,
                match.second_draw_status,
            ), "unexpected rnd: {} ds1: {} ds2: {}".format(
                match.rnd, match.first_draw_status, match.second_draw_status
            )
            if match.first_draw_status != "qual" and match.second_draw_status == "qual":
                scoresets = scoresets[1], scoresets[0]
            self.qualR1_sets_histo_from_lspairs[(tour.level, tour.surface)].hit(
                [scoresets]
            )
        elif match.rnd == tennis.Round("Second", "vs bye"):
            assert "bye" in (
                match.first_draw_status,
                match.second_draw_status,
            ), "unfound bye in rnd: {} ds1: {} ds2: {}".format(
                match.rnd, match.first_draw_status, match.second_draw_status
            )
            if match.first_draw_status != "bye" and match.second_draw_status == "bye":
                scoresets = scoresets[1], scoresets[0]
            self.byeR2_sets_histo_from_lspairs[(tour.level, tour.surface)].hit(
                [scoresets]
            )

    def __process_match(self, tour, match, qual_rounds):
        self.__lucky_looser_process(tour, match, qual_rounds)
        if match.rnd not in (
            tennis.Round("Second", "vs bye"),
            tennis.Round("First", "vs qual"),
        ):
            return
        self.__sets_process(tour, match)
        keys = co.keys_combinations([{"level": tour.level}, {"surface": tour.surface}])
        if match.rnd == tennis.Round("Second", "vs bye"):
            bye_is_win = match.first_draw_status == "bye"
            self.byeR2_wlpair_histo.hit(
                keys, (match.score.normalized_total(), bye_is_win)
            )
        elif match.rnd == tennis.Round("First", "vs qual"):
            qual_is_win = match.first_draw_status == "qual"
            self.qualR1_wlpair_histo.hit(
                keys, (match.score.normalized_total(), qual_is_win)
            )

    def reporting(self):
        log.info("drawstatus {} output start...".format(self.sex))
        dirname = os.path.join(cfg_dir.stat_misc_dir(self.sex), "drawstatus")
        dict_tools.dump(
            self.byeR2_wlpair_histo,
            dirname + "/bye-at-second/winloss.txt",
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        dict_tools.dump(
            self.qualR1_wlpair_histo,
            dirname + "/qual-at-first/winloss.txt",
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            alignvalues=True,
        )
        dict_tools.dump(
            self.qualLL_winloss_histo,
            dirname + "/qual-at-first/lucky-looser-winloss.txt",
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(1),
            sortreverse=True,
            alignvalues=True,
        )

        for (level, surface), histo in self.qualR1_sets_histo_from_lspairs.items():
            rpt_filename = "{}/qual-at-first/scoresets/{}-{}.txt".format(
                dirname, level, surface
            )
            dict_tools.dump(
                histo, rpt_filename, keyfun=str, valuefun=str, alignvalues=True
            )

        for (level, surface), histo in self.byeR2_sets_histo_from_lspairs.items():
            rpt_filename = "{}/bye-at-second/scoresets/{}-{}.txt".format(
                dirname, level, surface
            )
            dict_tools.dump(
                histo, rpt_filename, keyfun=str, valuefun=str, alignvalues=True
            )
        log.info("drawstatus {} output finish".format(self.sex))


def process(
    processors,
    sex,
    min_date=None,
    max_date=None,
    time_reverse=False,
    rnd_detailing=False,
    with_paired=False,
    with_bets=False,
    with_stat=False,
    with_ratings=False,
    with_pers_det=False,
):
    if not processors:
        return
    progress = tt.YearWeekProgress(head=processors[0].__class__.__name__)
    for tour in trmt.tours_generator(
        sex,
        min_date=min_date,
        max_date=max_date,
        time_reverse=time_reverse,
        with_paired=with_paired,
        rnd_detailing=rnd_detailing,
        with_bets=with_bets,
        with_stat=with_stat,
        with_ratings=with_ratings,
        with_pers_det=with_pers_det,
    ):
        for proc in processors:
            proc.process(tour)
        progress.put_tour(tour)

    for proc in processors:
        proc.reporting()


def week_splited_process(
    processors,
    sex,
    min_date=None,
    max_date=None,
    time_reverse=False,
    with_paired=False,
    with_mix=False,
    rnd_detailing=False,
    with_bets=False,
):
    progress = tt.YearWeekProgress(head=processors[0].__class__.__name__)
    for tour in trmt.week_splited_tours(
        sex,
        min_date=min_date,
        max_date=max_date,
        time_reverse=time_reverse,
        with_paired=with_paired,
        with_mix=with_mix,
        rnd_detailing=rnd_detailing,
        with_bets=with_bets,
    ):
        for proc in processors:
            proc.process(tour)
        progress.put_tour(tour)

    for proc in processors:
        proc.reporting()


def matchstat_process(sex):
    def get_processors():
        result = []
        for fun_name in matchstat.fun_names():
            result.append(
                matchstat.MatchStatGenericProcessor(sex, fun_name, history_days_generic)
            )
            result.append(
                matchstat.MatchStatPersonalProcessor(sex, fun_name, short_history=False)
            )
        return result

    matchstat.initialize(sex=sex)
    history_days_generic = round(365.25 * 1.5)
    max_history_days = max(
        history_days_generic,
        matchstat.history_days_personal(short=False),
        matchstat.history_days_personal(short=True),
    )
    min_date = tt.now_minus_history_days(max_history_days)
    process(get_processors(), sex, min_date=min_date, with_stat=True, with_ratings=True)
    dirs_oper = matchstat.PlayersDirsBinaryOperation(
        sex, "break_points_saved", "service_win", sub
    )
    dirs_oper.run()
    dirs_oper = matchstat.PlayersDirsBinaryOperation(
        sex, "break_points_converted", "receive_win", sub
    )
    dirs_oper.run()


def do_stat():
    log.initialize(
        co.logname(__file__, instance=str(args.sex)),
        file_level="debug",
        console_level="info",
    )
    try:
        log.info(__file__ + " started sex:" + str(args.sex))
        start_datetime = datetime.datetime.now()
        dba.open_connect()
        ratings_std.initialize(sex=args.sex)
        # bet_coefs.initialize(sex, min_date=datetime.date(1999, 1, 1))
        qual_seeds.initialize()
        oncourt_players.initialize(yearsnum=18)

        matchstat_process(sex=args.sex)

        #       process((
        #                WinAfterWin1stSetProcessor(sex),
        #                WinShortSet2AfterWinSet1Processor(sex=sex),
        #                CountryMatesProcessor(sex, ('mean_value',)),
        #               Cmp75Vs64Processor(sex),
        #           ),
        #          sex, min_date=datetime.date(1990,1,1),
        #          time_reverse=True, rnd_detailing=True)
        #
        #         process((
        #                 # WinFavoriteProcessor(sex),
        #                 # WinAfterWin1stSetProcessor(sex),
        #                 # WinAfterWin2ndSetProcessor(sex),
        #                 # Set12TotalProcessor(sex),
        #                 # TourAverageSetProcessor(sex),
        #                 # total.ToursTotalProcessor(sex),
        #                 ), sex)

        #       min_date = tt.now_minus_history_days(total.history_days_count())
        #       process([total.PlayersTotalProcessor('atp'),],
        #               'atp', min_date=min_date, time_reverse=False)
        #       process([total.PlayersTotalProcessor('wta'),],
        #               'wta', min_date=min_date, time_reverse=False)

        #       process((
        #                DrawStatusProcessor(sex='wta'),
        #                Cmp75Vs64Processor('wta'),
        #              ),
        #             'wta', min_date=datetime.date(1999,1,1),
        #              time_reverse=True, rnd_detailing=False)
        #       process((
        #                DrawStatusProcessor(sex='atp'),
        #              Cmp75Vs64Processor('atp'),
        #             ),
        #            'atp', min_date=datetime.date(1999,1,1),
        #            time_reverse=True, rnd_detailing=False)
        #

        #       process((rm.RematchesProcessor('wta'),), 'wta', time_reverse=False, with_bets=True)
        #       process((rm.RematchesProcessor('atp'),), 'atp', time_reverse=False, with_bets=True)

        dba.close_connect()
        log.info(
            "{} finished within {}".format(
                __file__, str(datetime.datetime.now() - start_datetime)
            )
        )
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        sys.exit(do_stat())
    else:
        logfilename = "./make_stat-test.log"
        log.initialize(logfilename, file_level="info", console_level="info")
        unittest.main()
        sys.exit(0)
