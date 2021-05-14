# -*- coding: utf-8 -*-
import os
import sys
import datetime
from collections import Counter, defaultdict
from operator import itemgetter, concat
import argparse
import itertools
import copy
import functools
from typing import Optional

import log
import common as co
import file_utils as fu
import tennis_time as tt
import cfg_dir
import stat_cont as st
import dict_tools
import dba
import oncourt_players
import make_stat
import miss_points
import tie_spec_view
from detailed_score_dbsa import open_db
from detailed_score import (
    tie_minibreaks_count,
    tie_leadchanges_count,
    point_score_error,
    find_detailed_score2,
)

import score as sc
import detailed_score_misc as dsm
import detailed_score_srvtries as dstries
import ratings_std
from tie_spec_edge import TieEdges

# best of predicates (score, set_idx):


# for best of three case:
def bo3_pred(scr, set_idx):
    return set_idx <= 2 and not scr.best_of_five()


# for best of five case:
def bo5_pred(scr, set_idx):
    return scr.best_of_five()


# for any case:
def bo35_pred(scr, set_idx):
    return True


# for decided set case: bo3 -> set_idx=2, bo5 -> set_idx=4
def bo35dec_pred(scr, set_idx):
    return (set_idx == 2 and not scr.best_of_five()) or (
        set_idx == 4 and scr.best_of_five()
    )


def root_report_dir(sex, subdir=""):
    dirname = os.path.join(cfg_dir.stat_misc_dir(sex), "detailed_score")
    if subdir:
        dirname = os.path.join(dirname, subdir)
    return dirname


class SetInfo(object):
    typenames = ("open", "decided", "pressunder", "open2")

    def __init__(self, typename, bestof_name, bestof_pred, indicies):
        self.typename = typename
        assert typename in self.typenames, "bad typename " + typename
        self.pressundername = None  # define after call make_pressunder_name

        self.bestof_name = bestof_name
        self.bestof_pred = bestof_pred
        self.indicies = indicies  # set indicies

    def name(self):
        return "set_{}_{}_{}".format(
            "".join([str(i + 1) for i in self.indicies]),
            self.pressundername if self.typename == "pressunder" else self.typename,
            self.bestof_name,
        )

    def generic_name(self):
        this_name = self.name()
        if "open" in this_name:
            result = "open"
        elif "decided" in this_name:
            result = "decided"
        elif "under" in this_name and "press" not in this_name:
            result = "under"
        elif "press" in this_name and "under" not in this_name:
            result = "press"
        elif "pressunder" in this_name:
            result = "pressunder"
        else:
            return ""
        return "set_" + result

    def make_pressunder_name(self, left_serve, set_idx, score):
        self.pressundername = None
        if self.typename == "pressunder":
            left_win, left_loss = 0, 0  # счет по завершившимся партиям
            for sidx in range(set_idx):
                cur = score[sidx]
                if cur[0] >= cur[1]:
                    left_win += 1
                else:
                    left_loss += 1
            if left_win == (left_loss + 1) or left_loss == (left_win + 1):
                left_press = left_win == (left_loss + 1)
                if left_serve:
                    self.pressundername = "press" if left_press else "under"
                else:
                    self.pressundername = "under" if left_press else "press"


class ScoreInfo(object):
    """счет по геймам в текущей партии"""

    def __init__(self, gscore):
        self.curset_gscore = gscore  # simple tuple

    def __eq__(self, other):
        return (
            self.curset_gscore == other.curset_gscore
            and self.__class__.__name__ == other.__class__.__name__
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def make_score_key(score, set_idx, curset_gscore):
        beg_sc = ()
        for sidx in range(set_idx):
            beg_sc += (score[sidx],)
        return beg_sc + (curset_gscore,)

    def get_item(self, score, det_score, set_idx):
        score_key = self.make_score_key(score, set_idx, self.curset_gscore)
        return det_score.item(lambda k, v: k == score_key and v.valid)

    @property
    def value(self):
        return self.curset_gscore

    def left_lead(self):
        return self.curset_gscore[0] > self.curset_gscore[1]

    def name(self):
        return "{}-{}".format(self.curset_gscore[0], self.curset_gscore[1])


class ScoreInfoFlipable(ScoreInfo):
    def __init__(self, gscore):
        super(ScoreInfoFlipable, self).__init__(gscore)
        self.flip = None

    def get_item(self, score, det_score, set_idx):
        self.flip = None
        item = super(ScoreInfoFlipable, self).get_item(score, det_score, set_idx)
        if item is not None:
            self.flip = False
            return item
        flip_score = (self.curset_gscore[1], self.curset_gscore[0])
        score_key = self.make_score_key(score, set_idx, flip_score)
        item = det_score.item(lambda k, v: k == score_key and v.valid)
        if item is not None:
            self.flip = True
        return item

    def name(self):
        minv = min(self.curset_gscore[0], self.curset_gscore[1])
        maxv = max(self.curset_gscore[0], self.curset_gscore[1])
        return "{}-{}".format(maxv, minv)

    def left_lead(self):
        assert self.flip is not None, "ScoreInfoFlipable with None flip"
        left_adv = self.curset_gscore[0] > self.curset_gscore[1]
        return left_adv is not self.flip

    @property
    def value(self):
        if self.flip:
            return self.curset_gscore[1], self.curset_gscore[0]
        return self.curset_gscore


class ScorePursueInfo(ScoreInfo):
    def __init__(self, gscore):
        super(ScorePursueInfo, self).__init__(gscore)
        assert (gscore[0] + 1) == gscore[1], "bad pursue info {}".format(gscore)

    def get_item(self, score, det_score, set_idx):
        minv, maxv = min(self.curset_gscore), max(self.curset_gscore)
        score_key_ladv = self.make_score_key(score, set_idx, (maxv, minv))
        score_key_radv = self.make_score_key(score, set_idx, (minv, maxv))
        predicate = lambda k, v: v.valid and (
            (k == score_key_ladv and v.right_opener)
            or (k == score_key_radv and v.left_opener)
        )
        return det_score.item(predicate)


class ScoreBreakupInfo(ScoreInfo):
    def get_item(self, score, det_score, set_idx):
        minv, maxv = min(self.curset_gscore), max(self.curset_gscore)
        score_key_ladv = self.make_score_key(score, set_idx, (maxv, minv))
        score_key_radv = self.make_score_key(score, set_idx, (minv, maxv))
        predicate = lambda k, v: v.valid and (
            (k == score_key_ladv and v.left_opener)
            or (k == score_key_radv and v.right_opener)
        )
        return det_score.item(predicate)


class Processor(object):
    def __init__(
        self,
        sex,
        workers,
        max_rating : Optional[int] = 350,
        max_rating_dif: Optional[int] = 250,
    ):
        self.sex = sex
        self.workers = workers
        self.max_rating = max_rating
        self.max_rating_dif = max_rating_dif

    def is_ratings_care(self):
        return (
            self.max_rating is not None
            and self.max_rating_dif is not None
            and ratings_std.initialized()
        )

    def process(self, tour):
        if tour.level in ("junior", "future"):
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if (
                rnd == "Pre-q"
                or
                # (tour.level != 'masters' and rnd.qualification())
                (tour.level == "chal" and rnd.qualification())
            ):
                continue
            keys = co.Keys.soft_main_maker(tour, rnd)
            for match in matches:
                if match.paired():
                    continue
                if not match.set_decided_tiebreak_info(tour, rnd):
                    continue
                if find_detailed_score2(dbdet.records, tour, rnd, match):
                    if self.is_ratings_care():
                        match.read_ratings(tour.sex, tour.date)
                        if match.is_ranks_both_below(self.max_rating) in (True, None):
                            continue
                        if match.is_ranks_dif_wide(self.max_rating_dif) in (True, None):
                            continue
                    for worker in self.workers:
                        worker.process_match(match, keys, sex=self.sex)

    def reporting(self):
        log.info("{} detscore output start...".format(self.sex))
        for worker in self.workers:
            worker.reporting(self.sex)
        log.info("{} detscore output finish".format(self.sex))


class TinkleSubdirStorage(object):
    def __init__(self, sex, subdir):
        self.sex = sex
        self.subdir = subdir
        # item ~ one file: name -> container
        self.histo_dict = defaultdict(st.create_winloss_histogram)

    def put(self, name, keys, result):
        """keys must be co.Keys object"""
        self.histo_dict[name].hit(keys.combinations(), result)

    def write_reports(self):
        dirname = os.path.join(root_report_dir(self.sex), self.subdir)
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class TinkleProcessor(object):
    def __init__(self, sex):
        self.sex = sex
        self.subdir_stores = [TinkleSubdirStorage(sex, "tinkle")]

    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q":
                continue
            keys = co.Keys.soft_main_maker(tour, rnd, with_surface=False)
            for match in matches:
                if match.score.valid() and not match.paired():
                    if find_detailed_score2(dbdet.records, tour, rnd, match):
                        self.process_match(match, keys)

    def process_match(self, match, keys):
        for sc_key, det_game in match.detailed_score.items():
            set_num = len(sc_key)
            if not det_game.valid or set_num not in (1, 3, 5):
                continue
            if set_num == 3 and (
                (sc_key[0][0] > sc_key[0][1] and sc_key[1][0] > sc_key[1][1])
                or (sc_key[0][0] < sc_key[0][1] and sc_key[1][0] < sc_key[1][1])
            ):
                continue
            set_sc_t = sc_key[-1]
            if abs(set_sc_t[0] - set_sc_t[1]) > 1 or det_game.tiebreak:
                continue
            if set_sc_t[0] == (set_sc_t[1] + 1) and det_game.left_opener:
                continue  # left leads break up
            if (set_sc_t[0] + 1) == set_sc_t[1] and det_game.right_opener:
                continue  # right leads break up
            next_item = match.detailed_score.item_after(sc_key)
            if (
                next_item is None
                or len(next_item[0]) != set_num
                or next_item[1].tiebreak
            ):
                continue
            if (
                next_item[1].valid
                and next_item[1].left_opener is not det_game.left_opener
            ):
                self.process_case(keys, (sc_key, det_game), next_item)

    def process_case(self, keys, item, next_item):
        def get_ext_keys():
            ext_keys = copy.deepcopy(keys)
            ext_keys["hold"] = "Y" if item[1].opener_wingame else "N"
            ext_keys["len"] = str(min(len(item[1].points) / 5, 5))
            gstat = dsm.GameStat.from_detailed_game(item[1])
            tinkle_sval = gstat.tinkle_gp_ratio(left=item[1].left_opener, min_size=5)
            if tinkle_sval:
                tval = tinkle_sval.value
                if tval >= 0.64:
                    ext_keys["tinkle"] = "pos"
                elif tval <= 0.36:
                    ext_keys["tinkle"] = "neg"
            return ext_keys

        self.subdir_stores[0].put(
            name="tinkle", keys=get_ext_keys(), result=not next_item[1].opener_wingame
        )

    def reporting(self, sex=None):
        log.info("tinkle output start (sex={})...".format(self.sex))
        for subdir_store in self.subdir_stores:
            subdir_store.write_reports()
        log.info("tinkle output finish (sex={})...".format(self.sex))


class BidirectTrendsWorker(object):
    """wta 1set bidirect-trends-hold(22_to_54) and then is_breakpoints:
        level=main________________59.1% (22) here 22 - n_matches
        all_______________________57.1% (28)
    wta 3set bidirect-trends-hold(33_to_65) and then is_breakpoints:
        level=main________________66.7% (6)
        all_______________________62.5% (8)

    """

    def __init__(self, setinfo, result_type):
        self.setinfo = setinfo
        self.result_type = result_type  # 'is_hold', 'is_bp'
        # filename -> winloss_histogram (result_type ....with keys views)
        self.histo_dict = defaultdict(st.create_winloss_histogram)
        self.beg_scores = ((0, 0), (1, 1), (2, 2), (1, 1), (3, 3))
        self.fin_scores = ((3, 2), (4, 3), (5, 4), (5, 4), (6, 5))
        # start/finish game tense conditions
        self.improver_beg_cond = lambda t: t.value >= 1.0
        self.improver_end_cond = lambda t: t.value <= 1.0
        self.worser_beg_cond = lambda t: t.value <= 3.0
        self.worser_end_cond = lambda t: t.value >= 4.0

    @staticmethod
    def make_workers(sex):
        if sex == "atp":
            predicate = lambda s: "boA" in s.bestof_name and s.typename in (
                "open",
                "decided",
            )
        else:
            predicate = lambda s: s.typename in ("open", "decided")
        return [
            BidirectTrendsWorker(si, res_type)
            for si in setinfos(sex, predicate)
            for res_type in ("is_hold", "is_bp", "lead_ratio")
        ]

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            setnum = set_idx + 1
            if setnum > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            if not match.detailed_score.set_valid(setnum):
                continue
            set_items = list(match.detailed_score.set_items(setnum))
            self.process_set(keys, set_items, setnum)

    def process_set(self, keys, set_items, setnum):
        def get_sub_items(beg_score, fin_score):
            beg_idx = sum(beg_score)
            fin_idx = sum(fin_score) - 1
            res_items = set_items[beg_idx : fin_idx + 1]
            beg_item = res_items[0]
            if beg_item[0][setnum - 1] == beg_score:
                return res_items

        def is_breaks(items):
            for item in items:
                if not item[1].hold:
                    return True
            return False

        def opener_has_desc_tenses(items):
            n_games = len(items)
            tenses = [dsm.GameTense(items[i][1]) for i in range(0, n_games, 2)]
            if not self.improver_beg_cond(tenses[0]):
                return False
            if not self.improver_end_cond(tenses[-1]):
                return False
            prev_tense = tenses[0]
            for tense in tenses[1:]:
                if not (tense <= prev_tense):
                    return False
                prev_tense = tense
            return True

        def closer_has_asc_tenses(items):
            n_games = len(items)
            tenses = [dsm.GameTense(items[i][1]) for i in range(1, n_games, 2)]
            if not self.worser_beg_cond(tenses[0]):
                return False
            if not self.worser_end_cond(tenses[-1]):
                return False
            prev_tense = tenses[0]
            for tense in tenses[1:]:
                if tense < prev_tense:
                    return False
                prev_tense = tense
            return True

        for beg_score, fin_score in zip(self.beg_scores, self.fin_scores):
            n_games = sum(fin_score)
            if len(set_items) < (n_games + 1):
                continue
            sub_items = get_sub_items(beg_score, fin_score)
            if not sub_items:
                continue
            if is_breaks(sub_items):
                continue
            if not opener_has_desc_tenses(sub_items):
                continue
            if not closer_has_asc_tenses(sub_items):
                continue
            if self.result_type == "is_hold":
                result = set_items[n_games][1].hold
            elif self.result_type == "is_bp":
                result = dsm.break_points_count(set_items[n_games][1]) > 0
            elif self.result_type == "lead_ratio":
                result = dsm.lead_ratio(set_items[n_games][1])
            else:
                raise co.TennisError(
                    "unexpected result_type: '{}'".format(self.result_type)
                )
            self.histo_dict[self.key_filename(beg_score, fin_score)].hit(
                keys.combinations(), result
            )

    def key_filename(self, beg_score, fin_score):
        return co.joined_name(
            [
                self.result_type,
                self.setinfo.name(),
                co.joined_name(beg_score),
                "to",
                co.joined_name(fin_score),
            ]
        )

    def reporting(self, sex):
        dirname = root_report_dir(sex, subdir="bidirect_trends")
        fu.ensure_folder(dirname)
        self.reporting_subdir(dirname, self.histo_dict)

    @staticmethod
    def reporting_subdir(dirname, histo_dict):
        assert os.path.isdir(dirname), "not found dir " + dirname
        for keyname, histo in histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class BreaksMovesWorker(object):
    """Работа с раскладкой волн (BreaksMoves) в сете.
    Когда первая волна симметрична (count == back_count)
    т.е. есть равновеликий отыгрыш брейков со стороны оппонента и count in (1, 2).
    Определяем: A) частоту выигрыша сета инициатором первой волны.
                B) в файлах с 'nextinitorsame' определяем частоту того,
                   что инициатор второй волны совпадет с инициатором первой.
                C) в подпапке '3way' определяем процентное распределение 3 вариантов
                   после первой волны с level=main и count=1. 3 варианта:
                   - после 1-й волны конец ('end') - нет новой волны,
                   - после 1-й волны инициатор второй волны тот же ('same'),
                   - после 1-й волны инициатор второй волны поменялся ('change')
                   Недостаток реализации C (некримнальный): в 'end' попали случаи
                   когда после обратного брейка счет 6-6 и уже просто конец дистанции
                   без всякой возможности 2-й волны (эти случаи лучше пропускать).
                D) в подпапке 'tfft' доля winset инитором 1 волны в 3 файлах:
                   - общий случай.
                   - открыватель сета есть инитор 1 волны,
                   - открыватель сета не есть инитор 1 волны
    """

    def __init__(self, setinfo):
        self.setinfo = setinfo
        self.histo_dict = defaultdict(st.create_winloss_histogram)
        self.way3_from_descname = defaultdict(Counter)
        self.tfft_histo_dict = defaultdict(st.create_winloss_histogram)

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            setnum = set_idx + 1
            if setnum > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            if not match.detailed_score.set_valid(setnum):
                continue
            left_winset = match.score[set_idx][0] > match.score[set_idx][1]
            breaks_moves = list(
                match.detailed_score.breaks_moves(
                    lambda k, v, sn=setnum: len(k) == sn and v.valid
                )
            )
            if breaks_moves:
                fst_item = match.detailed_score.item(
                    lambda k, v, sn=setnum: len(k) == sn and k[-1] == (0, 0) and v.valid
                )
                self.process_case(
                    keys, breaks_moves, left_winset, fst_item[1].left_opener
                )

    def process_case(self, keys, breaks_moves, left_winset, set_left_opener):
        fst_move = breaks_moves[0]
        symmetric = fst_move.count == fst_move.back_count
        if not symmetric or fst_move.count not in (1, 2):
            return
        initor_winset = fst_move.initor is left_winset
        name = "{}_nwav{}_cnt{}".format(
            self.setinfo.name(), len(breaks_moves), fst_move.count
        )
        self.histo_dict[name].hit(keys.combinations(), initor_winset)

        if len(breaks_moves) > 1:
            next_initor_same = fst_move.initor is breaks_moves[1].initor
            name2 = "{}_nwav{}_cnt{}_nextinitorsame".format(
                self.setinfo.name(), len(breaks_moves), fst_move.count
            )
            self.histo_dict[name2].hit(keys.combinations(), next_initor_same)
        self.process_case_3way(keys, breaks_moves)
        self.process_case_tfft(keys, breaks_moves, left_winset, set_left_opener)

    def process_case_3way(self, keys, breaks_moves):
        fst_move = breaks_moves[0]
        if fst_move.count != 1 or keys["level"] not in ("main", "chal", "qual"):
            return
        descname = keys["level"] + "_" + self.setinfo.name()
        if len(breaks_moves) == 1:
            self.way3_from_descname[descname]["end"] += 1
        elif len(breaks_moves) > 1:
            next_initor_same = fst_move.initor is breaks_moves[1].initor
            if next_initor_same:
                self.way3_from_descname[descname]["same"] += 1
            else:
                self.way3_from_descname[descname]["change"] += 1

    def process_case_tfft(self, keys, breaks_moves, left_winset, set_left_opener):
        if len(breaks_moves) <= 1 or keys["level"] not in ("main", "chal", "qual"):
            return
        fst_move = breaks_moves[0]
        snd_move = breaks_moves[1]
        if (
            fst_move.count != 1
            or fst_move.back_count != 1
            or snd_move.count != 1
            or snd_move.back_count != 1
            or fst_move.initor is snd_move.initor
        ):
            return
        m1initor_winset = fst_move.initor is left_winset
        name = keys["level"] + "_" + self.setinfo.name()
        self.tfft_histo_dict[name].hit(keys.combinations(), m1initor_winset)
        if set_left_opener is fst_move.initor:
            self.tfft_histo_dict[name + "_openisinit"].hit(
                keys.combinations(), m1initor_winset
            )
        else:
            self.tfft_histo_dict[name + "_openisnotinit"].hit(
                keys.combinations(), m1initor_winset
            )

    def reporting(self, sex):
        dirname = root_report_dir(sex, subdir="breaks_moves")
        self.reporting_subdir(dirname, self.histo_dict)

        subdirname = os.path.join(dirname, "tfft")
        self.reporting_subdir(subdirname, self.tfft_histo_dict)

        self.reporting_3way(dirname)

    @staticmethod
    def reporting_subdir(dirname, histo_dict):
        assert os.path.isdir(dirname), "not found dir " + dirname
        for keyname, histo in histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )

    def reporting_3way(self, dirname):
        subdirname = os.path.join(dirname, "3way")
        assert os.path.isdir(subdirname), "not found dir " + subdirname
        for descname, cntr in self.way3_from_descname.items():
            if not cntr:
                continue
            filename = os.path.join(subdirname, descname + ".txt")
            dict_tools.dump(
                cntr,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                alignvalues=True,
            )


class BreaksAverageCountWorker(object):
    """ср число брейков в сете"""

    def __init__(self, setinfo):
        self.setinfo = setinfo
        self.histo_dict = defaultdict(st.create_summator_histogram)

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            setnum = set_idx + 1
            if setnum > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            if not match.detailed_score.set_valid(setnum):
                continue
            breaks = list(
                match.detailed_score.breaks(lambda k, _, sn=setnum: len(k) == sn)
            )
            self.process_case(keys, len(breaks))

    def process_case(self, keys, breaks_count):
        name = "avgbrkcount_{}".format(self.setinfo.name())
        self.histo_dict[name].hit(keys.combinations(), breaks_count)

    def reporting(self, sex):
        dirname = root_report_dir(sex, subdir="breaks_count")
        assert os.path.isdir(dirname), "not found dir " + dirname
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class BreaksSeqCountWorker(object):
    """для каждой последовательности брейков в сете подсчет количества.
    Например запись 101___282 означает 282 случая такой посл-ти:
    выиграл брейк, проиграл брейк, выиграл брейк"""

    def __init__(self, setinfo):
        self.setinfo = setinfo
        self.histo_dict = defaultdict(st.create_histogram)

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            setnum = set_idx + 1
            if setnum > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            if not match.detailed_score.set_valid(setnum):
                continue
            breaks = list(
                match.detailed_score.breaks(
                    lambda k, v, sn=setnum: len(k) == sn and v.valid
                )
            )
            self.process_case(keys, breaks)

    def process_case(self, keys, breaks):
        def breaks_to_str():
            if len(breaks) == 0:
                return "z"
            if breaks[0]:
                return functools.reduce(concat, (str(int(v)) for v in breaks), "")
            # инвертируем чтобы первый элемент был 1 (True) в любом случае
            return functools.reduce(concat, (str(int(not v)) for v in breaks), "")

        def keys_to_str():
            return "-".join((str(v) for v in keys.values()))

        name = "{}_{}".format(keys_to_str(), self.setinfo.name())
        self.histo_dict[name].hit([breaks_to_str()])

    def reporting(self, sex):
        dirname = root_report_dir(sex, subdir="breaks_seq_count")
        assert os.path.isdir(dirname), "not found dir " + dirname
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                sortfun=itemgetter(1),
                sortreverse=True,
                alignvalues=True,
            )


class SrvTriesWorker(object):
    eq_scores = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6))
    beg_eq_scores = ((0, 0), (1, 1), (2, 2))
    end_eq_scores = ((3, 3), (4, 4), (5, 5), (6, 6))
    lt_scores = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
    beg_lt_scores = ((0, 1), (1, 2), (2, 3))
    end_lt_scores = ((3, 4), (4, 5), (5, 6))
    gt_scores = ((1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5))
    beg_gt_scores = ((1, 0), (2, 1), (3, 2))
    end_gt_scores = ((4, 3), (5, 4), (6, 5))

    @staticmethod
    def make_workers(sex):
        if sex == "atp":
            predicate = lambda s: "boA" in s.bestof_name and s.typename in (
                "open",
                "decided",
            )
        else:
            predicate = lambda s: s.typename in ("open", "decided")
        return [SrvTriesWorker(si) for si in setinfos(sex, predicate)]

    def __init__(self, setinfo):
        self.setinfo = setinfo
        self.eq_outer_dict = dstries.make_outer_dict(self.eq_scores)
        self.lt_outer_dict = dstries.make_outer_dict(self.lt_scores)
        self.gt_outer_dict = dstries.make_outer_dict(self.gt_scores)

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            if (set_idx + 1) > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            for scores, cmpval in (
                (self.eq_scores, co.EQ),
                (self.lt_scores, co.LT),
                (self.gt_scores, co.GT),
            ):
                for scr in scores:
                    item = None
                    if cmpval == co.EQ:
                        inner_dct = self.eq_outer_dict[scr]
                        item = match.detailed_score.item(
                            lambda k, g: (
                                len(k) == (set_idx + 1)
                                and k[-1] == scr
                                and not g.tiebreak
                            )
                        )
                    elif cmpval == co.LT:
                        inner_dct = self.lt_outer_dict[scr]
                        scr_rev = (scr[1], scr[0])
                        item = match.detailed_score.item(
                            lambda k, g: (
                                len(k) == (set_idx + 1)
                                and (
                                    (k[-1] == scr and g.left_opener)
                                    or (k[-1] == scr_rev and g.right_opener)
                                )
                            )
                        )
                    elif cmpval == co.GT:
                        inner_dct = self.gt_outer_dict[scr]
                        scr_rev = (scr[1], scr[0])
                        item = match.detailed_score.item(
                            lambda k, g: (
                                len(k) == (set_idx + 1)
                                and (
                                    (k[-1] == scr and g.left_opener)
                                    or (k[-1] == scr_rev and g.right_opener)
                                )
                            )
                        )
                    if item is not None:
                        det_game = item[1]
                        if (
                            det_game.valid
                            and not point_score_error(det_game.error)
                            and not det_game.extra_points_error()
                        ):
                            dstries.collect_detgame_eq_routes(det_game, inner_dct)

    def reporting(self, sex):
        def report_by_cmp(cmpval):
            if cmpval == co.EQ:
                scores = self.eq_scores
                beg_scores, end_scores = self.beg_eq_scores, self.end_eq_scores
                outer_dict = self.eq_outer_dict
            elif cmpval == co.LT:
                scores = self.lt_scores
                beg_scores, end_scores = self.beg_lt_scores, self.end_lt_scores
                outer_dict = self.lt_outer_dict
            elif cmpval == co.GT:
                scores = self.gt_scores
                beg_scores, end_scores = self.beg_gt_scores, self.end_gt_scores
                outer_dict = self.gt_outer_dict
            else:
                raise co.TennisError("bad cmpval: {}".format(cmpval))

            for scr in scores:
                shortname = self.setinfo.generic_name() + "_{}_{}-{}.txt".format(
                    cmpval.name, scr[0], scr[1]
                )
                filename = os.path.join(dirname, shortname)
                self.output_dict(filename, outer_dict[scr])

            filename = os.path.join(
                dirname, self.setinfo.generic_name() + "_{}_beg.txt".format(cmpval.name)
            )
            self.output_dict(
                filename, dstries.plus_dict_by_score(outer_dict, beg_scores)
            )

            filename = os.path.join(
                dirname, self.setinfo.generic_name() + "_{}_end.txt".format(cmpval.name)
            )
            self.output_dict(
                filename, dstries.plus_dict_by_score(outer_dict, end_scores)
            )

        dirname = root_report_dir(sex, subdir="srvtries")
        fu.ensure_folder(dirname)
        report_by_cmp(co.EQ)
        report_by_cmp(co.LT)
        report_by_cmp(co.GT)

    @staticmethod
    def output_dict(filename, inner_dict):
        with open(filename, "w") as fhandle_out:
            for routename, tvals in inner_dict.items():
                wls_text = " ".join(
                    [
                        "try" + str(n) + " " + str(wl)
                        for n, wl in enumerate(tvals, start=1)
                    ]
                )
                fhandle_out.write(routename + "\t" + wls_text + "\n")


class DebugLogCount(object):
    """designed for TiebreakWorker::adveqadv"""

    def __init__(self):
        self.max_log_count = 3
        self.advn_result_dict = defaultdict(lambda: 0)  # (advn,_result) -> count

    def is_admit_log(self, advn, result):
        self.advn_result_dict[(advn, result)] += 1
        return self.advn_result_dict[(advn, result)] <= self.max_log_count


tie_adveqadv_log_count = DebugLogCount()


class TiebreakWorker(object):
    # level -> {к-во изменений лидерства в тб -> к-во т.е. сколько таких ситуаций}
    leadchangescnt_from_level = defaultdict(Counter)
    # (dif, trynum) -> WinLoss(успешность after cycle)
    eqdifeq_dict = defaultdict(st.WinLoss)
    adveqadv_scat_dict_static = defaultdict(lambda: defaultdict(st.WinLoss))
    tie_edges = TieEdges()
    perf_sn_dicts = tie_spec_view.PerfDicts(keys=[], skip_super=False)
    perf_surf_dicts = tie_spec_view.PerfDicts(
        keys=["Hard", "Clay", "Carpet", "Grass"], skip_super=False
    )

    @staticmethod
    def reporting_leadchanges(sex):
        dirname = root_report_dir(sex, subdir="tie")
        subdirname = os.path.join(dirname, "leadchanges")
        fu.ensure_folder(subdirname)
        for level, cntr in TiebreakWorker.leadchangescnt_from_level.items():
            if not cntr:
                continue
            filename = os.path.join(subdirname, str(level) + ".txt")
            cntr_perc = dict_tools.transed_value_percent_text(cntr)
            dict_tools.dump(
                cntr_perc,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                alignvalues=True,
            )

    @staticmethod
    def reporting_eqdifeq(sex):
        dirname = root_report_dir(sex, subdir="tie")
        filename = os.path.join(dirname, "eqdifeq.txt")
        dict_tools.dump(
            TiebreakWorker.eqdifeq_dict,
            filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            sortreverse=True,
        )

    @staticmethod
    def reporting_adveqadv_static(sex):
        dir_root = root_report_dir(sex, subdir="tie")
        dirname = os.path.join(dir_root, "adveqadv")
        for adv_n, scat_dict in TiebreakWorker.adveqadv_scat_dict_static.items():
            filename = os.path.join(
                dirname, "adv{}_scat_static.txt".format(adv_n, adv_n)
            )
            dict_tools.dump(
                scat_dict,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=lambda i: i[1].ratio,
            )

    @staticmethod
    def reporting_tie_edge(sex):
        dir_root = root_report_dir(sex, subdir="tie")
        dirname = os.path.join(dir_root, "tie_edge")
        TiebreakWorker.tie_edges.report(dirname)

    @staticmethod
    def reporting_perf_dict(sex):
        dir_root = root_report_dir(sex, subdir="tie")
        dirname = os.path.join(dir_root, "perf_dict")
        TiebreakWorker.perf_sn_dicts.report(dirname, file_prefix="sn")
        TiebreakWorker.perf_surf_dicts.report(dirname, file_prefix="surf")

    @staticmethod
    def clear_static():
        TiebreakWorker.leadchangescnt_from_level = defaultdict(Counter)
        TiebreakWorker.eqdifeq_dict.clear()
        TiebreakWorker.adveqadv_scat_dict_static = st.WinLoss()

    @staticmethod
    def make_workers(sex):
        def wta_cond(setinfo):
            return "open" in setinfo.typename or "decided" in setinfo.typename

        def atp_cond(setinfo):
            return (
                "boA" in setinfo.bestof_name and setinfo.typename in ("open", "decided")
            ) or ("bo5" in setinfo.bestof_name and setinfo.typename == "open2")

        return [
            TiebreakWorker(si)
            for si in setinfos(sex, wta_cond if sex == "wta" else atp_cond)
        ]

    def __init__(self, setinfo):
        self.setinfo = setinfo

        # setnametiescore -> {levsurfkey -> WinLoss(успешность тб)}
        self.histo_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> WinLoss(успешность тб)} после 44hold, 55hold
        self.after44h55h_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> WinLoss(успешность тб)} после 56hold
        self.after56hold_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> WinLoss(успешность тб)} после 65brk
        self.after65brk_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> WinLoss(успешность beg6тб)} после 56hold
        self.beg6after56hold_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> WinLoss(успешность beg6тб)} после 65brk
        self.beg6after65brk_dict = defaultdict(st.create_winloss_histogram)

        # setname -> {levsurfkey -> Sumator(ср. число минитб)}
        self.minibreaks_dict = defaultdict(st.create_summator_histogram)

        # setname -> {levsurfkey -> Sumator(ср. число изменений лидерства в тб)}
        self.leadchanges_dict = defaultdict(st.create_summator_histogram)

        # setname -> {levsurfkey -> Sumator(ср. число розыгрышей в тб)}
        self.points_dict = defaultdict(st.create_summator_histogram)

        # advN -> {scattered_key -> WinLoss(успешность next_advance2 for init_leader)}
        self.adveqadv_scat_dict = defaultdict(lambda: defaultdict(st.WinLoss))

        for set_idx in self.setinfo.indicies:
            self.tie_edges.register(setname=self.setinfo.generic_name(), idx=set_idx)
            self.perf_sn_dicts.add_key(key=self.setinfo.generic_name())

    def process_match(self, match, keys, sex=None):
        sets_count = match.score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            if (set_idx + 1) > sets_count:
                break
            if not self.setinfo.bestof_pred(match.score, set_idx):
                continue
            item = match.detailed_score.item(
                lambda k, g, si=set_idx: len(k) == (si + 1) and g.tiebreak
            )
            if item is not None:
                det_game = item[1]
                if (
                    len(det_game) == 0
                    or (item[0][-1] not in ((0, 0), (6, 6), (12, 12)))
                    or (
                        item[0][-1] == (0, 0)
                        and (not det_game.supertiebreak or set_idx not in (2, 4))
                    )
                ):
                    return
                self.process_case(match, keys, set_idx, det_game)

    def process_case(self, match, keys, set_idx, det_game):
        self.setinfo.make_pressunder_name(det_game.left_opener, set_idx, match.score)
        tsc_mbname_mbdif = (
            (("1", "0"), "mbeq", 0),
            (("0", "1"), "mbdw", -1),
            (("4", "2"), "mbup", 1),
            (("2", "4"), "mbdw", -1),
            (("6", "5"), "mbup", 1),
            (("5", "6"), "mbeq", 0),
        )
        points = [p for p in det_game]
        for tsc, mbname, mbdif in tsc_mbname_mbdif:
            pnt_idx = int(tsc[0]) + int(tsc[1]) - 1
            if pnt_idx >= len(points):
                continue
            point = points[pnt_idx]
            if (
                point.text_score(before=False) == tsc
                and point.minibreaks_diff() == mbdif
            ):
                name = "{}_{}-{}_{}".format(self.setinfo.name(), tsc[0], tsc[1], mbname)
                self.histo_dict[name].hit(keys.combinations(), det_game.opener_wingame)
        self.tie_edges.put_data(
            det_game=det_game, setname=self.setinfo.generic_name(), idx=set_idx
        )
        self.perf_surf_dicts.cumulate_data(keys["surface"], det_game)
        self.perf_sn_dicts.cumulate_data(self.setinfo.generic_name(), det_game)
        self.process_eqdifeq_case(det_game)
        result, advn = self.process_adveqadv_scat_cases(det_game, keys)
        if result is not None and tie_adveqadv_log_count.is_admit_log(advn, result):
            log.info(
                "-----tie_adveqadv res {} advn {} seti {}\n\t{}\n\t{}".format(
                    result, advn, set_idx, match, det_game
                )
            )
        self.process_minibreaks_case(keys, det_game)
        self.process_leadchanges_case(keys, det_game)
        self.process_points_case(keys, det_game)
        self.process_after44h55h_case(match, keys, set_idx, det_game)
        set_items = list(match.detailed_score.set_items(setnum=set_idx + 1))
        cmp_val = self.set_allow_bp_compare(set_items, det_game.left_opener)
        ext_keys = copy.copy(keys)
        ext_keys["abp"] = str(cmp_val)
        self.process_after56hold_case(ext_keys, set_idx, set_items, det_game)
        self.process_after65brk_case(ext_keys, set_idx, set_items, det_game)
        self.process_beg6after56hold_case(ext_keys, set_idx, set_items, det_game)
        self.process_beg6after65brk_case(ext_keys, set_idx, set_items, det_game)

    @staticmethod
    def process_eqdifeq_case(det_game):
        points = list(iter(det_game))
        for dif in (6, 5, 4, 3, 2):
            res = tie_spec_view.info_eqdifeq(points, dif)
            if not res:
                continue
            dct = tie_spec_view.result_to_dict(res)
            for trynum in dct:
                TiebreakWorker.eqdifeq_dict[(dif, trynum)] += dct[trynum]
            break

    def process_adveqadv_scat_cases(self, det_game, keys):
        def next_dif2(x, y, i):
            return abs(x - y) >= 2

        def do_adv_n(adv_n):
            init_leader_lead_again = tie_spec_view.tie_adv_equal_next_adv_ext(
                det_game,
                adv_cond=lambda x, y: abs(x - y) == adv_n,
                is_adv_unique=True,
                next_adv_cond_ext=next_dif2,
            )
            if init_leader_lead_again is not None:
                for skey in keys.combinations():
                    self.adveqadv_scat_dict[adv_n][str(skey)].hit(
                        init_leader_lead_again
                    )
                return init_leader_lead_again

        result = None, None
        res = do_adv_n(adv_n=2)
        if res is not None:
            result = (res, 2)
        res = do_adv_n(adv_n=3)
        if res is not None:
            result = (res, 3)
        res = do_adv_n(adv_n=4)
        if res is not None:
            result = (res, 4)
        res = do_adv_n(adv_n=5)
        if res is not None:
            result = (res, 5)
        return result

    def process_after44h55h_case(self, match, keys, set_idx, det_game):
        item44hold = match.detailed_score.item(
            (
                lambda k, g: len(k) == (set_idx + 1)
                and k[set_idx] == (4, 4)
                and g.opener_wingame
            )
        )
        if item44hold is None:
            return
        item55hold = match.detailed_score.item(
            (
                lambda k, g: len(k) == (set_idx + 1)
                and k[set_idx] == (5, 5)
                and g.opener_wingame
            )
        )
        if item55hold is None:
            return
        self.after44h55h_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), det_game.opener_wingame
        )

    @staticmethod
    def set_allow_bp_compare(set_items, tie_left_opener):
        """для tie_left_opener вернем LT or EQ or GT.
        LT если он предоставил оппон-ту меньше BP на своих подачах,
        EQ - если примерно равное к-во, GT - если большее к-во.
        Например: 0-40 и тут же нас брейканули: мы предоставили 3 bp.
        """
        # allow - к-во bp которое игрок предоставил оппоненту:
        left_allow, right_allow = dsm.allow_breakpoints_not_tie(set_items)
        cmp_val = dsm.allow_breakpoints_compare(left_allow, right_allow)
        if not tie_left_opener:
            cmp_val = co.flip_compare(cmp_val)
        return cmp_val

    def process_after65brk_case(self, keys, set_idx, set_items, det_game):
        item65brk = co.find_first(
            set_items,
            (
                lambda i: (
                    (i[0][set_idx] == (6, 5) and i[1].left_opener)
                    or (i[0][set_idx] == (5, 6) and i[1].right_opener)
                )
                and not i[1].opener_wingame
            ),
        )
        if item65brk is None:
            return
        self.after65brk_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), det_game.opener_wingame
        )

    def process_after56hold_case(self, keys, set_idx, set_items, det_game):
        item56hold = co.find_first(
            set_items,
            (
                lambda i: (
                    (i[0][set_idx] == (6, 5) and i[1].right_opener)
                    or (i[0][set_idx] == (5, 6) and i[1].left_opener)
                )
                and i[1].opener_wingame
            ),
        )
        if item56hold is None:
            return
        self.after56hold_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), det_game.opener_wingame
        )

    def process_beg6after56hold_case(self, keys, set_idx, set_items, det_game):
        item56hold = co.find_first(
            set_items,
            (
                lambda i: (
                    (i[0][set_idx] == (6, 5) and i[1].right_opener)
                    or (i[0][set_idx] == (5, 6) and i[1].left_opener)
                )
                and i[1].opener_wingame
            ),
        )
        if item56hold is None:
            return
        left_right_cnts = sc.tie_begin6_score(det_game)
        if left_right_cnts is None:
            return
        left_cnt, right_cnt = left_right_cnts
        if left_cnt == right_cnt:
            return  # equal 3-3
        left_win_beg6 = left_cnt > right_cnt
        opener_win_beg6 = det_game.left_opener is left_win_beg6
        self.beg6after56hold_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), opener_win_beg6
        )

    def process_beg6after65brk_case(self, keys, set_idx, set_items, det_game):
        item65brk = co.find_first(
            set_items,
            (
                lambda i: (
                    (i[0][set_idx] == (6, 5) and i[1].left_opener)
                    or (i[0][set_idx] == (5, 6) and i[1].right_opener)
                )
                and not i[1].opener_wingame
            ),
        )
        if item65brk is None:
            return
        left_right_cnts = sc.tie_begin6_score(det_game)
        if left_right_cnts is None:
            return
        left_cnt, right_cnt = left_right_cnts
        if left_cnt == right_cnt:
            return  # equal 3-3
        left_win_beg6 = left_cnt > right_cnt
        opener_win_beg6 = det_game.left_opener is left_win_beg6
        self.beg6after65brk_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), opener_win_beg6
        )

    def process_minibreaks_case(self, keys, det_game):
        minibr_cnt = tie_minibreaks_count(det_game)
        self.minibreaks_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), minibr_cnt
        )

    def process_leadchanges_case(self, keys, det_game):
        leadchanges_cnt = tie_leadchanges_count(det_game)
        self.leadchanges_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), leadchanges_cnt
        )
        cnt = min(1, leadchanges_cnt)  # 0 - no leadchanges, 1 - exist leadchanges
        levname = keys["level"]
        TiebreakWorker.leadchangescnt_from_level[levname][cnt] += 1
        fst_point = next(iter(det_game))
        if fst_point is not None:
            if fst_point.win():
                TiebreakWorker.leadchangescnt_from_level[levname + "_p1WinSrv"][
                    cnt
                ] += 1
            else:
                TiebreakWorker.leadchangescnt_from_level[levname + "_p1LossSrv"][
                    cnt
                ] += 1

    def process_points_case(self, keys, det_game):
        points_cnt = len(det_game)
        self.points_dict[self.setinfo.generic_name()].hit(
            keys.combinations(), points_cnt
        )

    def reporting(self, sex):
        dirname = root_report_dir(sex, subdir="tie")
        self.reporting_dict_dir(sex, "tie", self.histo_dict)
        self.reporting_dict_dir(sex, "tie/after44h55h", self.after44h55h_dict)
        self.reporting_dict_dir(sex, "tie/after56hold", self.after56hold_dict)
        self.reporting_dict_dir(sex, "tie/after65brk", self.after65brk_dict)
        self.reporting_dict_dir(sex, "tie/beg6after56hold", self.beg6after56hold_dict)
        self.reporting_dict_dir(sex, "tie/beg6after65brk", self.beg6after65brk_dict)
        self.reporting_counts(sex, dirname, "minibreaks_count", self.minibreaks_dict)
        self.reporting_counts(sex, dirname, "leadchanges_count", self.leadchanges_dict)
        self.reporting_counts(sex, dirname, "points_count", self.points_dict)
        self.reporting_adveqadv_scat(sex)

    def reporting_adveqadv_scat(self, sex):
        dir_root = root_report_dir(sex, subdir="tie")
        dirname = os.path.join(dir_root, "adveqadv")

        for adv_n, scat_dict in self.adveqadv_scat_dict.items():
            filename = os.path.join(
                dirname, self.setinfo.typename + "_adv{}_scat.txt".format(adv_n)
            )
            dict_tools.dump(
                scat_dict,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=lambda i: i[1].ratio,
            )

            for key, wl in scat_dict.items():
                TiebreakWorker.adveqadv_scat_dict_static[adv_n][key] += wl

    @staticmethod
    def reporting_dict_dir(sex, subdir, dictionary):
        dirname = root_report_dir(sex, subdir=subdir)
        fu.ensure_folder(dirname)
        for keyname, histo in dictionary.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )

    @staticmethod
    def reporting_counts(sex, dirname, count_name, dictionary):
        fu.ensure_folder(dirname)
        for keyname, histo in dictionary.items():
            filename = os.path.join(dirname, keyname + "_" + count_name + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class Worker(object):
    def __init__(
        self, setinfo, scoreinfo, dirname="", innerdictmaker=st.create_winloss_histogram
    ):
        self.setinfo = setinfo
        self.scoreinfo = scoreinfo
        self.dirname = dirname

        # name(setinfo, scoreinfo) -> {keyi -> opener_winx}
        # name(setinfo, scoreinfo) reported in one file
        self.histo_dict = defaultdict(innerdictmaker)

    def process_match(self, match, keys, sex=None):
        score, det_score = match.score, match.detailed_score
        sets_count = score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            if (set_idx + 1) > sets_count or not self.setinfo.bestof_pred(
                score, set_idx
            ):
                continue
            item_at_score = self.scoreinfo.get_item(score, det_score, set_idx)
            if item_at_score is not None:
                if self.scoreinfo.value == (6, 6) and not item_at_score[1].tiebreak:
                    continue
                self.process_case(match, set_idx, keys, item_at_score)

    def report_dirname(self, sex):
        return root_report_dir(sex, self.dirname)

    def name(self):
        return self.setinfo.name() + "_" + self.scoreinfo.name()

    def reporting(self, sex):
        dirname = self.report_dirname(sex)
        fu.ensure_folder(dirname)
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class PairsWorker(object):
    def __init__(self, setinfo, dirname="", innerdictmaker=st.create_winloss_histogram):
        self.setinfo = setinfo
        self.dirname = dirname

        # name(setinfo, bef_item_server_adv, item_hold) -> {keyi -> opener_winx}
        # name(setinfo, bef_item_server_adv, item_hold) reported in one file
        self.histo_dict = defaultdict(innerdictmaker)

    def process_match(self, match, keys, sex=None):
        score, det_score = match.score, match.detailed_score
        sets_count = score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            if (
                (set_idx + 1) > sets_count
                or not self.setinfo.bestof_pred(score, set_idx)
                or not det_score.set_valid(setnum=set_idx + 1)
            ):
                continue
            set_items_it = det_score.set_items(setnum=set_idx + 1)
            for item, next_item in co.neighbor_pairs(set_items_it):
                item_key, item_dg = item
                if item_dg.left_opener:
                    bef_item_server_adv = sc.breaks_advantage(item_key[-1], co.LEFT)
                else:
                    bef_item_server_adv = sc.breaks_advantage(
                        co.reversed_tuple(item_key[-1]), co.LEFT
                    )  # reverse
                self.process_case(
                    match, set_idx, keys, bef_item_server_adv, item, next_item
                )

    def report_dirname(self, sex):
        return root_report_dir(sex, self.dirname)

    def name(self):
        return self.setinfo.name()

    def reporting(self, sex):
        dirname = self.report_dirname(sex)
        fu.ensure_folder(dirname)
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class PairsDecidedG1_040_G2_Worker(PairsWorker):
    def __init__(self):
        super(PairsDecidedG1_040_G2_Worker, self).__init__(
            setinfo=SetInfo("decided", "bo3", bo3_pred, (2,)),
            dirname="gamepairs_dec_g1_040",
        )
        self.histo_dict2 = defaultdict(st.create_winloss_histogram)

    def process_case(self, match, set_idx, keys, bef_item_server_adv, item, next_item):
        if set_idx not in (2, 4):
            raise co.TennisError("bad decided set_idx {}", set_idx)
        if item[0][-1] != (0, 0):
            return  # game1 must be first game in set
        det_game1 = item[1]
        if not det_game1.exist_text_score(("0", "40")):
            return
        game1_hold = det_game1.opener_wingame
        name = self.setinfo.name() + "_g1_040_ishold_perc"
        self.histo_dict[name].hit(keys.combinations(), game1_hold)

        if game1_hold:
            game2_hold = next_item[1].opener_wingame
            name = self.setinfo.name() + "_g1_040_hold_nextgwin_perc"
            self.histo_dict2[name].hit(keys.combinations(), not game2_hold)

    def reporting(self, sex):
        super(PairsDecidedG1_040_G2_Worker, self).reporting(sex)
        dirname = self.report_dirname(sex)
        for keyname, histo in self.histo_dict2.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )


class WinpointWorker(Worker):
    """Для анализа ситуаций: подающий в гейме при данном счете point_score
    выигрывает или нет розыгрыш очка
    """

    def __init__(self, setinfo, scoreinfo, point_score):
        super(WinpointWorker, self).__init__(setinfo, scoreinfo, dirname="winpoint")
        self.point_score = point_score

    def process_case(self, match, set_idx, keys, item_at_score):
        det_game = item_at_score[1]
        if point_score_error(det_game.error) or det_game.extra_points_error():
            return
        self.setinfo.make_pressunder_name(det_game.left_opener, set_idx, match.score)
        for point in det_game.points_text_score(self.point_score):
            self.histo_dict[self.name()].hit(keys.combinations(), point.win())

    def name(self):
        return (
            self.setinfo.name()
            + "_"
            + self.scoreinfo.name()
            + "_"
            + str(self.point_score[0])
            + "-"
            + str(self.point_score[1])
        )


class WinfirstpointWorker(object):
    """Для анализа ситуаций: подающий в гейме, первый розыгрыш.
    выигрывает или нет розыгрыш очка
    """

    def __init__(self):
        # (filed sc.Event.name/or 'noevent') -> {keyi->WL}
        self.histo_dict = defaultdict(st.create_winloss_histogram)

    def process_match(self, match, keys, sex=None):
        det_score = match.detailed_score
        for sckey, det_game in det_score.items():
            if not det_game.valid or len(det_game) == 0 or det_game.tiebreak:
                continue
            fst_point = next(iter(det_game))
            if fst_point:
                evt = sc.at_keyscore_event(sckey)
                evt_name = "noevent" if evt is None else evt.name
                self.histo_dict[evt_name].hit(keys.combinations(), fst_point.win())

    def reporting(self, sex):
        dirname = self.report_dirname(sex)
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )

    @staticmethod
    def report_dirname(sex):
        return os.path.join(root_report_dir(sex), "winpoint", "first")


class Winset1wayWorker(object):
    """1-й сет выигран 4 способами. каков рез-т второго сета для этих способов
    с точки зрения победителя первого сета
    """

    def __init__(self):
        # (filed sc.Winset1way.name/or error log) -> {keyi->WL}
        self.histo_dict = defaultdict(st.create_winloss_histogram)

    def process_match(self, match, keys, sex=None):
        sets_cnt = match.score.sets_count(full=True)
        if sets_cnt < 2:
            return
        det_score = match.detailed_score
        if not det_score.set_valid(setnum=1) or not det_score.set_valid(setnum=2):
            return
        leftwin1_winset1way = sc.winsetway(det_score, setnum=1)
        if leftwin1_winset1way is None:
            log.error("bad winset1way in {} in dscore {}".format(match, det_score))
            return
        leftwin1, winset1way = leftwin1_winset1way
        snd_set = match.score[1]
        leftwin2 = snd_set[0] > snd_set[1]
        winner1_win2 = leftwin1 is leftwin2
        self.histo_dict[winset1way.name].hit(keys.combinations(), winner1_win2)

    def reporting(self, sex):
        dirname = self.report_dirname(sex)
        fu.ensure_folder(dirname)
        for keyname, histo in self.histo_dict.items():
            filename = os.path.join(dirname, keyname + ".txt")
            dict_tools.dump(
                histo,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )

    def report_dirname(self, sex):
        return os.path.join(root_report_dir(sex), "winset1way")


class GameWorker(Worker):
    def __init__(
        self,
        setinfo,
        scoreinfo,
        dirname,
        game_fun,
        innerdictmaker=st.create_winloss_histogram,
    ):
        super(GameWorker, self).__init__(
            setinfo, scoreinfo, dirname=dirname, innerdictmaker=innerdictmaker
        )
        self.game_fun = game_fun

    def process_case(self, match, set_idx, keys, item_at_score):
        det_game = item_at_score[1]
        self.setinfo.make_pressunder_name(det_game.left_opener, set_idx, match.score)
        self.histo_dict[self.name()].hit(keys.combinations(), self.game_fun(det_game))


class WingameWorker(GameWorker):
    def __init__(self, setinfo, scoreinfo, innerdictmaker=st.create_winloss_histogram):
        super(WingameWorker, self).__init__(
            setinfo,
            scoreinfo,
            dirname="wingame",
            game_fun=lambda dg: dg.opener_wingame,
            innerdictmaker=innerdictmaker,
        )


class WinsetWorker(Worker):
    def __init__(self, setinfo, scoreinfo, innerdictmaker=st.create_winloss_histogram):
        super(WinsetWorker, self).__init__(
            setinfo, scoreinfo, dirname="winset", innerdictmaker=innerdictmaker
        )

    def process_case(self, match, set_idx, keys, item_at_score):
        self.setinfo.make_pressunder_name(
            item_at_score[1].left_opener, set_idx, match.score
        )
        left_winset = match.score[set_idx][0] >= match.score[set_idx][1]
        opener_winset = left_winset is item_at_score[1].left_opener
        self.histo_dict[self.name()].hit(keys.combinations(), opener_winset)


class LeadratioWorker(Worker):
    def __init__(self, setinfo, scoreinfo):
        super(LeadratioWorker, self).__init__(
            setinfo,
            scoreinfo,
            dirname="leadratio",
            innerdictmaker=st.create_summator_histogram,
        )

    @staticmethod
    def make_workers(sex):
        return [
            LeadratioWorker(si, ci)
            for si in setinfos(sex, predicate=lambda s: s.bestof_name == "bo3")
            for ci in itertools.chain(scoreinfos(co.EQ), scoreinfos(co.LT))
        ]

    def process_case(self, match, set_idx, keys, item_at_score):
        det_game = item_at_score[1]
        if (
            (self.scoreinfo.value == (6, 6) and not det_game.tiebreak)
            or point_score_error(det_game.error)
            or det_game.extra_points_error()
        ):
            return

        self.setinfo.make_pressunder_name(det_game.left_opener, set_idx, match.score)
        leadratio_wl = dsm.lead_ratio(det_game)
        if leadratio_wl:
            self.histo_dict[self.name()].hit(keys.combinations(), leadratio_wl.ratio)


class WingameWorkerOtherCondition(WingameWorker):
    def __init__(self, setinfo, scoreinfo, other_condition):
        super(WingameWorkerOtherCondition, self).__init__(setinfo, scoreinfo)
        self.other_condition = other_condition

    def process_case(self, match, set_idx, keys, item_at_score):
        det_game = item_at_score[1]
        self.setinfo.make_pressunder_name(det_game.left_opener, set_idx, match.score)
        if self.other_condition.exist_item(
            match.detailed_score, set_idx + 1, item_at_score[1].left_opener
        ):
            self.histo_dict[self.name()].hit(
                keys.combinations(), det_game.opener_wingame
            )

    def name(self):
        return (
            self.other_condition.name
            + "_"
            + self.setinfo.name()
            + "_"
            + self.scoreinfo.name()
        )


class WinsetWorkerOtherCondition(WinsetWorker):
    def __init__(self, setinfo, scoreinfo, other_condition):
        super(WinsetWorkerOtherCondition, self).__init__(setinfo, scoreinfo)
        self.other_condition = other_condition

    def process_case(self, match, set_idx, keys, item_at_score):
        self.setinfo.make_pressunder_name(
            item_at_score[1].left_opener, set_idx, match.score
        )
        left_winset = match.score[set_idx][0] >= match.score[set_idx][1]
        opener_winset = left_winset is item_at_score[1].left_opener
        if self.other_condition.exist_item(
            match.detailed_score, set_idx + 1, item_at_score[1].left_opener
        ):
            self.histo_dict[self.name()].hit(keys.combinations(), opener_winset)

    def name(self):
        return (
            self.other_condition.name
            + "_"
            + self.setinfo.name()
            + "_"
            + self.scoreinfo.name()
        )


class WinsetedgeWorkerOtherNegCondition(Worker):
    """успешность реализации 1-го, 2-го, 3-го сетбола подащим на сет.
    Условие: первый сетбол возникает после счета ровно (т.е. нет отрыва)"""

    _dict = defaultdict(st.create_winloss_histogram)  # group by setsinfo
    # here extern key in ('first', 'second', 'third') only:
    _begnums_dict = defaultdict(st.create_winloss_histogram)

    @classmethod
    def _reporting_dict(cls, sex, dictionary):
        dirname = os.path.join(root_report_dir(sex), "winsetedge")
        assert os.path.isdir(dirname), "not found dir " + dirname
        for key, dct in dictionary.items():
            if not dct:
                continue
            filename = os.path.join(dirname, str(key) + ".txt")
            dict_tools.dump(
                dct,
                filename,
                keyfun=str,
                valuefun=str,
                sortfun=itemgetter(0),
                sortreverse=True,
                alignvalues=True,
            )

    @classmethod
    def reporting_classdata(cls, sex):
        cls._reporting_dict(sex, cls._dict)
        cls._reporting_dict(sex, cls._begnums_dict)

    @classmethod
    def clear_classdata(cls):
        cls._dict = defaultdict(st.create_winloss_histogram)
        cls._begnums_dict = defaultdict(st.create_winloss_histogram)

    def __init__(self, setinfo, scoreinfo, other_condition):
        super(WinsetedgeWorkerOtherNegCondition, self).__init__(
            setinfo, scoreinfo, dirname="winsetedge"
        )
        self.other_condition = other_condition  # process if not this condition

    def process_case(self, match, set_idx, keys, item_at_score):
        def process_game_part(points, partnum):
            eqidx, p_at_eq = co.find_indexed_first(
                points, lambda p: p.equal_score(before=True)
            )
            if eqidx < 0:
                return None  # not found
            if p_at_eq.win() and (eqidx + 1) < len(points):
                p_at_adv = points[eqidx + 1]
                self.histo_dict[self.name() + "_all"].hit(
                    keys.combinations(), p_at_adv.win_game()
                )
                self._dict[self.scoreinfo.name() + "_all"].hit(
                    keys.combinations(), p_at_adv.win_game()
                )
                if partnum == 1:
                    self.histo_dict[self.name() + "_first"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                    self._dict[self.scoreinfo.name() + "_first"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                    self._begnums_dict["first"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                elif partnum == 2:
                    self.histo_dict[self.name() + "_second"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                    self._dict[self.scoreinfo.name() + "_second"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                    self._begnums_dict["second"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                elif partnum == 3:
                    self._begnums_dict["third"].hit(
                        keys.combinations(), p_at_adv.win_game()
                    )
                if p_at_adv.win_game():
                    return None  # end of game
            return eqidx + 2  # return possible next eqidx for work

        self.setinfo.make_pressunder_name(
            item_at_score[1].left_opener, set_idx, match.score
        )
        if self.other_condition.exist_item(
            match.detailed_score, set_idx + 1, item_at_score[1].left_opener
        ):
            return  # opener has been served for set already before this game
        if (
            point_score_error(item_at_score[1].error)
            or item_at_score[1].extra_points_error()
        ):
            return
        points = [pnt for pnt in item_at_score[1]]
        for partidx in range(40):
            next_idx = process_game_part(points, partidx + 1)
            if next_idx is None:
                break
            points = copy.copy(points[next_idx:])

    def name(self):
        return (
            self.other_condition.name
            + "_"
            + self.setinfo.name()
            + "_"
            + self.scoreinfo.name()
        )


class WorkerAttribute(co.Attribute):
    def __init__(self, name, infilename="", predicate=None):
        super(WorkerAttribute, self).__init__(name, predicate)
        self.__infilename = infilename

    def foldable(self):
        return self.predicate is None

    def apply_predicate(self, worker):
        """проверим условие над атрибутом об-та worker"""
        if self.foldable():
            return True
        return super(WorkerAttribute, self).apply_predicate(worker)

    @property
    def infilename(self):
        if self.foldable():
            return ""
        return self.__infilename


def make_winpoint_workers(sex):
    def get_scoreinfos():
        return (
            ScoreInfo((0, 0)),
            ScoreInfo((1, 1)),
            ScoreInfo((2, 2)),
            ScoreInfo((3, 3)),
            ScoreInfo((4, 4)),
            ScoreInfo((5, 5)),
            ScorePursueInfo((0, 1)),
            ScorePursueInfo((1, 2)),
            ScorePursueInfo((2, 3)),
            ScorePursueInfo((3, 4)),
            ScorePursueInfo((4, 5)),
            ScorePursueInfo((5, 6)),
        )

    def get_setinfos():
        if sex == "wta":
            return (
                SetInfo("open", "bo3", bo3_pred, (0,)),
                SetInfo("decided", "bo3", bo3_pred, (2,)),
            )
        elif sex == "atp":
            return (
                SetInfo("open", "boA", bo35_pred, (0,)),
                SetInfo("decided", "boA", bo35dec_pred, (2, 4)),
            )

    return [
        WinpointWorker(si, ci, ps)
        for si in get_setinfos()
        for ci in get_scoreinfos()
        for ps in [
            ("0", "0"),
            ("0", "15"),
            ("15", "0"),
            ("15", "15"),
            ("30", "0"),
            ("0", "30"),
            ("30", "30"),
            ("30", "15"),
            ("15", "30"),
            ("40", "15"),
            ("15", "40"),
            ("40", "30"),
            ("30", "40"),
            ("40", "40"),
            ("40", "A"),
            ("A", "40"),
        ]
        # if not (si.typename == 'open' and ci.value in ((0, 0), (0, 1)))
    ]


def winpoitworker_attrs_variants():
    attr1fold = WorkerAttribute(name="setinfo")
    attr1seq = [
        WorkerAttribute(
            name="setinfo",
            infilename="setopen",
            predicate=lambda a: "open" == a.typename,
        ),
        WorkerAttribute(
            name="setinfo",
            infilename="setdecided",
            predicate=lambda a: "decided" == a.typename,
        ),
    ]

    attr2fold = WorkerAttribute(name="scoreinfo")
    attr2seq = [
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset00",
            predicate=lambda a: a.value == (0, 0),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset11",
            predicate=lambda a: a.value == (1, 1),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset22",
            predicate=lambda a: a.value == (2, 2),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset33",
            predicate=lambda a: a.value == (3, 3),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset44",
            predicate=lambda a: a.value == (4, 4),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset55",
            predicate=lambda a: a.value == (5, 5),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset01",
            predicate=lambda a: a.value == (0, 1),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset12",
            predicate=lambda a: a.value == (1, 2),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset23",
            predicate=lambda a: a.value == (2, 3),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset34",
            predicate=lambda a: a.value == (3, 4),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset45",
            predicate=lambda a: a.value == (4, 5),
        ),
        WorkerAttribute(
            name="scoreinfo",
            infilename="inset56",
            predicate=lambda a: a.value == (5, 6),
        ),
    ]

    attr3seq = [
        WorkerAttribute(
            name="point_score",
            infilename="ingame0-0",
            predicate=lambda a: a == ("0", "0"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame15-0",
            predicate=lambda a: a == ("15", "0"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame0-15",
            predicate=lambda a: a == ("0", "15"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame15-15",
            predicate=lambda a: a == ("15", "15"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame15-30",
            predicate=lambda a: a == ("15", "30"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame30-15",
            predicate=lambda a: a == ("30", "15"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame30-0",
            predicate=lambda a: a == ("30", "0"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame0-30",
            predicate=lambda a: a == ("0", "30"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame30-30",
            predicate=lambda a: a == ("30", "30"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame40-15",
            predicate=lambda a: a == ("40", "15"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame15-40",
            predicate=lambda a: a == ("15", "40"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame40-30",
            predicate=lambda a: a == ("40", "30"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame30-40",
            predicate=lambda a: a == ("30", "40"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingameA-40",
            predicate=lambda a: a == ("A", "40"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame40-A",
            predicate=lambda a: a == ("40", "A"),
        ),
        WorkerAttribute(
            name="point_score",
            infilename="ingame40-40",
            predicate=lambda a: a == ("40", "40"),
        ),
    ]
    a1fold_vars = [(attr1fold, a2, a3) for a2 in attr2seq for a3 in attr3seq]
    a2fold_vars = [(a1, attr2fold, a3) for a1 in attr1seq for a3 in attr3seq]
    a1fold_a2fold_vars = [(attr1fold, attr2fold, a3) for a3 in attr3seq]
    return a1fold_vars + a2fold_vars + a1fold_a2fold_vars


def foldable_winpoitworkers_report(sex, workers):
    for worker_attrs in winpoitworker_attrs_variants():
        foldable_variant_report(sex, worker_attrs, workers)


def foldable_variant_report(sex, worker_attrs, workers):
    def worker_admit(worker):
        result = True
        for attr in worker_attrs:
            if not attr.apply_predicate(worker):
                result = False
                break
        return result

    def summarize():
        result = st.create_winloss_histogram()
        for worker in workers:
            if worker_admit(worker):
                for cont in worker.histo_dict.values():
                    result = result + cont
        return result

    def get_filename():
        classnames = set([w.__class__.__name__ for w in workers])
        assert len(classnames) == 1, "workers in foldable_variant_report too many types"
        assert all(
            [isinstance(w, Worker) for w in workers]
        ), "workers in foldable_variant_report bad type"
        dirname = workers[0].report_dirname(sex)
        shortname = "_".join([a.infilename for a in worker_attrs])
        return os.path.join(dirname, shortname) + ".txt"

    data = summarize()
    if data:
        filename = get_filename()
        dict_tools.dump(
            data,
            filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            sortreverse=True,
            alignvalues=True,
        )


class OtherSetItemCondition(object):
    """помошник для определения ситуаций вроде: подает при (5, 5) правый,
    но ДОП. УСЛОВИЕ: он уже подавал на этот сет при (3, 5)
    т.е. для доп. усл. создадим obj = OtherSetItemCondition((5, 3), 'fsrv53srv55')
    и вызовем obj.exist_item(det_score, setnum, left_opener=False)"""

    def __init__(self, other_gscore, name):
        self.other_gscore = other_gscore  # simple tuple. min always at right.
        self.name = name

    def exist_item(self, det_score, setnum, left_opener):
        """есть ли в сете setnum эл-т с открывателем left_opener
        и счетом self.other_gscore (который инверт-ся если left_opener==False)"""
        need_reverse = not left_opener
        seek_sc = (
            tuple(reversed(self.other_gscore)) if need_reverse else self.other_gscore
        )
        it = det_score.item(
            predicate=(
                lambda k, v: len(k) == setnum
                and k[-1] == seek_sc
                and v.left_opener is left_opener
            )
        )
        return it is not None


def make_winset_othercond_workers(sex):
    sci_cond_1 = (ScorePursueInfo((5, 6)), OtherSetItemCondition((5, 4), "fsrv54srv56"))
    sci_cond_2 = (ScoreInfo((5, 5)), OtherSetItemCondition((5, 3), "fsrv53srv55"))
    sci_cond_3 = (ScorePursueInfo((5, 6)), OtherSetItemCondition((5, 2), "fsrv52srv56"))
    sci_cond_4 = (ScoreInfo((5, 5)), OtherSetItemCondition((5, 1), "fsrv51srv55"))
    return [
        WinsetWorkerOtherCondition(si, ci, cond)
        for si in setinfos(sex)
        for ci, cond in (sci_cond_1, sci_cond_2, sci_cond_3, sci_cond_4)
    ]


def make_wingame_othercond_workers(sex):
    sci_cond_1 = (ScorePursueInfo((5, 6)), OtherSetItemCondition((5, 4), "fsrv54srv56"))
    sci_cond_2 = (ScoreInfo((5, 5)), OtherSetItemCondition((5, 3), "fsrv53srv55"))
    sci_cond_3 = (ScorePursueInfo((5, 6)), OtherSetItemCondition((5, 2), "fsrv52srv56"))
    sci_cond_4 = (ScoreInfo((5, 5)), OtherSetItemCondition((5, 1), "fsrv51srv55"))
    return [
        WingameWorkerOtherCondition(si, ci, cond)
        for si in setinfos(sex)
        for ci, cond in (sci_cond_1, sci_cond_2, sci_cond_3, sci_cond_4)
    ]


def make_winsetedge_othercond_workers(sex):
    sci_cond_1 = (ScoreBreakupInfo((6, 5)), OtherSetItemCondition((5, 4), "edge65"))
    sci_cond_2 = (ScoreBreakupInfo((5, 4)), OtherSetItemCondition((5, 2), "edge54"))
    sci_cond_3 = (ScoreBreakupInfo((5, 3)), OtherSetItemCondition((5, 1), "edge53"))
    return [
        WinsetedgeWorkerOtherNegCondition(si, ci, cond)
        for si in setinfos(sex)
        for ci, cond in (sci_cond_1, sci_cond_2, sci_cond_3)
    ]


class MissProcessor(Processor):
    def process(self, tour):
        if tour.level.junior:
            return
        for rnd, matches in tour.matches_from_rnd.items():
            if rnd == "Pre-q" or (
                tour.level in ("future", "chal") and rnd.qualification()
            ):
                continue
            keys = co.Keys()  # co.Keys.soft_main_maker(tour, rnd, with_surface=False)
            for match in matches:
                if match.score.valid() and not match.paired():
                    if find_detailed_score2(dbdet.records, tour, rnd, match):
                        miss_dict = miss_points.detailed_score_miss_points(
                            match.detailed_score, match.score.best_of_five()
                        )
                        for worker in self.workers:
                            worker.process_match(match, keys, miss_dict)


def make_miss_processor(sex):
    def miss_scoreinfos():
        return (
            ScoreInfoFlipable((5, 0)),
            ScoreInfoFlipable((5, 1)),
            ScoreInfoFlipable((5, 2)),
            ScoreInfoFlipable((5, 3)),
            ScoreInfoFlipable((5, 4)),
            ScoreInfoFlipable((6, 5)),
            ScoreInfo((6, 6)),
        )

    workers = [
        MissScopeWorker(co.SET, si, ci)
        for si in setinfos(sex)
        for ci in miss_scoreinfos()
    ]
    return MissProcessor(sex, workers)


class MissWorker(Worker):
    def process_match(self, match, keys, miss_dict):
        score, det_score = match.score, match.detailed_score
        sets_count = score.sets_count(full=True)
        for set_idx in self.setinfo.indicies:
            if (set_idx + 1) > sets_count or not self.setinfo.bestof_pred(
                score, set_idx
            ):
                continue
            item_at_score = self.scoreinfo.get_item(score, det_score, set_idx)
            if item_at_score is not None:
                if self.scoreinfo.value == (6, 6) and not item_at_score[1].tiebreak:
                    continue
                self.process_case(match, set_idx, keys, item_at_score, miss_dict)


class MissScopeWorker(MissWorker):
    def __init__(self, scope, setinfo, scoreinfo):
        super(MissScopeWorker, self).__init__(setinfo, scoreinfo)
        self.scope = scope

    def process_case(self, match, set_idx, keys, item_at_score, miss_dict):
        def miss_key(side):
            cnt1 = miss_points.failed_count_before(
                side, miss_dict, item_at_score[0], self.scope, last_set_only=True
            )
            side2 = co.RIGHT if side == co.LEFT else co.LEFT
            cnt2 = miss_points.failed_count_before(
                side2, miss_dict, item_at_score[0], self.scope, last_set_only=True
            )
            assert cnt1 >= 0 and cnt2 >= 0, "invalid miss cnt {} {}".format(cnt1, cnt2)
            if cnt1 == 0 and cnt2 == 0:
                symb1, symb2 = "Z", "Z"
            elif cnt1 == cnt2:
                symb1, symb2 = "E", "E"
            elif cnt1 < cnt2:
                symb1, symb2 = "L", "G"
            else:
                symb1, symb2 = "G", "L"
            return "{},{}".format(symb1, symb2)

        score = match.score
        self.setinfo.make_pressunder_name(item_at_score[1].left_opener, set_idx, score)
        if item_at_score[1].tiebreak:
            check_side = co.LEFT if item_at_score[1].left_opener else co.RIGHT
            checker_win = item_at_score[1].opener_wingame
        else:
            if self.scoreinfo.left_lead() != item_at_score[1].left_opener:
                return
            if self.scope == co.SET:
                left_win = score[set_idx][0] >= score[set_idx][1]
            elif self.scope == co.MATCH:
                side = miss_points.matchball_chance_side(
                    item_at_score[0], item_at_score[1].tiebreak, score.best_of_five()
                )
                if side is None:
                    return  # счет не допускает матчболы
                left_win = score[-1][0] >= score[-1][1]

            if self.scoreinfo.left_lead():
                check_side = co.LEFT
                checker_win = left_win
            else:
                check_side = co.RIGHT
                checker_win = not left_win
        ext_keys = co.Keys(keys, miss=miss_key(check_side))
        self.histo_dict[self.name()].hit(ext_keys.combinations(), checker_win)

    def name(self):
        return "miss_{0}_win_{0}_{1}_{2}".format(
            str(self.scope), self.setinfo.name(), self.scoreinfo.name()
        )


def setinfos(sex, predicate=lambda si: True):
    def results():
        if sex == "wta":
            return (
                SetInfo("open", "bo3", bo3_pred, (0,)),
                SetInfo("decided", "bo3", bo3_pred, (2,)),
                SetInfo("pressunder", "bo3", bo3_pred, (1,)),
            )
        else:
            return (
                SetInfo("open", "bo3", bo3_pred, (0,)),
                SetInfo("open", "bo5", bo5_pred, (0,)),
                SetInfo("open2", "bo5", bo5_pred, (2,)),
                SetInfo("open", "boA", bo35_pred, (0,)),
                SetInfo("decided", "bo3", bo3_pred, (2,)),
                SetInfo("decided", "bo5", bo5_pred, (4,)),
                SetInfo("decided", "boA", bo35dec_pred, (2, 4)),
                SetInfo("pressunder", "bo3", bo3_pred, (1,)),
                SetInfo("pressunder", "bo5", bo5_pred, (3,)),
                SetInfo("pressunder", "boA", bo35_pred, (1, 3)),
            )

    return (si for si in results() if predicate(si))


def scoreinfos(compare):
    if compare == co.LT:
        return (
            ScorePursueInfo((0, 1)),
            ScorePursueInfo((1, 2)),
            ScorePursueInfo((2, 3)),
            ScorePursueInfo((3, 4)),
            ScorePursueInfo((4, 5)),
            ScorePursueInfo((5, 6)),
        )
    elif compare == co.EQ:
        return (
            ScoreInfo((0, 0)),
            ScoreInfo((1, 1)),
            ScoreInfo((2, 2)),
            ScoreInfo((3, 3)),
            ScoreInfo((4, 4)),
            ScoreInfo((5, 5)),
            ScoreInfo((6, 6)),
        )
    elif compare == co.GT:
        return (
            ScoreBreakupInfo((1, 0)),
            ScoreBreakupInfo((2, 1)),
            ScoreBreakupInfo((3, 2)),
            ScoreBreakupInfo((4, 3)),
            ScoreBreakupInfo((5, 4)),
            ScoreBreakupInfo((6, 5)),
        )


__prev_scores_fromcur = {
    (5, 0): (),
    (0, 5): (),
    (5, 1): (),
    (1, 5): (),
    (5, 2): ((5, 0),),
    (2, 5): ((0, 5),),
    (5, 3): ((5, 1),),
    (3, 5): ((1, 5),),
    (5, 4): ((5, 2), (5, 0)),
    (4, 5): ((2, 5), (0, 5)),
    (6, 5): ((5, 4), (5, 2), (5, 0)),
    (5, 6): ((4, 5), (2, 5), (0, 5)),
}


def prev_serveon_try_count(score_key, set_idx, det_score):
    """число предыдущих попыток подать на сет с индексом set_idx.
    предположено что в score_key кто-то подает на сет set_idx.
    """
    beg_sc = ()
    for sidx in range(set_idx):
        beg_sc += (score_key[sidx],)

    try_count = 0
    for prev_sc in __prev_scores_fromcur[score_key[set_idx]]:
        key = beg_sc + (prev_sc,)
        if key in det_score:
            try_count += 1
    return try_count


def make_processor(sex):
    workers = []
    workers += [
        WingameWorker(si, ci)
        for si in setinfos(sex)
        for ci in itertools.chain(
            scoreinfos(co.EQ), scoreinfos(co.LT), scoreinfos(co.GT)
        )
    ]

    workers += [
        GameWorker(si, ci, "break_points_exist", dsm.break_points_exist)
        for si in setinfos(sex)
        for ci in itertools.chain(
            scoreinfos(co.EQ), scoreinfos(co.LT), scoreinfos(co.GT)
        )
        if ci != ScoreInfo((6, 6))
    ]

    workers += [
        GameWorker(
            si, ci, "break_points_or_trail2_exist", dsm.break_points_or_trail2_exist
        )
        for si in setinfos(sex)
        for ci in itertools.chain(
            scoreinfos(co.EQ), scoreinfos(co.LT), scoreinfos(co.GT)
        )
        if ci != ScoreInfo((6, 6))
    ]

    workers += [
        GameWorker(
            si, ci, "break_points_or_trail1_exist", dsm.break_points_or_trail1_exist
        )
        for si in setinfos(sex)
        for ci in itertools.chain(
            scoreinfos(co.EQ), scoreinfos(co.LT), scoreinfos(co.GT)
        )
        if ci != ScoreInfo((6, 6))
    ]

    workers += [
        WinsetWorker(si, ci)
        for si in setinfos(sex)
        for ci in itertools.chain(scoreinfos(co.EQ), scoreinfos(co.LT))
    ]

    workers += [
        WinsetWorker(si, ci)
        for si in setinfos(sex, predicate=lambda s: s.typename in ("open", "decided"))
        for ci in scoreinfos(co.GT)
    ]

    workers += make_winset_othercond_workers(sex)
    workers += make_wingame_othercond_workers(sex)
    return Processor(sex, workers)


def make_breaksmoves_workers(sex):
    return [
        BreaksMovesWorker(si)
        for si in setinfos(sex, predicate=(lambda si: si.typename != "pressunder"))
    ]


def make_breakscount_workers(sex):
    return [
        BreaksAverageCountWorker(si)
        for si in setinfos(
            sex,
            predicate=(
                lambda si: si.bestof_name != "boA" and si.typename != "pressunder"
            ),
        )
    ]


def make_breaks_seqcount_workers(sex):
    return [
        BreaksSeqCountWorker(si)
        for si in setinfos(
            sex,
            predicate=(
                lambda si: si.bestof_name != "boA" and si.typename != "pressunder"
            ),
        )
    ]


class Integrator(object):
    def __init__(self, workers, filename):
        self.workers = workers
        self.filename = filename

    def reporting(self):
        wl_histo = self.__summarize()
        dict_tools.dump(
            wl_histo,
            self.filename,
            keyfun=str,
            valuefun=str,
            sortfun=itemgetter(0),
            sortreverse=True,
            alignvalues=True,
        )

    def __summarize(self):
        result = st.create_winloss_histogram()
        for worker in self.workers:
            for cont in worker.histo_dict.values():
                result = result + cont
        return result


def make_integrators(sex, workers):
    def select_workers(
        wrk_cls, setinfo_pred=lambda si: True, scoreinfo_pred=lambda si: True
    ):
        return [
            worker
            for worker in workers
            if (
                isinstance(worker, wrk_cls)
                and setinfo_pred(worker.setinfo)
                and scoreinfo_pred(worker.scoreinfo)
            )
        ]

    def setinfo_pred(si):
        return "boA" not in si.bestof_name

    result = []
    dirname = root_report_dir(sex)
    for wrk_cls in (WingameWorker, WinsetWorker):
        for score_value in (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ):
            fn = "{}_{}-{}.txt".format(wrk_cls.__name__, score_value[0], score_value[1])
            result.append(
                Integrator(
                    select_workers(
                        wrk_cls, setinfo_pred, lambda si, v=score_value: si.value == v
                    ),
                    filename=os.path.join(dirname, fn),
                )
            )
    return result


def process_sex(sex):
    min_date = tt.past_monday_date(dbdet.query_min_date())
    max_date = tt.past_monday_date(dbdet.query_max_date()) + datetime.timedelta(days=7)
    #   decided_set.initialize_results(sex=sex,
    #                                  min_date=min_date - datetime.timedelta(days=365 * 3),
    #                                  max_date=max_date)
    #     proc = make_processor(sex)
    #     winpoint_workers = make_winpoint_workers(sex)
    make_stat.process(
        [
            # Processor(sex, [DecidedSetDataWorker()]),  # need dec_set.init res, with_bets
            # Processor(sex, LeadratioWorker.make_workers(sex)),
            Processor(
                sex,
                TiebreakWorker.make_workers(sex),
                max_rating=None,
                max_rating_dif=None,
            ),
            # Processor(sex, SrvTriesWorker.make_workers(sex), max_rating=300),
            # Processor(sex, BidirectTrendsWorker.make_workers(sex), max_rating=350),
            Processor(
                sex, make_breaksmoves_workers(sex), max_rating=400, max_rating_dif=250
            ),
            Processor(
                sex, make_breakscount_workers(sex), max_rating=400, max_rating_dif=250
            ),
            Processor(
                sex,
                make_breaks_seqcount_workers(sex),
                max_rating=400,
                max_rating_dif=250,
            ),
            # Processor(sex, make_winsetedge_othercond_workers(sex)),
            # Processor(sex, winpoint_workers, max_rating=300),
            # Processor(sex, [WinfirstpointWorker()], max_rating=300),
            # Processor(sex, [Winset1wayWorker()]),
            # Processor(sex, [PairsDecidedG1_040_G2_Worker()]),
            # proc,
        ],
        sex,
        min_date=min_date,
        max_date=max_date,
        with_bets=False,
    )

    # WinsetedgeWorkerOtherNegCondition.reporting_classdata(sex)
    # WinsetedgeWorkerOtherNegCondition.clear_classdata()

    TiebreakWorker.reporting_leadchanges(sex)
    TiebreakWorker.reporting_eqdifeq(sex)
    TiebreakWorker.reporting_adveqadv_static(sex)
    TiebreakWorker.reporting_tie_edge(sex)
    TiebreakWorker.reporting_perf_dict(sex)
    TiebreakWorker.clear_static()

    # foldable_winpoitworkers_report(sex, winpoint_workers)

    # for integrator in make_integrators(sex, proc.workers):
    #     integrator.reporting()


def process_miss_sex(sex):
    make_stat.process(
        [make_miss_processor(sex)],
        sex,
        min_date=tt.past_monday_date(dbdet.query_min_date()),
        max_date=tt.past_monday_date(dbdet.query_max_date()),
    )


# не забудь сохранить текущие р-ты в другое место на время отладки
def debug_single_match(sex):
    make_miss_processor(sex).debug_single_match()


def do_stat(sex, miss):
    modename = "{}_{}".format(sex, "miss" if miss else "usial")
    log.initialize(
        co.logname(__file__, instance=modename),
        file_level="debug",
        console_level="info",
    )
    try:
        log.info(__file__ + " started as " + modename)
        start_datetime = datetime.datetime.now()
        dba.open_connect()
        oncourt_players.initialize(yearsnum=11)
        ratings_std.initialize(min_date=datetime.date(2009, 12, 25))

        dbdet.query_matches()
        if miss:
            process_miss_sex(sex)
        else:
            process_sex(sex)

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
    parser.add_argument("--sex", choices=["wta", "atp"], required=True)
    parser.add_argument("--miss", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        dbdet = open_db(sex=args.sex)
        sys.exit(do_stat(sex=args.sex, miss=args.miss))

