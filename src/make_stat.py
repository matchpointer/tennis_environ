import os
import sys
from collections import defaultdict
import datetime
from operator import sub
import argparse

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
import qual_seeds
import oncourt_players
import matchstat


class WinFavoriteProcessor:
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


class TourAverageSetProcessor:
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
        log.info(__file__ + " started sex: " + str(args.sex))
        start_datetime = datetime.datetime.now()
        dba.open_connect()
        ratings_std.initialize(sex=args.sex)
        qual_seeds.initialize()
        oncourt_players.initialize(yearsnum=18)

        matchstat_process(sex=args.sex)

        process(
            (
                WinFavoriteProcessor(args.sex),
                TourAverageSetProcessor(args.sex),
            ),
            args.sex
        )

        dba.close_connect()
        log.info(
            "{} finished within {}".format(
                __file__, str(datetime.datetime.now() - start_datetime)
            )
        )
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exc_info=True)
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    sys.exit(do_stat())
