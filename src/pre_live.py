# -*- coding=utf-8 -*-
r"""
module gives standalone process (prepare data) for matches which soon be live.
"""
import sys
import signal
from typing import Optional
from pages.create_page import create_page
from pages.base import WebPage
from wdriver import stop_web_driver


wpage: Optional[WebPage] = None


def signal_handler(signal, frame):
    print("\nprogram exiting gracefully after signaled")
    global wpage
    if wpage is not None:
        print("\nwpage stoping...")
        stop_web_driver(wpage.get_web_driver())
        wpage = None
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

import time
import datetime
from collections import defaultdict
import argparse

import stopwatch

from loguru import logger as log
import common as co
import feature
import tennis_time as tt
import bet_coefs
from oncourt import dbcon, automate, extplayers
import matchstat
import ratings
import decided_set
import weeked_tours
from live import (
    MatchStatus,
    LiveMatch,
    skip_levels_work,
)
from score_company import get_company
import pre_live_dir
from stat_cont import WinLoss
from report_line import SizedValue
import atfirst_after
import advantage_tie_ratio

try:
    pass
except ImportError:
    automate = None


_DEBUG_MATCH_NAME = ""  # "Coppejans K. - Gerch L."


# (sex, tour_name) -> (tour_id, level, surface)
_tourinfo_cache = defaultdict(lambda: None)


class Script(object):
    def __init__(self, timer):
        self.timer = timer

    def work(self):
        raise NotImplementedError("work() must be implemented")


class OncourtUpdateScript(Script):
    def __init__(self, wait_timeout):
        super(OncourtUpdateScript, self).__init__(stopwatch.OverTimer(wait_timeout))

    def work(self):
        result = automate.oncourt_update_connect()
        if result:
            time.sleep(7)
            initialize()
            log.info("OncourtUpdateScript work done")
        else:
            log.error("OncourtUpdateScript failed")


class MatchDataScript(Script):
    important_features = (
        'fst_plr_tour_adapt',
        'snd_plr_tour_adapt',
        'fst_fatigue',
        'snd_fatigue',
        'decset_ratio_dif',
        'fst_win_coef',
        'snd_win_coef',
    )

    def __init__(self, wait_timeout, prelive_threshold):
        """ prelive_threshold на сколько секунд от настоящего момента
            смотрим в будущее для отбора матчей, которые скоро начнуться.
            wait_timeout на сколько засыпаем после очередной работы """
        super(MatchDataScript, self).__init__(stopwatch.OverTimer(wait_timeout))
        self.prelive_threshold = prelive_threshold  # in seconds
        self.work_count = 0

    def prelive_thresholded(self, match):
        if match.time:
            dt_now = datetime.datetime.now()
            dt_match = match.datetime
            if dt_match > dt_now:
                dif = (dt_match - dt_now).seconds
                return dif <= self.prelive_threshold
            elif self.work_count == 0:
                return True  # at start we want work possibly start-suspended matches

    def is_match_towork(self, match):
        if match.name == _DEBUG_MATCH_NAME:
            return True
        pre_live_name = match.pre_live_name()
        pre_live_dir.prepare_dir("matches")
        if pre_live_dir.is_data(pre_live_name):
            return False  # match is already prepared
        return self.prelive_thresholded(match)

    def is_event_towork(self, event):
        for match in event.matches:
            if self.is_match_towork(match):
                return True
        return False

    @staticmethod
    def prepare_dict(match):
        """ for serializing: return dct with attrs of match (features and some others)
            for serialize in json file """

        def prep_advleft_tie_ratio(name: str):
            """ input name sample: 'sd_tie_ratio' output key sample: 'advleft_' + name """
            fst_name = f"fst_{name}"
            snd_name = f"snd_{name}"
            fst_feat = co.find_first(match.features, lambda f: f.name == fst_name)
            snd_feat = co.find_first(match.features, lambda f: f.name == snd_name)
            if (
                fst_feat is not None
                and snd_feat is not None
                and isinstance(fst_feat.value, (SizedValue, WinLoss))
                and isinstance(snd_feat.value, (SizedValue, WinLoss))
            ):
                adv_side = advantage_tie_ratio.get_adv_side(
                    sex=match.sex, setpref=name[:3],
                    fst_sv=fst_feat.value, snd_sv=snd_feat.value)
                newname = f'{advantage_tie_ratio.ADV_PREF}{name}'
                if adv_side is None:
                    dct[newname] = None
                else:
                    dct[newname] = adv_side.is_left()

        def prep_simple_features():
            for feat in match.features:
                if isinstance(feat, feature.RigidFeature):
                    if isinstance(feat.value, SizedValue):
                        dct[feat.name] = (feat.value.value, feat.value.size)
                    else:
                        dct[feat.name] = feat.value

        def prep_plr_sv_feature(name):
            fst_name = f"fst_{name}"
            snd_name = f"snd_{name}"
            fst_feat = co.find_first(match.features, lambda f: f.name == fst_name)
            snd_feat = co.find_first(match.features, lambda f: f.name == snd_name)
            if (
                fst_feat is not None
                and snd_feat is not None
                and isinstance(fst_feat.value, (SizedValue, WinLoss))
                and isinstance(snd_feat.value, (SizedValue, WinLoss))
            ):
                dct[fst_name] = (fst_feat.value.value, fst_feat.value.size)
                dct[snd_name] = (snd_feat.value.value, snd_feat.value.size)

        def prep_plr_feature(name):
            fst_name = f"fst_{name}"
            snd_name = f"snd_{name}"
            fst_feat = co.find_first(match.features, lambda f: f.name == fst_name)
            snd_feat = co.find_first(match.features, lambda f: f.name == snd_name)
            if fst_feat is not None and snd_feat is not None:
                dct[fst_name] = fst_feat.value
                dct[snd_name] = snd_feat.value

        def update_dict(dct, prefix, player, hist_srv_win, hist_rcv_win):
            if player is not None:
                dct[prefix + "_player_id"] = player.ident
                dct[prefix + "_player_name"] = player.name
                dct[prefix + "_player_rating"] = player.rating
                if player.birth_date:
                    ymd = (
                        player.birth_date.year,
                        player.birth_date.month,
                        player.birth_date.day,
                    )
                    dct[prefix + "_player_bdate"] = ymd
                else:
                    dct[prefix + "_player_bdate"] = None
            else:
                dct[prefix + "_player_id"] = None
                dct[prefix + "_player_name"] = None
                dct[prefix + "_player_rating"] = {}
                dct[prefix + "_player_bdate"] = None
            dct[prefix + "_hist_srv_win"] = (hist_srv_win.value, hist_srv_win.size)
            dct[prefix + "_hist_rcv_win"] = (hist_rcv_win.value, hist_rcv_win.size)

        dct = {}
        prep_simple_features()
        update_dict(
            dct,
            "fst",
            match.first_player,
            match.hist_fst_srv_win,
            match.hist_fst_rcv_win,
        )
        update_dict(
            dct,
            "snd",
            match.second_player,
            match.hist_snd_srv_win,
            match.hist_snd_rcv_win,
        )
        if match.rnd is not None:
            dct["rnd"] = str(match.rnd)
        if match.features:
            prep_plr_feature("fatigue")
            prep_plr_feature("plr_tour_adapt")
            prep_plr_feature("prevyear_tour_rnd")
            prep_plr_feature("set2win_after_set1loss")
            prep_plr_feature("set2win_after_set1win")
            prep_plr_feature("decided_begin_sh-3")
            prep_plr_feature("decided_keep_sh-3")
            prep_plr_feature("decided_recovery_sh-3")
            prep_plr_feature("trail")
            prep_plr_feature("choke")
            prep_plr_feature("absence")
            prep_plr_feature("retired")
            prep_plr_feature("pastyear_nmatches")
            prep_plr_sv_feature("sd_tie_ratio")
            prep_advleft_tie_ratio("sd_tie_ratio")

        dct["decset_ratio_dif"] = match.decset_ratio_dif
        dct["decset_bonus_dif"] = match.decset_bonus_dif
        dct["h2h_direct"] = match.h2h_direct
        if match.offer and match.offer.win_coefs:
            dct["fst_win_coef"] = match.offer.win_coefs.first_coef
            dct["snd_win_coef"] = match.offer.win_coefs.second_coef
        dct["fst_draw_status"] = (
            match.first_draw_status if match.first_draw_status is not None else "")
        dct["snd_draw_status"] = (
            match.second_draw_status if match.second_draw_status is not None else "")
        if match.fst_last_res:
            dct["fst_last_res"] = match.fst_last_res.week_results_list
        if match.snd_last_res:
            dct["snd_last_res"] = match.snd_last_res.week_results_list
        return dct

    def check_dict(self, dct, match):
        absent_names = [fn for fn in self.important_features if dct.get(fn) is None]
        if absent_names:
            log.warning(f"prelivedct check: {match.name}"
                        f" ABSENT: {', '.join(absent_names)}")

    def prepare(self, match):
        dct = self.prepare_dict(match)
        pre_live_dir.prepare_dir("matches")
        pre_live_dir.save_data(match.pre_live_name(), dct)
        self.check_dict(dct, match)

    def work(self):
        tbeg = time.perf_counter()
        wpage.refresh()
        events = get_company(args.company_name).fetch_events(
            page_source=wpage.get_page_source(),
            skip_levels=skip_levels_work(),
            match_status=MatchStatus.scheduled,
        )
        for event in events:
            if self.is_event_towork(event):
                if (event.sex, event.tour_name) in _tourinfo_cache:
                    tour_id, level, surface = _tourinfo_cache[
                        (event.sex, event.tour_name)
                    ]
                    event.tour_id = tour_id
                    event.tour_info.level = level
                    event.tour_info.surface = surface
                    event.define_features()
                    tour = co.find_first(
                        weeked_tours.tail_tours(event.sex),
                        lambda t: t.ident == event.tour_id,
                    )
                    if tour is not None:
                        for match in event.matches:
                            if self.is_match_towork(match):
                                match.define_players(event.sex)
                                # weeked_tours may updated:
                                match.fill_details_tried = False
                                match.fill_details(tour)
                                log_preparing_match(match, comment='YEScached')
                                self.prepare(match)
                else:
                    # two next statements makes more long algo
                    event.define_features()
                    event.define_players()
                    event.define_level()
                    if event.tour_info is not None and event.tour_id is not None:
                        tour = co.find_first(
                            weeked_tours.tail_tours(event.sex),
                            lambda t: t.ident == event.tour_id,
                        )
                        if tour is not None:
                            for match in event.matches:
                                if self.is_match_towork(match):
                                    # weeked tours may updated
                                    match.fill_details_tried = False
                                    match.fill_details(tour)
                                    log_preparing_match(match, comment='NOTcached')
                                    self.prepare(match)
                            if event.tour_id is not None:
                                _tourinfo_cache[(event.sex, event.tour_name)] = (
                                    event.tour_id,
                                    event.level,
                                    event.surface,
                                )
        self.work_count += 1
        tend = time.perf_counter()
        log.info(f"MatchDataScript work done in {tend - tbeg:0.1f} seconds")


def log_preparing_match(match: LiveMatch, comment: str):
    log.info(
        f"prep {match.datetime.hour}:{match.datetime.minute} {match.name}"
        f" {match.tour_name} {comment} {match.sex} {match.level} {match.surface}"
    )


def get_min_timer_script(scripts) -> Script:
    minimum = 1e10
    script_selected = None
    for script in scripts:
        timer_wait = script.timer.remind_to_overtime()
        if timer_wait < minimum:
            minimum = timer_wait
            script_selected = script
    return script_selected


def main(scripts):
    global wpage
    company = get_company(args.company_name)
    wpage = create_page(score_company=company, is_main=True, headless=True)
    company.initialize_players_cache(wpage.get_page_source())

    if not atfirst_after.is_initialized_day(datetime.date.today()):
        atfirst_after.initialize_day(
            company.fetch_events(
                page_source=wpage.get_page_source(),
                skip_levels=skip_levels_work(),
                match_status=MatchStatus.scheduled,
            ),
            datetime.date.today(),
        )

    log.info("begin main cycle")
    exiting = False
    first_exec = True
    while not exiting:
        try:
            script = get_min_timer_script(scripts)
            if not first_exec:
                time.sleep(script.timer.remind_to_overtime())
            script.work()
            script.timer.restart()
            first_exec = False
        except co.TennisError as err:
            log.exception(str(err))
            break
    stop_web_driver(wpage.get_web_driver())
    wpage = None


def make_scripts():
    result = [
        MatchDataScript(wait_timeout=60 * 12, prelive_threshold=60 * 30),
    ]
    if args.oncourt and automate is not None:
        result.append(OncourtUpdateScript(wait_timeout=2000))
    return result


def initialize():
    date = tt.past_monday_date(datetime.date.today())
    min_date = date - datetime.timedelta(days=7 * matchstat.HIST_WEEKS_NUM)

    matchstat.initialize(min_date=min_date, time_reverse=True, tour_surface=True)

    decided_set.initialize_results(
        sex=None, min_date=LiveMatch.min_decset_date, max_date=datetime.date.today()
    )
    bet_coefs.initialize(sex="wta", min_date=min_date)
    bet_coefs.initialize(sex="atp", min_date=min_date)
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
    weeked_tours.use_tail_tours_cache = True


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company_name", choices=["T24", "FS"], default="T24")
    parser.add_argument("--cleardir", action="store_true")
    parser.add_argument("--oncourt", action="store_true")
    parser.add_argument("--dump_tail_tours", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    log.remove()
    log.add(sys.stderr, level='INFO')
    log.add('../log/pre_live.log', level='INFO', rotation='10:00', compression='zip')
    args = parse_command_line_args()
    log.info("started with company_name {}".format(args.company_name))
    dbcon.open_connect()
    extplayers.initialize(yearsnum=1.2)
    ratings.initialize(
        sex=None,
        rtg_names=("std", "elo"),
        min_date=datetime.date.today() - datetime.timedelta(days=21),
    )
    ratings.Rating.register_rtg_name("elo")
    ratings.Rating.register_rtg_name("elo_alt")
    mon_date = tt.past_monday_date(datetime.date.today())
    # tie_stat.initialize_results(sex=None,
    #                             min_date=mon_date - datetime.timedelta(days=365 * 3),
    #                             max_date=None)
    # impgames_stat.initialize_results(
    #     'wta', min_date=datetime.date.today() - datetime.timedelta(
    #         days=impgames_stat.HISTORY_DAYS))
    # impgames_stat.initialize_results(
    #     'atp', min_date=datetime.date.today() - datetime.timedelta(
    #         days=impgames_stat.HISTORY_DAYS))
    initialize()
    get_company(args.company_name).initialize()
    if args.cleardir:
        pre_live_dir.remove_all()
    if args.dump_tail_tours:
        weeked_tours.dump_tail_tours(sex="wta", tail_weeks=2)
        weeked_tours.dump_tail_tours(sex="atp", tail_weeks=2)
    main(make_scripts())
    dbcon.close_connect()
    sys.exit(0)
