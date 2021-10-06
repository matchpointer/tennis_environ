r"""
module gives standalone process (prepare data) for matches which soon be live.
"""
import sys
import signal
from typing import Optional
from common_wdriver import WDriver, wdriver

_drv: Optional[WDriver] = None


def signal_handler(signal, frame):
    print("\nprogram exiting gracefully after signaled")
    global _drv
    if _drv is not None:
        print("\n_drv stoping...")
        _drv.stop()
        _drv = None
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

import time
import datetime
from collections import defaultdict
import argparse

import stopwatch

import log
import common as co
import feature
import tennis_time as tt
import bet_coefs
import dba
import oncourt_players
import matchstat
import ratings
import decided_set
import weeked_tours
from live import (
    MatchStatus,
    get_events,
    LiveMatch,
    initialize as live_initialize,
    initialize_players_cache,
    skip_levels_work,
)
import pre_live_dir
from stat_cont import WinLoss
from report_line import SizedValue
import atfirst_after

try:
    import automate2
except ImportError:
    automate2 = None

_DEBUG_MATCH_NAME = ""  # sample: "Samsonova L. - Konjuh A."


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
        result = automate2.oncourt_update_connect()
        if result:
            time.sleep(7)
            initialize()
            log.info("OncourtUpdateScript work done")
        else:
            log.error("OncourtUpdateScript failed")


class MatchDataScript(Script):
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
            prep_plr_feature("absence")
            prep_plr_feature("retired")
            # other features temporary removed

        return dct

    def check_dict(self, dct, match):
        if match.level in ("main", "gs", "masters"):
            none_cnt = sum([1 for v in dct.values() if v is None])
            if none_cnt > 0 or len(dct) < 16:
                log.warn(
                    "prelivedct check none_cnt: {} len: {} in {}".format(
                        none_cnt, len(dct), match.name
                    )
                )

    def prepare(self, match):
        dct = self.prepare_dict(match)
        pre_live_dir.prepare_dir("matches")
        pre_live_dir.save_data(match.pre_live_name(), dct)
        self.check_dict(dct, match)

    def work(self):
        tbeg = time.perf_counter()
        _drv.live_page_refresh()
        events = get_events(
            _drv.page(),
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
                                match.define_players("FS", event.sex)
                                # weeked_tours may updated:
                                match.fill_details_tried = False
                                match.fill_details(tour)
                                log_preparing_match(match, comment='YEScached')
                                self.prepare(match)
                else:
                    # two next statements makes more long algo
                    event.define_features()
                    event.define_players(company_name="FS")
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
    global _drv
    _drv = wdriver(company_name="FS", headless=True)
    _drv.start()
    _drv.go_live_page()
    initialize_players_cache(_drv.page())

    atfirst_after.initialize_day(
        get_events(
            _drv.page(),
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
            log.error(str(err), exc_info=True)
            break
    _drv.stop()
    _drv = None


def make_scripts():
    result = [
        MatchDataScript(wait_timeout=60 * 12, prelive_threshold=60 * 30),
    ]
    if args.oncourt and automate2 is not None:
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
    parser.add_argument("--company_name", choices=["BC", "FS"], default="FS")
    parser.add_argument("--cleardir", action="store_true")
    parser.add_argument("--oncourt", action="store_true")
    parser.add_argument("--dump_tail_tours", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    log.info("started with company_name {}".format(args.company_name))
    dba.open_connect()
    oncourt_players.initialize(yearsnum=1.2)
    ratings.initialize(
        sex=None,
        rtg_names=("std", "elo"),
        min_date=datetime.date.today() - datetime.timedelta(days=21),
    )
    ratings.Rating.register_rtg_name("elo")
    ratings.Rating.register_rtg_name("elo_alt")
    mon_date = tt.past_monday_date(datetime.date.today())
    initialize()
    live_initialize()
    if args.cleardir:
        pre_live_dir.remove_all()
    if args.dump_tail_tours:
        weeked_tours.dump_tail_tours(sex="wta", tail_weeks=2)
        weeked_tours.dump_tail_tours(sex="atp", tail_weeks=2)
    main(make_scripts())
    dba.close_connect()
    sys.exit(0)
