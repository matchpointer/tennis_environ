r"""
в модуле общая логика обработки live обьектов.
"""
import os
import sys
import copy
from collections import defaultdict
import time
import datetime
import random
import argparse
import unittest
import winsound

import tkinter
import tkinter.ttk

from selenium.common.exceptions import TimeoutException

import stopwatch

import common as co
import log
import cfg_dir
import config_personal
import ratings
import tennis
import tennis_time as tt
import bet_coefs
import dba
import sms_svc
import oncourt_players
import common_wdriver
import matchstat
import clf_decided_00

import clf_secondset_00
import decided_set
import weeked_tours
from live import (
    get_events,
    LiveTourEvent,
    LiveMatch,
    TourInfo,
    skip_levels_work,
    initialize as live_initialize,
    initialize_players_cache,
    debug_match_data_save,
    set_debug_match_name,
    get_debug_match_name,
)
from tournament_misc import log_events, events_tostring
import betfair_client

# import decided_win_by_two_sets_stat
from live_alerts import (
    AlertMessage,
    AlertSetDecidedCloserClf,
    AlertSet2WinClf,
    AlertScEqualTieRatio,
)

DEFAULT_MARKET_NAME = "MATCH_ODDS"


class EventsFile(object):
    def __init__(self, filename):
        self.fhandler = open(filename, "w+")
        self.periodic_timer = stopwatch.OverTimer(60 + 30)

    def periodic_write(self, events):
        if self.periodic_timer.overtime():
            time_txt = tt.formated_datetime(datetime.datetime.now())
            self.write(events, head=time_txt)
            self.periodic_timer.restart()

    def write(self, events, head="", extended=True):
        self.fhandler.seek(0)
        self.fhandler.truncate()
        self.fhandler.write(
            events_tostring(
                [ev for ev in events if ev.level != "future"],
                head=head,
                extended=extended,
            )
        )
        self.fhandler.flush()

    def close(self):
        self.fhandler.close()


get_events_from_files = None  # see sample in flashscore.py


DEBUG = False
debug_fhandle = open("./debug_live_mon.txt", "w")


def debug_write(data):
    timestamp = "[" + tt.formated_datetime(datetime.datetime.now()) + "]"
    debug_fhandle.write(timestamp + " " + data)
    debug_fhandle.flush()


class MatchGenStatTest(unittest.TestCase):
    def make_tour_info(self):
        tour_info = TourInfo()
        tour_info.sex = "wta"
        tour_info.tour_name = "Indian Wells"
        tour_info.qualification = False
        tour_info.level = tennis.Level("masters")
        tour_info.surface = tennis.Surface("Hard")
        return tour_info

    def test_tour_event_vesnina_babos(self):
        tour_event = LiveTourEvent(tour_info=self.make_tour_info())
        match = LiveMatch(live_event=tour_event)
        match.date = datetime.date(2017, 3, 14)
        match.rnd = tennis.Round("Third")
        match.first_player = tennis.Player(ident=4123, name="Elena Vesnina", cou="RUS")
        match.second_player = tennis.Player(ident=10600, name="Timea Babos", cou="HUN")
        match.define_players_tried = True
        tour_event.matches = [match]


root = tkinter.Tk()


class Application(tkinter.Frame):
    def __init__(self, company_name, quick_timeout, slow_timeout):
        tkinter.Frame.__init__(self, None)
        self.grid()
        self.company_name = company_name
        self.quick_timeout = quick_timeout
        self.slow_timeout = slow_timeout
        self.timer = stopwatch.OverTimer(quick_timeout)
        self.events = []
        self.events_file = EventsFile(
            os.path.join(cfg_dir.log_dir(), "live_events.txt")
        )
        self.REMOVING_EVENT_SECS = 780  # delay before real remove of event
        self.REMOVING_MATCH_SECS = 750  # delay before real remove of match
        self.updating = False
        self.alerts_from_sex = defaultdict(list)
        self.make_alerts()
        self.drv = common_wdriver.wdriver(company_name, headless=True)
        if not DEBUG:
            self.drv.start()
            self.drv.go_live_page()
            initialize_players_cache(self.drv.page())
        self.wake_timer = None  # stopwatch.PointTimer(datetime.datetime(
        # year=2018, month=2, day=11, hour=8, minute=0))
        self.sms_start_timer = stopwatch.PointTimer(
            datetime.datetime(year=2021, month=2, day=23, hour=7, minute=15)
        )

        row = 0

        self.log_btn = tkinter.ttk.Button(
            self, text="save page", command=self.save_page
        )
        self.log_btn.grid(row=row, column=2)
        row += 1

        self.show_btn = tkinter.ttk.Button(
            self, text="log events", command=self.log_events
        )
        self.show_btn.grid(row=row, column=2)
        row += 1

        self.update_btn = tkinter.ttk.Button(
            self, text="update events", command=self.update_events
        )
        self.update_btn.grid(row=row, column=2)
        row += 1

        self.sleepmode_var = tkinter.IntVar()
        self.sleepmode_cbtn = tkinter.Checkbutton(
            self, text="sleep mode", variable=self.sleepmode_var, offvalue=0, onvalue=1
        )
        self.sleepmode_cbtn.grid(row=row, column=2)
        row += 1

        self.smsmode_var = tkinter.IntVar()
        self.smsmode_cbtn = tkinter.Checkbutton(
            self, text="sms mode", variable=self.smsmode_var, offvalue=0, onvalue=1
        )
        self.smsmode_cbtn.grid(row=row, column=2)
        if args.sms:
            self.smsmode_cbtn.select()
        row += 1

        self.betfairmode_var = tkinter.IntVar()
        self.betfairmode_cbtn = tkinter.Checkbutton(
            self,
            text="betfair mode",
            variable=self.betfairmode_var,
            offvalue=0,
            onvalue=1,
            command=self.betfair_change,
        )
        self.betfairmode_cbtn.grid(row=row, column=2)
        if args.betfair:
            self.betfairmode_cbtn.select()
        row += 1

        self.slow_allow_mode_var = tkinter.IntVar()
        self.slow_allow_mode_cbtn = tkinter.Checkbutton(
            self,
            text="slow_allow_mode",
            variable=self.slow_allow_mode_var,
            offvalue=0,
            onvalue=1,
        )
        self.slow_allow_mode_cbtn.grid(row=row, column=2)
        row += 1

        self.message_lbl = tkinter.ttk.Label(self, text="Silence...", font="sans 12")
        self.message_lbl.grid(row=row, column=2)
        self.message_lbl.after_idle(self.check_time)
        row += 1

    def sleep_mode(self):
        return bool(self.sleepmode_var.get())

    def sms_mode(self):
        return bool(self.smsmode_var.get())

    def betfair_mode(self):
        return bool(self.betfairmode_var.get())

    def betfair_change(self):
        if self.betfair_mode():  # enabling
            if not betfair_client.is_initialized():
                betfair_client.initialize()
        else:  # disabling
            if betfair_client.is_initialized():
                betfair_client.de_initialize()

    def slow_allow_mode(self):
        return bool(self.slow_allow_mode_var.get())

    def make_alerts(self):
        clf_decided_00.initialize()
        clf_secondset_00.initialize()
        sex = "wta"

        self.alerts_from_sex[sex].append(
            AlertScEqualTieRatio(
                "decided",
                max_rating=400,
                max_dif_rating=200,
                min_proba=0.57,
                max_lay_ratio=0.5,
                min_size=12,
            )
        )

        self.alerts_from_sex[sex].append(AlertSet2WinClf())
        self.alerts_from_sex[sex].append(AlertSetDecidedCloserClf())

        # -------------------------- atp -----------------------------
        sex = "atp"

        self.alerts_from_sex[sex].append(AlertSet2WinClf())

        self.alerts_from_sex[sex].append(
            AlertScEqualTieRatio(
                "decided",
                max_rating=400,
                max_dif_rating=200,
                min_proba=0.57,
                max_lay_ratio=0.5,
                min_size=14,
            )
        )

        self.alerts_from_sex[sex].append(AlertSetDecidedCloserClf())

    def is_slow_mode(self):
        return int(self.timer.threshold) == self.slow_timeout

    def is_quick_mode(self):
        return int(self.timer.threshold) == self.quick_timeout

    def save_page(self):
        debug_match_data_save()
        self.drv.save_page(filename="./live_mon_cur_page.html", encoding="utf8")

    def log_events(self, head="", extended=True, flush=False):
        log_events(self.events, head=head, extended=extended, flush=flush)

    def wait_random(self):
        if self.company_name == "BC":
            random_sec_max = int(self.timer.threshold * 0.1)
            if random_sec_max >= 1:
                random_sec = random.randrange(0, random_sec_max)
                time.sleep(random_sec)

    def dispatch(self):
        is_attention = False
        fresh_alert_messages = []
        for live_evt in self.events:
            if not live_evt.live_betable():
                continue
            for m in live_evt.matches:
                if m.left_service is None:
                    continue
                for alert in self.alerts_from_sex[live_evt.sex]:
                    if alert in m.worked_alerts:
                        continue
                    result = alert.find_alertion(m)
                    if result is False or isinstance(result, AlertMessage):
                        m.worked_alerts.append(alert)
                        if isinstance(result, AlertMessage):
                            fresh_alert_messages.append(result)
                    elif result is True:
                        is_attention = True
        return is_attention, fresh_alert_messages

    def get_fresh_events(self):
        fresh_events = []
        if DEBUG:
            try:
                fresh_events = next(get_events_from_files)
            except StopIteration:
                log.info("StopIteration in debug")
        else:
            for try_num in range(1, 4):
                try:
                    self.drv.live_page_refresh()
                    fresh_events = get_events(
                        self.drv.page(),
                        skip_levels=skip_levels_work(),
                        company_name=self.company_name,
                    )
                    break
                except (UnicodeEncodeError, TimeoutException, ValueError) as err:
                    # may be inet problems
                    log.error(
                        "drv.page_source {}\nfail try_num: {}".format(err, try_num),
                        exc_info=True,
                    )
                    time.sleep(20)
                    try:
                        self.drv.current_page_refresh()
                    except TimeoutException as err2:
                        log.error(
                            "drv.get {}\nfail try_num: {}".format(err2, try_num),
                            exc_info=True,
                        )
                        time.sleep(20)

        return fresh_events

    def update_events(self):
        if self.updating:
            return
        self.updating = True
        fresh_events = self.get_fresh_events()

        self.remove_completed(fresh_events)
        self.update_with_fresh_events(fresh_events)

        for event in self.events:
            event.define_features()
            event.define_players(self.company_name)
            event.define_level()
            event.set_matches_key()
            if event.need_fill_matches_details():
                event.fill_matches_details()
            for match in event.matches:
                match.log_check_unknown()

        self.events_file.periodic_write(self.events)
        self.updating = False

    def remove_completed(self, fresh_events):
        def try_remove(cont, idx, timer_seconds):
            obj = cont[idx]
            if obj.kill_timer:
                if obj.kill_timer.overtime():
                    if isinstance(obj, LiveTourEvent):
                        is_actual_matches, remind_threshold = False, 0
                        for m in obj.matches:
                            if m.kill_timer is None:
                                is_actual_matches = True
                                remind_threshold = self.REMOVING_MATCH_SECS
                                m.kill_timer = stopwatch.OverTimer(
                                    self.REMOVING_MATCH_SECS
                                )
                            elif m.kill_timer and not m.kill_timer.overtime():
                                is_actual_matches = True
                                remind_threshold = max(
                                    remind_threshold, m.kill_timer.remind_to_overtime()
                                )
                        if is_actual_matches:
                            obj.kill_timer.set_threshold(remind_threshold + 10)
                            obj.kill_timer.restart()
                            return False
                    del cont[idx]
                    return True
            else:
                obj.kill_timer = stopwatch.OverTimer(timer_seconds)
            return False

        for event_idx in reversed(range(len(self.events))):
            stored_event = self.events[event_idx]
            if stored_event in fresh_events:
                # remove completed matches (if exist)
                fresh_evt_idx = fresh_events.index(stored_event)
                for match_idx in reversed(range(len(stored_event.matches))):
                    if (
                        stored_event.matches[match_idx]
                        not in fresh_events[fresh_evt_idx].matches
                    ):
                        try_remove(
                            stored_event.matches, match_idx, self.REMOVING_MATCH_SECS
                        )
            else:
                try_remove(self.events, event_idx, self.REMOVING_EVENT_SECS)

    def update_with_fresh_events(self, fresh_events):
        for fresh_event in fresh_events:
            if fresh_event in self.events:
                evt_idx = self.events.index(fresh_event)
                stored_event = self.events[evt_idx]
                stored_event.kill_timer = None
                for fresh_m in fresh_event.matches:
                    if fresh_m in stored_event.matches:
                        m_idx = stored_event.matches.index(fresh_m)
                        self.events[evt_idx].matches[m_idx].update_with(
                            fresh_m.score, fresh_m.left_service, fresh_m.ingame
                        )
                        self.events[evt_idx].matches[m_idx].kill_timer = None
                    else:
                        fresh_m.live_event = stored_event
                        self.events[evt_idx].matches.append(copy.copy(fresh_m))
            else:
                self.events.append(fresh_event)

    def check_time(self):
        if self.timer.overtime():
            self.wait_random()
            is_attention, alert_messages = False, []
            if not self.sleep_mode():
                self.update_events()
                is_attention, alert_messages = self.dispatch()
            elif self.wake_timer and self.wake_timer.overtime():
                self.sleepmode_cbtn.deselect()
                self.timer.set_threshold(self.quick_timeout)
                log.info("wake from sleep to quick mode")

            if alert_messages:
                self.show_alert_messages(alert_messages)
            else:
                self.show_silence()
            if is_attention and self.is_slow_mode():
                log.info("switch to quick mode (exist alert)")
                self.timer.set_threshold(self.quick_timeout)
            elif not is_attention and self.is_quick_mode() and self.slow_allow_mode():
                log.info("switch to slow mode (no attention)")
                self.timer.set_threshold(self.slow_timeout)
            self.timer.restart()
            self.message_lbl.after(int(self.timer.threshold * 1000), self.check_time)
        else:
            self.message_lbl.after(
                int(self.timer.remind_to_overtime() * 1000), self.check_time
            )

    def show_alert_messages(self, alert_messages):
        text = "\n".join([a.text for a in alert_messages])
        self.message_lbl["text"] = text
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        Application.bring_to_front()
        if not DEBUG:
            if self.betfair_mode():
                for msg in alert_messages:
                    if (
                        msg.case_name == "decided_00" and msg.prob > 0.51
                    ) or msg.case_name == "secondset_00":
                        summary_href = msg.summary_href if msg.summary_href else ""
                        betfair_client.send_message(
                            DEFAULT_MARKET_NAME,
                            msg.sex,
                            msg.case_name,
                            msg.fst_name,
                            msg.snd_name,
                            msg.back_side,
                            summary_href,
                            msg.fst_betfair_name,
                            msg.snd_betfair_name,
                            msg.prob,
                        )
            if self.sms_mode():
                if not self.sms_start_timer or self.sms_start_timer.overtime():
                    try:
                        sms_svc.send_alert_messages(
                            [
                                a
                                for a in alert_messages
                                if a.case_name != "secondset_win"
                            ]
                        )
                    except sms_svc.SMSError as err:
                        log.error("{}".format(err))
        log.info("alerted: " + text)

    def show_silence(self):
        self.message_lbl["text"] = "Silence..."

    @staticmethod
    def bring_to_front():
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-topmost", False)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--sms", action="store_true")
    parser.add_argument("--betfair", action="store_true")
    parser.add_argument("--company_name", choices=["BC", "FS"], default="FS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    log.info("started with company_name {} gui {}".format(args.company_name, args.gui))
    config_personal.initialize_from_file('personal.cfg', sign='matchpointer')
    if args.betfair:
        betfair_client.initialize()
    dba.open_connect()
    oncourt_players.initialize(yearsnum=1.2)
    ratings.initialize(
        sex=None,
        rtg_names=("std", "elo"),
        min_date=datetime.date.today() - datetime.timedelta(days=21),
    )
    ratings.Rating.register_rtg_name("elo")
    ratings.Rating.register_rtg_name("elo_alt")
    matchstat.initialize(
        min_date=(
            tt.past_monday_date(datetime.date.today())
            - datetime.timedelta(days=7 * matchstat.HIST_WEEKS_NUM)
        ),
        time_reverse=True,
        tour_surface=True,
    )
    if args.gui:
        decided_set.initialize_results(
            sex=None, min_date=LiveMatch.min_decset_date, max_date=datetime.date.today()
        )
        date = tt.past_monday_date(datetime.date.today())
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
        weeked_tours.use_tail_tours_cache = True
        # impgames_stat.initialize_results(
        #     'wta', min_date=datetime.date.today() - datetime.timedelta(
        #                                      days=impgames_stat.HISTORY_DAYS))
        # impgames_stat.initialize_results(
        #     'atp', min_date=datetime.date.today() - datetime.timedelta(
        #                                      days=impgames_stat.HISTORY_DAYS))
        # tie_stat.initialize_results(sex=None,
        #                             min_date=date - datetime.timedelta(days=365 * 3),
        #                             max_date=None)
        live_initialize()
        set_debug_match_name("Papashvili D. - Kremnev A.")
        log.info("DEBUG_MATCH_DATA_NAME: {}".format(get_debug_match_name()))
        app = Application(
            company_name=args.company_name, quick_timeout=2.0, slow_timeout=132
        )
        app.master.title("Live monitor")
        app.mainloop()
        if not DEBUG:
            app.drv.stop()
        app.drv = None
        app.events_file.close()

        dba.close_connect()
        debug_fhandle.close()
        sys.exit(0)
    else:
        unittest.main()
        dba.close_connect()
        debug_fhandle.close()
