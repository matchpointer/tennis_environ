# -*- coding=utf-8 -*-
r"""
module gives Application with upper level logic.
"""
import os
import sys
import copy
from collections import defaultdict
import time
import datetime
import argparse
import winsound

import tkinter
import tkinter.ttk

from selenium.common.exceptions import TimeoutException
from requests.exceptions import RequestException

# import clf_decided_00_apply
import clf_decided_00dog_apply
import stopwatch

from loguru import logger as log
import cfg_dir
import config_personal
import ratings
import tennis_time as tt
import bet_coefs
from oncourt import dba, extplayers
import sms_svc
from pages.create_page import create_page
from wdriver import stop_web_driver
from score_company import get_company, ScoreCompany
import matchstat
import file_utils as fu

from clf_common import load_variants
import decided_set
import weeked_tours
from live import (
    LiveTourEvent,
    LiveMatch,
    skip_levels_work,
)
from debug_helper import get_debug_match_name, set_debug_match_name, debug_match_data_save
from tournament_misc import log_events, events_tostring
import betfair_client
from betfair_bet import winmatch_market, winset2_market, winset1_market

from live_alerts import (
    AlertMessage,
    AlertSetDecidedWinDispatchClf,
)
import predicts_db


class EventsFile:
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


root = tkinter.Tk()


class Application(tkinter.Frame):
    def __init__(self, company: ScoreCompany, quick_timeout, slow_timeout):
        tkinter.Frame.__init__(self, None)
        self.grid()
        self.company = company
        self.quick_timeout = quick_timeout
        self.slow_timeout = slow_timeout
        self.timer = stopwatch.OverTimer(quick_timeout)
        self.events = []
        self.events_file = None
        if args.eventsfile:
            self.events_file = EventsFile(
                os.path.join(cfg_dir.log_dir(), "live_events.txt"))
        self.REMOVING_EVENT_SECS = 780  # delay before real remove of event
        self.REMOVING_MATCH_SECS = 750  # delay before real remove of match
        self.updating = False
        self.alerts_from_sex = defaultdict(list)
        self.make_alerts()
        self.wpage = create_page(score_company=company, is_main=True,
                                 headless=True)
        company.initialize_players_cache(self.wpage.get_page_source())
        self.wake_timer = None  # or stopwatch.PointTimer(datetime)
        self.sms_start_timer = stopwatch.PointTimer(
            datetime.datetime(year=2021, month=7, day=27, hour=8, minute=20)
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
            command=self.change_betfair_mode,
        )
        self.betfairmode_cbtn.grid(row=row, column=2)
        if args.betfair:
            self.betfairmode_cbtn.select()
        row += 1

        self.allow_slow_mode_var = tkinter.IntVar()
        self.allow_slow_mode_cbtn = tkinter.Checkbutton(
            self,
            text="slow_allow_mode",
            variable=self.allow_slow_mode_var,
            offvalue=0,
            onvalue=1,
        )
        self.allow_slow_mode_cbtn.grid(row=row, column=2)
        row += 1

        self.ext_market_var = tkinter.IntVar()
        self.ext_market_cbtn = tkinter.Checkbutton(
            self,
            text="ext_market",
            variable=self.ext_market_var,
            offvalue=0,
            onvalue=1,
        )
        self.ext_market_cbtn.grid(row=row, column=2)
        self.ext_market_cbtn.select()
        row += 1

        self.message_lbl = tkinter.ttk.Label(self, text="Silence...", font="sans 12")
        self.message_lbl.grid(row=row, column=2)
        self.message_lbl.after_idle(self.check_time)
        row += 1

    def is_sleep_mode(self):
        return bool(self.sleepmode_var.get())

    def is_sms_mode(self):
        return bool(self.smsmode_var.get())

    def is_betfair_mode(self):
        return bool(self.betfairmode_var.get())

    def change_betfair_mode(self):
        if self.is_betfair_mode():  # enabling
            if not betfair_client.is_initialized():
                betfair_client.initialize()
        else:  # disabling
            if betfair_client.is_initialized():
                betfair_client.de_initialize()

    def is_allow_slow_mode(self):
        return bool(self.allow_slow_mode_var.get())

    def is_ext_market(self):
        return bool(self.ext_market_var.get())

    def make_alerts(self):
        load_variants(clf_decided_00dog_apply.apply_variants)

        sex = "wta"

        self.alerts_from_sex[sex].append(AlertSetDecidedWinDispatchClf())

        # -------------------------- atp -----------------------------
        sex = "atp"

        self.alerts_from_sex[sex].append(AlertSetDecidedWinDispatchClf())

    def is_slow_mode(self):
        return int(self.timer.threshold) == self.slow_timeout

    def is_quick_mode(self):
        return int(self.timer.threshold) == self.quick_timeout

    def save_page(self):
        debug_match_data_save()
        fu.write(filename="./live_mon_cur_page.html",
                 data=self.wpage.get_page_source(), encoding="utf8")

    def log_events(self, head="", extended=True, flush=False):
        log_events(self.events, head=head, extended=extended, flush=flush)

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
        for try_num in range(1, 4):
            try:
                fresh_events = self.company.fetch_events(
                    page_source=self.wpage.get_page_source(),
                    skip_levels=skip_levels_work())
                break
            except (
                RequestException, UnicodeEncodeError, TimeoutException, ValueError
            ) as err:
                # may be inet problems
                log.exception(
                    "drv.page_source {}\nfail try_num: {}".format(err, try_num))
                time.sleep(20)
                try:
                    self.wpage.refresh()
                except (RequestException, TimeoutException) as err2:
                    log.exception(
                        "drv.get {}\nfail try_num: {}".format(err2, try_num)
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
            event.define_players()
            event.define_level()
            event.set_matches_key()
            if event.need_fill_matches_details():
                event.fill_matches_details()
            for match in event.matches:
                match.log_check_unknown()

        if self.events_file is not None:
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
            is_attention, alert_messages = False, []
            if not self.is_sleep_mode():
                self.update_events()
                is_attention, alert_messages = self.dispatch()
            elif self.wake_timer and self.wake_timer.overtime():
                self.sleepmode_cbtn.deselect()
                self.timer.set_threshold(self.quick_timeout)
                log.info("wake from sleep to quick mode")

            if alert_messages:
                self.out_alert_messages(alert_messages)
            else:
                self.show_silence()
            if is_attention and self.is_slow_mode():
                log.info("switch to quick mode (exist alert)")
                self.timer.set_threshold(self.quick_timeout)
            elif not is_attention and self.is_quick_mode() and self.is_allow_slow_mode():
                log.info("switch to slow mode (no attention)")
                self.timer.set_threshold(self.slow_timeout)
            self.timer.restart()
            self.message_lbl.after(int(self.timer.threshold * 1000), self.check_time)
        else:
            self.message_lbl.after(
                int(self.timer.remind_to_overtime() * 1000), self.check_time
            )

    def out_alert_messages(self, alert_messages):
        text = "\n".join([a.text for a in alert_messages])
        self.message_lbl["text"] = text
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        Application.bring_to_front()
        if self.is_betfair_mode():
            self.out_betfair_messages(alert_messages)
            detail = '[B]'
        else:
            detail = '[b]'
        if self.is_sms_mode():
            self.out_sms_messages(alert_messages)
            detail += '[S]'
        else:
            detail += '[s]'
        log.info(f"alerted: {detail} {text}")

    def out_sms_messages(self, alert_messages):
        if not self.sms_start_timer or self.sms_start_timer.overtime():
            try:
                sms_svc.send_alert_messages(
                    [
                        a
                        for a in alert_messages
                        if self.out_betfair_permit(a)
                    ]
                )
            except sms_svc.SMSError as err:
                log.error("{}".format(err))

    @staticmethod
    def out_betfair_permit(msg: AlertMessage) -> bool:
        return (
            (msg.case_name.startswith("decided_00") and 'backdog' in msg.comment)
            #
            # or (msg.case_name.startswith("decided_00")
            #     and 'backfav' in msg.comment)
            #
            or (msg.case_name.startswith("decided_00nodog"))
            #
            # or (msg.case_name.startswith("secondset_00") and 'revenge' in msg.comment)
            #
            or msg.case_name.startswith("decided_66")
            or msg.case_name.startswith("open_66")
        )

    def out_betfair_messages(self, alert_messages):
        for msg in alert_messages:
            if self.out_betfair_permit(msg):
                summary_href = msg.summary_href if msg.summary_href else ""
                betfair_client.send_message(
                    self.get_market(case_name=msg.case_name, sex=msg.sex),
                    msg.sex,
                    msg.case_name,
                    msg.fst_name,
                    msg.snd_name,
                    msg.back_side,
                    summary_href,
                    msg.fst_betfair_name,
                    msg.snd_betfair_name,
                    msg.prob,
                    msg.book_prob,
                    msg.fst_id,
                    msg.snd_id,
                    msg.comment,
                    msg.level,
                )

    def get_market(self, case_name: str, sex: str):
        if self.is_ext_market():
            if case_name.startswith('secondset'):
                return winset2_market
            elif case_name.startswith('open'):
                return winset1_market
        return winmatch_market

    def show_silence(self):
        self.message_lbl["text"] = "Silence..."

    @staticmethod
    def bring_to_front():
        root.wm_attributes("-topmost", True)
        root.wm_attributes("-topmost", False)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sms", action="store_true")
    parser.add_argument("--betfair", action="store_true")
    parser.add_argument("--eventsfile", action="store_true")
    parser.add_argument("--company_name", choices=["T24", "FS"], default="T24")
    return parser.parse_args()


def main():
    log.info("started with company_name {}".format(args.company_name))
    config_personal.initialize_from_file('../personal.cfg', sign='matchpointer')
    if args.betfair:
        betfair_client.initialize()
    dba.open_connect()
    extplayers.initialize(yearsnum=1.2)
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
    decided_set.initialize_results(
        sex=None, min_date=LiveMatch.min_decset_date, max_date=datetime.date.today()
    )
    past_monday = tt.past_monday_date(datetime.date.today())
    bet_coefs.initialize(sex="wta", min_date=past_monday - datetime.timedelta(days=7))
    bet_coefs.initialize(sex="atp", min_date=past_monday - datetime.timedelta(days=7))
    # for fatigue, live-matches details(rnd-details, tour.level, tour.surface):
    weeked_tours.initialize_sex(
        "wta",
        min_date=past_monday - datetime.timedelta(days=7 * 55),
        max_date=past_monday + datetime.timedelta(days=11),
        with_today=True,
        with_paired=True,
        with_ratings=True,
        with_bets=True,
        with_stat=True,
        rnd_detailing=True,
    )
    weeked_tours.initialize_sex(
        "atp",
        min_date=past_monday - datetime.timedelta(days=7 * 55),
        max_date=past_monday + datetime.timedelta(days=11),
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
    #                             min_date=now_date - datetime.timedelta(days=365 * 3),
    #                             max_date=None)
    company = get_company(args.company_name)
    company.initialize()
    set_debug_match_name("no debug")
    log.info("DEBUG_MATCH_DATA_NAME: {}".format(get_debug_match_name()))
    app = Application(company=company, quick_timeout=2.7, slow_timeout=132)
    app.master.title("Live monitor")
    predicts_db.initialize()
    app.mainloop()
    log.info("Live monitor exiting")
    stop_web_driver(app.wpage.get_web_driver())
    if app.events_file is not None:
        app.events_file.close()
    predicts_db.finalize()
    dba.close_connect()


if __name__ == "__main__":
    log.remove()
    log.add(sys.stderr, level='INFO')
    log.add('../log/live_mon.log', level='INFO', rotation='10:00', compression='zip')
    args = parse_command_line_args()
    main()
