""" пополнение базы детальных счетов (DetailedScore) c сайта flashscore.com
"""
import sys
import datetime
import time
from collections import defaultdict
import unittest
import argparse
import winsound

import log
import common as co
import cmds
import weeked_tours
import tennis_time as tt
import oncourt_players
import dba
from detailed_score_dbsa import open_db, MatchRec
from detailed_score import detailed_score_side_reversed, error_code
import common_wdriver
from live import MatchStatus, skip_levels_default
import tournament_misc as trmt_misc
import flashscore
from interrupt_matches import is_interrupt


def fetch_initialize(sex=None, yearsnum=2):
    dba.open_connect()
    oncourt_players.initialize(yearsnum=yearsnum)
    now_date = datetime.datetime.now().date()
    cur_week_monday = tt.past_monday_date(now_date)
    if sex in (None, "wta"):
        weeked_tours.initialize_sex(
            "wta",
            min_date=cur_week_monday - datetime.timedelta(days=14),
            max_date=None,
            with_today=True,
        )
    if sex in (None, "atp"):
        weeked_tours.initialize_sex(
            "atp",
            min_date=cur_week_monday - datetime.timedelta(days=14),
            max_date=None,
            with_today=True,
        )

    fsdrv = common_wdriver.wdriver(company_name="FS")
    fsdrv.start()
    time.sleep(5)
    return fsdrv


def fetch_finalize(fsdrv):
    fsdrv.stop()
    dba.close_connect()


def show_errors(tour_events, daysago, n_warns):
    """return True if stop work, False if ok or ignore"""
    sex_match_err_list = []
    for tour_evt in tour_events:
        if tour_evt.tour_id is None:
            continue
        for match in tour_evt.matches:
            if (
                match.rnd is not None
                and match.date is not None
                and match.first_player is not None
                and match.first_player.ident is not None
                and match.second_player is not None
                and match.second_player.ident is not None
                and hasattr(match, "detailed_score")
                and match.detailed_score is not None
                and match.detailed_score.error > 0
                and match.detailed_score.error != error_code("RAIN_INTERRUPT")
            ):
                sex_match_err_list.append(
                    (tour_evt.sex, match, match.detailed_score.error)
                )
    if len(sex_match_err_list) > 0 or n_warns > 0:
        for sex_match_err in sex_match_err_list:
            sex, match, err = sex_match_err
            log.error("{} {} err:{}".format(sex, match, err))

        if args.interactive:
            msg = (
                "day-{} {} dscore error(s) found, {} warns. Continue? "
                + "(don't forget switch to browser)"
            ).format(daysago, len(sex_match_err_list), n_warns)
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            is_continue_ask = cmds.CommandAsk(msg)
            answer = is_continue_ask.fire()
            if answer == "n":
                return True
    return False


def match_to_string(match):
    return "{} {} {} md: {} r: {} s: {}".format(
        match.sex, match.tour_name, match.name, match.date, match.rnd, match.score
    )


def tour_events_put_db(tour_events):
    ok_wta_count, ok_atp_count = 0, 0
    for tour_evt in tour_events:
        if tour_evt.tour_id is None:
            continue
        for match in tour_evt.matches:
            if (
                match.rnd is not None
                and match.date is not None
                and match.first_player is not None
                and match.first_player.ident is not None
                and match.second_player is not None
                and match.second_player.ident is not None
                and hasattr(match, "detailed_score")
                and match.detailed_score is not None
            ):
                fst_id, snd_id = match.first_player.ident, match.second_player.ident
                det_score = match.detailed_score
                sets_score = match.score.sets_score(full=True)
                if sets_score[0] < sets_score[1]:
                    # in db left should be winner
                    fst_id, snd_id = snd_id, fst_id
                    det_score = detailed_score_side_reversed(det_score)
                do_rain_interrupt_match(match)
                mrec = MatchRec(
                    date=match.date,
                    tour_id=tour_evt.tour_id,
                    rnd=match.rnd,
                    left_id=fst_id,
                    right_id=snd_id,
                    detailed_score=det_score,
                    score=match.score,
                )
                if tour_evt.sex == "wta":
                    dbdet_wta.insert_obj(mrec)
                elif tour_evt.sex == "atp":
                    dbdet_atp.insert_obj(mrec)
                else:
                    raise co.TennisError(f"sexless tour_evt {tour_evt}")
                log.info("inserted {}\n".format(match_to_string(match)))
                if tour_evt.sex == "wta":
                    ok_wta_count += 1
                else:
                    ok_atp_count += 1
    return ok_wta_count, ok_atp_count


def commit_sex(sex, day, cnt):
    if cnt > 0:
        if sex in ("wta", None):
            dbdet_wta.commit()
        if sex == ("atp", None):
            dbdet_atp.commit()
        log.info("{} commited done day {}".format(sex, day))


def do_rain_interrupt_match(match):
    if is_interrupt(
        sex=match.sex,
        date=match.date,
        left_id=match.first_player.ident,
        right_id=match.second_player.ident,
    ):
        match.detailed_score.error = error_code("RAIN_INTERRUPT")
        log.info("rain interrupt match marked {}".format(match))


def fetch_main(mindaysago, maxdaysago, sex=None):
    def is_prev_week(in_date):
        if args.current_week:
            return False
        cur_monday = tt.past_monday_date(datetime.date.today())
        in_monday = tt.past_monday_date(in_date)
        return in_monday <= cur_monday

    flashscore.deep_find_player_mode = True
    fsdrv = fetch_initialize(sex=sex, yearsnum=1.5)
    start_datetime = datetime.datetime.now()
    log.info(
        "started with daysago {}-{} current_week: {}".format(
            mindaysago, maxdaysago, args.current_week
        )
    )
    wta_tours = weeked_tours.all_tours("wta")
    atp_tours = weeked_tours.all_tours("atp")
    warn_dict = defaultdict(lambda: 0)  # day -> n_warns
    wta_cnt, atp_cnt = 0, 0
    is_stop = False
    for daysago in reversed(range(mindaysago, maxdaysago + 1)):
        date = datetime.date.today() - datetime.timedelta(days=daysago)
        flashscore.initialize(prev_week=is_prev_week(date))
        target_date = flashscore.goto_date(fsdrv, daysago, start_datetime.date())
        if target_date is None:
            raise co.TennisError("fail goto target_date for daysago {}".format(daysago))
        tour_events = flashscore.make_events(
            fsdrv.page(),
            skip_levels=skip_levels_default(),
            match_status=MatchStatus.finished,
            target_date=target_date,
        )
        if len(tour_events) > 0:
            warn_cnt = trmt_misc.events_deep_ident(
                tour_events,
                wta_tours,
                atp_tours,
                from_scored=True,
                warnloghead="unk_id_tours for {}".format(date),
            )
            warn_dict[daysago] += warn_cnt
            err_cnt = trmt_misc.tour_events_parse_detailed_score(tour_events, fsdrv)
            warn_dict[daysago] += err_cnt
            day_wta_cnt, day_atp_cnt = tour_events_put_db(tour_events)
            log.info(
                "day {} db-inserted wta_cnt: {} atp_cnt: {}".format(
                    daysago, day_wta_cnt, day_atp_cnt
                )
            )
            is_stop = show_errors(tour_events, daysago, warn_dict[daysago])
            if is_stop:
                break
            wta_cnt += day_wta_cnt
            atp_cnt += day_atp_cnt
            commit_sex("wta", daysago, day_wta_cnt)
            commit_sex("atp", daysago, day_atp_cnt)
        if daysago != mindaysago:
            fsdrv.goto_start()  # prepare for next goto date
    if not is_stop:
        log.info("all db-inserted wta_cnt: {} atp_cnt: {}".format(wta_cnt, atp_cnt))
    fetch_finalize(fsdrv)
    log.info(
        "{} finished within {}".format(
            __file__, str(datetime.datetime.now() - start_datetime)
        )
    )
    return 0


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp", "both"])
    parser.add_argument("--dayago1", type=int, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--dayago2", type=int, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--current_week", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if not args.dayago1 and not args.dayago2:
        log.initialize(
            co.logname(__file__, test=True), file_level="info", console_level="info"
        )
        unittest.main()
    else:
        log.initialize(co.logname(__file__), file_level="info", console_level="info")
        assert args.dayago1 <= args.dayago2, "daysago bad interval [{}, {}]".format(
            args.dayago1, args.dayago2
        )
        dbdet_wta = open_db(sex="wta")
        dbdet_atp = open_db(sex="atp")
        sys.exit(
            fetch_main(
                args.dayago1, args.dayago2, sex=None if args.sex == "both" else args.sex
            )
        )
