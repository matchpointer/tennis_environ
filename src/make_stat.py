import sys
import datetime
from operator import sub
import argparse

from loguru import logger as log
import tournament as trmt
import tennis_time as tt
import dba
import ratings_std

import qual_seeds
import oncourt_players

import matchstat
import pandemia


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
        if pandemia.exhib_tour_id(sex) == tour.ident:
            continue
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
        if pandemia.exhib_tour_id(sex) == tour.ident:
            continue
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
    try:
        log.info(__file__ + " started sex:" + str(args.sex))
        start_datetime = datetime.datetime.now()
        dba.open_connect()
        ratings_std.initialize(sex=args.sex)
        qual_seeds.initialize()
        oncourt_players.initialize(yearsnum=18)

        matchstat_process(sex=args.sex)

        dba.close_connect()
        log.info(
            "{} finished within {}".format(
                __file__, str(datetime.datetime.now() - start_datetime)
            )
        )
        return 0
    except Exception as err:
        log.exception("{0} [{1}]".format(err, err.__class__.__name__))
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    log.add(f'../log/make_stat_{args.sex}.log', level='INFO',
            rotation='10:00', compression='zip')
    sys.exit(do_stat())
