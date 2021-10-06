# -*- coding: utf-8 -*-
import os
import datetime
from typing import Optional
from contextlib import closing
from datetime import date, timedelta
from collections import defaultdict
from pprint import pprint

import stopwatch
import dba
import predicts_dbsa
from side import Side
from stat_cont import WinLoss
from common import PredictResult


predicts_db_hnd: Optional[predicts_dbsa.Handle] = None

NUM_HOURS_TO_COMMIT = 4
commit_timer = stopwatch.OverTimer(3600 * NUM_HOURS_TO_COMMIT)


def _commit(bytime: bool = False):
    if bytime and not commit_timer.overtime():
        return
    predicts_db_hnd.commit()
    commit_timer.restart()


def initialize():
    global predicts_db_hnd
    if predicts_db_hnd is None:
        predicts_db_hnd = predicts_dbsa.open_db()


def finalize():
    if predicts_db_hnd is not None:
        _commit(bytime=False)


def write_rejected(match, case_name: str, back_side: Side, reason: str = ""):
    if predicts_db_hnd is None:
        return
    if back_side.is_left():
        back_id, oppo_id = match.first_player.ident, match.second_player.ident
    else:
        back_id, oppo_id = match.second_player.ident, match.first_player.ident
    rec = predicts_dbsa.find_predict_rec_by(
        predicts_db_hnd.session, match.sex, match.date,
        case_name, back_id=back_id, oppo_id=oppo_id)
    if isinstance(rec, predicts_dbsa.PredictRec):
        rec.rejected = 1
        if reason:
            rec.comments += (" " + reason)
        _commit(bytime=False)


def write_bf_live_coef(date: datetime.date, sex: str, fst_id: int, snd_id: int,
                       case_name: str, back_side: Side, bf_live_coef: float):
    if predicts_db_hnd is None:
        return
    if back_side.is_left():
        back_id, oppo_id = fst_id, snd_id
    else:
        back_id, oppo_id = snd_id, fst_id
    rec = predicts_dbsa.find_predict_rec_by(
        predicts_db_hnd.session, sex, date,
        case_name, back_id=back_id, oppo_id=oppo_id)
    if isinstance(rec, predicts_dbsa.PredictRec):
        rec.bf_live_coef = bf_live_coef
        _commit(bytime=False)


def write_predict(match, case_name: str, back_side: Side, proba: float,
                  comments: str = '', rejected: int = -1):
    if predicts_db_hnd is None:
        return
    if back_side.is_left():
        back_id, oppo_id = match.first_player.ident, match.second_player.ident
        back_name = match.first_player.name
        oppo_name = match.second_player.name
        book_start_chance = match.first_player_bet_chance()
    else:
        back_id, oppo_id = match.second_player.ident, match.first_player.ident
        back_name = match.second_player.name
        oppo_name = match.first_player.name
        book_start_chance = 1 - match.first_player_bet_chance()
    rec = predicts_dbsa.PredictRec(
        date=match.date,
        sex=match.sex,
        case_name=case_name,
        tour_name=str(match.tour_name),
        level=match.level,
        surface=match.surface,
        rnd=match.rnd,
        back_id=back_id,
        oppo_id=oppo_id,
        predict_proba=proba,
        predict_result=-1,
        comments=comments,
        back_name=back_name,
        oppo_name=oppo_name,
        book_start_chance=book_start_chance,
        rejected=rejected,
    )
    predicts_db_hnd.insert_obj(rec)
    _commit(bytime=False)


def run_dump_all_to_csv():
    assert predicts_db_hnd is not None, 'not opened db'
    predicts_db_hnd.query_predicts()
    dbfilename = predicts_dbsa.dbfilename()
    csvfilename = os.path.splitext(dbfilename)[0] + '.csv'
    predicts_db_hnd.records_to_csv_file(csvfilename)


def run_stat():
    from bet import FlatLimitedProfiter
    assert predicts_db_hnd is not None, 'not opened db'

    def get_rnd(rnd_db):
        return rnd_db.split(';')[0]

    def get_back_coef():
        if rec.case_name == 'decided_00':
            return max(1.44, round(1 / (rec.predict_proba - 0.05), 2))

    wl = WinLoss()
    surf_wl_dct = defaultdict(WinLoss)
    rnd_wl_dct = defaultdict(WinLoss)
    bystartbook_wl = WinLoss()
    book_start_chances = []
    profiter = FlatLimitedProfiter(start_money=9)
    predicts_db_hnd.query_predicts()
    for rec in predicts_db_hnd.records:
        if (
            rec.predict_result in (0, 1)
            and (not args.sex or (args.sex == rec.sex))
            and (not args.casename or (rec.case_name == args.casename))
            and (args.rejected is None or
                 (args.rejected == 1 and rec.rejected == args.rejected) or
                 (args.rejected == 0 and rec.rejected in (-1, 0))
            )
            and (not args.minproba or (rec.predict_proba >= args.minproba))
            and (not args.maxproba or (rec.predict_proba <= args.maxproba))
            and (args.opener is None
                 or (args.opener and ('OPENER' in rec.comments))
                 or (not args.opener and ('CLOSER' in rec.comments))
            )
        ):
            predict_ok = bool(rec.predict_result)
            wl.hit(iswin=predict_ok)
            surf_wl_dct[rec.surface].hit(iswin=predict_ok)
            rnd_wl_dct[get_rnd(rec.rnd)].hit(iswin=predict_ok)
            book_start_chances.append(rec.book_start_chance)
            back_coef = get_back_coef()
            if back_coef is not None:
                profiter.calculate_bet(
                    coef=back_coef, iswin=bool(rec.back_win_match))

            if (
                rec.back_win_match in (0, 1)
                and abs(rec.book_start_chance - 0.5) > 0.001
            ):
                bystartbook_win = (bool(rec.back_win_match)
                                   is (rec.book_start_chance > 0.5))
                bystartbook_wl.hit(bystartbook_win)

    print(f"stat: {wl} nw: {wl.win_count} nl: {wl.loss_count}")
    pprint([(s, str(w).strip()) for s, w in surf_wl_dct.items()])
    pprint([(r, str(w).strip()) for r, w in rnd_wl_dct.items()])
    avg_book_start_chance = round(sum(book_start_chances)/len(book_start_chances), 3)
    print(f"avg book_start_chance: {avg_book_start_chance} profit: {profiter}")
    print(f"bystartbook wl : {bystartbook_wl}")


def _read_match_score(sex: str, pid1: int, pid2: int, rec_date: date):
    from score import Score

    min_date = rec_date - timedelta(days=1)
    max_date = rec_date + timedelta(days=2)
    sql = """select games.ID1_G, games.ID2_G, games.RESULT_G                     
             from games_{0} AS games
             where games.RESULT_G IS NOT NULL
               and games.DATE_G IS NOT NULL 
               and (
                  (games.ID1_G = {1} and games.ID2_G = {2}) or
                  (games.ID1_G = {2} and games.ID2_G = {1})
               )                   
          """.format(sex, pid1, pid2)
    sql += dba.sql_dates_condition(min_date, max_date, dator='games.DATE_G')
    sql += ";"
    with closing(dba.get_connect().cursor()) as cursor:
        cursor.execute(sql)
        row = cursor.fetchone()
    if row:
        winner_id = row[0]
        loser_id = row[1]
        res_score = Score(text=row[2])
        return winner_id, loser_id, res_score


def run_matchwinner_flag():
    assert predicts_db_hnd is not None, 'not opened db'
    predicts_db_hnd.query_predicts()
    n_ok, n_err = 0, 0
    for rec in predicts_db_hnd.records:
        if rec.back_win_match == -1:
            res = _read_match_score(sex=rec.sex, pid1=rec.back_id, pid2=rec.oppo_id,
                                    rec_date=rec.date)
            if res is None:
                print(f"NOT FOUND RES {rec.date} {rec.back_name} {rec.oppo_name}")
                n_err += 1
            else:
                wid, lid, res_scr = res
                if res_scr.retired:
                    rec.back_win_match = -2
                else:
                    rec.back_win_match = int(rec.back_id == wid)
                n_ok += 1
    if n_ok > 0:
        predicts_db_hnd.commit()
    print(f"n_ok: {n_ok}, n_err: {n_err}")


def run_predictresult_flag():
    assert predicts_db_hnd is not None, 'not opened db'
    predicts_db_hnd.query_predicts()
    n_ok, n_err = 0, 0
    for rec in predicts_db_hnd.records:
        if rec.predict_result == PredictResult.empty:
            res = _read_match_score(sex=rec.sex, pid1=rec.back_id, pid2=rec.oppo_id,
                                    rec_date=rec.date)
            if res is None:
                print(f"NOT FOUND RES {rec.date} {rec.back_name} {rec.oppo_name}")
                n_err += 1
            else:
                wid, lid, res_scr = res
                if res_scr.retired:
                    rec.predict_result = PredictResult.retired.value
                else:
                    if rec.case_name == 'secondset_00':
                        rec.predict_result = int(res_scr.sets_count() > 2)
                    else:
                        rec.predict_result = int(rec.back_id == wid)
                n_ok += 1
    if n_ok > 0:
        predicts_db_hnd.commit()
    print(f"n_ok: {n_ok}, n_err: {n_err}")


def _test_db_add_two_records():
    def add_rec(*args, **kwargs):
        r = predicts_dbsa.PredictRec(*args, **kwargs)
        predicts_db_hnd.insert_obj(r)

    assert predicts_db_hnd is not None, 'not opened db'

    add_rec(
        date=datetime.date(2021, 7, 21),
        sex='wta',
        case_name='decided_00',
        tour_name='Olympics',
        level='main',
        surface='Hard',
        rnd='First',
        back_id=123,
        oppo_id=456,
        predict_proba=0.7,
        predict_result=-1,
        comments='CLOSER',
    )
    add_rec(
        date=datetime.date(2021, 7, 22),
        sex='wta',
        case_name='decided_00',
        tour_name='Olympics',
        level='main',
        surface='Hard',
        rnd='First',
        back_id=234,
        oppo_id=567,
        predict_proba=0.75,
        predict_result=-1,
        comments='OPENER',
    )
    predicts_db_hnd.commit()


if __name__ == "__main__":
    import argparse

    def parse_command_line_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_tocsv", action="store_true")
        parser.add_argument("--run_matchwinner", action="store_true")
        parser.add_argument("--run_predictresult", action="store_true")

        parser.add_argument("--run_stat", action="store_true")
        parser.add_argument("--sex", type=str, default="")
        parser.add_argument("--casename", type=str, default="")
        parser.add_argument("--minproba", type=float, default=None)
        parser.add_argument("--maxproba", type=float, default=None)

        parser.add_argument("--rejected", dest="rejected", action="store_true")
        parser.add_argument("--no-rejected", dest="rejected", action="store_false")

        parser.add_argument("--opener", dest="opener", action="store_true")
        parser.add_argument("--no-opener", dest="opener", action="store_false")
        parser.set_defaults(opener=None, rejected=None)
        return parser.parse_args()

    args = parse_command_line_args()
    if args.run_tocsv:
        initialize()
        run_dump_all_to_csv()
    elif args.run_stat:
        initialize()
        run_stat()
    elif args.run_matchwinner:
        dba.open_connect()
        initialize()
        run_matchwinner_flag()
        dba.close_connect()
    elif args.run_predictresult:
        dba.open_connect()
        initialize()
        run_predictresult_flag()
        dba.close_connect()
