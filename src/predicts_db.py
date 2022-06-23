# -*- coding=utf-8 -*-
"""
утилиты для использования базы прогнозов классификаторов.
Структура базы в predicts_dbsa.
Простые сценарии из командной строки  (--run) (см parse_command_line_args).

Сейчас схема записи такова:
    - классификаторы создают запись (write_predict)
    - live_alerts добавляют (если есть факторы против) в нее rejected=1 с comments
       (write_rejected)
    - betfair_svc_impl добавляют в нее coefs
       (write_bf_live_coef, write_bf_live_coef_matched)
"""
from sys import exit
import os
import datetime
from typing import Optional
from contextlib import closing
from datetime import date, timedelta
from collections import defaultdict
from pprint import pprint

import stopwatch
from oncourt import dba
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


def cut_comments(comments):
    return comments[:min(predicts_dbsa.MAX_COMMENTS_LEN, len(comments))]


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
            rec.comments = cut_comments(rec.comments + " " + reason)
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


def write_bf_live_coef_matched(date: datetime.date, sex: str, fst_id: int, snd_id: int,
                               case_name: str, back_side: Side,
                               bf_live_coef_matched: float):
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
        rec.bf_live_coef_matched = bf_live_coef_matched
        _commit(bytime=False)


def write_predict(match, case_name: str, back_side: Side, proba: float,
                  comments: str = '', rejected: int = -1, clf_hash: str = None):
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
        comments=cut_comments(comments),
        back_name=back_name,
        oppo_name=oppo_name,
        book_start_chance=book_start_chance,
        rejected=rejected,
        clf_hash=clf_hash,
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

    def get_prof_coef_result():
        nonlocal set2_loss_mched_cnt
        if rec.case_name == 'decided_00':
            return rec.bf_live_coef_matched, bool(rec.predict_result)
        elif rec.case_name == 'secondset_00':
            coef = 1 + (rec.bf_live_coef_matched - 1) * 0.4
            iswin = bool(rec.predict_result)
            if not iswin:
                set2_loss_mched_cnt += 1
                win_one_of = 4
                if (set2_loss_mched_cnt % win_one_of) == 0:
                    # assume win 1 of win_one_of by hedging, our plr looks ok
                    iswin = True
            return coef, iswin
        return None, None

    def do_profiter():
        if rec.bf_live_coef_matched is not None:
            prof_coef, prof_res = get_prof_coef_result()
            if prof_coef is not None:
                profiter.calculate_bet(coef=prof_coef, iswin=prof_res)

    set2_loss_mched_cnt = 0
    wl = WinLoss()
    surf_wl_dct = defaultdict(WinLoss)
    rnd_wl_dct = defaultdict(WinLoss)
    bystartbook_wl = WinLoss()
    book_start_chances = []
    profiter = FlatLimitedProfiter(start_money=9)
    predicts_db_hnd.query_predicts()
    min_date = None
    if args.min_date is not None:
        min_date = datetime.datetime.strptime(args.min_date, "%Y-%m-%d").date()
    for rec in predicts_db_hnd.records:
        if (
            rec.predict_result in (0, 1)
            and (not args.sex or (args.sex == rec.sex))
            and (not args.comment or (args.comment in rec.comments))
            and (not args.clf_hash
                 or (rec.clf_hash is not None and args.clf_hash in rec.clf_hash))
            and (min_date is None or rec.date >= min_date)
            and (not args.casename or (rec.case_name.startswith(args.casename)))
            and (not args.level or (rec.level == args.level))
            and (not args.surface or (rec.surface == args.surface))
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
            if args.stat_use_matchwinner_only:
                predict_ok = bool(rec.back_win_match)
            # print(f"{rec.back_name} {rec.oppo_name} -> {predict_ok}")
            wl.hit(iswin=predict_ok)
            surf_wl_dct[rec.surface].hit(iswin=predict_ok)
            rnd_wl_dct[get_rnd(rec.rnd)].hit(iswin=predict_ok)
            book_start_chances.append(rec.book_start_chance)
            do_profiter()
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
    if book_start_chances:
        avg_book_start_chance = round(sum(book_start_chances)/len(book_start_chances), 3)
    else:
        avg_book_start_chance = None
    print(f"avg book_start_chance: {avg_book_start_chance}")
    print(f"profit by matched coefs: {profiter}")
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


def run_matchresult_flags():
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
                win_pid, lose_pid, res_scr = res
                n_full_sets = res_scr.sets_count(full=True)
                if res_scr.retired:
                    rec.back_win_match = -2
                else:
                    rec.back_win_match = int(rec.back_id == win_pid)

                if rec.predict_result == PredictResult.empty:
                    if rec.case_name.startswith('open'):
                        trg_set_idx = 0
                    elif rec.case_name.startswith('secondset'):
                        trg_set_idx = 1
                    elif rec.case_name.startswith('decided'):
                        if rec.sex == 'wta':
                            trg_set_idx = 2
                        else:
                            bo5 = (rec.tour_name in ('Australian Open', 'australian-open',
                                                     'French Open', 'french-open',
                                                     'Wimbledon', 'wimbledon',
                                                     'U.S. Open', 'us-open')
                                   and rec.rnd not in ('q-First', 'q-Second', 'Qualifying')
                                   )
                            trg_set_idx = 4 if bo5 else 2
                    else:
                        n_err += 1
                        print(f"unexpected case_name: {rec.case_name}")
                        continue
                    trg_set = res_scr[trg_set_idx] if trg_set_idx < n_full_sets else None
                    if trg_set is None:
                        if res_scr.retired:
                            rec.predict_result = -2
                        else:
                            print(f"NOT calc {rec.case_name} {rec.date} {rec.back_name}"
                                  f" {rec.oppo_name} scr: {res_scr}"
                                  f" trg_set_idx: {trg_set_idx}")
                            n_err += 1  # ok if match is suspended
                    else:
                        is_set_win_left = trg_set[0] >= trg_set[1]
                        is_back_left = rec.back_id == win_pid
                        rec.predict_result = int(is_set_win_left is is_back_left)
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
        date=datetime.date(2031, 7, 21),
        sex='wta',
        case_name='decided_00',
        tour_name='Olympics',
        level='main',
        surface='Hard',
        rnd='First',
        back_id=123,
        oppo_id=456,
        back_name='test-backname',
        oppo_name='test-opponame',
        book_start_chance=0.6,
        predict_proba=0.7,
        predict_result=-1,
        comments='0123456789abcdefghij0123456789ABCDEFGHIJ',
    )
    add_rec(
        date=datetime.date(2031, 7, 22),
        sex='wta',
        case_name='decided_00',
        tour_name='Olympics',
        level='main',
        surface='Hard',
        rnd='First',
        back_id=234,
        oppo_id=567,
        back_name='test-backname',
        oppo_name='test-opponame',
        book_start_chance=0.6,
        predict_proba=0.75,
        predict_result=-1,
        comments='0123456789abcdefghij0123456789ABCDEFGHIJ',
    )
    predicts_db_hnd.commit()


if __name__ == "__main__":
    import argparse

    def parse_command_line_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_testadd", action="store_true")
        parser.add_argument("--run_tocsv", action="store_true")
        parser.add_argument("--run_matchresult", action="store_true")

        parser.add_argument("--run_stat", action="store_true")
        parser.add_argument("--sex", type=str, default="")
        parser.add_argument("--casename", type=str, default="")
        parser.add_argument("--level", type=str, default="")
        parser.add_argument("--surface", type=str, default="")
        parser.add_argument("--minproba", type=float, default=None)
        parser.add_argument("--maxproba", type=float, default=None)
        parser.add_argument("--min_date", type=str, default=None)
        parser.add_argument("--comment", type=str, default="")
        parser.add_argument("--clf_hash", type=str, default="")

        parser.add_argument("--rejected", dest="rejected", action="store_true")
        parser.add_argument("--no-rejected", dest="rejected", action="store_false")

        parser.add_argument("--opener", dest="opener", action="store_true")
        parser.add_argument("--no-opener", dest="opener", action="store_false")

        parser.add_argument("--stat-use-matchwinner-only", dest="stat_use_matchwinner_only",
                            action="store_true")

        parser.set_defaults(opener=None, rejected=None)
        return parser.parse_args()

    args = parse_command_line_args()
    if args.run_testadd:
        print('test_add start')
        initialize()
        _test_db_add_two_records()
        print('test_add finish')
        exit(0)

    if args.run_tocsv:
        initialize()
        run_dump_all_to_csv()
    elif args.run_stat:
        initialize()
        run_stat()
    elif args.run_matchresult:
        dba.open_connect()
        initialize()
        run_matchresult_flags()
        dba.close_connect()
