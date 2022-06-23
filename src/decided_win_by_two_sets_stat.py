# -*- coding=utf-8 -*-
import os
import sys
from collections import defaultdict, namedtuple
import argparse
from contextlib import closing

from loguru import logger as log
import cfg_dir
import common as co
import file_utils as fu
import dict_tools
import lev
import stat_cont as st
import score as sc
import tennis
import tennis_time as tt
from oncourt import dba, read_db_helper
import ratings_std
import feature
from clf_common import RANK_STD_BOTH_ABOVE, RANK_STD_MAX_DIF
import pandemia

ASPECT = "decided_win_by_set2_winner"

DISABLE_LEVELS = (lev.team, lev.future, lev.junior)
ENABLE_SOFT_LEVELS = (lev.main, lev.chal, "qual")

MIN_SIZE_DEFAULT = 5

# (sex, soft_level) -> ASC ordered dict{(year, weeknum) ->
#           dict{(s1_score, s2_score) -> WinLoss}}
# here s2_score[0] > s2_score[1], s1_score[0] < s1_score[1]
data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(st.WinLoss)))


def add_feature(features, sex, soft_level, date, set1_score, set2_score):
    """ добавить в features суммарный рез-т из памяти по неделям, предшествующим date """
    wl = decided_winloss_by_set2_winner(sex, soft_level, date, set1_score, set2_score)
    if wl.size <= MIN_SIZE_DEFAULT:
        features.append(feature.RigidFeature(name=ASPECT))
        return False
    features.append(feature.RigidFeature(name=ASPECT, value=wl.ratio))
    return True


def add_empty_feature(features):
    """uses for best_of_five matches (not implemented)"""
    features.append(feature.RigidFeature(name=ASPECT))


def decided_winloss_by_set2_winner(sex, soft_level, date, set1_score, set2_score):
    """ получить суммарный результат из памяти по неделям, предшествующим date """
    set1, set2 = _make_set2winner_orient(set1_score, set2_score)
    in_ywn = tt.get_year_weeknum(date)
    wl_res = st.WinLoss()
    dct = data_dict[(sex, soft_level)]
    for ywn, scr_dct in dct.items():
        if ywn >= in_ywn:
            break
        wl_res += scr_dct[(set1, set2)]
    return wl_res


def read_decided_winloss_by_set2_winner(sex: str, soft_level: str, set1_score, set2_score):
    if sex is None or soft_level:
        log.error("bad sex {}, or soft_level {}".format(sex, soft_level))
        return None
    set1, set2 = _make_set2winner_orient(set1_score, set2_score)
    score_dict = read_scores_dict(sex, "main" if soft_level == "team" else soft_level)
    if not score_dict:
        log.error("fail score_dict sex: {}, soft_level {}".format(sex, soft_level))
        return None
    return score_dict.get((set1, set2))


def read_decided_win_ratio(match, side):
    s1_scr, s2_scr = match.score[0], match.score[1]
    set2winer_dec_win = read_decided_winloss_by_set2_winner(
        match.sex, match.soft_level, s1_scr, s2_scr
    )
    if set2winer_dec_win is None:
        log.error(
            "fail read_decided_win_ratio for slevel '{}' side {} {}".format(
                match.soft_level, side, match
            )
        )
        return None
    if side == co.LEFT:
        sider_win_set2 = s2_scr[0] > s2_scr[1]
    else:
        sider_win_set2 = s2_scr[1] > s2_scr[0]
    set2winer_dec_win_val = set2winer_dec_win.ratio
    if not sider_win_set2 and set2winer_dec_win_val is not None:
        set2winer_dec_win_val = 1.0 - set2winer_dec_win_val
    return set2winer_dec_win_val


def _make_set2winner_orient(set1_score, set2_score):
    if set2_score[0] > set2_score[1]:
        set1, set2 = set1_score, set2_score
    else:
        set1, set2 = (set1_score[1], set1_score[0]), (set2_score[1], set2_score[0])
    return set1, set2


def fill_data(sex, min_date=None, max_date=None):
    """ ratings_std уже д.б. инициализирован для переданного интервала времени """
    if not dba.initialized():
        dba.open_connect()

    if sex in ("wta", None):
        _fill_results_sex(
            "wta",
            max_rating=RANK_STD_BOTH_ABOVE,
            max_rating_dif=RANK_STD_MAX_DIF,
            min_date=min_date,
            max_date=max_date,
        )
    if sex in ("atp", None):
        _fill_results_sex(
            "atp",
            max_rating=RANK_STD_BOTH_ABOVE,
            max_rating_dif=RANK_STD_MAX_DIF,
            min_date=min_date,
            max_date=max_date,
        )


def _fill_results_sex(
    sex, max_rating, max_rating_dif, min_date=None, max_date=None
):
    sql = """select tours.ID_T, tours.DATE_T, tours.NAME_T, tours.RANK_T, tours.PRIZE_T, 
                   games.ID_R_G, games.RESULT_G, games.ID1_G, games.ID2_G
             from Tours_{0} AS tours, games_{0} AS games, Players_{0} AS fst_plr
             where games.ID_T_G = tours.ID_T 
               and games.ID1_G = fst_plr.ID_P
               and (tours.NAME_T Not Like '%juniors%')
               and (fst_plr.NAME_P Not Like '%/%') """.format(
        sex
    )
    sql += dba.sql_dates_condition(min_date, max_date)
    sql += " order by tours.DATE_T;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (
            tour_id,
            tour_dt,
            tour_name,
            db_rank,
            db_money,
            rnd_id,
            score_txt,
            fst_id,
            snd_id,
        ) in cursor.execute(sql):
            date = tour_dt.date() if tour_dt else None
            if date is None:
                raise co.TennisScoreError("none date {}".format(tour_name))
            if pandemia.exhib_tour_id(sex) == tour_id:
                continue
            if not score_txt:
                continue
            scr = sc.Score(score_txt)
            if scr.retired:
                continue
            sets_count = scr.sets_count(full=True)
            if sets_count != 3 or scr.best_of_five():
                continue
            set3_score = scr[2]
            if set3_score[0] < set3_score[1]:
                raise co.TennisScoreError("right winner unexpected {}".format(scr))
            money = read_db_helper.get_money(db_money)
            rank = None if db_rank is None else int(db_rank)
            if rank is None:
                log.error(
                    "none rank date: {} scr: {} name: {}".format(date, scr, tour_name)
                )
            if not isinstance(rank, int):
                raise co.TennisError(
                    "not int rank '{}' date: {} scr: {} name: {}".format(
                        rank, date, scr, tour_name
                    )
                )
            rawname, level = read_db_helper.get_name_level(
                sex, tour_name.strip(), rank, money, date
            )
            if level in DISABLE_LEVELS:
                continue
            if not level:
                raise co.TennisError(
                    "empty level date: {} scr: {} name: {}".format(date, scr, tour_name)
                )
            rnd = tennis.Round.from_oncourt_id(rnd_id)
            soft_level = lev.soft_level(level, rnd)
            if not soft_level:
                raise co.TennisError(
                    "none soft_level date: {} scr: {} name: {}".format(
                        date, scr, tour_name
                    )
                )
            mdata = _get_match_data(
                sex, date, fst_id, snd_id, scr, max_rating, max_rating_dif
            )
            if mdata is not None:
                past_monday = tt.past_monday_date(date)
                ywn = tt.get_year_weeknum(past_monday)
                data_dict[(sex, soft_level)][ywn][
                    (mdata.set1_score, mdata.set2_score)
                ].hit(mdata.decided_win)


MatchData = namedtuple("MatchData", "set1_score set2_score decided_win")


def _get_match_data(sex, date, fst_id, snd_id, scr, max_rating, max_rating_dif):
    def admit_rtg_dif():
        win_rtg, loss_rtg = fst_rtg, snd_rtg
        if loss_rtg > (win_rtg + max_rating_dif):
            return False  # strong underdog fail or strong favorite prevail
        return True

    fst_rtg = ratings_std.get_rank(sex, fst_id, date)
    if fst_rtg is None or fst_rtg > max_rating:
        return None
    snd_rtg = ratings_std.get_rank(sex, snd_id, date)
    if snd_rtg is None or snd_rtg > max_rating:
        return None
    if not admit_rtg_dif():
        return None
    set1, set2 = scr[0], scr[1]
    if set1[0] < set1[1] and set2[0] > set2[1]:
        # ok score orientation as set2 winner
        return MatchData(set1_score=set1, set2_score=set2, decided_win=True)
    elif set1[0] > set1[1] and set2[0] < set2[1]:
        # make reverse score orientation (opponent is set2 winner)
        return MatchData(
            set1_score=(set1[1], set1[0]),
            set2_score=(set2[1], set2[0]),
            decided_win=False,
        )


# --------------------- work with data files ----------------------------


def get_history_days(sex):
    """for read data from already prepared files"""
    if sex == "atp":
        return int(365 * 20)
    elif sex == "wta":
        return int(365 * 20)
    raise co.TennisError("bad sex {}".format(sex))


class StatMaker(object):
    """for write data into files"""

    def __init__(self, sex):
        self.sex = sex

    def process_all(self):
        for soft_level in ENABLE_SOFT_LEVELS:
            rpt_level_dict = self.summarize_dict(soft_level)
            self.output(soft_level, rpt_level_dict)

    def output(self, soft_level, rpt_level_dict):
        dirname = generic_dirname(self.sex)
        fu.ensure_folder(dirname)
        filename = generic_filename(self.sex, soft_level)
        lst = [(s1, s2, wl) for (s1, s2), wl in rpt_level_dict.items()]
        lst.sort(key=lambda i: i[2].ratio, reverse=True)
        with open(filename, "w") as fhandle_out:
            for (s1, s2, wl) in lst:
                fhandle_out.write("({}, {})__{}\n".format(s1, s2, str(wl)))

    def summarize_dict(self, soft_level):
        dct = defaultdict(st.WinLoss)
        for ywn in data_dict[(self.sex, soft_level)]:
            for (s1, s2), wl in data_dict[(self.sex, soft_level)][ywn].items():
                dct[(s1, s2)] += wl
        return dct


def generic_dirname(sex):
    return "{}/{}_stat".format(cfg_dir.stat_misc_dir(sex), ASPECT)


def generic_filename(sex, level):
    return "{}/avg_{}.txt".format(generic_dirname(sex), level)


def read_scores_dict(sex, level):
    """:return dict: (set1_score, set2_score) -> WinLoss"""
    if (sex, level) in read_scores_dict.cache:
        return read_scores_dict.cache[(sex, level)]
    filename = generic_filename(sex, level)
    if not os.path.isfile(filename):
        log.error("FILE {} NOT EXIST".format(filename))
        return None
    dct = dict_tools.load(filename, valuefun=st.WinLoss.from_text)
    if dct:
        read_scores_dict.cache[(sex, level)] = dct
        return dct
    return None


read_scores_dict.cache = dict()  # (sex, level) -> scores_dict


def do_stat():
    try:
        dba.open_connect()
        msg = "sex: {} max_rtg: {}  max_rtg_dif: {}".format(
            args.sex, args.max_rating, args.max_rating_dif
        )
        log.info(__file__ + " started {}".format(msg))
        ratings_std.initialize(sex=args.sex, min_date=None)
        fill_data(args.sex, min_date=None, max_date=None)
        maker = StatMaker(args.sex)
        maker.process_all()
        log.info(__file__ + " done {}".format(ASPECT))

        dba.close_connect()
        log.info(__file__ + " finished sex: {}".format(args.sex))
        return 0
    except Exception as err:
        log.exception("{0} [{1}]".format(err, err.__class__.__name__))
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--min_size", type=int, default=MIN_SIZE_DEFAULT)
    parser.add_argument("--max_rating", type=int, default=RANK_STD_BOTH_ABOVE)
    parser.add_argument("--max_rating_dif", type=int, default=RANK_STD_MAX_DIF)
    parser.add_argument("--sex", choices=["wta", "atp"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.stat:
        log.add('../log/decided_win_by_two_sets_stat.log', level='INFO',
                rotation='10:00', compression='zip')

        sys.exit(do_stat())
