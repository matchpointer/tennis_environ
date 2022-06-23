import copy
from collections import defaultdict
from contextlib import closing
from typing import DefaultDict, List, Tuple

from loguru import logger as log
import tennis_time as tt
import tennis
import bet
from oncourt import dba

SexBettor_YearWeeknum_DbOfferList = DefaultDict[
    Tuple[
        str,  # sex
        int  # bettor_id
    ],
    DefaultDict[
        Tuple[
            int,  # year
            int  # week num
        ],
        List[bet.DbOffer]
    ]
]

dboffers_dict: SexBettor_YearWeeknum_DbOfferList = defaultdict(lambda: defaultdict(list))

PINNACLE_ID = 2
MARATHON_ID = 1

ALL_BETTORS = True
TOTAL_COEFS = False

DEFAULT_BETTOR = PINNACLE_ID
ALTER_BETTOR = MARATHON_ID


def db_offers(sex, date=None, bettor_id=DEFAULT_BETTOR):
    assert sex in ("atp", "wta"), "db_offers: invalid sex {}".format(sex)
    if date is None:
        result = []
        for value in dboffers_dict[(sex, bettor_id)].values():
            result += value
        return result

    year_weeknum = tt.get_year_weeknum(date)
    return dboffers_dict[(sex, bettor_id)][year_weeknum]


def initialize(sex=None, min_date=None, max_date=None, bettor_id=DEFAULT_BETTOR):
    _initialize_impl(sex=sex, min_date=min_date, max_date=max_date, bettor_id=bettor_id)
    if ALL_BETTORS:
        other_bettor_id = (
            ALTER_BETTOR if bettor_id == DEFAULT_BETTOR else DEFAULT_BETTOR
        )
        _initialize_impl(
            sex=sex, min_date=min_date, max_date=max_date, bettor_id=other_bettor_id
        )


def _initialize_impl(sex, min_date, max_date, bettor_id):
    if sex in ("wta", None):
        if ("wta", bettor_id) in dboffers_dict:
            dboffers_dict[("wta", bettor_id)].clear()
        _initialize_sex(
            "wta", min_date=min_date, max_date=max_date, bettor_id=bettor_id
        )
    if sex in ("atp", None):
        if ("atp", bettor_id) in dboffers_dict:
            dboffers_dict[("atp", bettor_id)].clear()
        _initialize_sex(
            "atp", min_date=min_date, max_date=max_date, bettor_id=bettor_id
        )


def _initialize_sex(sex, min_date, max_date, bettor_id):
    """
    Fof MARATHON_ID четверть ставок тоталов с целочисленной линией:
    ROUND(odds.TOTAL, 0) = odds.TOTAL, а условие НЕцелочисл. линии
    можно записать как: ROUND(odds.TOTAL, 0) = (odds.TOTAL + 0.5)
    """
    company = bet.get_company_by_id(bettor_id)
    sql = """select tours.DATE_T, odds.ID_T_O, Rounds.NAME_R,
                    odds.ID1_O, odds.ID2_O, 
                    odds.K1, odds.K2,
                    odds.TOTAL, odds.KTM, odds.KTB
             from Odds_{0} AS odds, Rounds, Tours_{0} AS tours
             where odds.ID_B_O = {1}
               and tours.ID_T = odds.ID_T_O
               and odds.ID_R_O = Rounds.ID_R""".format(
        sex, bettor_id
    )
    sql += dba.sql_dates_condition(min_date, max_date)
    sql += " order by tours.DATE_T;"
    with closing(dba.get_connect().cursor()) as cursor:
        for (
            tour_dt,
            tour_id,
            rnd_name,
            first_player_id,
            second_player_id,
            first_win_coef,
            first_loss_coef,
            total_value,
            total_less_coef,
            total_great_coef,
        ) in cursor.execute(sql):
            date = tour_dt.date() if tour_dt else None
            rnd = tennis.Round(rnd_name)
            dboffer = bet.DbOffer(
                company, tour_id, rnd, first_player_id, second_player_id
            )
            win_coefs = bet.WinCoefs(first_win_coef, first_loss_coef)
            if win_coefs:
                dboffer.win_coefs = win_coefs
            if TOTAL_COEFS:
                total_coefs = bet.TotalCoefs(
                    total_value, total_less_coef, total_great_coef
                )
                if total_coefs:
                    dboffer.total_coefs = total_coefs
            year_weeknum = tt.get_year_weeknum(date)
            dboffers_dict[(sex, bettor_id)][year_weeknum].append(dboffer)


def do_compare_bettors(sex, min_date=None, max_date=None):
    import tournament as trmt
    import stat_cont as st

    def trade(winloss, chances, left_win):
        if chances[0] > chances[1]:
            iswin = left_win
        else:
            iswin = not left_win
        winloss.hit(iswin)

    dba.open_connect()
    tours = list(
        trmt.tours_generator(
            sex,
            min_date=min_date,
            max_date=max_date,
            time_reverse=False,
            with_paired=False,
            with_bets=True,
        )
    )
    initialize(sex=sex, min_date=min_date, max_date=max_date, bettor_id=2)
    dboffers2 = db_offers(sex, bettor_id=2)
    wl = st.WinLoss()
    wl2 = st.WinLoss()
    progress = tt.YearWeekProgress(head=sex)
    eps = 0.022
    for tour in tours:
        for rnd, matches in tour.matches_from_rnd.items():
            for m in matches:
                if (
                    m.paired()
                    or m.score is None
                    or m.score.retired
                    or not m.offer.win_coefs
                ):
                    continue
                offer = bet.DbOffer(
                    m.offer.company,
                    tour.ident,
                    rnd,
                    m.first_player.ident,
                    m.second_player.ident,
                )
                offer.win_coefs = copy.copy(m.offer.win_coefs)
                try:
                    idx = dboffers2.index(offer)
                    offer2 = copy.copy(dboffers2[idx])
                    if offer.first_player_id == offer2.second_player_id:
                        offer2.flip()
                    chances = offer.win_coefs.chances()
                    chances2 = offer2.win_coefs.chances()
                except ValueError:
                    continue
                sets_score = m.score.sets_score(full=True)
                if sets_score[0] == sets_score[1]:
                    continue
                left_win = sets_score[0] > sets_score[1]
                if (
                    abs(chances[0] - chances2[0]) <= eps
                    or abs(chances[1] - chances2[1]) <= eps
                    or abs(max(chances) - 0.5) <= eps
                    or abs(max(chances2) - 0.5) <= eps
                ):
                    continue
                if (max(chances) == chances[0] and max(chances2) == chances2[0]) or (
                    max(chances) == chances[1] and max(chances2) == chances2[1]
                ):
                    continue  # favorit == favorit2
                trade(wl, chances, left_win)
                trade(wl2, chances2, left_win)
        progress.put_tour(tour)
        if progress.new_week_begining:
            print("{}".format(tour.year_weeknum))
    dba.close_connect()
    log.info("wl {}".format(wl))
    log.info("wl2 {}".format(wl2))
