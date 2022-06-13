# -*- coding=utf-8 -*-
from collections import namedtuple

Market = namedtuple("Market",
                    ['type', 'name'])
winmatch_market = Market(type='MATCH_ODDS', name='Match Odds')
winset1_market = Market(type='SET_WINNER', name='Set 1 Winner')
winset2_market = Market(type='SET_WINNER', name='Set 2 Winner')


def round_coef(coef: float):
    if 1.01 <= coef < 2:
        odds_inc = 0.01
    elif 2 <= coef < 3:
        odds_inc = 0.02
    elif 3 <= coef < 4:
        odds_inc = 0.05
    elif 4 <= coef < 6:
        odds_inc = 0.1
    elif 6 <= coef < 10:
        odds_inc = 0.2
    elif 10 <= coef < 20:
        odds_inc = 0.5
    elif 20 <= coef < 30:
        odds_inc = 1
    elif 30 <= coef < 50:
        odds_inc = 2
    elif 50 <= coef < 100:
        odds_inc = 5
    elif 100 <= coef < 1000:
        odds_inc = 10
    else:
        return None
    if round(coef + odds_inc, 2) <= 1000:
        return round(
            round(coef / odds_inc, 0) * odds_inc,
            2
        )
    else:
        return 1000.


def get_lay_liability(lay_coef: float, lay_size: float):
    if lay_size is not None and lay_coef is not None and lay_coef >= 1.01:
        return round(lay_size * (lay_coef - 1.0), 2)


def get_lay_size(lay_coef: float, lay_liability: float):
    if lay_coef is not None and lay_liability is not None and lay_coef >= 1.01:
        return round(lay_liability / (lay_coef - 1), 2)


def get_lay_coef(lay_size: float, lay_liability: float):
    if lay_size is not None and lay_liability is not None and lay_size >= 0.01:
        return round_coef(1 + (lay_liability / lay_size))


def get_ticks(coef_from: float, coef_to: float):
    """ если это актуально, то надо это согласовать с логикой ф-ии round_coef """
    if coef_from is not None and coef_to is not None:
        return int(round(coef_to * 100)) - int(round(coef_from * 100))


_MIN_BET_SIZE = 3.0
