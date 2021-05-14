# -*- coding: utf-8 -*-

import common as co
from detailed_score import point_score_error
import miss_points
import point_importance as pimportance
import tie_point_importance as tpimportance


AVG_DEFAULT_GAME_VALUE = 1.5
MAX_GAME_VALUE = 9.0
MAX_TIEBREAK_VALUE = 16.0


def initialize():
    pimportance.initialize()
    tpimportance.initialize()


def detailed_game_values(sex, level, surface, qualification, det_game, misspoints):
    """выдает пару рез-тов для левого и правого игроков,
    отражающих их игру на важных очках. Оценки м.б. не симметричны.
    большее зн-ние не всегда у выигравшего гейм"""
    if (
        not det_game.valid
        or point_score_error(det_game.error)
        or det_game.extra_points_error()
    ):
        return None
    left_sum, right_sum = 0.0, 0.0
    for point in det_game:
        if det_game.tiebreak:
            imp = tpimportance.point_importance(
                sex,
                level,
                surface,
                qualification,
                point.num_score(),
                co.side(point.serve()),
            )
        else:
            imp = pimportance.point_importance(
                sex, level, surface, qualification, point.text_score()
            )
        if point.win(left=True):
            left_sum += imp
        else:
            right_sum += imp

    min_sum = min(left_sum, right_sum)
    left_val = left_sum - min_sum
    right_val = right_sum - min_sum
    max_value = MAX_TIEBREAK_VALUE if det_game.tiebreak else MAX_GAME_VALUE
    if left_val > max_value:
        left_val = max_value
    if right_val > max_value:
        right_val = max_value

    left_miss_penalty, right_miss_penalty = miss_points.get_penalty(
        sex, misspoints, scope_predicate=lambda s: s != co.GAME
    )
    return left_val - left_miss_penalty, right_val - right_miss_penalty
