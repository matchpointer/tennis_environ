# -*- coding=utf-8 -*-
r""" module for calculate tie point importance.
     www.princeton.edu/~dixitak/Teaching/IntroductoryGameTheory/Precepts/Prcpt03.pdf
     where article 'Illustration of Rollback in a Decision Problem,
                    and Dynamic Games of Competition'
"""

import copy
from collections import defaultdict

import common as co
import score as sc
import matchstat


MAX_NORMALIZED_COUNT = 7
test_mode = False

# name -> dict(PointCell -> importance)
_impdict_from_name = defaultdict(dict)


def point_importance(sex, level, surface, qualification, ingame, srv_side):
    """вернем важность (float) розыгрыша при счете ingame (2-num tuple).
    Считаем, что это одинаковая важность и для подающего и для принимающего.
    при инициализации данные предполагают, что открыватель tiebreak слева
    """
    srv_num = sc.tie_serve_number_at(ingame)  # in [1, 2]
    if sc.tie_opener_side(ingame, srv_side).is_right():  # need flip
        cell = PointCell(ingame[1], ingame[0], srv_side.fliped(), srv_num)
    else:
        cell = PointCell(ingame[0], ingame[1], srv_side, srv_num)

    if level == "teamworld":
        level = "main"
    if qualification:
        name = "sex=%s,level=q-%s,surface=%s" % (sex, level, surface)
        if name in _impdict_from_name:
            return _impdict_from_name[name][cell]
    name = "sex=%s,level=%s,surface=%s" % (sex, level, surface)
    return _impdict_from_name[name][cell]


def initialize():
    if not initialize.field_cells:
        initialize.field_cells = list(make_allfield_cells())

    for sex in ("wta", "atp"):
        dct = matchstat.generic_result_dict(sex, "service_win")
        # here dct: StructKey -> probability as SizedValue
        assert bool(dct), "bad initialize in point_importance_by_name %s" % sex
        for key, svalue in dct.items():
            if len(key) == 2:
                name = ("sex=%s," % sex) + str(key)
                _prepare(name, svalue.value)
                if not test_mode:
                    _prepare_importance_normalize(name)


initialize.field_cells = None


def _prepare(name, srv_win_point_prob):
    if name in _impdict_from_name:
        return
    dct = {}
    for cell in build_importance_cells(srv_win_point_prob):
        if not cell.terminated():
            assert cell.importance is not None, "badimp cell %s %s" % (cell, name)
            dct[copy.copy(cell)] = cell.importance
    _impdict_from_name[name] = dct


def _prepare_importance_normalize(name):
    """normalize imp values so max_imp_value = 1"""
    dct = _impdict_from_name[name]
    if dct:
        max_imp = max(dct.values())
        if max_imp > 0.001:
            norm_coef = 1.0 / max_imp
            for key in dct:
                dct[key] *= norm_coef


class PointCell(object):
    """состояние при счете left_count, right_count,
    подает srv_side подачу номер srv_num (1-st or 2-nd).
    """

    def __init__(self, left_count, right_count, srv_side, srv_num):
        self.left_count = left_count
        self.right_count = right_count
        self.srv_side = srv_side
        self.srv_num = srv_num

        assert srv_num in (1, 2), "bad srv_num %d" % srv_num
        assert srv_side in (co.LEFT, co.RIGHT), "bad srv_side %s" % srv_side

        self.importance = None
        wingame_side = self.terminated()
        if wingame_side == co.LEFT:
            self.left_wingame_prob = 1.0
        elif wingame_side == co.RIGHT:
            self.left_wingame_prob = 0.0
        else:
            self.left_wingame_prob = None

    def terminated(self):
        """None - не завершено, side - сторона победителя гейма"""
        if abs(self.left_count - self.right_count) <= 1:
            return None
        if (7 <= self.left_count) and (self.left_count > self.right_count):
            return co.LEFT
        if (7 <= self.right_count) and (self.right_count > self.left_count):
            return co.RIGHT

    @staticmethod
    def normalized_counts(fst_count, snd_count):
        while max(fst_count, snd_count) >= (MAX_NORMALIZED_COUNT + 1):
            if min(fst_count, snd_count) < 2:
                break
            fst_count -= 2
            snd_count -= 2
        return fst_count, snd_count

    def __hash__(self):
        norm_counts = self.normalized_counts(self.left_count, self.right_count)
        return hash((norm_counts[0], norm_counts[1], self.srv_side, self.srv_num))

    def __str__(self):
        if self.importance is None:
            imp_text = ""
        else:
            imp_text = " I: {0:.2f}".format(self.importance)
        return "({},{}) {} {} P:{}{}".format(
            self.left_count,
            self.right_count,
            self.srv_side,
            self.srv_num,
            self.left_wingame_prob,
            imp_text,
        )

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__,
            self.left_count,
            self.right_count,
            self.srv_side,
            self.srv_num,
        )

    def __eq__(self, other):
        eq_counts = self.normalized_counts(
            self.left_count, self.right_count
        ) == self.normalized_counts(other.left_count, other.right_count)
        return (
            eq_counts
            and self.srv_side == other.srv_side
            and self.srv_num == other.srv_num
        )

    def step_forward(self, winpoint_side):
        """шаг вперед из self при условии что розыгрыш в
        self выиграл winpoint_side. Вернем None или новый PointCell
        """
        if self.terminated():
            return None

        if winpoint_side.is_left():
            next_left_count = self.left_count + 1
            next_right_count = self.right_count
        else:
            next_left_count = self.left_count
            next_right_count = self.right_count + 1

        if self == PointCell(0, 0, co.LEFT, 1):
            next_srv_num = 1
            next_srv_side = co.RIGHT
        else:
            if self.srv_num == 1:
                next_srv_num = 2
                next_srv_side = self.srv_side
            else:
                next_srv_num = 1
                next_srv_side = self.srv_side.fliped()
        return PointCell(next_left_count, next_right_count, next_srv_side, next_srv_num)


def make_allfield_cells():
    def step_advance():
        adv_cells = set()
        for cell in result_cells:
            if cell.terminated():
                continue
            for winpoint_side in (co.LEFT, co.RIGHT):
                new_cell = cell.step_forward(winpoint_side)
                if new_cell is None or new_cell in adv_cells:
                    continue
                max_count = max(new_cell.left_count, new_cell.right_count)
                if max_count > (MAX_NORMALIZED_COUNT + 1):
                    continue
                if (
                    max_count == (MAX_NORMALIZED_COUNT + 1)
                    and not new_cell.terminated()
                ):
                    continue
                adv_cells.add(new_cell)
        prev_len = len(result_cells)
        result_cells.update(adv_cells)
        return prev_len < len(result_cells)

    result_cells = {PointCell(0, 0, co.LEFT, 1)}
    grow = True
    while grow:
        grow = step_advance()
    return result_cells | {PointCell(8, 6, co.RIGHT, 2), PointCell(6, 8, co.RIGHT, 2)}


def split_start_cells(field_cells, srv_win_prob):
    """assume field_cells is list of all possible cells.
    return (start_cells, nostart_cells)
    """
    # according (13) at page 247 P66 = 0.5 from
    # Paul K. Newton, Joseph B. Keller "Probability of Winning at Tennis. Theory and Data"
    c66L2_idx = field_cells.index(PointCell(6, 6, co.LEFT, 2))
    c66L2 = field_cells.pop(c66L2_idx)
    c66L2.left_wingame_prob = 0.5

    c77R2_idx = field_cells.index(PointCell(7, 7, co.RIGHT, 2))
    c77R2 = field_cells.pop(c77R2_idx)
    c77R2.left_wingame_prob = 0.5

    c76R1_idx = field_cells.index(PointCell(7, 6, co.RIGHT, 1))
    c76R1 = field_cells.pop(c76R1_idx)
    c76R1.left_wingame_prob = 1.0 - srv_win_prob * 0.5

    c67R1_idx = field_cells.index(PointCell(6, 7, co.RIGHT, 1))
    c67R1 = field_cells.pop(c67R1_idx)
    c67R1.left_wingame_prob = (1.0 - srv_win_prob) * 0.5
    return ([c66L2, c77R2, c76R1, c67R1], field_cells)


def split_term_cells(cells):
    """return (terminated_cells, noterminated_cells)"""
    term_cells = []
    cells_len = len(cells)
    for i in reversed(range(cells_len)):
        if cells[i].terminated():
            term_cells.append(copy.deepcopy(cells[i]))
            del cells[i]
    return term_cells, cells


def make_prob_cell(cell, prob_cells, srv_win_prob):
    """define prob in cell. return True if success, False if failed"""
    assert cell.left_wingame_prob is None, "already probed cell %s" % cell
    cell_next_win = cell.step_forward(winpoint_side=co.LEFT)
    if cell_next_win not in prob_cells:
        return False
    next_win_idx = prob_cells.index(cell_next_win)

    cell_next_loss = cell.step_forward(winpoint_side=co.RIGHT)
    if cell_next_loss not in prob_cells:
        return False
    next_loss_idx = prob_cells.index(cell_next_loss)

    if cell.srv_side.is_left():
        prob = (
            srv_win_prob * prob_cells[next_win_idx].left_wingame_prob
            + (1.0 - srv_win_prob) * prob_cells[next_loss_idx].left_wingame_prob
        )
    else:
        prob = (1.0 - srv_win_prob) * prob_cells[
            next_win_idx
        ].left_wingame_prob + srv_win_prob * prob_cells[next_loss_idx].left_wingame_prob
    cell.left_wingame_prob = prob
    return True


def make_prob_cells(cells, prob_cells, srv_win_prob):
    """transfer probed items from cells into prob_cells"""
    is_transfer = True
    while is_transfer:
        is_transfer = False
        cells_len = len(cells)
        for i in reversed(range(cells_len)):
            if make_prob_cell(cells[i], prob_cells, srv_win_prob):
                prob_cells.append(copy.deepcopy(cells[i]))
                del cells[i]
                is_transfer = True


def make_importance_cells(cells):
    for cell in cells:
        if cell.terminated():
            continue
        cell_next_win = cell.step_forward(winpoint_side=co.LEFT)
        next_win_idx = cells.index(cell_next_win)
        cell_next_loss = cell.step_forward(winpoint_side=co.RIGHT)
        next_loss_idx = cells.index(cell_next_loss)
        cell.importance = (
            cells[next_win_idx].left_wingame_prob
            - cells[next_loss_idx].left_wingame_prob
        )


def build_importance_cells(srv_win_prob):
    field_cells = copy.deepcopy(initialize.field_cells)
    start_cells, nostart_cells = split_start_cells(field_cells, srv_win_prob)
    term_cells, cells = split_term_cells(nostart_cells)
    prob_cells = start_cells + term_cells
    make_prob_cells(cells, prob_cells, srv_win_prob)
    assert not cells, "notprobedcells_len: %d" % len(cells)
    make_importance_cells(prob_cells)
    return prob_cells


def print_cells(cells):
    print(("-----(%d)------" % len(cells)))
    for cell in cells:
        print(("%s" % cell))


if __name__ == "__main__":
    import doctest

    test_mode = True
    doctest.testmod()
