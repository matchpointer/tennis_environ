# -*- coding=utf-8 -*-
import unittest

import common as co
from tie_point_importance import (
    initialize,
    _impdict_from_name,
    PointCell,
    build_importance_cells,
)
import tie_point_importance


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize()

    def test_build_importance_cells_50perc(self):
        eps = 0.01
        cells = build_importance_cells(srv_win_prob=0.5)
        self.assertTrue(PointCell(8, 6, co.RIGHT, 2) in cells)
        self.assertTrue(PointCell(6, 8, co.RIGHT, 2) in cells)
        self.assertTrue(all([c.left_wingame_prob is not None for c in cells]))
        self.assertTrue(
            all([c.importance is not None for c in cells if not c.terminated()])
        )

        # check left_wingame_prob=0.5 for 00, 11, 22, 33, 44, 55, 66, 77
        for i, srv_side, srv_num in (
            (0, co.LEFT, 1),
            (1, co.RIGHT, 2),
            (2, co.LEFT, 2),
            (3, co.RIGHT, 2),
            (4, co.LEFT, 2),
            (5, co.RIGHT, 2),
            (6, co.LEFT, 2),
            (7, co.RIGHT, 2),
        ):
            cell = PointCell(i, i, srv_side, srv_num)
            self.assertTrue(cell in cells)
            obj = cells[cells.index(cell)]
            self.assertTrue(abs(obj.left_wingame_prob - 0.5) < eps)

        # not terminated cells number according Demetris Spanias giagram: 48
        #     and add starting 4 cells: (6,6), (7,6), (6,7), (7,7)
        noterm_cnt = sum([1 for c in cells if not c.terminated()])
        self.assertEqual(48 + 4, noterm_cnt)

    def test_beyond_allfield_cells(self):
        eps = 0.01
        name = "sex=wta,level=main,surface=Hard"

        cell = PointCell(7, 6, co.RIGHT, 1)
        imp = _impdict_from_name[name][cell]
        cell2 = PointCell(11, 10, co.RIGHT, 1)
        imp2 = _impdict_from_name[name][cell2]
        self.assertTrue(abs(imp - imp2) < eps)

    def test_point_cell_equal_method(self):
        cell1 = PointCell(6, 6, co.LEFT, 2)
        cell2 = PointCell(6, 6, co.LEFT, 2)
        self.assertEqual(cell1, cell2)

        s = {cell1}
        s.update({cell2})
        self.assertEqual(1, len(s))


if __name__ == "__main__":
    tie_point_importance.test_mode = True
    unittest.main()
