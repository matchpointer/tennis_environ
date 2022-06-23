# -*- coding=utf-8 -*-
import unittest

from oncourt import dba
from decided_win_by_two_sets_stat import ENABLE_SOFT_LEVELS, read_scores_dict


class ReadFilesTest(unittest.TestCase):
    def test_read_score_dict(self):
        for soft_level in ENABLE_SOFT_LEVELS:
            dct = read_scores_dict("wta", soft_level)
            self.assertEqual(len(dct), 49)

            if soft_level == "main":
                sval = dct[((4, 6), (6, 0))]
                self.assertTrue(0.6 < sval.value < 0.8)
                self.assertTrue(120 < sval.size < 200)


if __name__ == '__main__':
    unittest.main()
    dba.close_connect()

