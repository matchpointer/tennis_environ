# -*- coding=utf-8 -*-
import unittest
from datetime import date

from pprint import pprint
import dba
from atfirst_after import after_retired_results


class AfterRetiredTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not dba.initialized():
            dba.open_connect()

    def test_rybakina_2022_02_21(self):
        # в этот день Рыбакина выходила на корт после снятия 11 дней назад
        pid = 36558
        oppo_pid = 17614
        appear_date = date(2022, 2, 21)
        player_result_list = after_retired_results(
            {oppo_pid, pid}, 'wta', appear_date)
        self.assertTrue(len(player_result_list) > 0)
        pprint(player_result_list)


if __name__ == '__main__':
    unittest.main()
