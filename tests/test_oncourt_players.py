import unittest

from loguru import logger as log
import common as co
import dba
from oncourt_players import initialize, players


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dba.open_connect()
        initialize(yearsnum=8)

    def test_min_players(self):
        plrs = players("atp")
        self.assertGreaterEqual(len(plrs), 1000)

    def test_wta_count_hand_players(self):
        plrs = players("wta")
        cnt = sum([1 for p in plrs if p.lefty is not None])
        print(f"wta n_hand={cnt}")
        self.assertTrue(cnt >= 400)

    def test_wta_kerber_lefty(self):
        plrs = players("wta")
        results = [p for p in plrs if p.ident == 4076]
        self.assertEqual(len(results), 1)
        if len(results) == 1:
            kerber = results[0]
            self.assertTrue('Kerber' in kerber.name)
            self.assertTrue(kerber.lefty)

    def test_atp_exist_lefty_player(self):
        plrs = players("atp")
        lefty_cnt = sum([1 for p in plrs if p.lefty])
        self.assertTrue(lefty_cnt >= 5)
        our_plrs = [p for p in plrs if p.name == "Gerald Melzer" and p.cou == "AUT"]
        self.assertTrue(len(our_plrs) == 1)
        if our_plrs:
            self.assertTrue(our_plrs[0].lefty)


if __name__ == '__main__':
    unittest.main()
