import datetime
import unittest

import common as co
import log
import dba
import score as sc
from tennis import Round, Surface, Player, Match, HeadToHead


class RoundTest(unittest.TestCase):
    def test_cmp(self):
        self.assertTrue(Round("Rubber 3") < "Rubber 5")
        self.assertTrue(Round("Rubber 3") > "Rubber 2")
        self.assertTrue("Rubber 3" < Round("Rubber 5"))
        self.assertTrue("Rubber 3" > Round("Rubber 2"))

        self.assertTrue("Rubber 3" == Round("Rubber 3"))
        self.assertTrue(Round("Rubber 3") == "Rubber 3")

        self.assertFalse(Round("Rubber 2").qualification())

        self.assertTrue(Round("First", details="vs qual") == "First")


class SurfaceTest(unittest.TestCase):
    def test_cmp(self):
        self.assertTrue("Clay" == Surface("Clay"))
        self.assertTrue(Surface("Carpet") == "Carpet")


class HeadToHeadTest(unittest.TestCase):
    def test_ito_kukushkin(self):
        plr1 = Player(ident=8313, name="Tatsuma Ito", cou="JPN")
        plr2 = Player(ident=9043, name="Mikhail Kukushkin", cou="KAZ")
        match = Match(
            first_player=plr1,
            second_player=plr2,
            score=sc.Score("4-6 7-6(3) 7-6(6)"),
            rnd=Round("Second"),
            date=datetime.date(2017, 2, 1),
        )
        h2hdir = HeadToHead(sex="atp", match=match, completed_only=True).direct()
        self.assertEqual(h2hdir, None)

    @unittest.skip("actual only was 2019,10,18 and recent h2h match was 2019,9,26")
    def test_recent_winner(self):
        plr1 = Player(ident=6590, name="Cagla Buyukakcay", cou="TUR")
        plr2 = Player(ident=8951, name="Veronica Cepede Royg", cou="PAR")
        match = Match(first_player=plr1, second_player=plr2,
                      score=None, rnd=Round('Second'),
                      date=datetime.date(2019, 10, 18))
        h2h = HeadToHead(sex='wta', match=match, completed_only=True)
        recent_winner_id = h2h.recently_won_player_id()
        self.assertEqual(recent_winner_id, plr1.ident)


if __name__ == '__main__':
    import doctest

    log.initialize(co.logname(__file__, test=True), "debug", None)
    dba.open_connect()
    doctest.testmod()
    unittest.main()
    dba.close_connect()
