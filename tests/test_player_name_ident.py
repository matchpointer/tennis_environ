import unittest

import dba
import common as co
import log
import oncourt_players
from tennis import Player
from player_name_ident import AbbrName, identify_player


class AbbrNameTest(unittest.TestCase):
    def setUp(self):
        self.players_wta = oncourt_players.players("wta")
        self.players_atp = oncourt_players.players("atp")

    def test_find(self):
        players = [
            Player(name="Daniel Baraldi Marcos"),
            Player(name="Dekel Bar"),
        ]
        self.assertEqual(
            AbbrName("Bar D.").find_player(players), Player(name="Dekel Bar")
        )

        players = [
            Player(name="Daniel Baraldi Marcos"),
            Player(name="Dekel Bar"),
            Player(name="Dekel Bar Foo"),
        ]
        self.assertEqual(
            AbbrName("Bar D.").find_player(players), Player(name="Dekel Bar")
        )

        players = [Player(name="Dekel Bar Foo")]
        self.assertEqual(AbbrName("Bar D.").find_player(players), players[0])

        players = [Player(name="Nils Brinkman")]
        self.assertEqual(AbbrName("Brinkman N.").find_player(players), players[0])

        players = [Player(name="Michelle Larcher De Brito")]
        self.assertEqual(
            AbbrName("Larcher de Brito M.").find_player(players), players[0]
        )

    @unittest.skip(reason='not resolved by algo')
    def test_wta_identify_player_ambigious(self):
        """problem 2019.6.25 case, instead correct result plr got: 'Jiaxi Lu' pid=23358.
        It is ambigious problem: too short display name 'Lu J.' can represent more 1 names.
        Also from CHN: 'Jia-Jing Lu' have pid=6744, 'Jia Xiang Lu' pid=7832,
                       'Jing-Jing Lu' pid=7833, 'Jing-Jue Lu' pid=10773.
        My first look give problem in AbbrName.regexp() not designed for such case
        More radical treatment is to read full plr-name in flashscore_match page.
        """
        # ambigious: wrong answer gives:  'Jiaxi Lu'
        plr = identify_player('FS', 'wta', 'Lu J.', cou='CHN')
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, 'Jia-Jing Lu')  # pid=6744

    def test_wta_identify_player(self):
        plr = identify_player("FS", "wta", "Rybakina E.", cou="KAZ")
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Elena Rybakina")

        plr = identify_player("FS", "wta", "Chirico L.", cou="USA")
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Louisa Chirico")

        plr = identify_player("FS", "wta", "Fernandez L. A.", cou="CAN")
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Leylah Annie Fernandez")

        plr = identify_player("FS", "wta", "Marino R.", cou="CAN")
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Rebecca Marino")

        plr: Player = identify_player("FS", "wta", "Wang Xin.", cou="CHN")
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Xin Yu Wang")
            self.assertEqual(plr.disp_name('betfair'), "Xinyu Wang")

    def test_find_wang_in_oncourt_players(self):
        plr: Player = co.find_first(
            self.players_wta,
            lambda p: p.cou == 'CHN' and p.disp_name("flashscore") == "Wang Xin."
        )
        self.assertTrue(plr is not None)
        if plr is not None:
            self.assertTrue(plr.ident is not None)
            self.assertEqual(plr.name, "Xin Yu Wang")
            self.assertEqual(plr.disp_name('betfair'), "Xinyu Wang")

    def test_wta_find(self):
        self.assertEqual(
            AbbrName("Van Uytvanck A.").find_player(self.players_wta).name,
            "Alison Van Uytvanck",
        )
        self.assertEqual(
            AbbrName("McHale C.").find_player(self.players_wta).name, "Christina Mchale"
        )

        self.assertEqual(
            AbbrName("Begu I.-C.").find_player(self.players_wta).name,
            "Irina-Camelia Begu",
        )
        self.assertEqual(
            AbbrName("Begu I. -C.").find_player(self.players_wta).name,
            "Irina-Camelia Begu",
        )
        self.assertEqual(
            AbbrName("Pliskova Ka.").find_player(self.players_wta).name,
            "Karolina Pliskova",
        )

    @unittest.skip(reason='not resolved by algo')
    def test_atp_find_ambigious_1(self):
        # ---- ambigious china problem: also exist 'Ze Zhang'
        self.assertEqual(AbbrName('Zhang Z.').find_player(self.players_atp).name,
                         'Zhizhen Zhang')

    def test_atp_find(self):
        self.assertEqual(
            AbbrName("Brinkman N.").find_player(self.players_atp).name, "Nils Brinkman"
        )
        self.assertEqual(
            AbbrName("De Schepper K.").find_player(self.players_atp).name,
            "Kenny de Schepper",
        )
        self.assertEqual(
            AbbrName("Nam Ji Sung").find_player(self.players_atp).name, "Ji Sung Nam"
        )
        self.assertEqual(
            AbbrName("Lu Yen-Hsun").find_player(self.players_atp).name, "Yen-Hsun Lu"
        )
        self.assertEqual(
            AbbrName("Schwartzman D.S.").find_player(self.players_atp).name,
            "Diego Sebastian Schwartzman",
        )

    def test_resemble(self):
        self.assertFalse(AbbrName("Renamar Sh.").resemble("Sherazad Reix"))
        self.assertTrue(
            AbbrName("Larcher de Brito M.").resemble("Michelle Larcher De Brito")
        )
        self.assertTrue(AbbrName("Brinkman N.").resemble("Nils Brinkman"))
        self.assertTrue(AbbrName("Cox D.").resemble("Daniel Cox"))
        self.assertTrue(AbbrName("de Bakker T.").resemble("Thiemo de Bakker"))
        self.assertTrue(AbbrName("Ferrer-Suarez I.").resemble("Ines Ferrer-Suarez"))
        self.assertTrue(AbbrName("Nedovyesov A.").resemble("Aleksandr Nedovesov"))

    def test_parts(self):
        def check_impl(abbr, last_parts, init_parts):
            abbrname = AbbrName(abbr)
            self.assertEqual(last_parts, abbrname.last_parts())
            self.assertEqual(init_parts, abbrname.init_parts())

        check_impl("Van De Velde B.", ["Van", "De", "Velde"], ["B"])
        check_impl("Makarova E.", ["Makarova"], ["E"])
        check_impl("Some E.-N.", ["Some"], ["E", "N"])
        check_impl("Some E.N.", ["Some"], ["E", "N"])
        check_impl("Some Any E.", ["Some", "Any"], ["E"])
        check_impl("Some Any", ["Some", "Any"], [])
        check_impl("Some And.", ["Some"], ["And"])
        check_impl("Bar D.", ["Bar"], ["D"])
        check_impl("Some A", ["Some"], ["A"])
        check_impl("Some A. K", ["Some"], ["A", "K"])
        check_impl("Some Any E. V.", ["Some", "Any"], ["E", "V"])
        check_impl("Some Any E.V.", ["Some", "Any"], ["E", "V"])
        check_impl("Some Any E.-V.", ["Some", "Any"], ["E", "V"])
        check_impl("Some Any E.-V", ["Some", "Any"], ["E", "V"])
        check_impl("Some Any R. - S.", ["Some", "Any"], ["R", "S"])
        check_impl("Hsieh Su-Wei", ["Hsieh", "Su Wei"], [])


def setUpModule():
    if not dba.initialized():
        dba.open_connect()
    oncourt_players.initialize()


if __name__ == "__main__":
    log.initialize(
        co.logname(__file__, test=True), file_level="info", console_level="info"
    )
    if not dba.initialized():
        dba.open_connect()
    unittest.main()
    dba.close_connect()
