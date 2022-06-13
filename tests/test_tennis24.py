# -*- coding=utf-8 -*-
import unittest

from tennis24 import (
    itf_wta_money_tourname,
    itf_atp_money_tourname,
    TourInfoFlashscore
)


class ITFtournameTest(unittest.TestCase):
    def test_wta_money_tourname(self):
        money_tourname = itf_wta_money_tourname("W15 Antalya (Turkey)")
        self.assertTrue(money_tourname is not None)
        if money_tourname:
            self.assertEqual(money_tourname[0], 15000)
            self.assertEqual(money_tourname[1], "Antalya")

    def test_wta_money_tourname_num(self):
        money_tourname = itf_wta_money_tourname("W15 Antalya 2 (Turkey)")
        self.assertTrue(money_tourname is not None)
        if money_tourname:
            self.assertEqual(money_tourname[0], 15000)
            self.assertEqual(money_tourname[1], "Antalya 2")

    def test_atp_money_tourname(self):
        money_tourname = itf_atp_money_tourname("M15 Bagnoles-de-l'Orne (France)")
        self.assertTrue(money_tourname is not None)
        if money_tourname:
            self.assertEqual(money_tourname[0], 15000)
            self.assertEqual(money_tourname[1], "Bagnoles-de-l'Orne")


class TourInfoFlashscoreSimpleTest(unittest.TestCase):
    def test_itf_wta_orlando(self):
        # <div class="icon--flag event__title fl_3473164">
        #   <div class="event__titleBox">
        #       <span class="event__title--type">ITF WOMEN - SINGLES</span>
        #       <span class="event__title--name" title="W60 Orlando, FL 2 (USA), hard">W60 Orlando, FL 2 (USA), hard</span>
        #   </div>
        # </div>
        part_one = "ITF WOMEN - SINGLES"
        part_two = "W60 Orlando, FL 2 (USA), hard"

        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("orlando", obj.tour_name.name)
        self.assertEqual("", obj.tour_name.desc_name)
        self.assertEqual(None, obj.tour_name.number)  # 2 после запятой НЕ считаем номером
        self.assertEqual("Hard", obj.surface)
        self.assertEqual("USA", obj.country)

    def test_wta_melbourne(self):
        part_one = "WTA"
        part_two = "Melbourne (Gippsland Trophy 2) (Australia), hard"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("melbourne", obj.tour_name.name)
        self.assertEqual("gippsland-trophy", obj.tour_name.desc_name)
        self.assertEqual(obj.tour_name.number, 2)
        self.assertEqual("Hard", obj.surface)
        self.assertEqual("Australia", obj.country)

    def test_wta_finals_playoffs(self):
        part_one = "WTA"
        part_two = "Finals - Guadalajara (World) - Play Offs, hard"
        obj = TourInfoFlashscore(part_one, part_two)
        print(f"obj: {obj}")
        self.assertTrue("finals" in obj.tour_name)
        self.assertEqual("Hard", obj.surface)

    def test_atp_quimper2(self):
        part_one = "CHALLENGER MEN"
        part_two = "Quimper 2 (descript) (France) - Qualification, hard (indoor)"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("quimper", obj.tour_name.name)
        self.assertEqual("quimper 2", str(obj.tour_name))
        self.assertEqual(2, obj.tour_name.number)
        self.assertTrue(obj.qualification)
        self.assertEqual("chal", obj.level)
        self.assertEqual("descript", obj.tour_name.desc_name)
        self.assertEqual("Carpet", obj.surface)
        self.assertEqual("France", obj.country)


@unittest.skip(reason='not actual')
class TourInfoTeamsTest(unittest.TestCase):
    def test_atp_cup(self):
        part_one = "ATP - TEAMS"
        part_two = "ATP Cup - World Group (World) - Qualification"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("atp-cup", str(obj.tour_name))

    def test_init(self):
        part_one = "WTA - SINGLES"
        part_two = "Fed Cup - World Group (World) - Qualification"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("fed-cup", str(obj.tour_name))


if __name__ == '__main__':
    unittest.main()
