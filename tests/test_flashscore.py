import datetime
import os
import time
import unittest
from collections import defaultdict

import dba
import common as co
import common_wdriver
import log
import oncourt_players
import tennis_time as tt
import tournament_misc as trmt_misc
import weeked_tours

from flashscore import (
    itf_wta_money_tourname,
    itf_atp_money_tourname,
    TourInfoFlashscore,
    make_events,
    initialize,
    split_ontwo_enclosed
)
from live import MatchStatus


class SplitOntwoEnclosedTest(unittest.TestCase):
    def test_split_ontwo_enclosed(self):
        text = "ab(c)-d(ef)"
        p1, p2 = split_ontwo_enclosed(text, delim_op="(", delim_cl=")")
        self.assertTrue(p1 == "ab(c)-d" and p2 == "ef")


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
    def test_wta_melbourne(self):
        part_one = "WTA"
        part_two = "Melbourne (Gippsland Trophy) (Australia), hard"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("Melbourne", obj.tour_name.name)
        self.assertEqual("Gippsland Trophy", obj.tour_name.desc_name)
        self.assertEqual("Hard", obj.surface)
        self.assertEqual("Australia", obj.country)

    def test_atp_quimper2(self):
        part_one = "CHALLENGER MEN"
        part_two = "Quimper 2 (descript) (France) - Qualification, hard (indoor)"
        obj = TourInfoFlashscore(part_one, part_two)
        self.assertEqual("Quimper", obj.tour_name.name)
        self.assertEqual("Quimper 2", str(obj.tour_name))
        self.assertEqual(2, obj.tour_name.number)
        self.assertTrue(obj.qualification)
        self.assertEqual("chal", obj.level)
        self.assertEqual("descript", obj.tour_name.desc_name)
        self.assertEqual("Carpet", obj.surface)
        self.assertEqual("France", obj.country)


class FlashscoreSiteTest(unittest.TestCase):
    fsdrv = None
    page = None
    actual_file_name = "../log/test_flashscore_actual_page.html"
    match_status = MatchStatus.finished  # other matches will be skip

    @classmethod
    def is_actual_file(cls):
        if os.path.isfile(cls.actual_file_name):
            n_secs = os.path.getctime(cls.actual_file_name)
            struct_time = time.localtime(n_secs)
            file_create_date = datetime.date(
                struct_time.tm_year, struct_time.tm_mon, struct_time.tm_mday
            )
            return file_create_date == datetime.date.today()

    @classmethod
    def read_page_from_file(cls):
        from file_utils import read

        cls.page = read(cls.actual_file_name)

    @classmethod
    def setUpClass(cls):
        if not cls.is_actual_file():
            cls.fsdrv = common_wdriver.wdriver(company_name="FS", faked=False)
            cls.fsdrv.start()
            time.sleep(5)
            cls.fsdrv.save_page(cls.actual_file_name)

        cls.read_page_from_file()
        # here possible initialize_players_cache:
        # initialize_players_cache(cls.page, match_status=cls.match_status)

    @classmethod
    def tearDownClass(cls):
        if cls.fsdrv:
            cls.fsdrv.stop()

    def test_make_and_log_events(self):
        def count_defined_matches(evt):
            n_ok, n_fail = 0, 0
            for match in evt.matches:
                if (
                    match.first_player
                    and match.first_player.ident
                    and match.second_player
                    and match.second_player.ident
                ):
                    n_ok += 1
                else:
                    n_fail += 1
            return n_ok, n_fail

        events = make_events(
            webpage=self.page,
            skip_levels=defaultdict(list),
            match_status=self.match_status,
        )
        for event in events:
            event.define_players(company_name="FS")
            event.define_level()

            n_ok, n_fail = count_defined_matches(event)
            msg = "{} tour_id: {} n_ok: {} n_fail: {}".format(
                event.tour_name, event.tour_id, n_ok, n_fail
            )
            log.info(msg)
            if (n_ok + n_fail) > 0:
                self.assertTrue(n_ok > 0)

        trmt_misc.log_events(
            events,
            head="test_make_and_log_events {}".format(self.match_status),
            extended=True,
        )


if __name__ == "__main__":
    log.initialize(
        co.logname(__file__, test=True), file_level="info", console_level="info"
    )
    dba.open_connect()
    oncourt_players.initialize(yearsnum=1.2)
    min_date = tt.past_monday_date(datetime.date.today()) - datetime.timedelta(days=7)
    weeked_tours.initialize_sex(
        "wta", min_date=min_date, max_date=None, with_today=True, with_bets=True
    )
    weeked_tours.initialize_sex(
        "atp", min_date=min_date, max_date=None, with_today=True, with_bets=True
    )
    initialize()

    unittest.main()
    dba.close_connect()


# class TourInfoFlashscoreTeamsTest(unittest.TestCase):
#     def test_atp_cup(self):
#         part_one = "ATP - TEAMS"
#         part_two = "ATP Cup - World Group (World) - Qualification"
#         obj = TourInfoFlashscore(part_one, part_two)
#         self.assertEqual("ATP Cup", str(obj.tour_name))
#
#     def test_init(self):
#         part_one = "WTA - SINGLES"
#         part_two = "Fed Cup - World Group (World) - Qualification"
#         obj = TourInfoFlashscore(part_one, part_two)
#         self.assertEqual("Fed Cup", str(obj.tour_name))


# ----- Davis Cup, not top (not World Group) event header:

# <div class="event__header">
#   <div class="event__check"></div>
#   <div class="icon--flag event__title fl_3473163">TEAMS - MEN:&nbsp;
#     <span class="event__title--name" title="Davis Cup - Group II (World)">Davis Cup - Group II (World)</span>
#     <span class="toggleMyLeague 2_9992_QVQVyHNa"></span>
#   </div>
#   <a href="#" class="event__info active"></a>
#   <div class="event__expander icon--expander collapse" title="Hide all matches of this league!"></div>
# </div>

# Davis Cup, not top (not World Group) team result:

# <div id="g_2_SdN0YX5C" title="Click for match detail!" class="event__match event__match--twoLine">
#   <div class="event__check"/>
#   <div class="event__stage">
#     <div class="event__stage--block">After
#       <br/>
# day 1</div>
#   </div>
#   <div class="event__participant event__participant--home">Thailand</div>
#   <div class="event__participant event__participant--away">Philippines</div>
#   <div class="event__score event__score--home">2</div>
#   <div class="event__score event__score--away">0</div>
#   <div class="event__icon icon--info event__icon--slim"/>
# </div>
