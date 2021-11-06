import os
import unittest
import datetime

import dba
import file_utils as fu
import tennis_time as tt
import tennis
import tour_name
from tournament import best_of_five, tours_generator, tours_write_file


class BestOfFiveTest(unittest.TestCase):
    def test_best_of_five(self):
        isbo5 = best_of_five(
            date=datetime.date(2020, 9, 3),
            sex="atp",
            tour_name=tour_name.TourName("U.S. Open"),
            level="gs",
            rnd=tennis.Round("Third"),
        )
        self.assertTrue(isbo5)

        isbo5 = best_of_five(
            date=datetime.date(2018, 9, 3),
            sex="atp",
            tour_name=tour_name.TourName("Davis Cup"),
            level="teamworld",
            rnd=tennis.Round("Robin"),
        )
        self.assertTrue(isbo5)

        isbo5 = best_of_five(
            date=datetime.date(2020, 9, 3),
            sex="atp",
            tour_name=tour_name.TourName("ATP Cup"),
            level="teamworld",
            rnd=tennis.Round("Robin"),
        )
        self.assertFalse(isbo5)

        isbo5 = best_of_five(
            date=datetime.date(2020, 9, 3),
            sex="atp",
            tour_name=tour_name.TourName("Davis Cup"),
            level="teamworld",
            rnd=tennis.Round("Robin"),
        )
        self.assertFalse(isbo5)

        isbo5 = best_of_five(
            date=datetime.date(2020, 9, 3),
            sex="atp",
            tour_name="Davis Cup",
            level="teamworld",
            rnd=tennis.Round("Robin"),
        )
        self.assertFalse(isbo5)

        isbo5 = best_of_five(
            date=datetime.date(2018, 9, 3),
            sex="atp",
            tour_name="Davis Cup",
            level="team",
            rnd=tennis.Round("Robin"),
        )
        self.assertFalse(isbo5)


class ToursGeneratorTest(unittest.TestCase):
    @staticmethod
    def matches_ordering(tour):
        for rnd in tour.matches_from_rnd.keys():
            tour.matches_from_rnd[rnd].sort(
                key=lambda m: m.first_player.name + m.second_player.name
            )

    @staticmethod
    def tours_matches_ordering(tours):
        for tour in tours:
            ToursGeneratorTest.matches_ordering(tour)

    def test_common(self):
        def write_tours(
            sex,
            fname,
            todaymode,
            min_date,
            max_date,
            with_paired,
            with_mix,
            rnd_detailing,
        ):
            tours = list(
                tours_generator(
                    sex,
                    todaymode=todaymode,
                    min_date=min_date,
                    max_date=max_date,
                    with_paired=with_paired,
                    with_mix=with_mix,
                    rnd_detailing=rnd_detailing,
                )
            )
            if tours:
                dirname = "./tmp/debug"
                fu.ensure_folder(dirname)
                self.tours_matches_ordering(tours)
                tours_write_file(tours, os.path.join(dirname, fname))
                self.assertTrue(len(tours) > 0)

        sex = "atp"
        min_date = tt.past_monday_date(datetime.date.today())
        max_date = min_date + datetime.timedelta(days=7)
        write_tours(
            sex,
            "tours.txt",
            todaymode=False,
            min_date=min_date,
            max_date=max_date,
            with_paired=True,
            with_mix=False,
            rnd_detailing=True,
        )

        write_tours(
            sex,
            "today_tours.txt",
            todaymode=True,
            min_date=min_date,
            max_date=max_date,
            with_paired=True,
            with_mix=False,
            rnd_detailing=True,
        )


if __name__ == "__main__":
    dba.open_connect()
    unittest.main()
    dba.close_connect()
