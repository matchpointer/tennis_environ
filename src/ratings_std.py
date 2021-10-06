from collections import defaultdict
import datetime
import unittest
from contextlib import closing

import log
import common as co
import dba
import tennis

""" access to oncourt ratings (since 2003.01.06)
"""

# sex->
#  OrderedDict{date(order by desc) -> defaultdict(player_id -> (rating_pos, rating_pts))}
_sex_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: (None, None))))


def clear(sex):
    if sex is None:
        _sex_dict.clear()
    else:
        key_list = list(_sex_dict.keys())
        for key in key_list:
            if key == sex:
                _sex_dict[sex].clear()
                del _sex_dict[sex]


def initialize(sex=None, min_date=None):
    if sex in ("wta", None):
        if "wta" in _sex_dict:
            _sex_dict["wta"].clear()
        __initialize_sex("wta", min_date=min_date)

    if sex in ("atp", None):
        if "atp" in _sex_dict:
            _sex_dict["atp"].clear()
        __initialize_sex("atp", min_date=min_date)


def initialized():
    return len(_sex_dict) > 0


def get_rank(sex, player_id, date):
    return _get_pos_pts(sex, player_id, date)[0]


def get_pts(sex, player_id, date):
    return _get_pos_pts(sex, player_id, date)[1]


def _get_pos_pts(sex, player_id, date):
    valid_date, data_from_pid = __date_entry(sex, date)
    if valid_date is None:
        return None, None
    return data_from_pid[player_id]


default_min_diff = 260


def compare_rank(first_rating, second_rating, min_diff=default_min_diff):
    """
    Cравнение двух рейтинговых позиций с допуском min_diff.
    дает  LT если first позиция выше (значение first меньше),
          EQ если равны с допуском min_diff,
          GT если first ниже (значение first больше).
    """
    left_rating = first_rating if first_rating is not None else 950  # max_pos is 900
    right_rating = second_rating if second_rating is not None else 950  # max_pos is 900
    return co.value_compare(left_rating, right_rating, min_diff)


class RatingsTest(unittest.TestCase):
    @staticmethod
    def is_all_handred_even(iterable):
        return all(((n % 100) == 0 for n in iterable))

    @staticmethod
    def write_pts_csv(sex, date, pts_list):
        filename = "./{}_{:4d}_{:02d}_{:02d}_pts.csv".format(
            sex, date.year, date.month, date.day
        )
        with open(filename, "w") as fh:
            line = ",".join([str(i) for i in pts_list])
            fh.write(line + "\n")

    def test_write_pts_csv(self):
        dates = [
            datetime.date(2010, 6, 7),
            datetime.date(2014, 1, 13),
            datetime.date(2015, 5, 18),
            datetime.date(2019, 9, 30),
        ]
        for date in dates:
            for sex in ("wta", "atp"):
                pts_list = top_players_pts_list(sex, date, top=500)
                is_100_even = self.is_all_handred_even(pts_list)
                if sex == "wta":
                    self.assertTrue(is_100_even)
                    self.write_pts_csv(sex, date, [n // 100 for n in pts_list])
                else:
                    print("atp is_100_even", is_100_even)
                    self.write_pts_csv(sex, date, pts_list)
                self.assertTrue(len(pts_list) >= 500)

    def test_common_wta(self):
        sex = "wta"
        sex_dict = _sex_dict[sex]
        dates = list(sex_dict.keys())
        n_rtg = 0
        for date in dates:
            n_rtg += len(sex_dict[date])
        self.assertTrue(len(dates) >= 40)

    def test_get_rank_wta(self):
        plr = tennis.Player(ident=14364, name="Angeliki Kairi", cou="GRE")
        pos = get_rank("wta", plr.ident, datetime.date(2014, 7, 21))
        self.assertEqual(pos, None)

        plr = tennis.Player(ident=14010, name="Beatrice Cedermark", cou="SWE")
        pos = get_rank("wta", plr.ident, datetime.date(2014, 7, 21))
        self.assertEqual(pos, 723)

        plr = tennis.Player(ident=431, name="Vera Zvonareva", cou="RUS")
        pos = get_rank("wta", plr.ident, datetime.date(2012, 5, 28))
        self.assertEqual(pos, 11)
        pts = get_pts("wta", plr.ident, datetime.date(2012, 5, 28))
        self.assertEqual(pts, 344000)

        pos = get_rank("wta", plr.ident, datetime.date(2003, 1, 13))
        self.assertEqual(pos, 43)

        plr = tennis.Player(ident=7574, name="Petra Martic", cou="CRO")
        pos = get_rank("wta", plr.ident, datetime.date(2020, 8, 3))
        self.assertEqual(pos, 15)

    def test_get_rank_atp(self):
        date = datetime.date(2014, 7, 21)

        plr = tennis.Player(ident=21812, name="Rodrigo Arus", cou="URU")
        pos = get_rank("atp", plr.ident, date)
        self.assertEqual(pos, None)

        plr = tennis.Player(ident=13962, name="Valentin Florez", cou="ARG")
        pos = get_rank("atp", plr.ident, date)
        self.assertEqual(pos, 689)


def __date_entry(sex, date):
    """return date, data_from_pid"""
    ratings_dct = _sex_dict[sex]
    data_from_pid = ratings_dct.get(date, None)
    if data_from_pid is not None:
        return date, data_from_pid
    for d in ratings_dct:  # dates are ordered by desc
        if d < date:
            return d, ratings_dct[d]
    return None, None


def __initialize_sex(sex, min_date=None):
    sql = """select DATE_R, ID_P_R, POS_R, POINT_R 
             from Ratings_{} """.format(
        sex
    )
    dates_cond = dba.sql_dates_condition(min_date, max_date=None, dator="DATE_R")
    if dates_cond:
        sql += """
               where 1 = 1 {} """.format(
            dates_cond
        )
    sql += """
           order by DATE_R desc;"""
    with closing(dba.get_connect().cursor()) as cursor:
        for (dtime, player_id, rating_pos, rating_pts) in cursor.execute(sql):
            _sex_dict[sex][dtime.date()][player_id] = (rating_pos, rating_pts)


def top_players_id_list(sex, top):
    """at max date which initialized"""
    from operator import itemgetter

    maxdate = list(_sex_dict[sex].keys())[0]
    pos_from_pid = _sex_dict[sex][maxdate]
    pos_pid_lst = [(pos, pid) for pid, (pos, _) in pos_from_pid.items() if pos <= top]
    pos_pid_lst.sort(key=itemgetter(0))
    return [pid for (_, pid) in pos_pid_lst]


def top_players_pts_list(sex, date, top=500):
    """:returns list of pts"""
    pos_from_pid = _sex_dict[sex][date]
    pts_lst = [
        pts for _pid, (pos, pts) in pos_from_pid.items() if pos <= top and pts > 0
    ]
    pts_lst.sort(reverse=False)
    return pts_lst


def setUpModule():
    initialize(sex=None, min_date=datetime.date(2003, 1, 6))


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", None)
    dba.open_connect()
    unittest.main()
    dba.close_connect()
