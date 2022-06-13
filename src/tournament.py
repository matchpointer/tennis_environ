# -*- coding=utf-8 -*-
import datetime
import copy
from collections import defaultdict, namedtuple
from functools import cmp_to_key
from contextlib import closing

import common as co
from loguru import logger as log
import file_utils as fu
from surf import make_surf
import dba
import oncourt_db
import qual_seeds
import tennis_time as tt
import score as sc
import tennis
import stat_cont as st
import report_line as rl
from detailed_score_dbsa import MatchRec
import tour_name
import pandemia
import lev


def best_of_five(date, sex, tourname, level, rnd):
    if sex == "wta":
        return False
    elif sex == "atp":
        if level == lev.gs:
            if rnd is not None and rnd.main_draw():
                return True
            if "wimbledon" in tourname and rnd == "Qualifying":
                return True
            return None
        elif level == lev.teamworld:
            if date is not None and date.year >= 2019:
                return False  # new best of three format was applied
            return "atp-cup" not in tourname
    return False


def best_of_five2(sex, level, is_qual):
    """less precise than above function
    (atp-Wim-qualifying and 'ATP Cup' is not recognized here)"""
    if sex == "wta":
        return False
    elif sex == "atp":
        if level == lev.gs:
            if is_qual is False:
                return True
        elif level == lev.teamworld:
            return True
    return False


class Tournament:
    def __init__(self, ident, name, sex=None, surface=None, rank=None,
                 date=None, money=None, cou=None):
        self.ident = ident
        self.sex = sex
        self.surface = surface
        self.money = oncourt_db.get_money(money)
        self.rank = None if rank is None else int(rank)
        self.date = Tournament.endyear_date_correction(date)
        self.raw_name, self.level = oncourt_db.get_name_level(
            sex, name.strip(), self.rank, self.money, self.date
        )
        self.cou = cou.strip()
        self.name = tour_name.TourName.from_oncourt(self.raw_name)
        self.matches_from_rnd = defaultdict(list)
        # T - потерян квал, F - есть квал, N - нет его и не знаем потерян ли:
        self.qual_lost = None

    def __str__(self):
        return "{} {} {} ({}) {} {}".format(
            self.sex,
            self.date,
            self.name,
            self.cou or "unk-cou",
            self.level,
            self.surface,
        )

    def best_of_five(self, rnd):
        return best_of_five(self.date, self.sex, self.name, self.level, rnd)

    @property
    def year_weeknum(self):
        return tt.get_year_weeknum(self.date)

    def has_qualification(self):
        return any(
            [
                (rnd.qualification() or rnd.pre_qualification())
                for rnd in self.matches_from_rnd
            ]
        )

    def has_main_draw(self):
        return any(
            [
                not (rnd.qualification() or rnd.pre_qualification())
                for rnd in self.matches_from_rnd
            ]
        )

    def has_round(self, rnd, paired=None):
        """если paired is None то неважна парность-непарность"""
        if rnd not in list(self.matches_from_rnd.keys()):
            return False
        if paired is None:
            return len(self.matches_from_rnd[rnd]) > 0
        for m in self.matches_from_rnd[rnd]:
            match_is_paired = m.paired()
            if paired and match_is_paired:
                return True
            elif not paired and not match_is_paired:
                return True
        return False

    def players_id_list(self, match_splited=False):
        result = []
        for matches in list(self.matches_from_rnd.values()):
            for m in matches:
                if not m.paired():
                    first_id, second_id = None, None
                    if m.first_player and m.first_player.ident:
                        first_id = m.first_player.ident
                    if m.second_player and m.second_player.ident:
                        second_id = m.second_player.ident
                    if match_splited and (first_id or second_id):
                        result.append((first_id, second_id))
                    if not match_splited and first_id:
                        result.append(first_id)
                    if not match_splited and second_id:
                        result.append(second_id)
        return result

    def players_list(self, match_splited=False):
        result = []
        for matches in list(self.matches_from_rnd.values()):
            for m in matches:
                if not m.paired():
                    first, second = None, None
                    if m.first_player:
                        first = m.first_player
                    if m.second_player:
                        second = m.second_player
                    if match_splited and (first or second):
                        result.append((first, second))
                    if not match_splited and first:
                        result.append(first)
                    if not match_splited and second:
                        result.append(second)
        return result

    def grand_slam(self):
        return self.level == lev.gs

    @staticmethod
    def endyear_date_correction(date):
        """
        Иногда турнир начинается перед новым годом, попадая по ISO в первую неделю след. года.
        Чтобы оставить турнир в текущем году его дата двигается назад на несколько дней.
        Надеемя что после этого сдвига year_weeknum вернет текущий год.
        Пример: итоговый atp 'Capitala World Tennis Championship' в 2008
        """
        if date and date.year == 2008 and date.month == 12 and date.day == 28:  # sunday
            return datetime.date(year=2008, month=12, day=25)
        return date

    def rounds_detailing(self):
        for matches in list(self.matches_from_rnd.values()):
            for m in matches:
                m.first_draw_status = ""
                m.second_draw_status = ""
        qual_r1st_plr_ids = [
            m.first_player.ident
            for m in self.matches_from_rnd[tennis.Round("q-First")]
            if not m.paired() and not m.second_player.unknown()
        ] + [
            m.second_player.ident
            for m in self.matches_from_rnd[tennis.Round("q-First")]
            if not m.paired() and not m.first_player.unknown()
        ]
        qual_r2nd_plr_ids = [
            m.first_player.ident
            for m in self.matches_from_rnd[tennis.Round("q-Second")]
            if not m.paired() and not m.second_player.unknown()
        ] + [
            m.second_player.ident
            for m in self.matches_from_rnd[tennis.Round("q-Second")]
            if not m.paired() and not m.first_player.unknown()
        ]
        qual_final_plr_ids = [
            m.first_player.ident
            for m in self.matches_from_rnd[tennis.Round("Qualifying")]
            if not m.paired() and not m.second_player.unknown()
        ] + [
            m.second_player.ident
            for m in self.matches_from_rnd[tennis.Round("Qualifying")]
            if not m.paired() and not m.first_player.unknown()
        ]
        qual_plr_ids = (
            set(qual_r1st_plr_ids) | set(qual_r2nd_plr_ids) | set(qual_final_plr_ids)
        )
        if qual_plr_ids:
            self.qual_lost = False
        else:
            qual_plr_ids = self.__qual_idents_from_dbseed()
        md_r1st_plr_ids = [
            m.first_player.ident
            for m in self.matches_from_rnd[tennis.Round("First")]
            if not m.paired() and not m.second_player.unknown()
        ] + [
            m.second_player.ident
            for m in self.matches_from_rnd[tennis.Round("First")]
            if not m.paired() and not m.first_player.unknown()
        ]
        md_r2nd_plr_ids = [
            m.first_player.ident
            for m in self.matches_from_rnd[tennis.Round("Second")]
            if not m.paired() and not m.second_player.unknown()
        ] + [
            m.second_player.ident
            for m in self.matches_from_rnd[tennis.Round("Second")]
            if not m.paired() and not m.first_player.unknown()
        ]
        bye_plr_ids = set(md_r2nd_plr_ids) - set(md_r1st_plr_ids) - set(qual_plr_ids)
        if not qual_plr_ids and not bye_plr_ids:
            return
        for rnd in sorted(self.matches_from_rnd.keys()):
            if rnd.qualification() or rnd.pre_qualification() or rnd.robin():
                continue
            for match in self.matches_from_rnd[rnd]:
                self.__round_detailing_impl(
                    match,
                    qual_plr_ids,
                    bye_plr_ids,
                    len(md_r1st_plr_ids),
                    len(md_r2nd_plr_ids),
                )

    def __qual_idents_from_dbseed(self):
        if not self.ident:
            return set()
        result = set(qual_seeds.qual_players_idents(self.sex, self.ident))
        if result:
            self.qual_lost = True
        return result

    def __round_detailing_impl(
        self,
        match,
        qual_plr_ids,
        bye_plr_ids,
        dbg_first_rnd_plr_count,
        dbg_second_rnd_plr_count,
    ):
        lhs_qual = match.first_player.ident in qual_plr_ids
        rhs_qual = match.second_player.ident in qual_plr_ids
        if lhs_qual and rhs_qual:
            match.first_draw_status = "qual"
            match.second_draw_status = "qual"
            return
        if match.rnd == "First" or match.rnd.robin():
            if lhs_qual or rhs_qual:
                if lhs_qual:
                    match.first_draw_status = "qual"
                else:
                    match.second_draw_status = "qual"
            return  # need not analyze for bye
        lhs_bye = match.first_player.ident in bye_plr_ids
        rhs_bye = match.second_player.ident in bye_plr_ids
        consistent = True
        if lhs_qual and lhs_bye:
            consistent = False
            log.error(
                "lhs_qual AND!! lhs_bye. sex: {} tid: {} lhsplr: {}"
                " rnd1_plr_cnt: {} rnd2_plr_cnt: {}".format(
                    self.sex,
                    self.ident,
                    match.first_player,
                    dbg_first_rnd_plr_count,
                    dbg_second_rnd_plr_count,
                )
            )
        if rhs_qual and rhs_bye:
            if consistent:
                consistent = False
                log.error(
                    "rhs_qual AND!! rhs_bye. sex: {} tid: {} rhsplr: {}"
                    " rnd1_plr_cnt: {} rnd2_plr_cnt: {}".format(
                        self.sex,
                        self.ident,
                        match.second_player,
                        dbg_first_rnd_plr_count,
                        dbg_second_rnd_plr_count,
                    )
                )
        if lhs_bye and rhs_bye and consistent:
            match.first_draw_status = "bye"
            match.second_draw_status = "bye"
        elif (lhs_qual or rhs_qual) and (lhs_bye or rhs_bye) and consistent:
            match.first_draw_status = "qual" if lhs_qual else "bye"
            match.second_draw_status = "qual" if rhs_qual else "bye"
        elif lhs_qual or rhs_qual:
            if lhs_qual:
                match.first_draw_status = "qual"
            else:
                match.second_draw_status = "qual"
        elif lhs_bye or rhs_bye and consistent:
            if lhs_bye:
                match.first_draw_status = "bye"
            else:
                match.second_draw_status = "bye"

    def matches(self, predicate=lambda m: True, reverse=False):
        for rnd in sorted(list(self.matches_from_rnd.keys()), reverse=reverse):
            for match in self.matches_from_rnd[rnd]:
                if predicate(match):
                    yield match

    def match_rnd_list(self, predicate=lambda m: True, reverse=False):
        """return list of (macth, rnd), ordered by time"""

        def compare(mr1, mr2):
            """compare (match1, round1) vs (match2, round2)"""
            if mr1[0].date is not None and mr2[0].date is not None:
                return co.cmp((mr1[0].date, mr1[1]), (mr2[0].date, mr2[1]))
            return co.cmp(mr1[1], mr2[1])

        match_rnd_lst = []
        for rnd, matches in self.matches_from_rnd.items():
            for match in matches:
                if predicate(match):
                    match_rnd_lst.append((match, rnd))
        return sorted(match_rnd_lst, key=cmp_to_key(compare), reverse=reverse)

    def avgset(self, isqual):
        """return SizedValue"""
        res_sumator = st.Sumator()
        for rnd in list(self.matches_from_rnd.keys()):
            is_best_of_five = self.best_of_five(rnd)
            if rnd.qualification() is isqual:
                for match in self.matches_from_rnd[rnd]:
                    if match.score is None or match.paired():
                        continue
                    setnum = match.score.sets_count(full=True)
                    if setnum == 0:
                        continue
                    stotals = [
                        min(13, match.score[i][0] + match.score[i][1])
                        for i in range(setnum)
                    ]
                    # skips if winner is very strong dominated:
                    if max(stotals) <= 7:
                        continue  # skip if dominated to zero (or to one) in each set
                    if not isqual:
                        if not is_best_of_five and stotals in ([8, 6], [6, 8]):
                            continue  # skip if dominated 6-2, 6-0
                        if is_best_of_five and stotals in (
                            [8, 6, 6],
                            [6, 8, 6],
                            [6, 6, 8],
                        ):
                            continue  # skip if dominated 6-2, 6-0, 6-0
                    for set_total in stotals:
                        res_sumator.hit(set_total)
        if not res_sumator:
            return rl.SizedValue()
        return rl.SizedValue(value=res_sumator.average(), size=res_sumator.size)

    def write_file(self, filename):
        with open(filename, "a") as fhandle:
            fhandle.write("***Tournament: id:{} {}\n".format(self.ident, str(self)))
            for rnd in sorted(list(self.matches_from_rnd.keys()), reverse=True):
                matches = self.matches_from_rnd[rnd]
                if matches:
                    fhandle.write("<<<tennis.Round: {}\n".format(rnd))
                    for m in matches:
                        fhandle.write("\t{}\n".format(m))
            fhandle.write("\n")


class SqlBuilder:
    Row = namedtuple(
        "Row",
        [
            "rnd_txt", "rnd_id", "surf_txt", "tour_id", "tour_name",
            "tour_rank", "tour_dt", "tour_money", "tour_cou",
            "fst_plr_id", "fst_plr_name", "fst_plr_dt", "fst_plr_cou",
            "snd_plr_id", "snd_plr_name", "snd_plr_dt", "snd_plr_cou",
            "score_txt", "match_dt"
        ],
    )

    def __init__(self, sex, todaymode, min_date, max_date, time_reverse=False,
                 with_paired=False, with_mix=False):
        self.sex = sex
        self.todaymode = todaymode
        self.min_date = min_date
        self.max_date = max_date
        self.time_reverse = time_reverse
        self.with_paired = with_paired
        self.with_mix = with_mix
        assert not (not with_paired and with_mix), "inconsistent with_paired with_mix"
        self.sql = self.build()

    def build(self):
        result = (
            self.select_clause(mix=False)
            + self.from_clause(mix=False)
            + self.where_clause()
            + dba.sql_dates_condition(self.min_date, self.max_date)
        )
        if self.with_mix:
            result += (
                "\nUNION \n"
                + self.select_clause(mix=True)
                + self.from_clause(mix=True)
                + self.where_clause()
                + dba.sql_dates_condition(self.min_date, self.max_date)
            )
        result += self.orderby() + ";"
        return result

    def orderby(self):
        if self.time_reverse:
            # tours.DATE_T desc,  tours.ID_T, Rounds.ID_R
            return "ORDER BY 7 desc, 4, 2"
        else:
            return "ORDER BY 7, 4, 2"

    def select_clause(self, mix):
        result = (
            "SELECT Rounds.NAME_R, Rounds.ID_R, Courts.NAME_C, tours.ID_T, \n"
            + "  tours.NAME_T, tours.RANK_T, tours.DATE_T, tours.PRIZE_T, "
            + "  tours.COUNTRY_T, \n"
        )
        if mix:
            result += (
                "  fst_plr.ID_P, fst_plr.NAME_P, null, fst_plr.COUNTRY_P, \n"
                + "  snd_plr.ID_P, snd_plr.NAME_P, null, snd_plr.COUNTRY_P"
            )
        else:
            result += (
                "  fst_plr.ID_P, fst_plr.NAME_P, fst_plr.DATE_P, fst_plr.COUNTRY_P, \n"
                + "  snd_plr.ID_P, snd_plr.NAME_P, snd_plr.DATE_P, snd_plr.COUNTRY_P"
            )
        if self.todaymode:
            result += ", today.RESULT, today.DATE_GAME \n"
        else:
            result += ", games.RESULT_G, games.DATE_G \n"
        return result

    def from_clause(self, mix):
        result = "FROM Rounds, Courts, "
        if mix:
            result += (
                "Tours_mxt AS tours, "
                + "Players_mxt AS fst_plr, Players_mxt AS snd_plr"
            )
            if self.todaymode:
                result += ", today_mxt AS today"
            else:
                result += ", games_mxt AS games"

        else:
            result += (
                "Tours_{0} AS tours, "
                + "Players_{0} AS fst_plr, "
                + "Players_{0} AS snd_plr"
            ).format(self.sex)
            if self.todaymode:
                result += ", today_{0} AS today".format(self.sex)
            else:
                result += ", games_{0} AS games".format(self.sex)
        return result + "\n"

    def where_clause(self):
        result = (
            "WHERE tours.ID_C_T = Courts.ID_C \n"
            + "\t and (tours.NAME_T Not Like '%juniors%') \n"
            + "\t and (tours.NAME_T Not Like '%Wildcard%') \n"
        )
        if self.todaymode:
            result += (
                "\t and today.ID1 = fst_plr.ID_P \n"
                + "\t and today.ID2 = snd_plr.ID_P \n"
                + "\t and fst_plr.NAME_P Not Like '%Unknown%'\n"
                + "\t and snd_plr.NAME_P Not Like '%Unknown%'\n"
                + "\t and today.TOUR = tours.ID_T \n"
                + "\t and today.ROUND = Rounds.ID_R \n"
            )
        else:
            result += (
                "\t and games.ID1_G = fst_plr.ID_P \n"
                + "\t and games.ID2_G = snd_plr.ID_P \n"
                + "\t and games.ID_T_G = tours.ID_T \n"
                + "\t and games.ID_R_G = Rounds.ID_R \n"
            )
        if not self.with_paired:
            result += "\t and (fst_plr.NAME_P Not Like '%/%') \n"
        return result

    def rows(self):
        with closing(dba.get_connect().cursor()) as cursor:
            cursor.execute(self.sql)
            for row in dba.result_iter(cursor):
                yield SqlBuilder.Row(*row)


def tours_generator(
    sex,
    todaymode=False,
    min_date=None,
    max_date=None,
    time_reverse=False,
    with_paired=False,
    with_mix=False,
    rnd_detailing=False,  # if True players draw_status will be defined
    with_bets=False,
    with_stat=False,
    with_ratings=False,
    with_pers_det=False,
):
    from matchstat import get

    assert not (not with_paired and with_mix), "inconsistent with_paired with_mix"
    sql_builder = SqlBuilder(
        sex,
        todaymode,
        min_date=min_date,
        max_date=max_date,
        time_reverse=time_reverse,
        with_paired=with_paired,
        with_mix=with_mix,
    )
    cur_tour = None
    for row in sql_builder.rows():
        tour = Tournament(
            row.tour_id,
            row.tour_name,
            sex=sex,
            surface=make_surf(row.surf_txt),
            rank=row.tour_rank,
            date=row.tour_dt.date() if row.tour_dt else None,
            money=row.tour_money,
            cou=row.tour_cou,
        )
        rnd = tennis.Round(row.rnd_txt)
        scr = sc.Score(row.score_txt) if row.score_txt else None
        fst_player = tennis.Player(
            ident=row.fst_plr_id,
            name=row.fst_plr_name,
            cou=row.fst_plr_cou,
            birth_date=row.fst_plr_dt.date() if row.fst_plr_dt else None,
        )
        snd_player = tennis.Player(
            ident=row.snd_plr_id,
            name=row.snd_plr_name,
            cou=row.snd_plr_cou,
            birth_date=row.snd_plr_dt.date() if row.snd_plr_dt else None,
        )
        m = tennis.Match(
            fst_player,
            snd_player,
            scr,
            rnd,
            row.match_dt.date() if row.match_dt else None,
        )
        if not m.paired():
            if with_pers_det:
                m.read_pers_det(sex)
            if with_ratings:
                m.read_ratings(sex, tour.date)
            if with_bets:
                m.fill_offer(sex, tour.ident, tour.date, alter_bettor=False)
            if with_stat:
                m.stat = get(
                    sex,
                    tour.ident,
                    rnd,
                    fst_player.ident,
                    snd_player.ident,
                    tt.get_year_weeknum(tour.date),
                )

        if cur_tour is not None and cur_tour.ident != tour.ident:
            if rnd_detailing:
                cur_tour.rounds_detailing()
            yield cur_tour
            cur_tour = tour
        elif cur_tour is None:
            cur_tour = tour
        cur_tour.matches_from_rnd[rnd].append(m)

    if cur_tour is not None:
        if rnd_detailing:
            cur_tour.rounds_detailing()
        yield cur_tour


def tours_write_file(tours, filename):
    fu.remove_file(filename)
    for tour in tours:
        tour.write_file(filename)


def mix_unioned_tours(
    sex,
    todaymode=False,
    min_date=None,
    max_date=None,
    time_reverse=False,
    rnd_detailing=False,
    with_bets=False,
    with_ratings=False,
    with_pers_det=False,
):
    def find_and_union(tours):
        slam_indicies = [
            i
            for i in range(len(tours))
            if tours[i].grand_slam() or tours[i].name in ("Perth", "Olympics")
        ]
        slam_size = len(slam_indicies)
        if slam_size <= 1:
            return False
        for i in range(slam_size - 1):
            fst_idx = slam_indicies[i]
            for j in range(i + 1, slam_size):
                snd_idx = slam_indicies[j]
                if (
                    tours[fst_idx].date != tours[snd_idx].date
                    or tours[fst_idx].name != tours[snd_idx].name
                ):
                    continue
                if (
                    tours[fst_idx].has_round(tennis.Round("First"), paired=False)
                    or tours[fst_idx].has_round(tennis.Round("Robin"), paired=False)
                ) and (
                    not tours[snd_idx].has_round(tennis.Round("First"), paired=False)
                    and not tours[snd_idx].has_round(
                        tennis.Round("Robin"), paired=False
                    )
                ):
                    main_idx, mix_idx = fst_idx, snd_idx
                elif (
                    tours[snd_idx].has_round(tennis.Round("First"), paired=False)
                    or tours[snd_idx].has_round(tennis.Round("Robin"), paired=False)
                ) and (
                    not tours[fst_idx].has_round(tennis.Round("First"), paired=False)
                    and not tours[fst_idx].has_round(
                        tennis.Round("Robin"), paired=False
                    )
                ):
                    main_idx, mix_idx = snd_idx, fst_idx
                else:
                    continue
                for rnd, matches in list(tours[mix_idx].matches_from_rnd.items()):
                    tours[main_idx].matches_from_rnd[rnd] += copy.deepcopy(matches)
                del tours[mix_idx]
                return True
        return False

    tours = list(
        tours_generator(
            sex,
            todaymode=todaymode,
            min_date=min_date,
            max_date=max_date,
            time_reverse=time_reverse,
            with_paired=True,
            rnd_detailing=rnd_detailing,
            with_bets=with_bets,
            with_ratings=with_ratings,
            with_pers_det=with_pers_det,
        )
    )
    found_and_unioned = True
    while found_and_unioned:
        found_and_unioned = find_and_union(tours)
    return tours


def week_splited_tours(
    sex,
    todaymode=False,
    min_date=None,
    max_date=None,
    time_reverse=False,
    with_paired=False,
    with_mix=False,
    rnd_detailing=False,
    with_bets=False,
    with_ratings=False,
    with_pers_det=False,
):
    """Если todaymode is None, это совмещенный режим с дополнением сегодн-ми т-рами"""
    assert not (not with_paired and with_mix), "inconsistent with_paired with_mix"
    max_date_rigid = None
    if max_date:  # relax for catch qual next tours after weeksplit in our range
        max_date_rigid = max_date
        max_date += datetime.timedelta(days=7)
    if with_mix:
        if todaymode in (None, True):
            td_tours = mix_unioned_tours(
                sex,
                todaymode=True,
                min_date=min_date,
                max_date=max_date,
                time_reverse=time_reverse,
                rnd_detailing=rnd_detailing,
                with_bets=with_bets,
                with_ratings=with_ratings,
            )
        if todaymode in (None, False):
            tours = mix_unioned_tours(
                sex,
                todaymode=False,
                min_date=min_date,
                max_date=max_date,
                time_reverse=time_reverse,
                rnd_detailing=rnd_detailing,
                with_bets=with_bets,
                with_ratings=with_ratings,
            )
    else:
        if todaymode in (None, True):
            td_tours = list(
                tours_generator(
                    sex,
                    todaymode=True,
                    min_date=min_date,
                    max_date=max_date,
                    time_reverse=time_reverse,
                    with_paired=with_paired,
                    rnd_detailing=rnd_detailing,
                    with_bets=with_bets,
                    with_ratings=with_ratings,
                    with_pers_det=with_pers_det,
                )
            )
        if todaymode in (None, False):
            tours = list(
                tours_generator(
                    sex,
                    todaymode=False,
                    min_date=min_date,
                    max_date=max_date,
                    time_reverse=time_reverse,
                    with_paired=with_paired,
                    rnd_detailing=rnd_detailing,
                    with_bets=with_bets,
                    with_ratings=with_ratings,
                    with_pers_det=with_pers_det,
                )
            )
    if todaymode is None:
        complete_with_today(tours, td_tours)

    out_tours = td_tours if todaymode is True else tours
    result_tours = []
    for tour in out_tours:
        result_tours += split_tour_by_weeks(tour)
    if max_date_rigid or min_date:
        min_date_rigid = min_date if min_date else datetime.date(1990, 1, 1)
        max_date_rigid = max_date if max_date else datetime.date(2060, 1, 1)
        result_tours = [
            t for t in result_tours if min_date_rigid <= t.date < max_date_rigid
        ]
    result_tours.sort(
        key=lambda t: (t.date, not t.has_main_draw()), reverse=time_reverse
    )
    return result_tours


def round_week_shift_by_struct(tour, rnd):
    """
    -1 если раунд играется на прошлой неделе по отношению к основной неделе турнира
     0 если раунд играется на основной неделе турнира
     1 если раунд играется на следующей неделе по отношению к основной неделе турнира
    """

    def qualifying_shift_by_struct(tour):
        if (tour.level == "main" and tour.surface == "Carpet") or (
            tour.level == "main" and tour.surface == "Clay" and tour.sex == "wta"
        ):
            return 0
        else:
            return -1

    if tour.grand_slam():
        if rnd.qualification() or rnd.pre_qualification():
            return -1
        elif rnd >= tennis.Round("Fourth"):
            return 1
    elif tour.has_round(tennis.Round("Fourth")):  # example: Indian Wells, Miami tours
        if (tour.sex == "atp" and rnd >= tennis.Round("Third")) or (
            tour.sex == "wta" and rnd >= tennis.Round("Fourth")
        ):
            return 1
    else:
        if rnd == tennis.Round("Qualifying"):
            return qualifying_shift_by_struct(tour)
        if rnd.qualification() or rnd.pre_qualification():
            return -1
    return 0  # most often case


def split_tour_by_weeks(tour):
    def get_qual_match_date_shifted(tour_date, match_date, rnd):
        if match_date is not None or not rnd.qualification():
            return None
        # qualifying -> sunday, q-Second -> saturday, q-First -> friday
        if rnd == "Qualifying":
            return tour_date + datetime.timedelta(days=6)
        elif rnd == "q-Second":
            return tour_date + datetime.timedelta(days=5)
        elif rnd == "q-First":
            return tour_date + datetime.timedelta(days=4)

    def get_mdraw_match_date(tour_date, match_date, rnd):
        if match_date is not None:
            return None
        # qualifying->monday, 1st->tuesday, 2nd->wednesday, 1/4->thursday, 1/2->friday
        if rnd == "Qualifying":
            return tour_date
        elif rnd == "First":
            return tour_date + datetime.timedelta(days=1)
        elif rnd == "Second":
            return tour_date + datetime.timedelta(days=2)
        elif rnd == "1/4":
            return tour_date + datetime.timedelta(days=3)
        elif rnd == "1/2":
            return tour_date + datetime.timedelta(days=4)
        elif rnd == "Final":
            return tour_date + datetime.timedelta(days=5)

    prevweek_matches_from_rnd = defaultdict(list)
    curweek_matches_from_rnd = defaultdict(list)
    nextweek_matches_from_rnd = defaultdict(list)
    curweek_date = tt.past_monday_date(tour.date)
    prevweek_date = curweek_date - datetime.timedelta(days=7)
    nextweek_date = curweek_date + datetime.timedelta(days=7)

    for rnd, matches in tour.matches_from_rnd.items():
        for m in matches:
            if m.date is not None:
                if prevweek_date <= m.date < curweek_date:
                    prevweek_matches_from_rnd[rnd].append(m)
                elif curweek_date <= m.date < nextweek_date:
                    curweek_matches_from_rnd[rnd].append(m)
                elif nextweek_date <= m.date:
                    nextweek_matches_from_rnd[rnd].append(m)
                else:
                    raise co.TennisError(
                        "match date {} is not binded with tour: {} match: {}".format(
                            m.date, str(tour), str(m)
                        )
                    )
            else:
                week_shift = round_week_shift_by_struct(tour, rnd)
                if week_shift == -1:
                    if rnd.qualification():
                        m.date = get_qual_match_date_shifted(prevweek_date, m.date, rnd)
                    prevweek_matches_from_rnd[rnd].append(m)
                elif week_shift == 0:
                    m.date = get_mdraw_match_date(curweek_date, m.date, rnd)
                    curweek_matches_from_rnd[rnd].append(m)
                else:
                    nextweek_matches_from_rnd[rnd].append(m)

    result = []
    if len(prevweek_matches_from_rnd) > 0:
        prevweek_tour = copy.deepcopy(tour)
        prevweek_tour.date = prevweek_date
        prevweek_tour.matches_from_rnd = prevweek_matches_from_rnd
        result.append(prevweek_tour)

    if len(curweek_matches_from_rnd) > 0:
        # м.б. пусто если онкорт еще не прислал матчи основной сетки
        curweek_tour = copy.deepcopy(tour)
        curweek_tour.matches_from_rnd = curweek_matches_from_rnd
        result.append(curweek_tour)

    if len(nextweek_matches_from_rnd) > 0:
        nextweek_tour = copy.deepcopy(tour)
        nextweek_tour.date = nextweek_date
        nextweek_tour.matches_from_rnd = nextweek_matches_from_rnd
        result.append(nextweek_tour)
    return result


def tours_from_ywn_dict(sex, min_date=None, max_date=None, with_bets=True):
    tours_from_yearweeknum = defaultdict(list)
    for tour in tours_generator(
        sex,
        min_date=min_date,
        max_date=max_date,
        time_reverse=False,
        with_paired=False,
        with_bets=with_bets,
    ):
        if tour.level == lev.junior:
            continue
        for rnd in list(tour.matches_from_rnd.keys()):
            if rnd.pre_qualification():
                del tour.matches_from_rnd[rnd]
        tours_from_yearweeknum[tour.year_weeknum].append(tour)
    return tours_from_yearweeknum


def complete_with_today(tours, today_tours, with_unscored_matches=False):
    """дополняем tours недостающими матчами из today_tours (или целыми недост. тур-ми).
    Предв-но из today_tours del матчи без счета-рез-та (if not with_unscored_matches)
    """

    def same_players(fst_match, snd_match):
        if (
            fst_match.first_player is None
            or fst_match.second_player is None
            or snd_match.first_player is None
            or snd_match.second_player is None
        ):
            return False
        if (
            not fst_match.first_player.name
            or not fst_match.second_player.name
            or not snd_match.first_player.name
            or not snd_match.second_player.name
        ):
            return False
        return (
            fst_match.first_player.name == snd_match.first_player.name
            and fst_match.second_player.name == snd_match.second_player.name
        ) or (
            fst_match.first_player.name == snd_match.second_player.name
            and fst_match.second_player.name == snd_match.first_player.name
        )

    def remain_scored_tours(tours):
        for tidx in list(reversed(list(range(len(tours))))):
            tour = tours[tidx]
            for rnd in list(tour.matches_from_rnd.keys()):
                for midx in list(
                    reversed(list(range(len(tour.matches_from_rnd[rnd]))))
                ):
                    match = tour.matches_from_rnd[rnd][midx]
                    if match.score is None:
                        del tour.matches_from_rnd[rnd][midx]
                if len(tour.matches_from_rnd[rnd]) == 0:
                    del tour.matches_from_rnd[rnd]
            if len(tour.matches_from_rnd) == 0:
                del tours[tidx]

    if not with_unscored_matches:
        remain_scored_tours(today_tours)
    for today_tour in today_tours:
        tour = co.find_first(tours, lambda t, tot=today_tour: t.ident == tot.ident)
        if tour is None:
            tours.append(today_tour)
        else:
            for rnd in today_tour.matches_from_rnd.keys():
                for tm in today_tour.matches_from_rnd[rnd]:
                    match = co.find_first(
                        tour.matches_from_rnd[rnd],
                        lambda m, tom=tm: same_players(m, tom),
                    )
                    if match is None and tm.first_player and tm.second_player:
                        tour.matches_from_rnd[rnd].append(copy.deepcopy(tm))


class PandemiaTours:
    """exhibition tournaments during pandemia since 2020 ~ march"""

    def __init__(self):
        self._sex_recs = defaultdict(
            set
        )  # sex -> set of Tuple[rnd_txt, left_id, right_id]

    def init_from_det_db(self, sex: str, session):
        tour_id = pandemia.exhib_tour_id(sex)
        records = session.query(MatchRec).filter(MatchRec.tour_id == tour_id).all()
        if records:
            self._sex_recs[sex] = {(r.rnd, r.left_id, r.right_id) for r in records}
            log.info(f"{self.__class__.__name__} init {sex} len: {len(records)}")

    def is_match_in_det_db(self, sex, tour_id, rnd, left_id, right_id):
        rnd_txt = rnd if isinstance(rnd, str) else rnd.value
        return (
            tour_id == pandemia.exhib_tour_id(sex)
            and (rnd_txt, left_id, right_id) in self._sex_recs[sex]
        )

    def is_match_should_skip(self, sex, tour_id, rnd, left_id, right_id):
        """None not pandemia event (матч можно добавлять в базу);
        False этот пандемийный матч можно добавлять в базу;
        True этот пандемийный матч следует пропустить (встречался уже)"""
        if tour_id != pandemia.exhib_tour_id(sex):
            return None  # not pandemia event
        rnd_txt = rnd if isinstance(rnd, str) else rnd.value
        is_exist = (rnd_txt, left_id, right_id) in self._sex_recs[sex]
        if not is_exist:
            self._sex_recs[sex].add((rnd_txt, left_id, right_id))
            return False
        return True
