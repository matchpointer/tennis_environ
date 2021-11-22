import datetime

import stat_cont as st
import tennis_time as tt
import weeked_tours


class LastResults(object):
    def __init__(self, sex, player_id, weeks_ago):
        """weeks_ago - how many weeks from past.
        weeked_tours must be inited aleady"""
        self.MIN_GAMES_IN_MATCH = 9  # if retired case

        # list of list of bool: [current week iswin list, then 1 week ago iswin list,...]
        # недельный список упорядочен по времени с реверсом (сначала свежие рез-ты)
        if sex is not None and player_id is not None:
            self.week_results_list = self.__init_data(sex, player_id, weeks_ago)

    @staticmethod
    def from_week_results(week_results):
        obj = LastResults(None, None, 0)
        obj.week_results_list = week_results
        return obj

    def exist_win_streak(self, min_ratio, min_size):
        """ищем начиная со свежайшего результата уходя в прошлое разными размерами"""
        iswins = self.__iswin_list()
        iswins_len = len(iswins)
        if iswins_len >= min_size:
            for size in range(min_size, iswins_len + 1):
                wl = st.WinLoss.from_iter(iswins[0:size])
                if wl.ratio >= min_ratio:
                    return True
        return False

    def best_win_streak_ratio(self, min_ratio, min_size):
        """ищем начиная со свежайшего результата уходя в прошлое разными размерами.
        Если не выполнен min_size, то вернем float(0).
        """
        best_ratio = 0.0
        iswins = self.__iswin_list()
        iswins_len = len(iswins)
        if iswins_len >= min_size:
            for size in range(min_size, iswins_len + 1):
                wl = st.WinLoss.from_iter(iswins[0:size])
                if wl.ratio >= min_ratio and wl.ratio > best_ratio:
                    best_ratio = wl.ratio
        return best_ratio

    def prev_weeks_empty(self, weeks_num=2):
        """not take into account current tour week"""
        for week_num in range(1, weeks_num + 1):
            if self.week_results_list[week_num]:
                return False  # week week_num is not empty
        return True

    def last_weeks_empty(self, weeks_num):
        """with current tour week"""
        for week_num in range(weeks_num + 1):
            if self.week_results_list[week_num]:
                return False  # week week_num is not empty
        return True

    def poor_practice(self):
        """нет побед, нет матчей тек. недели, не более 1 матча в прошлых 3-х нед."""
        if len(self.week_results_list) >= 4 and not self.week_results_list[0]:
            prev_weeks_matches_cnt = (
                len(self.week_results_list[1])
                + len(self.week_results_list[2])
                + len(self.week_results_list[3])
            )
            if prev_weeks_matches_cnt <= 1 and self.__iswin_list().count(True) == 0:
                return True

    def min_load_vs_poor_practice(self):
        if len(self.week_results_list) >= 3:
            # at least 2 matches in {current, -1week, -2week}
            return sum([len(wlst) for wlst in self.week_results_list[:3]]) >= 2

    def __str__(self):
        def week_str(iswins):
            result = ""
            if iswins:
                for iswin in iswins:
                    result += str(int(iswin))
            else:
                result = "x"
            return result

        return "-".join([week_str(iswins) for iswins in self.week_results_list])

    def __iswin_list(self):
        """список упорядочен по времени с реверсом (сначала свежие рез-ты)"""
        return [iswin for weeklst in self.week_results_list for iswin in weeklst]

    def __init_data(self, sex, player_id, weeks_ago):
        result = []
        monday_date = tt.past_monday_date(datetime.date.today())
        start_ywn = tt.get_year_weeknum(monday_date)
        for ywn in tt.year_weeknum_reversed(start_ywn, weeks_ago):
            result.append(self.__week_results(player_id, weeked_tours.tours(sex, ywn)))
        return result

    def __week_results(self, player_id, tours):
        results = []
        for tour in sorted(tours, key=lambda t: t.date, reverse=True):
            for rnd in sorted(iter(tour.matches_from_rnd.keys()), reverse=True):
                for match in tour.matches_from_rnd[rnd]:
                    if match.paired() or match.score is None:
                        continue
                    if (
                        match.first_player.ident != player_id
                        and match.second_player.ident != player_id
                    ):
                        continue
                    if (
                        match.score.retired
                        and match.score.games_count() < self.MIN_GAMES_IN_MATCH
                    ):
                        continue
                    results.append(match.first_player.ident == player_id)
        return results
