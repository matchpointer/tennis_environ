# -*- coding=utf-8 -*-
r"""
module gives main entities for live.
"""
import copy
import datetime
from collections import defaultdict
from typing import Optional
from enum import IntEnum

import common as co
import decided_set
import feature
import get_features
from last_results import LastResults
from loguru import logger as log
from surf import Clay, Grass
from bet import WinCoefs
import matchstat
import pre_live_dir
import ratings
import report_line as rl
import score as sc
import set2_after_set1loss_stat
import stat_cont as st
import stopwatch
from tennis import Match, Round, Player, HeadToHead
import player_name_ident
import tennis_time as tt
import tournament as trmt
import weeked_tours
from clf_common import FeatureError
import atfirst_after
from live_tourinfo import TourInfo
from tour_name import TourName
from debug_helper import (
    add_debug_match_data, add_debug_match_feat_dict, get_debug_match_name
)
import advantage_tie_ratio
import lev


class LiveEventToskipError(co.TennisError):
    pass


class MatchStatus(IntEnum):
    scheduled = 1
    live = 2
    finished = 3


class MatchSubStatus(IntEnum):
    interrupted = 4
    retired = 5  # also covers 'walkover'
    cancelled = 6
    # also exist 'awaiting updates'


def skip_levels_default():
    return {
        "wta": ("junior", "future"),
        "atp": ("junior", "future"),
    }


def skip_levels_work():
    return {
        "wta": ("junior", "future", "team", "teamworld"),
        "atp": ("junior", "future", "team", "teamworld", "gs"),
    }


def back_year_weeknums(max_weeknum_dist):
    if (
        back_year_weeknums.lst is not None
        and back_year_weeknums.max_dist == max_weeknum_dist
    ):
        return back_year_weeknums.lst
    back_year_weeknums.lst = list(
        tt.year_weeknum_reversed(
            tt.get_year_weeknum(datetime.date.today()),
            max_weeknum_dist=max_weeknum_dist,
        )
    )
    back_year_weeknums.max_dist = max_weeknum_dist
    return back_year_weeknums.lst


back_year_weeknums.lst = None
back_year_weeknums.max_dist = None


class LiveMatch(Match):
    min_decset_date = datetime.date.today() - datetime.timedelta(days=365 * 3)

    def __init__(self, live_event):
        super().__init__()
        self.live_event = live_event
        self.name = None  # sample: 'Federer R. - Nadal R.'
        self.href = None
        self.define_players_tried = False
        self.fill_details_tried = False
        self.left_service = None
        self.ingame = None  # sample: ('A', '40')
        self.worked_alerts = []
        self.srvw_stat = (
            None  # obj for players history for serve/receive (matchstat.py)
        )
        self.fst_last_res: Optional[LastResults] = None
        self.snd_last_res: Optional[LastResults] = None
        self.quad_stat = st.QuadServiceStat()
        self.set_opener_tracker = st.SetOpenerMatchTracker()
        self.kill_timer = None
        self.h2h_direct = None  # 1 dominate of first; -1 dominate of second
        self.decset_ratio_dif = 0.0
        self.decset_bonus_dif = 0.0
        self.hist_fst_srv_win = rl.SizedValue()
        self.hist_snd_srv_win = rl.SizedValue()
        self.hist_fst_rcv_win = rl.SizedValue()
        self.hist_snd_rcv_win = rl.SizedValue()
        self.features = []  # Features obj: fst_fatigue, snd_fatigue
        self.hot_nums = None  # uses for tie point numbering
        self.time = None  # uses in today match
        self.def_plr_seconds = 0.0
        self.check_unknown = False
        # cache values: LEFT or RIGHT if pre-estimated back side, False if reject pre-estimated
        #               None if cache value is absent
        self.cache_dict = defaultdict(lambda: None)

    def get_cache_value(self, name):
        return self.cache_dict.get(name)

    def set_cache_value(self, name, value):
        self.cache_dict[name] = value

    def pointbypoint_href(self, setnum=1):
        if self.href:
            return f"{self.href}/#match-summary/point-by-point/{setnum - 1}"

    def summary_href(self) -> str:
        if self.href:
            return "{}/#match-summary/match-summary".format(self.href)

    @property
    def datetime(self) -> Optional[datetime.datetime]:
        if self.date is None:
            return None
        if self.time is None:
            return datetime.datetime(
                self.date.year, self.date.month, self.date.day, 0, 0, 0
            )
        return datetime.datetime(
            self.date.year,
            self.date.month,
            self.date.day,
            self.time.hour,
            self.time.minute,
            self.time.second,
        )

    @property
    def sex(self) -> Optional[str]:
        if self.live_event is not None:
            return self.live_event.sex

    @property
    def level(self):
        if self.live_event is not None:
            return self.live_event.level

    @property
    def soft_level(self) -> str:
        return lev.soft_level(self.level, self.rnd, self.qualification)

    @property
    def surface(self):
        if self.live_event is not None:
            return self.live_event.surface

    @property
    def tour_name(self) -> Optional[TourName]:
        if self.live_event is not None:
            return self.live_event.tour_name

    @property
    def best_of_five(self) -> Optional[bool]:
        if self.rnd is not None:
            return trmt.best_of_five(
                self.date, self.sex, self.tour_name, self.level, self.rnd
            )
        if self.live_event is not None:
            return self.live_event.best_of_five

    @property
    def qualification(self) -> Optional[bool]:
        if self.rnd is not None:
            return self.rnd.qualification()
        if self.live_event is not None and self.live_event.qualification is not None:
            return self.live_event.qualification

    @property
    def decided_tiebreak(self) -> Optional[sc.TieInfo]:
        result = super().decided_tiebreak
        if result is not None:
            return result
        return self.live_event.decided_tiebreak

    def paired(self):
        if self.name:
            return "/" in self.name
        return super(LiveMatch, self).paired()

    def read_ratings(self, sex, date=None, surfaces=("all",)):
        if self.surface is not None:
            surfaces = ("all", self.surface)
        if self.first_player:
            self.first_player.read_rating(sex, date=date, surfaces=surfaces)

        if self.second_player:
            self.second_player.read_rating(sex, date=date, surfaces=surfaces)

    def flip(self):
        super(LiveMatch, self).flip()
        if self.name:
            fst, snd = self.name.split(" - ")
            self.name = snd + " - " + fst
        if self.features:
            for feat in self.features:
                feat.flip_side()
        if self.h2h_direct is not None:
            self.h2h_direct = -self.h2h_direct
        if self.decset_ratio_dif is not None:
            self.decset_ratio_dif = -self.decset_ratio_dif
        if self.decset_bonus_dif is not None:
            self.decset_bonus_dif = -self.decset_bonus_dif
        self.hist_fst_srv_win, self.hist_snd_srv_win = (
            self.hist_snd_srv_win,
            self.hist_fst_srv_win,
        )
        self.hist_fst_rcv_win, self.hist_snd_rcv_win = (
            self.hist_snd_rcv_win,
            self.hist_fst_rcv_win,
        )
        self.fst_last_res, self.snd_last_res = self.snd_last_res, self.fst_last_res

    def update_with(self, score, left_service, ingame):
        def new_set_begining():
            if n_prev_sets >= 1:
                return (
                    (n_prev_sets + 1) == n_cur_sets
                    and score[-1] == (0, 0)
                    and sc.exist_point_increment(self.score[-1], score[-2])
                )

        if score is None:
            return
        n_prev_sets = len(self.score)
        n_cur_sets = len(score)
        consilience = self.set_opener_tracker.put(n_cur_sets, score[-1], left_service)
        if consilience is False:
            left_service = not left_service
        elif consilience is None and new_set_begining():
            prev_left_service_last = self.set_opener_tracker.is_left_opener(
                n_prev_sets, score[n_prev_sets - 1], at=False
            )
            left_service_new = (
                not prev_left_service_last
                if prev_left_service_last is not None
                else None
            )
            if left_service_new is not None and left_service_new != left_service:
                left_service = left_service_new

        if self.name == get_debug_match_name():
            add_debug_match_data(score, ingame, left_service)
            if len(score) == 2 and score[-1] == (0, 0) and ingame == ('0', '0'):
                add_debug_match_feat_dict('secondset_00', self.features)
            elif len(score) == 3 and score[-1] == (0, 0) and ingame == ('0', '0'):
                add_debug_match_feat_dict('decided_00', self.features)

        scores_same = (self.score, self.ingame, self.left_service) == (
            score,
            ingame,
            left_service,
        )
        if not scores_same:
            self.quad_stat.update_with(
                self.score, self.ingame, self.left_service, score, ingame, left_service
            )
            # here only score.wl_games copy т.к. при all score copy при
            # завершении тайбрейка правый score c пустым idx_mintie затрет левый с непустым
            self.score.wl_games = copy.copy(score.wl_games)
            self.score.live_update_tie_loser_result(ingame)
            self.left_service = left_service
            self.ingame = ingame

    def curset_opener_side(self):
        if self.score is None or self.left_service is None:
            return None
        if (sum(self.score[-1]) % 2) == 0:
            return co.LEFT if self.left_service else co.RIGHT
        else:
            return co.RIGHT if self.left_service else co.LEFT

    def hashid(self, sex=None, add_text=""):
        text = ""
        if sex:
            text += sex
        if self.first_player:
            text += str(self.first_player.name)
        if self.second_player:
            text += str(self.second_player.name)
        text += add_text
        return hash(text)

    def check_hist_min_size(self, min_size):
        return (
            self.hist_fst_srv_win is not None
            and self.hist_fst_srv_win.size >= min_size
            and self.hist_fst_srv_win.value is not None
            and
            #
            self.hist_fst_rcv_win is not None
            and self.hist_fst_rcv_win.size >= min_size
            and self.hist_fst_rcv_win.value is not None
            and
            #
            self.hist_snd_srv_win is not None
            and self.hist_snd_srv_win.size >= min_size
            and self.hist_snd_srv_win.value is not None
            and
            #
            self.hist_snd_rcv_win is not None
            and self.hist_snd_rcv_win.size >= min_size
            and self.hist_snd_rcv_win.value is not None
        )

    def log_check_unknown(self):
        if self.check_unknown:
            return
        self.check_unknown = True
        if self.first_player is None or self.first_player.ident is None:
            log.warning("unknown left {} {}".format(self.name, self.live_event))
        if self.second_player is None or self.second_player.ident is None:
            log.warning("unknown right {} {}".format(self.name, self.live_event))

    def players_defined(self):
        return (
            self.first_player is not None
            and self.first_player.ident is not None
            and self.second_player is not None
            and self.second_player.ident is not None
        )

    def define_players(self, sex: str):
        if self.define_players_tried:
            return self.players_defined()
        self.define_players_tried = True
        pre_live_dir.prepare_dir("matches")
        if pre_live_dir.is_data(self.pre_live_name()):
            self.load_pre_live_data()
            return self.players_defined()
        abbr_names = self.name.split(" - ")
        if len(abbr_names) == 2:
            timer = stopwatch.Timer()
            if not self.first_player:
                self.first_player = player_name_ident.identify_player(
                    sex, abbr_names[0].strip())
            if not self.second_player:
                self.second_player = player_name_ident.identify_player(
                    sex, abbr_names[1].strip())
            if self.first_player and self.second_player:
                self.read_ratings(sex)
                self.read_birth_date(sex)
                self.head_to_head = HeadToHead(sex, self, completed_only=True)
                self.h2h_direct = self.head_to_head.direct()
                self.fst_last_res = LastResults(sex, self.first_player.ident)
                self.snd_last_res = LastResults(sex, self.second_player.ident)
                self.decset_ratio_dif = decided_set.get_dif_ratio(
                    self.sex,
                    self.first_player.ident,
                    self.second_player.ident,
                    min_date=self.min_decset_date,
                    max_date=datetime.date.today())
                self.decset_bonus_dif = decided_set.get_dif_bonus(
                    self.sex,
                    self.first_player.ident,
                    self.second_player.ident,
                    min_date=self.min_decset_date,
                    max_date=datetime.date.today())
                (
                    self.hist_fst_srv_win,
                    self.hist_fst_rcv_win,
                ) = matchstat.player_srvrcv_win(
                    self.sex,
                    self.first_player.ident,
                    back_year_weeknums(max_weeknum_dist=matchstat.HIST_WEEKS_NUM),
                    self.surface,
                    12)
                (
                    self.hist_snd_srv_win,
                    self.hist_snd_rcv_win,
                ) = matchstat.player_srvrcv_win(
                    self.sex,
                    self.second_player.ident,
                    back_year_weeknums(max_weeknum_dist=matchstat.HIST_WEEKS_NUM),
                    self.surface,
                    12)
                self.features = []
                self.features.append(
                    feature.RigidFeature(
                        "recently_winner_id", self.head_to_head.recently_won_player_id()
                    )
                )
                get_features.add_fatigue_features(
                    self.features,
                    self.sex,
                    back_year_weeknums(max_weeknum_dist=30),
                    self,
                    rnd=None,
                )
                get_features.add_prevyear_tour_features(
                    self.features,
                    self.sex,
                    back_year_weeknums(max_weeknum_dist=56),
                    self.live_event,
                    self,
                )
                get_features.add_nmatches_features(
                    self.features,
                    self.sex,
                    back_year_weeknums(max_weeknum_dist=56)[:52],
                    self.first_player.ident,
                    self.second_player.ident,
                    featprefix='pastyear_nmatches',
                )
                f1tradp, f2tradp = get_features.tour_adapt_features(None, self)
                self.features.append(f1tradp)
                self.features.append(f2tradp)
                f1set2r, f2set2r = set2_after_set1loss_stat.read_features_pair(
                    self.sex,
                    "set2win_after_set1loss",
                    self.first_player.ident,
                    self.second_player.ident,
                )
                self.features.append(f1set2r)
                self.features.append(f2set2r)
                f1set2k, f2set2k = set2_after_set1loss_stat.read_features_pair(
                    self.sex,
                    "set2win_after_set1win",
                    self.first_player.ident,
                    self.second_player.ident,
                )
                self.features.append(f1set2k)
                self.features.append(f2set2k)

                atfirst_after.add_read_features(
                    self.features,
                    self.sex,
                    self.first_player.ident,
                    self.second_player.ident,
                )

            self.def_plr_seconds = timer.elapsed
        return self.players_defined()

    def load_pre_live_data(self):
        """ deserializing: read dct from json file,
              and load values mainly to self.features and some other attrs """

        def simple_feature(name):
            if name in dct:
                self.features.append(feature.RigidFeature(name=name, value=dct[name]))

        def simple_sv_feature(name):
            if name in dct:
                sval_args = dct[name]
                if sval_args is not None:
                    sval = rl.SizedValue(*sval_args)
                    self.features.append(feature.RigidFeature(name=name, value=sval))

        def plr_sv_feature(name):
            fst_name = f"fst_{name}"
            snd_name = f"snd_{name}"
            if fst_name in dct and snd_name in dct:
                fst_val = rl.SizedValue(*dct[fst_name])
                snd_val = rl.SizedValue(*dct[snd_name])
                self.features.append(
                    feature.Feature(name=fst_name, value=fst_val, flip_value=snd_val))
                self.features.append(
                    feature.Feature(name=snd_name, value=snd_val, flip_value=fst_val))

        def plr_feature(name):
            fst_name = f"fst_{name}"
            snd_name = f"snd_{name}"
            if fst_name in dct and snd_name in dct:
                fst_val = dct[fst_name]
                snd_val = dct[snd_name]
                self.features.append(
                    feature.Feature(name=fst_name, value=fst_val, flip_value=snd_val))
                self.features.append(
                    feature.Feature(name=snd_name, value=snd_val, flip_value=fst_val))

        def player_load(prefix: str, player: Player):
            player.ident = dct.get(prefix + "_player_id")
            player.name = dct.get(prefix + "_player_name")
            player.rating = ratings.Rating(dct.get(prefix + "_player_rating"))
            ymd = dct.get(prefix + "_player_bdate")
            if ymd is not None:
                player.birth_date = datetime.date(ymd[0], ymd[1], ymd[2])
            setattr(
                self,
                "hist_" + prefix + "_srv_win",
                rl.SizedValue(*dct[prefix + "_hist_srv_win"]),
            )
            setattr(
                self,
                "hist_" + prefix + "_rcv_win",
                rl.SizedValue(*dct[prefix + "_hist_rcv_win"]),
            )

        dct = pre_live_dir.get_data(self.pre_live_name())
        if self.first_player is None:
            self.first_player = Player()
        if self.second_player is None:
            self.second_player = Player()
        player_load("fst", self.first_player)
        player_load("snd", self.second_player)
        if "rnd" in dct:
            rnd_txt = dct["rnd"]
            if rnd_txt:
                rnd_txt_parts = rnd_txt.split(";")
                self.rnd = Round(rnd_txt_parts[0])
        self.decset_ratio_dif = dct.get("decset_ratio_dif")
        self.decset_bonus_dif = dct.get("decset_bonus_dif")
        self.h2h_direct = dct.get("h2h_direct")

        self.features = []
        simple_feature("recently_winner_id")
        plr_feature("fatigue")
        plr_feature("plr_tour_adapt")
        plr_feature("prevyear_tour_rnd")
        plr_feature("set2win_after_set1loss")
        plr_feature("set2win_after_set1win")
        plr_feature("decided_begin_sh-3")
        plr_feature("decided_keep_sh-3")
        plr_feature("decided_recovery_sh-3")
        plr_feature("trail")
        plr_feature("choke")
        plr_feature("absence")
        plr_feature("retired")
        plr_feature("pastyear_nmatches")
        plr_sv_feature("sd_tie_ratio")
        simple_feature(f"{advantage_tie_ratio.ADV_PREF}sd_tie_ratio")

        if "fst_win_coef" in dct and "snd_win_coef" in dct:
            fst_wcoef, snd_wcoef = dct["fst_win_coef"], dct["snd_win_coef"]
            if fst_wcoef is not None and snd_wcoef is not None:
                if not self.offer.win_coefs:
                    self.offer.win_coefs = WinCoefs(fst_wcoef, snd_wcoef)

        if "fst_draw_status" in dct and "snd_draw_status" in dct:
            self.first_draw_status = dct["fst_draw_status"]
            self.second_draw_status = dct["snd_draw_status"]

        if "fst_last_res" in dct:
            self.fst_last_res = LastResults.from_week_results(
                dct["fst_last_res"])
        if "snd_last_res" in dct:
            self.snd_last_res = LastResults.from_week_results(
                dct["snd_last_res"])

    def abbr_rating_info(self, rtg_name="elo"):
        result = "-"
        if self.first_player and self.first_player.rating.rank(rtg_name):
            result = str(self.first_player.rating.rank(rtg_name)) + "-"
        if self.second_player and self.second_player.rating.rank(rtg_name):
            result += str(self.second_player.rating.rank(rtg_name))
        return "R:" + result

    def tostring(self, extended=False):
        rnd_txt = (" " + str(self.rnd)[:3]) if self.rnd is not None else ""
        fst_bet_chance = self.first_player_bet_chance()
        result = "{}{} hh:{} bet:{}".format(
            str(self),
            rnd_txt,
            "-" if self.h2h_direct is None else "{:.2f}".format(self.h2h_direct),
            "-" if fst_bet_chance is None else "{:.2f}".format(fst_bet_chance),
        )
        if extended:
            if self.quad_stat:
                result += "\n\t\t{} sec: {:.1f}".format(
                    str(self.quad_stat),
                    self.def_plr_seconds,
                )
        return result

    def __str__(self):
        if self.ingame is None:
            abbr_ingame = ""
        else:
            abbr_ingame = self.ingame[0] + "-" + self.ingame[1]

        if not self.first_draw_status and not self.second_draw_status:
            draws = ""
        else:
            draws = "(%s-%s)" % (
                "" if not self.first_draw_status else self.first_draw_status,
                "" if not self.second_draw_status else self.second_draw_status,
            )
        return "{}{} {} {} S{} {}".format(
            self.name,
            draws,
            self.score,
            abbr_ingame,
            "L" if self.left_service else ("R" if self.left_service is False else "X"),
            self.abbr_rating_info(),
        )

    def __eq__(self, other):
        return self.name == other.name

    def change_surf_advantage_side(self):
        if not self.fst_last_res or not self.snd_last_res:
            return None
        if (
            self.surface in (Clay, Grass)
            and self.rnd in (Round('q-First'), Round('q-Second'),
                             Round('First'), Round('Second'))
        ):
            srf_in_fst_hist = self.surface in self.fst_last_res.surface_list(weeks_ago=8)
            srf_in_snd_hist = self.surface in self.snd_last_res.surface_list(weeks_ago=8)
            if srf_in_fst_hist and not srf_in_snd_hist:
                return co.LEFT
            if not srf_in_fst_hist and srf_in_snd_hist:
                return co.RIGHT

    def last_res_advantage_side(self):
        if not self.fst_last_res or not self.snd_last_res:
            return None
        if (self.level in ("masters", "gs") and self.rnd >= Round("Fourth")) or (
            self.level == "main" and self.rnd >= Round("1/2")
        ):
            return None  # players are deep in tournament
        fst_long_poor = self.fst_last_res.long_poor_practice()
        snd_long_poor = self.snd_last_res.long_poor_practice()
        if fst_long_poor and not snd_long_poor:
            return co.RIGHT
        if not fst_long_poor and snd_long_poor:
            return co.LEFT

        fst_bonuses, snd_bonuses = 0, 0
        # fst_ratio = self.fst_last_res.best_win_streak_ratio(min_ratio=0.777, min_size=8)
        # snd_ratio = self.snd_last_res.best_win_streak_ratio(min_ratio=0.777, min_size=8)
        # if fst_ratio and fst_ratio > (snd_ratio + 0.2):
        #     fst_bonuses += 1
        # if not fst_ratio and snd_ratio > (fst_ratio + 0.2):
        #     snd_bonuses += 1

        fst_poor = self.fst_last_res.poor_practice()
        snd_poor = self.snd_last_res.poor_practice()
        if fst_poor and not snd_poor and self.snd_last_res.min_load_vs_poor_practice():
            snd_bonuses += 1
        elif (
            not fst_poor and snd_poor and self.fst_last_res.min_load_vs_poor_practice()
        ):
            fst_bonuses += 1

        if self.rnd in ("q-First", "q-Second"):
            fst_prev_empty = self.fst_last_res.prev_weeks_empty(weeks_num=2)
            snd_prev_empty = self.snd_last_res.prev_weeks_empty(weeks_num=2)
            if not fst_prev_empty and snd_prev_empty:
                fst_bonuses += 1
            if fst_prev_empty and not snd_prev_empty:
                snd_bonuses += 1
        elif self.rnd == "First":
            fst_prev_empty = self.fst_last_res.last_weeks_empty(weeks_num=3)
            snd_prev_empty = self.snd_last_res.last_weeks_empty(weeks_num=3)
            if not fst_prev_empty and snd_prev_empty:
                fst_bonuses += 1
            if fst_prev_empty and not snd_prev_empty:
                snd_bonuses += 1

        if fst_bonuses > 0 and snd_bonuses == 0:
            return co.LEFT
        elif snd_bonuses > 0 and fst_bonuses == 0:
            return co.RIGHT

    def vs_incomer_advantage_side(self):
        """qual vs not qual at First, or not bye vs bye at Second"""
        income_side = self.incomer_side()
        if income_side is not None:
            return income_side.fliped()

    def recently_won_h2h_side(self):
        feat = co.find_first(self.features, lambda f: f.name == "recently_winner_id")
        if feat and feat.value:
            winner_pid = feat.value
            if winner_pid == self.first_player.ident:
                return co.LEFT
            elif winner_pid == self.second_player.ident:
                return co.RIGHT

    def pre_live_name(self):
        if self.live_event and self.name:
            return "{}_{}__{}".format(
                self.live_event.sex, self.name, self.live_event.tour_name
            )

    def fill_details(self, tour):
        def find_match():
            for matches in tour.matches_from_rnd.values():
                for m in matches:
                    if (
                        not m.paired()
                        and m.score is None
                        and (
                            (
                                m.first_player.ident == self.first_player.ident
                                and m.second_player.ident == self.second_player.ident
                            )
                            or (
                                m.first_player.ident == self.second_player.ident
                                and m.second_player.ident == self.first_player.ident
                            )
                        )
                    ):
                        return m

        if (
            self.offer.win_coefs
            and self.first_draw_status is not None
            and self.second_draw_status is not None
        ):
            return  # all ok
        if not self.fill_details_tried and self.players_defined():
            match = find_match()
            if match is not None:
                self.rnd = match.rnd
                self.offer.win_coefs = match.offer.win_coefs
                self.first_draw_status = match.first_draw_status
                self.second_draw_status = match.second_draw_status
                if self.first_player.ident == match.second_player.ident:
                    self.offer.win_coefs.flip()
                    self.first_draw_status = match.second_draw_status
                    self.second_draw_status = match.first_draw_status
            self.fill_details_tried = True

    def get_ranks_cmp(self, rtg_name: str, is_surface: bool) -> ratings.CompareResult:
        if self.first_player and self.second_player:
            r1 = self.first_player.rating.rank(
                rtg_name, surface=str(self.surface) if is_surface else "all")
            r2 = self.second_player.rating.rank(
                rtg_name, surface=str(self.surface) if is_surface else "all")
            return ratings.compare_result(r1, r2, inranks=True)

    def elo_pts_dif_mixed(self, gen_part: float = 0.55, isalt: bool = True):
        elo_name = "elo_alt" if isalt else "elo"
        fst_pts = self.first_player.rating.pts(elo_name)
        snd_pts = self.second_player.rating.pts(elo_name)

        fst_srf_pts = self.first_player.rating.pts(elo_name, str(self.surface))
        snd_srf_pts = self.second_player.rating.pts(elo_name, str(self.surface))

        if (
            fst_pts is None
            or snd_pts is None
            or fst_srf_pts is None
            or snd_srf_pts is None
        ):
            raise FeatureError(
                "none {} in: 1: {}  2: {} 1s: {}  2s: {}".format(
                    elo_name, fst_pts, snd_pts, fst_srf_pts, snd_srf_pts
                )
            )
        return (snd_pts - fst_pts) * gen_part + (snd_srf_pts - fst_srf_pts) * (
            1 - gen_part
        )


class LiveTourEvent:
    def __init__(self, tour_info=None, matches=None):
        self.tour_info: Optional[TourInfo] = tour_info
        self.tour_id = None
        self.matches = matches if matches is not None else []
        self.kill_timer = None
        self.matches_key = ""
        self.filled_matches_key = (
            None  # if matches_key - need not fill matches by detail
        )
        self.features = []
        self.features_tried = False

    def need_fill_matches_details(self):
        return self.matches_key != self.filled_matches_key

    def set_matches_key(self):
        self.matches_key = "-".join((str(m.name) for m in self.matches))

    @property
    def doubles(self):
        return self.tour_info is not None and self.tour_info.doubles

    @property
    def decided_tiebreak(self) -> Optional[sc.TieInfo]:
        if self.tour_info is not None:
            return self.tour_info.decided_tiebreak

    def live_betable(self):
        if self.doubles or self.level in ("future", "junior"):
            return False
        if self.qualification and self.level not in ("gs", "masters", "main"):
            return False
        return True

    def ident_by_players(self, tours, fst_id, snd_id):
        def ident_in_tour():
            for matches in tour.matches_from_rnd.values():
                for m in matches:
                    if m.score is None and (
                        (
                            m.first_player.ident == fst_id
                            and m.second_player.ident == snd_id
                        )
                        or (
                            m.first_player.ident == snd_id
                            and m.second_player.ident == fst_id
                        )
                    ):
                        self.tour_id = tour.ident
                        self.tour_info.level = tour.level
                        self.tour_info.surface = tour.surface
                        return True
            return False

        for tour in co.find_all(tours, lambda t: t.name == self.tour_name):
            if ident_in_tour():
                return True
        return False

    def define_players(self):
        for match in self.matches:
            match.define_players(self.sex)

    def pre_live_name(self):
        if self.sex and self.tour_name:
            return "{}_{}_{}_qual{}".format(
                self.sex, self.tour_name, self.level, self.qualification
            )

    @property
    def avgset(self):
        feat = co.find_first(self.features, lambda f: f.name == "tour_avgset")
        if feat is not None:
            return feat.value

    def define_features(self):
        if not self.features_tried and self.level in (
            lev.gs, lev.main, lev.masters, lev.chal
        ):
            self.features_tried = True
            pre_live_dir.prepare_dir("tours")
            if pre_live_dir.is_data(self.pre_live_name()):
                dct = pre_live_dir.get_data(self.pre_live_name())
                for fname, fval in dct.items():
                    self.features.append(feature.RigidFeature(name=fname, value=fval))
                return

    def define_level(self):
        if self.tour_info is not None and (
            not self.tour_info.level or self.tour_id is None
        ):
            tours = weeked_tours.tail_tours(self.sex)
            for m in self.matches:
                if m.players_defined():
                    if self.ident_by_players(
                        tours, m.first_player.ident, m.second_player.ident
                    ):
                        return
            if not self.tour_info.level and self.tour_info.itf:
                self.tour_info.level = lev.future

    def fill_matches_details(self):
        if self.tour_info is not None and self.tour_id is not None:
            tour = co.find_first(
                weeked_tours.tail_tours(self.sex), lambda t: t.ident == self.tour_id
            )
            if tour is None:
                log.error(
                    "no found tour by id {} for {}".format(self.tour_id, str(self))
                )
                return
            for match in self.matches:
                match.fill_details(tour)
        self.filled_matches_key = copy.copy(self.matches_key)

    @property
    def sex(self):
        if self.tour_info is not None:
            return self.tour_info.sex

    @property
    def level(self):
        if self.tour_info is not None:
            return self.tour_info.level

    @property
    def surface(self):
        if self.tour_info is not None:
            return self.tour_info.surface

    @property
    def best_of_five(self):
        if self.tour_info is not None:
            return self.tour_info.best_of_five

    @property
    def qualification(self):
        if self.tour_info is not None:
            return self.tour_info.qualification

    @property
    def tour_name(self) -> Optional[TourName]:
        if self.tour_info:
            return self.tour_info.tour_name

    name = tour_name

    def __str__(self):
        return str(self.tour_info)

    def __eq__(self, other):
        return self.tour_info == other.tour_info
