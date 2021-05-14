# -*- coding: utf-8 -*-
import datetime

import log
import common as co
import bet
import weeked_tours

# import clf_openset_versus_incomer
import clf_qual_at_first_cb


wta_tail_tours = list()
atp_tail_tours = list()


def initialize():
    clf_qual_at_first_cb.initialize()
    fill_tail_tours()


def get_tail_tours(sex):
    return wta_tail_tours if sex == "wta" else atp_tail_tours


def fill_tail_tours():
    """tail tours are recent from weeked_tours which will help identify current events"""
    tail_weeks = 2
    if datetime.date.today().isoweekday() >= 6:
        tail_weeks = 3
    atp_tours = weeked_tours.tail_tours(sex="atp", tail_weeks=tail_weeks)
    if not atp_tours:
        log.warn("empty atp tail_tours {}".format(tail_weeks))
    else:
        atp_tail_tours.extend(atp_tours)
        # tours_write_file(atp_tours, filename='./atp_tours_tail.txt')
    wta_tours = weeked_tours.tail_tours(sex="wta", tail_weeks=tail_weeks)
    if not wta_tours:
        log.warn("empty wta tail_tours {}".format(tail_weeks))
    else:
        wta_tail_tours.extend(wta_tours)
        # tours_write_file(wta_tours, filename='./wta_tours_tail.txt')


class QualAtFirst(object):
    def __init__(self):
        pass

    def find_picks(self, today_match):
        result = []
        if today_match.sex == "atp" or today_match.level not in (
            "main",
            "gs",
            "masters",
        ):
            return result
        fst_draw, snd_draw = (
            today_match.first_draw_status,
            today_match.second_draw_status,
        )
        if (
            today_match.rnd == "First"
            and "qual" in (fst_draw, snd_draw)
            and not (fst_draw == "qual" and snd_draw == "qual")
        ):
            qual_side = co.LEFT if fst_draw == "qual" else co.RIGHT
            qual_id = (
                today_match.first_player.ident
                if fst_draw == "qual"
                else today_match.second_player.ident
            )
            other_id = (
                today_match.second_player.ident
                if qual_id == today_match.first_player.ident
                else today_match.first_player.ident
            )
            # tour_id = today_match.live_event.tour_id
            # tour = co.find_first(get_tail_tours(today_match.sex),
            #                      lambda t: t.ident == tour_id)
            # if tour is None:
            #     log.error("not found tour #{} '{}' among tail_tours for {}".format(
            #         tour_id, today_match.live_event.tour_name, today_match))
            #     return result
            back_side, proba = clf_qual_at_first_cb.match_has_min_proba(
                today_match, qual_side
            )
            if back_side in (co.LEFT, co.RIGHT):
                back_plr_id = qual_id if back_side == qual_side else other_id
                result.append(
                    bet.PickWin(
                        today_match.offer.company,
                        None,
                        back_plr_id,
                        margin=1.0,
                        probability=proba,
                        explain=self.__class__.__name__,
                    )
                )
        return result


# class FirstSetVsNofited(object):
#     """ интересна ситуация (для лайв) когда прибитый (fited) игрок
#         может выиграть 1 сет против только что вышедшего на этот турнир """
#     def __init__(self):
#         pass
#
#     def find_picks(self, today_match):
#         def fited_better_at_first_set(fst_plr_fited, min_size=20, min_diff=2.5):
#             fst_sval = tie_importance_stat.personal_value(today_match.sex, 'open_sets',
#                                                   today_match.first_player.ident)
#             snd_sval = tie_importance_stat.personal_value(today_match.sex, 'open_sets',
#                                                   today_match.second_player.ident)
#             if (not fst_sval or not snd_sval or
#                     fst_sval.size < min_size or snd_sval.size < min_size):
#                 return False
#
#             if fst_plr_fited:
#                 fited_val = fst_sval.value
#                 other_val = snd_sval.value
#             else:
#                 fited_val = snd_sval.value
#                 other_val = fst_sval.value
#             return (other_val + min_diff) <= fited_val
#
#         result = []
#         qual_at_fst = (
#             today_match.rnd == 'First' and
#             'qual' in (today_match.first_draw_status, today_match.second_draw_status) and
#             not (today_match.first_draw_status == 'qual' and
#                  today_match.second_draw_status == 'qual') and
#             not (today_match.sex == 'atp' and today_match.level == 'gs')
#         )
#         bye_at_snd = (
#             today_match.rnd == 'Second' and
#             'bye' in (today_match.first_draw_status, today_match.second_draw_status)
#         )
#         if today_match.level == 'future' or (not qual_at_fst and not bye_at_snd):
#             return result
#         if qual_at_fst:
#             fst_plr_fited = today_match.first_draw_status == 'qual'
#         else:
#             fst_plr_fited = today_match.second_draw_status == 'bye'
#         if fited_better_at_first_set(fst_plr_fited):
#             win_coefs = copy.deepcopy(today_match.offer.win_coefs)
#             coef = win_coefs.first_coef if fst_plr_fited else win_coefs.second_coef
#             fited_id = (today_match.first_player.ident if fst_plr_fited
#                         else today_match.second_player.ident)
#             if 1.5 <= coef <= 4.0:
#                 result.append(bet.PickWin(
#                     today_match.offer.company, coef, fited_id, margin=1.,
#                     probability=0.5, explain=self.__class__.__name__))
#         return result
#
#
# class VsLowLoadIncomer(object):
#     """ против только что вышедшего на этот турнир без практики в прошлой неделе.
#         First rnd with qualifier. Second rnd as bye """
#     def __init__(self, sex):
#         date = tt.past_monday_date(datetime.date.today())
#         self.tours_cache = tours_cache.get_cache(sex, date)
#
#     def find_picks(self, today_match):
#         result = []
#         incomer_side = today_match.incomer_side()
#         if incomer_side is None or today_match.level not in ('gs', 'masters', 'main'):
#             return result
#         if incomer_side.is_left():
#             incomer_plr = today_match.first_player
#             back_plr = today_match.second_player
#         else:
#             incomer_plr = today_match.second_player
#             back_plr = today_match.first_player
#         load_obj = load.player_load(incomer_plr, today_match, self.tours_cache,
#                                     weeks_before_current=1, units=load.Units.matches)
#         if load_obj.get_value(units=load.Units.matches) > 0.2:
#             return result
#         result.append(bet.PickWin(today_match.offer.company,
#                                   coef=1., sider_id=back_plr.ident, margin=1.,
#                                   probability=0.5, explain=self.__class__.__name__))
#         return result
