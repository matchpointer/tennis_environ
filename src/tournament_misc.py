# -*- coding=utf-8 -*-
import datetime
from collections import Counter
import copy

from loguru import logger as log
import common as co
import player_name_ident
import flashscore_match


def log_events(events, head="", extended=True, flush=False):
    log.info(events_tostring(events, head, extended))
    if flush:
        log.flush()


def events_tostring(events, head="", extended=True):
    result = "-----{}------ {} events  -------------\n".format(head, len(events))
    for event in events:
        result += "{}\n".format(event)
        for match in event.matches:
            result += "\t{}\n".format(match.tostring(extended=extended))
    return result


def print_events(events, head="", extended=False):
    """for debug purpose"""
    print(head)
    for event in events:
        print(str(event))
        for match in event.matches:
            print(match.tostring(extended=extended))


def events_deep_ident(tour_events, wta_tours, atp_tours, from_scored, warnloghead=""):
    warn_cnt = 0
    if len(tour_events) > 0:
        _ident_players(tour_events)
        noid_matches = _log_noident_players(tour_events)
        _ident_tours(tour_events, wta_tours, atp_tours, from_scored)
        unk_id_tevents = [t for t in tour_events if t.tour_id is None]
        if len(unk_id_tevents) > 0:
            log_events(unk_id_tevents, head="{} unk_id_tours".format(warnloghead))
        if noid_matches and from_scored:
            _reident_players(noid_matches, wta_tours, atp_tours)
        warn_cnt = _ident_rnd_date(tour_events, wta_tours, atp_tours, from_scored)
    return warn_cnt


def tour_events_parse_detailed_score(tour_events, fsdrv):
    """suppose that tour_events are already idented (match.rnd, match.date,...)"""
    n_errors = 0
    for tour_evt in tour_events:
        if tour_evt.tour_id is None:
            continue
        for match in tour_evt.matches:
            if match.rnd is not None:
                res = flashscore_match.parse_match_detailed_score(match, fsdrv)
                if res is False:
                    n_errors += 1
    return n_errors


def _ident_tours(tour_events, wta_tours, atp_tours, from_scored):
    def known_players_ids(matches):
        return set(
            [
                m.first_player.ident
                for m in matches
                if m.first_player is not None
                and m.first_player.ident
                and (m.score is not None) is from_scored
            ]
        ) | set(
            [
                m.second_player.ident
                for m in matches
                if m.second_player is not None
                and m.second_player.ident
                and (m.score is not None) is from_scored
            ]
        )

    def ident_tour_by_name_surf(tour_evt, all_tours, all_knowns):
        def proved_tour_by_knowns(tour):
            if len(all_knowns) >= 2:
                intersect_ids = all_knowns & set(tour.players_id_list())
                return len(intersect_ids) > 0

        our_name_tours = [t for t in all_tours if t.name == tour_evt.tour_name]
        if len(our_name_tours) == 1:
            if proved_tour_by_knowns(our_name_tours[0]) is not False:
                return our_name_tours[0]
            else:
                log.warning(
                    "finded tour {} NOT PROVED1 with {}".format(
                        our_name_tours[0], tour_evt
                    )
                )
        if len(our_name_tours) > 1 and tour_evt.surface:
            our_name_surf_tours = [
                t for t in our_name_tours if tour_evt.surface == t.surface
            ]
            if len(our_name_surf_tours) == 1:
                if proved_tour_by_knowns(our_name_surf_tours[0]) is not False:
                    return our_name_surf_tours[0]
                else:
                    log.warning(
                        "finded tour {} NOT PROVED2 with {}".format(
                            our_name_surf_tours[0], tour_evt
                        )
                    )
        log.warning(
            "tevt {} NOT ID-by-name-srf n_names: {}".format(
                tour_evt, len(our_name_tours)
            )
        )

    def ident_tour_by_match_players(match, all_tours):
        def two_players_in_tour(two_players_id_list, tour):
            tour_match_ids = tour.players_id_list(match_splited=True)
            if (
                tuple(two_players_id_list) in tour_match_ids
                or tuple(reversed(two_players_id_list)) in tour_match_ids
            ):
                return True
            return False

        def match_known_players_id_list():
            result = []
            if match.first_player is not None and match.first_player.ident is not None:
                result.append(match.first_player.ident)
            if (
                match.second_player is not None
                and match.second_player.ident is not None
            ):
                result.append(match.second_player.ident)
            return result

        our_knowns = match_known_players_id_list()
        if len(our_knowns) == 2:
            tour_flag_lst = [(t, two_players_in_tour(our_knowns, t)) for t in all_tours]
            if [flag for (_, flag) in tour_flag_lst].count(True) == 1:
                return [t for (t, flag) in tour_flag_lst if flag][0]
        if len(our_knowns) == 1:
            our_tours = [t for t in all_tours if our_knowns[0] in t.players_id_list()]
            if len(our_tours) == 1:
                return our_tours[0]

    for tour_evt in tour_events:
        all_tours = wta_tours if tour_evt.sex == "wta" else atp_tours
        all_knowns = known_players_ids(tour_evt.matches)
        tour = ident_tour_by_name_surf(tour_evt, all_tours, all_knowns)
        if tour is not None and tour.ident is not None:
            tour_evt.tour_id = tour.ident
            continue

        candidate_tours = [
            ident_tour_by_match_players(m, all_tours) for m in tour_evt.matches
        ]
        tours_cnt = Counter([tour for tour in candidate_tours if tour is not None])
        if tours_cnt:
            tour_evt.tour_id = tours_cnt.most_common(1)[0][0].ident


def _ident_rnd_date(tour_events, wta_tours, atp_tours, from_scored):
    def ident_match_rnd_date(match, tour):
        today_date = datetime.date.today()
        for rnd, matches in tour.matches_from_rnd.items():
            for m in matches:
                if not m.paired() and ((m.score is not None) is from_scored):
                    if (
                        m.first_player == match.first_player
                        and m.second_player == match.second_player
                    ) or (
                        m.first_player == match.second_player
                        and m.second_player == match.first_player
                    ):
                        match.rnd = rnd
                        if m.date is not None:
                            if (
                                match.date is not None
                                and abs((m.date - match.date).days) > 1
                            ):
                                log.warning(
                                    "{} mdatedif rn: {}, m: {}, match: {} q:{}".format(
                                        tour.sex, rnd, m, match, match.qualification
                                    )
                                )
                                continue
                            if m.date <= (today_date + datetime.timedelta(days=1)):
                                match.date = m.date
                            else:
                                log.warning(
                                    "{}mdate-fut {} found\n{} identing\n{}".format(
                                        tour.sex, m.date, m, match
                                    )
                                )
                                continue
                        return True

    n_warns = 0
    for tour_evt in tour_events:
        all_tours = wta_tours if tour_evt.sex == "wta" else atp_tours
        if tour_evt.tour_id is not None:
            tour = co.find_first(all_tours, lambda t: t.ident == tour_evt.tour_id)
            assert tour is not None, "not found early idented tour {}".format(tour_evt)
            for match in tour_evt.matches:
                if not ident_match_rnd_date(match, tour):
                    n_warns += 1
                    log.warning(
                        "in {} {} not idented rnd-date match: {}".format(
                            tour_evt.sex, tour.name, match
                        )
                    )
    return n_warns


def _log_noident_players(tour_events):
    noid_matches = []  # here only one player is not idented
    for tour_evt in tour_events:
        for match in tour_evt.matches:
            fst_ok, snd_ok = True, True
            if match.first_player is None or match.first_player.ident is None:
                fst_ok = False
                log.warning(
                    "IN {} not idented [1] match: {}".format(tour_evt.tour_name, match)
                )
            if match.second_player is None or match.second_player.ident is None:
                snd_ok = False
                log.warning(
                    "IN {} not idented [2] match: {}".format(tour_evt.tour_name, match)
                )
            if (int(fst_ok) + int(snd_ok)) == 1:
                noid_matches.append(match)
    return noid_matches


def _ident_players(tour_events):
    def get_abbr_name_side(name, side):
        abbr_names = name.split(" - ")
        if len(abbr_names) == 2:
            result = abbr_names[0] if side.is_left() else abbr_names[1]
            return result.strip()

    for tour_evt in tour_events:
        for match in tour_evt.matches:
            if match.first_player is None or match.first_player.ident is None:
                match.first_player = player_name_ident.identify_player(
                    tour_evt.sex, get_abbr_name_side(match.name, co.LEFT)
                )
            if match.second_player is None or match.second_player.ident is None:
                match.second_player = player_name_ident.identify_player(
                    tour_evt.sex, get_abbr_name_side(match.name, co.RIGHT)
                )


def _reident_players(noid_matches, wta_tours, atp_tours):
    def candidate_players():
        result = []
        for rnd, matches in the_tour.matches_from_rnd.items():
            if rnd.qualification() != match.live_event.qualification:
                continue
            for mch in matches:
                if mch.paired() or not mch.score:
                    continue
                if known_id not in (mch.first_player.ident, mch.second_player.ident):
                    continue
                if known_side == win_side and known_id != mch.first_player.ident:
                    continue
                if known_side != win_side and known_id != mch.second_player.ident:
                    continue
                if mch.score != winoriented_score:
                    continue
                if (
                    match.date
                    and mch.date
                    and abs(match.date - mch.date) > datetime.timedelta(days=1)
                ):
                    continue
                unk_plr = (
                    mch.first_player
                    if known_id == mch.second_player.ident
                    else mch.second_player
                )
                if not unk_plr.name.endswith(unk_last_name):
                    continue
                result.append(unk_plr)
        return result

    for match in noid_matches:
        if (
            match.live_event is None
            or match.live_event.tour_id is None
            or match.live_event.qualification is None
            or not match.score
        ):
            continue
        tours = wta_tours if match.sex == "wta" else atp_tours
        the_tours = co.find_all(tours, lambda t: t.ident == match.live_event.tour_id)
        if len(the_tours) != 1:
            continue
        known_side = co.LEFT if (match.first_player is not None) else co.RIGHT
        if known_side == co.LEFT:
            known_id = match.first_player.ident
        else:
            known_id = match.second_player.ident
        last_set = match.score[-1]
        win_side = co.LEFT if last_set[0] > last_set[1] else co.RIGHT
        winoriented_score = (
            match.score.fliped() if win_side == co.RIGHT else match.score
        )
        abbr_names = match.name.split(" - ")
        unk_abbr_name = abbr_names[0] if known_side == co.RIGHT else abbr_names[1]
        unk_last_name = unk_abbr_name.split(" ")[0]
        the_tour = the_tours[0]
        candidates = candidate_players()
        if len(candidates) != 1:
            continue
        if known_side == co.LEFT:
            match.second_player = copy.deepcopy(candidates[0])
        else:
            match.first_player = copy.deepcopy(candidates[0])
        log.info(
            "ok reidented player {} for {} in {}".format(
                candidates[0], match.name, match.live_event
            )
        )
