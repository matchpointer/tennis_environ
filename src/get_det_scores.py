# -*- coding=utf-8 -*-
"""
access to some concrete scores (for test purposes)
"""
from typing import Tuple

from score import Score
from detailed_score_dbsa import Handle
from detailed_score import DetailedScore


def get_match_scores(dbdet: Handle, tour_id, rnd, left_id, right_id
                     ) -> Tuple[DetailedScore, Score]:
    det_score = dbdet.get_detailed_score(
        tour_id=tour_id, rnd=rnd, left_id=left_id, right_id=right_id
    )
    scor = dbdet.get_score(
        tour_id=tour_id, rnd=rnd, left_id=left_id, right_id=right_id
    )
    return det_score, scor


def scores_boulter_makarova_2019_ao(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 6-0 4-6 7-6 (supertie: 10-6, max dif was 9-6). AO 2019.
    return get_match_scores(
        dbdet, tour_id=12357, rnd="First", left_id=14175, right_id=4159)


def scores_gracheva_mladenovic(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 1-6 7-6(2) 6-0
    return get_match_scores(
        dbdet, tour_id=13059, rnd="Second", left_id=44510, right_id=9458)


def scores_koukalova_safarova(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 6-0 6-7(3) 6-4
    # Koukalova leads in set2: 6-5 0-0 rcv; tie 1-0. But she losed set2.
    return get_match_scores(
        dbdet, tour_id=7233, rnd="Second", left_id=454, right_id=4089)


def scores_navarro_floris_fes2010(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 6-1 5-7 6-1
    return get_match_scores(
        dbdet, tour_id=6602, rnd="First", left_id=5873, right_id=661)


def scores_lepchenko_tomova_ao2019(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 4-6 6-2 7-6 (supertie: 10-6, max dif was 9-4).
    return get_match_scores(
        dbdet, tour_id=12357, rnd="q-First", left_id=461, right_id=12432)


def scores_kvitova_goerges_spb2018(dbdet: Handle) -> Tuple[DetailedScore, Score]:
    # 7-5 4-6 6-2 Kvitova opens set1, at 5-5 she holds, then at 6-5 break
    return get_match_scores(
        dbdet, tour_id=11757, rnd="1/2", left_id=8234, right_id=7102)

