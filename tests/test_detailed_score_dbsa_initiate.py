from os.path import isfile
import datetime
from enum import Enum

import pytest

import file_utils as fu
import common as co
import detailed_score_dbsa as dbsa
from detailed_score import DetailedScore, DetailedGame
from score import Score
from tennis import Round


DB_INSTANCE_TEST = 5
DBTYPENAME = dbsa.SQLITE_TYPENAME
SEX = "wta"

if DB_INSTANCE_TEST == dbsa.DEFAULT_DB_INSTANCE:
    raise ValueError(f'testing dbsa instance {DB_INSTANCE_TEST} == work dbsa instance')


def make_mrec_0():
    items = [
        (
            ((0, 0),),
            DetailedGame("1111", left_wingame=True, left_opener=True, tiebreak=False),
        ),
        (
            ((1, 0),),
            DetailedGame("0000", left_wingame=True, left_opener=False, tiebreak=False),
        ),
    ]
    retired = True
    ds = DetailedScore(items, retired=retired)
    scr = Score.from_pairs(ds.final_score(), retired=retired)
    return dbsa.MatchRec(
        date=datetime.date(2020, 6, 24),
        tour_id=101,
        rnd=Round("1/4"),
        left_id=555,
        right_id=111,
        detailed_score=ds,
        score=scr,
    )


def make_mrec_1():
    items = [
        (
            ((0, 0),),
            DetailedGame("11011", left_wingame=True, left_opener=True, tiebreak=False),
        ),
        (
            ((1, 0),),
            DetailedGame("00100", left_wingame=True, left_opener=False, tiebreak=False),
        ),
    ]
    ds = DetailedScore(items, retired=False)
    scr = Score.from_pairs(ds.final_score(), retired=False)
    return dbsa.MatchRec(
        date=datetime.date(2020, 6, 25),
        tour_id=101,
        rnd=Round("1/2"),
        left_id=555,
        right_id=222,
        detailed_score=ds,
        score=scr,
    )


def make_mrec_2():
    items = [
        (
            ((0, 0),),
            DetailedGame("100111", left_wingame=True, left_opener=True, tiebreak=False),
        ),
        (
            ((1, 0),),
            DetailedGame(
                "011000", left_wingame=True, left_opener=False, tiebreak=False
            ),
        ),
    ]
    ds = DetailedScore(items, retired=False)
    scr = Score.from_pairs(ds.final_score(), retired=False)
    return dbsa.MatchRec(
        date=datetime.date(2020, 6, 26),
        tour_id=101,
        rnd=Round("Final"),
        left_id=555,
        right_id=333,
        detailed_score=ds,
        score=scr,
    )


def remove_db_file(sex: str):
    sexes = ("wta", "atp") if sex is None else (sex,)
    for sex in sexes:
        filename = dbsa.dbfilename(sex, DB_INSTANCE_TEST, DBTYPENAME)
        if isfile(filename):
            fu.remove_file(filename)


def add_few_matches_to_db(handle):
    m0 = make_mrec_0()
    m1 = make_mrec_1()
    m2 = make_mrec_2()
    handle.insert_obj(m1)
    handle.insert_obj(m2)
    handle.insert_obj(m0)
    handle.commit()


def create_few_matches_db(sex):
    remove_db_file(sex)
    dbsa.create_empty_db(sex=sex, instance=DB_INSTANCE_TEST, dbtypename=DBTYPENAME)
    handle = dbsa.open_db(sex, instance=DB_INSTANCE_TEST, dbtypename=DBTYPENAME)
    add_few_matches_to_db(handle)
    return handle


class InitMode(Enum):
    CREATE = 1  # create db and commit stage
    OPEN = 2  # open existed db stage


init_mode = InitMode.OPEN


@pytest.fixture(scope="module")
def all_dbdet_init():
    # Setup :
    if init_mode == InitMode.CREATE:
        handle = create_few_matches_db(sex=SEX)
    elif init_mode == InitMode.OPEN:
        handle = dbsa.open_db(sex=SEX, instance=DB_INSTANCE_TEST, dbtypename=DBTYPENAME)
        handle.query_matches()
    else:
        raise ValueError(f'unsupported init_mode {init_mode}')

    yield handle  # this is where the testing happens


@pytest.mark.detscoredb
class TestQueries:
    @staticmethod
    def test_min_max_date(all_dbdet_init: dbsa.Handle):
        min_date = all_dbdet_init.query_min_date()
        max_date = all_dbdet_init.query_max_date()
        assert min_date is not None and max_date is not None
        assert isinstance(min_date, datetime.date)
        assert isinstance(max_date, datetime.date)
        assert min_date <= max_date

    @staticmethod
    def test_find(all_dbdet_init: dbsa.Handle):
        s = all_dbdet_init.session
        mrecs = (
            s.query(dbsa.MatchRec)
            .filter(
                (datetime.date(2020, 6, 24) < dbsa.MatchRec.date)
                & (dbsa.MatchRec.date < datetime.date(2020, 6, 26))
            )
            .all()
        )
        assert bool(mrecs)
        if mrecs:
            assert len(mrecs) == 1
            tst_mrec = make_mrec_1()
            assert mrecs[0] == tst_mrec
            assert mrecs[0].score == tst_mrec.score

    @staticmethod
    def test_get_all_order_by(all_dbdet_init: dbsa.Handle):
        s = all_dbdet_init.session
        mrecs = s.query(dbsa.MatchRec).order_by(dbsa.MatchRec.date).all()
        assert bool(mrecs)
        if mrecs:
            assert len(mrecs) == 3
            assert mrecs[0] == make_mrec_0()
            assert mrecs[1] == make_mrec_1()
            assert mrecs[2] == make_mrec_2()

    @staticmethod
    def test_get_all_order_by_desc(all_dbdet_init: dbsa.Handle):
        s = all_dbdet_init.session
        mrecs = s.query(dbsa.MatchRec).order_by(dbsa.MatchRec.date.desc()).all()
        assert bool(mrecs)
        if mrecs:
            assert len(mrecs) == 3
            assert mrecs[0] == make_mrec_2()
            assert mrecs[1] == make_mrec_1()
            assert mrecs[2] == make_mrec_0()

    @staticmethod
    def test_query_matches(all_dbdet_init: dbsa.Handle):
        all_dbdet_init.query_matches(min_date=datetime.date(2020, 6, 25))
        assert bool(all_dbdet_init.records)
        if all_dbdet_init.records:
            assert len(all_dbdet_init.records) == 2
            assert all_dbdet_init.records[0] == make_mrec_1()
            assert all_dbdet_init.records[1] == make_mrec_2()


def delete_record(sex, tour_id, rnd, left_id, right_id):
    """ простая ф-я из отдельного сценария (main). Пока не оформлена как test """
    handle = dbsa.open_db(sex, instance=DB_INSTANCE_TEST)
    handle.query_matches()
    mrec = dbsa.MatchRec(
        date=datetime.date.today(),
        tour_id=tour_id,
        rnd=rnd,
        left_id=left_id,
        right_id=right_id,
        detailed_score="",
        score="",
    )
    mrecs = co.find_all(
        handle.records,
        (
            lambda o: o.tour_id == mrec.tour_id
            and o.rnd == mrec.rnd
            and o.left_id == mrec.left_id
            and o.right_id == mrec.right_id
        ),
    )
    if mrecs and len(mrecs) == 1:
        handle.delete_obj(mrecs[0])
        handle.commit()
        print(
            f"deleted {sex} tour_id {tour_id} rnd {rnd} pid1 {left_id} pid2 {right_id}"
        )
    else:
        print(f"fail delete mrecs {mrecs}")
        print(
            f"fail delete {sex} tour_id {tour_id} rnd {rnd} pid1 {left_id} pid2 {right_id}"
        )

