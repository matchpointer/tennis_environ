# -*- coding: utf-8 -*-
""" database of MatchRec with DetailedScore
"""
import os
import datetime

from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    MetaData,
    Date,
    PrimaryKeyConstraint,
)

import cfg_dir
import common as co
from score import Score
from detailed_score import DetailedScore

DEFAULT_DB_INSTANCE = 4


metadata = MetaData()

matches_table = Table(
    "matches",
    metadata,
    Column("date", Date, index=True),
    Column("tour_id", Integer),
    Column("rnd", String(16)),
    Column("left_id", Integer),
    Column("right_id", Integer),
    Column("score_txt", String(60)),
    Column("detailed_score_txt", String),
    PrimaryKeyConstraint("tour_id", "rnd", "left_id", "right_id"),
)


class MatchRec(object):
    def __init__(self, date, tour_id, rnd, left_id, right_id, detailed_score, score):
        self.date = date
        self.tour_id = tour_id
        self.rnd = rnd if isinstance(rnd, str) else rnd.value
        self.left_id = left_id
        self.right_id = right_id
        self.score_txt = str(score) if isinstance(score, Score) else score
        self.detailed_score_txt = (
            detailed_score.tostring()
            if isinstance(detailed_score, DetailedScore)
            else detailed_score
        )

    def __eq__(self, other):
        return (
            self.tour_id == other.tour_id
            and self.rnd == other.rnd
            and self.left_id == other.left_id
            and self.right_id == other.right_id
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.tour_id, self.rnd, self.left_id, self.right_id))

    @property
    def detailed_score(self):
        if self.detailed_score_txt:
            return DetailedScore.from_str(self.detailed_score_txt)

    @property
    def score(self):
        if self.score_txt:
            return Score(self.score_txt)

    def __repr__(self):
        return "{}({}, {}, '{}', {}, {}, '{}', '{}')".format(
            self.__class__.__name__,
            repr(self.date),
            self.tour_id,
            self.rnd,
            self.left_id,
            self.right_id,
            self.score_txt,
            self.detailed_score_txt,
        )

    def __str__(self):
        return self.tostring(is_det_score=False)

    def tostring(self, is_det_score=False):
        det_score_txt = "\n{}".format(self.detailed_score_txt) if is_det_score else ""
        return "{} tour={} {} fst=={} snd={} {}{}".format(
            self.date,
            self.tour_id,
            self.rnd,
            self.left_id,
            self.right_id,
            self.score_txt,
            det_score_txt,
        )


mapper(
    MatchRec,
    matches_table,
    properties={
        "date": matches_table.c.date,
        "tour_id": matches_table.c.tour_id,
        "rnd": matches_table.c.rnd,
        "left_id": matches_table.c.left_id,
        "right_id": matches_table.c.right_id,
        "score_txt": matches_table.c.score_txt,
        "detailed_score_txt": matches_table.c.detailed_score_txt,
    },
)


class Handle:
    def __init__(self, engine, session):
        self.engine = engine
        self.session = session
        self.records = None  # list of MatchRec

    def commit(self):
        self.session.commit()

    def insert_obj(self, record):
        self.session.add(record)

    def delete_obj(self, record):
        self.session.delete(record)

    def query_matches(self, min_date=None, max_date=None):
        if min_date is None and max_date is None:
            self.records = self.session.query(MatchRec).order_by(MatchRec.date).all()
        elif min_date is not None and max_date is None:
            self.records = (
                self.session.query(MatchRec)
                .filter(min_date <= MatchRec.date)
                .order_by(MatchRec.date)
                .all()
            )
        elif min_date is None and max_date is not None:
            self.records = (
                self.session.query(MatchRec)
                .filter(MatchRec.date < max_date)
                .order_by(MatchRec.date)
                .all()
            )
        elif min_date is not None and max_date is not None:
            self.records = (
                self.session.query(MatchRec)
                .filter((min_date <= MatchRec.date) & (MatchRec.date < max_date))
                .order_by(MatchRec.date)
                .all()
            )

    def query_match(self, tour_id, rnd, left_id, right_id):
        rec = (
            self.session.query(MatchRec)
            .filter(
                (MatchRec.tour_id == tour_id)
                & (MatchRec.rnd == rnd)
                & (MatchRec.left_id == left_id)
                & (MatchRec.right_id == right_id)
            )
            .first()
        )
        if rec is None:
            self.records = []
        else:
            self.records = [rec]

    def query_min_date(self):
        return get_min_date(self.engine)

    def query_max_date(self):
        return get_max_date(self.engine)

    def get_min_date(self):
        if self.records:
            # assume that matches ordered by date asc
            return self.records[0].date

    def get_max_date(self):
        if self.records:
            # assume that matches ordered by date asc
            return self.records[-1].date

    def get_detailed_score(self, tour_id, rnd, left_id, right_id):
        if self.records:
            match_rec = co.find_first(
                self.records,
                (
                    lambda r: r.tour_id == tour_id
                    and r.rnd == rnd
                    and r.left_id == left_id
                    and r.right_id == right_id
                ),
            )
            if match_rec is not None:
                return match_rec.detailed_score

    def get_score(self, tour_id, rnd, left_id, right_id):
        if self.records:
            match_rec = co.find_first(
                self.records,
                (
                    lambda r: r.tour_id == tour_id
                    and r.rnd == rnd
                    and r.left_id == left_id
                    and r.right_id == right_id
                ),
            )
            if match_rec is not None:
                return match_rec.score


max_date_stmt = text("""SELECT MAX(date) FROM matches""")
min_date_stmt = text("""SELECT MIN(date) FROM matches""")


def do_scalar_select(engine, stmt):
    with engine.connect() as con:
        rs = con.execute(stmt)
        if rs is not None:
            return next(iter(rs))[0]


def get_max_date(engine):
    date_txt = do_scalar_select(engine, max_date_stmt)
    if date_txt:
        return datetime.datetime.strptime(date_txt, "%Y-%m-%d").date()


def get_min_date(engine):
    date_txt = do_scalar_select(engine, min_date_stmt)
    if date_txt:
        return datetime.datetime.strptime(date_txt, "%Y-%m-%d").date()


def find_match_rec_by(session, tour_id, rnd, left_id, right_id):
    return (
        session.query(MatchRec)
        .filter_by(tour_id=tour_id, rnd=rnd, left_id=left_id, right_id=right_id)
        .first()
    )


SQLITE_TYPENAME = "sqlite"

_opened = dict()


def open_db(sex, instance=DEFAULT_DB_INSTANCE, dbtypename=SQLITE_TYPENAME):
    if (sex, instance, dbtypename) in _opened:
        raise KeyError("already opened {} {} {}".format(sex, instance, dbtypename))
    e = make_engine(sex, instance, dbtypename)
    s = make_session(e)
    _opened[(sex, instance, dbtypename)] = (e, s)
    return Handle(engine=e, session=s)


def make_engine(sex, instance, dbtypename=SQLITE_TYPENAME):
    return create_engine(
        "{}:///{}".format(
            dbtypename,
            os.path.abspath(dbfilename(sex, instance=instance, dbtypename=dbtypename)),
        )
    )


def make_session(engine):
    return sessionmaker(bind=engine)()


def create_empty_db(sex, instance, dbtypename=SQLITE_TYPENAME):
    """GIVEN: no db file
    THEN create empty db file"""
    e = make_engine(sex, instance=instance, dbtypename=dbtypename)
    metadata.create_all(e)


def dbfilename(sex, instance, dbtypename=SQLITE_TYPENAME):
    return "{}_{}/{}_{}".format(cfg_dir.detailed_score_dir(), instance, sex, dbtypename)
