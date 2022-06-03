"""
database of classifier's predicts
"""
import os
import datetime
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import (
    Table,
    Column,
    Integer,
    Enum as DbEnum,
    String,
    Float,
    MetaData,
    Date,
    PrimaryKeyConstraint,
)

import cfg_dir
import common as co


metadata = MetaData()


MAX_COMMENTS_LEN = 36

predicts_table = Table(
    "predicts",
    metadata,
    Column("sex", String(4)),
    Column("date", Date, index=True),
    Column("case_name", String(24)),
    Column("tour_name", String(36)),
    Column("level", String(16)),
    Column("surface", String(16)),
    Column("rnd", String(16)),
    Column("back_id", Integer),
    Column("oppo_id", Integer),
    Column("predict_proba", Float),
    Column("predict_result", Integer),
    Column("comments", String(MAX_COMMENTS_LEN)),
    Column("back_name", String(40)),
    Column("oppo_name", String(40)),
    Column("book_start_chance", Float),
    Column("rejected", Integer),
    Column("back_win_match", Integer),
    Column("bf_live_coef", Float),
    Column("bf_live_coef_matched", Float),
    Column("clf_hash", String(50)),
    PrimaryKeyConstraint("sex", "date", "case_name", "rnd", "back_id", "oppo_id"),
)


class PredictRec:
    def __init__(self, sex: str, date: datetime.date, case_name: str,
                 tour_name: str, level, surface, rnd, back_id: int, oppo_id: int,
                 predict_proba: float, predict_result: int, comments: str,
                 back_name: str, oppo_name: str,
                 book_start_chance: float, rejected: int = -1,
                 back_win_match: int = -1, bf_live_coef: float = None,
                 bf_live_coef_matched: float = None, clf_hash: str = None
                 ):
        self.sex = sex
        self.date = date
        self.case_name = case_name
        self.tour_name = str(tour_name)
        self.level = str(level)
        self.surface = str(surface)
        self.rnd = str(rnd)
        self.back_id = back_id
        self.oppo_id = oppo_id
        self.predict_proba = predict_proba
        self.predict_result = predict_result
        self.comments = comments
        self.back_name = back_name
        self.oppo_name = oppo_name
        self.book_start_chance = book_start_chance
        self.rejected = rejected
        self.back_win_match = back_win_match
        self.bf_live_coef = bf_live_coef
        self.bf_live_coef_matched = bf_live_coef_matched
        self.clf_hash = clf_hash

    def get_predict_result(self):
        return co.PredictResult(self.predict_result)

    def __eq__(self, other):
        return (
            self.sex == other.sex
            and self.date == other.date
            and self.case_name == other.case_name
            and self.rnd == other.rnd
            and self.back_id == other.back_id
            and self.oppo_id == other.oppo_id
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(
            (self.sex, self.date, self.case_name, self.rnd, self.back_id, self.oppo_id)
        )

    def __repr__(self):
        return (
            f"{self.date} {self.sex} case {self.case_name} tour {self.tour_name}"
            f" lev {self.level} srf {self.surface} rnd {self.rnd}"
            f" back {self.back_id} oppo {self.oppo_id}"
            f" backn {self.back_name} oppon {self.oppo_name}"
            f" book_start {self.book_start_chance} back_win_m {self.back_win_match}"
            f" pred_P {self.predict_proba} res {self.predict_result} {self.comments}"
        )


mapper(
    PredictRec,
    predicts_table,
    properties={
        "sex": predicts_table.c.sex,
        "date": predicts_table.c.date,
        "case_name": predicts_table.c.case_name,
        "tour_name": predicts_table.c.tour_name,
        "level": predicts_table.c.level,
        "surface": predicts_table.c.surface,
        "rnd": predicts_table.c.rnd,
        "back_id": predicts_table.c.back_id,
        "oppo_id": predicts_table.c.oppo_id,
        "predict_proba": predicts_table.c.predict_proba,
        "predict_result": predicts_table.c.predict_result,
        "comments": predicts_table.c.comments,
        "back_name": predicts_table.c.back_name,
        "oppo_name": predicts_table.c.oppo_name,
        "book_start_chance": predicts_table.c.book_start_chance,
        "rejected": predicts_table.c.rejected,
        "back_win_match": predicts_table.c.back_win_match,
        "bf_live_coef": predicts_table.c.bf_live_coef,
        "bf_live_coef_matched": predicts_table.c.bf_live_coef_matched,
        "clf_hash": predicts_table.c.clf_hash,
    },
)


class Handle:
    def __init__(self, engine, session):
        self.engine = engine
        self.session = session
        self.records: List[PredictRec] = []

    def commit(self):
        self.session.commit()

    def insert_obj(self, record):
        self.session.add(record)

    def delete_obj(self, record):
        self.session.delete(record)

    def query_predicts(self, min_date=None, max_date=None):
        if min_date is None and max_date is None:
            self.records = self.session.query(PredictRec).order_by(PredictRec.date).all()
        elif min_date is not None and max_date is None:
            self.records = (
                self.session.query(PredictRec)
                .filter(min_date <= PredictRec.date)
                .order_by(PredictRec.date)
                .all()
            )
        elif min_date is None and max_date is not None:
            self.records = (
                self.session.query(PredictRec)
                .filter(PredictRec.date < max_date)
                .order_by(PredictRec.date)
                .all()
            )
        elif min_date is not None and max_date is not None:
            self.records = (
                self.session.query(PredictRec)
                .filter((min_date <= PredictRec.date) & (PredictRec.date < max_date))
                .order_by(PredictRec.date)
                .all()
            )

    def query_predict(self, sex: str, date: datetime.date, case_name: str,
                      back_id: int, oppo_id: int):
        rec = (
            self.session.query(PredictRec)
            .filter(
                (PredictRec.sex == sex)
                & (PredictRec.date == date)
                & (PredictRec.case_name == case_name)
                & (PredictRec.back_id == back_id)
                & (PredictRec.oppo_id == oppo_id)
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

    def get_predict_result_proba(self, sex: str, date: datetime.date, case_name: str,
                                 back_id: int, oppo_id: int):
        if self.records:
            rec = co.find_first(
                self.records,
                (
                    lambda r: r.sex == sex
                    and r.date == date
                    and r.case_name == case_name
                    and r.back_id == back_id
                    and r.oppo_id == oppo_id
                ),
            )
            if rec is not None:
                return rec.predict_result, rec.predict_proba

    def records_to_csv_file(self, filename,
                            columns_to_exclude=("_sa_instance_state",)):
        csv_str = self.records_to_csv(columns_to_exclude=columns_to_exclude)
        with open(filename, mode="w", encoding='utf-8') as fh:
            fh.write(csv_str)

    def records_to_csv(self, columns_to_exclude=("_sa_instance_state",)) -> str:
        """ Converts output from a SQLAlchemy query to a .csv string.

        Parameters:
        columns_to_exclude (list of str): names of columns to exclude from .csv output.

        Returns:
        csv (str): query_output represented in .csv format.

        Example usage:
        csv = handle.records_to_csv(("id", "age", "address"))
        """
        rows = self.records
        columns_to_exclude = set(columns_to_exclude)

        # create list of column names
        column_names = [i for i in rows[0].__dict__]
        for column_name in columns_to_exclude:
            column_names.pop(column_names.index(column_name))

        # add column titles to csv
        # column_names.sort()
        csv = ", ".join(column_names) + "\n"

        # add rows of data to csv
        for row in rows:
            for column_name in column_names:
                if column_name not in columns_to_exclude:
                    data = str(row.__dict__[column_name])
                    # Escape (") symbol by preceeding with another (")
                    data.replace('"', '""')
                    # Enclose each datum in double quotes so commas
                    # within are not treated as separators
                    csv += '"' + data + '"' + ","
            csv += "\n"
        return csv


max_date_stmt = text("""SELECT MAX(date) FROM predicts""")
min_date_stmt = text("""SELECT MIN(date) FROM predicts""")


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


def find_predict_rec_by(session, sex: str, date: datetime.date, case_name: str,
                        back_id: int, oppo_id: int):
    return (
        session.query(PredictRec)
        .filter_by(sex=sex, date=date, case_name=case_name,
                   back_id=back_id, oppo_id=oppo_id)
        .first()
    )


SQLITE_TYPENAME = "sqlite"


def open_db(dbtypename=SQLITE_TYPENAME):
    e = make_engine(dbtypename)
    s = make_session(e)
    return Handle(engine=e, session=s)


def make_engine(dbtypename=SQLITE_TYPENAME):
    return create_engine(
        "{}:///{}".format(
            dbtypename,
            os.path.abspath(dbfilename(dbtypename=dbtypename)),
        )
    )


def make_session(engine):
    return sessionmaker(bind=engine)()


def create_empty_db(dbtypename=SQLITE_TYPENAME):
    """GIVEN: no db file
    THEN create empty db file"""
    e = make_engine(dbtypename=dbtypename)
    metadata.create_all(e)


def dbfilename(dbtypename=SQLITE_TYPENAME):
    return os.path.join(cfg_dir.predicts_db_dir(), dbtypename)
