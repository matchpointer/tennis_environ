# -*- coding: utf-8 -*-
import os
import pyodbc
from contextlib import contextmanager

""" at windows 10 you must install Access Database Engine redistributable  (2010)
"""

__conn = None


def open_connect():
    def odbc_msaccess_drv_name():
        names = [
            x
            for x in pyodbc.drivers()
            if x.startswith("Microsoft Access Driver") and r"*.mdb" in x
        ]
        if len(names) >= 1:
            return names[0]

    global __conn
    if __conn is None:
        db_filename = os.environ.get("ODBNAME")
        assert bool(db_filename), "ODBNAME must be as environment variable"
        assert os.path.isfile(db_filename), "file {} not found".format(db_filename)

        db_pwd = os.environ.get("ODBPASSWORD")
        assert bool(db_pwd), "ODBPASSWORD must be as environment variable"

        odbc_drv_name = odbc_msaccess_drv_name()
        assert odbc_drv_name is not None, "can not found Microsoft Access Driver"

        conn_str = f"DRIVER={odbc_drv_name};PWD={db_pwd};DBQ={db_filename}"
        __conn = pyodbc.connect(conn_str)


def get_connect():
    assert __conn is not None, "get_connect to none (not opened) object"
    return __conn


def close_connect():
    global __conn
    if __conn is not None:
        __conn.close()
        __conn = None


@contextmanager
def connect_cntx():
    open_connect()
    try:
        yield get_connect()
    finally:
        close_connect()


def initialized():
    return __conn is not None


def result_iter(cursor, arraysize=1000):
    """
    An iterator that uses fetchmany to keep memory usage down
    """
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        for result in results:
            yield result


def msaccess_date(date):
    return "#{}/{}/{}#".format(date.month, date.day, date.year)


def sql_dates_condition(min_date, max_date, dator="tours.DATE_T"):
    result = ""
    if min_date is not None:
        result += " and {} >= {}\n".format(dator, msaccess_date(min_date))
    if max_date is not None:
        result += " and {} < {}\n".format(dator, msaccess_date(max_date))
    return result
