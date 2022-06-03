"""Define some fixtures to use in the project."""
import os
import pytest

import dba
from loguru import logger as log
from score_company import get_company
import common_wdriver
from detailed_score_dbsa import open_db


@pytest.fixture(scope="session")
def log_init():
    log.add("../log/pytest.log", level='INFO')
    yield  # this is where the testing happens


@pytest.fixture(scope="session")
def log_init_debug():
    log.add("../log/pytest.log", level='DEBUG')
    yield  # this is where the testing happens


@pytest.fixture(scope="session")
def get_dba():
    """Connect to oncourt db before tests, disconnect after."""
    # Setup : start db
    dba.open_connect()

    yield dba.get_connect()  # this is where the testing happens

    # Teardown : stop db
    dba.close_connect()


@pytest.fixture(scope="session")
def dbdet_wta():
    """ get ORM access to detailed scores db wta before tests """
    # Setup :
    dbdet = open_db(sex='wta')

    yield dbdet  # this is where the testing happens


@pytest.fixture(scope="session")
def all_dbdet_wta():
    """ get ORM access to all-inited detailed scores db wta before tests """
    # Setup :
    dbdet = open_db(sex='wta')
    dbdet.query_matches()

    yield dbdet  # this is where the testing happens


@pytest.fixture(scope="session")
def dbdet_atp():
    """ get ORM access to  detailed scores db atp before tests """
    # Setup :
    dbdet = open_db(sex='atp')

    yield dbdet  # this is where the testing happens


@pytest.fixture(scope="session")
def get_fs_comp():
    company = get_company("FS")

    yield company


@pytest.fixture(scope="session")
def get_t24_comp():
    company = get_company("T24")

    yield company


@pytest.fixture(scope="session")
def fscore_driver(get_fs_comp):
    drv = common_wdriver.wdriver(company=get_fs_comp, headless=True)
    drv.start()

    yield drv

    drv.stop()


@pytest.fixture(scope="session")
def t24_driver(get_t24_comp):
    drv = common_wdriver.wdriver(company=get_t24_comp, headless=True)
    drv.start()

    yield drv

    drv.stop()
