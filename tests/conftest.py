# -*- coding=utf-8 -*-
"""Define some fixtures to use in the project."""
import pytest

from oncourt import dbcon
from loguru import logger as log
from score_company import get_company
from pages.create_page import create_page
from pages.tennis24page import MainPageTennis24
from wdriver import stop_web_driver
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
    dbcon.open_connect()

    yield dbcon.get_connect()  # this is where the testing happens

    # Teardown : stop db
    dbcon.close_connect()


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
    raise NotImplementedError()


@pytest.fixture(scope="session")
def t24_driver(get_t24_comp) -> MainPageTennis24:
    wpage = create_page(score_company=get_t24_comp, is_main=True, headless=False)

    yield wpage

    stop_web_driver(wpage.get_web_driver())
