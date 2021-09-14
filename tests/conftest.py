"""Define some fixtures to use in the project."""

import pytest

import dba
from common import logname
import log
import common_wdriver


@pytest.fixture(scope="session")
def log_init():
    log.initialize(
        logname(__file__, test=True), file_level="info", console_level="info"
    )
    yield  # this is where the testing happens


@pytest.fixture(scope="session")
def log_init_debug():
    log.initialize(
        logname(__file__, test=True), file_level="debug", console_level="info"
    )
    yield  # this is where the testing happens


@pytest.fixture(scope="session")
def get_dba():
    """Connect to db before tests, disconnect after."""
    # Setup : start db
    dba.open_connect()

    yield dba.get_connect()  # this is where the testing happens

    # Teardown : stop db
    dba.close_connect()


@pytest.fixture(scope="session")
def fscore_driver():
    fsdrv = common_wdriver.wdriver(company_name="FS", headless=True)
    fsdrv.start()

    yield fsdrv

    fsdrv.stop()
