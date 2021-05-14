# -*- coding: utf-8 -*-
import os
import time
import logging
import logging.handlers
import unittest

import file_utils as fu


LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

logger = None


def initialized():
    return logger is not None


def initialize(filename, file_level="info", console_level=None):
    if initialized():
        raise Exception("log is already initialized")
    __check_levels([file_level, console_level])

    global logger
    logger = logging.getLogger()
    logger.setLevel(__min_level_code([file_level, console_level]))

    if file_level:
        need_roll = os.path.isfile(filename)
        file_handler = logging.handlers.RotatingFileHandler(
            filename, backupCount=50, encoding="utf8"
        )
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%d.%m.%Y %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(LEVELS[file_level])
        logger.addHandler(file_handler)

        if need_roll:
            # Roll over on application start
            logger.handlers[0].doRollover()

    if console_level:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(LEVELS[console_level])
        logger.addHandler(console_handler)


def info(text):
    __check_logger()
    logger.info(text)


def debug(text):
    __check_logger()
    logger.debug(text)


def warn(text):
    __check_logger()
    logger.warn(text)


def error(text, exception=False):
    __check_logger()
    logger.error(text, exc_info=exception)


def flush():
    __check_logger()
    if logger.handlers:
        logger.handlers[0].flush()


def __check_logger():
    if logger is None:
        raise Exception(__name__ + ".initialize() must be call before using")


def __check_level(level):
    if level is not None and level not in iter(LEVELS.keys()):
        raise Exception(
            "invalid parameter level ({0}) in {1}.initializion".format(
                str(level), __name__
            )
        )


def __check_levels(levels):
    for level in levels:
        __check_level(level)
    if not any(levels):
        raise Exception(
            "used all None level-parameters in {}.initializion".format(__name__)
        )


def __min_level_code(levels):
    level_codes = [
        ((logging.CRITICAL + 1) if level is None else LEVELS[level]) for level in levels
    ]
    return min(level_codes)


class LogRotatingTest(unittest.TestCase):
    DIRNAME = "./logingtest"
    SHORTNAME = "test.log"
    prev_session_files_count = 0

    @classmethod
    def setUpClass(cls):
        fu.ensure_folder(cls.DIRNAME)
        files = fu.find_files_in_folder(cls.DIRNAME)
        cls.prev_session_files_count = len(files)
        initialize(cls.DIRNAME + "/" + cls.SHORTNAME)

    @classmethod
    def tearDownClass(cls):
        global logger
        del logger
        logger = None

    def test_rotating(self):
        """должно быть на 1 больше файлов чем после предыдущих запусков"""
        info(str(time.asctime()))
        info(str("\xc3\xb4", "utf-8"))
        flush()
        files = fu.find_files_in_folder(LogRotatingTest.DIRNAME)
        self.assertEqual(len(files), LogRotatingTest.prev_session_files_count + 1)


if __name__ == "__main__":
    unittest.main()
