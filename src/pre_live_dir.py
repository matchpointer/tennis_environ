# -*- coding: utf-8 -*-
import os
import unittest
import json

import log
import cfg_dir
import file_utils as fu
import common as co


dirname = cfg_dir.pre_live_dir("matches")


def prepare_dir(name):
    assert name in ("matches", "tours"), "bad pre_live subrir {}".format(name)
    global dirname
    dirname = cfg_dir.pre_live_dir(name)


def is_data(pre_live_name):
    return os.path.isfile(_filename(pre_live_name))


def get_data(pre_live_name):
    filename = _filename(pre_live_name)
    return fu.json_load(filename)


def save_data(pre_live_name, dictionary):
    filename = _filename(pre_live_name)
    fu.json_dump(dictionary, filename)


def remove_data(pre_live_name):
    filename = _filename(pre_live_name)
    if os.path.isfile(filename):
        os.remove(filename)


def remove_all():
    global dirname
    dirname_mem = dirname
    for dname in ("matches", "tours"):
        prepare_dir(dname)
        fu.remove_files_in_folder(dirname, recursive=False)
    dirname = dirname_mem  # restore current state


def _filename(pre_live_name):
    return os.path.join(dirname, pre_live_name)


class JsonfileTest(unittest.TestCase):
    dirname = cfg_dir.unit_test_data_dir()

    def test_json_read(self):
        filename = os.path.join(self.dirname, "json_dict.txt")
        with open(filename, "r") as fh:
            dct = json.load(fh)
        self.assertTrue(8.5399 <= dct["fst_win_coef"] <= 8.54)
        self.assertTrue(1.085 <= dct["snd_win_coef"] <= 1.086)
        self.assertTrue("snd_player_name" in dct)
        self.assertTrue(isinstance(dct["snd_player_name"], str))


class FilesTest(unittest.TestCase):
    def test_remove_all(self):
        print("dirname: ", dirname)
        self.assertTrue(os.path.isdir(dirname))
        remove_all()
        self.assertTrue(len("OK") > 0)


if __name__ == "__main__":
    log.initialize(
        co.logname(__file__, test=True), file_level="info", console_level="info"
    )
    unittest.main()
