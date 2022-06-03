import os

import cfg_dir
import file_utils as fu


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
