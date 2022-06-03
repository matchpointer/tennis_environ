import os
import re
import fnmatch
from collections import defaultdict
import json
import pickle
import hashlib
from typing import DefaultDict
try:
    import win32api
except ImportError:
    win32api = None

import common as co


def json_load(filename, encoding="utf8"):
    with open(filename, mode="r", encoding=encoding) as fh:
        return json.load(fh)


def json_dump(dictionary, filename, indent=4):
    with open(filename, mode="w") as fh:
        json.dump(dictionary, fh, indent=indent)


def json_load_intkey_dict(filename):
    with open(filename, mode="r") as fh:
        dct = json.load(fh)
    return {int(k): v for k, v in dct.items()}


def json_edit_item(filename: str, edit_fun):
    with open(filename) as fh:
        data = json.load(fh)

    for item in data.items():
        edit_fun(item)

    with open(filename, "w") as fh:
        json.dump(data, fh)


def read(filename, encoding="utf8"):
    with open(filename, mode="r", encoding=encoding) as fh:
        return fh.read()


def write(filename, data, encoding="utf8"):
    with open(filename, mode="w", encoding=encoding) as fh:
        fh.write(data)


def single_write(filename, data, encoding="utf8"):
    """для кокретного filename пишем только первый раз,
    остальные попытки игнорируются в оставшееся время исполнения"""
    if not _writed_from_filename[filename]:
        write(filename, data, encoding=encoding)
        _writed_from_filename[filename] = True


_writed_from_filename: DefaultDict[str, bool] = defaultdict(lambda: False)


def pickle_object(obj, filename):
    with open(filename, "wb") as out:  # Overwrites any existing file.
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)


def unpickle_object(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_file_version(filename):
    if os.path.isfile(filename) and win32api:
        info = win32api.GetFileVersionInfo(filename, "\\")
        ms = info["ProductVersionMS"]
        ls = info["ProductVersionLS"]
        return (
            win32api.HIWORD(ms),
            win32api.LOWORD(ms),
            win32api.HIWORD(ls),
            win32api.LOWORD(ls),
        )
    return None


def md5sum(filename, block_size=4096):
    """for large file you may use block_size=65536"""
    with open(filename, mode="rb") as f:
        d = hashlib.md5()
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            d.update(buf)
        return d.hexdigest()


def get_file_hash(filename):
    with open(filename, "r") as fhandle:
        return hash(fhandle.read())


def get_most_fresh_file(files, file_filter=None):
    def admit_file(filename):
        if file_filter is None:
            return True
        return file_filter(filename)

    fresh_file, fresh_time = None, 0.0
    for fname in files:
        last_modif_time = os.path.getmtime(fname)
        if last_modif_time >= fresh_time and admit_file(fname):
            fresh_time = last_modif_time
            fresh_file = fname
    return fresh_file


def file_lines_count(filename):
    count = 0
    with open(filename, "r") as fhandle:
        while fhandle.readline():
            count += 1
        return count


def count_in_file(filename, pattern, re_flags=0):
    pattern_re = re.compile(pattern, re_flags)
    result_count = 0
    with open(filename, "r") as fhandle:
        if re_flags == 0:
            for line in fhandle.readlines():
                result_count += len(re.findall(pattern_re, line))
        else:
            data = ""
            for line in fhandle.readlines():
                data += line
            result_count = len(re.findall(pattern_re, data))
        return result_count


def folder_lines_count(folder, filemask="*", recursive=True):
    return apply_to_files(
        co.Functor(file_lines_count, ret_init=0, ret_set=lambda r, c: r + c),
        folder,
        filemask=filemask,
        recursive=recursive,
    ).returned_value


def folder_files_count(folder, filemask="*", recursive=True):
    return apply_to_files(
        co.Functor(None, ret_init=0, ret_set=lambda r, c: r + 1),
        folder,
        filemask=filemask,
        recursive=recursive,
    ).returned_value


def line_pos_in_file(filename, pattern):
    """Get Position  for first presence pattern in file."""
    found_line_pos = None
    line_pos = 0
    with open(filename, "r") as fhandle:
        for line in fhandle.readlines():
            line_pos += 1
            if pattern and found_line_pos is None and re.search(pattern, line):
                found_line_pos = line_pos
        return co.Position(found_line_pos, line_pos)


def ensure_folder_for_filename(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def is_empty_folder(folder):
    assert os.path.isdir(folder), "try check non-existed folder %s for empty" % folder
    for root, dirs, files in os.walk(folder):
        return len(dirs) == 0 and len(files) == 0
    return True


def is_non_zero_file(filename: str):
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        return False
    # if file is damaged then very likely it consist of zero bytes only
    with open(filename, mode='rb') as fd:
        return any((char != 0 for char in fd.read()))


def find_files_in_folder(folder, filemask="*", recursive=True):
    """Retrive list of full-filenames which are match with given mask."""
    functor = co.Functor(lambda fn: fn, ret_init=[], ret_set=lambda r, c: r + [c])
    return apply_to_files(
        functor, folder, filemask=filemask, recursive=recursive
    ).returned_value


def find_single_file_in_folder(folder, filemask="*", recursive=True):
    """Retrive full-filename which matches with given mask.
    If such files is more than one then return None.
    """
    fnames = find_files_in_folder(folder, filemask=filemask, recursive=recursive)
    if len(fnames) == 1:
        return fnames[0]
    else:
        return None


def find_folders_in_folder(folder, dirmask="*", recursive=True):
    functor = co.Functor(lambda fn: fn, ret_init=[], ret_set=lambda r, c: r + [c])
    return apply_to_folders(
        functor, folder, dirmask=dirmask, recursive=recursive
    ).returned_value


def replace_in_file(filename, source, target, re_flags=0):
    repl_re = re.compile(source, re_flags)
    with open(filename, "r") as fhandle:
        data = ""
        if re_flags == 0:
            for line in fhandle.readlines():
                data += re.sub(repl_re, target, line)
        else:
            for line in fhandle.readlines():
                data += line
            data = re.sub(repl_re, target, data)

    # rewrite modified data into our file
    with open(filename, "w") as fhandle_out:
        fhandle_out.write(data)


def replace_in_folder(folder, source, target, filemask="*", recursive=True, re_flags=0):
    functor = co.Functor(
        replace_in_file, source=source, target=target, re_flags=re_flags
    )
    apply_to_files(functor, folder, filemask=filemask, recursive=recursive)


def replace_defined_right_value(filename, left_value, new_right_value):
    """For each matched line (in file): define left_value = \"oldRightValue\" ==>
    define left_value = \"rightValue\" """
    replace_in_file(
        filename,
        r"(\s*define\s+%s\s*=\s*)\"[-a-zA-Z0-9_]*\"(.*$)" % left_value,
        '\\1"%s"\\2' % new_right_value,
    )


def remove_files_in_folder(folder, filemask="*", recursive=True):
    """Removes files which are match with given mask."""
    apply_to_files(
        co.Functor(os.remove), folder, filemask=filemask, recursive=recursive
    )


def remove_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def apply_to_files(functor, folder, filemask="*", recursive=True):
    if recursive:
        for root, dirs, files in os.walk(folder):
            for fname in files:
                filename = os.path.join(root, fname)
                if fnmatch.fnmatch(fname, filemask):
                    functor(filename)
    else:
        for fname in os.listdir(folder):
            filename = os.path.join(folder, fname)
            if os.path.isfile(filename) and fnmatch.fnmatch(fname, filemask):
                functor(filename)
    return functor


def apply_to_folders(functor, folder, dirmask="*", recursive=True):
    """Call functor(dirname) for each dir in folder. Not apply call with start folder."""
    if recursive:
        for root, dirs, files in os.walk(folder):
            for dname in dirs:
                dirname = os.path.join(root, dname)
                if fnmatch.fnmatch(dname, dirmask):
                    functor(dirname)
    else:
        for dname in os.listdir(folder):
            dirname = os.path.join(folder, dname)
            if os.path.isdir(dirname) and fnmatch.fnmatch(dname, dirmask):
                functor(dirname)
    return functor


ORA_ERROR_RE = re.compile(".*ORA-(?P<code>[0-9][0-9][0-9][0-9][0-9]).*")


def ora_error_in_file(filename, always_check_last_line=False, ignore_codes=None):
    """Returns False if file is ok. Returns True if file with errors."""
    ignore_codes = [] if ignore_codes is None else ignore_codes
    is_error = False
    ora_errors = []
    with open(filename, "r") as fhandle:
        curline = ""
        for line in fhandle.readlines():
            curline = line
            match_ora_err = ORA_ERROR_RE.search(line)
            if match_ora_err:
                ora_err = match_ora_err.group("code")
                ora_errors.append(ora_err)
                if ora_err not in ignore_codes:
                    is_error = True

    lastline = curline.strip()
    table_or_view_absence_code = "00942"
    is_last_line_ok = table_or_view_absence_code not in ora_errors and (
        "No errors found." in lastline
        or (
            lastline.startswith("The application")
            and lastline.endswith("successfully installed.")
        )
    )
    if is_error and is_last_line_ok:
        is_error = False  # existed ORA errors finally recompiled successfully
    elif not is_error and not is_last_line_ok and always_check_last_line:
        is_error = True  # implicit errors without displayed ORA error code
    return is_error

