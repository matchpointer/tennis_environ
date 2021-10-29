import os
import re
from collections import Counter, defaultdict, OrderedDict, namedtuple
from operator import itemgetter
import functools
import unittest

import log
import common as co
import report_line as rl
import stat_cont as st


class OrderedDefaultDictTest(unittest.TestCase):
    def test_simple(self):
        dct = defaultdict(list)
        dct["x"].append(1)
        dct["b"].append(2)
        dct["x"].append(11)
        dct["c"].append(3)
        val = dct["zzz"]
        self.assertEqual([], val)
        self.assertEqual(["x", "b", "c", "zzz"], list(dct.keys()))
        self.assertEqual([[1, 11], [2], [3], []], list(dct.values()))
        del dct["x"]
        self.assertEqual(["b", "c", "zzz"], list(dct.keys()))


default_sep = "_"
load_line_re = re.compile(
    "(?P<key>[^" + default_sep + "]+)" + default_sep + "+(?P<value>.+)"
)


def dump(
    dictionary,
    filename,
    filterfun=None,
    keyfun=None,
    valuefun=None,
    sep=default_sep,
    alignvalues=False,
    sortfun=None,
    sortreverse=False,
):
    """
    save to file.
    filterfun применяется до keyfun, valuefun к key, value and skip item if False.
    sortfun applyable to one item-arg: (key, value).
    If sortfun given then keyfun, valuefun will apply after sorting.
    каждая пара ключ-значение должна быть сохранена в одной отдельной строке
    (чтобы восстановиться)
    """
    keydumpfun = keyfun if keyfun is not None else repr
    valuedumpfun = valuefun if valuefun is not None else repr
    if alignvalues and dictionary:
        key_len_max = max((len(keydumpfun(k)) for k in dictionary.keys()))
        fmt = "{0:" + sep + "<" + str(key_len_max) + "}{1}{2}\n"
    else:
        fmt = "{}{}{}\n"
    with open(filename, "w") as fh:
        if sortfun is not None:
            items = sorted(iter(dictionary.items()), key=sortfun, reverse=sortreverse)
        else:
            items = iter(dictionary.items())
        for key, value in items:
            if filterfun is not None and not filterfun(key, value):
                continue
            fh.write(fmt.format(keydumpfun(key), sep, valuedumpfun(value)))


def load(filename, createfun=None, keyfun=None, valuefun=None, filterfun=None):
    """
    restore from file
    каждая пара ключ-значение восстанавливается из одной отдельной строки
    filterfun применяется после keyfun, valuefun к key, value and skip item if False.
    """

    keyloadfun = keyfun if keyfun is not None else eval
    valueloadfun = valuefun if valuefun is not None else eval
    result = createfun() if createfun is not None else {}
    if os.path.isfile(filename):
        with open(filename, "r") as fh:
            for line in fh.readlines():
                if line.startswith("#"):
                    continue
                match = load_line_re.match(line)
                if match:
                    key = keyloadfun(match.group("key").strip())
                    value = valueloadfun(match.group("value").strip())
                    if filterfun is not None and not filterfun(key, value):
                        continue
                    result[key] = value
                else:
                    raise co.TennisError("unparsed dict line: '{}'".format(line))
    return result


def transed(
    dictionary,
    createfun=None,
    filterfun=None,
    keyfun=None,
    valuefun=None,
    sortfun=None,
    sortreverse=False,
):
    """
    return new transformed dict.
    filterfun применяется до keyfun, valuefun к key, value and skip item if False.
    sortfun applyable to one item-arg: (key, value)
    If sortfun given then keyfun, valuefun will apply after sorting.
    """
    keyfun = keyfun if keyfun is not None else co.identity
    valuefun = valuefun if valuefun is not None else co.identity
    result = createfun() if createfun is not None else {}
    if sortfun is not None:
        items = sorted(iter(dictionary.items()), key=sortfun, reverse=sortreverse)
    else:
        items = iter(dictionary.items())
    for key, value in items:
        if filterfun is not None and not filterfun(key, value):
            continue
        result[keyfun(key)] = valuefun(value)
    return result


def transed_value_percent_text(
    dictionary, createfun=None, keyfun=None, full_format=False
):
    """return new transformed dict.
    вместо value имеем (как текст) % текущего value от суммы value по всем key
    """
    keyfun = keyfun if keyfun is not None else co.identity
    value_sum = sum((v for v in dictionary.values()))
    result = createfun() if createfun is not None else {}
    for key, value in dictionary.items():
        result[keyfun(key)] = co.percented_text(
            value, value_sum, full_format=full_format
        )
    return result


def filter_value_size_fun(min_size):
    def check_min_size(key, value, min_size):
        if min_size == 0:
            return True
        elif hasattr(value, "size"):
            return value.size >= min_size
        else:
            return value >= min_size  # treat value as size

    return functools.partial(check_min_size, min_size=min_size)


LevSurf = namedtuple("LevSurf", "level surface")


class DictPersistTest(unittest.TestCase):
    def test_namedtuple_counter_dict(self):
        def make_tuple_counter_dict():
            result = dict()
            result[()] = Counter(dict([(-444, 222), (-222, 111)]))
            result[("main",)] = Counter(dict([(-44, 22), (-22, 11)]))
            result[("main", "Clay")] = Counter(dict([(-4, 2), (-2, 1)]))
            result[("masters", "Clay")] = Counter(dict([(-3, 4), (-2, 2)]))
            return result

        self.dump_load_test_impl(
            dictionary=make_tuple_counter_dict(),
            filename="./utest_tuple_counter_dict.txt",
        )

        def make_namedtuple_counter_dict():
            result = dict()
            result[LevSurf(level="main", surface="Hard")] = "This is text"
            result[LevSurf(level="main", surface="Clay")] = Counter(
                dict([(-4, 2), (-2, 1)])
            )
            result[LevSurf(level="masters", surface="Clay")] = Counter(
                dict([(-3, 4), (-2, 2)])
            )
            return result

        self.dump_load_test_impl(
            dictionary=make_namedtuple_counter_dict(),
            filename="./utest_namedtuple_counter_dict.txt",
        )

    def test_structkey_counter_dict(self):
        def make_structkey_counter_dict():
            result = dict()
            result[co.StructKey(level="main", surface="Clay")] = Counter(
                dict([(-4, 2), (-2, 1)])
            )
            result[co.StructKey(level="masters", surface="Clay")] = Counter(
                dict([(-3, 4), (-2, 2)])
            )
            result[co.StructKey(level="main", rnd="First")] = Counter(
                dict([(-1, 4), (0, 5)])
            )
            result[co.StructKey(level="chal", surface="Grass", rnd="1/4")] = Counter(
                dict([(2, 12), (-1, 48)])
            )
            return result

        self.dump_load_test_impl(
            dictionary=make_structkey_counter_dict(),
            filename="./utest_structkey_counter_dict.txt",
        )

    def test_structkey_winloss_dict(self):
        def make_structkey_winloss_dict():
            result = dict()
            result[co.StructKey(surface="Hard", level="main")] = st.WinLoss(999, 1)
            result[co.StructKey(level="masters", surface="Clay")] = st.WinLoss(70, 30)
            result[co.StructKey(level="main", rnd="First")] = st.WinLoss(66, 34)
            result[co.StructKey(level="chal", surface="Grass", rnd="1/4")] = st.WinLoss(
                0, 100
            )
            return result

        self.dump_load_test_impl(
            dictionary=make_structkey_winloss_dict(),
            filename="./utest_structkey_winloss_dict.txt",
        )

    def test_struct_winloss_dict(self):
        def make_struct_winloss_dict():
            result = dict()
            result[co.Struct()] = st.WinLoss(9999, 1)
            result[co.Struct(surface="Hard", level="main")] = st.WinLoss(999, 1)
            result[co.Struct(level="masters", surface="Clay")] = st.WinLoss(70, 30)
            result[co.Struct(level="main", rnd="First")] = st.WinLoss(66, 34)
            result[co.Struct(level="chal", surface="Grass", rnd="1/4")] = st.WinLoss(
                0, 100
            )
            return result

        self.dump_load_test_impl(
            dictionary=make_struct_winloss_dict(),
            filename="./utest_struct_winloss_dict.txt",
        )

    def test_empty_dict(self):
        self.dump_load_test_impl(dictionary={}, filename="./utest_empty_dict.txt")

    def dump_load_test_impl(self, dictionary, filename):
        if os.path.isfile(filename):
            os.remove(filename)
        dump(dictionary, filename)
        dictionary_loaded = load(filename)
        self.assertEqual(dictionary, dictionary_loaded)
        for val in dictionary_loaded.values():
            if isinstance(val, (bytes, str)):
                self.assertTrue(isinstance(val, str))
        os.remove(filename)


def value_pos(filename, key, keyfun=None, valuefun=None):
    """найти в файле указанный ключ и вернуть пару (value, Position in file)"""
    dct = load(filename, createfun=OrderedDict, keyfun=keyfun, valuefun=valuefun)
    line_num = 0
    for the_key, the_value in dct.items():
        line_num += 1
        if the_key == key:
            return the_value, co.Position(line_num, len(dct))
    return None, None


# ------------------------ misc ------------------------------


def values_to_str(dictionary, value_fmt="{}", sep="-"):
    result = ""
    for key_val in sorted(iter(dictionary.items()), key=itemgetter(0)):
        if result:
            result += sep
        result += value_fmt.format(key_val[1])
    return result


def binary_operation(oper_fun, left_dict, right_dict):
    """только для словарей где значения типа SizedValue"""
    result = {}
    for key, left_sval in left_dict.items():
        if key not in right_dict:
            continue
        right_sval = right_dict[key]
        if (
            left_sval is None
            or left_sval.value is None
            or right_sval is None
            or right_sval.value is None
        ):
            continue
        result[key] = rl.SizedValue(
            oper_fun(left_sval.value, right_sval.value),
            max(left_sval.size, right_sval.size),
        )
    return result


def make_counter(iterable, key_extractor):
    """вернем словарь-счетчик (Counter)"""
    cntr = Counter()
    for elem in iterable:
        key = key_extractor(elem)
        if key is not None:
            cntr[key] += 1
    return cntr


def get_items(dictionary, key_predicate=lambda k: True, value_predicate=lambda v: True):
    return [
        (key, value)
        for (key, value) in list(dictionary.items())
        if key_predicate(key) and value_predicate(value)
    ]


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", None)
    unittest.main()
