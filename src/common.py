# -*- coding=utf-8 -*-
import os
from sys import argv
import copy
import functools
import time
import datetime
import random
import unicodedata
from itertools import tee
from platform import node
from typing import List

from enum import IntEnum

from loguru import logger as log

import lev
from side import Side


class PlatformNodes:
    FIRST_NODE = "DESKTOP-first"
    SECOND_NODE = "DESKTOP-second"

    @staticmethod
    def is_first_node():
        return node() == PlatformNodes.FIRST_NODE

    @staticmethod
    def is_second_node():
        return node() == PlatformNodes.SECOND_NODE


def abscurfilename(basefilename):
    abspath = os.path.abspath(os.path.curdir)
    return os.path.join(abspath, basefilename)


def random_sleep(min_sec, max_sec):
    min_dsec = round(min_sec * 10)
    max_dsec = round(max_sec * 10)
    sleep_sec = random.randint(min(min_dsec, max_dsec), max(min_dsec, max_dsec))
    time.sleep(float(sleep_sec) * 0.1)


def random_seeds(size):
    """return list of uniq positive integers. len(list) = size"""
    return random.sample(range(10000000), size)


def log_command_line_args(head="cmd-line-args"):
    if len(argv) > 1:
        log.info(f"{head} {argv[1:]}")


class RandomArguments:
    sep = ","
    generate_prefix = "generate"
    generate_max = 10000

    def __init__(self, arguments=None, default_seed=None):
        self.default_seed = default_seed
        if arguments is not None and arguments.random_seeds:
            self.seeds = self._randoms_from_str(arguments.random_seeds)
        else:
            self.seeds = []

        if arguments is not None and arguments.random_states:
            self.states = self._randoms_from_str(arguments.random_states)
        else:
            self.states = []

    @staticmethod
    def from_pair(random_seed, random_state):
        result = RandomArguments()
        result.seeds = [random_seed]
        result.states = [random_state]
        return result

    @staticmethod
    def _randoms_from_str(text):
        if text.startswith(RandomArguments.generate_prefix):
            size = int(text[len(RandomArguments.generate_prefix) :])
            return [
                random.randint(0, RandomArguments.generate_max) for _ in range(size)
            ]
        else:
            return [int(item) for item in text.split(RandomArguments.sep)]

    def iter_seeds(self):
        if len(self.seeds) > 0:
            return iter(self.seeds)
        if self.default_seed is not None:
            return iter([self.default_seed])

    def _seeds_len(self):
        if len(self.seeds) > 0:
            return len(self.seeds)
        if self.default_seed is not None:
            return 1
        return 0

    def iter_states(self):
        return iter(self.states)

    def is_states(self):
        return len(self.states) > 0

    def get_any_state(self):
        if len(self.states) > 0:
            return self.states[0]

    def get_any_seed(self):
        if len(self.seeds) > 0:
            return self.seeds[0]
        return self.default_seed

    def is_seeds(self):
        return self._seeds_len() > 0

    def space_size(self):
        return self._seeds_len() * len(self.states)

    @staticmethod
    def prepare_parser(parser):
        msg = (
            f"coma-separated ints "
            f'(or "{RandomArguments.generate_prefix}N" for generating N integers)'
        )
        parser.add_argument("--random_seeds", help=msg, type=str)
        parser.add_argument("--random_states", help=msg, type=str)


def make_type_instance(dictionary, type_name="_"):
    return type(type_name, (), dictionary)()


def enum_from_enum_or_str(enum_type, arg):
    if isinstance(arg, enum_type):
        return arg
    elif isinstance(arg, str):
        return enum_type[arg]
    else:
        raise Exception(
            "can't create enum " + str(enum_type) + " from type " + str(type(arg))
        )


class Struct(object):
    """
    Создается из словаря. Ключи становятся членами с соответствующими значениями.
    Объекты, у которых элементы отличаются лишь порядком считаюся равными.

    >>> Struct(surface='Clay', rnd='First', level='main') == Struct(level='main', rnd='First', surface='Clay')
    True
    >>> Struct(surface='Clay', rnd='First', level='main') != Struct(level='main', rnd='First')
    True
    >>> Struct() == Struct()
    True
    """

    text_sep = " "

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return self.text_sep.join(
            [
                "{}={}".format(name, value)
                for name, value in sorted(self.__dict__.items())
            ]
        )

    def __repr__(self):
        kvs = ",".join(
            [
                "{}={}".format(name, repr(value))
                for name, value in self.__dict__.items()
            ]
        )
        return f"{self.__class__.__name__}({kvs})"

    def __hash__(self):
        return hash(self.__str__())

    def __len__(self):
        return len(self.__dict__)

    def updated(self, other):
        kw = self.__dict__.copy()
        kw.update(other.__dict__)
        return Struct(**kw)

    def key_names(self):
        return self.__dict__.keys()

    def cardinality(self):
        return len(list(self.__dict__.keys()))


@functools.total_ordering
class StructKey(Struct):
    """
    Отличается от предка возможностью инициализации из строки,
    Отличие также возможно (было ранее) в разделителе для строкового представления.

    >>> k1 = StructKey(surface='Clay', rnd='First', level='main')
    >>> k2 = StructKey(rnd='First', level='main', surface='Clay')
    >>> k1 == k2
    True
    >>> k3 = StructKey.create_from_text('rnd=First,surface=Clay,level=main')
    >>> k1 == k3
    True
    """

    text_sep = ","
    text_if_empty = "all"

    def __str__(self):
        result = super(StructKey, self).__str__()
        return result if result else StructKey.text_if_empty

    def __lt__(self, other):
        return str(self) < str(other)

    @staticmethod
    def create_from_text(text):
        if text == StructKey.text_if_empty:
            return StructKey()
        if text:
            dct = {}
            for item in text.split(StructKey.text_sep):
                name, value = item.split("=")
                dct[name] = value
            return StructKey(**dct)
        raise TennisError("StructKey failed init-from_text: empty arg")

    def keys_string(self, sep=None):
        if sep is None:
            sep = StructKey.text_sep
        return sep.join(self.__dict__.keys())

    def values_string(self, sep=None):
        if sep is None:
            sep = StructKey.text_sep
        return sep.join(self.__dict__.values())


def maps_combinations(maps, initializer=None):
    """
    Выдает словари, являющиеся всевозм-ми сочетаниями из вх-х элементов-словарей 2x местных

    >>> maps_combinations([{'a': 1}, {'b': 2}, {'c': 3}])
    [{}, {'a': 1}, {'b': 2}, {'a': 1, 'b': 2}, {'c': 3}, {'a': 1, 'c': 3}, {'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}]
    """

    def expand(maps, x):
        expand_list = []
        for m in maps:
            mcopy = copy.copy(m)
            mcopy.update(x)
            expand_list.append(mcopy)
        return maps + expand_list

    if maps is None:
        return []
    if initializer is None:
        initializer = [{}]
    return functools.reduce(expand, maps, initializer)


def keys_combinations(maps, initializer=None):
    if initializer is None:
        initializer = [{}]
    return [StructKey(**m) for m in maps_combinations(maps, initializer)]


class Keys(dict):
    """
    кроме словаря содержит член __combinations
    определяемый при вызове combinations и сбрасываемый при добавлении
    новой пары ключ-значение
    """

    def __init__(self, dictionary=None, **kwargs):
        dictionary = dictionary or {}
        super(Keys, self).__init__(dictionary)
        if kwargs:
            super(Keys, self).update(kwargs)
        self.__combinations = None

    @staticmethod
    def soft_main_maker(tour, rnd, with_surface=True):
        return Keys.soft_main_from_raw(
            tour.level, rnd, surface=tour.surface if with_surface else None
        )

    @staticmethod
    def soft_main_from_raw(level, rnd, surface=None):
        import tennis

        soft_level = lev.soft_level(level, rnd)
        if surface is not None:
            return Keys(level=soft_level, surface=str(surface))
        else:
            return Keys(level=soft_level)

    def combinations(self):
        if self.__combinations is not None:
            return self.__combinations
        args = []
        for key, value in self.items():
            args.append({key: value})
        self.__combinations = keys_combinations(args)
        return self.__combinations

    def __setitem__(self, key, value):
        self.__combinations = None
        super(Keys, self).__setitem__(key, value)


class Attribute(object):
    def __init__(self, name, predicate=None):
        self.name = name
        self.predicate = predicate

    def value(self, obj):
        return getattr(obj, self.name)

    def apply_predicate(self, obj):
        if self.predicate is not None:
            attr = self.value(obj)
            return self.predicate(attr)


def identity(arg):
    return arg


def compose(*functions):
    """
    Возврат подготовленной функции-композита функций одного аргумента.
    Когда будет вызов, то в порядке справа налево.
    """

    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions)


def operation(oper, *functions):
    """
    Возврат oper (callable) над функциями одного аргумента.
    Сама по себе oper - callable for 2 args.
    """

    def oper2(f, g):
        return lambda x: oper(f(x), g(x))

    return functools.reduce(oper2, functions)


class Functor(object):
    """
    Resembles functools.partial, but some differings (listed below).
    Differings from functools.partial:
    a) After call we have returned value as usial return and also stored as member returned_value.
    b) constructor can take pair ret_init=val for returned_value member initializing, default is None;
    c) constructor can take pair ret_set=fun_set for returned_value member updating after call with
           fun_set(previuos returned_value, current returned_value),
           default fun_set behavior is: returned_value = current returned_value;
    d) constructor can take pair ret_check=fun_check for returned_value member checking after call with
           assert fun_set(current returned_value)
           skip by default.
    """

    def __init__(self, func, *args, **kwargs):
        """func - function, if given None then return value assumed as None without call."""
        self._func = func
        self._args = args

        kw = kwargs.copy()
        # extracting special key=val (not for main call,
        # but for initialize, update, check of result member)
        if "ret_init" in kw:
            self.returned_value = kw["ret_init"]
            del kw["ret_init"]
        else:
            self.returned_value = None

        if "ret_set" in kw:
            self.returned_value_setter = kw["ret_set"]
            del kw["ret_set"]
        else:
            self.returned_value_setter = lambda ret_val, cur_val: cur_val

        if "ret_check" in kw:
            self.returned_value_check = kw["ret_check"]
            del kw["ret_check"]
        else:
            self.returned_value_check = None

        self._keywords = kw

    def __call__(self, *args_rest, **kwargs_rest):
        if self._func is None:
            ret_value = None
        else:
            kw = self._keywords.copy()
            kw.update(kwargs_rest)
            ret_value = self._func(*(self._args + args_rest), **kw)

        if self.returned_value_check is not None:
            assert self.returned_value_check(
                ret_value
            ), "functor ret val check fail " + str(ret_value)

        self.returned_value = self.returned_value_setter(self.returned_value, ret_value)
        return self.returned_value


class TennisError(Exception):
    pass


class TennisScoreError(TennisError):
    pass


class TennisScoreSuperTieError(TennisScoreError):
    pass


class TennisUnknownScoreError(TennisScoreError):
    pass


class TennisScoreMaxError(TennisScoreError):
    pass


class TennisScoreOrderError(TennisScoreError):
    pass


class TennisParseError(TennisError):
    pass


class TennisNotFoundError(TennisError):
    pass


class TennisAbortError(TennisError):
    pass


class TennisDebugError(TennisError):
    pass


class TennisInternetError(TennisError):
    pass


def split_ontwo(text: str, delim: str, right_find=True):
    pos = text.rfind(delim) if right_find else text.find(delim)
    if pos >= 0:
        return text[:pos].strip(), text[pos + len(delim):].strip()
    return text.strip(), ""


def delete_first_find(text, sub, most_right=False, repetition=1):
    """
    удаляет первый найденый фрагмент текста. Возможность повтора repetition раз.

    >>> delete_first_find('abcdef', 'cd')
    'abef'
    >>> delete_first_find('abcdecf', 'c', True)
    'abcdef'
    >>> delete_first_find('abcdecf', 'c', False)
    'abdecf'
    >>> delete_first_find('abcfdecf', 'cf', True)
    'abcfde'
    >>> delete_first_find('abcfdecfxcf', 'cf', True, 2)
    'abcfdex'
    """

    def delete_first_find_impl(text, sub, most_right=False):
        idx = text.rfind(sub) if most_right else text.find(sub)
        if idx < 0:
            return text
        else:
            return text[0:idx] + text[idx + len(sub) :]

    for _ in range(repetition):
        text = delete_first_find_impl(text, sub, most_right)
    return text


def strip_after_find(text, sub):
    """
    удаляет начиная от найденного фрагмента и до правого конца (НЕ включая найденное).

    >>> strip_after_find('abcdef', 'cd')
    'abcd'
    >>> strip_after_find('abcdecf', 'c')
    'abc'
    """
    idx = text.find(sub)
    if idx < 0:
        return text
    else:
        return text[0 : idx + len(sub)]


def strip_from_find(text, sub, most_right=False):
    """
    удаляет начиная от найденного фрагмента и до правого конца (включая найденное).

    >>> strip_from_find('abcdef', 'cd')
    'ab'
    >>> strip_from_find('abcdecf', 'c', True)
    'abcde'
    """
    idx = text.rfind(sub) if most_right else text.find(sub)
    if idx < 0:
        return text
    else:
        return text[0:idx]


def strip_until_find(text, sub, most_right=False):
    """
    удаляет с начала и до найденного фрагмента (включая найденное).

    >>> strip_until_find('abcdef', 'cd')
    'ef'
    >>> strip_until_find('abcdecf', 'c', True)
    'f'
    """
    idx = text.rfind(sub) if most_right else text.find(sub)
    if idx < 0:
        return text
    else:
        return text[idx + len(sub) :]


def strip_fragment(text, begin, end):
    """remove begin,...,end if begin and end are found in text"""
    begin_idx = text.find(begin)
    if begin_idx < 0:
        return text

    end_idx = text.find(end)
    if end_idx < 0 or end_idx < begin_idx:
        return text
    return text[0:begin_idx] + text[end_idx + len(end) :]


def strip(text):
    if text is not None and isinstance(text, (str, bytes)):
        return text.strip()
    return text


def to_ascii(text):
    """ascii-normalized text.
    >>> to_ascii(u'\xa0')
    ' '
    >>> to_ascii(u'\xe9')
    'e'
    >>> to_ascii('bats\u00E0')
    'batsa'
    """
    if not text.isascii():
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text


def remove_position(slicable, pos_idx):
    """вернем исходное с удаленной позицией с индексом pos_idx

    >>> remove_position('012345', 3)
    '01245'
    >>> remove_position('012345', 5)
    '01234'
    >>> remove_position('012345', 6)
    '012345'
    """
    return slicable[0:pos_idx] + slicable[pos_idx + 1 :]


def cyrillic_misprint_to_latin(text):
    """
    Возращаются преобразованный текст:
    Вместо кириллических символов, пишущихся одинаково с латинскими, ставим латинские.
    """
    if text is None:
        return None
    converted = text
    rus_to_latin = {
        "\0": "",
        "\xe0": "a",
        "\xe5": "e",
        "\xea": "k",
        "\xec": "m",
        "\xee": "o",
        "\xf0": "p",
        "\xf1": "c",
        "\xf3": "y",
        "\xf5": "x",
        "\xc0": "A",
        "\xc5": "E",
        "\xca": "K",
        "\xcc": "M",
        "\xce": "O",
        "\xd0": "P",
        "\xd1": "C",
        "\xd3": "Y",
        "\xd5": "X",
        "\xc2": "B",
        "\xd2": "T",
        "\xcd": "H",
        "\xe8": "u",
    }
    for idx, symb in enumerate(text):
        if symb in rus_to_latin:
            converted = converted[:idx] + rus_to_latin[symb] + converted[idx + 1:]
    return converted


def joined_name(iterable, sep="_"):
    return "_".join((str(i) for i in iterable))


def date_to_str(date: datetime.date, sep: str = "-") -> str:
    return f"{date.year:04d}{sep}{date.month:02d}{sep}{date.day:02d}"


def date_from_str(date_str: str) -> datetime.date:
    """
    date('yyyy-mm-dd')

    >>> date_from_str('2020-08-17')
    datetime.date(2020, 8, 17)
    """
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def binary_oper_result_name(oper_fun, first_arg_name, second_arg_name):
    return oper_fun.__name__ + "_" + first_arg_name + "_" + second_arg_name


def find_first_xpath(from_element, xpath):
    elements = from_element.xpath(xpath)
    if elements:
        return elements[0]


def find_first(iterable, predicate, default_value=None):
    if iterable is None:
        return default_value
    for item in iterable:
        if predicate(item):
            return item
    return default_value


def find_all(iterable, predicate):
    if iterable is None:
        return []
    return [item for item in iterable if predicate(item)]


def find_indexed_first(iterable, predicate, default_value=None):
    if iterable is None:
        return -1, default_value
    for idx, item in enumerate(iterable):
        if predicate(item):
            return idx, item
    return -1, default_value


def add_pair(appendable: List, first, second):
    appendable.append(first)
    appendable.append(second)


def count(iterable, predicate):
    return sum((1 for obj in iterable if predicate(obj)))


def reversed_tuple(in_tuple):
    return tuple(reversed(in_tuple))


def neighbor_pairs(iterable):
    """
    Выдает соседствующие пары, где первый элемент левее второго.
    (s0,s1), (s1,s2), (s2, s3), ...

    >>> list(neighbor_pairs([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    >>> list(neighbor_pairs([1, 1]))
    [(1, 1)]
    >>> list(neighbor_pairs([1]))
    []
    >>> list(neighbor_pairs([]))
    []
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# serve quadrants:
DEUCE, ADV = ("DEUCE", "ADV")


(LEFT, RIGHT, ANY) = (Side("LEFT"), Side("RIGHT"), Side("ANY"))


def side(left):
    if left is not None:
        return Side(left)


class PredictResult(IntEnum):
    undefined = -3
    retired = -2
    empty = -1
    lose = 0
    win = 1


@functools.total_ordering
class Position(object):
    def __init__(self, current=None, size=None):
        self.current = current
        self.size = size

    def exist(self):
        return self.current is not None

    def ratio(self):
        if self.size is None or self.size == 0:
            return None
        return float(self.current) / float(self.size)

    def __eq__(self, other):
        rat = self.ratio()
        rat_other = other.ratio()
        if rat is None and rat_other is None:
            return True
        if rat is None or rat_other is None:
            return False
        return equal_float(rat, rat_other)

    def __lt__(self, other):
        return self.ratio() < other.ratio()

    def __str__(self):
        return self.percented_text(with_size=False)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.current, self.size)

    def __hash__(self):
        return hash((self.current, self.size))

    def percented_text(self, with_size=False):
        if self.current is None or self.size is None:
            return "[_%]"
        if self.size == 0:
            return "[0% (0)]" if with_size else "0%"
        else:
            result = "{0:4.1f}%".format(float(self.current * 100) / float(self.size))
            if with_size:
                return "[{0} ({1})]".format(result, self.size)
            else:
                return "[" + result + "]"


def make_weights(sizes):
    sum_all = sum(sizes)
    if sum_all > 0:
        return [float(i) / float(sum_all) for i in sizes]
    return [None for i in sizes]


def percented_text(part_count, all_count, full_format=True):
    """
    выдает part_count / all_count в процентном виде с плав. точкой
    если full_format, то еще в скобках общее к-во all_count и выравнивающие пробелы
    (заключ. пробелы исходя из того что макс. к-во all_count занимает 5 знаков)
    """
    if part_count is None or all_count is None:
        return "_%"
    if all_count == 0:
        return "0% (0)" if full_format else "0%"
    else:
        result = "{0:4.1f}%".format(float(part_count * 100) / float(all_count))
        if full_format:
            return "{0} ({1}) {2}".format(
                result, all_count, " " * (5 - len(str(all_count)))
            )
        else:
            return result


def float_to_str(value, round_digits=2, none_str="-"):
    if value is None:
        return none_str
    return str(round(value, round_digits))


def formated(value, size, round_digits=2):
    value_str = float_to_str(value, round_digits)
    return "{0:5} ({1}) {2}".format(value_str, size, " " * (5 - len(str(size))))


def to_align_text(rows, completer=" ", sep=" "):
    def max_col_widths(rows):
        results = [0 for j in range(len(rows[0]))]
        for i in range(len(rows)):
            row = rows[i]
            for j in range(len(row)):
                width = len(str(row[j]))
                if width > results[j]:
                    results[j] = width
        return results

    if not rows:
        return ""
    assert len(completer) <= 1, "too wide completer: '{}'".format(completer)
    col_widths = max_col_widths(rows)
    result = ""
    for i in range(len(rows)):
        row = rows[i]
        line = ""
        for j in range(len(row)):
            field = row[j]
            width = len(str(field))
            line += "{}{}".format(field, completer * (col_widths[j] - width))
            if j < (len(row) - 1):
                line += sep
        result += line
        if i < (len(rows) - 1):
            result += "\n"
    return result


def to_align_list(rows, completer=" ", sep=" "):
    def max_col_widths(rows):
        results = [0 for j in range(len(rows[0]))]
        for i in range(len(rows)):
            row = rows[i]
            for j in range(len(row)):
                width = len(str(row[j]))
                if width > results[j]:
                    results[j] = width
        return results

    if not rows:
        return []
    assert len(completer) <= 1, "too wide completer: '{}'".format(completer)
    col_widths = max_col_widths(rows)
    result = []
    for i in range(len(rows)):
        row = rows[i]
        line = ""
        for j in range(len(row)):
            field = row[j]
            width = len(str(field))
            line += "{}{}".format(field, completer * (col_widths[j] - width))
            if j < (len(row) - 1):
                line += sep
        result.append(line)
    return result


class Using(IntEnum):
    NO = 0
    IFEXIST = 1
    ALWAYS = 2


def is_odd(num):
    return (num % 2) == 1


def is_even(num):
    return (num % 2) == 0


def sign(x):
    return (x > 0) - (x < 0)


def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.
    """
    return (x > y) - (x < y)


# Отказ от sys.float_info.epsilon был связан со спецификой теннисных расчетов
# (кажется с ReportLine) где хотелось, чтобы блюлось что-то вроде assert(0 <= p(i) <= 1)
epsilon = 0.00001


def equal_float(a, b):
    return abs(a - b) <= epsilon


def to_interval(value, min_value=0.0, max_value=1.0):
    if value is None:
        return value
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


class Scope(IntEnum):
    POINT = 0
    GAME = 1
    SET = 2
    MATCH = 3


POINT, GAME, SET, MATCH = (Scope.POINT, Scope.GAME, Scope.SET, Scope.MATCH)


class Compare(IntEnum):
    LT = -1
    EQ = 0
    GT = 1


LT, EQ, GT = (Compare.LT, Compare.EQ, Compare.GT)


def value_compare(first_value, second_value, eps=0):
    """
    Cравнение двух величин. выдает LT если первое меньше,
    EQ если равны (с точностью до eps), GT если первое больше.
    """
    diff = first_value - second_value
    if abs(diff) <= eps:
        return EQ
    return LT if diff < 0 else GT


def flip_compare(value):
    if value == LT:
        return GT
    elif value == GT:
        return LT
    elif value == EQ:
        return EQ
    else:
        raise TennisError("unexpected compare value {}".format(value))


def centered_int_flip(value, range_min, range_max):
    if (range_min + range_max) % 2 == 0:
        center = (range_min + range_max) / 2
        return int(center + center - value)
    else:
        center = float(range_min + range_max) * 0.5
        return round(center + center - value)


def centered_float_flip(value, range_min, range_max):
    center = float(range_min + range_max) * 0.5
    return float(center + center) - value


def proportion_value(inval, inminval, inmaxval, outminval, outmaxval, infloatfun=float):
    """
     пропорциональное значение.

    >>> proportion_value(0.4, 0., 1., -5, 5)
    -1.0
    >>> from datetime import date
    >>> proportion_value(date(2016,1,5), date(2016,1,1), date(2016,1,11), -5, 5, infloatfun=lambda t: float(t.days))
    -1.0
    """
    outdelta = float(outmaxval - outminval)
    indelta = infloatfun(inmaxval - inminval)
    return outminval + outdelta * infloatfun(inval - inminval) / indelta


def balanced_value(value_first, size_first, value_second, size_second):
    size_all = size_first + size_second
    if size_all > 0:
        coef_first = float(size_first) / float(size_all)
        coef_second = float(size_second) / float(size_all)
        return value_first * coef_first + value_second * coef_second


def twoside_values(left_sizedvalue, right_sizedvalue, oppose_fun=lambda v: 1.0 - v):
    if left_sizedvalue is not None and left_sizedvalue.value is not None:
        left_value = left_sizedvalue.value
    else:
        left_value = None
    if right_sizedvalue is not None and right_sizedvalue.value is not None:
        right_value = right_sizedvalue.value
    else:
        right_value = None
    if left_value is None or right_value is None:
        return left_value, right_value
    left_value_opp = oppose_fun(left_value)
    right_value_opp = oppose_fun(right_value)
    left_value_result = balanced_value(
        left_value, left_sizedvalue.size, right_value_opp, right_sizedvalue.size
    )
    right_value_result = balanced_value(
        right_value, right_sizedvalue.size, left_value_opp, left_sizedvalue.size
    )
    return left_value_result, right_value_result


def oneside_values(first_sizedvalue, second_sizedvalue):
    if first_sizedvalue is not None and first_sizedvalue.value is not None:
        first_value = first_sizedvalue.value
    else:
        first_value = None
    if second_sizedvalue is not None and second_sizedvalue.value is not None:
        second_value = second_sizedvalue.value
    else:
        second_value = None

    if (
        first_value is None
        or second_value is None
        or first_sizedvalue.size == 0
        or second_sizedvalue.size == 0
    ):
        return first_value, second_value

    sum_size = first_sizedvalue.size + second_sizedvalue.size
    fst_coef = float(first_sizedvalue.size) / float(sum_size)
    snd_coef = float(second_sizedvalue.size) / float(sum_size)

    avg_value = (first_value + second_value) * 0.5
    fst_result = avg_value + (first_value - avg_value) * fst_coef
    snd_result = avg_value + (second_value - avg_value) * snd_coef
    sum_result = fst_result + snd_result
    return fst_result / sum_result, snd_result / sum_result


def smooth_proba(proba, factor=0.5):
    """
    >>> smooth_proba(proba=0.9)
    0.7
    >>> smooth_proba(proba=0.1)
    0.3
    """
    if proba > 0.5:
        return 0.5 + (proba - 0.5) * factor
    elif proba < 0.5:
        return 0.5 - (0.5 - proba) * factor
    else:
        return proba


if __name__ == "__main__":
    import doctest

    doctest.testmod()
