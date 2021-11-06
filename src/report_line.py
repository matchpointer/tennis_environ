import os
import re
import copy
from collections import defaultdict, Counter
import unittest

import common as co
import log
import cfg_dir
import dict_tools


# dict: keyi -> Counter
#               Counter: valuei -> int
# если этот объект пуст, то режим работы 'сегодня' (читать из файла)
dict_from_pid = defaultdict(lambda: defaultdict(Counter))


def player_total_report_lines(sex, player, total_value, is_less=True, keys=None):
    import total

    predicate = total.total_predicate(total_value, is_less)
    if not dict_from_pid:
        # режим работы 'сегодня' - читаем данные из текущей файловой статистики
        dirname = cfg_dir.stat_players_total_dir(sex)
        filename = os.path.join(
            dirname, "{}.txt".format(player.ident if player else None)
        )
        return ReportLineList.create_ratio_from_kcntr_file(filename, predicate)
    else:
        assert keys is not None, "None keys in player_total_report_lines given"
        dct = dict_from_pid[player.ident]
        items = []
        for key, cntr in dct.items():
            if key in keys:
                sized_ratio = SizedValue.create_ratio_from_cntr(cntr, predicate)
                items.append(
                    ReportLine(key=key, value=sized_ratio.value, size=sized_ratio.size)
                )
        return ReportLineList(items=items)


def find_report_line_in_file(filename, key):
    report_lines = ReportLineList(
        filename=filename, struct_key=isinstance(key, co.StructKey)
    )
    line_num = 0
    for rpt_line in report_lines:
        line_num += 1
        if rpt_line.key == key:
            return rpt_line, co.Position(line_num, len(report_lines))
    return None, None


class SizedValue:
    """
    представляет пару: value, size
    """

    line_re = re.compile(r"(?P<value>-?\d+\.\d+) +\((?P<size>\d+)\)")

    def __init__(self, value=None, size=0):
        self.value = value
        self.size = size

    @staticmethod
    def create_from_sumator(sumator):
        return SizedValue(value=sumator.average(), size=sumator.count)

    @staticmethod
    def create_from_text(text):
        line_match = SizedValue.line_re.match(text)
        if line_match:
            value = float(line_match.group("value"))
            size = int(line_match.group("size"))
            return SizedValue(value, size)
        raise co.TennisError("SizedValue not created from text: '{}'".format(text))

    @staticmethod
    def create_ratio_from_cntr(cntr, predicate):
        return SizedValue.create_from_cntr(
            cntr, num_sumitem_f=lambda k, c: c, num_preditem_f=lambda k, c: predicate(k)
        )

    @staticmethod
    def create_avg_from_cntr(cntr):
        return SizedValue.create_from_cntr(cntr, num_sumitem_f=lambda k, c: k * c)

    @staticmethod
    def create_from_cntr(
        cntr,
        num_sumitem_f,
        num_preditem_f=lambda k, c: True,
        denum_f=co.compose(sum, lambda x: x.values()),
    ):
        if cntr is None:
            return SizedValue()
        numerator = sum(
            (num_sumitem_f(k, c) for k, c in cntr.items() if num_preditem_f(k, c))
        )
        denumerator = denum_f(cntr)
        if denumerator == 0:
            return SizedValue()
        else:
            return SizedValue(float(numerator) / float(denumerator), denumerator)

    def __bool__(self):
        return self.value is not None and self.size > 0

    __nonzero__ = __bool__

    def tostring(self, precision=3):
        if self:
            fmt = "{:." + str(precision) + "f} ({})"
            return fmt.format(self.value, self.size)
        return "{} ({})".format(self.value, self.size)

    def __str__(self):
        if self.value is None or isinstance(self.value, str):
            return "{} ({})".format(self.value, self.size)
        else:
            return "{0:.2f} ({1})".format(self.value, self.size)

    def __repr__(self):
        return "{}(value={}, size={})".format(
            self.__class__.__name__, self.value, self.size
        )

    def __eq__(self, other):
        return co.equal_float(self.value, other.value) and self.size == other.size

    def __ne__(self, other):
        return not self.__eq__(other)

    def ballanced_with(self, other):
        if other is None or other.value is None:
            return self
        if self.value is None:
            return other
        value = co.balanced_value(self.value, self.size, other.value, other.size)
        return SizedValue(value, self.size + other.size)


class SizedValueTest(unittest.TestCase):
    def test_create_from_cntr(self):
        cntr = Counter(dict([(12, 2), (13, 3), (14, 2), (8, 0)]))
        sized_avg = SizedValue.create_avg_from_cntr(cntr)
        self.assertEqual(sized_avg.value, 13.0)
        self.assertEqual(sized_avg.size, sum(cntr.values()))


class ReportLine(SizedValue):
    """
    представляет троицу: key, value, size
    """

    line_re = re.compile(
        r"(?P<key>[^_]+)_* *(?P<value>-?\d+\.\d+)%? +\((?P<size>\d+)\) ?.*"
    )

    def __init__(self, key=None, value=None, size=0):
        super().__init__(value, size)
        self.key = key

    @staticmethod
    def create_from_text(text, struct_key=True):
        line_match = ReportLine.line_re.match(text)
        if line_match:
            if struct_key:
                key = co.StructKey.create_from_text(line_match.group("key"))
            else:
                key = line_match.group("key")
            value = float(line_match.group("value"))
            size = int(line_match.group("size"))
            return ReportLine(key, value, size)
        else:
            raise co.TennisError(
                "ReportLine not created when struct_key={} "
                "from text: '{}'".format(struct_key, text)
            )

    def __str__(self):
        return "{}__{}".format(str(self.key), super(ReportLine, self).__str__())

    def __repr__(self):
        return "{}(key={}, value={}, size={})".format(
            self.__class__.__name__, repr(self.key), self.value, self.size
        )

    def __eq__(self, other):
        return self.key == other.key and super(ReportLine, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        """ very resemble at common.balanced_value """
        assert isinstance(other, ReportLine), "invalid other type: '{}'".format(
            type(other)
        )
        sum_size = self.size + other.size
        if sum_size == 0:
            return ReportLine(self.key, None, 0)
        if other.size == 0:
            return ReportLine(self.key, self.value, self.size)
        if self.size == 0:
            return ReportLine(self.key, other.value, other.size)
        value = ((self.value * self.size) + (other.value * other.size)) / float(
            sum_size
        )
        return ReportLine(self.key, value, sum_size)

    def __sub__(self, other):
        assert isinstance(other, ReportLine), "invalid other type: '{}'".format(
            type(other)
        )
        assert (
            self.size >= other.size
        ), "fail substract, noncorrespond sizes, self: {} other: {}".format(
            self.size, other.size
        )
        diff_size = self.size - other.size
        if diff_size == 0:
            return ReportLine(self.key, None, 0)
        if other.size == 0:
            return ReportLine(self.key, self.value, self.size)
        if self.size == 0:
            return ReportLine(self.key, -other.value, other.size)
        value = ((self.value * self.size) - (other.value * other.size)) / float(
            diff_size
        )
        return ReportLine(self.key, value, diff_size)


class ReportLineTest(unittest.TestCase):
    def test_parse(self):
        rpt = ReportLine.create_from_text(
            "cou=QAT,tour=15th Asian Games Doha 2006___ 8.26 (70)", struct_key=True
        )
        self.assertEqual(
            (rpt.key, rpt.value, rpt.size),
            (co.StructKey(cou="QAT", tour="15th Asian Games Doha 2006"), 8.26, 70),
        )

        rpt = ReportLine.create_from_text(
            "cou=TUR,tour=Turkey #2-w3___ 8.52 (60)", struct_key=True
        )
        self.assertEqual(
            (rpt.key, rpt.value, rpt.size),
            (co.StructKey(cou="TUR", tour="Turkey #2-w3"), 8.52, 60),
        )

        rpt = ReportLine.create_from_text(
            "all_____________________  5.42 (271)", struct_key=False
        )
        self.assertEqual((rpt.key, rpt.value, rpt.size), ("all", 5.42, 271))

        rpt = ReportLine.create_from_text(
            "U.S. Open_____________________   999.942 (271)", struct_key=False
        )
        self.assertEqual((rpt.key, rpt.value, rpt.size), ("U.S. Open", 999.942, 271))

        rpt = ReportLine.create_from_text(
            "Vignesh Peranamallur-Chandrasekhar_  18.655 (87)", struct_key=False
        )
        self.assertEqual(
            (rpt.key, rpt.value, rpt.size),
            ("Vignesh Peranamallur-Chandrasekhar", 18.655, 87),
        )

    def test_add(self):
        r1 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=1.0,
            size=1,
        )
        r2 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=0.0,
            size=1,
        )
        radd = r1 + r2
        self.assertEqual(
            radd,
            ReportLine(
                key=co.StructKey(level="chal", surface="Hard", rnd="Final"),
                value=0.5,
                size=2,
            ),
        )
        r1 += r2
        self.assertEqual(r1, radd)
        r0 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=3.0,
            size=0,
        )
        self.assertEqual(r2, r2 + r0)

    def test_sub(self):
        r1 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=0.5,
            size=2,
        )
        r2 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=1.0,
            size=1,
        )
        rsub = r1 - r2
        self.assertEqual(
            rsub,
            ReportLine(
                key=co.StructKey(level="chal", surface="Hard", rnd="Final"),
                value=0.0,
                size=1,
            ),
        )
        r1 -= r2
        self.assertEqual(r1, rsub)
        r0 = ReportLine(
            key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
            value=3.0,
            size=0,
        )
        self.assertEqual(r2, r2 - r0)


class ReportLineList:
    def __init__(self, filename=None, struct_key=True, eval_key=False, items=None):
        """
        Варианты создания: 1) filename, struct_key
                           2) посл-ть items
                           3) без аргументов - будет пустой.
        """
        if items:
            self._report_lines = items[:]
            assert not filename, "ReportLineList init unexpected file: '{}'".format(
                filename
            )
        elif filename:
            self._report_lines = []
            assert not (
                struct_key and eval_key
            ), "struct_key and eval_key are not compatible when init"
            assert not items, "ReportLineList init unexpected items: '{}'".format(items)
            if os.path.isfile(filename):
                try:
                    with open(filename, "r") as fhandle:
                        for line in fhandle.readlines():
                            if line.startswith("#"):
                                continue
                            rpt_line = ReportLine.create_from_text(line, struct_key)
                            if eval_key:
                                rpt_line.key = eval(rpt_line.key)
                            self._report_lines.append(rpt_line)
                except Exception as err:
                    raise co.TennisError(
                        "{} - ReportLineList init failed at file: '{}'".format(
                            err, filename
                        )
                    )
        else:
            self._report_lines = []

    @staticmethod
    def create_ratio_from_kcntr_file(filename, predicate):
        dct = dict_tools.load(
            filename,
            createfun=lambda: defaultdict(lambda: None),
            keyfun=co.StructKey.create_from_text,
        )
        items = []
        for key, cntr in dct.items():
            sized_ratio = SizedValue.create_ratio_from_cntr(cntr, predicate)
            items.append(
                ReportLine(key=key, value=sized_ratio.value, size=sized_ratio.size)
            )
        return ReportLineList(items=items)

    def __len__(self):
        return len(self._report_lines)

    def __bool__(self):
        return len(self._report_lines) > 0

    __nonzero__ = __bool__

    def __getitem__(self, index):
        return self._report_lines[index]

    def __setitem__(self, index, value):
        raise co.TennisError("ReportLineList immutable set error")

    def __delitem__(self, index):
        raise co.TennisError("ReportLineList immutable del error")

    def __iter__(self):
        return iter(self._report_lines)

    def sort(self, key=lambda i: i.key, reverse=False):
        self._report_lines.sort(key=key, reverse=reverse)

    def find_first(self, key):
        for rpt_line in self._report_lines:
            if rpt_line.key == key:
                return rpt_line

    def to_str(self):
        if len(self._report_lines) == 0:
            return "empty report line list"
        key_len_max = max([len(str(r.key)) for r in self._report_lines])
        fmt = "{0:_<" + str(key_len_max + 2) + "}  {1}\n"
        result = ""
        for rpt_line in self._report_lines:
            result += fmt.format(
                str(rpt_line.key), str(SizedValue(rpt_line.value, rpt_line.size))
            )
        return result

    def to_interval(self):
        for rpt_line in self._report_lines:
            rpt_line.value = co.to_interval(rpt_line.value)

    def sub_detailed(self, keys):
        results = []
        for rpt_line in self._report_lines:
            for key in keys:
                if rpt_line.key == key:
                    results.append(rpt_line)
                    break
        return ReportLineList(items=results)

    def value_reversed(self):
        """
        Возращается ReportLineList где value = 1 - old_value.
        (чтобы получить противоположную вероятность).
        """
        results = []
        for rpt_line in self._report_lines:
            assert (
                -co.epsilon <= rpt_line.value <= (1.0 + co.epsilon)
            ), "invalid value ({}) during reversing".format(rpt_line.value)
            results.append(
                ReportLine(
                    key=rpt_line.key,
                    size=rpt_line.size,
                    value=co.to_interval(
                        1.0 - rpt_line.value, min_value=0.0, max_value=1.0
                    ),
                )
            )
        return ReportLineList(items=results)

    def normalized(self):
        rpt_from_key = {}
        for item in self._report_lines:
            if item.size > 0:
                rpt_from_key[item.key] = copy.copy(item)
        zero_keys = [k for k in rpt_from_key.keys() if len(k) == 0]
        assert len(zero_keys) <= 1, "only single 0-key must be"

        min_len = min([len(k) for k in rpt_from_key.keys()])
        max_len = max([len(k) for k in rpt_from_key.keys()])
        for cur_len_high in reversed(list(range(min_len + 1, max_len + 1))):
            for cur_key_high in [
                kh for kh in rpt_from_key.keys() if len(kh) == cur_len_high
            ]:
                for cur_len_low in range(min_len, cur_len_high):
                    for cur_key_low in [
                        kl for kl in rpt_from_key.keys() if len(kl) == cur_len_low
                    ]:
                        if set(cur_key_low.key_names()) < set(cur_key_high.key_names()):
                            rpt_from_key[cur_key_low] -= rpt_from_key[cur_key_high]
        return ReportLineList(
            items=[rpt for rpt in rpt_from_key.values() if rpt.size > 0]
        )

    def mixed(self, other_report_lines):
        mixed_lines = []
        for report_line in self._report_lines:
            rl_key = report_line.key
            oposite_line = co.find_first(
                other_report_lines, predicate=lambda x, rk=rl_key: x.key == rk
            )
            if oposite_line:
                mixed_lines.append(report_line + oposite_line)
            else:
                mixed_lines.append(report_line)

        for report_line in other_report_lines:
            if (
                co.find_first(mixed_lines, predicate=lambda x: x.key == report_line.key)
                is None
            ):
                mixed_lines.append(report_line)
        return ReportLineList(items=mixed_lines)

    def size_sum(self):
        return sum([r.size for r in self._report_lines])

    def size_at_all_key(self):
        """Ищется размер элемента с пустым ключем (соответсвует категории all)"""
        allkey_rpt_line = self.find_first(key=co.StructKey())
        if allkey_rpt_line:
            return allkey_rpt_line.size

    def weight_value(self, weights):
        nw_sum = sum([weights[len(r.key)] * r.size for r in self._report_lines])
        if co.equal_float(nw_sum, 0.0):
            return None
        # '~probability of 1 unit': so that sum([coef])=1,
        # where coef(i) = size(i)*weight(i)*mult
        mult = 1.0 / float(nw_sum)
        return sum(
            [
                weights[len(r.key)] * r.size * mult * r.value
                for r in self._report_lines
                if r.value is not None
            ]
        )


class ReportLineListTest(unittest.TestCase):
    def test_mixed(self):
        reports1 = ReportLineList(
            items=[
                ReportLine(
                    key=co.StructKey(level="chal", rnd="Final"), value=1.0, size=1
                ),
                ReportLine(
                    key=co.StructKey(surface="Hard", level="chal", rnd="Final"),
                    value=1.0,
                    size=1,
                ),
                ReportLine(key=co.StructKey(rnd="Final"), value=1.0, size=1),
                ReportLine(key=co.StructKey(level="chal"), value=1.0, size=1),
                ReportLine(key=co.StructKey(surface="Hard"), value=1.0, size=1),
            ]
        )

        reports2 = ReportLineList(
            items=[
                ReportLine(
                    key=co.StructKey(level="chal", rnd="Final"), value=0.0, size=1
                ),
                ReportLine(key=co.StructKey(rnd="Final"), value=0.0, size=1),
                ReportLine(key=co.StructKey(level="chal"), value=0.0, size=1),
            ]
        )

        mix = reports1.mixed(reports2)
        self.assertEqual(
            True,
            ReportLine(key=co.StructKey(level="chal", rnd="Final"), value=0.5, size=2)
            in mix,
        )

    def test_normalized2(self):
        a = ReportLine(key=co.StructKey(a="a"), value=0.5, size=100)
        b = ReportLine(key=co.StructKey(b="b"), value=0.6, size=100)
        ab = ReportLine(key=co.StructKey(a="a", b="b"), value=0.54, size=10)

        # do full
        fa = a + ab
        fb = b + ab
        full = ReportLineList(items=[fa, fb, ab])

        normaled = full.normalized()
        self.assertTrue(a in normaled)
        self.assertTrue(b in normaled)
        self.assertTrue(ab in normaled)
        self.assertEqual(len(normaled), 3)

    def test_normalized2milk(self):
        a = ReportLine(key=co.StructKey(a="a"), value=0.5, size=100)
        b = ReportLine(key=co.StructKey(b="b"), value=0.6, size=100)
        ab = ReportLine(key=co.StructKey(a="a", b="b"), value=0.54, size=10)
        milk = ReportLine(key=co.StructKey(), value=0.3, size=50)

        # do full
        fa = a + ab
        fb = b + ab
        fmilk = milk + a + b + ab
        full = ReportLineList(items=[fmilk, fa, fb, ab])

        normaled = full.normalized()
        self.assertTrue(a in normaled)
        self.assertTrue(b in normaled)
        self.assertTrue(ab in normaled)
        self.assertTrue(milk in normaled)
        self.assertEqual(len(normaled), 4)

    def test_normalized3(self):
        a = ReportLine(key=co.StructKey(a="a"), value=0.5, size=100)
        b = ReportLine(key=co.StructKey(b="b"), value=0.4, size=100)
        c = ReportLine(key=co.StructKey(c="c"), value=0.6, size=100)

        ab = ReportLine(key=co.StructKey(a="a", b="b"), value=0.42, size=10)
        ac = ReportLine(key=co.StructKey(a="a", c="c"), value=0.58, size=10)
        bc_ = ReportLine(key=co.StructKey(b="b", c="c"), value=0.52, size=10)

        abc = ReportLine(key=co.StructKey(a="a", b="b", c="c"), value=0.5, size=5)
        milk = ReportLine(key=co.StructKey(), value=0.3, size=50)

        # do full
        fa = a + ab + ac + abc
        fb = b + ab + bc_ + abc
        fc = c + ac + bc_ + abc

        fab = ab + abc
        fbc = bc_ + abc
        fac = ac + abc

        fmilk = milk + a + b + c + ab + ac + bc_ + abc
        full = ReportLineList(items=[fmilk, fa, fb, fc, fab, fac, fbc, abc])

        normaled = full.normalized()
        self.assertTrue(a in normaled)
        self.assertTrue(b in normaled)
        self.assertTrue(c in normaled)
        self.assertTrue(ab in normaled)
        self.assertTrue(ac in normaled)
        self.assertTrue(bc_ in normaled)
        self.assertTrue(abc in normaled)
        self.assertTrue(milk in normaled)
        self.assertEqual(len(normaled), 8)

    def test_normalized4(self):
        a = ReportLine(key=co.StructKey(a="a"), value=0.5, size=100)
        b = ReportLine(key=co.StructKey(b="b"), value=0.4, size=100)
        c = ReportLine(key=co.StructKey(c="c"), value=0.6, size=100)
        d = ReportLine(key=co.StructKey(d="d"), value=0.7, size=100)

        ab = ReportLine(key=co.StructKey(a="a", b="b"), value=0.42, size=10)
        ac = ReportLine(key=co.StructKey(a="a", c="c"), value=0.58, size=10)
        ad = ReportLine(key=co.StructKey(a="a", d="d"), value=0.54, size=10)
        bc_ = ReportLine(key=co.StructKey(b="b", c="c"), value=0.52, size=10)
        bd = ReportLine(key=co.StructKey(b="b", d="d"), value=0.53, size=10)
        cd = ReportLine(key=co.StructKey(c="c", d="d"), value=0.51, size=10)

        abc = ReportLine(key=co.StructKey(a="a", b="b", c="c"), value=0.5, size=5)
        abd = ReportLine(key=co.StructKey(a="a", b="b", d="d"), value=0.55, size=5)
        acd = ReportLine(key=co.StructKey(a="a", c="c", d="d"), value=0.48, size=5)
        bcd = ReportLine(key=co.StructKey(b="b", c="c", d="d"), value=0.57, size=5)

        abcd = ReportLine(
            key=co.StructKey(a="a", b="b", c="c", d="d"), value=0.59, size=2
        )
        milk = ReportLine(key=co.StructKey(), value=0.33, size=50)

        # do full
        fa = a + ab + ac + ad + abc + abd + acd + abcd
        fb = b + ab + bc_ + bd + abc + abd + bcd + abcd
        fc = c + ac + bc_ + cd + abc + acd + bcd + abcd
        fd = d + ad + bd + cd + abd + acd + bcd + abcd

        fab = ab + abc + abd + abcd
        fbc = bc_ + abc + bcd + abcd
        fac = ac + abc + acd + abcd
        fad = ad + abd + acd + abcd
        fbd = bd + abd + bcd + abcd
        fcd = cd + acd + bcd + abcd

        fabc = abc + abcd
        fabd = abd + abcd
        facd = acd + abcd
        fbcd = bcd + abcd

        fmilk = (
            milk
            + a
            + b
            + c
            + d
            + ab
            + ac
            + ad
            + bc_
            + bd
            + cd
            + abc
            + abd
            + acd
            + bcd
            + abcd
        )
        full = ReportLineList(
            items=[
                fmilk,
                fa,
                fb,
                fc,
                fd,
                fab,
                fbc,
                fac,
                fad,
                fbd,
                fcd,
                fabc,
                fabd,
                facd,
                fbcd,
                abcd,
            ]
        )
        normaled = full.normalized()
        self.assertTrue(a in normaled)
        self.assertTrue(b in normaled)
        self.assertTrue(c in normaled)
        self.assertTrue(d in normaled)

        self.assertTrue(ab in normaled)
        self.assertTrue(bc_ in normaled)
        self.assertTrue(ac in normaled)
        self.assertTrue(ad in normaled)
        self.assertTrue(bd in normaled)
        self.assertTrue(cd in normaled)

        self.assertTrue(abc in normaled)
        self.assertTrue(abd in normaled)
        self.assertTrue(acd in normaled)
        self.assertTrue(bcd in normaled)

        self.assertTrue(abcd in normaled)
        self.assertTrue(milk in normaled)
        self.assertEqual(len(normaled), 16)


if __name__ == "__main__":
    log.initialize(co.logname(__file__, test=True), "debug", None)
    unittest.main()
