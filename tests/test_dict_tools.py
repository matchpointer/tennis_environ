import os
import unittest
from collections import Counter

import common as co
from dict_tools import dump, load, LevSurf
import stat_cont as st


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


if __name__ == '__main__':
    unittest.main()
