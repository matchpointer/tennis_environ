import unittest

from common import (
    Keys,
    strip_fragment,
    cyrillic_misprint_to_latin,
    value_compare,
    EQ,
    LT,
    GT,
    centered_int_flip,
    centered_float_flip,
    twoside_values,
)


class KeysTest(unittest.TestCase):
    def test_soft_main_init(self):
        import tennis

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("main"), rnd=tennis.Round("Second")
        )
        self.assertEqual(keys["level"], "main")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("main"), rnd=tennis.Round("q-First")
        )
        self.assertEqual(keys["level"], "qual")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("masters"), rnd=tennis.Round("Second")
        )
        self.assertEqual(keys["level"], "main")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("masters"), rnd=tennis.Round("q-First")
        )
        self.assertEqual(keys["level"], "main")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("gs"), rnd=tennis.Round("Second")
        )
        self.assertEqual(keys["level"], "main")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("gs"), rnd=tennis.Round("q-First")
        )
        self.assertEqual(keys["level"], "qual")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("chal"), rnd=tennis.Round("Second")
        )
        self.assertEqual(keys["level"], "chal")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("chal"), rnd=tennis.Round("q-First")
        )
        self.assertEqual(keys["level"], "chal")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("future"), rnd=tennis.Round("Second")
        )
        self.assertEqual(keys["level"], "future")

        keys = Keys.soft_main_from_raw(
            level=tennis.Level("future"), rnd=tennis.Round("q-First")
        )
        self.assertEqual(keys["level"], "future")


class MiscTextTest(unittest.TestCase):
    def test_strip_fragment(self):
        self.assertEqual(strip_fragment("abc(123)def", "(", ")"), "abcdef")
        self.assertEqual(strip_fragment("(abc123)def", "(", ")"), "def")
        self.assertEqual(strip_fragment("abc(123def)", "(", ")"), "abc")
        self.assertEqual(strip_fragment("(abc123def)", "(", ")"), "")

    def test_cyrillic_misprint_to_latin(self):
        latin_etalon = "Brkic"
        self.assertEqual(cyrillic_misprint_to_latin(latin_etalon), latin_etalon)

        missprinted = "Brki\xf1"  # cyrillic 'с'
        self.assertEqual(cyrillic_misprint_to_latin(missprinted), latin_etalon)

        missprinted = "\xc2rkic"  # cyrillic 'В'
        self.assertEqual(cyrillic_misprint_to_latin(missprinted), latin_etalon)

        missprinted = "\xc2rki\xf1"  # cyrillic 'В', 'c'
        self.assertEqual(cyrillic_misprint_to_latin(missprinted), latin_etalon)


class ValueCompareTest(unittest.TestCase):
    def test_value_compare(self):
        self.assertEqual(value_compare(1.999, 2.0, eps=0.001), EQ)
        self.assertEqual(value_compare(1.999, 2.0, eps=0.0001), LT)
        self.assertEqual(value_compare(2.0, 1.999, eps=0.0001), GT)
        self.assertEqual(value_compare(-1, 1, eps=2), EQ)
        self.assertEqual(value_compare(-1, 1, eps=1), LT)
        self.assertEqual(value_compare(1, -1, eps=1), GT)
        self.assertEqual(value_compare(-3, -1, eps=1), LT)
        self.assertEqual(value_compare(-1, -3, eps=1), GT)
        self.assertEqual(value_compare(-1, -3, eps=2), EQ)
        self.assertEqual(value_compare(-3, -1, eps=2), EQ)


class CenteredFlipTest(unittest.TestCase):
    def test_centered_flip(self):
        self.assertEqual(centered_int_flip(12, 12, 19), 19)
        self.assertEqual(centered_int_flip(13, 12, 19), 18)
        self.assertEqual(centered_int_flip(14, 12, 19), 17)
        self.assertEqual(centered_int_flip(15, 12, 19), 16)
        self.assertEqual(centered_int_flip(16, 12, 19), 15)
        self.assertEqual(centered_int_flip(17, 12, 19), 14)
        self.assertEqual(centered_int_flip(18, 12, 19), 13)
        self.assertEqual(centered_int_flip(19, 12, 19), 12)

        self.assertEqual(centered_int_flip(12, 12, 18), 18)
        self.assertEqual(centered_int_flip(13, 12, 18), 17)
        self.assertEqual(centered_int_flip(14, 12, 18), 16)
        self.assertEqual(centered_int_flip(15, 12, 18), 15)
        self.assertEqual(centered_int_flip(16, 12, 18), 14)
        self.assertEqual(centered_int_flip(17, 12, 18), 13)
        self.assertEqual(centered_int_flip(18, 12, 18), 12)

    def test_centered_float_flip(self):
        self.assertEqual(centered_float_flip(12.0, 12, 19), 19.0)
        self.assertEqual(centered_float_flip(12.1, 12, 19), 18.9)
        self.assertEqual(centered_float_flip(13.0, 12, 19), 18.0)
        self.assertEqual(centered_float_flip(14.0, 12, 19), 17.0)
        self.assertEqual(centered_float_flip(15.0, 12, 19), 16.0)
        self.assertEqual(centered_float_flip(16.0, 12, 19), 15.0)
        self.assertEqual(centered_float_flip(17.0, 12, 19), 14.0)
        self.assertEqual(centered_float_flip(18.0, 12, 19), 13.0)
        self.assertEqual(centered_float_flip(19.0, 12, 19), 12.0)

        self.assertEqual(centered_float_flip(12.25, 12, 18), 17.75)
        self.assertEqual(centered_float_flip(13.0, 12, 18), 17.0)
        self.assertEqual(centered_float_flip(14.0, 12, 18), 16.0)
        self.assertEqual(centered_float_flip(15.0, 12, 18), 15.0)
        self.assertEqual(centered_float_flip(16.0, 12, 18), 14.0)
        self.assertEqual(centered_float_flip(17.0, 12, 18), 13.0)
        self.assertEqual(centered_float_flip(17.8, 12, 18), 12.2)


class TwosideValuesTest(unittest.TestCase):
    def test_twoside_values(self):
        from report_line import SizedValue

        self.assertEqual(
            (0.6, 0.4), twoside_values(SizedValue(0.6, 100), SizedValue(0.4, 1))
        )
        self.assertEqual(
            (0.55, 0.45), twoside_values(SizedValue(0.65, 100), SizedValue(0.55, 100))
        )
        self.assertEqual(
            (0.65, 0.35), twoside_values(SizedValue(0.65, 100), SizedValue(0.5, 0))
        )

        p1, p2 = twoside_values(SizedValue(0.65, 1000), SizedValue(0.5, 1))
        self.assertTrue((0.65 - 0.01) <= p1 <= 0.65)
        self.assertTrue(0.35 <= p2 <= (0.35 + 0.01))


if __name__ == "__main__":
    unittest.main()
