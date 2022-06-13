# -*- coding=utf-8 -*-
import unittest

from tour_name import tourname_number_split, normalized_name


class NormalizedNameTest(unittest.TestCase):
    def test_normalized_name(self):
        self.assertEqual(normalized_name('St. Petersburg'), 'st-petersburg')
        self.assertEqual(normalized_name('St.-Petersburg'), 'st-petersburg')
        self.assertEqual(normalized_name('Madrid'), 'madrid')
        self.assertEqual(normalized_name('Saint Malo'), 'saint-malo')
        self.assertEqual(normalized_name('Saint  Malo'), 'saint-malo')
        self.assertEqual(normalized_name('Saint - Malo'), 'saint-malo')
        self.assertEqual(normalized_name('U.S. Open'), 'us-open')


class TournameNumberSplitTest(unittest.TestCase):
    def test_tourname_number_split(self):
        name, num = tourname_number_split("Charleston 2")
        self.assertEqual((name, num), ("Charleston", 2))

        name, num = tourname_number_split("New Charleston 2")
        self.assertEqual((name, num), ("New Charleston", 2))

        name, num = tourname_number_split("Charleston 10")
        self.assertEqual((name, num), ("Charleston", 10))

        name, num = tourname_number_split("Charleston 2a")
        self.assertEqual((name, num), ("Charleston 2a", None))

        name, num = tourname_number_split("Charleston2")
        self.assertEqual((name, num), ("Charleston2", None))


if __name__ == '__main__':
    unittest.main()
