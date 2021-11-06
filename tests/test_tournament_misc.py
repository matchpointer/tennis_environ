import unittest

from tournament_misc import tourname_number_split


class TournameNumberSplitTest(unittest.TestCase):
    def test_tourname_number_split(self):
        name, num = tourname_number_split("Charleston 2")
        self.assertEqual((name, num), ("Charleston", 2))

        name, num = tourname_number_split("Charleston 10")
        self.assertEqual((name, num), ("Charleston", 10))

        name, num = tourname_number_split("Charleston 2a")
        self.assertEqual((name, num), ("Charleston 2a", None))

        name, num = tourname_number_split("Charleston2")
        self.assertEqual((name, num), ("Charleston2", None))


if __name__ == '__main__':
    unittest.main()
