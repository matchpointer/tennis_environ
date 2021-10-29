import unittest

from tennis import Round
import stat_cont as st


class OrderedTypesTest(unittest.TestCase):
    def test_round_compare(self):
        self._strict_lt_compare(Round("q-Second"), Round("First"))
        self._strict_lt_compare(Round("q-Second"), "First")
        self._strict_lt_compare(Round("Pre-q"), Round("q-First"))

        self.assertFalse(Round("Final") != Round("Final"))
        self.assertEqual(Round("q-Second"), Round("q-Second"))
        self.assertEqual(Round("q-Second"), "q-Second")
        self.assertEqual("q-Second", Round("q-Second"))

        lst = [Round("First"), Round("Pre-q"), Round("q-First")]
        lst_asc = [Round("Pre-q"), Round("q-First"), Round("First")]
        lst_desc = [Round("First"), Round("q-First"), Round("Pre-q")]
        self.assertEqual(sorted(lst, reverse=False), lst_asc)
        self.assertEqual(sorted(lst, reverse=True), lst_desc)

    def test_winloss_compare(self):
        self._strict_lt_compare(st.WinLoss(1, 3), st.WinLoss(2, 5))

        self.assertEqual(st.WinLoss(1, 3), st.WinLoss(10, 30))

    def _strict_lt_compare(self, first, second):
        """
        at input we have case: first < second
        """
        self.assertTrue(first < second)
        self.assertTrue(first <= second)

        self.assertTrue(second > first)
        self.assertTrue(second >= first)

        self.assertTrue(first != second)
        self.assertTrue(second != first)

        self.assertFalse(first == second)

        self.assertFalse(first > second)
        self.assertFalse(first >= second)


if __name__ == "__main__":
    unittest.main()
