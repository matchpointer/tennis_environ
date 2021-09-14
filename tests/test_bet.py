import unittest
from bet import WinCoefs, Offer


class WinCoefsTest(unittest.TestCase):
    def test_win_coefs(self):
        self.assertEqual(WinCoefs(1.9, 1.9).chances(), (0.5, 0.5))
        self.assertGreater(
            WinCoefs(1.8, 2.0).chances()[0], WinCoefs(1.8, 2.0).chances()[1]
        )
        self.assertLess(
            WinCoefs(1.86, 1.84).chances()[0], WinCoefs(1.86, 1.84).chances()[1]
        )

        self.assertLess(
            WinCoefs(1.86, 1.86).bookmaker_margin(),
            WinCoefs(1.85, 1.86).bookmaker_margin(),
        )
        self.assertEqual(WinCoefs(2.0, 2.0).bookmaker_margin(), 0.0)

        self.assertEqual(not WinCoefs(2.0, 2.0), False)
        self.assertEqual(not WinCoefs(2.0), True)
        self.assertEqual(not WinCoefs(), True)
        self.assertEqual(type(bool(WinCoefs())), bool)

    def test_offer_as_bool(self):
        offer = Offer(None)
        offer.win_coefs = WinCoefs(2.0, 2.0)
        self.assertEqual(not offer, False)
        self.assertEqual(type(not offer), bool)
        self.assertEqual(type(bool(offer)), bool)


if __name__ == '__main__':
    unittest.main()
