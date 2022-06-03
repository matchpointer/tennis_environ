import unittest

from surf import make_surf, Clay, Grass, get_code, get_name


class SurfTest(unittest.TestCase):
    def test_cmp(self):
        self.assertTrue(make_surf('Clay') is Clay)
        self.assertTrue(make_surf('Clay') == 'Clay')

        self.assertTrue('Clay' == str(make_surf('Clay')))
        self.assertTrue('clay' != make_surf('Clay'))

        self.assertRaises(
            ValueError,
            lambda: int(Clay),
        )

    def test_grass_values(self):
        expect_grass_code = 3
        self.assertEqual(get_code(Grass), expect_grass_code)
        self.assertEqual(get_name(expect_grass_code), Grass)


if __name__ == '__main__':
    unittest.main()
