import unittest

from geo import city_in_europe


class CityLocationTest(unittest.TestCase):
    def test_city_in_europe(self):
        self.assertTrue(city_in_europe("Istanbul"))
        self.assertTrue(city_in_europe("Rotterdam"))
        self.assertTrue(city_in_europe("Cologne"))
        self.assertTrue(city_in_europe("Bergamo"))


if __name__ == '__main__':
    unittest.main()
