import unittest

from geo import city_in_europe
from country_code import alpha_3_code


class Alpha3CountryCodeTest(unittest.TestCase):
    def test_alpha_3_country_code(self):
        self.assertEqual("GEO", alpha_3_code("Georgia"))
        self.assertEqual("ESP", alpha_3_code("Spain"))
        self.assertEqual("GRE", alpha_3_code("Greece"))
        self.assertEqual("USA", alpha_3_code("USA"))
        self.assertEqual("HKG", alpha_3_code("Hong Kong"))


class CityLocationTest(unittest.TestCase):
    def test_city_in_europe(self):
        self.assertTrue(city_in_europe("Istanbul"))
        self.assertTrue(city_in_europe("Rotterdam"))
        self.assertTrue(city_in_europe("Cologne"))
        self.assertTrue(city_in_europe("Bergamo"))


if __name__ == '__main__':
    unittest.main()
