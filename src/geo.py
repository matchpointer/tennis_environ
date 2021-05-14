# -*- coding: utf-8 -*-

import unittest

cou_to_cities = {
    "ITA": ("Biella", "Ortisei", "Bergamo"),
    "GBR": ("London", "Birmingham", "Barnstaple", "Loughborough"),
    "FRA": ("Paris", "Lion", "Nantes"),
    "SWE": ("Stockholm",),
    "SUI": ("Basel",),
    "GER": ("Eckental", "Berlin", "Hamburg", "Munich", "Cologne"),
    "ESP": ("Madrid", "Valencia", "Barcelona"),
    "NED": ("Rotterdam", "Amsterdam"),
    "RUS": ("St. Petersburg", "Moscow", "Kazan"),
    "KAZ": ("Nur-Sultan",),
    "SVK": ("Bratislava",),
    "IND": ("Pune",),
    "TUR": ("Istanbul",),
    "USA": ("Knoxville",),
    "BRA": ("Sao Leopoldo",),
    "ECU": ("Guayaquil",),
}


# http://goeasteurope.about.com/od/easterneuropedestinations/ss/Countries-Of-Eastern-Europe.htm
sub_parts = {
    "east_europe": (
        "RUS",
        "CZE",
        "POL",
        "HUN",
        "ROM",
        "MDA",
        "CRO",
        "LAT",
        "EST",
        "LTU",
        "SLO",
        "SVK",
        "BUL",
        "UKR",
        "SRB",
        "BLR",
        "MKD",
        "KAZ",
    ),
    "west_europe": ("ITA", "GBR", "FRA", "SWE", "ESP", "SUI", "GER", "NED"),
    "south_europe": ("TUR",),
    "states": ("USA",),
    "canada": ("CAN",),
    "mexico": ("MEX",),
}

parts = {
    "europe": ("south_europe", "east_europe", "west_europe"),
    "north_america": ("states", "canada", "mexico"),
    "south_america": ("south_caribes", "brasil", "argentina"),
}


def part_countries(partname):
    result = []
    for subpart in parts[partname]:
        for cou in sub_parts[subpart]:
            result.append(cou)
    return result


def city_in_europe(city):
    for cou, cities in cou_to_cities.items():
        if city in cities:
            return cou in part_countries("europe")


class CityLocationTest(unittest.TestCase):
    def test_city_in_europe(self):
        self.assertTrue(city_in_europe("Istanbul"))
        self.assertTrue(city_in_europe("Rotterdam"))
        self.assertTrue(city_in_europe("Cologne"))
        self.assertTrue(city_in_europe("Bergamo"))


if __name__ == "__main__":
    unittest.main()
