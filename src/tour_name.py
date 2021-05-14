# -*- coding: utf-8 -*-
"""
TODO - flashscore.py when Davis Cup uses tour_name.add_part(part: str)
     - Fed Cup tours from oncourt db loaded with coma in tour_name (differ from atp)
"""


class TourName:
    def __init__(self, name: str, desc_name: str = "", number=None):
        self.name = name
        self.desc_name = desc_name
        self.number = number
        if (
            self.desc_name
            in (
                "Australian Open",
                "U.S. Open",
                "French Open",
                "Wimbledon",
                "Fed Cup",
                "Davis Cup",
                "ATP Cup",
                "Olympics",
            )
            or self.desc_name.endswith("ATP Finals")
            or self.desc_name.endswith("WTA Finals")
        ):
            self.name, self.desc_name = self.desc_name, self.name

    def __contains__(self, item):
        return item in self.name

    def __hash__(self):
        return hash((self.name, self.number, self.desc_name))

    def __eq__(self, other):
        if self.desc_name and other.desc_name:
            return (
                self.name == other.name
                and self.number == other.number
                and self.desc_name == other.desc_name
            )
        return self.name == other.name and self.number == other.number

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.number is None:
            return self.name
        return f"{self.name} {self.number}"

    def grand_slam(self):
        return self.name in ("Australian Open", "U.S. Open", "French Open", "Wimbledon")

    @staticmethod
    def from_oncourt(raw_name):
        from common import to_ascii, split_ontwo

        raw = to_ascii(raw_name)
        desc_name, name = split_ontwo(raw, delim=" - ")
        if name == "":
            name = raw
            desc_name = ""
        number = None
        if name and name[-1].isdigit() and name[-2] == " ":
            number = int(name[-1])
            if number == 1:
                number = None  # to compare with flashscore 1-less name
            name = name[:-2]
        return TourName(name=name, desc_name=desc_name, number=number)
