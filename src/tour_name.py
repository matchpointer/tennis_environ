# -*- coding=utf-8 -*-
"""
предоставляет класс TourName
мотивация:
    - в связи с пандемией появились турниры, стартующие на той же неделе
      и в том же городе и различ-ся лишь номером (ну и длинным описанием)
      например в Австралии в 2021 january, febrary.
      Описание в модели никак не было задествовано ранее и главным был город,
      то теперь главным будет (город, номер) ну и вводится описание для доп. информации

    - традиционная нумерация тоже остается, когда след. турнир сдвинут позже,
      но если он сдвинут всего на неделю вперед, то это в каких-то моментах
      напоминает предельную ситуацию (на той же неделе), отличие от предельной
      ситуации в том что не привлекается описание, которое тут не играет различительной роли

особенность нумерации:
    - flashscore не использует номер 1 (со второго если есть номер)
    - oncourt использует номер 1. Поэтому для совместимости есть метод from_oncourt,
      удаляющий 1


TODO - flashscore.py when Davis Cup uses tour_name.add_part(part: str)
     - Fed Cup tours from oncourt db loaded with coma in tour_name (differ from atp)
"""
import re


class TourName:
    def __init__(self, name: str, desc_name: str = "", number=None):
        self.name = normalized_name(name) if name else name  # location (usially)
        self.desc_name = normalized_name(desc_name) if desc_name else desc_name
        self.number = number
        if (
            self.desc_name
            in (
                "australian-open",
                "us-open",
                "french-open",
                "wimbledon",
                "fed-cup",
                "davis-cup",
                "atp-cup",
                "olympics",
            )
            or self.desc_name.endswith("atp-finals")
            or self.desc_name.endswith("wta-finals")
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

    def __repr__(self):
        if self.number is None:
            return self.name
        return f"{self.name} {self.number}"

    def grand_slam(self):
        return self.name in ("australian-open", "us-open", "french-open", "wimbledon")

    def remove_location_from_desc(self):
        if self.name and self.desc_name and self.desc_name.startswith(self.name):
            self.desc_name = self.desc_name[len(self.name):].strip()

    def remove_usa_state_suffix(self):
        self.name = removed_usa_state_suffix(self.name)

    def remove_number(self):
        self.number = None

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
                number = None  # to compare with flashscore where 1 is not used in name
            name = name[:-2]
        # oncourt c 2022.01.09 ставит number в конце desc, а не в конце локации
        if number is None:
            desc_name, number = tourname_number_split(desc_name)
            if number == 1:
                number = None
        tourname = TourName(name=name, desc_name=desc_name, number=number)
        # чтобы согласовать с flashscore где нет дублирования локации внутри desc_name
        # (пример: WTA 2022.01.03 location: Melbourne  desc_name: Summer Set 1)
        tourname.remove_location_from_desc()
        return tourname


def normalized_name(name):
    return '-'.join(
        (w.lower() for w in name.replace('.', '').replace('-', ' ').split() if w)
    )


def tourname_number_split(tour_name):
    m_endswith_num = _ENDSWITH_NUMBER_RE.match(tour_name)
    if m_endswith_num:
        return m_endswith_num.group("name"), int(m_endswith_num.group("number"))
    return tour_name, None


_ENDSWITH_NUMBER_RE = re.compile(
    r"(?P<name>[a-zA-Z].*) (?P<number>\d\d?)$"
)


def removed_usa_state_suffix(name):
    """ remove ', ' + redundant_suffix  (USA states mainly) """
    pos = name.find(",-")
    if pos > 0 and name[pos + 2:] in _USA_STATES:
        return name[0:pos]
    return name


_USA_STATES = (
    "oklahoma",
    "california",
    "florida",
    "south-carolina",
    "georgia",
    "illinois",
    "pennsylvania",
    "messachusetts",
    "connecticut",
    "arizona",
    "tennessee",
    "hawaii",
)
