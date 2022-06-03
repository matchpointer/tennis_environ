from typing import Optional

from common import to_ascii

Clay = 'Clay'
Hard = 'Hard'
Grass = 'Grass'
Carpet = 'Carpet'
Acrylic = 'Acrylic'


def make_surf(name: str) -> str:
    name_low = to_ascii(name.strip()).lower()
    if (
        ("carpet" in name_low and "outdoor" not in name_low)
        or
        ("hard" in name_low and ("indoor" in name_low or name_low.startswith("i.")))
    ):
        return Carpet
    elif "grass" in name_low:
        return Grass
    elif "clay" in name_low:
        return Clay
    elif "hard" in name_low or ("carpet" in name_low and "outdoor" in name_low):
        return Hard
    elif "acrylic" in name_low:
        return Acrylic
    else:
        raise TypeError("unexpected surface '{}'".format(name))


_codes = {
    Clay: 1,
    Hard: 2,  # outdoor
    Grass: 3,
    Carpet: 4,  # indoor
    Acrylic: 5,
}


def get_code(name: str) -> Optional[int]:
    return _codes.get(name)


def get_name(code: int) -> Optional[str]:
    for srf, value in _codes.items():
        if value == code:
            return srf
    return None


_abbrev = {
    Clay: 'C',
    Hard: 'H',
    Grass: 'G',
    Carpet: 'I',
    Acrylic: 'A',
}


def get_abbr_name(name: str) -> Optional[str]:
    return _abbrev.get(name)
