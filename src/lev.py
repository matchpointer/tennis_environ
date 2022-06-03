"""
level of tournament
"""
from typing import Dict

junior = "junior"
future = "future"
chal = "chal"
main = "main"
masters = "masters"
gs = "gs"

team = "team"
teamworld = "teamworld"


def soft_level(level, rnd, qualification=None) -> str:
    """
    дает 'усредненный' уровень. Может вернуть 'qual' не входящее в базовые уровни.
    """
    qual = qualification
    if qual is None and rnd is not None:
        qual = rnd.qualification() or rnd.pre_qualification()

    if qual and level not in (masters, chal, future):
        result = 'qual'
    elif level in (masters, gs, teamworld):
        result = main
    else:
        result = str(level)
    return result


_code_level: Dict[int, str] = {
    0: future, 1: "qual", 2: chal, 3: main, 4: team, 5: teamworld
}


def level_code(level) -> int:
    for code, lev_name in _code_level.items():
        if level == lev_name:
            return code
    raise ValueError(f"unexpected level: {level}")


_rawlevel_code: Dict[str, int] = {chal: -1, main: 1, masters: 2, gs: 3}


def raw_level_code(rnd_text: str, raw_level_text: str) -> int:
    if 'team' in raw_level_text:
        raise ValueError(f"bad in raw_level_text {raw_level_text} {rnd_text}")
    if rnd_text in ('q-First', 'q-Second', 'Qualifying'):
        return 0
    return _rawlevel_code[raw_level_text]



