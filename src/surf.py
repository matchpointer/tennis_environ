from typing import Optional
from enum import Enum


class Surf(Enum):
    Clay = 1
    Hard = 2
    Grass = 3
    Carpet = 4
    Acrylic = 5

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Surf):
            return self.name == other.name and self.value == other.value
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def get_code(name: str) -> Optional[int]:
    for surf in Surf:
        if surf.name == name:
            return surf.value


def get_name(code: int) -> Optional[str]:
    for surf in Surf:
        if surf.value == code:
            return surf.name
