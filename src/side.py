
class Side:
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ANY = "ANY"

    def __init__(self, value):
        if value in (True, Side.LEFT):
            self.value = Side.LEFT
        elif value in (False, Side.RIGHT):
            self.value = Side.RIGHT
        elif value == Side.ANY:
            self.value = Side.ANY
        else:
            raise TypeError("unwaited side({})".format(value))

    def is_left(self):
        return self.value == Side.LEFT

    def is_right(self):
        return self.value == Side.RIGHT

    def is_any(self):
        return self.value == Side.ANY

    def is_oppose(self, other):
        if other is None:
            return False
        return (self.value == Side.LEFT and other.value == Side.RIGHT) or (
            self.value == Side.RIGHT and other.value == Side.LEFT
        )

    def fliped(self):
        if self.value == Side.LEFT:
            return Side(Side.RIGHT)
        elif self.value == Side.RIGHT:
            return Side(Side.LEFT)
        return Side(Side.ANY)

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.value}')"

    def __eq__(self, other):
        if other is None or other is True or other is False:
            return False
        if isinstance(other, str):
            return self.value == other
        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def abbr_text(self):
        return self.value[0]
