from side import Side
import common as co
import score as sc


class Pair(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dif_pair(self, other, unit):
        res_x = (self.x - other.x) // unit
        res_y = (self.y - other.y) // unit
        return Pair(res_x, res_y)

    def incr_decr(self):
        return Pair(self.x + 1, self.y - 1)

    def decr_incr(self):
        return Pair(self.x - 1, self.y + 1)

    def is_incr_to(self, other):
        if other is not None:
            return sc.exist_point_increment((self.x, self.y), other)

    def is_negative(self):
        return self.x < 0 or self.y < 0

    @staticmethod
    def from_text(text):
        pair = text.split("-")
        return Pair(int(pair[0]), int(pair[1]))

    def __add__(self, other):
        return Pair(self.x + other[0], self.y + other[1])

    def __iadd__(self, other):
        self.x += other[0]
        self.y += other[1]
        return self

    def __str__(self):
        return "{}-{}".format(self.x, self.y)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.x, self.y)

    def __iter__(self):
        return iter((self.x, self.y))

    def __len__(self):
        return 2

    def __eq__(self, other):
        return (self.x, self.y) == (other[0], other[1]) and len(other) == 2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return (self.x, self.y) < (other[0], other[1])

    def __le__(self, other):
        return (self.x, self.y) <= (other[0], other[1])

    def __gt__(self, other):
        return (self.x, self.y) > (other[0], other[1])

    def __ge__(self, other):
        return (self.x, self.y) >= (other[0], other[1])

    def __getitem__(self, index):
        if index in (0, -2):
            return self.x
        elif index in (1, -1):
            return self.y
        else:
            raise co.TennisError("unexpected Pair index {}".format(index))

    def __setitem__(self, index, value):
        if index in (0, -2):
            self.x = value
        elif index in (1, -1):
            self.y = value
        else:
            raise co.TennisError("unexpected Pair index {}".format(index))

    def flip(self):
        self.x, self.y = self.y, self.x


class SrvPair(Pair):
    def __init__(self, srv_side: Side, x, y):
        super(SrvPair, self).__init__(x, y)
        self.srv_side = srv_side

    @staticmethod
    def from_text(text_item, strict=True):
        pair = (
            text_item.strip()
            .replace("[", "")
            .replace("]", "")
            .replace("40", "45")
            .replace("AD", "60")
            .replace("A", "60")
            .split("-")
        )
        if len(pair) < 2:
            raise ValueError("no 2-column scr '{}'".format(text_item))
        if "*" in pair[0]:
            srv_side = co.LEFT
        elif "*" in pair[1]:
            srv_side = co.RIGHT
        elif not strict:
            srv_side = None
        else:
            raise co.TennisScoreError("absent * in '{}'".format(text_item))
        return SrvPair(
            srv_side, int(pair[0].replace("*", "")), int(pair[1].replace("*", ""))
        )

    def __str__(self):
        if self.srv_side == co.LEFT:
            result = "*{}-{}".format(self.x, self.y)
        elif self.srv_side == co.RIGHT:
            result = "{}-{}*".format(self.x, self.y)
        else:
            result = super(SrvPair, self).__str__()
        return result

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            "LEFT" if self.srv_side == co.LEFT else "RIGHT",
            self.x,
            self.y,
        )

    def __eq__(self, other):
        if type(other) in (Pair, tuple):
            return super(SrvPair, self).__eq__(other)
        return self.srv_side == other.srv_side and super(SrvPair, self).__eq__(other)

    def flip(self):
        super(SrvPair, self).flip()
        self.srv_side = self.srv_side.fliped()

    def is_local_left_opener_tie_orientation(self):
        """True if x:y correspond for x is tie opener side
        Attention! for x scores is may be RIGHT player in match
        """
        return sc.get_tie_open_side((self.x, self.y), self.srv_side) == co.LEFT

