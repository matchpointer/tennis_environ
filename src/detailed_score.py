""" database of DetailedScore records
"""
from collections import OrderedDict, namedtuple
import copy
import functools
import operator
from typing import Tuple, List, Optional

import common as co
from score import (
    Score,
    Scr,
    nil_scr,
    exist_point_increment,
    is_full_set,
    tie_opener_serve_at,
    text_from_num,
)
from side import Side


class DetailedScoreError(co.TennisScoreError):
    pass


class DetailedScoreTransError(DetailedScoreError):
    pass


class DetailedScoreTransBeforeError(DetailedScoreTransError):
    pass


__error_code_from_text = {
    "SPANING": 0x01,
    "KEY_CHAIN": 0x01 << 1,
    "KEY_CHAIN_SETSNUM": 0x01 << 2,
    "KEY_CHAIN_00": 0x01 << 3,
    "SERVER_CHAIN": 0x01 << 4,
    "KEY_EMPTY": 0x01 << 5,
    "KEY_VALUE_TIEBREAK": 0x01 << 6,
    "CREATE_REVERSE": 0x01 << 7,  # new from 2020.4.10
    "GAME_SCORE_SIMULATED": 0x01 << 8,  # new from 2020.4.10
    "CREATE_WINGAME": 0x01 << 9,
    "RAIN_INTERRUPT": 0x01 << 10,
    "GAME_ORIENT": 0x01 << 11,
    "POINT_SCORE_REVERT": 0x01 << 12,
    "POINT_SCORE_UNKNOWN": 0x01 << 13,
    "POINT_SCORE_FEW_DATA": 0x01 << 14,  # sample: '40:A' only
    "POINT_SCORE_SIMULATED": 0x01 << 15,  # new from 2020.4.20
    "SET1_SCORE_PROBLEMS": 0x01 << 16,
    "SET2_SCORE_PROBLEMS": 0x01 << 17,
    "SET3_SCORE_PROBLEMS": 0x01 << 18,
    "SET4_SCORE_PROBLEMS": 0x01 << 19,
    "SET5_SCORE_PROBLEMS": 0x01 << 20,
}


def error_code(text):
    return __error_code_from_text[text]


def error_contains(error, text):
    return error & __error_code_from_text[text]


def point_score_error(error):
    return (
        error_contains(error, "POINT_SCORE_REVERT")
        or error_contains(error, "POINT_SCORE_UNKNOWN")
        or error_contains(error, "POINT_SCORE_FEW_DATA")
    )


def error_text(error, sep=", "):
    if error == 0:
        return ""
    texts = []
    for txt, code in __error_code_from_text.items():
        # if txt in ('POINT_SCORE_SIMULATED', 'POINT_SCORE_UNKNOWN', 'POINT_SCORE_FEW_DATA'):
        #     continue
        if error_contains(error, txt):
            texts.append(txt)
    return sep.join(texts)


def left_win_game(beforekey, afterkey):
    if len(beforekey) == len(afterkey):
        diff_left = afterkey[-1][0] - beforekey[-1][0]
        diff_right = afterkey[-1][1] - beforekey[-1][1]
        if diff_left == 1 and diff_right == 0:
            return True
        elif diff_left == 0 and diff_right == 1:
            return False


def key_reversed(key):
    return tuple((tuple(reversed(pair)) for pair in key))


def key_after(item):
    """инкремент счета без учета перехода к новому сету"""
    key, value = item
    inc_pair = (1, 0) if value.left_wingame else (0, 1)
    key_lstset = key[-1]
    next_key_wait_lstset = (key_lstset[0] + inc_pair[0], key_lstset[1] + inc_pair[1])
    return key[0 : len(key) - 1] + (next_key_wait_lstset,)


class Point(object):
    """Информирует о розыгрыше (в контексте гейма).
    члены-функции информируют о состоянии ПОСЛЕ данного розыгрыша.
    члены-функции с окончанием _score (с параметром before) могут
     говорить о сост-ии ДО розыгрыша (before=True) или ПОСЛЕ него (before=False).
    Причем с точки зрения игрока означенного аргументом left.
    при left=None игрок - откр-тель гейма, иначе левый (True) или правый (False).
    """

    min_for_win_game = 4

    def __init__(
        self, left_opener, opener_win, is_simulated, opener_count, other_count
    ):
        self.left_opener = left_opener  # true if left service at game begin
        self.opener_win = opener_win  # выиграл ли данный розыгрыш подающий
        self.is_simulated = is_simulated
        # сколько розыгрышей выиграно от начала гейма и до данного розыгрыша ВКЛЮЧАЯ:
        self.opener_count = opener_count
        self.other_count = other_count
        assert (
            opener_count + other_count
        ) >= 1, "invalid Point create with zero counts"

    def _opener_host(self, left: Optional[bool] = None) -> bool:
        """вернуть ориентацию начала гейма:
        если указана сторона left (T/F), то вернем ориентацию для этой стороны,
        если не указана сторона left(None), то default ориентация откр-ля True"""
        if left is None:
            return True  # ориентировано относительно открывателя
        elif left is True:
            return self.left_opener
        else:  # left is False
            return not self.left_opener

    def _host_oppose_counts(self, left=None, before=True):
        """вернуть 2-tuple счетчиков очков хоста и его оппонента с учетом
        запрошенной стороны (left), и before - флага счета ДО РОЗЫГРЫША"""
        opener_cnt, other_cnt = self.opener_count, self.other_count
        if before:
            if self.opener_win:
                opener_cnt -= 1
            else:
                other_cnt -= 1
        if self._opener_host(left):
            return opener_cnt, other_cnt
        else:
            return other_cnt, opener_cnt

    def num_score(self, left=None, before=True):
        """вернем 2-tuple целочисленных элементов, напр. (1, 4).
        примеры завершающего гейм розыгрыша: (5, 3), (4, 0).
        если не указана сторона (left is None), то счет относительно откр-ля"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left, before)
        return host_cnt, oppose_cnt

    def text_score(self, left=None, before=True) -> Tuple[str, str]:
        """вернем 2-tuple текстовых элементов, напр. ('A', '40').
        примеры счета после завершения гейма: ('A', '0'), ('60', '40').
        если не указана сторона (left is None), то счет относительно откр-ля"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left, before)
        if max(host_cnt, oppose_cnt) < self.min_for_win_game:
            return text_from_num(host_cnt), text_from_num(oppose_cnt)
        elif host_cnt == oppose_cnt:
            return text_from_num(3), text_from_num(3)

        if host_cnt < oppose_cnt:
            min_cnt, host_min = host_cnt, True
        else:
            min_cnt, host_min = oppose_cnt, False
        if min_cnt >= 3:
            if host_min:
                return (text_from_num(3), text_from_num(3 + oppose_cnt - min_cnt))
            else:
                return (text_from_num(3 + host_cnt - min_cnt), text_from_num(3))
        return text_from_num(host_cnt), text_from_num(oppose_cnt)

    def win(self, left=None):
        """выигран ли розыгрыш"""
        return self._opener_host(left) is self.opener_win

    def loss(self, left=None):
        """проигран ли розыгрыш"""
        return not self.win(left)

    def equal_score(self, before=True):
        """ровно (30:30) или (40:40)"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left=None, before=before)
        return host_cnt == oppose_cnt and host_cnt >= (self.min_for_win_game - 2)

    def game_point_score(self, left=None, before=True):
        """есть ли геймпойнт. 0 - нет, n>0 - есть с кратностью n"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left, before)
        gamepoint_cond = host_cnt > oppose_cnt and host_cnt >= (
            self.min_for_win_game - 1
        )
        if not gamepoint_cond or (not before and self.win_game(left)):
            return 0
        return host_cnt - oppose_cnt

    def break_point_score(self, left=None, before=True):
        """есть ли брейкпойнт у оппонента. 0 - нет, n>0 - есть с кратностью n"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left, before)
        brkpoint_cond = oppose_cnt > host_cnt and oppose_cnt >= (
            self.min_for_win_game - 1
        )
        if not brkpoint_cond or (not before and self.loss_game(left)):
            return 0
        return oppose_cnt - host_cnt

    def win_game(self, left=None):
        """выигран ли гейм данным розыгрышем"""
        if self._opener_host(left):
            host_cnt, oppose_cnt = self.opener_count, self.other_count
        else:
            host_cnt, oppose_cnt = self.other_count, self.opener_count
        return host_cnt >= (oppose_cnt + 2) and host_cnt >= self.min_for_win_game

    def loss_game(self, left=None):
        """проигран ли гейм данным розыгрышем"""
        if self._opener_host(left):
            host_cnt, oppose_cnt = self.opener_count, self.other_count
        else:
            host_cnt, oppose_cnt = self.other_count, self.opener_count
        return oppose_cnt >= (host_cnt + 2) and oppose_cnt >= self.min_for_win_game

    def serve(self, left: Optional[bool] = None) -> bool:
        """кто подавал в розыгрыше"""
        return self._opener_host(left)


class TiebreakPoint(Point):
    """особенности: left_opener - открывал ли левый игрок тайбрейк;
    serve - кто подавал ВО ВРЕМЯ ЭТОГО розыгрыша"""

    min_for_win_game = 7

    def __init__(
        self, left_opener, opener_win, is_simulated, opener_count, other_count
    ):
        super(TiebreakPoint, self).__init__(
            left_opener, opener_win, is_simulated, opener_count, other_count
        )

    def text_score(self, left=None, before=True):
        """вернем 2-tuple текстовых элементов, напр. ('0', '1').
        примеры завершающего гейм розыгрыша: ('7', '4'), ('10', '8')"""
        host_cnt, oppose_cnt = self._host_oppose_counts(left, before)
        return str(host_cnt), str(oppose_cnt)

    def serve(self, left: Optional[bool] = None) -> bool:
        """кто подавал в розыгрыше"""
        # here -1 emulate score before/at this point
        return self._opener_host(left) is tie_opener_serve_at(
            (self.opener_count - 1, self.other_count)
        )

    def win_minibreak(self, left=None):
        """сделан ли минибрейк"""
        return self.win(left) and not self.serve(left)

    def loss_minibreak(self, left=None):
        """проигран ли минибрейк"""
        return not self.win(left) and self.serve(left)

    def minibreaks_diff(self, left=None):
        """выдает разницу в минибреках после этого розыгрыша.
        0 - статус кво, n>0 - на n больше, n<0 - на n меньше.
        если не указана сторона (left is None), то речь про откр-ля"""
        n_four, rem_four = divmod(self.opener_count + self.other_count - 1, 4)
        if self._opener_host(left):
            opener_srv_cnt = 1 + n_four * 2 + (1 if rem_four == 3 else 0)
            return self.opener_count - opener_srv_cnt
        else:
            notopener_srv_cnt = n_four * 2 + rem_four - (1 if rem_four == 3 else 0)
            return self.other_count - notopener_srv_cnt


class SuperTiebreakPoint(TiebreakPoint):
    min_for_win_game = 10


def tie_minibreaks_count(detailed_tiebreak):
    return sum(
        (1 for pnt in detailed_tiebreak if pnt.win_minibreak() or pnt.loss_minibreak())
    )


def tie_leadchanges_count(detailed_tiebreak):
    changes_count = 0
    left_win_cnt, right_win_cnt = 0, 0
    prev_left_lead = None
    for pnt in detailed_tiebreak:
        if pnt.win(left=True):
            left_win_cnt += 1
        else:
            right_win_cnt += 1
        if left_win_cnt == right_win_cnt:
            continue
        left_lead = left_win_cnt > right_win_cnt
        if prev_left_lead is not None and prev_left_lead is not left_lead:
            changes_count += 1
        prev_left_lead = left_lead
    return changes_count


def filter_points(det_game, point_pred, pass_point_pred=None, with_pass_point=False):
    """if pass_point_pred then firstly seek pass_point, after that seek pred points"""
    pass_point_ok = pass_point_pred is None
    for point in det_game:
        if pass_point_ok:
            if point_pred(point):
                yield point
        elif pass_point_pred(point):
            pass_point_ok = True
            if with_pass_point and point_pred(point):
                yield point


GameChar = namedtuple("GameChar", "pos neg none")


def value_to_char(value, name):
    game_char = gameCharDict[name]
    if value is None:
        return game_char.none
    bool_val = bool(value)
    if bool_val:
        return game_char.pos
    if bool_val is False:
        return game_char.neg
    raise ValueError("unexpected value '{}' name '{}'".format(value, name))


def value_from_char(char, name):
    game_char = gameCharDict[name]
    if char == game_char.pos:
        return True
    if char == game_char.neg:
        return False
    if char == game_char.none:
        return None
    raise ValueError("unexpected char '{}' name '{}'".format(char, name))


gameCharDict = {
    "left_wingame": GameChar("W", "w", "_"),
    "left_opener": GameChar("O", "o", "_"),
    "tiebreak": GameChar("T", "t", "_"),
    "supertiebreak": GameChar("S", "s", "_"),
    "error": GameChar("E", "e", "_"),
    "points": GameChar("P", "p", "p"),
}


DG_STATE_TEXT_MIN_LEN = 5


class DetailedGame(object):

    __slots__ = ("state", "points")

    def __init__(
        self,
        points,
        left_wingame=None,
        left_opener=None,
        tiebreak=None,
        error=None,
        supertiebreak=None,
        state_str=None,
    ):
        """points (строка) д.б. ориентирована для game-открывателя
        ('1' or 'W' при win, '0' or 'L' при loss)
        """
        self.points = points
        if state_str is not None:
            self.__init_state_from_str(state_str)
        else:
            self.state = 0
            if left_wingame is not None:
                self.left_wingame = left_wingame
            if left_opener is not None:
                self.left_opener = left_opener
            if tiebreak is not None:
                self.tiebreak = tiebreak
            if error is not None:
                self.error = error
            if supertiebreak:
                if not tiebreak:
                    raise co.TennisScoreError("supertiebreak must come with tiebreak")
                self.supertiebreak = supertiebreak
            if self.orientation_error():
                was_error = self.error
                self.error = was_error | error_code("GAME_ORIENT")
        self.__check()

    def __set_bit(self, bit_idx):
        self.state = self.state | (1 << bit_idx)

    def __clear_bit(self, bit_idx):
        self.state = self.state & ~(1 << bit_idx)

    def __read_bit(self, bit_idx):
        return bool(self.state & (1 << bit_idx))

    @property
    def left_wingame(self):
        return self.__read_bit(0)

    @left_wingame.setter
    def left_wingame(self, value):
        if value:
            self.__set_bit(0)
        else:
            self.__clear_bit(0)

    @property
    def left_opener(self):
        return self.__read_bit(1)

    @left_opener.setter
    def left_opener(self, value):
        if value:
            self.__set_bit(1)
        else:
            self.__clear_bit(1)

    @property
    def tiebreak(self):
        return self.__read_bit(2)

    @tiebreak.setter
    def tiebreak(self, value):
        if value:
            self.__set_bit(2)
        else:
            self.__clear_bit(2)

    @property
    def supertiebreak(self):
        if self.tiebreak:
            return self.__read_bit(3)

    @supertiebreak.setter
    def supertiebreak(self, value):
        if value:
            self.__set_bit(3)
        else:
            self.__clear_bit(3)

    @property
    def error(self):
        return self.state >> 4

    @error.setter
    def error(self, value):
        self.state = (self.state & 0b00001111) | (value << 4)

    def __eq__(self, other):
        return self.state == other.state and self.points == other.points

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        # error_text(self.error)
        return "{}{}{}".format(
            self.state_to_str(), value_to_char(self.points, "points"), self.points
        )

    def __iter__(self):
        """вернем объекты Point (или TiebreakPoint)"""
        opener_count, other_count = 0, 0
        for winsymb in self.points:
            is_simulated = winsymb in ("W", "L")
            is_win = winsymb in ("W", "1")
            if is_win:
                opener_count += 1
            else:
                other_count += 1

            if self.tiebreak:
                if self.supertiebreak:
                    point = SuperTiebreakPoint(
                        self.left_opener,
                        is_win,
                        is_simulated,
                        opener_count,
                        other_count,
                    )
                else:
                    point = TiebreakPoint(
                        self.left_opener,
                        is_win,
                        is_simulated,
                        opener_count,
                        other_count,
                    )
            else:
                point = Point(
                    self.left_opener, is_win, is_simulated, opener_count, other_count
                )
            yield point

            # было закоментировано чтобы разобраться с лишними розыгрышами, удалить их
            if point.win_game(left=True) or point.win_game(left=False):
                break

    def __len__(self):
        """число розыгрышей"""
        return len(self.points)

    def reverse_side(self):
        self.left_wingame = not self.left_wingame
        self.left_opener = not self.left_opener

    @property
    def valid(self):
        return bool(self.points) and not self.error

    @property
    def right_opener(self):
        return not self.left_opener

    @property
    def opener_wingame(self):
        return self.left_opener is self.left_wingame

    @property
    def right_wingame(self):
        return not self.left_wingame

    def final_num_score(self, left=None, before=True):
        """return 2-tuple of int.
        Если не указан left, то счет относительно открывателя"""
        points = list(iter(self))
        if points:
            return points[-1].num_score(left=left, before=before)

    def exist_text_score(self, text_score, left=None, before=True):
        for point in self:
            if point.text_score(left, before) == text_score:
                return True
        return False

    def point_text_score(self, text_score, left=None, before=True):
        for point in self:
            if point.text_score(left, before) == text_score:
                return point

    def orientation_error(self):
        """0 и 1 не ориентированы по открывателю (может исказить оценивание гейма)"""
        if (
            len(self.points) >= 2
            and not error_contains(self.error, "POINT_SCORE_FEW_DATA")
            and not error_contains(self.error, "POINT_SCORE_UNKNOWN")
            and not error_contains(self.error, "POINT_SCORE_REVERT")
        ):
            ok_orientation = self.opener_wingame is (self.points[-1] in ("1", "W"))
            return not ok_orientation
        return False

    def nocomplete_error(self, is_retired):
        """не завершен (может исказить оценивание гейма)"""
        points = [p for p in self]
        if points:
            last = points[-1]
            if (last.win_game(left=True) and self.left_wingame) or (
                last.win_game(left=False) and self.right_wingame
            ):
                return False
        elif is_retired:
            return False
        return True

    def extra_points_error(self):
        """есть лишние розыгрыши (могут исказить оценивание гейма)"""
        size = len(self)
        for (num, point) in enumerate(self, start=1):
            if num < size and (point.win_game(left=True) or point.win_game(left=False)):
                return True
        return False

    @property
    def hold(self):
        if not self.tiebreak:
            return self.left_wingame is self.left_opener

    def points_text_score(self, text_score, left=None, before=True):
        for point in self:
            if point.text_score(left, before) == text_score:
                yield point

    def state_to_str(self):
        return "{}{}{}{}{}{}".format(
            value_to_char(self.left_wingame, "left_wingame"),
            value_to_char(self.left_opener, "left_opener"),
            value_to_char(self.tiebreak, "tiebreak"),
            value_to_char(self.supertiebreak, "supertiebreak"),
            value_to_char(self.error, "error"),
            "" if not self.error else self.error,
        )

    def __init_state_from_str(self, text):
        self.state = 0
        if not text or len(text) < DG_STATE_TEXT_MIN_LEN:
            raise ValueError("unexpected state text '{}'".format(text))
        self.left_wingame = value_from_char(text[0], "left_wingame")
        self.left_opener = value_from_char(text[1], "left_opener")
        self.tiebreak = value_from_char(text[2], "tiebreak")
        supertiebreak = value_from_char(text[3], "supertiebreak")
        if supertiebreak and not self.tiebreak:
            raise co.TennisScoreError("supertiebreak must come with tiebreak")
        self.supertiebreak = supertiebreak
        is_err = value_from_char(text[4], "error")
        if is_err:
            err_text = text[DG_STATE_TEXT_MIN_LEN:]
            if not err_text:
                raise ValueError("was expected err '{}'".format(text))
            err_val = int(err_text)
            self.error = err_val

    @staticmethod
    def from_str(text):
        if not text:
            raise ValueError("unexpected detgame text '{}'".format(text))
        i, p_char = co.find_indexed_first(text, lambda c: c in ("P", "p"))
        if i < DG_STATE_TEXT_MIN_LEN:
            raise ValueError(
                "unexpected detgame len <{} text '{}'".format(
                    text, DG_STATE_TEXT_MIN_LEN
                )
            )
        if p_char == "P":
            points = text[i + 1 :]
        else:
            points = ""
        return DetailedGame(points=points, state_str=text[:i])

    def __check(self):
        assert self.left_wingame is not None, "empty left_wingame at DetailedGame"
        assert self.left_opener is not None, "empty left_opener at DetailedGame"
        assert self.tiebreak is not None, "empty tiebreak at DetailedGame"


# брейк-волна. initor=True если инициатор волны левый игрок.
# count - сколько последовательных брейков сделал initor
# back_count - сколько последовательных брейков отыграл потом в ответ соперник initorа.
# между последовательными брейками одного типа (туда ну или обратно) м.б. только
# обычные геймы когда берут свою подачу.
BreaksMove = namedtuple("BreaksMove", "initor count back_count")


class DetailedScore(OrderedDict):
    """key sample: ((6,4), (0,0) ) value: DetailedGame
    счет в key отражает состояние матча ПЕРЕД розыгрыша гейма в value.
    In database records orientation: left_id(winer), right_id(loser).
    """

    def __init__(self, items=None, retired=False):
        OrderedDict.__init__(self, items if items else [])
        self.__error = 0
        self.retired = retired

    @property
    def error(self):
        return functools.reduce(
            operator.or_, (v.error for v in self.values()), self.__error
        )

    @error.setter
    def error(self, value):
        self.__error = self.__error | value

    def __str__(self):
        ret_txt = "retired " if self.retired else ""
        if self.error:
            err_txt = "all_err: {}".format(error_text(self.error))
        else:
            err_txt = ""
        return (
            ret_txt
            + err_txt
            + "\n"
            + "\n".join([str(k) + ": " + str(v) for k, v in self.items()])
        )

    def __eq__(self, other):
        return (
            self.error == other.error
            and self.retired == other.retired
            and super(DetailedScore, self).__eq__(other)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def tostring(self):
        result = "err={}\n".format(self.error)
        result += "ret={}\n".format(int(self.retired))
        result += "len={}\n".format(len(self))
        return result + "\n".join([str(k) + "=" + str(v) for k, v in self.items()])

    @staticmethod
    def from_str(text):
        result = DetailedScore()
        size = None
        for i, line in enumerate(text.split("\n")):
            line = line.strip()
            k, v = line.split("=")
            if i == 0:
                if k != "err":
                    raise ValueError("unexpected {} (wait err=)".format(line))
                result.error = int(v)
                continue
            elif i == 1:
                if k != "ret":
                    raise ValueError("unexpected {} (wait ret=)".format(line))
                result.retired = bool(int(v))
                continue
            elif i == 2:
                if k != "len":
                    raise ValueError("unexpected {} (wait len=)".format(line))
                size = int(v)
                continue
            elif size is not None and (i - 3) <= (size - 1):
                key = eval(k)
                value = DetailedGame.from_str(v)
                result[key] = value
        return result

    def __setitem__(self, key, value):
        prev_item = self.last_item()
        error = self.__valid_item((key, value), prev_item)
        value.error = value.error | error
        OrderedDict.__setitem__(self, key, value)

    def __valid_item(self, item, prev_item=None):
        """вернем код ошибки или 0 если все ОК"""
        key, value = item
        if not bool(key):
            return error_code("KEY_EMPTY")
        if value.tiebreak and key[-1] not in ((6, 6), (12, 12), (0, 0)):
            return error_code("KEY_VALUE_TIEBREAK")
        if prev_item is not None:
            err = self.__next_key_valid(prev_item, key)
            if err:
                return err
            if (
                len(prev_item[0]) == len(item[0])
                and not prev_item[1].tiebreak
                and not item[1].tiebreak
            ):
                srv_altering = prev_item[1].left_opener is not item[1].left_opener
                if not srv_altering:
                    return error_code("SERVER_CHAIN")
            if len(prev_item[0]) == len(item[0]) and prev_item[1].tiebreak:
                return error_code("KEY_VALUE_TIEBREAK")
        return 0

    def __next_key_valid(self, item, next_key):
        """вернем код ошибки или 0 если все ОК"""
        key = item[0]
        prev_len = len(key)
        next_len = len(next_key)
        if next_len != prev_len and next_len != (prev_len + 1):
            return error_code("KEY_CHAIN_SETSNUM")
        if next_len == (prev_len + 1):
            if next_key[-1] != (0, 0):
                return error_code("KEY_CHAIN_00")  # new set must be started with 0-0
            next_key = next_key[0 : len(next_key) - 1]  # del last 0-0

        adv_score = key_after(item)
        if adv_score and next_key != adv_score:
            return error_code("KEY_CHAIN")
        return 0

    def valid_items_count(self):
        return self.count(lambda _, v: v.valid)

    def count(self, predicate):
        return sum((1 for (k, v) in self.items() if predicate(k, v)))

    def last_item(self):
        if bool(self):
            last_key = next(reversed(self))
            last_value = self[last_key]
            return last_key, last_value

    def final_score(self, setnum=None):
        if setnum is None:
            lst_item = self.last_item()
            if lst_item is not None:
                return key_after(lst_item)
        else:
            end_set_key = co.find_first(reversed(self), lambda k: len(k) == setnum)
            if end_set_key:
                return key_after((end_set_key, self[end_set_key]))
        return tuple()

    def item(self, predicate):
        for gsc, det_game in self.items():
            if predicate(gsc, det_game):
                return gsc, det_game

    def item_before(self, key):
        """вернем item предшествующий элементу со cчетом key"""
        prev_item = None
        for item in self.items():
            if item[0] == key:
                return prev_item
            prev_item = item

    def item_after(self, key):
        """вернем item после элемента со cчетом key"""
        key_reached = False
        for item in self.items():
            if key_reached:
                return item
            if item[0] == key:
                key_reached = True

    def filter_items(self, predicate):
        return ((k, v) for k, v in self.items() if predicate(k, v))

    def set_items(self, setnum):
        return self.filter_items(lambda k, _: len(k) == setnum)

    def set_valid(self, setnum, is_decided=None, decided_tie_info=None):
        atsc_dg_valid_lst = [
            (k[-1], v, v.valid) for k, v in self.items() if len(k) == setnum
        ]
        size = len(atsc_dg_valid_lst)
        if not atsc_dg_valid_lst or size < 6:
            return False
        if not all((tpl[-1] for tpl in atsc_dg_valid_lst)):
            return False  # есть невалидные геймы
        beg_atsc = atsc_dg_valid_lst[0][0]
        if beg_atsc != (0, 0):
            return False
        end_atsc = atsc_dg_valid_lst[-1][0]
        if max(end_atsc) < 5 or size < (end_atsc[0] + end_atsc[1] + 1):
            return False
        # проверим правильное окончание
        end_dg = atsc_dg_valid_lst[-1][1]
        if end_dg.left_wingame:
            end_atsc = (end_atsc[0] + 1, end_atsc[1])
        else:
            end_atsc = (end_atsc[0], end_atsc[1] + 1)
        return is_full_set(end_atsc, is_decided, decided_tie_info)

    def breaks(self, predicate):
        """вернем посл-ть: True - левый брейканул, False - правый"""
        for _, v in self.filter_items(predicate):
            if not v.tiebreak and not v.opener_wingame:
                yield v.right_opener

    def breaks_moves(self, predicate):
        """генератор эл-тов BreaksMove"""
        monoseries = []  # список моносерий (в моносерии посл-ные брейки одного игрока)
        for brk in self.breaks(predicate):
            if not monoseries:
                monoseries.append([brk])
                continue
            if monoseries[-1][0] is brk:
                monoseries[-1].append(brk)
            else:
                monoseries.append([brk])

        last_back_cnt = 0
        for idx in range(len(monoseries)):
            monoseria = monoseries[idx]
            if idx == (len(monoseries) - 1):  # last seria
                if len(monoseria) > last_back_cnt:
                    yield BreaksMove(
                        initor=monoseria[0],
                        count=len(monoseria) - last_back_cnt,
                        back_count=0,
                    )
                break
            cnt = len(monoseria) - last_back_cnt
            if cnt > 0:
                next_monoseria = monoseries[idx + 1]
                back_cnt = min(cnt, len(next_monoseria))
                yield BreaksMove(initor=monoseria[0], count=cnt, back_count=back_cnt)
                last_back_cnt = back_cnt
            else:
                # вся тек. серия ушла на обр. брейки пред. цикла и след. цикл д.б. чистым
                last_back_cnt = 0


def detailed_score_side_reversed(detail_score):
    """вернет инвертированный относительно игроков детальный счет"""
    if detail_score is None:
        return None
    result = DetailedScore()
    for key, value in detail_score.items():
        value.reverse_side()
        result[key_reversed(key)] = copy.copy(value)
    result.error = detail_score.error
    return result


def match_win_side(detail_score):
    last_item = detail_score.last_item()
    if last_item is None:
        return None
    is_last_supertie = last_item[1].supertiebreak
    fin_score = key_after(last_item)
    n_sets = len(fin_score)
    left_cnt, right_cnt = 0, 0
    for set_idx in range(n_sets):
        set_scr = fin_score[set_idx]
        if set_scr[0] > set_scr[1]:
            left_cnt += 1
        elif set_scr[0] < set_scr[1]:
            right_cnt += 1
    last_sc = fin_score[-1]
    # if is_last_supertie and n_sets in (3, 5):
    #     is_dec, dec_tie = True, sc.TieInfo(beg_scr=(0, 0), is_super=True)
    # else:
    #     is_dec, dec_tie = None,  None
    # last_full = sc.is_full_set(last_sc, is_decided=is_dec, decided_tie_info=dec_tie)
    last_full = is_full_set(last_sc)
    if left_cnt > right_cnt and last_sc[0] > last_sc[1] and last_full:
        return co.LEFT
    if left_cnt < right_cnt and last_sc[0] < last_sc[1] and last_full:
        return co.RIGHT
    if not last_full and abs(left_cnt - right_cnt) > 1:
        if left_cnt > right_cnt and last_sc[0] > last_sc[1]:
            return co.LEFT
        if left_cnt < right_cnt and last_sc[0] < last_sc[1]:
            return co.RIGHT


def is_best_of_five(detail_score):
    """only use for completed detail_score"""
    fin_score_t = detail_score.final_score()
    score = Score.from_pairs(fin_score_t, retired=detail_score.retired)
    return score.best_of_five()


def is_full(detail_score, score):
    if error_contains(detail_score.error, "GAME_SCORE_SIMULATED"):
        return False
    n_sets = score.sets_count()
    n_games = 0
    for setnum in range(1, n_sets + 1):
        set_scr = score[setnum - 1]
        n_games += sum(set_scr)
        fin_scr_t = detail_score.final_score(setnum=setnum)
        if not fin_scr_t:
            return False
        if fin_scr_t[-1] != set_scr:
            return False
    if len(detail_score) != n_games:
        return False
    return True


def find_detailed_score(session, tour, rnd, match, predicate=lambda ds: True):
    from detailed_score_dbsa import find_match_rec_by

    if hasattr(match, "detailed_score"):
        return match.detailed_score
    match_rec = find_match_rec_by(
        session, tour.ident, rnd, match.first_player.ident, match.second_player.ident
    )
    if match_rec:
        det_score = match_rec.detailed_score
        if predicate(det_score):
            setattr(match, "detailed_score", det_score)
            return det_score


def find_detailed_score2(mrecs, tour, rnd, match, predicate=lambda ds: True):
    if hasattr(match, "detailed_score"):
        return match.detailed_score
    det_score = get_detailed_score(
        mrecs, tour.ident, rnd, match.first_player.ident, match.second_player.ident
    )
    if det_score is not None:
        if predicate(det_score):
            setattr(match, "detailed_score", det_score)
            return det_score


def get_detailed_score(mrecs, tour_id, rnd, left_id, right_id):
    match_rec = co.find_first(
        mrecs,
        (
            lambda r: r.tour_id == tour_id
            and r.rnd == rnd
            and r.left_id == left_id
            and r.right_id == right_id
        ),
    )
    if match_rec:
        return match_rec.detailed_score


class SetItems:
    Items = List[Tuple[Scr, DetailedGame]]  # here 'at' Scr (before game)

    def __init__(self, items: Items):
        self.items: SetItems.Items = items
        if items:
            self.beg_scr: Scr = self.items[0][0]
            last_scr, last_dg = items[-1]
            self.fin_scr: Scr = (
                (last_scr[0] + 1, last_scr[1])
                if last_dg.left_wingame
                else (last_scr[0], last_scr[1] + 1)
            )
            self.tie_scr: Scr = last_dg.final_num_score(left=True, before=False)
        else:
            self.beg_scr: Scr = nil_scr
            self.fin_scr: Scr = nil_scr
            self.tie_scr: Scr = nil_scr
        self._ok_set = self.ok_set()

    @staticmethod
    def from_scores(setnum: int, detailed_score: DetailedScore, score: Score):
        items: SetItems.Items = [
            (k[setnum - 1], v) for (k, v) in detailed_score.set_items(setnum)
        ]
        obj = SetItems(items)
        obj.fin_scr = score[setnum - 1] if 1 <= setnum <= len(score) else nil_scr
        obj.init_tie_scr_from_score(setnum, score)
        obj._ok_set = obj.ok_set()
        return obj

    def init_tie_scr_from_score(self, setnum: int, score: Score):
        if not (1 <= setnum <= len(score)):
            self.tie_scr = nil_scr
            return
        tie_min = score.tie_loser_result(setnum)
        if tie_min is None:
            self.tie_scr = nil_scr
            return
        min_win = 7
        if (setnum == score.decsupertie_setnum and score.decsupertie) or (
            setnum in (3, 5)
            and setnum == len(score)
            and self.items
            and self.items[-1][1].supertiebreak
        ):
            min_win = 10
        win_left = self.fin_scr[0] >= self.fin_scr[1]
        if win_left:
            tie_scr = max(min_win, tie_min + 2), tie_min
        else:
            tie_scr = tie_min, max(min_win, tie_min + 2)
        self.tie_scr = tie_scr

    def __bool__(self):
        return bool(self.items)

    __nonzero__ = __bool__

    def flip(self):
        fliped_items = []
        for ((x, y), dg) in self.items:
            dgc = copy.copy(dg)
            dgc.reverse_side()
            fliped_items.append(((y, x), dgc))
        self.items = fliped_items
        self.beg_scr = (self.beg_scr[1], self.beg_scr[0])
        self.fin_scr = (self.fin_scr[1], self.fin_scr[0])

    def set_winner_side(self) -> Optional[Side]:
        if self.fin_scr[0] > self.fin_scr[1]:
            return co.LEFT
        elif self.fin_scr[0] < self.fin_scr[1]:
            return co.RIGHT
        return None

    def set_opener_side(self):
        if self.items:
            beg_scr, beg_dg = self.items[0]
            if sum(beg_scr) % 2 == 0:
                return co.side(beg_dg.left_opener)
            else:
                return co.side(not beg_dg.left_opener)

    def exist_scr(self, scr: Scr, left_opener: bool):
        """:return None if unknown"""
        same_scr_it = co.find_first(self.items, lambda i: scr == i[0])
        if self._ok_set:
            if same_scr_it is None:
                return False
            return same_scr_it[1].left_opener == left_opener

        if same_scr_it is not None:
            if same_scr_it[1].left_opener != left_opener:
                return False
            is_ok = self.is_admit_error(same_scr_it[1].error)
            return True if is_ok else None

        same_sum_it = co.find_first(self.items, lambda i: sum(scr) == sum(i[0]))
        if same_sum_it is not None:
            return False
        if self.ok_last_scr():
            last_scr = self.items[-1][0]
            if last_scr[0] < scr[0] or last_scr[1] < scr[1]:
                return False  # scr is not reachable

    def ok_set(self, strict=True) -> bool:
        if not self.items or self.beg_scr != (0, 0) or not self.ok_last_scr():
            return False
        prev_scr = (0, 0)
        idx = 1
        while idx < len(self.items):
            scr, dg = self.items[idx]
            if strict and not self.is_admit_error(dg.error):
                return False
            if not exist_point_increment(prev_scr, scr):
                return False
            prev_scr = scr
            idx += 1
        return True

    def ok_last_scr(self):
        if self.items:
            last_scr = self.items[-1][0]
            return exist_point_increment(last_scr, self.fin_scr)

    def tostring(self, full=True):
        if full:
            return " ".join((f"{k}{str(v)}" for k, v in self.items))
        return " ".join((f"{k}{str(v)[:5]}" for k, v in self.items))

    def __str__(self):
        return self.tostring(full=False)

    def __getitem__(self, idx):
        if not self.ok_idx(idx):
            raise co.TennisUnknownScoreError(
                f"SetItems not okidx {idx} beg:{self.beg_scr} fin:{self.fin_scr} "
                f"okset:{self._ok_set} len:{len(self.items)}"
            )
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def ok_idx(self, idx) -> bool:
        if self.beg_scr == (0, 0) and len(self.items) > idx == sum(self.items[idx][0]):
            return all(
                [self.is_admit_error(it[1].error) for it in self.items[: idx + 1]]
            )
        return False

    def exist_equal_after(self, idx: int, maxlen: int):
        """:return None if unknown"""
        prev_scr = nil_scr
        n = 0
        while idx < len(self.items) and n < maxlen:
            item = self.items[idx]
            if error_contains(item[1].error, "GAME_SCORE_SIMULATED"):
                return None
            scr = item[0]
            if prev_scr != nil_scr and not exist_point_increment(prev_scr, scr):
                return None  # may be we have a gap
            if scr[0] == scr[1]:
                return True
            prev_scr = scr
            idx += 1
            n += 1
        if n > 0 and self.ok_last_scr():
            return False
        return None  # may be we are cut at end

    @staticmethod
    def is_admit_error(error):
        return (
            not error_contains(error, "GAME_SCORE_SIMULATED")
            and not error_contains(error, "SERVER_CHAIN")
        )
