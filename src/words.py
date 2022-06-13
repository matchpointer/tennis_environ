# -*- coding=utf-8 -*-
import common as co


class SyllabledName:
    synonym_prefixes = ("saint", "sant", "st")

    def __init__(self, name):
        self.name = co.to_ascii(name)
        self.parts = []  # lower case syllables
        self.syllabled = ""  # lower case syllables departed by '-'
        self._init_parts()

    def add_part(self, part):
        part_low = part.lower()
        self.parts.append(part_low)
        if self.syllabled:
            self.syllabled += "-" + part_low
        else:
            self.syllabled = part_low

    def _init_parts(self):
        if self.name:
            name = self.name.replace(".", "").replace(",", "").lower()
            # here remain problem wta 2013, 2014 chal clay tour: 'St.Petersburg'

            split_idx_list = [
                i for i in range(len(name)) if name[i] in (" ", "-", "'", "`")
            ]
            if not split_idx_list and name:
                self.parts.append(name)
                self.syllabled = name
            elif split_idx_list:
                beg_idx = 0
                for split_idx in split_idx_list:
                    name_part = name[beg_idx:split_idx]
                    if name_part:
                        self.parts.append(name_part)
                    beg_idx = split_idx + 1
                name_part = name[beg_idx:]
                if name_part:
                    self.parts.append(name_part)
                self.syllabled = "-".join(self.parts)

    def resemble(self, other):
        if len(self.parts) != len(other.parts):
            return False
        for i in range(len(self.parts)):
            if (
                i == 0
                and self.parts[i] in self.synonym_prefixes
                and other.parts[i] in self.synonym_prefixes
            ):
                continue
            if not resemble_words(
                self.parts[i], other.parts[i], max_errors=1, ignore_case=False
            ):
                return False
        return True

    def __eq__(self, other):
        if isinstance(other, SyllabledName):
            if self.syllabled == other.syllabled:
                return True
            return self.resemble(other)
        elif isinstance(other, str):
            if self.name == other:
                return True
            return resemble_words(self.name, other, max_errors=1, ignore_case=False)
        raise co.TennisError(
            "bad compare {} and {}".format(self.__class__.__name__, type(other))
        )

    def __contains__(self, item):
        return item in self.name

    def __str__(self):
        return self.syllabled

    def __hash__(self):
        return hash(self.syllabled)

    def __bool__(self):
        return bool(self.name)

    __nonzero__ = __bool__


def resemble_words_err2(first_word, second_word, ignore_case=False):
    size1st = len(first_word)
    size2nd = len(second_word)
    if abs(size1st - size2nd) > 2:
        return False
    max_errors = 1
    if max(size1st, size2nd) <= 3:
        max_errors = 0
    if max(size1st, size2nd) >= 9:
        max_errors = 2
    return resemble_words(
        first_word, second_word, max_errors=max_errors, ignore_case=ignore_case
    )


def resemble_words(first_word, second_word, max_errors=0, ignore_case=False):
    word1st = first_word.lower() if ignore_case else first_word
    word2nd = second_word.lower() if ignore_case else second_word
    size1st = len(word1st)
    size2nd = len(word2nd)
    if min(size1st, size2nd) <= 5:
        return word1st == word2nd
    if size1st == size2nd:
        err_count = errors_count(word1st, word2nd)
        return err_count <= max_errors
    elif size1st == (size2nd + 1) and max_errors >= 1:
        err_count = errors_count_after_symb_del(word1st, word2nd)
        return err_count <= (max_errors - 1)
    elif size2nd == (size1st + 1) and max_errors >= 1:
        err_count = errors_count_after_symb_del(word2nd, word1st)
        return err_count <= (max_errors - 1)
    else:
        return False


def errors_count(word1st, word2nd):
    err_count = 0
    for i in range(len(word1st)):
        if word1st[i] != word2nd[i]:
            err_count += 1
    return err_count


def errors_count_after_symb_del(word1st, word2nd):
    """ " must be: len(ward1st) >= len(ward2nd)"""
    size1st = len(word1st)
    err_count_min = size1st
    for i in range(size1st):
        err_count = errors_count(word1st[0:i] + word1st[i + 1:], word2nd)
        if err_count < err_count_min:
            err_count_min = err_count
        if err_count_min == 0:
            return err_count_min
    return err_count_min
