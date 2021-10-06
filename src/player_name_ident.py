import re

import common as co
import log
import oncourt_players
import words


def identify_player(company_name, sex, player_short_name, cou=None):
    if company_name == "FS":
        company_key = "flashscore"
    else:
        raise co.TennisError("unexpected company_name '{}'".format(company_name))
    if cou is None:
        player = co.find_first(
            oncourt_players.players(sex),
            lambda p: p.disp_name(company_key) == player_short_name,
        )
    else:
        player = co.find_first(
            oncourt_players.players(sex),
            lambda p: p.cou == cou and p.disp_name(company_key) == player_short_name,
        )
    if player is not None:
        return player
    abbrname = AbbrName(player_short_name)
    return abbrname.find_player(oncourt_players.players(sex), sex)


class AbbrName(object):
    """Исходя из сокр. имени с помощью рег. выр-ния ищет полное имя игрока.
    Основано на разборе до 5 частей: last1 last2 last3 init1 init2
    пока не можем например следующее (тут и last4):
    'Neffa De Los Rios A.P.'  --> fullname 'Ana Paula Neffa De Los Rios'
     another problem sample: full name "Jia-Jing Lu" flashscore displays as "Lu J."
            uses regex() 'J[a-z]+ Lu\\b' and miss found player "Jiaxi Lu"
    """

    bc_abbr_name_re = re.compile(
        r"(?P<last1>(de|di|da|van|der|[A-Z]([a-z]|')+[-A-Za-z]*)\b)"
        + r"(?P<last2> (de|di|da|van|der|[A-Z][a-z]+[-A-Za-z]*)\b)?"
        + r"(?P<last3> [A-Z][a-z]+[-A-Za-z]*\b)?"
        + r"(?P<init1> [A-Z][a-z]?[a-z]?[a-z]?\.)?"
        + r"(( |-|- )?(?P<init2>[A-Z]\.?))?$"
    )

    # lastname prefixes samples:
    #  'Van De Graaf G.'  --> 'Gabriela Van De Graaf'
    #  'Van De Velde B.'  --> 'Bernice Van De Velde'
    #  'Van Riet L.'      --> 'Lisanne Van Riet'
    last_prefixes = ("De", "de", "Di", "di", "Van", "van", "Der", "der")

    between_part_sep = " "

    def __init__(self, abbr):
        self.dispname = abbr
        self.abbr = abbr
        self.last1 = None
        self.last2 = None
        self.last3 = None
        self.init1 = None
        self.init2 = None
        self.__first_correction()
        if self.abbr is not None:
            self.__restore_missing_point()
            self.__make_parts()

    def last_parts(self):
        result = []
        if self.last1:
            result.append(self.last1.replace("-", AbbrName.between_part_sep))
        if self.last2:
            result.append(self.last2.replace("-", AbbrName.between_part_sep))
        if self.last3:
            result.append(self.last3.replace("-", AbbrName.between_part_sep))
        return result

    def init_parts(self):
        result = []
        if self.init1:
            result.append(self.init1)
        if self.init2:
            result.append(self.init2)
        return result

    def resemble(self, playername):
        def name_lexems(name):
            return [i.strip() for i in name.replace("-", " ").split(" ") if i.strip()]

        def two_last_name_lexems(name):
            results = []
            lexems = name_lexems(name)
            if len(lexems) >= 2:
                results.append(lexems[len(lexems) - 2])
            if len(lexems) >= 1:
                results.append(lexems[len(lexems) - 1])
            return results

        player_lexems = two_last_name_lexems(playername.lower())
        ok_length = 0
        last_part_lexems = sum([lp.split(" ") for lp in self.last_parts()], [])
        for last_part in last_part_lexems:
            last_part_low = last_part.lower()
            for plr_lexem in player_lexems:
                if (
                    last_part_low not in AbbrName.last_prefixes
                    and words.resemble_words_err2(last_part_low, plr_lexem)
                ):
                    ok_length += len(last_part_low)
        return ok_length >= 4 or (
            ok_length == 3 and 3 == max([len(i) for i in last_part_lexems])
        )

    def find_player(self, players, sex=None):
        def re_find_player_impl(players, name_exp, flags=0, openend=True):
            name_re = re.compile(name_exp if openend else name_exp + "$", flags)
            return [
                p
                for p in players
                if name_re.match(p.name.replace("-", AbbrName.between_part_sep))
            ]

        name_exp = self.regex()
        if not name_exp:
            return None
        found_players = re_find_player_impl(players, name_exp)
        found_count = len(found_players)
        if found_count == 1:
            return found_players[0]
        elif found_count > 1:
            found_players = re_find_player_impl(players, name_exp, openend=False)
            if len(found_players) == 1:
                return found_players[0]
        elif found_count == 0:
            found_players = re_find_player_impl(players, name_exp, flags=re.IGNORECASE)
            size = len(found_players)
            if size == 1:
                return found_players[0]
            elif size > 1:
                found_players = re_find_player_impl(
                    players, name_exp, flags=re.IGNORECASE, openend=False
                )
                if len(found_players) == 1:
                    return found_players[0]
        found_txt = "NOT FOUND"
        if len(found_players) > 1:
            found_txt = "AMBIGIOUS " + "\n\t".join([p.name for p in found_players])
        log.warn(
            "fail {} get name by abbr '{}' cause: {}".format(
                "" if sex is None else sex, self.abbr, found_txt
            )
        )

    def regex(self):
        last_parts = self.last_parts()
        init_parts = self.init_parts()
        if not last_parts:
            return None
        name_exp = ""
        for init_part in init_parts:
            name_exp += init_part + "[a-z]+" + AbbrName.between_part_sep

        if len(last_parts) > 1:
            if init_parts:
                name_exp += AbbrName.between_part_sep.join(last_parts)
            else:
                # parts coming after first last_part are init_parts (more probably)
                name_exp = (
                    AbbrName.between_part_sep.join(last_parts[1:])
                    + AbbrName.between_part_sep
                    + last_parts[0]
                )
        else:
            name_exp += last_parts[0]
        return name_exp + "\\b"

    def __first_correction(self):
        self.abbr = co.cyrillic_misprint_to_latin(self.abbr)
        self.abbr = self.abbr.replace("  ", " ")
        self.abbr = self.abbr.replace(". -", ".-")
        self.abbr = self.abbr.replace(".- ", ".-")

    def __make_parts(self):
        m = self.bc_abbr_name_re.match(self.abbr)
        if m:
            self.last1 = m.group("last1")

            self.last2 = m.group("last2")
            if self.last2 is not None:
                self.last2 = self.last2.strip()

            self.last3 = m.group("last3")
            if self.last3 is not None:
                self.last3 = self.last3.strip()

            self.init1 = m.group("init1")
            if self.init1 is not None:
                self.init1 = self.init1.strip(" .")

            self.init2 = m.group("init2")
            if self.init2 is not None:
                self.init2 = self.init2.strip(" .")

    two_inits_name_re = re.compile(
        r"(?P<last_name>[A-Z][a-z]+) (?P<first_init>[A-Z])\.?"
        + r"(?P<sep>[- ])(?P<second_init>[A-Z])\.?$"
    )

    def __restore_missing_point(self):
        """betcity typical error: 'Struff J-L.' (corrected: 'Struff J.-L.')"""
        match = self.two_inits_name_re.match(self.abbr)
        if match:
            self.abbr = "{} {}.{}{}.".format(
                match.group("last_name"),
                match.group("first_init"),
                match.group("sep"),
                match.group("second_init"),
            )
