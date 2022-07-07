import sys
import os
from collections import defaultdict
from operator import sub
import argparse
import tkinter
import tkinter.ttk
import tkinter.constants as tkc

import common as co
from loguru import logger as log
from surf import make_surf
import score as sc
import tennis
import cfg_dir
from oncourt import dbcon, extplayers
import stat_cont as st
import dict_tools
import matchstat
import decided_set
import set2_after_set1loss_stat

import ratings


def get_size(left_sized, right_sized):
    size = 0
    if left_sized is not None:
        size += left_sized.size
    if right_sized is not None:
        size += right_sized.size
    return size


def draw_size(size_widget, left_sized, right_sized):
    size_widget["text"] = "({})".format(get_size(left_sized, right_sized))


def draw_value(widget, sizedvalue):
    value = sizedvalue.value if sizedvalue is not None else None
    widget["text"] = "{0:.2f}".format(value) if value is not None else "-"


def draw_left_right_chances(
    left_widget, right_widget, left_sizedvalue, right_sizedvalue
):
    left_prob, right_prob = co.twoside_values(left_sizedvalue, right_sizedvalue)
    left_widget["text"] = "{0:.2f}".format(left_prob) if left_prob is not None else "-"
    right_widget["text"] = (
        "{0:.2f}".format(right_prob) if right_prob is not None else "-"
    )


def draw_oneside_chances(
    first_widget, second_widget, first_sizedvalue, second_sizedvalue
):
    first_prob, second_prob = co.oneside_values(first_sizedvalue, second_sizedvalue)
    first_widget["text"] = (
        "{0:.2f}".format(first_prob) if first_prob is not None else "-"
    )
    second_widget["text"] = (
        "{0:.2f}".format(second_prob) if second_prob is not None else "-"
    )


class MoneyBallancePage(tkinter.ttk.Frame):
    def __init__(self, parent):
        tkinter.ttk.Frame.__init__(self, parent)

        row = 1
        self.estimate_btn = tkinter.ttk.Button(
            self, text="Estimate", command=self.estimate
        )
        self.estimate_btn.grid(row=row, column=0)
        row += 1

        self.coef_lbl = tkinter.ttk.Label(self, text="coef:")
        self.coef_lbl.grid(row=row, column=1)
        self.lhs_coef_var = tkinter.StringVar()
        self.lhs_coef_var.set("0.0")
        self.lhs_coef_entry = tkinter.ttk.Entry(self, textvariable=self.lhs_coef_var)
        self.lhs_coef_entry.config(width=5)
        self.lhs_coef_entry.grid(row=row, column=3)

        self.rhs_coef_var = tkinter.StringVar()
        self.rhs_coef_var.set("0.0")
        self.rhs_coef_entry = tkinter.ttk.Entry(self, textvariable=self.rhs_coef_var)
        self.rhs_coef_entry.config(width=5)
        self.rhs_coef_entry.grid(row=row, column=5)
        row += 1

        self.sum_lbl = tkinter.ttk.Label(self, text="sum:")
        self.sum_lbl.grid(row=row, column=1)
        self.lhs_sum_var = tkinter.StringVar()
        self.lhs_sum_var.set("0.0")
        self.lhs_sum_entry = tkinter.ttk.Entry(self, textvariable=self.lhs_sum_var)
        self.lhs_sum_entry.config(width=5)
        self.lhs_sum_entry.grid(row=row, column=3)

        self.rhs_sum_var = tkinter.StringVar()
        self.rhs_sum_var.set("0.0")
        self.rhs_sum_entry = tkinter.ttk.Entry(self, textvariable=self.rhs_sum_var)
        self.rhs_sum_entry.config(width=5)
        self.rhs_sum_entry.grid(row=row, column=5)
        row += 2

        # for output:
        self.profit_lbl = tkinter.ttk.Label(self, text="$W*")
        self.profit_lbl.grid(row=row, column=4)
        row += 1
        self.any_lbl = tkinter.ttk.Label(self, text="any orient:")
        self.any_lbl.grid(row=row, column=0)

        self.any_orient_sum_txt = tkinter.ttk.Label(self, text="")
        self.any_orient_sum_txt.grid(row=row, column=1)
        self.any_orient_profit_txt = tkinter.ttk.Label(self, text="")
        self.any_orient_profit_txt.grid(row=row, column=4)
        row += 1

        self.profit_if_lhs_lbl = tkinter.ttk.Label(self, text="$WL")
        self.profit_if_lhs_lbl.grid(row=row, column=3)
        self.profit_if_rhs_lbl = tkinter.ttk.Label(self, text="$WR")
        self.profit_if_rhs_lbl.grid(row=row, column=5)
        row += 1

        self.rhs_zero_lbl = tkinter.ttk.Label(self, text="right zero:")
        self.rhs_zero_lbl.grid(row=row, column=0)
        self.rhs_zero_sum_txt = tkinter.ttk.Label(self, text="")
        self.rhs_zero_sum_txt.grid(row=row, column=1)
        self.rhs_zero_profit_if_left_txt = tkinter.ttk.Label(self, text="")
        self.rhs_zero_profit_if_left_txt.grid(row=row, column=3)
        self.rhs_zero_profit_if_right_txt = tkinter.ttk.Label(self, text="")
        self.rhs_zero_profit_if_right_txt.grid(row=row, column=5)
        row += 1

        self.lhs_zero_lbl = tkinter.ttk.Label(self, text="left zero:")
        self.lhs_zero_lbl.grid(row=row, column=0)
        self.lhs_zero_sum_txt = tkinter.ttk.Label(self, text="")
        self.lhs_zero_sum_txt.grid(row=row, column=1)
        self.lhs_zero_profit_if_left_txt = tkinter.ttk.Label(self, text="")
        self.lhs_zero_profit_if_left_txt.grid(row=row, column=3)
        self.lhs_zero_profit_if_right_txt = tkinter.ttk.Label(self, text="")
        self.lhs_zero_profit_if_right_txt.grid(row=row, column=5)
        row += 1

    def left_coef(self):
        return float(self.lhs_coef_var.get())

    def right_coef(self):
        return float(self.rhs_coef_var.get())

    def left_sum(self):
        return float(self.lhs_sum_var.get())

    def right_sum(self):
        return float(self.rhs_sum_var.get())

    def data_inputed(self):
        return (
            self.left_coef() > 0.0
            and self.right_coef() > 0.0
            and (
                (self.left_sum() > 0.0 and self.right_sum() == 0.0)
                or (self.left_sum() == 0.0 and self.right_sum() > 0.0)
            )
        )

    def estimate_right_zero(self):
        """справа мы в нуле, слева - что получится"""
        if self.right_sum() > 0:
            money = (self.right_coef() - 1.0) * self.right_sum()  # all right profit
            profit_if_left = (self.left_coef() - 1.0) * money - self.right_sum()
            profit_if_right = (self.right_coef() - 1.0) * self.right_sum() - money
        else:
            money = self.left_sum() / (self.right_coef() - 1.0)
            profit_if_left = (self.left_coef() - 1.0) * self.left_sum() - money
            profit_if_right = (self.right_coef() - 1.0) * money - self.left_sum()
        self.rhs_zero_sum_txt["text"] = str(round(money, 1))
        self.rhs_zero_profit_if_left_txt["text"] = str(round(profit_if_left, 1))
        self.rhs_zero_profit_if_right_txt["text"] = str(round(profit_if_right, 1))

    def estimate_left_zero(self):
        """слева мы в нуле, справа - что получится"""
        if self.left_sum() > 0:
            money = (self.left_coef() - 1.0) * self.left_sum()  # all left profit
            profit_if_right = (self.right_coef() - 1.0) * money - self.left_sum()
            profit_if_left = (self.left_coef() - 1.0) * self.left_sum() - money
        else:
            money = self.right_sum() / (self.left_coef() - 1.0)
            profit_if_right = (self.right_coef() - 1.0) * self.right_sum() - money
            profit_if_left = (self.left_coef() - 1.0) * money - self.right_sum()
        self.lhs_zero_sum_txt["text"] = str(round(money, 1))
        self.lhs_zero_profit_if_left_txt["text"] = str(round(profit_if_left, 1))
        self.lhs_zero_profit_if_right_txt["text"] = str(round(profit_if_right, 1))

    def estimate_equal(self):
        """ждем одинаковую денежную отдачу независимо от исхода"""
        if self.left_sum() > 0:
            money = (self.left_coef() * self.left_sum()) / self.right_coef()
            profit = (self.left_coef() - 1.0) * self.left_sum() - money
        else:
            money = (self.right_coef() * self.right_sum()) / self.left_coef()
            profit = (self.right_coef() - 1.0) * self.right_sum() - money
        self.any_orient_sum_txt["text"] = str(round(money, 1))
        self.any_orient_profit_txt["text"] = str(round(profit, 1))

    def clear_indicators(self):
        self.any_orient_sum_txt["text"] = ""
        self.any_orient_profit_txt["text"] = ""
        self.rhs_zero_sum_txt["text"] = ""
        self.rhs_zero_profit_if_left_txt["text"] = ""
        self.rhs_zero_profit_if_right_txt["text"] = ""
        self.lhs_zero_sum_txt["text"] = ""
        self.lhs_zero_profit_if_left_txt["text"] = ""
        self.lhs_zero_profit_if_right_txt["text"] = ""

    def estimate(self):
        self.clear_indicators()
        if self.data_inputed():
            self.estimate_left_zero()
            self.estimate_right_zero()
            self.estimate_equal()


class MatchstatPageLGR(tkinter.ttk.Frame):
    def __init__(self, parent, application, short, fun_name):
        tkinter.ttk.Frame.__init__(self, parent)
        self.application = application
        self.fun_name = fun_name
        self.short = short
        self.page_builder = matchstat.PageBuilderLGR(
            self.application.sex(),
            self.application.level(),
            self.application.surface(),
            self.application.rnd(),
            short,
            self.application.left_player(),
            self.application.right_player(),
            fun_name,
        )
        self.page_builder.create_widgets(self)

    def estimate(self):
        self.page_builder.set_data(
            self.application.sex(),
            self.application.level(),
            self.application.surface(),
            self.application.rnd(),
            self.short,
            self.application.left_player(),
            self.application.right_player(),
            self.fun_name,
        )
        self.page_builder.clear(self)
        self.page_builder.update(self)


class DecidedSetPageLGR(tkinter.ttk.Frame):
    def __init__(self, parent, application):
        tkinter.ttk.Frame.__init__(self, parent)
        self.application = application
        self.page_builder = decided_set.PageBuilderDecidedSetLGR(
            self.application.sex(),
            self.application.left_player(),
            self.application.right_player(),
        )
        self.page_builder.create_widgets(self)

    def estimate(self):
        self.page_builder.set_data(
            self.application.sex(),
            self.application.left_player(),
            self.application.right_player(),
        )
        self.page_builder.clear(self)
        self.page_builder.update(self)


class MatchstatPageMutual(tkinter.ttk.Frame):
    def __init__(self, parent, application, short, fst_fun_name, snd_fun_name):
        tkinter.ttk.Frame.__init__(self, parent)
        self.application = application
        self.fst_fun_name = fst_fun_name
        self.snd_fun_name = snd_fun_name
        self.short = short
        self.page_builder = matchstat.PageBuilderMutual(
            self.application.sex(),
            self.application.level(),
            self.application.surface(),
            self.application.rnd(),
            short,
            self.application.left_player(),
            self.application.right_player(),
            fst_fun_name,
            snd_fun_name,
        )
        self.page_builder.create_widgets(self)

    def estimate(self):
        self.page_builder.set_data(
            self.application.sex(),
            self.application.level(),
            self.application.surface(),
            self.application.rnd(),
            self.short,
            self.application.left_player(),
            self.application.right_player(),
            self.fst_fun_name,
            self.snd_fun_name,
        )
        self.page_builder.clear(self)
        self.page_builder.update(self)


class ThirdBestOfThreePage(tkinter.ttk.Frame):
    def __init__(self, parent, application):
        tkinter.ttk.Frame.__init__(self, parent)
        self.application = application

        row = 1
        self.estimate_btn = tkinter.ttk.Button(
            self, text="Estimate", command=self.estimate
        )
        self.estimate_btn.grid(row=row, column=0)
        row += 1

        self.genscore_chances_lbl = tkinter.ttk.Label(self, text="level generic score:")
        self.genscore_chances_lbl.grid(row=row, column=0)
        self.genscore_lhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.genscore_lhs_chances_txt.grid(row=row, column=1)
        self.genscore_rhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.genscore_rhs_chances_txt.grid(row=row, column=2)
        self.genscore_chances_size_txt = tkinter.ttk.Label(self, text="(0)")
        self.genscore_chances_size_txt.grid(row=row, column=3)
        row += 1

        self.level_chances_lbl = tkinter.ttk.Label(self, text="level (unordered):")
        self.level_chances_lbl.grid(row=row, column=0)
        self.level_lhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.level_lhs_chances_txt.grid(row=row, column=1)
        self.level_rhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.level_rhs_chances_txt.grid(row=row, column=2)
        self.level_chances_size_txt = tkinter.ttk.Label(self, text="(0)")
        self.level_chances_size_txt.grid(row=row, column=3)
        row += 1

        self.horizont_sep_1 = tkinter.ttk.Separator(self, orient=tkc.HORIZONTAL)
        self.horizont_sep_1.grid(row=row, columnspan=3, sticky=(tkc.W, tkc.E))
        row += 1

        self.sets_score_chances_lbl = tkinter.ttk.Label(self, text="sets score:")
        self.sets_score_chances_lbl.grid(row=row, column=0)
        self.sets_lhs_score_chances_txt = tkinter.ttk.Label(self, text="-")
        self.sets_lhs_score_chances_txt.grid(row=row, column=1)
        self.sets_rhs_score_chances_txt = tkinter.ttk.Label(self, text="-")
        self.sets_rhs_score_chances_txt.grid(row=row, column=2)
        self.sets_score_size_txt = tkinter.ttk.Label(self, text="(0)")
        self.sets_score_size_txt.grid(row=row, column=3)
        row += 1

        self.horizont_sep_2 = tkinter.ttk.Separator(self, orient=tkc.HORIZONTAL)
        self.horizont_sep_2.grid(row=row, columnspan=3, sticky=(tkc.W, tkc.E))
        row += 1

        self.score_chances_lbl = tkinter.ttk.Label(self, text="score:")
        self.score_chances_lbl.grid(row=row, column=0)
        self.score_lhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.score_lhs_chances_txt.grid(row=row, column=1)
        self.score_rhs_chances_txt = tkinter.ttk.Label(self, text="-")
        self.score_rhs_chances_txt.grid(row=row, column=2)
        self.score_chances_size_txt = tkinter.ttk.Label(self, text="(0)")
        self.score_chances_size_txt.grid(row=row, column=3)
        row += 1

    def clear_indicators(self):
        self.genscore_lhs_chances_txt["text"] = "-"
        self.genscore_rhs_chances_txt["text"] = "-"
        self.genscore_chances_size_txt["text"] = "(0)"
        self.level_lhs_chances_txt["text"] = "-"
        self.level_rhs_chances_txt["text"] = "-"
        self.level_chances_size_txt["text"] = "(0)"
        self.sets_lhs_score_chances_txt["text"] = "-"
        self.sets_rhs_score_chances_txt["text"] = "-"
        self.sets_score_size_txt["text"] = "(0)"
        self.score_lhs_chances_txt["text"] = "-"
        self.score_rhs_chances_txt["text"] = "-"
        self.score_chances_size_txt["text"] = "(0)"

    def estimate(self):
        def key_wl_dict(sex, player):
            if not player:
                return {}
            filename = "{}/win_long_bo3/{}.txt".format(
                cfg_dir.stat_players_dir(sex), player.name
            )
            return dict_tools.load(
                filename,
                keyfun=str,
                createfun=lambda: defaultdict(lambda: None),
                valuefun=st.WinLoss.from_text,
            )

        self.clear_indicators()
        left_key_wl_dict = key_wl_dict(
            self.application.sex(), self.application.left_player()
        )
        right_key_wl_dict = key_wl_dict(
            self.application.sex(), self.application.right_player()
        )
        self.estimate_level(left_key_wl_dict, right_key_wl_dict)
        self.estimate_sets_score(left_key_wl_dict, right_key_wl_dict)
        self.estimate_score(left_key_wl_dict, right_key_wl_dict)
        self.estimate_generic_score()

    def estimate_level(self, left_key_wl_dict, right_key_wl_dict):
        level = self.application.level()
        left_wl = left_key_wl_dict[str(level)]
        right_wl = right_key_wl_dict[str(level)]
        draw_left_right_chances(
            self.level_lhs_chances_txt, self.level_rhs_chances_txt, left_wl, right_wl
        )
        draw_size(self.level_chances_size_txt, left_wl, right_wl)

    def estimate_sets_score(self, left_key_wl_dict, right_key_wl_dict):
        score = self.application.score()
        last_set_idx = score.sets_count() - 1
        if last_set_idx < 0 or last_set_idx > 2:
            return
        set_first = score[0]
        if set_first[0] == set_first[1]:
            return  # unknown winner in first set

        if last_set_idx <= 0:
            set_second = score[0][1], score[0][0]  # 2nd set = inverse(1st set)
        else:
            set_second = score[1]
            if (set_first[0] > set_first[1] and set_second[0] > set_second[1]) or (
                set_first[0] < set_first[1] and set_second[0] < set_second[1]
            ):
                return  # straight in the first, second sets
            if set_second[0] == set_second[1]:
                if set_second[0] == 6:
                    set_second = (6, 7) if set_first[0] > set_first[1] else (7, 6)
                else:
                    set_second = score[0][1], score[0][0]  # 2nd set = inverse(1st set)
        key_left = "({}, {})".format(
            int(set_first[0] > set_first[1]), int(set_second[0] > set_second[1])
        )
        left_wl = left_key_wl_dict[key_left]

        key_right = "({}, {})".format(
            int(set_first[0] < set_first[1]), int(set_second[0] < set_second[1])
        )
        right_wl = right_key_wl_dict[key_right]
        draw_left_right_chances(
            self.sets_lhs_score_chances_txt,
            self.sets_rhs_score_chances_txt,
            left_wl,
            right_wl,
        )
        draw_size(self.sets_score_size_txt, left_wl, right_wl)

    def estimate_score(self, left_key_wl_dict, right_key_wl_dict):
        score = self.application.score()
        if score.sets_count() < 2:
            return
        set_first = score[0]
        set_second = score[1]
        if set_second == (6, 6):
            set_second = (6, 7) if set_first[0] > set_first[1] else (7, 6)
        if max(set_second[0], set_second[1]) < 6 or set_second[0] == set_second[1]:
            return

        key_left = "{}-{} {}-{}".format(
            set_first[0], set_first[1], set_second[0], set_second[1]
        )
        left_wl = left_key_wl_dict[key_left]

        key_right = "{}-{} {}-{}".format(
            set_first[1], set_first[0], set_second[1], set_second[0]
        )
        right_wl = right_key_wl_dict[key_right]
        draw_left_right_chances(
            self.score_lhs_chances_txt, self.score_rhs_chances_txt, left_wl, right_wl
        )
        draw_size(self.score_chances_size_txt, left_wl, right_wl)

    def estimate_generic_score(self):
        level = self.application.level()
        score = self.application.score()
        sex = self.application.sex()
        if score.sets_count() < 2:
            return
        set_first = score[0]
        set_second = score[1]
        if set_second == (6, 6):
            set_second = (6, 7) if set_first[0] > set_first[1] else (7, 6)
        if max(set_second[0], set_second[1]) < 6 or set_second[0] == set_second[1]:
            return  # second set is not full
        inverse = False
        if set_second[0] < set_second[1]:
            inverse = True  # second set must be winned
            set_first = set_first[1], set_first[0]
            set_second = set_second[1], set_second[0]

        filename = "{}/winmatch-after-winset2/{}.txt".format(
            cfg_dir.stat_misc_dir(sex), level
        )
        if not os.path.isfile(filename):
            return
        score_wl_dict = dict_tools.load(
            filename,
            createfun=lambda: defaultdict(lambda: None),
            keyfun=str,
            valuefun=st.WinLoss.from_text,
        )
        key = "{}-{} {}-{}".format(
            set_first[0], set_first[1], set_second[0], set_second[1]
        )
        wl = score_wl_dict[key]
        if wl:
            left_prob = wl.ratio
            right_prob = 1.0 - left_prob
            if inverse:
                left_prob, right_prob = right_prob, left_prob
            self.genscore_lhs_chances_txt["text"] = "{0:.2f}".format(left_prob)
            self.genscore_rhs_chances_txt["text"] = "{0:.2f}".format(right_prob)
            self.genscore_chances_size_txt["text"] = "({})".format(wl.size)


class Application(tkinter.Frame):
    def __init__(self, wta_players, atp_players):
        tkinter.Frame.__init__(self, None)
        self.grid()

        self.wta_players = wta_players
        self.atp_players = atp_players
        self.players = self.wta_players

        self.__create_widgets()

    def sex(self):
        return self.sex_var.get()

    def level(self):
        return self.level_var.get()

    def left_player(self):
        return self.__find_player(self.left_player_name_var.get())

    def right_player(self):
        return self.__find_player(self.right_player_name_var.get())

    def __find_player(self, name):
        if name:
            return co.find_first(self.players, lambda p: p.name == name)

    def left_player_name(self):
        return self.left_player_name_var.get()

    def right_player_name(self):
        return self.right_player_name_var.get()

    def score(self):
        return sc.Score(self.score_var.get())

    def surface(self):
        return make_surf(self.surface_var.get())

    def rnd(self):
        return tennis.Round(self.round_var.get())

    def best_of_five(self):
        return self.best_of_five_var.get() > 0

    def __create_widgets(self):
        column_p1, column_r1, column_p2, column_r2 = 0, 1, 2, 3
        row = 0
        self.sex_var = tkinter.StringVar()
        self.sex_rbtn_wta = tkinter.Radiobutton(
            self,
            text="wta",
            variable=self.sex_var,
            value="wta",
            command=self.__sex_select,
        )
        self.sex_rbtn_wta.grid(row=row, column=column_p1)
        self.sex_rbtn_wta.select()

        self.rtg1_lbl = tkinter.ttk.Label(self, text="rtg1")
        self.rtg1_lbl.grid(row=row, column=column_r1)

        self.sex_rbtn_atp = tkinter.Radiobutton(
            self,
            text="atp",
            variable=self.sex_var,
            value="atp",
            command=self.__sex_select,
        )
        self.sex_rbtn_atp.grid(row=row, column=column_p2)
        self.sex_rbtn_atp.deselect()

        self.rtg2_lbl = tkinter.ttk.Label(self, text="rtg2")
        self.rtg2_lbl.grid(row=row, column=column_r2)
        row += 1

        self.left_player_name_var = tkinter.StringVar()
        self.left_player_combo = tkinter.ttk.Combobox(
            self,
            textvariable=self.left_player_name_var,
            state="normal",
            postcommand=self.__filter_left_players,
        )
        self.left_player_combo.bind("<<ComboboxSelected>>", self.__on_left_player_event)
        self.left_player_combo.grid(row=row, column=column_p1)
        self.left_player_combo.focus_set()

        self.elo1_var = tkinter.StringVar()
        self.elo1_lbl = tkinter.ttk.Label(self, textvariable=self.elo1_var)
        self.elo1_lbl.grid(row=row, column=column_r1)

        self.right_player_name_var = tkinter.StringVar()
        self.right_player_combo = tkinter.ttk.Combobox(
            self,
            textvariable=self.right_player_name_var,
            state="normal",
            postcommand=self.__filter_right_players,
        )
        self.right_player_combo.bind(
            "<<ComboboxSelected>>", self.__on_right_player_event
        )
        self.right_player_combo.grid(row=row, column=column_p2)

        self.elo2_var = tkinter.StringVar()
        self.elo2_lbl = tkinter.ttk.Label(self, textvariable=self.elo2_var)
        self.elo2_lbl.grid(row=row, column=column_r2)
        row += 1

        self.level_var = tkinter.StringVar()
        self.level_lbl = tkinter.ttk.Label(self, text="level:")
        self.level_lbl.grid(row=row, column=column_p1)
        self.level_combo = tkinter.ttk.Combobox(
            self, textvariable=self.level_var, state="readonly"
        )
        self.level_combo["values"] = ("main", "masters", "gs", "chal", "future")
        self.level_combo.current(0)
        self.level_combo.grid(row=row, column=column_p2)
        row += 1

        self.surface_var = tkinter.StringVar()
        self.surface_lbl = tkinter.ttk.Label(self, text="surface:")
        self.surface_lbl.grid(row=row, column=column_p1)

        self.elosurf1_var = tkinter.StringVar()
        self.elosurf1_lbl = tkinter.ttk.Label(self, textvariable=self.elosurf1_var)
        self.elosurf1_lbl.grid(row=row, column=column_r1)

        self.surface_combo = tkinter.ttk.Combobox(
            self, textvariable=self.surface_var, state="readonly"
        )
        self.surface_combo["values"] = ("Hard", "Clay", "Carpet", "Grass")
        self.surface_combo.current(1)
        self.surface_combo.grid(row=row, column=column_p2)

        self.elosurf2_var = tkinter.StringVar()
        self.elosurf2_lbl = tkinter.ttk.Label(self, textvariable=self.elosurf2_var)
        self.elosurf2_lbl.grid(row=row, column=column_r2)
        row += 1

        self.round_var = tkinter.StringVar()
        self.round_lbl = tkinter.ttk.Label(self, text="round:")
        self.round_lbl.grid(row=row, column=column_p1)
        self.round_combo = tkinter.ttk.Combobox(
            self, textvariable=self.round_var, state="readonly"
        )
        self.round_combo["values"] = (
            "First",
            "Second",
            "1/4",
            "Third",
            "Fourth",
            "1/2",
            "Final",
            "q-First",
            "q-Second",
            "Qualifying",
        )
        self.round_combo.current(0)
        self.round_combo.grid(row=row, column=column_p2)
        row += 1

        self.score_var = tkinter.StringVar()
        self.score_var.set("6-6")
        self.score_lbl = tkinter.ttk.Label(self, text="score:")
        self.score_lbl.grid(row=row, column=column_p1)
        self.score_entry = tkinter.ttk.Entry(self, textvariable=self.score_var)
        self.score_entry.grid(row=row, column=column_p2)
        row += 1

        self.best_of_five_var = tkinter.IntVar()
        self.best_of_five_check = tkinter.Checkbutton(
            self,
            text="best of five",
            variable=self.best_of_five_var,
            onvalue=1,
            offvalue=0,
        )
        self.best_of_five_check.grid(row=row, column=column_p2)
        self.best_of_five_check.deselect()
        row += 1

        self.notebook = tkinter.ttk.Notebook(self, height=240)
        self.notebook.add(
            set2_after_set1loss_stat.Set2RecoveryPageLGR(self.notebook, self),
            text="S2rec",
        )

        self.notebook.add(DecidedSetPageLGR(self.notebook, self), text="dec")
        self.notebook.add(
            ThirdBestOfThreePage(self.notebook, application=self), text="third"
        )
        self.notebook.add(
            MatchstatPageMutual(
                self.notebook, self, False, "service_hold", "receive_hold"
            ),
            text="SRVHM",
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "service_hold"), text="SRVH"
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "receive_hold"), text="RCVH"
        )
        self.notebook.add(
            MatchstatPageMutual(
                self.notebook, self, False, "service_win", "receive_win"
            ),
            text="SRVWM",
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "service_win"), text="SRVW"
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "receive_win"), text="RCVW"
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "first_service_in"), text="FI"
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "break_points_saved"),
            text="BPS",
        )
        self.notebook.add(
            MatchstatPageLGR(
                self.notebook,
                self,
                False,
                co.binary_oper_result_name(sub, "break_points_saved", "service_win"),
            ),
            text="BPS-",
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "break_points_converted"),
            text="BPC",
        )
        self.notebook.add(
            MatchstatPageLGR(
                self.notebook,
                self,
                False,
                co.binary_oper_result_name(
                    sub, "break_points_converted", "receive_win"
                ),
            ),
            text="BPC-",
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "double_faults_pergame"),
            text="DF",
        )
        self.notebook.add(
            MatchstatPageLGR(self.notebook, self, False, "aces_pergame"), text="A"
        )
        self.notebook.add(MoneyBallancePage(self.notebook), text="$")
        self.notebook.grid(
            row=row, column=0, columnspan=3, sticky=(tkc.N, tkc.W, tkc.E, tkc.S)
        )

    def __sex_select(self):
        if self.sex() == "wta":
            self.players = self.wta_players
            self.best_of_five_check.deselect()
        else:
            self.players = self.atp_players
        self.left_player_name_var.set("")
        self.right_player_name_var.set("")
        self.left_player_combo.set("")
        self.right_player_combo.set("")

    def __filter_left_players(self):
        if self.left_player_name() == "" or self.left_player_name() in [
            p.name for p in self.players
        ]:
            self.left_player_combo["values"] = tuple([p.name for p in self.players])
        else:
            self.left_player_combo["values"] = tuple(
                [
                    p.name
                    for p in self.players
                    if self.left_player_name().lower() in p.name.lower()
                ]
            )

    def __filter_right_players(self):
        if self.right_player_name() == "" or self.right_player_name() in [
            p.name for p in self.players
        ]:
            self.right_player_combo["values"] = tuple([p.name for p in self.players])
        else:
            self.right_player_combo["values"] = tuple(
                [
                    p.name
                    for p in self.players
                    if self.right_player_name().lower() in p.name.lower()
                ]
            )

    def __on_left_player_event(self, _evt):
        plr = self.left_player()
        if not plr:
            self.elo1_var.set("-")
            self.elosurf1_var.set("-")
        else:
            plr.read_rating(self.sex(), surfaces=("all", self.surface()))
            self.elo1_var.set(str(plr.rating.rank("elo_alt")))
            self.elosurf1_var.set(
                str(plr.rating.rank("elo_alt", surface=self.surface()))
            )

    def __on_right_player_event(self, _evt):
        plr = self.right_player()
        if not plr:
            self.elo2_var.set("-")
            self.elosurf2_var.set("-")
        else:
            plr.read_rating(self.sex(), surfaces=("all", self.surface()))
            self.elo2_var.set(str(plr.rating.rank("elo_alt")))
            self.elosurf2_var.set(
                str(plr.rating.rank("elo_alt", surface=self.surface()))
            )


def main():
    try:
        args = parse_command_line_args()
        dbcon.open_connect()
        extplayers.initialize()
        ratings.initialize("wta", rtg_names=("elo",))
        ratings.initialize("atp", rtg_names=("elo",))
        ratings.Rating.register_rtg_name("elo_alt")
        app = Application(
            wta_players=extplayers.get_players("wta"),
            atp_players=extplayers.get_players("atp"),
        )
        app.master.title("Live")
        app.mainloop()
        log.info("sex: " + app.sex())
        log.info("left_player: " + app.left_player_name())
        log.info("right_player: " + app.right_player_name())
        log.info("score: {}".format(app.score()))
        log.info("level: {}".format(app.level()))
        log.info("surface: {}".format(app.surface()))
        dbcon.close_connect()
        return 0
    except Exception as err:
        log.exception("{0} [{1}]".format(err, err.__class__.__name__))
        return 1


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instance", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
