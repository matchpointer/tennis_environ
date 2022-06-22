# -*- coding=utf-8 -*-
r"""
module for TourInfo (used in live mode).
"""
import datetime
from typing import Optional, Dict

import config_file as cf
from loguru import logger as log
import score as sc
import tennis_time as tt
from tour_name import TourName


class TourInfo:
    def __init__(
        self,
        sex=None,
        tour_name=None,
        surface=None,
        level='',
        qualification=False,
        doubles=None,
        teams=None,
        exhibition=False,
        itf=None,
    ):
        self.teams = teams
        self.sex = sex
        self.doubles = doubles
        self.surface = surface
        self.exhibition = exhibition
        self.tour_name: TourName = tour_name or TourName(name='')
        self.qualification = qualification
        self.level = level
        self.itf = itf

    def __str__(self):
        result = "<{}> {} {} {}".format(
            self.level, self.sex, self.tour_name, self.surface
        )
        if self.qualification:
            result += " qual"
        if self.doubles:
            result += " doubles"
        return result

    def __eq__(self, other):
        return (
            self.tour_name == other.tour_name
            and self.sex == other.sex
            and self.teams == other.teams
            and self.qualification == other.qualification
            and self.doubles == other.doubles
            and self.itf == other.itf
        )

    @property
    def grand_slam(self):
        return self.level == "gs"

    @property
    def best_of_five(self):
        result = None
        if self.sex == "wta":
            result = False
        elif self.sex == "atp":
            if self.grand_slam:
                if not self.qualification:
                    result = True
                else:
                    result = "wimbledon" in self.tour_name  # at Wim qualify is bo5
            else:
                result = self.teams and not self.doubles
        return result

    @property
    def decided_tiebreak(self) -> Optional[sc.TieInfo]:
        return sc.decided_tiebreak(
            self.sex, datetime.date.today().year, self.tour_name,
            self.qualification, self.level
        )

    def dates_range_for_query(self, match_date):
        """return min_date, max_date for sql query in semiopen style:
                  min_date <= x < max_date"""
        if match_date is None:
            match_date = datetime.date.today()
        match_past_monday = tt.past_monday_date(match_date)
        match_further_monday = match_past_monday + datetime.timedelta(days=7)
        if self.itf:
            if self.qualification and match_date.isoweekday() in (6, 7):
                min_date = match_further_monday
                max_date = min_date + datetime.timedelta(days=7)
            else:
                min_date = match_past_monday
                max_date = match_further_monday
        else:
            if self.grand_slam:
                # самый широкий интервал
                min_date = match_past_monday - datetime.timedelta(days=14)
                max_date = match_further_monday + datetime.timedelta(days=14)
            else:
                # напр. 2-х недельный турнир играется (поэтому интервал чуть шире)
                min_date = match_past_monday - datetime.timedelta(days=7)
                max_date = match_further_monday + datetime.timedelta(days=7)
        return min_date, max_date

    @staticmethod
    def tour_name_map_to_oncourt(sex: str, tourname: str):
        section = "{}-tours-map-to-oncourt".format(sex)
        if cf.has_section(section) and cf.has_option(section, tourname):
            return cf.getval(section, tourname)
        return tourname


class TourInfoCache:
    def __init__(self, skip_exc_cls, skip_keys=None):
        """ для ускорения сканирования событий в skip_keys
            можно заранее передать странные ключи, чтобы на ранней
            стадии пропускались, не доходя до специфических исключений """
        self.skip_exc_cls = skip_exc_cls
        # does marking as skip (if encountered skip item -> raise self.skip_exc_cls()):
        self.skip_obj = TourInfo()
        self.cache: Dict[str, TourInfo] = dict()
        if skip_keys:
            for sk_key in skip_keys:
                self.cache[sk_key] = self.skip_obj

    def get(self, key: str) -> Optional[TourInfo]:
        obj = self.cache.get(key)
        if obj is self.skip_obj:
            raise self.skip_exc_cls()
        return obj

    def put(self, key: str, obj: TourInfo) -> None:
        self.cache[key] = obj

    def put_skip(self, key: str):
        self.cache[key] = self.skip_obj
        raise self.skip_exc_cls()

    def edit_skip(self, obj: TourInfo):
        key_mark = None
        for key, tinfo in self.cache.items():
            if tinfo == obj:
                key_mark = key
                break
        if key_mark is not None:
            self.cache[key_mark] = self.skip_obj
        raise self.skip_exc_cls()

    def log(self):
        log.info("ticache:")
        for key, tinfo in self.cache.items():
            if tinfo == self.skip_obj:
                log.info(f"key: {key} ti: SKIP")
            else:
                log.info(f"key: {key} ti: {tinfo}")
        if not self.cache:
            log.info("EMPTY ticache")
