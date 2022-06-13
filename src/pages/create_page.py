# -*- coding=utf-8 -*-

from wdriver import make_web_driver
from score_company import ScoreCompany
from pages.tennis24page import MainPageTennis24
from pages.tennisbetsitepage import MainPageTennisbetsite


def create_page(
    score_company: ScoreCompany,
    is_main: bool,
    headless: bool,
):
    if score_company.abbrname == 'T24' and is_main:
        web_driver = make_web_driver(headless=headless)
        return MainPageTennis24(web_driver=web_driver)
    elif score_company.abbrname == 'TBS' and is_main:
        web_driver = make_web_driver(headless=headless)
        return MainPageTennisbetsite(web_driver=web_driver)
    raise NotImplementedError(
        f'score_company: {score_company}'
        f' is_main: {is_main}'
        f' headless: {headless}'
    )
