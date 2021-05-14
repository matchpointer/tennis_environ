# -*- coding: utf-8 -*-
import wdriver


is_mobile = False


def start_url():
    return "http://www.flashscore.com/tennis/"


def start(headless=False):
    driver = wdriver.start(load_timeout=60 * 5, headless=headless)
    driver.get(start_url())
    driver.implicitly_wait(10)
    return driver


def stop(driver):
    wdriver.stop(driver)


def current_page_refresh(driver):
    driver.get(driver.current_url)


def goto_start(driver):
    driver.get(start_url())
    driver.implicitly_wait(10)
