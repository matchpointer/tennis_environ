# -*- coding: utf-8 -*-
import time
import io

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

import common as co
import wdriver


root_url = None
is_mobile = True


def start_url():
    if is_mobile:
        result = "http://wap.betsbc.com"
    else:
        result = "http://tny.im/56G"
    return result


def start():
    driver = wdriver.start(load_timeout=60 * 5)
    time.sleep(80)  # tempo for manual firefox USA socks puting
    driver.get(start_url())
    time.sleep(20)
    global root_url
    if "mts" not in driver.current_url:
        root_url = driver.current_url
    assert root_url is not None, "betcity root url is not defined"
    return driver


def stop(driver):
    wdriver.stop(driver)


def current_page_refresh(driver):
    driver.get(driver.current_url)


def switch_to_english(driver):
    if is_mobile:
        return switch_to_english_mobile(driver)
    frame = wdriver.find_element_by(driver, "//frame[@name='bottom']")
    driver.switch_to.frame(frame)
    sel = wdriver.find_element_by(driver, "select", bywhat=By.TAG_NAME)
    time.sleep(1)
    Select(sel).select_by_index(1)  # EN
    time.sleep(1)
    driver.switch_to.default_content()
    time.sleep(2)


def switch_to_english_mobile(driver):
    time.sleep(2)
    lng_items = list(driver.find_elements(By.XPATH, "//div[@class='head']/div/a"))
    lng_items[-1].click()
    time.sleep(3)
    en = wdriver.find_element_by(
        driver, "//div[@class='lang']/a[contains(text(), 'EN')]"
    )
    time.sleep(1)
    en.click()
    time.sleep(1)


def go_tennis_line_page(driver):
    driver.get(root_url + "/app/#/line/?line_id[]=2")
    time.sleep(2)

    checkall = wdriver.find_element_by(
        driver,
        "//input[@type='checkbox' and contains(@ng-click, 'checkAll')]",
        wait_seconds=15,
    )
    time.sleep(1)
    checkall.click()
    time.sleep(2)

    show = wdriver.find_element_by(driver, "//span[text()='Show']", wait_seconds=15)
    time.sleep(1)
    show.click()
    time.sleep(2)


def go_live_tennis_page(driver):
    if is_mobile:
        return go_live_tennis_page_mobile(driver)
    driver.get(root_url + "/app/#/live/bets/")
    time.sleep(3)
    filter_bt = wdriver.find_element_by(
        driver, "//a[contains(text(),'Filter by sports')]"
    )
    time.sleep(1.5)
    filter_bt.click()

    clear_all = wdriver.find_element_by(driver, "//a[text()='Clear all']")
    time.sleep(1.5)
    clear_all.click()

    tennis_bx = wdriver.find_element_by(
        driver, "//span[text()='Tennis' and @ng-bind='sport.text']"
    )
    time.sleep(1.5)
    tennis_bx.click()
    time.sleep(2)
    filter_bt.click()  # close filtering
    time.sleep(1)


def go_live_tennis_page_mobile(driver):
    livebets = wdriver.find_element_by(
        driver, "//div[@class='head']/div/a[text()='Live-bets']"
    )
    livebets.click()
    time.sleep(1)


def live_page_refresh(driver):
    if is_mobile:
        return live_page_refresh_mobile(driver)

    if live_page_refresh.refresh_bt is None:
        live_page_refresh.refresh_bt = wdriver.find_element_by(
            driver, "//span[text()='Refresh']"
        )
    live_page_refresh.refresh_bt.click()
    time.sleep(0.2)


live_page_refresh.refresh_bt = None


def live_page_refresh_mobile(driver):
    live_page_refresh_mobile.refresh_bt = wdriver.find_element_by(
        driver, "//td[@class='now']/a[text()='Now']"
    )
    time.sleep(0.1)
    live_page_refresh_mobile.refresh_bt.click()
    time.sleep(0.2)


live_page_refresh_mobile.refresh_bt = None


def goto_start(driver):
    driver.get(start_url())
    time.sleep(15)


def save_page(driver, filename, encoding=None):
    """depricate: use common_wdriver module"""
    if encoding is None:
        with open(filename, "w") as fh:
            fh.write(co.to_ascii(driver.page_source))
    else:
        with io.open(filename, "w", encoding=encoding) as fh:
            fh.write(driver.page_source)


def work_save_tennis_line(filename):
    driver = start()
    switch_to_english(driver)
    go_tennis_line_page(driver)
    save_page(driver, filename, encoding="utf8")  # tempo encoding
    stop(driver)


if __name__ == "__main__":
    drv = start()
    switch_to_english(drv)
    go_live_tennis_page(drv)
    save_page(drv, filename="./betcity_live_wap_utf.html", encoding="utf8")
    stop(drv)
