# -*- coding: utf-8 -*-
""" работа с selenium.webdriver
    chromedriver.exe 32bit from http://chromedriver.chromium.org/downloads
    geckodriver.exe 64bit (firefox) from https://github.com/mozilla/geckodriver/releases
"""
import time
import unittest
from contextlib import contextmanager

from enum import IntEnum

# to update selenium: pip install -U selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    NoAlertPresentException,
)
from selenium.webdriver.common.keys import Keys

from retry_decorator import retry

import log
import common as co


@retry(TimeoutException, tries=3)
def _make_driver_chrome(headless):
    opt = ChromeOptions()
    if headless:
        opt.add_argument("--headless")
        opt.add_argument("--disable-gpu")
    else:
        opt.add_argument("javascriptEnabled=True")
    driver = webdriver.Chrome(options=opt)  # here arg is optional by API
    return driver


@retry(TimeoutException, tries=3)
def _make_driver_firefox(headless):
    opt = FirefoxOptions()
    if headless:
        opt.add_argument("--headless")
        opt.add_argument("--disable-gpu")
    return webdriver.Firefox(executable_path="c:/utils/geckodriver.exe", options=opt)


@retry(TimeoutException, tries=3)
def _make_driver_opera(headless):
    return webdriver.Opera()


#   capabilities = DesiredCapabilities.OPERA.copy()
#   capabilities['turboEnabled'] = True
#   return webdriver.Opera(desired_capabilities=capabilities)


@retry(TimeoutException, tries=3)
def _make_driver_phantomjs():
    # return webdriver.PhantomJS(executable_path="c:/utils/phantomjs.exe")
    capabilities = DesiredCapabilities.PHANTOMJS.copy()
    capabilities["javascriptEnabled"] = True
    capabilities["takesScreenshot"] = False
    return webdriver.PhantomJS(
        executable_path="c:/utils/phantomjs.exe", desired_capabilities=capabilities
    )


class BROWSER(IntEnum):
    OPERA = 1
    FIREFOX = 2
    CHROME = 3
    PHANTOMJS = 4


def _make_driver(browser, headless):
    if browser == BROWSER.OPERA:
        result = _make_driver_opera(headless)
    elif browser == BROWSER.FIREFOX:
        result = _make_driver_firefox(headless)
    elif browser == BROWSER.CHROME:
        result = _make_driver_chrome(headless)
    elif browser == BROWSER.PHANTOMJS:
        result = _make_driver_phantomjs()
    else:
        raise co.TennisError("unknown browser: {}".format(browser))
    return result


def default_browser() -> BROWSER:
    if co.PlatformNodes.is_second_node():
        return BROWSER.FIREFOX
    return BROWSER.CHROME


def start(load_timeout=3 * 60, browser=default_browser(), headless=False):
    driver = _make_driver(browser, headless=headless)
    driver.set_page_load_timeout(load_timeout)  # secs
    driver.implicitly_wait(10)
    driver.maximize_window()
    driver.implicitly_wait(10)
    return driver


def stop(driver):
    driver.close()
    time.sleep(3)
    driver.quit()
    time.sleep(10)  # Дадим браузеру корректно завершиться


@contextmanager
def driver_cntx(load_timeout=3 * 60, browser=BROWSER.CHROME, headless=False):
    _drv = start(load_timeout=load_timeout, browser=browser, headless=headless)
    try:
        yield _drv
    finally:
        stop(_drv)


def load_url(driver, url, try_max=5):
    try:
        driver.get(url)
    except TimeoutException as err:
        log.error(
            "{} [{}] try_max: {} url: {}".format(
                err, err.__class__.__name__, try_max, url
            )
        )
        time.sleep(20)
        try:
            WebDriverWait(driver, 60 * 20).until(
                EC.alert_is_present(), "Timed out waiting for alert popup creation"
            )
            log.warn("Alert present passed")
            driver.switch_to_alert().accept()
            log.warn("Alert accepted successfully")
        except (TimeoutException, NoAlertPresentException) as alert_err:
            log.error(
                "waitforalert {} [{}] try_max: {} url: {}".format(
                    alert_err, alert_err.__class__.__name__, try_max, url
                )
            )
            if try_max >= 1:
                time.sleep(30)
                return load_url(driver, url, try_max=try_max - 1)
            else:
                raise err


def find_element_by(driver, content, wait_seconds=5, bywhat=By.XPATH):
    """:return WebElement if success. otherwise None"""
    try:
        element = WebDriverWait(driver, wait_seconds).until(
            EC.presence_of_element_located((bywhat, content))
        )
        return element
    except (NoSuchElementException, WebDriverException, TimeoutException):
        return None


# def find_element_clickable_by(driver, content, wait_seconds=5, bywhat=By.XPATH):
#     """ :return WebElement if success. otherwise None """
#     try:
#         element = WebDriverWait(driver, wait_seconds).until(
#             EC.presence_of_element_located((bywhat, content)) and
#             EC.element_to_be_clickable((bywhat, content))
#         )
#         return element
#     except (NoSuchElementException, WebDriverException, TimeoutException):
#         return None
#


def find_and_press_button(driver, button_text):
    """suppose: tagname = 'button'"""
    btn = find_element_by(driver, f"//button[text()='{button_text}']", wait_seconds=10)
    if btn is not None:
        btn.send_keys(Keys.ENTER)
    else:
        raise co.TennisNotFoundError(f"'{button_text}' button not found")
    return btn


def log_element(element, head_msg=""):
    if head_msg:
        log.info(head_msg)
    log.info(str(element.get_attribute("innerHTML")))


if __name__ == "__main__":
    log.initialize(
        co.logname(__file__, test=True), file_level="info", console_level="info"
    )
    #   drv = start()
    #   time.sleep(10)
    #   load_url(drv, "http://meduza.io")
    #   time.sleep(20)
    #   stop(drv)
    unittest.main()
