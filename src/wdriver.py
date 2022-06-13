# -*- coding=utf-8 -*-
""" работа с selenium.webdriver
    chromedriver.exe 32bit from http://chromedriver.chromium.org/downloads
    geckodriver.exe 64bit (firefox) from https://github.com/mozilla/geckodriver/releases
"""
import time
from enum import IntEnum

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    NoAlertPresentException,
)

from retry_decorator import retry

from loguru import logger as log
import common as co


@retry(TimeoutException, tries=3)
def _make_webdriver_chrome(headless):
    opt = ChromeOptions()
    opt.add_argument("javascriptEnabled=True")
    opt.add_argument('--no-sandbox')
    if headless:
        opt.add_argument("--headless")
    return webdriver.Chrome(options=opt)


@retry(TimeoutException, tries=3)
def _make_webdriver_firefox(headless):
    opt = FirefoxOptions()
    opt.add_argument("javascriptEnabled=True")
    opt.add_argument('--no-sandbox')
    if headless:
        opt.add_argument("--headless")
    return webdriver.Firefox(options=opt)


class BrowserKind(IntEnum):
    FIREFOX = 2
    CHROME = 3


def _default_browser_kind() -> BrowserKind:
    if co.PlatformNodes.is_second_node():
        return BrowserKind.FIREFOX
    return BrowserKind.CHROME


def make_web_driver(headless: bool,
                    browser_kind: BrowserKind = _default_browser_kind()):
    if browser_kind == BrowserKind.FIREFOX:
        result = _make_webdriver_firefox(headless)
    elif browser_kind == BrowserKind.CHROME:
        result = _make_webdriver_chrome(headless)
    else:
        raise co.TennisError(f"unknown browser kind: {browser_kind}")
    result.maximize_window()
    return result


def stop_web_driver(web_driver):
    if web_driver:
        web_driver.close()
        time.sleep(2)
        web_driver.quit()
        time.sleep(5)  # Дадим браузеру корректно завершиться


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
            log.warning("Alert present passed")
            driver.switch_to_alert().accept()
            log.warning("Alert accepted successfully")
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

