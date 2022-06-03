import wdriver


def start(start_url: str, headless=False):
    driver = wdriver.start(load_timeout=60 * 5, headless=headless)
    driver.get(start_url)
    driver.implicitly_wait(10)
    return driver


def stop(driver):
    wdriver.stop(driver)


def current_page_refresh(driver):
    driver.get(driver.current_url)


def goto_start(start_url, driver):
    driver.get(start_url)
    driver.implicitly_wait(10)
