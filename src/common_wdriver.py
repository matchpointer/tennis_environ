# -*- coding: utf-8 -*-
import time

import betcity_wdriver as bcdrv
import flashscore_wdriver as fsdrv
import inet
import file_utils as fu


def wdriver(company_name: str, headless=False, faked=False, random_sleep_max=None):
    if company_name == "BC":
        return (
            WDriverFaked(bcdrv.start_url(), random_sleep_max)
            if faked
            else WDriverBetcity()
        )
    elif company_name == "FS":
        return (
            WDriverFaked(fsdrv.start_url(), random_sleep_max)
            if faked
            else WDriverFlashscore(headless=headless)
        )


class WDriver(object):
    def __init__(self):
        self.drv = None

    def page(self):
        if self.drv is not None:
            return self.drv.page_source

    def start(self):
        raise NotImplementedError()

    def goto_start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def go_live_page(self):
        raise NotImplementedError()

    def go_line_page(self):
        raise NotImplementedError()

    def live_page_refresh(self):
        raise NotImplementedError()

    def current_page_refresh(self):
        raise NotImplementedError()

    def is_mobile(self):
        return False

    def save_page(self, filename, encoding="utf8"):
        fu.write(filename=filename, data=self.page(), encoding=encoding)

    def implicitly_wait(self, seconds):
        if self.drv is not None:
            self.drv.implicitly_wait(seconds)


class WDriverFaked(object):
    def __init__(self, url, random_sleep_max=None):
        self.wpage = inet.WebPage(url, random_sleep_max=random_sleep_max)

    def page(self):
        return self.wpage.content

    def start(self):
        self.wpage.refresh()

    def stop(self):
        pass

    def go_live_page(self):
        self.wpage.refresh()

    def live_page_refresh(self):
        self.wpage.refresh()

    def current_page_refresh(self):
        self.wpage.refresh()

    def is_mobile(self):
        return False

    def save_page(self, filename, encoding="utf8"):
        fu.write(filename=filename, data=self.page(), encoding=encoding)

    def implicitly_wait(self, seconds):
        time.sleep(seconds)


class WDriverBetcity(WDriver):
    def __init__(self):
        super(WDriverBetcity, self).__init__()

    def start(self):
        self.drv = bcdrv.start()
        time.sleep(20)
        bcdrv.switch_to_english(self.drv)

    def goto_start(self):
        bcdrv.goto_start(self.drv)

    def stop(self):
        bcdrv.stop(self.drv)

    def go_live_page(self):
        bcdrv.go_live_tennis_page(self.drv)

    def go_line_page(self):
        bcdrv.go_tennis_line_page(self.drv)

    def live_page_refresh(self):
        bcdrv.live_page_refresh(self.drv)

    def current_page_refresh(self):
        bcdrv.current_page_refresh(self.drv)

    def is_mobile(self):
        return bcdrv.is_mobile


class WDriverFlashscore(WDriver):
    def __init__(self, headless):
        super(WDriverFlashscore, self).__init__()
        self.headless = headless

    def start(self):
        self.drv = fsdrv.start(headless=self.headless)

    def goto_start(self):
        fsdrv.goto_start(self.drv)

    def stop(self):
        fsdrv.stop(self.drv)

    def go_live_page(self):
        pass

    def go_line_page(self):
        pass

    def live_page_refresh(self):
        pass

    def current_page_refresh(self):
        fsdrv.current_page_refresh(self.drv)


if __name__ == "__main__":
    drv = wdriver(company_name="FS", faked=False)
    drv.start()
    time.sleep(5)
    page1 = drv.page().strip()
    h1 = hash(page1)
    len1 = len(page1)
    filename = "./fs_test.html"
    fu.write(filename=filename, data=page1, encoding="utf8")
    drv.stop()
    page2 = fu.read(filename=filename, encoding="utf8")
    h2 = hash(page2)
    len2 = len(page2)
    print(f"hash1={h1} len1={len1}\nhash2={h2} len2={len2}")
