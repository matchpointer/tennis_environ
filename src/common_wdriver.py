# -*- coding: utf-8 -*-
import time
import requests
import gzip
import io
from typing import Optional

import betcity_wdriver as bcdrv
import flashscore_wdriver as fsdrv
import file_utils as fu


def wdriver(company_name: str, headless=False, faked=False):
    """ company_name 'FS' (flashscore, default now);
                     'BC' (betcity, not used now)
        faked: True than try experimental emulation without selenium.
               faked true mode is not work with flashscore site (selenium needed).
    """
    if company_name == "BC":
        return (
            WDriverFaked(bcdrv.start_url())
            if faked
            else WDriverBetcity()
        )
    elif company_name == "FS":
        return (
            WDriverFaked(fsdrv.start_url())
            if faked
            else WDriverFlashscore(headless=headless)
        )


class WDriver(object):
    def __init__(self):
        self.drv = None

    def page(self) -> Optional[str]:
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
    def __init__(self, url):
        self.url = url
        self.content = None

    def _get_session(self):
        return requests.Session()

    def _download_page(self) -> str:
        session = self._get_session()
        with session.get(self.url) as response:
            response.raise_for_status()
            gzip_filehandle = gzip.GzipFile(fileobj=io.BytesIO(response.content))
            return gzip_filehandle.read().decode()

    def _download_page_simple(self) -> str:
        response = requests.get(self.url)
        response.encoding = 'utf-8'
        response.raise_for_status()
        return response.text

    def _do_download(self):
        self.content = None
        self.content = self._download_page_simple()

    def page(self) -> Optional[str]:
        return self.content

    def start(self):
        self._do_download()

    def stop(self):
        pass

    def go_live_page(self):
        self._do_download()

    def live_page_refresh(self):
        self._do_download()

    def current_page_refresh(self):
        self._do_download()

    def is_mobile(self):
        pass

    def save_page(self, filename, encoding="utf8"):
        fu.write(filename=filename, data=self.content, encoding=encoding)

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

