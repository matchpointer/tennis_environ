# -*- coding=utf-8 -*-

from pages.base import WebPage
from pages.elements import WebElement


class MainPageTennisbetsite(WebPage):

    def __init__(self, web_driver, url=''):
        if not url:
            url = "http://www.tennisbetsite.com/"

        super().__init__(web_driver, url)
        self._start_url = url

    def get_start_url(self):
        """ Returns start browser URL. """

        return self._start_url
