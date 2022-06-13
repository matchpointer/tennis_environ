# -*- coding=utf-8 -*-

from datetime import date

from pages.base import WebPage
from pages.elements import WebElement


def get_tab_xpath(text):
    return (f'//div[@class="filters__group"]/'
            f'descendant::div[contains(text(), "{text}")]/'
            f'parent::div[starts-with(@class, "filters__tab")]')


class MainPageTennis24(WebPage):

    def __init__(self, web_driver, url=''):
        if not url:
            url = "http://www.tennis24.com/"

        super().__init__(web_driver, url)
        self._start_url = url

    def get_start_url(self):
        """ Returns start browser URL. """

        return self._start_url

    def parse_date(self):
        """ Returns date displayed at calendar. """
        date_txt = self.calendar.get_text()
        if date_txt:
            date_txt = date_txt.strip()
            if len(date_txt) >= 5:
                day_txt, month_txt = date_txt[:5].split(r"/")
                return date(year=date.today().year,
                            month=int(month_txt),
                            day=int(day_txt))

    # к этим стат членам при использовании будут присоединены ссылочные атрибуты
    # _web_driver на self._web_driver, и _page на self
    prev_day_button = WebElement(
        xpath='//div[contains(@class, "calendar__navigation--yesterday")]')

    # it displays current date
    calendar = WebElement(
        xpath='//div[starts-with(@class, "calendar__datepicker")]')

    next_day_button = WebElement(
        xpath='//div[contains(@class, "calendar__navigation--tomorrow")]')

    # filter tabs group (located above all tennis events)
    tab_all = WebElement(xpath=get_tab_xpath("All Matches"))
    tab_live = WebElement(xpath=get_tab_xpath("LIVE NOW"))
    tab_results = WebElement(xpath=get_tab_xpath("Results"))
    tab_scheduled = WebElement(xpath=get_tab_xpath("Upcoming Matches"))
