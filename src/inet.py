import time
import urllib.request, urllib.error, urllib.parse
from socket import error as SocketError
from http.client import HTTPException
from contextlib import closing
import gzip
import io

import log
import file_utils as fu
import common as co

"""
work in Internet
"""

__firefox_version = None


def get_firefox_version_string(
    filename=r"C:\Program Files\Mozilla Firefox\firefox.exe",
    default_version=(26, 0, 0, 0),
):
    global __firefox_version
    if __firefox_version is not None:
        return __firefox_version
    __firefox_version = fu.get_file_version(filename)
    if __firefox_version is None:
        __firefox_version = default_version
    return str(__firefox_version[0]) + "." + str(__firefox_version[1])


def write_firefox_user_agent(filename="./user_agent.txt"):
    with open(filename, "w") as fhandle:
        fhandle.write(default_mozilla_user_agent())


def default_mozilla_user_agent():
    return "Mozilla/5.0 (Windows NT 6.0; rv:{0}) Gecko/20100101 Firefox/{0}".format(
        get_firefox_version_string()
    )


def default_mozilla_headers():
    return {"User-Agent": default_mozilla_user_agent()}


def oncourt_headers():
    return {
        "Accept": "text/html, */*",
        "User-Agent": "Mozilla/3.0 (compatible; Indy Library)",
    }


def read_ziped(content):
    gzip_filehandle = gzip.GzipFile(fileobj=io.StringIO(content))
    return gzip_filehandle.read()


def fetch_url_wrap(url, data=None, headers=None, try_max=4, sleep_on_error=20):
    try:
        return fetch_url(url, data=data, headers=headers)
    except urllib.error.HTTPError as err:
        if err.code == 404:  # 'Not Found'
            return None
        else:
            raise err
    except (urllib.error.URLError, SocketError, HTTPException, OSError) as err:
        log.error("{} [{}] try_max: {}".format(err, err.__class__.__name__, try_max))
        if try_max > 1:
            time.sleep(sleep_on_error)
            return fetch_url_wrap(
                url,
                data=data,
                headers=headers,
                try_max=try_max - 1,
                sleep_on_error=sleep_on_error,
            )
        else:
            raise err


def fetch_url(url, data=None, headers=None):
    result = None
    req = urllib.request.Request(url, data=data, headers=headers)
    with closing(urllib.request.urlopen(req)) as response:
        if response.getcode() == 200:  # 'OK'
            result = response.read().decode()
    return result


class WebPage(object):
    def __init__(self, url, random_sleep_max=None):
        self.url = url
        self._content = None
        self.random_sleep_max = random_sleep_max

    def __hash__(self):
        return hash(self.url)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.url)

    def __random_sleep(self):
        if self.random_sleep_max is not None:
            co.random_sleep(1, self.random_sleep_max)

    @property
    def content(self):
        if self._content is None:
            self.__random_sleep()
            self._content = fetch_url_wrap(
                url=self.url, headers=default_mozilla_headers()
            )
        return self._content

    def refresh(self):
        self.__random_sleep()
        self._content = fetch_url_wrap(url=self.url, headers=default_mozilla_headers())
