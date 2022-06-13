# -*- coding=utf-8 -*-
from typing import Optional
import rpcclient

from loguru import logger as log

from multiprocessing.connection import Client

from betfair_bet import Market

ADDRESS = ('localhost', 17000)
AUTHKEY = b'peekaboo'


_proxy: Optional[rpcclient.RPCProxyNoAnswer] = None


def initialize(verbose=True):
    global _proxy
    try:
        _proxy = rpcclient.RPCProxyNoAnswer(Client(ADDRESS, authkey=AUTHKEY))
        if verbose:
            log.info("betfair client initialized")
    except ConnectionError as err:
        _proxy = None
        if verbose:
            log.error("betfair client init fail: {}".format(err))
        else:
            raise err


def is_initialized():
    return _proxy is not None


def de_initialize():
    global _proxy
    if _proxy is not None:
        del _proxy
        _proxy = None
        log.info("betfair client de_initialized")


def send_message(
    market: Market,
    sex: str,
    case_name,
    fst_name,
    snd_name,
    back_side,
    summary_href,
    fst_betfair_name="",
    snd_betfair_name="",
    prob=0.,
    book_prob=0.,
    fst_id: int = None,
    snd_id: int = None,
    comment: str = "",
    level: str = "main",
):
    global _proxy
    if _proxy is None:
        log.error("betfair client calling but it is none inited")
        return None
    try:
        # assumption: function put_message with same args is registered on server side
        _proxy.put_message(
            market,
            sex,
            case_name,
            fst_name,
            snd_name,
            back_side,
            summary_href,
            fst_betfair_name,
            snd_betfair_name,
            prob,
            book_prob,
            fst_id,
            snd_id,
            comment,
            level,
        )
    except ConnectionError as err:
        log.error("betfair client connect fail: {}".format(err))
    except Exception as err:
        log.error("betfair client fail: {}".format(err))
