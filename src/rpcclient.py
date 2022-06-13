# -*- coding=utf-8 -*-
"""
After davbeaz.com and his 'Python Cookbook' helps in rpcclient

# ---------------- Example use ----------------
from multiprocessing.connection import Client
import time

ADDRESS = ('localhost', 17000)
AUTHKEY = b'peekaboo'

proxy = RPCProxy(Client(ADDRESS, authkey=AUTHKEY))

print(proxy.add(2, 3))  # >>> 5
print(proxy.sub(2, 3))  # >>> -1
proxy.sub([1, 2], 4)  # TypeError: unsupported operand type(s) for -: 'list' and 'int'
"""

import pickle


class RPCProxy:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        def do_rpc(*args, **kwargs):
            self._connection.send(pickle.dumps((name, args, kwargs)))
            result = pickle.loads(self._connection.recv())
            if isinstance(result, Exception):
                raise result
            return result
        return do_rpc


class RPCProxyNoAnswer:
    """ No wait return value from server """
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        """ client side can catch ConnectionError (typical network problems) """
        def do_rpc(*args, **kwargs):
            self._connection.send(pickle.dumps((name, args, kwargs)))

        return do_rpc



