# -*- coding=utf-8 -*-
"""
After davbeaz.com and his 'Python Cookbook' helps in rpcserver

# ---------------- Example use ----------------
from multiprocessing.connection import Listener
from threading import Thread
import time

ADDRESS = ('localhost', 17000)
AUTHKEY = b'peekaboo'


def rpc_server(handler):
    listener = Listener(ADDRESS, authkey=AUTHKEY)
    while True:
        client = listener.accept()
        t = Thread(target=handler.handle_connection, args=(client,))
        t.daemon = True
        t.start()


# Some remote functions
def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def waiting(sleep_sec):
    time.sleep(sleep_sec)


# Register with a handler
handler = RPCHandler()
handler.register_function(add)
handler.register_function(sub)
handler.register_function(waiting, need_return=False)

# Run the server
rpc_server(handler)
"""

import pickle


class RPCHandler:
    def __init__(self):
        self._functions = {}
        self._need_return = {}

    def register_function(self, func, need_return=True):
        self._functions[func.__name__] = func
        self._need_return[func.__name__] = need_return

    def handle_connection(self, connection):
        try:
            while True:
                # Receive a message
                func_name, args, kwargs = pickle.loads(connection.recv())
                # Run the RPC and send a response
                try:
                    r = self._functions[func_name](*args, **kwargs)
                    if self._need_return[func_name]:
                        connection.send(pickle.dumps(r))
                except Exception as e:
                    if self._need_return[func_name]:
                        connection.send(pickle.dumps(e))
        except EOFError:
            pass  # connect has dropped on client side
