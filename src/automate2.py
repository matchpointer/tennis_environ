"""
module for update action in Oncourt.exe
uses coord methods from pyautogui (pywinauto doesn't work and bug fixed py3.8.1 and more)
"""
import time
from typing import Tuple
import pyautogui

import log
import common as co

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 2.5

update_pause = 30


def oncourt_update_connect():
    """return True. assume that oncourt is wide and update button is visible"""
    # pyautogui.hotkey('ctrl', 'U')  # ctrl-U to update
    _press_update_button()
    time.sleep(update_pause)
    _press_close_update_button()
    return True


def _mouse_move_wrap(x, y, duration, max_tries):
    try:
        _mouse_move(x, y, duration=duration)
        return
    except pyautogui.FailSafeException as err:
        log.error(f"{err} max_tries: {max_tries} x{x} y{y}")
        if max_tries > 1:
            time.sleep(10)
            _mouse_move_wrap(x, y, duration, max_tries=max_tries - 1)
        else:
            raise err


def _mouse_move(x, y, duration):
    pyautogui.moveTo(x, y, duration=duration)


def press_button(pos: Tuple[int, int]):
    if pos:
        _mouse_move_wrap(x=pos[0], y=pos[1], duration=1, max_tries=3)
        pyautogui.click()


def _press_update_button():
    press_button(_get_update_button_pos())


def _press_close_update_button():
    press_button(_get_update_close_button_pos())


def _get_update_button_pos():
    if co.PlatformNodes.is_first_node():
        return 510, 100
    if co.PlatformNodes.is_second_node():
        return 370, 80


def _get_update_close_button_pos():
    if co.PlatformNodes.is_first_node():
        return 1100, 390
    if co.PlatformNodes.is_second_node():
        return 750, 310


if __name__ == "__main__":
    # log.initialize(co.logname(__file__), file_level='info', console_level='info')
    # _press_close_update_button()
    res = oncourt_update_connect()
    print(f"answer: {res}")
