# -*- coding=utf-8 -*-
"""
module for update action in Oncourt app
"""
import os
import time
from typing import Tuple

from pywinauto.application import Application
from pywinauto.findwindows import ElementNotFoundError
from pywinauto import mouse

import cfg_dir
from loguru import logger as log


class AutoUpdateError(Exception):
    pass


def oncourt_update_connect():
    """return True if success update done (include without data renew)"""
    try:
        app = Application(backend="uia").connect(title_re="OnCourt.*")
        return _update_app_object(app)
    except ElementNotFoundError as err:
        log.exception(err)
        return False


def oncourt_update_start():
    """return True if success update done (include without data renew)"""
    filename = os.path.join(cfg_dir.oncourt_exe_dir(), "OnCourt.exe")
    try:
        app = Application(backend="uia").start(filename)
        result = _update_app_object(app)
        app.kill()
        return result
    except ElementNotFoundError as err:
        log.exception(err)
        return False


def _close_dlg_window(app: Application, win_title: str):
    dlg = app.window(title=win_title)
    if dlg and dlg.exists():
        dlg.window(title="OK", control_type="Button").click()
        time.sleep(1)
        return True
    return None


def _close_warning_window(app: Application):
    """ True if ok closed, False if failed and stuck, None if not exist warning """

    def get_warn():
        wrn_win = app.window(title='Warning')
        if wrn_win and wrn_win.exists():
            return wrn_win

    dlg = get_warn()
    if dlg is not None:
        dlg.type_keys("~")  # ENTER
        time.sleep(1.5)

        if get_warn() is None:
            return True  # ok closed
        return False
    return None


MAX_UPDATE_SEC = 30


def _update_app_object(app: Application):
    """return True if success, if critical fail then raise AutoUpdateError and
       client should avoid next auto update call """
    result = False
    main_form = app.window(title_re="OnCourt.*")
    if main_form:
        main_form.set_focus()
        main_form.type_keys("^+u")  # send Ctrl+U invoking 'Update'
        time.sleep(MAX_UPDATE_SEC)
        upd_dlg = app.window(title="Updating database")
        if upd_dlg and upd_dlg.exists():
            result = True

            if _close_warning_window(app) is False:
                msg = 'stucked to close warning window'
                log.critical(msg)
                raise AutoUpdateError(msg)

            if _close_dlg_window(app, win_title='Error'):
                # ошибка на сервере (обновления наверное не было), окно закрыто
                result = False

            # TODO if exist btn 'Cancel' (may be 'Try again') -> press it, error return
            upd_dlg.window(title="Close", control_type="Button").click()
    return result


def mouse_click(pos: Tuple[int, int]):
    """ left mouse button click on coord. """
    if pos is not None:
        mouse.click(button='left', coords=pos)
