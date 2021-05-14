"""
module for update action in Oncourt.exe
IMPORTANT: client python script must be run with administrator rights.
           If it is not supplied then type_keys() will not work.
"""
import os
import time

from pywinauto.application import Application
from pywinauto.findwindows import ElementNotFoundError

import cfg_dir
import log
import common as co


def oncourt_update_connect():
    """return True if success update done (include without data renew)"""
    try:
        app = Application(backend="uia").connect(title_re="OnCourt.*")
        return _update_app_object(app)
    except ElementNotFoundError as err:
        log.error(err, exception=True)
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
        log.error(err, exception=True)
        return False


def _update_app_object(app):
    """return True if success"""
    result = False
    main_form = app.window(title_re="OnCourt.*")
    if main_form:
        main_form.set_focus()
        main_form.type_keys("^+u")  # send Ctrl+U invoking 'Update'
        time.sleep(20)
        upd_dlg = app.window(title="Updating database")
        if upd_dlg and upd_dlg.exists():
            result = True
            err_dlg = app.window(title="Error")
            if err_dlg and err_dlg.exists():
                result = False
                err_dlg.window(title="OK", control_type="Button").click()  # was fail
                time.sleep(1)
            # TODO if exist btn 'Cancel' (may be 'Try again') -> press it, error return
            upd_dlg.window(title="Close", control_type="Button").click()
    return result


if __name__ == "__main__":
    log.initialize(co.logname(__file__), file_level="info", console_level="info")
    res = oncourt_update_connect()
    print("answer: {}".format(res))
