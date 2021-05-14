# -*- coding: utf-8 -*-
import sys
from contextlib import closing

import cfg_dir
import file_utils as fu
import common as co
import log
import dba


def execute_script_file(filename):
    with closing(dba.get_connect().cursor()) as cursor:
        with open(filename, "r") as fhandle:
            print("goto execute ", filename)
            sql = fhandle.read()
            cursor.execute(sql.strip())
            cursor.commit()
            print("executed ", filename)


def process_scripts(filemask):
    sql_scripts_dir = cfg_dir.oncourt_sql_dir()
    fu.apply_to_files(
        co.Functor(execute_script_file),
        sql_scripts_dir,
        filemask=filemask,
        recursive=True,
    )


def main():
    try:
        log.initialize(co.logname(__file__), file_level="debug", console_level="info")
        log.info("\n---------------------------------------------------------\nstarted")
        dba.open_connect()
        process_scripts(filemask="*.sql")
        log.info("finished.")
        dba.close_connect()
        return 0
    except Exception as err:
        log.error("{0} [{1}]".format(err, err.__class__.__name__), exception=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
