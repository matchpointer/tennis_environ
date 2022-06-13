# -*- coding=utf-8 -*-
import subprocess
import os
import time
import re
import random
import winsound

from common import TennisError
import file_utils as fu
from loguru import logger as log

WORK_DIR = "."
CMD_ENVIRON = os.environ
CMD_ENVIRON["PATH"] += ";."
CMD_ENVIRON["nls_lang"] = "american_cis.cl8mswin1251"

DECORED_FUN_RE = re.compile(r"<function (?P<fun_name>[a-zA-Z0-9._]*) .*>")

DECORED_MEMBER_FUN_RE = re.compile(
    r"<bound method (?P<class_name>[a-zA-Z0-9_]*)\.(?P<fun_name>[a-zA-Z0-9_]*) of <.*>>"
)


def quote_ifspace(text):
    return ('"' + text + '"') if " " in text else text


class TennisCommandError(TennisError):
    def __init__(self, cmd, comments=""):
        super().__init__(cmd, comments)
        self.cmd = cmd

    def __str__(self):
        result = ""
        comments = TennisError.__str__(self)
        if comments:
            result += comments + " "
        return result + str(self.cmd)


class TennisDatabaseCommandError(TennisCommandError):
    def __init__(self, cmd, comments=""):
        super().__init__(cmd, comments)


class Command:
    def __init__(
        self,
        cmd,
        args="",
        external=True,
        cwd=".",
        read_console_out=False,
        ok_return_value=None,
        check_return_fun=None,
    ):
        self.returned_value = None
        self.cmd = cmd
        self.args = args
        self.external = external
        self.cwd = cwd
        self.read_console_out = read_console_out
        self.console_out = None

        if check_return_fun:
            assert ok_return_value is None, (
                "check_return_fun and ok_return_value can"
                + " not be used together in Command"
            )

        self.ok_return_value = ok_return_value
        if external and ok_return_value is None:
            self.ok_return_value = 0  # default for external

        self.check_return_fun = check_return_fun
        if check_return_fun is None:
            self.check_return_fun = lambda x: x == self.ok_return_value

    def __str__(self):
        if type(self.args) == str:
            args_text = "(" + self.args + ")"
        else:
            args_text = (
                "(" + ", ".join([str(i) for i in self.args]) + ")" if self.args else ""
            )
        if self.external:
            return (
                "{0} {1} <dir: {2} waiting_return: {3} returned: {4} "
                "read_console_out: {5}>"
            ).format(
                self.cmd,
                args_text,
                self.cwd,
                str(self.ok_return_value),
                self.returned_value_as_string(),
                self.read_console_out,
            )
        else:
            cmd_text = "%s" % self.cmd
            match = DECORED_FUN_RE.search(cmd_text)
            if match:
                cmd_text = match.group("fun_name")  # clear from decorations
            else:
                match = DECORED_MEMBER_FUN_RE.search(cmd_text)
                if match:
                    cmd_text = (
                        match.group("class_name") + "." + match.group("fun_name")
                    )  # clear from decorations
            return "{0} {1} <waiting_return: {2} returned: {3}>".format(
                cmd_text,
                args_text,
                str(self.ok_return_value),
                self.returned_value_as_string(),
            )

    def returned_value_as_string(self):
        return str(self.returned_value)

    def external_human_model(self, towork_path):
        if self.external:
            assert self.cwd.startswith(towork_path), "cwd not startswith %s in %s" % (
                towork_path,
                self,
            )
            line_cd_into = "cd ." + self.cwd[len(towork_path) :] + "\n"
            line_cd_into = line_cd_into.replace("\\", "/")
            line_cmd = str(self.cmd)
            if self.args:
                line_cmd += " " + self.args
            line_cmd += "\n"
            line_check = "IF %ERRORLEVEL% NEQ 0 GOTO ERROR\n"
            line_cd_from = "cd " + "../" * line_cd_into.count("/") + "\n"
            return line_cd_into + line_cmd + line_check + line_cd_from + "\n"

    def fire(self):
        self.returned_value = None
        if self.external:
            if self.read_console_out:
                popen = subprocess.Popen(
                    "%s %s" % (self.cmd, self.args),
                    shell=True,
                    stdout=subprocess.PIPE,
                    env=CMD_ENVIRON,
                    cwd=self.cwd,
                )
                self.console_out = popen.communicate()[0]
                self.returned_value = popen.wait()
                return self.returned_value

            self.returned_value = subprocess.call(
                "%s %s" % (self.cmd, self.args),
                shell=True,
                env=CMD_ENVIRON,
                cwd=self.cwd,
            )
            return self.returned_value

        elif type(self.args) == list or type(self.args) == tuple:
            self.returned_value = self.cmd(*self.args)
            return self.returned_value
        elif type(self.args) == dict:
            self.returned_value = self.cmd(**self.args)
            return self.returned_value
        else:
            raise TennisError("unsupported args type: %s" % type(self.args))

    def fire_checked(self):
        self.fire()
        if not self.check_return_fun(self.returned_value):
            raise TennisCommandError(self)
        return self.returned_value

    def get_log_name(self):
        if (
            not self.external
            and (
                self.cmd in (fu.ora_error_in_file, fu.count_in_file)
                or "write_dummy_logfile" in str(self.cmd)
            )
            and type(self.args) == list
            and len(self.args) > 0
        ):
            return self.args[0]
        elif (
            self.external
            and type(self.cmd) == str
            and type(self.args) == str
            and self.cmd.startswith("ares_loader")
            and " -l " in self.args
        ):
            return self.cwd + "/" + self.args[self.args.find(" -l ") + 4 :]
        else:
            return None


class CommandAsk(Command):
    """Command for retrive answer from console user."""

    def __init__(
        self, ask, answers=("y", "n"), ok_return_value=None, check_return_fun=None
    ):
        assert answers, "answers must be defined"
        Command.__init__(
            self,
            ask,
            args=[answer.lower() for answer in answers],
            ok_return_value=ok_return_value,
            check_return_fun=check_return_fun,
        )

    def fire(self):
        answers_hint = "/".join(self.args)
        while True:
            print("%s (%s)" % (self.cmd, answers_hint))
            answer = input().lower()
            if answer in self.args:
                break
            else:
                print("your answer must be one of: %s" % answers_hint)

        self.returned_value = answer
        return self.returned_value


class CommandSqlplus(Command):
    """sqlplus command.
    ------ sample of package proc calling:
    r = CommandSqlplus(target_user.get_conn_str(),
                   plsql = 'begin idata_core.dummy(p_dummy => 1); end;', fetch = None)
    assert r == 0
    ----- sample of using sql_file as arg in CommandSqlplus (with many-rows query):
    set echo off
    set serveroutput on
    set feedback off
    set timing off
    set verify off
    declare
      CURSOR v_cur IS
        SELECT modl_id, modl_internal_name from hpc_modules;
    begin
      FOR v_rec IN v_cur
      LOOP
        dbms_output.put_line(
              to_char(v_rec.modl_id) || ' ' || to_char(v_rec.modl_internal_name));
      END LOOP;
    end;
    /
    exit;
    """

    tempo_count_in = 0
    tempo_count_out = 0
    MAX_ROWS_FETCH_ALL = 1000
    PLSQL_HEAD = (
        "set echo off\n"
        "set serveroutput on\n"
        "set feedback off\n"
        "set timing off\n"
        "set verify off\n\n"
    )
    PLSQL_TAIL = "\n/\nexit;\n"

    def __init__(
        self,
        connect_string,
        plsql=None,
        sql_file=None,
        result_file=None,
        binds=None,
        cwd=".",
        fetch="all",
        ok_return_value=None,
        check_return_fun=None,
    ):
        self.fetch = fetch
        assert (
            fetch is None or fetch == "all" or type(fetch) == int
        ), "unexpected fetch: " + str(fetch)
        self.plsql = plsql
        assert (sql_file or plsql) and not (
            sql_file and plsql
        ), "sql_file or plsql must be supplied in CommandSqlplus"
        if binds is None:
            binds = []
        if plsql and len(binds) > 0:
            assert "&" in plsql, "plsql must contain binds (&1,...)"

        self.result_file = result_file
        if result_file is None:
            self.result_file = CommandSqlplus.get_tempo_file_name_out()
            assert not os.path.isfile(self.result_file), (
                "can not create out temporary file " + self.result_file
            )
        self.sql_file = sql_file
        if sql_file is None:
            self.sql_file = CommandSqlplus.get_tempo_file_name_in()
        vars_str = ""
        for var in binds:
            vars_str += " %s" % str(var)
        Command.__init__(
            self,
            "sqlplus -s",
            args=(
                connect_string
                + " @"
                + quote_ifspace(self.sql_file)
                + vars_str
                + " > "
                + quote_ifspace(self.result_file)
            ),
            external=True,
            cwd=cwd,
            ok_return_value=ok_return_value,
            check_return_fun=check_return_fun,
        )

    def __str__(self):
        result = Command.__str__(self)
        result += "\n\tfetch: " + str(self.fetch)
        if self.plsql:
            result += " \n\tplsql: " + self.plsql
        return result

    def prepare_sql_file(self):
        assert self.sql_file is not None, "fail prepare sql file without name"
        if os.path.isfile(self.sql_file):
            if CommandSqlplus.is_tempo_file(self.sql_file):
                os.remove(self.sql_file)
            else:
                return  # by user prepared
        with open(self.sql_file, "w") as fhandle:
            fhandle.write(
                CommandSqlplus.PLSQL_HEAD + self.plsql + CommandSqlplus.PLSQL_TAIL
            )

    def returned_value_as_string(self):
        if self.returned_value is None:
            return "None"
        if type(self.returned_value) == str:
            return self.returned_value
        if self.fetch == 1:
            # one row
            return " ".join(self.returned_value)
        if (self.fetch == "all") or (type(self.fetch) == int):
            # matrix
            text = ""
            for row in self.returned_value:
                for column in row:
                    text += " " + str(column)
                text += "\n"
            return text
        raise TennisDatabaseCommandError(
            self, "unexpected returned_value_as_string: " + str(self.returned_value)
        )

    def fire(self):
        try:
            self.prepare_sql_file()
            log.debug("sqlplus firing: " + self.__str__())
            Command.fire(self)
            time.sleep(0.6)
            if self.returned_value != 0:
                raise TennisCommandError(self)
            self.returned_value = None

            if not os.path.isfile(self.result_file):
                raise TennisDatabaseCommandError(
                    self, "can not find result file " + self.result_file
                )
            if fu.ora_error_in_file(self.result_file):
                with open(self.result_file, "r") as fhandle:
                    self.returned_value = fhandle.read()
                log.debug("sqlplus ora: \n" + self.returned_value)
                raise TennisDatabaseCommandError(self, "ORA errors in result file")

            lines_max = 0
            if self.fetch == "all":
                lines_max = CommandSqlplus.MAX_ROWS_FETCH_ALL
            elif type(self.fetch) == int:
                lines_max = min(self.fetch, CommandSqlplus.MAX_ROWS_FETCH_ALL)
            rows = []
            with open(self.result_file, "r") as fhandle:
                for line in fhandle.readlines():
                    if len(rows) > lines_max:
                        break
                    rows.append(line.split())
            if self.fetch == 1 and len(rows) > 0:
                self.returned_value = rows[0]  # no matrix, as [ col1val, ..., colNval ]
            elif (type(self.fetch) == int) or (self.fetch == "all"):
                self.returned_value = rows
        finally:
            if CommandSqlplus.is_tempo_file(self.result_file) and os.path.isfile(
                self.result_file
            ):
                os.remove(self.result_file)
            if CommandSqlplus.is_tempo_file(self.sql_file) and os.path.isfile(
                self.sql_file
            ):
                os.remove(self.sql_file)
        log.debug("sqlplus out: \n" + self.returned_value_as_string())
        return self.returned_value

    def fire_checked(self):
        self.fire()
        if self.fetch is not None:
            if not self.check_return_fun(self.returned_value):
                raise TennisDatabaseCommandError(self, "invalid check")
        return self.returned_value

    @staticmethod
    def get_tempo_file_prefix():
        return "tempo_"

    @staticmethod
    def is_tempo_file(filename):
        return os.path.basename(filename).startswith(
            CommandSqlplus.get_tempo_file_prefix()
        )

    @staticmethod
    def get_tempo_file_name_in():
        CommandSqlplus.tempo_count_in += 1
        return "{0}/{1}{2}_{3}.in".format(
            WORK_DIR,
            CommandSqlplus.get_tempo_file_prefix(),
            CommandSqlplus.tempo_count_in,
            random.choice(range(10000000)),
        )

    @staticmethod
    def get_tempo_file_name_out():
        CommandSqlplus.tempo_count_out += 1
        return "{0}/{1}{2}_{3}.out".format(
            WORK_DIR,
            CommandSqlplus.get_tempo_file_prefix(),
            CommandSqlplus.tempo_count_out,
            random.choice(range(10000000)),
        )


class Commands:
    def __init__(self, commands=None):
        command_list = [] if commands is None else commands
        self.commands = [cmd for cmd in command_list if cmd is not None]

    def __str__(self):
        res = "Commands: [\n"
        for cmd in self.commands:
            res += "cmd: %s\n" % cmd
        return res + "]"

    def __getitem__(self, index):
        return self.commands[index]

    def __len__(self):
        return len(self.commands)

    def add(self, *commands):
        self.commands += [cmd for cmd in commands if cmd is not None]

    def fire(self):
        for cmd in self.commands:
            cmd.fire()
            time.sleep(1.5)
        return 0

    def fire_checked(self):
        for cmd in self.commands:
            cmd.fire_checked()
            time.sleep(1.5)
        return 0

    def get_log_names(self):
        lognames = []
        for cmd in self.commands:
            filename = cmd.get_log_name()
            if filename and filename not in lognames:
                lognames.append(filename)
        return lognames


def play_sound_file(filename):
    winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
