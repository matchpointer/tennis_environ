import os
import configparser as cfgparser

import crypt

__config = None


def initialize_from_file(filename, sign):
    if not os.path.isfile(filename):
        raise Exception(__name__ + " can not find config ini file: " + filename)
    global __config
    if __config is not None:
        return False  # already done
    if sign not in open(filename, "r").read():
        new_filename = filename + ".key"
        if not os.path.isfile(new_filename):
            raise Exception(__name__ + " can not find key file for: " + filename)
        key = crypt.load_key(new_filename)
        with open(filename, "rb") as file:
            encrypted_data = file.read()
        data = crypt.decrypt_data(encrypted_data, key)
        __config = cfgparser.ConfigParser(allow_no_value=True)
        __config.optionxform = str
        __config.read_string(data.decode("utf-8"))
    else:
        __config = cfgparser.ConfigParser()
        __config.optionxform = str
        __config.read([filename])
    return True


def sections():
    __check_config()
    return __config.sections()


def has_section(section):
    __check_config()
    return __config.has_section(section)


def items(section):
    __check_config()
    return __config.items(section)


def has_option(section, option):
    __check_config()
    return __config.has_option(section, option)


def getval(section, option, predicate=None, predicate_desc="", default_value=None):
    __check_config()
    try:
        if __config.has_section(section) and __config.has_option(section, option):
            value = __config.get(section, option)
            if predicate is not None:
                if not predicate(value):
                    raise ValueError("broken constraint: '{}'".format(predicate_desc))
            return value
        return default_value
    except ValueError as err:
        raise ValueError(
            "parameter '{}' in section '{}'. error: {}".format(option, section, err)
        )


def __check_config():
    if __config is None:
        raise ValueError('not inited config file')
