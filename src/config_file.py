# -*- coding: utf-8 -*-
import os
import configparser as cfgparser


__config = None


def initialize_from_file(filename):
    if not os.path.isfile(filename):
        raise Exception(__name__ + " can not find config ini file: " + filename)
    global __config
    __config = cfgparser.ConfigParser()
    __config.optionxform = str
    __config.read([filename])


def initialize_from_object(obj):
    global __config
    __config = obj


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


def setval(section, option, value):
    __check_config()
    __config.set(section, option, value)


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


def getint(section, option, predicate=None, predicate_desc="", default_value=None):
    __check_config()
    try:
        if __config.has_section(section) and __config.has_option(section, option):
            value = __config.getint(section, option)
            if predicate is not None:
                if not predicate(value):
                    raise ValueError("broken constraint: '{}'".format(predicate_desc))
            return value
        return default_value
    except ValueError as err:
        raise ValueError(
            "integer parameter '{}' in section '{}'. error: {}".format(
                option, section, err
            )
        )


def getfloat(section, option, predicate=None, predicate_desc="", default_value=None):
    __check_config()
    try:
        if __config.has_section(section) and __config.has_option(section, option):
            value = __config.getfloat(section, option)
            if predicate is not None:
                if not predicate(value):
                    raise ValueError("constraint is boken: '{}'".format(predicate_desc))
            return value
        return default_value
    except ValueError as err:
        raise ValueError(
            "float parameter '{}' in section '{}'. error: {}".format(
                option, section, err
            )
        )


def getboolean(section, option, default_value=None):
    __check_config()
    try:
        if __config.has_section(section) and __config.has_option(section, option):
            return __config.getboolean(section, option)
        return default_value
    except ValueError as err:
        raise ValueError(
            "boolean parameter '{}' in section '{}'. error: {}".format(
                option, section, err
            )
        )


def getlist(
    section,
    option,
    delimiter=",",
    item_final_fun=str,
    item_filter_fun=lambda i: True,
    default_value=None,
):
    __check_config()
    try:
        if __config.has_section(section) and __config.has_option(section, option):
            value = __config.get(section, option)
            return [
                item_final_fun(item)
                for item in value.split(delimiter)
                if item_filter_fun(item)
            ]
        return default_value
    except ValueError as err:
        raise ValueError(
            "read parameter '{}' (list) in section '{}'. error: {}".format(
                option, section, err
            )
        )


def getlist_notempty_values(
    section, option, delimiter=",", lower_case=True, strip=True, default_value=None
):
    if strip and lower_case:
        item_final_fun = lambda i: i.strip().lower()
    elif strip and not lower_case:
        item_final_fun = str.strip
    elif not strip and lower_case:
        item_final_fun = str.lower
    else:
        item_final_fun = str
    return getlist(
        section,
        option,
        delimiter=delimiter,
        item_final_fun=item_final_fun,
        item_filter_fun=lambda i: i.strip() != "",
        default_value=default_value,
    )


def __check_config():
    if __config is None:
        initialize_from_file(filename="../tennis.cfg")
