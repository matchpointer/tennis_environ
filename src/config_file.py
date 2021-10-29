import os
from configparser import ConfigParser
from typing import Optional

_config: Optional[ConfigParser] = None


def initialize_from_file(filename):
    if not os.path.isfile(filename):
        raise Exception(__name__ + " can not find config ini file: " + filename)
    global _config
    _config = ConfigParser()
    _config.optionxform = str
    _config.read([filename])
    length = len(_config)
    if length <= 1:
        raise Exception(f"{__name__} bad init(length={length}) from file {filename}")


def sections():
    _check_config()
    return _config.sections()


def has_section(section):
    _check_config()
    return _config.has_section(section)


def items(section):
    _check_config()
    return _config.items(section)


def has_option(section, option):
    _check_config()
    return _config.has_option(section, option)


def setval(section, option, value):
    _check_config()
    _config.set(section, option, value)


def getval(section, option, predicate=None, predicate_desc="", default_value=None):
    _check_config()
    try:
        if _config.has_section(section) and _config.has_option(section, option):
            value = _config.get(section, option)
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
    _check_config()
    try:
        if _config.has_section(section) and _config.has_option(section, option):
            value = _config.getint(section, option)
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
    _check_config()
    try:
        if _config.has_section(section) and _config.has_option(section, option):
            value = _config.getfloat(section, option)
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
    _check_config()
    try:
        if _config.has_section(section) and _config.has_option(section, option):
            return _config.getboolean(section, option)
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
    _check_config()
    try:
        if _config.has_section(section) and _config.has_option(section, option):
            value = _config.get(section, option)
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


def _check_config():
    if _config is None:
        initialize_from_file(filename="../tennis.cfg")
