def msaccess_date(date):
    return "#{}/{}/{}#".format(date.month, date.day, date.year)


def sql_dates_condition(min_date, max_date, dator="tours.DATE_T"):
    result = ""
    if min_date is not None:
        result += " and {} >= {}\n".format(dator, msaccess_date(min_date))
    if max_date is not None:
        result += " and {} < {}\n".format(dator, msaccess_date(max_date))
    return result