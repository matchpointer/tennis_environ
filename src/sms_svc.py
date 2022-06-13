# -*- coding=utf-8 -*-
import os
import urllib.request
import urllib.error
import urllib.parse
from urllib.parse import quote
from socket import error as SocketError
from collections import defaultdict, namedtuple
import datetime
import time

import smsc_api
from enum import IntEnum

import cmds
import cfg_dir
import config_personal


class Providers(IntEnum):
    sms_ru = 1
    smsc_ru = 2


CURRENT_PROVIDER = Providers.sms_ru

servicecodes = {
    100: ("Сообщение принято к отправке. На следующих строчках вы найдете идентификаторы"
          " отправленных сообщений в том же порядке, в котором вы указали номера,"
          " на которых совершалась отправка."),
    200: "Неправильный api_id",
    201: "Не хватает средств на лицевом счету",
    202: "Неправильно указан получатель",
    203: "Нет текста сообщения",
    204: "Имя отправителя не согласовано с администрацией",
    205: "Сообщение слишком длинное (превышает 8 СМС)",
    206: "Будет превышен или уже превышен дневной лимит на отправку сообщений",
    207: ("На этот номер (или один из номеров) нельзя отправлять сообщения,"
          " либо указано более 100 номеров в списке получателей"),
    208: "Параметр time указан неправильно",
    209: "Вы добавили этот номер (или один из номеров) в стоп-лист",
    210: "Используется GET, где необходимо использовать POST",
    211: "Метод не найден",
    220: "Сервис временно недоступен, попробуйте чуть позже.",
    300: "Неправильный token (возможно истек срок действия, либо ваш IP изменился)",
    301: "Неправильный пароль, либо пользователь не найден",
    302: ("Пользователь авторизован, но аккаунт не подтвержден"
          " (пользователь не ввел код, присланный в регистрационной смс)"),
}


sended_list_from_date = defaultdict(lambda: [])


def remove_old_sended():
    for date in list(sended_list_from_date.keys()):
        if date < datetime.date.today():
            del sended_list_from_date[date]


class SMSError(Exception):
    pass


SendedKeys = namedtuple("SendedKeys", "sex fst_name snd_name back_side setname")

sended = set()


def is_already_sended(alert_message):
    """True if resemble message has been already sended"""
    if alert_message.case_name:
        setname = alert_message.case_name.split("_")[0]
        sending_keys = SendedKeys(
            sex=alert_message.sex,
            fst_name=alert_message.fst_name,
            snd_name=alert_message.snd_name,
            back_side=alert_message.back_side,
            setname=setname,
        )
        is_sended = sending_keys in sended
        if not is_sended:
            sended.add(sending_keys)
        return is_sended


def send_alert_messages(alert_messages):
    #   dtnow = datetime.datetime.now()
    is_new_content = False
    for alert_msg in alert_messages:
        if is_already_sended(alert_msg):
            continue
        is_new_content = True
        msg_text = alert_msg.text
        send_sms(msg_text, to=config_personal.getval('sms', 'NUMBER1'))
        time.sleep(0.2)

    if is_new_content:
        # if sms is delaying then simple sound would be useful:
        sound_filename = os.path.join(cfg_dir.sounds_dir(), "gong.wav")
        cmds.play_sound_file(sound_filename)


def send_sms(text, to):
    if CURRENT_PROVIDER == Providers.sms_ru:
        send_sms_ru(text, to=to)
    elif CURRENT_PROVIDER == Providers.smsc_ru:
        send_smsc_ru(text, to=to)


def send_smsc_ru(text, to):
    """smsc.ru"""
    smsc = smsc_api.SMSC()
    smsc.send_sms(to, text)


def send_sms_ru(text, to):
    assert text, "invalid sms text: '{}'".format(text)
    url = "http://sms.ru/sms/send?api_id=%s&to=%s&text=%s" % (
        config_personal.getval('sms', 'SMSRU_PASSWORD'),
        to,
        quote(text),
    )

    try:
        res = urllib.request.urlopen(url, timeout=10)
    except (urllib.error.URLError, SocketError) as err:
        raise SMSError("send sms failed with error: '{}'".format(err))

    service_result = res.read().splitlines()
    res.close()
    if service_result is not None and int(service_result[0]) != 100:
        raise SMSError(
            "send sms failed with returned: '{}'".format(
                servicecodes[int(service_result[0])]
            )
        )
