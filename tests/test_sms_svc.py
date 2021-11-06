import unittest

from sms_svc import send_sms, SMSError

import config_personal


class SmsTest(unittest.TestCase):
    def test_send_me(self):
        error_text = ""
        try:
            send_sms("simple smsru unit_test",
                     to=config_personal.getval('sms', 'NUMBER1'))
        except SMSError as err:
            error_text = "{}".format(err)
        self.assertFalse(error_text)


if __name__ == '__main__':
    unittest.main()
