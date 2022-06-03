import unittest
import os

from file_utils import is_non_zero_file


class IsNoneZeroFileTestCase(unittest.TestCase):
    def test_damaged_file(self):
        damaged_filename = r'.\unit_test_data\damaged_files\file1'
        exist = os.path.isfile(damaged_filename)
        self.assertTrue(exist)
        if exist:
            is_non_zero = is_non_zero_file(damaged_filename)
            self.assertEqual(is_non_zero, False)

    def test_ok_file(self):
        filename = r'.\unit_test_data\json_ok_files\file1'
        exist = os.path.isfile(filename)
        self.assertTrue(exist)
        if exist:
            is_non_zero = is_non_zero_file(filename)
            self.assertEqual(is_non_zero, True)


if __name__ == '__main__':
    unittest.main()
