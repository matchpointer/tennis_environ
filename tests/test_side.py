import unittest

from side import Side


class TestSimple(unittest.TestCase):
    def test_simple(self):
        left = Side("LEFT")
        left2 = Side("LEFT")
        right = Side("RIGHT")
        self.assertEqual(left, "LEFT")
        self.assertTrue(left.is_left())
        self.assertFalse(left.is_right())
        self.assertTrue(left.is_oppose(right))
        self.assertTrue(left.is_oppose(left.fliped()))
        self.assertFalse(left.is_oppose(left2))
        self.assertTrue(left != right)
        self.assertTrue(left == left2)
        self.assertFalse(left2.fliped().is_left())


if __name__ == '__main__':
    unittest.main()
