import unittest


from set_naming import get_names


class ScoreTest(unittest.TestCase):
    def test_set_names(self):
        from score import Score

        sc = Score("6-4 3-6 7-6(5)")
        self.assertEqual(get_names(sc, setnum=1), ("open", "open"))
        self.assertEqual(get_names(sc, setnum=2), ("press", "under"))
        self.assertEqual(get_names(sc, setnum=3), ("decided", "decided"))

        sc = Score("6-4 3-6 7-5 1-6 7-6(5)")
        self.assertEqual(get_names(sc, setnum=1), ("open", "open"))
        self.assertEqual(get_names(sc, setnum=2), ("press", "under"))
        self.assertEqual(get_names(sc, setnum=3), ("open2", "open2"))
        self.assertEqual(get_names(sc, setnum=4), ("press", "under"))
        self.assertEqual(get_names(sc, setnum=5), ("decided", "decided"))

        sc = Score("6-4 6-3 5-7 1-6 7-6(5)")
        self.assertEqual(get_names(sc, setnum=1), ("open", "open"))
        self.assertEqual(get_names(sc, setnum=2), ("press", "under"))
        self.assertEqual(get_names(sc, setnum=3), ("press2", "under2"))
        self.assertEqual(get_names(sc, setnum=4), ("press", "under"))
        self.assertEqual(get_names(sc, setnum=5), ("decided", "decided"))

        sc = Score("4-6 3-6 7-5 6-1 7-6(5)")
        self.assertEqual(get_names(sc, setnum=1), ("open", "open"))
        self.assertEqual(get_names(sc, setnum=2), ("under", "press"))
        self.assertEqual(get_names(sc, setnum=3), ("under2", "press2"))
        self.assertEqual(get_names(sc, setnum=4), ("under", "press"))
        self.assertEqual(get_names(sc, setnum=5), ("decided", "decided"))


if __name__ == '__main__':
    unittest.main()
