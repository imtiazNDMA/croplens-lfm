import unittest

from size_report import accuracy


class AccuracyTests(unittest.TestCase):
    def test_all_and_subset(self):
        found = {"a": True, "b": False, "c": True, "d": True}
        self.assertEqual(accuracy(found), {"n": 4, "right": 3, "accuracy": 0.75})
        self.assertEqual(accuracy(found, {"b", "c"}), {"n": 2, "right": 1, "accuracy": 0.5})

    def test_empty_subset_has_no_accuracy_not_a_zero(self):
        self.assertIsNone(accuracy({"a": True}, {"zzz"})["accuracy"])


if __name__ == "__main__":
    unittest.main()
