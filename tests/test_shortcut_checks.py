import unittest

from shortcut_checks import min_distances, near_pair_counts, size_only_crop_accuracy


class ShortcutTests(unittest.TestCase):
    def test_min_distance_is_the_closest_other_split_image(self):
        zero = "00" * 32
        three_bits = "07" + "00" * 31
        far = "ff" * 32
        self.assertEqual(min_distances([zero, far], [three_bits, far]).tolist(), [3, 0])
        self.assertEqual(min_distances([], [zero]).tolist(), [])
        self.assertEqual(min_distances([zero], []).tolist(), [])

    def test_pair_counts_grow_with_the_threshold(self):
        zero = "00" * 32
        hashes = {"train": [zero], "val": ["0f" + "00" * 31], "test": ["ff" * 32]}
        counts = near_pair_counts(hashes, (3, 4, 256))
        self.assertEqual(counts["train_vs_val"], {"3": 0, "4": 1, "256": 1})
        self.assertEqual(counts["train_vs_test"], {"3": 0, "4": 0, "256": 1})

    def test_size_alone_predicts_a_crop_only_where_sizes_are_unique_to_a_crop(self):
        train = [("256x256", "Rice"), ("256x256", "Rice"), ("400x400", "Rice"), ("300x300", "Maize"), ("300x300", "Maize")]
        later = [("256x256", "Rice"), ("300x300", "Rice"), ("999x1", "Maize")]
        report = size_only_crop_accuracy(train, later)
        # 256x256 -> Rice (right); 300x300 -> Maize (wrong); 999x1 never seen -> the overall majority, Rice (wrong).
        self.assertAlmostEqual(report["accuracy"], 1 / 3)
        self.assertAlmostEqual(report["size_seen_in_training"], 2 / 3)
        self.assertAlmostEqual(report["accuracy_where_size_seen"], 1 / 2)
        self.assertAlmostEqual(report["majority_crop_baseline"], 2 / 3)  # always saying Rice


if __name__ == "__main__":
    unittest.main()
