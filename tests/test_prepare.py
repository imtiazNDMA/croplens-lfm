import unittest

from prepare import clean_training_pool, near_mask

ZERO = "00" * 32
THREE = "07" + "00" * 31
FAR = "ff" * 32


def row(path, split, dhash, label="rice_blast", status="kept"):
    return {"path": path, "split": split, "dhash": dhash, "label": label, "crop": "Rice", "status": status}


class NearMaskTests(unittest.TestCase):
    def test_flags_each_hash_within_the_threshold_of_any_other(self):
        self.assertEqual(near_mask([ZERO, FAR], [THREE], 3).tolist(), [True, False])
        self.assertEqual(near_mask([ZERO], [THREE], 2).tolist(), [False])

    def test_empty_other_side_flags_nothing(self):
        self.assertEqual(near_mask([ZERO], [], 32).tolist(), [False])


class CleanPoolTests(unittest.TestCase):
    def test_removes_near_copies_of_val_and_test_and_augmented_names_and_keeps_the_rest(self):
        rows = [
            row("a/clean.jpg", "train", FAR),
            row("a/near_val.jpg", "train", ZERO),
            row("a/near_test.jpg", "train", "ff" * 31 + "f8"),
            row("a/rotation_1.jpg", "train", "0f" * 32),
            row("a/dropped.jpg", "train", FAR, status="duplicate"),
            row("a/v.jpg", "val", THREE),
            row("a/t.jpg", "test", "ff" * 32),
        ]
        # near_test shares a hash with FAR's neighbour; make test hash close to it, and keep clean.jpg far from both.
        rows[-1]["dhash"] = "ff" * 31 + "f0"
        rows[0]["dhash"] = "a5" * 32
        pool, removed = clean_training_pool(rows, threshold=8)
        self.assertEqual([r["path"] for r in pool], ["a/clean.jpg"])
        reasons = {r["path"]: r["reason"] for r in removed}
        self.assertEqual(reasons, {"a/near_val.jpg": "near_val_or_test", "a/near_test.jpg": "near_val_or_test",
                                   "a/rotation_1.jpg": "augmentation_named"})

    def test_never_modifies_val_or_test_and_ignores_non_kept_rows(self):
        rows = [row("v.jpg", "val", ZERO), row("t.jpg", "test", FAR), row("x.jpg", "train", ZERO, status="conflict")]
        pool, removed = clean_training_pool(rows, threshold=32)
        self.assertEqual((pool, removed), ([], []))

    def test_pool_has_no_pair_within_the_threshold_of_val_or_test(self):
        rows = [row(f"t{i}.jpg", "train", format(i * 37 % 256, "02x") * 32) for i in range(40)]
        rows += [row("v.jpg", "val", "10" * 32), row("e.jpg", "test", "c0" * 32)]
        pool, _ = clean_training_pool(rows, threshold=32)
        evaluation = [r["dhash"] for r in rows if r["split"] != "train"]
        self.assertFalse(near_mask([r["dhash"] for r in pool], evaluation, 32).any())


if __name__ == "__main__":
    unittest.main()
