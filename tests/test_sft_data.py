import json
import random
import unittest

from PIL import Image

from contract import parse_photo_check
import tempfile
from pathlib import Path

from sft_data import LazySftDataset, answer_for, degrade, make_examples


def rows(n):
    return [{"path": f"p/{i}.jpg", "crop": ("Rice", "Wheat")[i % 2], "label": "x"} for i in range(n)]


class SftDataTests(unittest.TestCase):
    def test_answers_are_valid_contract_text_with_the_right_flag(self):
        self.assertEqual(parse_photo_check(answer_for("Rice"))[0]["blurred"], False)
        self.assertTrue(parse_photo_check(answer_for("Rice", "blurred"))[0]["blurred"])
        self.assertTrue(parse_photo_check(answer_for("Wheat", "dark"))[0]["too_dark"])
        self.assertTrue(parse_photo_check(answer_for("Wheat", "coarse"))[0]["blurred"])

    def test_every_pool_image_has_one_clean_example_and_some_have_a_degraded_copy(self):
        examples = make_examples(rows(400), poor_fraction=0.1)
        clean = [e for e in examples if e["how"] is None]
        poor = [e for e in examples if e["how"]]
        self.assertEqual(sorted(e["path"] for e in clean), sorted(r["path"] for r in rows(400)))
        self.assertTrue(20 < len(poor) < 60)
        for e in poor:
            self.assertTrue(any(parse_photo_check(e["answer"])[0][f] for f in ("blurred", "too_dark")))
        for e in clean:
            self.assertFalse(any(parse_photo_check(e["answer"])[0][f] for f in ("blurred", "too_dark", "not_close_up", "several_leaves")))

    def test_deterministic_for_a_seed_and_different_for_another(self):
        self.assertEqual(make_examples(rows(100)), make_examples(rows(100)))
        self.assertNotEqual(make_examples(rows(100), seed=1), make_examples(rows(100), seed=2))

    def test_degradation_changes_pixels_and_repeats_for_the_same_seed(self):
        image = Image.effect_noise((64, 64), 60).convert("RGB")
        for how in ("blurred", "dark", "coarse"):
            a, b = degrade(image, how, random.Random(5)), degrade(image, how, random.Random(5))
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertNotEqual(a.tobytes(), image.tobytes())
        with self.assertRaises(ValueError):
            degrade(image, "sepia", random.Random(1))


class LazyDatasetTests(unittest.TestCase):
    def test_item_is_a_two_turn_chat_with_a_capped_image_and_the_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.new("RGB", (2000, 1000), "green").save(Path(folder) / "a.jpg")
            examples = [{"path": "a.jpg", "how": "dark", "seed": 7, "answer": answer_for("Rice", "dark")}]
            item = LazySftDataset(examples, folder, max_side=256)[0]
            self.assertEqual(len(LazySftDataset(examples, folder)), 1)
        user, assistant = item["messages"]
        image = user["content"][0]["image"]
        self.assertEqual(max(image.size), 256)
        self.assertLess(sum(image.getpixel((5, 5))), 60)  # darkened
        self.assertEqual(assistant["content"][0]["text"], answer_for("Rice", "dark"))

    def test_hf_dataset_decodes_each_row_only_when_read(self):
        from datasets import Dataset
        from sft_data import as_hf_dataset
        with tempfile.TemporaryDirectory() as folder:
            Image.new("RGB", (300, 300), "green").save(Path(folder) / "a.jpg")
            examples = [{"path": "a.jpg", "how": None, "answer": answer_for("Rice")},
                        {"path": "missing.jpg", "how": None, "answer": answer_for("Wheat")}]
            table = as_hf_dataset(examples, folder, max_side=128)
            self.assertIsInstance(table, Dataset)
            self.assertEqual(len(table), 2)
            row = table[0]  # the missing second image is never opened
            self.assertEqual(max(row["messages"][0]["content"][0]["image"].size), 128)
            with self.assertRaises(FileNotFoundError):
                table[1]


if __name__ == "__main__":
    unittest.main()
