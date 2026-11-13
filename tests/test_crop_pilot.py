import unittest
from pathlib import Path
import tempfile

from PIL import Image

from finetune_crops import CROPS, load_image, score, select_rows


class CropPilotTests(unittest.TestCase):
    def rows(self):
        return [dict(status="kept", split=split, crop=crop, path=f"{split}/{crop}",
                     sha256=f"{split}/{crop}", group=f"{split}/{crop}")
                for split in ("train", "val", "test") for crop in CROPS]

    def test_selection_balances_and_excludes_test(self):
        selected = select_rows(self.rows(), 1, 1)
        self.assertEqual(len(selected["train"]), 5)
        self.assertEqual(len(selected["val"]), 5)
        self.assertTrue(all(row["split"] == "val" for row in selected["val"]))

    def test_held_out_overlap_is_rejected(self):
        rows = self.rows()
        rows[5]["sha256"] = rows[0]["sha256"]
        with self.assertRaises(ValueError):
            select_rows(rows, 1, 1)

    def test_invalid_outputs_count_as_failures(self):
        metrics = score(CROPS, ["INVALID"] * 5)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["macro_f1"], 0)
        self.assertEqual(metrics["invalid_answers"], 5)

    def test_orientation_is_applied_without_retaining_source_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oriented.jpg"
            image = Image.new("RGB", (40, 20), "red")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            loaded = load_image(path)
            self.assertEqual(loaded.size, (20, 40))
            self.assertEqual(loaded.info, {})


if __name__ == "__main__":
    unittest.main()
