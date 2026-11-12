import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import yaml

from audit_dataset import audit, image_path, inspect_image, near_pairs, split_conflicts


class AuditTests(unittest.TestCase):
    def test_path_escape_and_windows_paths_are_rejected(self):
        for path in ("../escape.png", "/absolute.png", "C:/data.png", "folder\\file.png"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                image_path(Path("/tmp/dataset"), path)

    def test_symlinks_cannot_lead_outside_the_image_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "inside").mkdir()
            Image.new("RGB", (8, 8), "red").save(Path(outside) / "secret.png")
            (root / "inside" / "link.png").symlink_to(Path(outside) / "secret.png")
            (root / "linked_dir").symlink_to(outside, target_is_directory=True)
            (root / "inside" / "real.png").write_bytes(b"x")
            for path in ("inside/link.png", "linked_dir/secret.png"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    image_path(root, path)
            self.assertEqual(image_path(root, "inside/real.png"), (root / "inside" / "real.png").resolve())

    def test_a_directory_is_resolved_once_however_many_images_it_holds(self):
        # On a Windows-mounted drive every Path.resolve() is several slow lookups; resolving per image made
        # validating the 17,952-row manifest take 426 of the audit's 594 seconds.
        calls = []
        real_resolve = Path.resolve

        def counting(self, *args, **kwargs):
            calls.append(self)
            return real_resolve(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "class").mkdir()
            with mock.patch.object(Path, "resolve", counting):
                for number in range(500):
                    image_path(root, f"class/{number}.png")
            self.assertLessEqual(len(calls), 4)

    def test_leakage_detection_includes_duplicates_with_assigned_splits(self):
        rows = [{"sha256": "same", "split": "train"}, {"sha256": "same", "split": "test"},
                {"sha256": "excluded", "split": ""}]
        self.assertEqual(split_conflicts(rows, "sha256"), 1)

    def test_near_hash_threshold_and_empty_input(self):
        zero = "00" * 32
        one_bit = "01" + "00" * 31
        self.assertEqual(near_pairs([zero], [one_bit], 1), 1)
        self.assertEqual(near_pairs([zero], [one_bit], 0), 0)
        self.assertEqual(near_pairs([], [zero], 12), 0)

    def test_missing_corrupt_and_hash_mismatch_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(inspect_image(root, "missing.png", "x")["errors"], ["missing_file"])
            (root / "bad.png").write_bytes(b"not an image")
            errors = inspect_image(root, "bad.png", "x")["errors"]
            self.assertIn("sha256_mismatch", errors)
            self.assertIn("UnidentifiedImageError", errors)

    def test_real_files_and_cross_split_near_copy_are_audited_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32), "red").save(root / "red.png")
            Image.new("RGB", (32, 32), "blue").save(root / "blue.png")
            rows = []
            for name, split in (("red", "train"), ("blue", "test")):
                digest = hashlib.sha256((root / f"{name}.png").read_bytes()).hexdigest()
                info = inspect_image(root, f"{name}.png", digest)
                rows.append(dict(path=f"{name}.png", label="maize_healthy", crop="Maize", category="healthy",
                                 sha256=digest, dhash=info["dhash"], group=name, status="kept", split=split))
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            taxonomy = root / "taxonomy.yaml"
            taxonomy.write_text(yaml.safe_dump({"near_duplicate_distance": 12, "classes": {
                "maize_healthy": {"crop": "Maize", "category": "healthy"}}}))
            before = manifest.read_bytes()
            report = audit(manifest, root, taxonomy)
            self.assertEqual(report["file_error_counts"], {})
            self.assertEqual(report["cross_split_near_pairs"]["train_vs_test"], 1)
            self.assertEqual(report["kept_crop_counts"], {"Maize": 2})
            self.assertIn("license", report["provenance_fields_absent"])
            self.assertEqual(manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
