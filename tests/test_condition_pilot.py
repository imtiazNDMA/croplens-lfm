import json
import unittest

from finetune_conditions import answer_for, parse_answer, score, select_rows


LABELS = ["maize_healthy", "maize_common_rust"]


class ConditionPilotTests(unittest.TestCase):
    def rows(self):
        return [dict(path=f"{split}/{label}.jpg", label=label, crop="Maize", status="kept", split=split,
                     sha256=f"{split}/{label}", group=f"{split}/{label}")
                for split in ("train", "val", "test") for label in LABELS]

    def config(self):
        return dict(classes=LABELS, train_per_class=1, val_per_class=1, seed=3407)

    def test_targets_roundtrip_and_invalid_pairs_are_not_rescued(self):
        for label in LABELS:
            self.assertEqual(parse_answer(json.dumps(answer_for(label)), LABELS), label)
        for value in ('{"crop":"Wheat","condition":"common_rust"}',
                      '{"crop":"Maize","condition":"healthy","confidence":1}', 'healthy'):
            self.assertEqual(parse_answer(value, LABELS), "INVALID")

    def test_selection_is_balanced_and_uses_only_cleaned_training_rows(self):
        rows = self.rows()
        selected = select_rows(rows[:2], rows, self.config())
        self.assertEqual(len(selected["train"]), 2)
        self.assertTrue(all(row["split"] == "train" for row in selected["train"]))
        self.assertTrue(all(row["split"] == "val" for row in selected["val"]))

    def test_validation_rows_cannot_enter_training_pool(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select_rows(rows[2:4], rows, self.config())

    def test_healthy_coverage_and_heldout_overlap_are_required(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select_rows(rows[:2], rows, {**self.config(), "classes": ["maize_common_rust"]})
        rows[2]["sha256"] = rows[0]["sha256"]
        with self.assertRaises(ValueError):
            select_rows(rows[:2], rows, self.config())

    def test_invalid_answers_fail_all_metrics_and_wrong_condition_is_not_jointly_correct(self):
        metrics = score(LABELS, ["INVALID", "INVALID"], LABELS)
        self.assertEqual(metrics["joint_accuracy"], 0)
        self.assertEqual(metrics["condition_macro_f1"], 0)
        metrics = score(LABELS, list(reversed(LABELS)), LABELS)
        self.assertEqual(metrics["crop_accuracy"], 1)
        self.assertEqual(metrics["joint_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
