import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from condition_search import candidate_config, completed_worker, decision, run_bounded, verified_metrics
from finetune_conditions import answer_for, score


class ConditionSearchTests(unittest.TestCase):
    def fixture(self):
        labels = ["maize_healthy", "maize_common_rust"]
        config = {"classes": labels, "steps": 400, "rank": 8, "learning_rate": 2e-4}
        answers = [{"sha256": str(i), "truth": label, "prediction": label, "raw": json.dumps(answer_for(label))}
                   for i, label in enumerate(labels)]
        report = {"status": "complete", "completed_steps": 400, "config": config, "input_hashes": {"pool": "same"},
                  "revision": "same", "question": "same", "finetuned": {**score(labels, labels, labels), "answers": answers}}
        return report, copy.deepcopy(report), config

    def test_only_one_allowlisted_positive_parameter_changes(self):
        base = {"rank": 8, "learning_rate": 2e-4, "steps": 400}
        self.assertEqual(candidate_config(base, "rank", 16), {**base, "rank": 16})
        self.assertEqual(base["rank"], 8)
        for key, value in (("steps", 800), ("rank", 0), ("rank", 1.5), ("rank", True), ("rank", 8)):
            with self.assertRaises(ValueError):
                candidate_config(base, key, value)

    def test_verification_rejects_incomplete_and_changed_gold_or_metrics(self):
        report, reference, config = self.fixture()
        self.assertEqual(verified_metrics(report, reference, config)["joint_macro_f1"], 1)
        for field, value in (("status", "timed_out"), ("completed_steps", 399), ("revision", "different")):
            with self.assertRaises(ValueError):
                verified_metrics({**report, field: value}, reference, config)
        report["finetuned"]["answers"][0]["truth"] = "maize_common_rust"
        with self.assertRaises(ValueError):
            verified_metrics(report, reference, config)
        report, reference, config = self.fixture()
        report["finetuned"]["joint_macro_f1"] = 0.2
        with self.assertRaises(ValueError):
            verified_metrics(report, reference, config)

    def test_keep_needs_margin_and_no_quality_regression(self):
        reference = {"crop_accuracy": 0.99}
        metrics = {"joint_macro_f1": 0.89, "crop_accuracy": 0.99, "invalid_answers": 0}
        self.assertEqual(decision(metrics, 0.878, reference), "keep")
        self.assertEqual(decision(metrics, 0.889, reference), "discard")
        self.assertEqual(decision({**metrics, "invalid_answers": 1}, 0.878, reference), "discard")
        self.assertEqual(decision({**metrics, "crop_accuracy": 0.95}, 0.878, reference), "discard")

    def test_process_deadline_terminates_a_slow_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            code = run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], 0.05, Path(folder) / "run.log")
            self.assertEqual(code, 124)

    def test_resume_only_adopts_completed_workers_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            self.assertIsNone(completed_worker(path))
            path.write_text(json.dumps({"status": "running"}))
            with self.assertRaises(ValueError):
                completed_worker(path)
            path.write_text(json.dumps({"status": "complete", "completed_steps": 400}))
            original = path.read_bytes()
            self.assertEqual(completed_worker(path)["completed_steps"], 400)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
