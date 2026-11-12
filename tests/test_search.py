import json
import unittest
from pathlib import Path

from run_search import decide, pending, settings_for, should_stop
from sft_data import balanced_sample

LIMITS = {"stop_after_consecutive_failures": 2, "session_seconds": 7200}
REFERENCE = {"lr": 2e-4, "rank": 16, "balance": False, "steps": 150}


class SearchLogicTests(unittest.TestCase):
    def test_a_trial_changes_exactly_one_known_setting(self):
        self.assertEqual(settings_for(REFERENCE, {"id": "t", "change": {"rank": 8}})["rank"], 8)
        with self.assertRaises(ValueError):
            settings_for(REFERENCE, {"id": "t", "change": {"rank": 8, "lr": 1e-4}})
        with self.assertRaises(ValueError):
            settings_for(REFERENCE, {"id": "t", "change": {"dropout": 0.1}})

    def test_keep_needs_the_margin(self):
        self.assertEqual(decide(0.83, 0.84, 0.01), "keep")
        self.assertEqual(decide(0.83, 0.835, 0.01), "discard")
        self.assertEqual(decide(0.83, 0.80, 0.01), "discard")

    def test_finished_trials_are_not_repeated_and_the_session_is_capped(self):
        trials = [{"id": c} for c in "abcdefgh"]
        self.assertEqual([t["id"] for t in pending(trials, {"a", "b"}, 3)], ["c", "d", "e"])

    def test_stop_conditions(self):
        self.assertIsNone(should_stop(1, 0, 100, LIMITS))
        self.assertEqual(should_stop(2, 0, 100, LIMITS), "consecutive failures")
        self.assertEqual(should_stop(0, 0, 7200, LIMITS), "session time limit")

    def test_shipped_search_config_is_valid_and_single_change(self):
        config = json.loads((Path(__file__).parent.parent / "configs" / "search.json").read_text(encoding="utf-8"))
        self.assertLessEqual(len(config["trials"]), config["limits"]["max_trials_per_session"])
        for trial in config["trials"]:
            settings_for(config["reference"], trial)


class BalancedSampleTests(unittest.TestCase):
    def test_each_crop_is_about_equally_likely_whatever_the_pool_looks_like(self):
        pool = [{"crop": "Maize", "path": f"m{i}"} for i in range(900)] + [{"crop": "Wheat", "path": f"w{i}"} for i in range(100)]
        sample = balanced_sample(pool, 2000, seed=1)
        self.assertEqual(len(sample), 2000)
        wheat = sum(e["crop"] == "Wheat" for e in sample) / 2000
        self.assertTrue(0.45 < wheat < 0.55)
        self.assertEqual(balanced_sample(pool, 50, 1), balanced_sample(pool, 50, 1))


if __name__ == "__main__":
    unittest.main()
