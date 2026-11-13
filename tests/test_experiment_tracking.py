from pathlib import Path
import tempfile
import unittest

import mlflow
from mlflow import MlflowClient

from experiment_tracking import log_report, setup_tracking


class TrackingTests(unittest.TestCase):
    def test_metrics_history_and_artifacts_persist_to_local_sqlite(self):
        previous = mlflow.get_tracking_uri()
        try:
            with tempfile.TemporaryDirectory() as folder:
                uri, experiment_id = setup_tracking(Path(folder))
                stage = {"crop_accuracy": 1, "joint_accuracy": 0.5, "joint_macro_f1": 0.4,
                         "condition_macro_f1": 0.3, "invalid_answers": 0, "seconds": 1,
                         "per_class_recall": {"maize_healthy": 0.5}}
                report = {"config": {"classes": ["maize_healthy"], "rank": 8, "seed": 3407}, "model": "fixture",
                          "revision": "fixture", "input_hashes": {"pool": "fixture"}, "untuned": stage, "finetuned": stage,
                          "training_seconds": 1, "elapsed_seconds": 2, "peak_allocated_gib": 1, "trainable_parameters": 8,
                          "losses": [1.0, 0.5]}
                with mlflow.start_run(experiment_id=experiment_id) as run:
                    run_id = run.info.run_id
                    log_report(report)
                    log_report(report, log_losses=False)
                client = MlflowClient(tracking_uri=uri)
                saved = client.get_run(run_id)
                self.assertEqual(saved.info.status, "FINISHED")
                self.assertEqual(saved.data.metrics["finetuned.joint_macro_f1"], 0.4)
                self.assertEqual(len(client.get_metric_history(run_id, "train.loss")), 2)
                self.assertIn("report.json", {artifact.path for artifact in client.list_artifacts(run_id)})
        finally:
            mlflow.set_tracking_uri(previous)


if __name__ == "__main__":
    unittest.main()
