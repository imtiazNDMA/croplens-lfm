"""Local MLflow storage and condition-run artifacts; never copies source images."""

from pathlib import Path
import subprocess
import sys

import mlflow


def setup_tracking(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name("croplens-conditions")
    if experiment is None:
        experiment_id = mlflow.create_experiment("croplens-conditions", artifact_location=(directory / "artifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return mlflow.get_tracking_uri(), experiment_id


def log_report(report, adapter=None, log_losses=True):
    config = report["config"]
    mlflow.log_params({key: value for key, value in config.items() if key != "classes"})
    mlflow.log_params({"model": report["model"], "model_revision": report["revision"],
                       "classes_count": len(config["classes"]), "lora_alpha": config["rank"]})
    mlflow.log_params({f"data_{key}_sha256": value for key, value in report["input_hashes"].items()})
    for stage in ("untuned", "finetuned"):
        for key in ("crop_accuracy", "joint_accuracy", "joint_macro_f1", "condition_macro_f1", "invalid_answers", "seconds"):
            mlflow.log_metric(f"{stage}.{key}", report[stage][key])
        for label, recall in report[stage]["per_class_recall"].items():
            mlflow.log_metric(f"{stage}.recall.{label}", recall)
    for key in ("training_seconds", "elapsed_seconds", "peak_allocated_gib", "trainable_parameters"):
        mlflow.log_metric(key, report[key])
    if log_losses:
        for step, value in enumerate(report["losses"], 1):
            mlflow.log_metric("train.loss", value, step=step)
    mlflow.log_dict(report, "report.json")
    mlflow.log_dict(config, "config.json")
    if adapter is not None:
        mlflow.log_artifacts(str(adapter), artifact_path="adapter")


def log_environment():
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    mlflow.log_text(environment, "environment.txt")
    root = Path(__file__).parent
    mlflow.set_tags({"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                     "git_dirty": str(bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))).lower()})
    for name in ("finetune_conditions.py", "finetune_crops.py", "condition_search.py", "experiment_tracking.py", "program.md"):
        mlflow.log_artifact(str(Path(__file__).parent / name), artifact_path="source")
