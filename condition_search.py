"""Execute the agent's short condition-search proposals with MLflow and hard deadlines."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import mlflow

from experiment_tracking import log_environment, log_report, setup_tracking


HERE = Path(__file__).parent
PROPOSALS = [
    ("lr-low", "learning_rate", 1e-4, "Lower LR may generalize better on weak condition classes"),
    ("lr-high", "learning_rate", 4e-4, "Higher LR may learn weak classes within the fixed 400 steps"),
    ("rank-16", "rank", 16, "Larger adapter may separate visually similar conditions"),
]


def candidate_config(reference, key, value):
    if key not in {"learning_rate", "rank"} or not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError("Only positive learning-rate/rank changes are allowed")
    if key == "rank" and not isinstance(value, int):
        raise ValueError("Rank must be an integer")
    if reference[key] == value:
        raise ValueError("Candidate must actually change the reference")
    return {**reference, key: value}


def verified_metrics(report, reference, config):
    from finetune_conditions import parse_answer, score
    if report["status"] != "complete" or report.get("completed_steps") != config["steps"]:
        raise ValueError("Incomplete trial")
    if (report["config"] != config or report["input_hashes"] != reference["input_hashes"]
            or report["question"] != reference["question"] or report["revision"] != reference["revision"]):
        raise ValueError("Task/data/config changed")
    gold = {row["sha256"]: row["truth"] for row in reference["finetuned"]["answers"]}
    answers = report["finetuned"]["answers"]
    if len(answers) != len(gold) or {row["sha256"] for row in answers} != set(gold):
        raise ValueError("Validation images changed or repeated")
    truth = [gold[row["sha256"]] for row in answers]
    predictions = [parse_answer(row["raw"], config["classes"]) for row in answers]
    if truth != [row["truth"] for row in answers] or predictions != [row["prediction"] for row in answers]:
        raise ValueError("Reported labels/predictions do not match gold/raw output")
    metrics = score(truth, predictions, config["classes"])
    for key in ("joint_macro_f1", "joint_accuracy", "crop_accuracy", "condition_macro_f1", "invalid_answers"):
        if abs(metrics[key] - report["finetuned"][key]) > 1e-9:
            raise ValueError(f"Stored metric disagrees with recomputed {key}")
    return metrics


def decision(metrics, best_f1, reference_metrics):
    return "keep" if (metrics["joint_macro_f1"] >= best_f1 + 0.005 and metrics["invalid_answers"] == 0
                       and metrics["crop_accuracy"] >= reference_metrics["crop_accuracy"] - 0.01) else "discard"


def completed_worker(path):
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("Cannot resume over an active or incomplete worker; review it first")
    return report


def run_bounded(command, seconds, log):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
        try:
            return process.wait(timeout=seconds)
        except BaseException as error:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                return 124
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pool", "images", "output", "summary"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=HERE / "reports" / "condition-pilot-001.json")
    parser.add_argument("--tracking-dir", type=Path, default=Path.home() / "croplens-mlflow")
    parser.add_argument("--resume", action="store_true", help="Reconcile completed workers and continue the same bounded session")
    args = parser.parse_args()
    if (args.summary.exists() and not args.resume) or not args.summary.parent.is_dir():
        parser.error("Summary must be new and its parent must exist")
    if args.resume and not args.summary.exists():
        parser.error("Resume requires an existing summary")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["status"] != "complete":
        parser.error("A complete reference is required")
    protected = [args.manifest, args.pool, args.reference, HERE / "configs/condition-pilot.json",
                 HERE / "finetune_conditions.py", HERE / "finetune_crops.py", HERE / "program.md"]

    def fingerprints():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}

    fixed = fingerprints()
    uri, _ = setup_tracking(args.tracking_dir)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    started = time.monotonic()
    best_f1 = reference["finetuned"]["joint_macro_f1"]
    best = {"trial": "condition-pilot-001", "joint_macro_f1": best_f1, "adapter": "outputs/condition-pilot-001/adapter"}
    summary = {"status": "running", "tracking_uri": uri, "reference_joint_macro_f1": best_f1,
               "limits": {"trials": 3, "trial_seconds": 600, "training_seconds": 300, "session_seconds": 1800},
               "protected_sha256": fixed, "trials": [], "best": best}
    if args.resume:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if summary["protected_sha256"] != fixed or summary["tracking_uri"] != uri or summary["status"] == "complete":
            parser.error("Cannot resume changed inputs/storage or an already completed session")
        existing_session = mlflow.get_run(summary["session_run_id"])
        started = time.monotonic() - max(0, time.time() - existing_session.info.start_time / 1000)
        best_f1 = summary["best"]["joint_macro_f1"]

    def save():
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def stop(signum, frame):
        raise TimeoutError("External session deadline reached")

    signal.signal(signal.SIGTERM, stop)
    with mlflow.start_run(run_id=summary.get("session_run_id"), run_name=args.output.name) as session:
        summary["session_run_id"] = session.info.run_id
        if not args.resume:
            log_environment()
            mlflow.log_dict(fixed, "protected-inputs.json")
            with mlflow.start_run(run_name="reference-historical", nested=True):
                mlflow.set_tags({"role": "historical_reference", "decision": "reference"})
                log_report(reference, HERE / "outputs/condition-pilot-001/adapter")
        else:
            mlflow.log_artifact(__file__, artifact_path="resume-source")
        failures = 0
        for entry in reversed(summary["trials"]):
            if entry["status"] != "failed":
                break
            failures += 1
        try:
            for name, key, value, hypothesis in PROPOSALS:
                if name in {entry["trial"] for entry in summary["trials"]}:
                    continue
                remaining = 1800 - (time.monotonic() - started)
                if remaining < 30 or failures >= 2:
                    summary["stop_reason"] = "session deadline" if remaining < 30 else "two consecutive failures"
                    break
                if fingerprints() != fixed:
                    raise ValueError("Protected source/data changed")
                config = candidate_config(reference["config"], key, value)
                trial_dir = args.output / name
                trial_dir.mkdir(exist_ok=args.resume)
                config_file, report_file = trial_dir / "config.json", trial_dir / "report.json"
                recovered = completed_worker(report_file)
                if recovered is None:
                    config_file.write_text(json.dumps(config, indent=2) + "\n")
                entry = {"trial": name, "hypothesis": hypothesis, "change": {key: value}, "status": "running"}
                print(f"TRIAL {name}: {hypothesis}", flush=True)
                try:
                    with mlflow.start_run(run_id=recovered.get("mlflow_run_id") if recovered else None, run_name=name, nested=True) as run:
                        entry["mlflow_run_id"] = run.info.run_id
                        mlflow.set_tags({"hypothesis": hypothesis, "changed_parameter": key})
                        mlflow.log_dict(config, "proposed-config.json")
                        command = [sys.executable, "finetune_conditions.py", "--manifest", str(args.manifest),
                            "--pool", str(args.pool), "--images", str(args.images), "--output", str(trial_dir / "training"),
                            "--report", str(report_file), "--config", str(config_file), "--reference", str(args.reference),
                            "--mlflow-run-id", run.info.run_id, "--training-seconds", "300"]
                        if recovered:
                            if recovered["status"] != "complete":
                                raise RuntimeError("Interrupted incomplete worker cannot be promoted or overwritten")
                            code = 0
                            entry["reconciled_after_supervisor_interruption"] = True
                        else:
                            code = run_bounded(command, min(600, remaining), trial_dir / "run.log")
                        if code != 0 or fingerprints() != fixed:
                            raise RuntimeError(f"Trial exit {code}, or protected input changed")
                        report = json.loads(report_file.read_text(encoding="utf-8"))
                        metrics = verified_metrics(report, reference, config)
                        choice = decision(metrics, best_f1, reference["finetuned"])
                        entry.update(status="complete", decision=choice, joint_macro_f1=metrics["joint_macro_f1"],
                                     joint_accuracy=metrics["joint_accuracy"], crop_accuracy=metrics["crop_accuracy"],
                                     training_seconds=report["training_seconds"], peak_allocated_gib=report["peak_allocated_gib"])
                        mlflow.set_tags({"decision": choice, "evaluation_verified": "true"})
                        if choice == "keep":
                            best_f1 = metrics["joint_macro_f1"]
                            summary["best"] = {"trial": name, "joint_macro_f1": best_f1, "adapter": str(trial_dir / "training/adapter"),
                                               "mlflow_run_id": run.info.run_id}
                        failures = 0
                except Exception as error:
                    failures += 1
                    entry.update(status="failed", decision="failed", error=f"{type(error).__name__}: {error}")
                summary["trials"].append(entry)
                save()
                print(json.dumps(entry), flush=True)
            summary["status"] = "complete"
            save()
            mlflow.log_dict(summary, "summary.json")
            mlflow.log_metric("best_joint_macro_f1", summary["best"]["joint_macro_f1"])
        except Exception as error:
            summary.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            raise
    print(f"BEST: {json.dumps(summary['best'])}", flush=True)


if __name__ == "__main__":
    main()
