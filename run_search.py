"""One bounded search session: train and score a few single-change trials from the reference run, one at a time.
Started by a person; it never restarts itself. Each step runs under a hard timeout. Selection uses the size-normalised
crop photos of the tune part only; the final part and the test split are never read here. Every attempt, including a
failed or discarded one, is appended to the ledger."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate import load_cases, score

HERE = Path(__file__).parent


def settings_for(reference, trial):
    """The reference settings with the trial's single change applied. A trial changing two settings is refused."""
    if len(trial["change"]) != 1:
        raise ValueError(f"Trial {trial['id']} changes {len(trial['change'])} settings; each trial changes exactly one")
    unknown = set(trial["change"]) - set(reference)
    if unknown:
        raise ValueError(f"Trial {trial['id']} changes unknown settings: {sorted(unknown)}")
    return {**reference, **trial["change"]}


def decide(best, value, margin):
    """keep when the score beats the best so far by at least `margin`; discard otherwise."""
    return "keep" if value >= best + margin else "discard"


def pending(trials, ledger_ids, limit):
    return [t for t in trials if t["id"] not in ledger_ids][:limit]


def should_stop(consecutive_failures, started, now, limits):
    if consecutive_failures >= limits["stop_after_consecutive_failures"]:
        return "consecutive failures"
    if now - started >= limits["session_seconds"]:
        return "session time limit"
    return None


def read_ledger(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def append(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


def run(command, seconds, log):
    with log.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds, cwd=HERE).returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def run_trial(trial, settings, args, limits):
    out = args.runs / trial["id"]
    started = time.monotonic()
    train = [sys.executable, "train.py", "--examples", str(args.examples), "--images", str(args.crop_data), "--output", str(out),
             "--steps", str(settings["steps"]), "--batch", str(settings["batch"]), "--accum", str(settings["accum"]),
             "--lr", str(settings["lr"]), "--rank", str(settings["rank"]), "--seed", str(settings["seed"]),
             "--max-seconds", str(limits["train_seconds"] - 60)] + (["--balance"] if settings["balance"] else [])
    args.runs.mkdir(parents=True, exist_ok=True)
    code = run(train, limits["train_seconds"], args.runs / f"{trial['id']}-train.log")
    report = out / "report.json"
    if code != 0 or not report.exists():
        return {"status": f"train_failed:{code}"}
    training = json.loads(report.read_text(encoding="utf-8"))
    if training["status"] != "complete":
        return {"status": "train_incomplete_time_capped", "training": {k: training[k] for k in ("completed_steps", "train_seconds")}}
    answers_file = args.runs / f"{trial['id']}-normalized-crop.json"
    ask = [sys.executable, "run_photo_check.py", "--cases", str(args.normalized_cases), "--part", "tune", "--groups", "crop",
           "--crop-data", str(args.normalized_dir), "--extra-images", str(args.normalized_dir), "--adapter", str(out / "adapter"),
           "--prompt", "short", "--max-side", "512", "--output", str(answers_file), "--max-seconds", str(limits["eval_seconds"] - 90)]
    code = run(ask, limits["eval_seconds"], args.runs / f"{trial['id']}-eval.log")
    if code != 0:
        return {"status": f"eval_failed:{code}"}
    cases = [c for c in load_cases(args.normalized_cases) if c["part"] == "tune" and c["group"] == "crop"]
    answers = json.loads(answers_file.read_text(encoding="utf-8"))["answers"]
    if len(answers) < len(cases):
        return {"status": "eval_incomplete_time_capped", "answered": len(answers)}
    result = score(cases, answers)
    if time.monotonic() - started > limits["trial_seconds"]:
        return {"status": "over_trial_limit", "seconds": round(time.monotonic() - started)}
    return {"status": "complete", "crop_identification": result["crop_identification"], "invalid_share": result["invalid_share"],
            "per_crop": result["per_crop"], "train_seconds": training["train_seconds"], "peak_vram_gib": training["peak_vram_gib"],
            "final_loss": training["losses"][-1], "trial_seconds": round(time.monotonic() - started)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "search.json")
    parser.add_argument("--ledger", type=Path, default=HERE / "reports" / "search" / "ledger.jsonl")
    parser.add_argument("--runs", type=Path, default=HERE / "outputs" / "search")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--normalized-cases", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    limits, reference = config["limits"], config["reference"]
    ledger = read_ledger(args.ledger)
    if not ledger:
        append(args.ledger, {"id": reference["id"], "status": "complete", "role": "reference", "decision": "reference",
                             "crop_identification": reference["normalized_crop_identification"], "settings": {
                                 k: reference[k] for k in ("steps", "batch", "accum", "lr", "rank", "balance", "seed")}})
        ledger = read_ledger(args.ledger)
    best = max(e["crop_identification"] for e in ledger if e["status"] == "complete")
    best_id = next(e["id"] for e in ledger if e["status"] == "complete" and e["crop_identification"] == best)
    started, failures, done = time.monotonic(), 0, 0
    for trial in pending(config["trials"], {e["id"] for e in ledger}, limits["max_trials_per_session"]):
        reason = should_stop(failures, started, time.monotonic(), limits)
        if reason:
            print(f"Stopping: {reason}", flush=True)
            break
        settings = settings_for(reference, trial)
        print(f"Trial {trial['id']}: {trial['hypothesis']}", flush=True)
        result = run_trial(trial, settings, args, limits)
        entry = {"id": trial["id"], "hypothesis": trial["hypothesis"], "reference": reference["id"], "settings": settings,
                 "change": trial["change"], **result}
        if result["status"] == "complete":
            failures = 0
            entry["decision"] = decide(best, result["crop_identification"], config["improvement_margin"])
            if entry["decision"] == "keep":
                best, best_id = result["crop_identification"], trial["id"]
        else:
            failures += 1
            entry["decision"] = "failed"
        entry["best_so_far"] = {"id": best_id, "crop_identification": best}
        append(args.ledger, entry)
        done += 1
        print(json.dumps({k: entry.get(k) for k in ("id", "status", "crop_identification", "decision")}), flush=True)
    print(f"Session finished: {done} trials, best {best_id} at {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
