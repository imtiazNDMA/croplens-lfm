"""Score a recorded run (this project's run_photo_check.py output, or the portal's Qwen readings file) on one part
of the photo-check set. Writes a new JSON report with strict and lenient scores side by side."""

import argparse
import json
from pathlib import Path

from evaluate import load_cases, score


def read_answers(path):
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if "readings" in body:  # the portal's format: objects, always valid JSON
        return {key: json.dumps(value) for key, value in body["readings"].items()}, body["model"], {}
    seconds = list(body.get("seconds", {}).values())
    return body["answers"], body["model"] + (" + adapter" if body.get("adapter") else " (untuned)"), {
        "seconds_per_photo_mean": sum(seconds) / len(seconds) if seconds else None,
        "peak_vram_gib": body.get("peak_vram_gib")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--part", choices=("tune", "final"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [c for c in load_cases(args.cases) if c["part"] == args.part]
    answers, model, extra = read_answers(args.run)
    report = {"model": model, "part": args.part, "run_file": args.run.name, **extra,
              "strict": score(cases, answers), "lenient": score(cases, answers, lenient=True)}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for mode in ("strict", "lenient"):
        r = report[mode]
        print(f"{model} [{mode}] crop-ID {r['crop_identification']:.1%}  unclear {r['unclear_share']:.1%}  false stops {r['false_stops']:.1%}  "
              f"non-plant stopped {r['non_plant_stopped']:.1%}  unsupported stopped {r['unsupported_plant_stopped']:.1%}  invalid {r['invalid_share']:.1%}")


if __name__ == "__main__":
    main()
