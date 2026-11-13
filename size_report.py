"""Crop identification for each system on three views of the same photos: as they are, size-normalised, and the
size-ambiguous subset of either. Reads recorded answers; runs no model."""

import argparse
import json
from pathlib import Path

from evaluate import load_cases, outcome


def accuracy(correct_by_id, ids=None):
    """Share of photos named correctly, with the count. `ids` limits it to a subset."""
    chosen = [v for k, v in correct_by_id.items() if ids is None or k in ids]
    return {"n": len(chosen), "right": sum(chosen), "accuracy": sum(chosen) / len(chosen) if chosen else None}


def lfm_correct(run_path, cases):
    answers = json.loads(Path(run_path).read_text(encoding="utf-8"))["answers"]
    return {c["id"]: (lambda o: o["kind"] == "crop" and o["crop"] == c["crop"])(outcome(answers[c["id"]], lenient=True))
            for c in cases if c["group"] == "crop"}


def convnext_correct(path):
    return {r["id"]: r["inferred_crop"] == r["crop"] for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="the original photo-check manifest")
    parser.add_argument("--part", default="tune")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True,
                        help="name=kind:original_file:normalized_file, kind lfm or convnext")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [c for c in load_cases(args.cases) if c["part"] == args.part]
    ids = set(json.loads(args.subset.read_text(encoding="utf-8"))["ids"])
    report = {"part": args.part, "size_ambiguous_photos": len(ids), "systems": {}}
    for spec in args.system:
        name, rest = spec.split("=", 1)
        kind, original, normalized = rest.split(":")
        read = (lambda p: lfm_correct(p, cases)) if kind == "lfm" else convnext_correct
        views = {"as_is": read(original), "size_normalised": read(normalized)}
        report["systems"][name] = {view: {"all": accuracy(found), "size_ambiguous": accuracy(found, ids)} for view, found in views.items()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
        stream.write("\n")
    for name, views in report["systems"].items():
        print(name, {v: (f"{d['all']['accuracy']:.1%}", f"{d['size_ambiguous']['accuracy']:.1%} of {d['size_ambiguous']['n']}") for v, d in views.items()})


if __name__ == "__main__":
    main()
