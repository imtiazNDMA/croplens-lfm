"""Score photo-check answers against the truth. A pure function of the cases and the answers, so any model's
recorded answers (untuned, tuned, or the portal's Qwen) are scored the same way. An invalid answer is a failure."""

from collections import Counter
import csv
from pathlib import Path

from contract import CROPS, parse_photo_check

STOP_KINDS = ("not_plant", "unsupported")


def outcome(text, lenient=False):
    """What the portal's photo check would do with this answer: one of the five crops, or a named result."""
    answer, reason = parse_photo_check(text, lenient)
    if answer is None:
        return {"kind": "invalid", "reason": reason, "crop": None, "flags": []}
    flags = [name for name in ("blurred", "too_dark", "not_close_up", "several_leaves") if answer[name]]
    if not answer["is_plant"]:
        return {"kind": "not_plant", "crop": None, "flags": flags}
    if answer["crop"] == "Other":
        return {"kind": "unsupported", "crop": None, "flags": flags}
    if answer["crop"] == "Unclear":
        return {"kind": "unclear", "crop": None, "flags": flags}
    return {"kind": "crop", "crop": answer["crop"], "flags": flags}


def load_cases(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def score(cases, answers, lenient=False):
    """`answers` maps case id to the model's raw text. Every case must be answered."""
    missing = [case["id"] for case in cases if case["id"] not in answers]
    if missing:
        raise KeyError(f"{len(missing)} photos have no answer, the first being {missing[0]}")
    results = {case["id"]: outcome(answers[case["id"]], lenient) for case in cases}
    groups = {g: [c for c in cases if c["group"] == g] for g in ("crop", "other-plant", "not-plant", "poor")}

    def share(group, test):
        return sum(test(c) for c in group) / len(group) if group else None

    def stopped(case):
        return results[case["id"]]["kind"] in STOP_KINDS

    crops = groups["crop"]
    per_crop = {crop: Counter() for crop in CROPS}
    confusions = Counter()
    for case in crops:
        result = results[case["id"]]
        if result["kind"] == "crop":
            slot = "right" if result["crop"] == case["crop"] else "wrong_crop"
            if slot == "wrong_crop":
                confusions[f"{case['crop']} -> {result['crop']}"] += 1
        elif result["kind"] in STOP_KINDS:
            slot = "stopped"
        else:
            slot = result["kind"]  # unclear or invalid
        per_crop[case["crop"]][slot] += 1
    return {
        "photos": {g: len(m) for g, m in groups.items()},
        "crop_identification": share(crops, lambda c: results[c["id"]]["kind"] == "crop" and results[c["id"]]["crop"] == c["crop"]),
        "unclear_share": share(crops, lambda c: results[c["id"]]["kind"] == "unclear"),
        "false_stops": share(crops, stopped),
        "non_plant_stopped": share(groups["not-plant"], stopped),
        "unsupported_plant_stopped": share(groups["other-plant"], stopped),
        "invalid_share": share(cases, lambda c: results[c["id"]]["kind"] == "invalid"),
        "quality_raised_on_poor": share(groups["poor"], lambda c: bool(results[c["id"]]["flags"])),
        "quality_raised_on_clean": share(crops, lambda c: bool(results[c["id"]]["flags"])),
        "per_crop": {crop: dict(counts) for crop, counts in per_crop.items()},
        "confusions": dict(sorted(confusions.items())),
    }
