"""Build the cleaned training pool from the crop manifest. Read-only on the source: val and test stay exactly as
they are; only train images are removed, and each removal is recorded with its reason."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from audit_dataset import AUGMENTED_NAME, REQUIRED, read_csv
from shortcut_checks import as_words


def near_mask(hashes, others, threshold):
    """True for each hash in `hashes` that is within `threshold` bits of at least one hash in `others`."""
    mask = np.zeros(len(hashes), dtype=bool)
    if not hashes or not others:
        return mask
    a, b = as_words(hashes), as_words(others)
    for start in range(0, len(a), 128):
        distance = np.bitwise_count(a[start:start + 128, None, :] ^ b[None, :, :]).sum(axis=2)
        mask[start:start + 128] = (distance <= threshold).any(axis=1)
    return mask


def clean_training_pool(rows, threshold=32):
    kept = [row for row in rows if row["status"] == "kept"]
    train = [row for row in kept if row["split"] == "train"]
    evaluation = [row["dhash"] for row in kept if row["split"] in ("val", "test")]
    near = near_mask([row["dhash"] for row in train], evaluation, threshold)
    pool, removed = [], []
    for row, is_near in zip(train, near):
        if is_near:
            removed.append({**row, "reason": "near_val_or_test"})
        elif AUGMENTED_NAME.match(Path(row["path"]).stem):
            removed.append({**row, "reason": "augmentation_named"})
        else:
            pool.append(row)
    return pool, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=32)
    args = parser.parse_args()
    pool_path, report_path = args.output_dir / "training_pool.csv", args.output_dir / "training_pool_report.json"
    for path in (pool_path, report_path):
        if path.exists():
            parser.error(f"{path} exists; choose a new output directory")
    rows, columns = read_csv(args.manifest, REQUIRED)
    pool, removed = clean_training_pool(rows, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with pool_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pool)
    train_kept = [r for r in rows if r["status"] == "kept" and r["split"] == "train"]
    report = {
        "threshold_bits": args.threshold,
        "train_kept": len(train_kept), "pool": len(pool), "removed": len(removed),
        "removed_by_reason": dict(Counter(r["reason"] for r in removed)),
        "per_class": {label: {"train_kept": sum(r["label"] == label for r in train_kept),
                              "pool": sum(r["label"] == label for r in pool),
                              "removed": {reason: sum(r["label"] == label and r["reason"] == reason for r in removed)
                                          for reason in ("near_val_or_test", "augmentation_named")}}
                      for label in sorted({r["label"] for r in train_kept})},
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("train_kept", "pool", "removed", "removed_by_reason")}, indent=2))


if __name__ == "__main__":
    main()
