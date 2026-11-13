"""Read-only checks that extend the dataset audit: how close augmented copies could sit across splits, and whether
image size alone gives the crop away. Reads image headers only; writes one new JSON report."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

from audit_dataset import AUGMENTED_NAME, image_path, read_csv, REQUIRED, near_pairs

THRESHOLDS = (12, 16, 24, 32, 48, 64)
SPLITS = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def as_words(hashes):
    return np.frombuffer(b"".join(bytes.fromhex(value) for value in hashes), dtype=np.uint64).reshape(-1, 4)


def min_distances(left, right):
    """For each hash in `left`, the Hamming distance to its closest hash in `right`."""
    if not left or not right:
        return np.zeros(0, dtype=np.int64)
    a, b = as_words(left), as_words(right)
    nearest = [np.bitwise_count(a[start:start + 128, None, :] ^ b[None, :, :]).sum(axis=2).min(axis=1)
               for start in range(0, len(a), 128)]
    return np.concatenate(nearest)


def near_pair_counts(hashes, thresholds):
    return {f"{a}_vs_{b}": {str(t): near_pairs(hashes[a], hashes[b], t) for t in thresholds} for a, b in PAIRS}


def size_only_crop_accuracy(train, later):
    """How well "the crop most often seen at this exact size in training" names the crop of later photos.
    `train` and `later` are lists of (size, crop). A size never seen in training gets the overall majority crop."""
    by_size = defaultdict(Counter)
    for size, crop in train:
        by_size[size][crop] += 1
    overall = Counter(crop for _, crop in train)
    fallback = overall.most_common(1)[0][0]
    hits = seen_hits = seen = 0
    for size, crop in later:
        known = size in by_size
        guess = by_size[size].most_common(1)[0][0] if known else fallback
        hits += guess == crop
        seen += known
        seen_hits += known and guess == crop
    return {"photos": len(later), "accuracy": hits / len(later), "size_seen_in_training": seen / len(later),
            "accuracy_where_size_seen": seen_hits / seen if seen else None,
            "majority_crop_baseline": sum(crop == fallback for _, crop in later) / len(later)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report name")
    started = time.monotonic()
    rows, _ = read_csv(args.manifest, REQUIRED)
    kept = [row for row in rows if row["status"] == "kept"]

    def header_size(row):
        with Image.open(image_path(args.images, row["path"])) as image:
            return "x".join(map(str, image.size))

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(header_size, kept))
    print(f"[{time.monotonic() - started:6.1f}s] read {len(sizes)} image headers", file=sys.stderr, flush=True)
    for row, size in zip(kept, sizes):
        row["size"] = size

    hashes = {split: [row["dhash"] for row in kept if row["split"] == split] for split in SPLITS}
    augmented = [row for row in kept if AUGMENTED_NAME.match(Path(row["path"]).stem)]
    train = [(row["size"], row["crop"]) for row in kept if row["split"] == "train"]
    report = {
        "check_version": 1,
        "kept_images": len(kept),
        "dhash_note": "Distances use the manifest's stored 256-bit dHash, which audit 001/002 recomputed and found unchanged.",
        "cross_split_near_pairs_by_threshold": near_pair_counts(hashes, THRESHOLDS),
        "closest_train_distance_percentiles": {
            split: {str(p): float(np.percentile(min_distances(hashes[split], hashes["train"]), p)) for p in (0, 1, 5, 25, 50)}
            for split in ("val", "test")},
        "augmentation_named_kept_by_split": dict(Counter(row["split"] for row in augmented)),
        "augmentation_named_kept_by_class": dict(sorted(Counter(row["label"] for row in augmented).items())),
        "augmentation_prefixes": dict(Counter(re.split(r"[_-]", Path(row["path"]).stem)[0].lower() for row in augmented)),
        "size_by_crop": {size: dict(Counter(row["crop"] for row in kept if row["size"] == size))
                         for size, _ in Counter(row["size"] for row in kept).most_common(12)},
        "distinct_sizes": len(set(sizes)),
        "size_only_crop_accuracy": {split: size_only_crop_accuracy(train, [(row["size"], row["crop"]) for row in kept if row["split"] == split])
                                    for split in ("val", "test")},
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
