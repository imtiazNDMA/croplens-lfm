"""The photos whose exact pixel size does not give the crop away: sizes that occur under more than one crop among the
kept manifest images. A model that scores well only on photos whose size is unique to a crop may be reading the size or
the source; its score on this subset is the fairer one. Reads image headers only."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from PIL import Image


def ambiguous_sizes(crop_by_size):
    """Sizes seen under two or more different crops. `crop_by_size` maps a size to the crops seen at that size."""
    return {size for size, crops in crop_by_size.items() if len(set(crops)) > 1}


def subset_ids(cases, sizes_by_path, crop_by_size):
    """Ids of the five-crop `cases` whose size is shared by several crops."""
    shared = ambiguous_sizes(crop_by_size)
    return sorted(case["id"] for case in cases if case["group"] == "crop" and sizes_by_path[case["path"]] in shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="the crop manifest")
    parser.add_argument("--cases", type=Path, required=True, help="the photo-check manifest (original paths)")
    parser.add_argument("--part", choices=("tune", "final"), required=True)
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        kept = [r for r in csv.DictReader(stream) if r["status"] == "kept"]
    crop_by_size = defaultdict(list)
    sizes = {}
    for row in kept:
        with Image.open(args.crop_data / row["path"]) as image:
            sizes[row["path"]] = image.size
        crop_by_size[sizes[row["path"]]].append(row["crop"])
    with args.cases.open(encoding="utf-8", newline="") as stream:
        cases = [c for c in csv.DictReader(stream) if c["part"] == args.part and c["group"] == "crop"]
    ids = subset_ids(cases, sizes, crop_by_size)
    chosen = set(ids)
    report = {"part": args.part, "crop_photos": len(cases), "size_ambiguous": len(ids),
              "definition": "exact pixel size also occurs under a different crop among the 11,453 kept images",
              "by_crop": {c: sum(1 for case in cases if case["id"] in chosen and case["crop"] == c) for c in sorted({x["crop"] for x in cases})},
              "ids": ids}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
        stream.write("\n")
    print({k: v for k, v in report.items() if k != "ids"})


if __name__ == "__main__":
    main()
