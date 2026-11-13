"""Read-only audit of a crop manifest, its images, and optional photo-check holdouts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np
from PIL import Image
import yaml


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENTED_NAME = re.compile(r"^(crop|translation|zoom|contrast|flip|rotation|rotate|brightness|blur)[_-]?\d", re.I)
REQUIRED = {"path", "label", "crop", "category", "sha256", "dhash", "group", "status", "split"}


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        if not required <= columns:
            raise ValueError(f"Missing manifest columns: {sorted(required - columns)}")
        return list(reader), sorted(columns)


@lru_cache(maxsize=None)
def real_directory(directory):
    # Resolving costs several filesystem lookups, and on a Windows-mounted drive each one is slow. Done per
    # image it took 426 of the audit's 594 seconds, so each directory is resolved once.
    return directory.resolve()


def image_path(root, relative):
    if "\\" in relative or ":" in relative:
        raise ValueError("Image path must be relative POSIX format")
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ValueError("Image path escapes the supplied root")
    real_root = real_directory(root)
    parent = real_directory(real_root.joinpath(*parts[:-1]))
    candidate = parent / parts[-1]
    # A symlinked directory resolves outside the root; a symlinked file is caught here.
    if not parent.is_relative_to(real_root) or candidate.is_symlink():
        raise ValueError("Image path escapes the supplied root")
    return candidate


def inspect_image(root, relative, expected_sha, expected_dhash=None):
    result = {"path": relative, "errors": []}
    try:
        path = image_path(root, relative)
        content = path.read_bytes()
        result["sha256"] = hashlib.sha256(content).hexdigest()
        if result["sha256"] != expected_sha:
            result["errors"].append("sha256_mismatch")
        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            image.load()  # Decode fully rather than accepting a readable header alone.
        with Image.open(path) as image:
            image.draft("L", (128, 128))
            grey = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
        result["dhash"] = np.packbits((grey[:, 1:] > grey[:, :-1]).flatten()).tobytes().hex()
        if expected_dhash is not None and result["dhash"] != expected_dhash:
            result["errors"].append("dhash_mismatch")
    except FileNotFoundError:
        result["errors"].append("missing_file")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        result["errors"].append(type(error).__name__)
    return result


def split_conflicts(rows, key):
    groups = defaultdict(set)
    for row in rows:
        if row["split"]:
            groups[row[key]].add(row["split"])
    return sum(len(splits) > 1 for splits in groups.values())


def near_pairs(left, right, threshold):
    if not left or not right:
        return 0
    a = np.frombuffer(b"".join(bytes.fromhex(value) for value in left), dtype=np.uint64).reshape(-1, 4)
    b = np.frombuffer(b"".join(bytes.fromhex(value) for value in right), dtype=np.uint64).reshape(-1, 4)
    count = 0
    for start in range(0, len(a), 128):
        distance = np.bitwise_count(a[start:start + 128, None, :] ^ b[None, :, :]).sum(axis=2)
        count += int(np.count_nonzero(distance <= threshold))
    return count


def audit(manifest, images, taxonomy_path, photo_manifest=None, photo_images=None):
    started = time.monotonic()

    def log(message):  # Progress goes to stderr so a run that is cut off still shows where its time went.
        print(f"[{time.monotonic() - started:7.1f}s] {message}", file=sys.stderr, flush=True)

    rows, columns = read_csv(manifest, REQUIRED)
    taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    classes = taxonomy["classes"]
    if not images.is_dir():
        raise ValueError("Image root does not exist")
    for row in rows:
        image_path(images, row["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", row["dhash"]):
            raise ValueError("Expected SHA-256 and 256-bit dHash hex values")
        if row["status"] not in {"kept", "duplicate", "conflict", "rare_class"}:
            raise ValueError("Unknown manifest status")
        if row["split"] not in {"", "train", "val", "test"} or (row["status"] == "kept" and not row["split"]):
            raise ValueError("Invalid or missing split")
        if row["status"] in {"conflict", "rare_class"} and row["split"]:
            raise ValueError("Excluded row has an assigned split")
        if not row["group"] or row["label"] not in classes:
            raise ValueError("Missing group or unknown taxonomy label")
        spec = classes[row["label"]]
        if row["crop"] != spec["crop"] or row["category"] != spec["category"]:
            raise ValueError("Crop/category disagrees with taxonomy")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Repeated paths in manifest")

    log(f"manifest and taxonomy validated: {len(rows)} rows")
    checks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(lambda row: inspect_image(images, row["path"], row["sha256"], row["dhash"]), rows):
            checks.append(item)
            if len(checks) % 2500 == 0:
                log(f"inspected {len(checks)}/{len(rows)} images")
    log(f"inspected all {len(checks)} images")
    by_path = {row["path"]: row for row in rows}
    checked = {item["path"]: item for item in checks}
    kept = [row for row in rows if row["status"] == "kept"]
    inventory = {p.relative_to(images).as_posix() for p in images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
    log(f"scanned the image folder: {len(inventory)} files")
    counts = {}
    for label in sorted(classes):
        members = [row for row in kept if row["label"] == label]
        counts[label] = {"crop": classes[label]["crop"], "category": classes[label]["category"],
                         "total": len(members), "splits": dict(Counter(row["split"] for row in members))}
    hashes = {split: [checked[row["path"]]["dhash"] for row in kept
                      if row["split"] == split and "dhash" in checked[row["path"]]]
              for split in ("train", "val", "test")}
    threshold = taxonomy["near_duplicate_distance"]
    cross_near = {f"{a}_vs_{b}": near_pairs(hashes[a], hashes[b], threshold)
                  for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    log("compared hashes across splits")
    report = {
        "audit_version": 1,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "manifest_rows": len(rows), "manifest_columns": columns,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "kept_split_counts": dict(Counter(row["split"] for row in kept)),
        "kept_crop_counts": dict(Counter(row["crop"] for row in kept)),
        "kept_category_counts": dict(Counter(row["category"] for row in kept)),
        "classes": counts,
        "filesystem_image_count": len(inventory),
        "unlisted_image_count": len(inventory - set(by_path)),
        "file_error_counts": dict(Counter(error for item in checks for error in item["errors"])),
        "file_errors": [item for item in checks if item["errors"]],
        "kept_dimensions": dict(Counter(f"{checked[row['path']].get('width')}x{checked[row['path']].get('height')}" for row in kept)),
        "augmentation_named_rows": sum(bool(AUGMENTED_NAME.match(Path(row["path"]).stem)) for row in rows),
        "augmentation_named_kept": sum(bool(AUGMENTED_NAME.match(Path(row["path"]).stem)) for row in kept),
        "assigned_sha256_cross_split_groups": split_conflicts(rows, "sha256"),
        "assigned_duplicate_cross_split_groups": split_conflicts(rows, "group"),
        "multiple_kept_per_group": sum(count > 1 for count in Counter(row["group"] for row in kept).values()),
        "dhash_threshold_bits": threshold,
        "cross_split_near_pairs": cross_near,
        "provenance_fields_absent": sorted({"source_url", "source_dataset_revision", "license", "original_photo_id", "field_id", "collection_session"} - set(columns)),
    }

    if photo_manifest is not None:
        photos, _ = read_csv(photo_manifest, {"id", "part", "group", "crop", "label", "source", "path"})
        if photo_images is None or not photo_images.is_dir():
            raise ValueError("Photo image root is required and must exist")
        log(f"checking {len(photos)} photo-check images")
        photo_checks = []
        parents = []
        for photo in photos:
            if photo["part"] not in {"tune", "final"} or photo["group"] not in {"crop", "poor", "other-plant", "not-plant"}:
                raise ValueError("Unknown photo-check part/group")
            if not re.fullmatch(r"[0-9a-f]{64}", photo["id"]):
                raise ValueError("Invalid photo image ID")
            root = images if photo["group"] == "crop" else photo_images
            photo_checks.append(inspect_image(root, photo["path"], photo["id"]))
            parent_path = photo["path"] if photo["group"] == "crop" else photo["source"].split(":", 1)[1] if photo["source"].startswith("val:") else None
            if parent_path is not None:
                parent = by_path.get(parent_path)
                parents.append({"part": photo["part"], "group": photo["group"],
                                "parent_found": parent is not None,
                                "parent_split": parent["split"] if parent else "unknown"})
        train_shas = {checked[row["path"]].get("sha256") for row in kept if row["split"] == "train"}
        report["photo_check"] = {
            "manifest_sha256": hashlib.sha256(photo_manifest.read_bytes()).hexdigest(),
            "rows": len(photos),
            "part_group_counts": dict(Counter(f"{p['part']}:{p['group']}" for p in photos)),
            "file_error_counts": dict(Counter(error for item in photo_checks for error in item["errors"])),
            "file_errors": [item for item in photo_checks if item["errors"]],
            "exact_overlap_with_crop_training": sum(item.get("sha256") in train_shas for item in photo_checks if "sha256" in item),
            "known_parent_split_counts": dict(Counter(f"{p['part']}:{p['group']}:{p['parent_split']}" for p in parents)),
            "missing_known_parents": sum(not p["parent_found"] for p in parents),
            "known_parents_in_training": sum(p["parent_split"] == "train" for p in parents),
            "near_pairs_with_crop_training": near_pairs([item["dhash"] for item in photo_checks if "dhash" in item], hashes["train"], threshold),
            "note": "Photo-check data are existing evaluation assets, not new training data. Parent links are only those explicitly represented in its manifest.",
        }
    log("done")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "images", "taxonomy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--photo-manifest", type=Path)
    parser.add_argument("--photo-images", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new audit version")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory must already exist")
    if bool(args.photo_manifest) != bool(args.photo_images):
        parser.error("Supply photo manifest and image root together")
    report = audit(args.manifest, args.images, args.taxonomy, args.photo_manifest, args.photo_images)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("manifest_rows", "status_counts", "file_error_counts", "cross_split_near_pairs", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
