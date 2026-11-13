"""Size-normalised copies of evaluation photos, to show how much of a score is the image-size / source shortcut.
Every photo is centre-cropped to a square, resized to one size and recompressed at one JPEG quality, with metadata
dropped, so size, aspect ratio and compression no longer differ between crops. Colour and texture still can."""

import argparse
import csv
import io
from pathlib import Path

from PIL import Image, ImageOps

SIDE = 512
QUALITY = 85


def normalize_image(image, side=SIDE, quality=QUALITY):
    """Return JPEG bytes of the square centre crop of `image`, resized to side x side."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    edge = min(image.size)
    left, top = (image.width - edge) // 2, (image.height - edge) // 2
    square = image.crop((left, top, left + edge, top + edge)).resize((side, side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    square.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--part", choices=("tune", "final"), required=True)
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--extra-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.cases.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        cases = [c for c in reader if c["part"] == args.part]
    rows = []
    for case in cases:
        source = (args.crop_data if case["source"] in ("val", "test") else args.extra_images) / case["path"]
        with Image.open(source) as image:
            (args.output_dir / f"{case['id']}.jpg").write_bytes(normalize_image(image))
        rows.append({**case, "source": "normalized", "path": f"{case['id']}.jpg"})
    with (args.output_dir / "manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} normalised photos in {args.output_dir}")


if __name__ == "__main__":
    main()
