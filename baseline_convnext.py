"""The portal's packaged ConvNeXt classifier (five-crop-001) on the photo-check crop photos, two ways: told the true crop
(as the portal does when a person picks it) and not told (the crop is the crop of its best class over all 37).
Preprocessing is the package's own: RGB, 224x224 bilinear stretch, ImageNet normalisation. Needs torch and Pillow; run
it in the portal's crop-disease environment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

CROP_INDEX = {"Cotton": 0, "Maize": 1, "Rice": 2, "Sugarcane": 3, "Wheat": 4}
MEAN, STD = np.asarray((0.485, 0.456, 0.406), np.float32), np.asarray((0.229, 0.224, 0.225), np.float32)


def prepare(path):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return torch.from_numpy(((np.asarray(image, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)


def macro_f1(truth, predicted):
    scores = []
    for name in sorted(set(truth)):
        hits = sum(t == name and p == name for t, p in zip(truth, predicted))
        named, actual = predicted.count(name), truth.count(name)
        scores.append(2 * hits / (named + actual) if hits else 0.0)
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--part", choices=("tune", "final"), required=True)
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {int(c["label_index"]): c for c in json.loads((args.package / "label_map.json").read_text(encoding="utf-8"))["classes"]}
    model = torch.jit.load(str(args.package / "model.ts"), map_location="cpu").eval()
    with args.cases.open(encoding="utf-8", newline="") as stream:
        cases = [c for c in csv.DictReader(stream) if c["part"] == args.part and c["group"] == "crop"]
    rows = []
    with torch.inference_mode():
        for case in cases:
            image = prepare(args.crop_data / case["path"])
            unmasked = model.model(image).argmax(1).item()  # temperature does not change the best class
            given = model(image, torch.tensor([CROP_INDEX[case["crop"]]])).argmax(1).item()
            rows.append({"id": case["id"], "truth": case["label"], "crop": case["crop"],
                         "inferred_class": labels[unmasked]["canonical_class_id"], "inferred_crop": labels[unmasked]["crop"],
                         "given_class": labels[given]["canonical_class_id"]})
    truth = [r["truth"] for r in rows]
    report = {"package": args.package.name, "part": args.part, "photos": len(rows),
              "crop_identification_inferred": sum(r["inferred_crop"] == r["crop"] for r in rows) / len(rows),
              "class_accuracy_crop_given": sum(r["given_class"] == r["truth"] for r in rows) / len(rows),
              "class_accuracy_crop_inferred": sum(r["inferred_class"] == r["truth"] for r in rows) / len(rows),
              "class_macro_f1_crop_given": macro_f1(truth, [r["given_class"] for r in rows]),
              "class_macro_f1_crop_inferred": macro_f1(truth, [r["inferred_class"] for r in rows]),
              "note": "Cannot refuse: other plants, non-plants and poor photos are not scored for it.",
              "rows": rows}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
        stream.write("\n")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k not in ("rows", "note")})


if __name__ == "__main__":
    main()
