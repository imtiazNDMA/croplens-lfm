"""Training examples for the photo check, from the cleaned pool. Each example is a path, a degradation (or none) and the
exact answer text the model is trained to produce. Degradations are applied when an image is loaded, never stored.

Poor-photo examples are made from training photos only, with the strength drawn at random per image so the check's
evaluation images (fixed recipe, made from val photos) are not the training recipe. "coarse" loses detail like a
smeared photo, so it is labelled blurred."""

import csv
import json
from pathlib import Path
import random

from PIL import Image, ImageEnhance, ImageFilter

from contract import render_photo_check

HOW_TO_FLAG = {"blurred": "blurred", "dark": "too_dark", "coarse": "blurred"}


def degrade(image, how, rng):
    image = image.convert("RGB")
    if how == "blurred":
        return image.filter(ImageFilter.GaussianBlur(max(image.size) / rng.uniform(40, 80)))
    if how == "dark":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.08, 0.2))
    if how == "coarse":
        factor = rng.randint(8, 16)
        small = image.resize((max(1, image.width // factor), max(1, image.height // factor)))
        return small.resize(image.size)
    raise ValueError(f"unknown degradation: {how}")


def answer_for(crop, how=None):
    flags = {HOW_TO_FLAG[how]: True} if how else {}
    return render_photo_check({"is_plant": True, "crop": crop, "other_plant": "", **flags})


def balanced_sample(examples, count, seed):
    """`count` examples where each crop is equally likely: pick a crop, then one of its examples, with replacement."""
    rng = random.Random(seed)
    by_crop = {}
    for example in examples:
        by_crop.setdefault(example["crop"], []).append(example)
    crops = sorted(by_crop)
    return [rng.choice(by_crop[rng.choice(crops)]) for _ in range(count)]


def make_examples(pool_rows, seed=3407, poor_fraction=0.08):
    """One clean example per pool image, plus a degraded copy for a random `poor_fraction` of them."""
    rng = random.Random(seed)
    examples = []
    for row in sorted(pool_rows, key=lambda r: r["path"]):
        examples.append({"path": row["path"], "how": None, "answer": answer_for(row["crop"]), "crop": row["crop"], "label": row["label"]})
        if rng.random() < poor_fraction:
            how = rng.choice(sorted(HOW_TO_FLAG))
            examples.append({"path": row["path"], "how": how, "answer": answer_for(row["crop"], how), "crop": row["crop"],
                             "label": row["label"], "seed": rng.randrange(2**31)})
    return examples


def load_image(root, example):
    with Image.open(Path(root) / example["path"]) as raw:
        image = raw.convert("RGB")
        image.load()
    if example["how"]:
        image = degrade(image, example["how"], random.Random(example["seed"]))
    return image


QUESTION = "Check this photo."


class LazySftDataset:
    """Examples as chat messages, with each image decoded (and degraded) only when it is asked for. Photos can be
    6000x4000, so the long side is capped at `max_side` to keep memory and step time bounded."""

    def __init__(self, examples, root, max_side=512):
        self.examples, self.root, self.max_side = examples, root, max_side

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        image = load_image(self.root, example)
        if max(image.size) > self.max_side:
            image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
        return {"messages": [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]},
            {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
        ]}


def as_hf_dataset(examples, root, max_side=512):
    """The same lazy items as a `datasets.Dataset`, which TRL's trainer requires: only the metadata is stored, and
    each row's image is decoded when the row is read."""
    from datasets import Dataset

    lazy = LazySftDataset(examples, root, max_side)
    table = Dataset.from_dict({"index": list(range(len(examples)))})
    table.set_transform(lambda batch: {"messages": [lazy[i]["messages"] for i in batch["index"]]})
    return table


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.pool.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    examples = make_examples(rows)
    with args.output.open("x", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example) + "\n")
    print(f"{len(examples)} examples ({sum(e['how'] is not None for e in examples)} degraded) from {len(rows)} pool images")


if __name__ == "__main__":
    main()
