"""Bounded crop-only LoRA pilot: matched untuned/tuned validation, no test scoring."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import signal
import time

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


MODEL = "LiquidAI/LFM2.5-VL-1.6B"
REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"
CROPS = ["Cotton", "Maize", "Rice", "Sugarcane", "Wheat"]
QUESTION = "Identify the crop in this image. Answer with exactly one of: Cotton, Maize, Rice, Sugarcane, Wheat."


def load_image(path):
    with Image.open(path) as raw:
        orientation = raw.getexif().get(274, 1)
        image = raw.convert("RGB")
        transforms = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                      4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                      6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                      8: Image.Transpose.ROTATE_90}
        if orientation in transforms:
            image = image.transpose(transforms[orientation])
        image.info.clear()  # Keep pixels/orientation without reserializing malformed source EXIF.
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        return image.copy()


def select_rows(rows, per_train=100, per_val=20, seed=3407):
    selected = {"train": [], "val": []}
    for split, count in (("train", per_train), ("val", per_val)):
        for crop in CROPS:
            pool = sorted((row for row in rows if row["status"] == "kept" and row["split"] == split
                           and row["crop"] == crop), key=lambda row: row["path"])
            if len(pool) < count:
                raise ValueError(f"Not enough {crop} {split} images")
            random.Random(f"{seed}:{split}:{crop}").shuffle(pool)
            selected[split].extend(pool[:count])
    held_out = [row for row in rows if row["split"] in ("val", "test")]
    for key in ("sha256", "group"):
        if {row[key] for row in selected["train"]} & {row[key] for row in held_out}:
            raise ValueError(f"Training overlaps held-out {key}")
    return selected


def score(truth, predictions):
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, labels=CROPS, average="macro", zero_division=0),
        "per_crop_recall": dict(zip(CROPS, recall_score(truth, predictions, labels=CROPS, average=None, zero_division=0).tolist())),
        "invalid_answers": sum(value == "INVALID" for value in predictions),
        "confusion_labels": CROPS + ["INVALID"],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CROPS + ["INVALID"]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("Report must be new and its parent must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "model": MODEL, "revision": REVISION, "seed": 3407,
              "task": "five-crop identification; in-corpus validation pilot, not disease/field evaluation",
              "question": QUESTION, "steps": 200, "batch_size": 4, "learning_rate": 2e-4,
              "rank": 8, "alpha": 8, "max_image_side": 384,
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "unsloth", "transformers", "peft")}}

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def stop(signum, frame):
        raise TimeoutError("External run deadline reached")

    signal.signal(signal.SIGTERM, stop)
    try:
        with args.manifest.open(encoding="utf-8", newline="") as stream:
            selected = select_rows(list(csv.DictReader(stream)))
        report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        report["counts"] = {split: dict(Counter(row["crop"] for row in rows)) for split, rows in selected.items()}
        (args.output / "subsets.json").write_text(json.dumps(selected, indent=2) + "\n")

        # Decode and hash only the 600 selected images, not the entire corpus.
        image_cache = {}
        for rows in selected.values():
            for row in rows:
                path = (args.images / row["path"]).resolve()
                if not path.is_relative_to(args.images.resolve()):
                    raise ValueError("Image path escapes dataset root")
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Image hash changed: {row['path']}")
                image_cache[row["path"]] = load_image(path)
        report["selected_images_verified"] = len(image_cache)
        save()
        print(f"Prepared {len(selected['train'])} train / {len(selected['val'])} validation images", flush=True)

        torch.manual_seed(3407)
        torch.cuda.reset_peak_memory_stats()
        model, processor = FastVisionModel.from_pretrained(
            MODEL, revision=REVISION, use_exact_model_name=True, max_seq_length=2048,
            dtype=torch.bfloat16, load_in_4bit=False,
        )
        report["gpu"] = torch.cuda.get_device_name(0)

        def validate(stage):
            FastVisionModel.for_inference(model)
            begin = time.monotonic()
            answers, predictions = [], []
            for number, row in enumerate(selected["val"], 1):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(image_cache[row["path"]], text, add_special_tokens=False,
                                   return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=8)
                answer = processor.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                prediction = next((crop for crop in CROPS if answer.strip(". ").casefold() == crop.casefold()), "INVALID")
                predictions.append(prediction)
                answers.append({"sha256": row["sha256"], "truth": row["crop"], "prediction": prediction, "raw": answer})
                if number % 20 == 0:
                    print(f"{stage}: {number}/100", flush=True)
            metrics = score([row["crop"] for row in selected["val"]], predictions)
            metrics["seconds"] = round(time.monotonic() - begin, 3)
            metrics["answers"] = answers
            report[stage] = metrics
            save()
            print(f"{stage}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)

        validate("untuned")
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=8, lora_alpha=8, lora_dropout=0, bias="none", random_state=3407,
        )
        model.peft_config["default"].revision = REVISION
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        report["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        collator = UnslothVisionDataCollator(
            model, processor, train_on_responses_only=True,
            instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
        )
        FastVisionModel.for_training(model)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4)
        rng = random.Random(3407)
        order = list(selected["train"])
        losses = []
        train_begin = time.monotonic()
        for step in range(200):
            if step * 4 % len(order) == 0:
                rng.shuffle(order)
            rows = [order[(step * 4 + offset) % len(order)] for offset in range(4)]
            examples = [{"messages": [
                {"role": "user", "content": [{"type": "image", "image": image_cache[row["path"]]}, {"type": "text", "text": QUESTION}]},
                {"role": "assistant", "content": [{"type": "text", "text": row["crop"]}]},
            ]} for row in rows]
            batch = collator(examples)
            if step == 0:
                for labels, row in zip(batch["labels"], rows):
                    assert processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=True).strip() == row["crop"]
            batch = {key: value.to(model.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            assert torch.isfinite(loss).item(), "Nonfinite training loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            if (step + 1) % 20 == 0:
                report["completed_steps"] = step + 1
                report["losses"] = losses
                save()
                print(f"TRAIN {step + 1}/200 loss={losses[-1]:.4f}", flush=True)
        report["training_seconds"] = round(time.monotonic() - train_begin, 3)
        model.save_pretrained(args.output / "adapter")
        processor.save_pretrained(args.output / "adapter")
        report["adapter_saved"] = True
        save()
        del optimizer, parameters, batch, loss
        validate("finetuned")
        report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        report["status"] = "complete"
        save()
    except Exception as error:
        report["status"] = "timed_out" if isinstance(error, TimeoutError) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
