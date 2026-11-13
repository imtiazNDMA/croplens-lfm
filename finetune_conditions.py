"""Crop-and-condition LoRA pilot using cleaned training data and matched validation."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from finetune_crops import MODEL, REVISION, load_image


CONFIG = Path(__file__).parent / "configs" / "condition-pilot.json"


def answer_for(label):
    crop, condition = label.split("_", 1)
    return {"crop": crop.capitalize(), "condition": condition}


def make_question(labels):
    by_crop = {}
    for label in labels:
        answer = answer_for(label)
        by_crop.setdefault(answer["crop"], []).append(answer["condition"])
    choices = "; ".join(f"{crop}: {', '.join(conditions)}" for crop, conditions in sorted(by_crop.items()))
    return ('Identify the crop and visible condition. Return only JSON with exactly the keys "crop" and "condition". '
            'Use one supported pair from: ' + choices + '. Example: {"crop":"Maize","condition":"healthy"}.')


def parse_answer(text, labels):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return "INVALID"
    if not isinstance(value, dict) or set(value) != {"crop", "condition"}:
        return "INVALID"
    return next((label for label in labels if value == answer_for(label)), "INVALID")


def select_rows(pool, original, config):
    originals = {row["path"]: row for row in original}
    for row in pool:
        source = originals.get(row["path"])
        if not source or source["status"] != "kept" or source["split"] != "train":
            raise ValueError("Pool contains a row outside the retained training split")
        if any(row[key] != source[key] for key in ("sha256", "group", "label", "crop", "split", "status")):
            raise ValueError("Pool row disagrees with original manifest")
    labels = config["classes"]
    if len(labels) != len(set(labels)):
        raise ValueError("Repeated condition class")
    crops = {answer_for(label)["crop"] for label in labels}
    if any(f"{crop.lower()}_healthy" not in labels for crop in crops):
        raise ValueError("Every supported crop needs a healthy class")
    selected = {"train": [], "val": []}
    for split, source, count in (("train", pool, config["train_per_class"]),
                                 ("val", original, config["val_per_class"])):
        for label in labels:
            candidates = sorted((row for row in source if row["status"] == "kept" and row["split"] == split
                                 and row["label"] == label), key=lambda row: row["path"])
            if len(candidates) < count:
                raise ValueError(f"Insufficient {split} data for {label}: {len(candidates)} < {count}")
            random.Random(f"{config['seed']}:{split}:{label}").shuffle(candidates)
            selected[split].extend(candidates[:count])
    held_out = [row for row in original if row["split"] in ("val", "test")]
    for key in ("sha256", "group"):
        if {row[key] for row in selected["train"]} & {row[key] for row in held_out}:
            raise ValueError(f"Training overlaps held-out {key}")
    return selected


def score(truth, predictions, labels):
    true_answers = [answer_for(label) for label in truth]
    predicted = [answer_for(label) if label != "INVALID" else {"crop": "INVALID", "condition": "INVALID"}
                 for label in predictions]
    conditions = sorted({answer_for(label)["condition"] for label in labels})
    return {
        "crop_accuracy": accuracy_score([a["crop"] for a in true_answers], [a["crop"] for a in predicted]),
        "joint_accuracy": accuracy_score(truth, predictions),
        "joint_macro_f1": f1_score(truth, predictions, labels=labels, average="macro", zero_division=0),
        "condition_macro_f1": f1_score([a["condition"] for a in true_answers], [a["condition"] for a in predicted],
                                        labels=conditions, average="macro", zero_division=0),
        "per_class_recall": dict(zip(labels, recall_score(truth, predictions, labels=labels, average=None, zero_division=0).tolist())),
        "invalid_answers": predictions.count("INVALID"),
        "confusion_labels": labels + ["INVALID"],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels + ["INVALID"]).tolist(),
    }


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pool", "images", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--reference", type=Path, help="Reuse a verified historical untuned evaluation")
    parser.add_argument("--mlflow-run-id", help="Attach metrics/artifacts to the supervisor's MLflow run")
    parser.add_argument("--training-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("Report must be new and its parent must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = config["classes"]
    question = make_question(labels)
    started = time.monotonic()
    report = {"status": "running", "model": MODEL, "revision": REVISION, "config": config, "question": question,
              "task": "closed-set crop-condition development validation; no test/field evaluation",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "unsloth", "transformers", "peft")}}

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def stop(signum, frame):
        raise TimeoutError("External run deadline reached")

    signal.signal(signal.SIGTERM, stop)
    tracking = False
    try:
        if args.mlflow_run_id:
            import mlflow
            from experiment_tracking import log_environment, log_report
            mlflow.start_run(run_id=args.mlflow_run_id)
            report["mlflow_run_id"] = args.mlflow_run_id
            mlflow.log_params({key: value for key, value in config.items() if key != "classes"})
            log_environment()
            tracking = True
        selected = select_rows(read_rows(args.pool), read_rows(args.manifest), config)
        report["input_hashes"] = {"manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                  "cleaned_pool": hashlib.sha256(args.pool.read_bytes()).hexdigest()}
        report["counts"] = {split: dict(Counter(row["label"] for row in rows)) for split, rows in selected.items()}
        (args.output / "subsets.json").write_text(json.dumps(selected, indent=2) + "\n")
        images = {}
        root = args.images.resolve()
        for rows in selected.values():
            for row in rows:
                path = (root / row["path"]).resolve()
                if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Unsafe path or image hash mismatch: {row['path']}")
                images[row["path"]] = load_image(path)
        report["selected_images_verified"] = len(images)
        save()
        print(f"Prepared {len(selected['train'])} train / {len(selected['val'])} validation images, {len(labels)} classes", flush=True)
        torch.manual_seed(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        model, processor = FastVisionModel.from_pretrained(MODEL, revision=REVISION, use_exact_model_name=True,
                                                         max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
        report["gpu"] = torch.cuda.get_device_name(0)

        def validate(stage):
            FastVisionModel.for_inference(model)
            begin = time.monotonic()
            answers, predictions = [], []
            for index, row in enumerate(selected["val"], 1):
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(images[row["path"]], prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=48)
                raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                prediction = parse_answer(raw, labels)
                predictions.append(prediction)
                answers.append({"sha256": row["sha256"], "truth": row["label"], "prediction": prediction, "raw": raw})
                if index % 40 == 0:
                    report[stage] = {"status": "partial", "answers": answers}
                    save()
                    print(f"{stage}: {index}/{len(selected['val'])}", flush=True)
            metrics = score([row["label"] for row in selected["val"]], predictions, labels)
            metrics.update(seconds=round(time.monotonic() - begin, 3), answers=answers)
            report[stage] = metrics
            save()
            print(f"{stage}: crop={metrics['crop_accuracy']:.3f} joint={metrics['joint_accuracy']:.3f} joint_f1={metrics['joint_macro_f1']:.3f}", flush=True)

        if args.reference:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            fixed = set(config) - {"learning_rate", "rank"}
            if (reference["status"] != "complete" or reference["input_hashes"] != report["input_hashes"]
                    or reference["question"] != question or reference["revision"] != REVISION
                    or any(config[key] != reference["config"][key] for key in fixed)):
                raise ValueError("Reference data, task or fixed training work differs")
            expected = {row["sha256"] for row in selected["val"]}
            if expected != {row["sha256"] for row in reference["untuned"]["answers"]}:
                raise ValueError("Reference validation images differ")
            report["untuned"] = reference["untuned"]
            report["untuned_reused_from_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            save()
        else:
            validate("untuned")
        model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True, r=config["rank"], lora_alpha=config["rank"],
            lora_dropout=0, bias="none", random_state=config["seed"])
        model.peft_config["default"].revision = REVISION
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        report["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        collator = UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
            instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
        FastVisionModel.for_training(model)
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
        order = list(selected["train"])
        rng = random.Random(config["seed"])
        losses = []
        begin = time.monotonic()
        for step in range(config["steps"]):
            if time.monotonic() - begin >= args.training_seconds:
                raise TimeoutError("Training safety cap reached; incomplete trials cannot win")
            if step * config["batch"] % len(order) == 0:
                rng.shuffle(order)
            rows = [order[(step * config["batch"] + i) % len(order)] for i in range(config["batch"])]
            examples = [{"messages": [
                {"role": "user", "content": [{"type": "image", "image": images[row["path"]]}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer_for(row["label"]), separators=(",", ":"))}]},
            ]} for row in rows]
            batch = collator(examples)
            if step == 0:
                for target, row in zip(batch["labels"], rows):
                    assert parse_answer(processor.tokenizer.decode(target[target != -100], skip_special_tokens=True).strip(), labels) == row["label"]
            batch = {key: value.to(model.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            assert torch.isfinite(loss).item(), "Nonfinite training loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            if tracking:
                mlflow.log_metric("train.loss", losses[-1], step=step + 1)
            if (step + 1) % 40 == 0:
                report.update(completed_steps=step + 1, losses=losses)
                save()
                print(f"TRAIN {step + 1}/{config['steps']} loss={losses[-1]:.4f}", flush=True)
        report["training_seconds"] = round(time.monotonic() - begin, 3)
        model.save_pretrained(args.output / "adapter")
        processor.save_pretrained(args.output / "adapter")
        (args.output / "adapter" / "condition-pilot.json").write_text(json.dumps(config, indent=2) + "\n")
        report["adapter_saved"] = True
        save()
        del optimizer, parameters, batch, loss
        validate("finetuned")
        report.update(status="complete", peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        save()
        if tracking:
            log_report(report, args.output / "adapter", log_losses=False)
            mlflow.end_run(status="FINISHED")
    except Exception as error:
        report["status"] = "timed_out" if isinstance(error, TimeoutError) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        if tracking:
            mlflow.log_dict(report, "failure-report.json")
            mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
