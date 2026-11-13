"""LoRA fine-tune of LFM2.5-VL-1.6B (language side only, vision encoder frozen) on photo-check examples.
Fixed training work: the number of optimizer steps is set, and the time cap only guards against an overrun (a capped run is
marked incomplete). Run under an external timeout. Selection happens elsewhere, on val; this script never reads test."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

MODEL = "LiquidAI/LFM2.5-VL-1.6B"
MODEL_REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--balance", action="store_true", help="draw the training examples so each crop is equally likely")
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from sft_data import as_hf_dataset, balanced_sample

    examples = [json.loads(line) for line in args.examples.read_text(encoding="utf-8").splitlines()]
    if args.balance:
        examples = balanced_sample(examples, args.steps * args.batch * args.accum, args.seed)
    random.Random(args.seed).shuffle(examples)
    started = time.monotonic()
    model, processor = FastVisionModel.from_pretrained(
        MODEL, revision=MODEL_REVISION, use_exact_model_name=True, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=False)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=args.seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    FastVisionModel.for_training(model)
    loaded = time.monotonic() - started

    class TimeCap(TrainerCallback):
        capped = False

        def on_step_end(self, a, state, control, **kw):
            if time.monotonic() - started > args.max_seconds:
                TimeCap.capped = True
                control.should_training_stop = True

    trainer = SFTTrainer(
        model=model, processing_class=processor, train_dataset=as_hf_dataset(examples, args.images, args.max_side),
        data_collator=UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
                                                instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"),
        callbacks=[TimeCap()],
        args=SFTConfig(
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum, max_steps=args.steps,
            learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=max(1, args.steps // 20), weight_decay=0.01,
            optim="adamw_8bit", bf16=True, seed=args.seed, logging_steps=1, report_to="none", output_dir=str(args.output / "trainer"),
            save_strategy="no", remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048, dataloader_num_workers=2))
    torch.cuda.reset_peak_memory_stats()
    trained = time.monotonic()
    trainer.train()
    train_seconds = time.monotonic() - trained
    model.save_pretrained(str(args.output / "adapter"))
    processor.save_pretrained(str(args.output / "adapter"))
    report = {
        "status": "incomplete_time_capped" if TimeCap.capped else "complete",
        "model": MODEL, "revision": MODEL_REVISION, "args": {k: str(v) for k, v in vars(args).items()},
        "examples": len(examples), "completed_steps": trainer.state.global_step,
        "examples_seen": trainer.state.global_step * args.batch * args.accum, "trainable_parameters": trainable,
        "losses": [row["loss"] for row in trainer.state.log_history if "loss" in row],
        "load_seconds": round(loaded, 1), "train_seconds": round(train_seconds, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "unsloth", "unsloth-zoo", "transformers", "trl", "peft")},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "completed_steps", "train_seconds", "peak_vram_gib")}), file=sys.stderr)


if __name__ == "__main__":
    main()
