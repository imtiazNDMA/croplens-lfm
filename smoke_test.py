"""Check the Unsloth VLM training path on synthetic colors, not crop accuracy.

Run under an external timeout; outputs include local adapters and a JSON report.
"""

import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import time

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import torch
from huggingface_hub import HfApi
from PIL import Image


MODEL = "LiquidAI/LFM2.5-VL-1.6B"
MODEL_REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"
QUESTION = "What is the dominant color in this image? Answer with one word."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "task": "synthetic-color engineering smoke test; not agricultural evaluation",
        "model": MODEL,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "unsloth", "unsloth-zoo",
                         "transformers", "trl", "peft", "accelerate", "bitsandbytes")
        },
        "checks": {},
    }

    def record(name, value):
        report["checks"][name] = value
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"CHECK {name}: {value}", flush=True)

    def deadline_reached(signum, frame):
        raise TimeoutError("External process deadline reached")

    signal.signal(signal.SIGTERM, deadline_reached)
    try:
        assert torch.cuda.is_available(), "CUDA unavailable"
        assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
        report["model_revision"] = HfApi().model_info(MODEL, revision=MODEL_REVISION).sha
        record("resolved_model_revision", report["model_revision"])

        torch.manual_seed(3407)
        torch.cuda.reset_peak_memory_stats()
        model, processor = FastVisionModel.from_pretrained(
            model_name=MODEL,
            revision=report["model_revision"],
            use_exact_model_name=True,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            return_logits=True,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=8,
            lora_alpha=8,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        model.peft_config["default"].revision = report["model_revision"]
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in name for name in trainable)
        assert not any("vision" in name or "visual" in name for name in trainable)
        record("trainable_adapter_parameters", sum(p.numel() for p in trainable.values()))
        record("only_language_adapters_trainable", True)

        images = [Image.new("RGB", (224, 224), color) for color in ("red", "blue")]
        samples = [{"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]} for image, answer in zip(images, ("red", "blue"))]
        collator = UnslothVisionDataCollator(
            model, processor, train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        batch = collator(samples)
        assert any("pixel" in key for key in batch), "Missing image tensors"
        for row, expected in zip(batch["labels"], ("red", "blue")):
            tokens = row[row != -100]
            assert tokens.numel() > 0, "All labels masked"
            decoded = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            assert decoded == expected, f"Unexpected supervised content: {decoded!r}"
        record("assistant_only_labels", True)
        batch = {key: value.to(model.device) for key, value in batch.items()}

        def image_inputs(image):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": QUESTION},
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            return processor(image, prompt, add_special_tokens=False,
                             return_tensors="pt").to(model.device)

        FastVisionModel.for_inference(model)
        inputs = [image_inputs(image) for image in images]
        assert torch.equal(inputs[0]["input_ids"], inputs[1]["input_ids"])
        with torch.no_grad():
            logits = [model(**item).logits[:, -1].float().cpu() for item in inputs]
        delta = (logits[0] - logits[1]).abs().max().item()
        assert delta > 1e-5, "Changing pixels did not affect output logits"
        record("image_conditioned_logit_max_difference", delta)

        before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        FastVisionModel.for_training(model)
        optimizer = torch.optim.AdamW(trainable.values(), lr=2e-4)
        losses = []
        training_started = time.monotonic()
        for step in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            assert torch.isfinite(loss).item(), "Nonfinite loss"
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all().item()
                       for p in trainable.values()), "Nonfinite gradient"
            optimizer.step()
            losses.append(loss.item())
            print(f"STEP {step + 1}/20 loss={losses[-1]:.6f}", flush=True)
        assert math.isfinite(losses[-1]) and losses[-1] < losses[0]
        assert any(not torch.equal(before[name], p.detach().cpu())
                   for name, p in trainable.items()), "Adapters did not change"
        report["training_seconds"] = round(time.monotonic() - training_started, 3)
        report["losses"] = losses
        record("finite_loss_gradients_and_adapter_updates", True)
        record("tiny_batch_loss_decreased", True)

        FastVisionModel.for_inference(model)
        with torch.no_grad():
            expected = model.generate(**inputs[0], do_sample=False, max_new_tokens=8)
        report["red_prediction"] = processor.tokenizer.decode(
            expected[0, inputs[0]["input_ids"].shape[1]:], skip_special_tokens=True)
        adapter = args.output / "adapter"
        model.save_pretrained(adapter)
        processor.save_pretrained(adapter)
        expected = expected.cpu()
        del optimizer, trainable, before, batch, inputs, logits, loss, model, processor, collator
        gc.collect()
        torch.cuda.empty_cache()

        model, processor = FastVisionModel.from_pretrained(
            model_name=str(adapter), max_seq_length=1024,
            dtype=torch.bfloat16, load_in_4bit=False,
        )
        FastVisionModel.for_inference(model)
        with torch.no_grad():
            actual = model.generate(**image_inputs(images[0]), do_sample=False, max_new_tokens=8)
        assert torch.equal(expected, actual.cpu()), "Save/reload generation changed"
        record("adapter_processor_reload_identical_generation", True)
        report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        report["status"] = "passed"
        record("completed", True)
    except Exception as error:
        report["status"] = "timed_out" if isinstance(error, TimeoutError) else "failed"
        record("error", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
