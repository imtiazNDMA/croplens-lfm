"""Run the saved closed-set crop/condition adapter on one image."""

import argparse
import json
from pathlib import Path

from unsloth import FastVisionModel
import torch

from finetune_conditions import make_question, parse_answer
from finetune_crops import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.adapter / "condition-pilot.json").read_text(encoding="utf-8"))
    model, processor = FastVisionModel.from_pretrained(
        str(args.adapter), max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": make_question(config["classes"])}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(load_image(args.image), prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=48)
    raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if parse_answer(raw, config["classes"]) == "INVALID":
        raise ValueError(f"Invalid or unsupported model answer: {raw!r}")
    print(raw)


if __name__ == "__main__":
    main()
