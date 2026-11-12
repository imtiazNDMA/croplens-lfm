"""Identify one of the five supported crops using the saved crop-only adapter."""

import argparse

from unsloth import FastVisionModel
import torch

from finetune_crops import QUESTION, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    model, processor = FastVisionModel.from_pretrained(
        args.adapter, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(load_image(args.image), text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=8)
    print(processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
