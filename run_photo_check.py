"""Ask a model (the untuned base, or base plus a saved adapter) the photo-check question about each photo of one
evaluation part and record the raw answers by photo hash. Resumes where an interrupted run stopped, and saves as it goes.
Run under an external timeout. Scoring is done separately by evaluate.py."""

import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps

MODEL = "LiquidAI/LFM2.5-VL-1.6B"
MODEL_REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"
HERE = Path(__file__).parent


def photo_path(case, crop_data, extra_images):
    return crop_data / case["path"] if case["source"] in ("val", "test") else extra_images / case["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="the photo-check manifest.csv")
    parser.add_argument("--part", choices=("tune", "final"), required=True)
    parser.add_argument("--crop-data", type=Path, required=True)
    parser.add_argument("--extra-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prompt", choices=("long", "short"), default="long",
                        help="long: the portal's instructions (untuned baselines); short: the question alone (tuned models)")
    parser.add_argument("--max-side", type=int, help="cap the photo's long side, as training did (512)")
    parser.add_argument("--groups", nargs="+", choices=("crop", "other-plant", "not-plant", "poor"),
                        help="ask only these groups of photos (the search scores crop photos only)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seconds", type=float, default=3300)
    args = parser.parse_args()
    if args.part == "final" and not args.adapter and not args.output.exists():
        print("Scoring the final part with an untuned model is allowed, but it is the one-time final part: "
              "pass only the chosen model.", file=sys.stderr)

    import torch
    from unsloth import FastVisionModel
    from evaluate import load_cases

    cases = [c for c in load_cases(args.cases) if c["part"] == args.part and (not args.groups or c["group"] in args.groups)][:args.limit]
    prompt = (HERE / "configs" / "photo_check_prompt.txt").read_text(encoding="utf-8").strip()
    question = (HERE / "configs" / "photo_check_question.txt").read_text(encoding="utf-8").strip()
    record = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {
        "model": MODEL, "revision": MODEL_REVISION, "adapter": str(args.adapter) if args.adapter else None,
        "part": args.part, "prompt": args.prompt, "max_side": args.max_side, "answers": {}, "seconds": {}}
    if record["adapter"] != (str(args.adapter) if args.adapter else None) or record["part"] != args.part:
        sys.exit("The existing output was made with a different adapter or part; choose a new output file.")

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(args.adapter) if args.adapter else MODEL, revision=None if args.adapter else MODEL_REVISION,
        use_exact_model_name=True, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    todo = [c for c in cases if c["id"] not in record["answers"]]
    print(f"{len(cases)} photos in part {args.part}, {len(todo)} to ask", flush=True)
    for done, case in enumerate(todo, 1):
        if time.monotonic() - started > args.max_seconds:
            print("Time cap reached; resume with the same command.", flush=True)
            break
        with Image.open(photo_path(case, args.crop_data, args.extra_images)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        if args.max_side and max(image.size) > args.max_side:
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        if args.prompt == "short":  # what a tuned model was trained on: the question alone
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Check this photo."}]}]
        else:
            messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, text, add_special_tokens=False, return_tensors="pt").to(model.device)
        begin = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        answer = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        record["answers"][case["id"]] = answer
        record["seconds"][case["id"]] = round(time.monotonic() - begin, 3)
        if done % 25 == 0 or done == len(todo):
            record["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
            print(f"{done}/{len(todo)} last: {answer[:90]!r}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
