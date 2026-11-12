# CropLens

**Fine-tuning a compact vision-language model for crop identification and crop health screening—with reproducible evaluations and bounded research-agent experiments.**

CropLens investigates whether [Liquid AI's LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B), adapted with [Unsloth](https://unsloth.ai/docs/models/tutorials/lfm2.5.md) and LoRA, can identify supported crops, reject unsuitable photos, and screen a limited set of crop conditions efficiently.

> **Actual crop fine-tune completed:** the crop-only LoRA pilot improved matched 100-image validation accuracy from **53% to 86%** and macro F1 from **0.524 to 0.861**. The adapter is saved locally. These are in-corpus development results, not field or disease accuracy. See the [crop pilot](docs/crop-pilot.md).

## The question

Can a small VLM improve on its untuned crop-photo predictions under a fixed compute budget—and when does a specialist classifier remain the better choice?

We will measure that question rather than assume fine-tuning helps. The project combines data engineering, parameter-efficient training, controlled experiments, model evaluation, and an eventual application demo.

## Planned capabilities

1. **Photo check:** identify the main supported crop, flag poor image quality, and return other/unclear outcomes.
2. **Crop health screening:** separately investigate supported healthy, disease, and pest classes.
3. **Budgeted research loop:** adapt [Karpathy's autoresearch workflow](https://github.com/karpathy/autoresearch) to supervised VLM training, with hard limits and no overnight runs.
4. **Reproducible demo and report:** compare against untuned LFM and a compatible specialist classifier where available.
5. **Later treatment RAG:** retrieve current, authoritative crop-management sources after a market survey. This is not part of initial vision fine-tuning.

The proposed crop scope is cotton, maize, rice, sugarcane, and wheat, subject to a data audit. Image-level identification is not bounding-box object detection. Photo-based screening is not confirmation of a biological cause.

## Initial technical direction

| Area | Proposed approach |
| --- | --- |
| Base model | `LiquidAI/LFM2.5-VL-1.6B` |
| Training | Image-conditioned SFT with Unsloth and language-side LoRA |
| Vision encoder | Frozen for the initial experiment |
| Precision | Native 16-bit base weights first; QLoRA only after separate verification |
| Evaluation | Macro F1, per-class recall, rejection behavior, output validity, latency, memory, and cost |
| Data controls | Provenance review, grouped splits, deduplication, and locked final tests |
| Experiment controls | Fixed training work; at most six trials and two hours per manually started session |
| Compute | Local Ubuntu WSL2 and RTX 6000 Ada; actual crop LoRA training completed |

The original autoresearch code trains a GPT model, not this VLM. CropLens adapts its experiment/measure/keep-or-discard idea; it does not claim an unchanged upstream training integration.

## Workflow

```text
Audit data → freeze splits → reproduce training setup → measure baselines
    → bounded LoRA experiments → confirm across seeds → locked evaluation
    → analyze failures → build demo → publish reproducible findings
```

Every reported number will link to a run, data revision, evaluator, and hardware context. Field-image results will be separated from in-corpus results. Invalid model answers will count as failures, and accepted-case metrics will include coverage.

## Start here

- [Living project plan](fine-tuning.md): scope, phases, decision log, and current checklist.
- [Crop-only fine-tune and inference](docs/crop-pilot.md): measured before/after results, saved adapter, and limitations.
- [Feasibility research](docs/research/lfm-vision-feasibility.md): primary sources, support evidence, and uncertainties.
- [Local WSL setup](docs/setup-wsl.md): verified setup, bounded smoke-test commands, and remaining agricultural training checks.
- [Smoke-test report](reports/smoke-002.json): actual losses, timing, memory, versions, and verification outcomes.
- [Environment snapshot](requirements-wsl-cu128.txt): exact observed Linux/Python 3.12/CUDA 12.8 package versions.
- [Substack article outline](docs/article/substack-outline.md): the complete teaching narrative we intend to write.
- [Evidence journal](docs/article/evidence-journal.md): actual setup observations and the record used to support that narrative.

Run the crop-only pilot with `finetune_crops.py` and single-image inference with `predict_crop.py`. Separate structured photo-check work is documented in `docs/baselines.md`. The next training extension is supported crop–condition identification using the existing cleaned pool; another exhaustive audit is not a prerequisite to that development run.

## Learning in public

The companion Substack article will teach the entire engineering process for beginner AI engineers and software engineers transitioning into AI: VLM basics, dataset pitfalls, LoRA, Unsloth, research automation, evaluation metrics, costs, debugging, and deployment.

It will use actual experiments—including negative results—not invented benchmark tables. The final article and model claims will reflect what we successfully reproduce.

## Attribution and artifact rights

The model remains subject to the [selected Liquid checkpoint's license](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/LICENSE). Unsloth and autoresearch retain their own licenses and attribution requirements. Dataset rights are source-specific; acquisition instructions and manifests do not grant image redistribution rights.

A license for original CropLens code has not yet been selected. No model weights, datasets, or private agriculture-portal assets are tracked; smoke-test adapters and generated caches remain local.
