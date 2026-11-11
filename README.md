# CropLens

**Fine-tuning a compact vision-language model for crop identification and crop health screening—with reproducible evaluations and bounded research-agent experiments.**

CropLens investigates whether [Liquid AI's LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B), adapted with [Unsloth](https://unsloth.ai/docs/models/tutorials/lfm2.5.md) and LoRA, can identify supported crops, reject unsuitable photos, and screen a limited set of crop conditions efficiently.

> **Status: setup in progress.** An isolated WSL environment exists, but large training-dependency downloads did not finish within the bounded attempts. No CropLens model has been trained, no evaluation results are available, and no demo has been deployed. This repository currently contains documentation, not an executable training pipeline.

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
| Compute | Local Ubuntu WSL2 and RTX 6000 Ada discovered; training compatibility not yet reproduced |

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
- [Feasibility research](docs/research/lfm-vision-feasibility.md): primary sources, support evidence, and uncertainties.
- [Local WSL setup](docs/setup-wsl.md): actual environment state, candidate dependencies, bounded resume command, and remaining smoke-test checks.
- [Substack article outline](docs/article/substack-outline.md): the complete teaching narrative we intend to write.
- [Evidence journal](docs/article/evidence-journal.md): actual setup observations and the record used to support that narrative.

The next implementation step is to complete the candidate dependency installation and reproduce an image-conditioned training/save/reload smoke test in the isolated GPU environment. The setup guide explicitly distinguishes commands already run from checks still pending. Freeze the initial task/data specification before agricultural training.

## Learning in public

The companion Substack article will teach the entire engineering process for beginner AI engineers and software engineers transitioning into AI: VLM basics, dataset pitfalls, LoRA, Unsloth, research automation, evaluation metrics, costs, debugging, and deployment.

It will use actual experiments—including negative results—not invented benchmark tables. The final article and model claims will reflect what we successfully reproduce.

## Attribution and artifact rights

The model remains subject to the [selected Liquid checkpoint's license](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/LICENSE). Unsloth and autoresearch retain their own licenses and attribution requirements. Dataset rights are source-specific; acquisition instructions and manifests do not grant image redistribution rights.

A license for original CropLens code has not yet been selected. No model weights, datasets, or private agriculture-portal assets are included in this bootstrap.
