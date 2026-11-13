# Actual five-crop LoRA fine-tune: crop-pilot-002

Completed locally on 2026-11-11. This is a crop-only pilot, separate from the existing structured photo-check experiments.

## Measured results

| Metric | Untuned | Fine-tuned |
| --- | ---: | ---: |
| Validation accuracy | 53% (53/100) | **86% (86/100)** |
| Macro F1 | 0.524455 | **0.861105** |
| Invalid answers | 0 | 0 |

Both evaluations used the same 100 images, prompt, preprocessing, greedy decoding, and scoring. The 33-percentage-point accuracy gain is an in-corpus development result, not locked-test or field performance.

| Crop | Validation images | Untuned recall | Fine-tuned recall |
| --- | ---: | ---: | ---: |
| Cotton | 20 | 55% | 95% |
| Maize | 20 | 60% | 95% |
| Rice | 20 | 75% | 100% |
| Sugarcane | 20 | 55% | 65% |
| Wheat | 20 | 20% | 75% |

## Configuration

- Base: `LiquidAI/LFM2.5-VL-1.6B`, revision `919fde3d022e3f90a4716006f993938ee8c2eb97`.
- Data: 500 retained training images, 100/crop; 100 retained validation images, 20/crop; deterministic seed 3407.
- Source manifest SHA-256: `f3e404e075ea9289545afb080b84a54e25f5016f0407560c5ffe869a750ec6ed`.
- All 600 selected files decoded and matched their recorded hashes. Training hashes/groups were checked against existing validation/test assignments. No test images were decoded or scored.
- Training: 200 direct AdamW steps, batch 4, learning rate 2e-4, rank/alpha 8, language-side LoRA, BF16, frozen vision encoder, assistant-only targets.
- Trainable parameters: 4,571,136; 800 image presentations across 500 training images.
- Training time: **86.736 seconds**; full run **147.283 seconds**, excluding interpreter startup and the separate reload check.
- Peak PyTorch allocated memory: **3.315 GiB**, RTX 6000 Ada.
- Preprocessing: apply orientation, discard EXIF without modifying source files, cap long side at 384 px.
- Prompt: `Identify the crop in this image. Answer with exactly one of: Cotton, Maize, Rice, Sugarcane, Wheat.`

The first preparation attempt (`crop-pilot-001`) failed before model loading because Pillow attempted to serialize malformed EXIF. Reading orientation and discarding metadata fixed it; four focused tests passed, including orientation and held-out overlap checks.

## Saved artifacts and inference

- [Complete metrics, raw answers, confusion matrices, and all losses](../reports/crop-pilot-002.json).
- Adapter/processor: `outputs/crop-pilot-002/adapter/` (local, ignored by Git).
- Exact selected rows: `outputs/crop-pilot-002/subsets.json` (local, ignored by Git).
- Training: [finetune_crops.py](../finetune_crops.py).
- Inference: [predict_crop.py](../predict_crop.py).

The saved adapter/processor loaded in a separate process and predicted `Cotton` on the first validation image, matching the in-memory prediction. This is one reload check, not an independent benchmark.

From the repository root in Ubuntu WSL2:

```bash
timeout --signal=TERM --kill-after=10s 120s \
  env HF_HUB_OFFLINE=1 \
  "$HOME/croplens-venv/bin/python" predict_crop.py \
  --adapter outputs/crop-pilot-002/adapter \
  --image "/path/to/your/crop-photo.jpg"
```

The adapter requires the cached base weights; it is not a standalone merged model.

## Limits and next training step

This closed-set model chooses among five crops. It does not identify diseases, reject nonplants, or recommend treatments. One seed and 20 validation images/crop do not establish generalization.

This pilot used the original retained training manifest, **not** the newer near-duplicate-cleaned pool used in the separate photo-check experiments. Exact/group checks do not exclude unrecorded augmented siblings or near copies outside the original threshold. Source/size shortcuts remain. Do not compare this 100-image crop-only score directly with the 700-photo structured photo-check results.

Next substantive run: use the already prepared cleaned pool and extend to supported crop–condition labels with a matched baseline. An additional exhaustive audit is not a prerequisite to that development run.
