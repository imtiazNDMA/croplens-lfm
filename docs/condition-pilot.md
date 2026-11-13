# Crop-and-condition LoRA pilot: condition-pilot-001

Completed locally on **2026-11-13**, using the existing cleaned training pool. This is a closed-set development experiment, not a field or locked-test evaluation.

## Matched before/after results

Both evaluations used the same 280 validation images, supported-class list, prompt, preprocessing, and greedy decoding. Unsupported/invalid output counted as failure, without fence stripping or fuzzy label matching.

| Metric | Untuned base | Fine-tuned |
| --- | ---: | ---: |
| Crop accuracy | 57.50% (161/280) | **99.29% (278/280)** |
| Exact crop–condition accuracy | 27.50% (77/280) | **87.86% (246/280)** |
| Joint macro F1 over 14 crop–condition classes | 0.231070 | **0.878469** |
| Condition-only macro F1 over 11 condition names | 0.254018 | **0.869507** |
| Invalid JSON/unsupported pair | 1 | 0 |

Joint accuracy/F1 require the correct crop and condition together. Condition-only F1 collapses the four healthy classes into one condition name; it does not require correct crop identity. The joint metrics are the primary screening measures.

### Recall by crop–condition class

Each class has 20 validation images.

| Class | Untuned | Fine-tuned |
| --- | ---: | ---: |
| Cotton aphid | 5% | 70% |
| Cotton bacterial blight | 15% | 80% |
| Cotton healthy | 85% | 100% |
| Cotton wilt | 15% | 95% |
| Maize armyworm | 0% | 95% |
| Maize common rust | 70% | 100% |
| Maize gray leaf spot | 15% | 95% |
| Maize healthy | 60% | 95% |
| Sugarcane healthy | 60% | 100% |
| Sugarcane mosaic | 0% | 60% |
| Sugarcane red rot | 10% | 85% |
| Sugarcane yellow rust | 0% | 60% |
| Wheat healthy | 0% | 100% |
| Wheat powdery mildew | 50% | 95% |

Sugarcane mosaic/yellow rust and cotton aphid are the clearest remaining weaknesses. Twenty examples per class do not establish precise population-level reliability.

## Data and training

- **700 training images:** 50 per class, sampled only from `data/pool-001/training_pool.csv`.
- **280 validation images:** 20 per class from the original retained validation split. Existing splits were not regenerated and test images were not decoded or scored.
- Each pool row was matched against the original retained training manifest. Selected training hashes/recorded groups were checked against all held-out assignments; all 980 selected files decoded and matched their SHA-256.
- Rice was excluded because healthy-rice examples are absent. Healthy examples are included for every supported crop. The 14 active labels are frozen in [condition-pilot.json](../configs/condition-pilot.json), without changing the broader frozen taxonomy.
- Base: `LiquidAI/LFM2.5-VL-1.6B`, pinned revision `919fde3d022e3f90a4716006f993938ee8c2eb97`; this run starts from the base, not the earlier crop-only adapter.
- BF16, frozen vision encoder, language-side LoRA rank/alpha 8, **4,571,136 trainable parameters**, assistant-only JSON targets, AdamW at 2e-4, seed 3407.
- **400 steps × batch 4:** 1,600 image presentations, approximately 2.29 passes over the 700-image subset. Long image side capped at 384 px using the existing orientation/metadata-safe loader.
- Training loop: **114.248 seconds**. Full preparation/baseline/train/evaluation run: **347.323 seconds**, excluding interpreter startup and the later independent reload check.
- Untuned validation: 78.269 seconds; tuned validation: 112.364 seconds. Tuned structured answers are often longer, so latency is not an equal-output-length benchmark.
- Peak PyTorch allocated memory: **3.387 GiB** on the RTX 6000 Ada. The run used cached weights offline and a 1,200-second external deadline.

## Artifacts and inference

- [Complete report: metrics, confusion matrices, raw answers, losses, hashes, and versions](../reports/condition-pilot-001.json).
- Training code: [finetune_conditions.py](../finetune_conditions.py).
- Single-image inference: [predict_condition.py](../predict_condition.py).
- Adapter/processor and the matching condition config: `outputs/condition-pilot-001/adapter/` (local, Git-ignored).
- Selected-row manifest: `outputs/condition-pilot-001/subsets.json` (local, Git-ignored).

From the repository root in Ubuntu WSL2:

```bash
timeout --signal=TERM --kill-after=10s 120s \
  env HF_HUB_OFFLINE=1 \
  "$HOME/croplens-venv/bin/python" predict_condition.py \
  --adapter outputs/condition-pilot-001/adapter \
  --image "/path/to/a/supported-crop-photo.jpg"
```

The saved adapter was reloaded in a separate process on a selected validation image and reproduced its in-memory answer:

```json
{"crop":"Cotton","condition":"aphid"}
```

This is a single-case persistence check, not another independent evaluation. The base weights must be cached locally. All **54 unit tests passed**, including five new condition-specific tests.

## Limits and next experiment

The cleaned pool removes known near-copy/augmentation risks but cannot prove independent physical leaves or source diversity. Source/style shortcuts, imperfect annotations, and domain shift remain. This model always chooses from the supported closed set: it does not validate arbitrary crops, reject nonplants/unknown conditions, confirm a biological cause, or prescribe treatments.

No final test, independent field evaluation, confidence calibration, or multi-seed confirmation has run. Results are not directly comparable to the earlier 100-image five-crop pilot (different crops, labels, validation set, and prompt). Next: a short bounded search using joint validation macro F1, with special attention to the weak classes; final-test evaluation stays deferred until model selection is finished.
