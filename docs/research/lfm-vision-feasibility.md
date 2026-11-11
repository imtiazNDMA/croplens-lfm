# Liquid vision fine-tuning: feasibility and evidence

Research date: 2026-11-11. This is the standalone version of the initial source review. No model download, training, agricultural evaluation, or dataset audit has been performed for CropLens. Published example outputs are not our measurements.

## Recommendation

Start with **`LiquidAI/LFM2.5-VL-1.6B`**, image-conditioned supervised fine-tuning, and language-side LoRA while freezing the vision encoder. Follow the explicit Unsloth vision recipe before attempting QLoRA, encoder training, or quantized export. [1–4]

## What the evidence establishes

| Question | Evidence and implication |
| --- | --- |
| Does Unsloth publish an exact vision recipe? | Yes: `FastVisionModel`, `UnslothVisionDataCollator`, SFTTrainer, `load_in_4bit=False`, and `finetune_vision_layers=False`. [1] |
| Is there published training output? | The pinned official notebook records a completed 30-step LaTeX OCR run on an NVIDIA L4/Linux. This establishes an example run, not crop accuracy or reproduction here. [3] |
| Are installation instructions a verified lock? | No. The notebook's installation pins and saved runtime output differ, and current model guidance has evolved. Reproduce and lock the versions actually used. [1–3] |
| Does text LFM training establish vision training? | No. Multimodal processing, labels, adapters, backward passes, and save/reload require exact-checkpoint checks. [1–4] |
| Is the checkpoint choice a claim of newest/best model? | No. It prioritizes inspected Unsloth training evidence. Liquid lists newer vision models and deprecates older LFM2-VL checkpoints. [5–6] |
| Is 4-bit training or GGUF export verified here? | No. These need separate image-conditioned regression checks; code snippets alone do not prove a successful round trip. [1–3] |

The model card lists eight supported languages, excluding Urdu. Urdu crop-management output needs separate evaluation. The model is an image-conditioned generator, not a crop specialist. [2]

## Compatibility gate

Before search, verify image-conditioned inference, finite forward/backward loss, actual LoRA updates, intended parameter freezing, correct assistant-label masking, intact image tokens, image-dependent predictions, tiny-batch overfitting, and adapter/processor save/reload.

Liquid's TRL guidance is a documented fallback if the exact Unsloth recipe cannot be reproduced. Do not silently change frameworks while continuing to describe the run as Unsloth training. [4]

## Dataset evidence

| Source | Scope and implications |
| --- | --- |
| PlantVillage author release | Controlled-condition leaf images, 14 crops and 38 crop–condition classes; metadata declares CC-BY-SA-3.0. Prose and split counts differ by one image, so inspect the actual revision/manifest. [7–8] |
| PlantDoc author release | Non-laboratory, internet-scraped images; the main repository is cropped classification data and a separate repository supplies object-detection data. The classification repository declares CC-BY-4.0; item-level rights still depend on provenance. [9–10] |

The inspected classification taxonomies contain maize/corn, but do not supply all five proposed crops: cotton, maize, rice, sugarcane, and wheat. Neither dataset establishes Pakistan field performance or open-world plant identification. [7,9]

The original PlantVillage paper reports 99.35% held-out accuracy under controlled conditions but about 31% on two small external-image sets. These are the paper's classifier results, not LFM results. They illustrate the danger of equating a random in-corpus split with field readiness. [8]

Audit all acquisition rights and cross-source duplicates. Keep derived crops and related views in the same split, include unsupported/ambiguous inputs, and reserve an independent field holdout when available.

## Bounded autoresearch adaptation

Karpathy's original repository trains a small GPT model with a five-minute training budget and selects on validation bits per byte. It is not an Unsloth VLM SFT pipeline. [11]

CropLens will adapt the experiment/measure/keep-or-discard workflow using frozen agricultural evaluators, fixed training work, validation-only selection, protected tests, and explicit session deadlines. See the [canonical plan](../../fine-tuning.md) for the current policy; that plan takes precedence over illustrative upstream defaults.

## Licenses and treatment boundary

The checkpoint is governed by the **LFM Open License v1.0**, not Apache-2.0. Retain the exact terms and notices applicable to the selected revision and intended distribution. The current Liquid guide describes commercial revenue conditions; do not assume all downstream deployments qualify identically. [12]

Model licensing, original code licensing, and dataset licensing are separate questions. A public dataset repository does not automatically make every image redistributable.

Later treatment RAG should retrieve current authoritative registration and product-label sources. Image SFT does not establish current pesticide knowledge. Source-corpus research and market survey have not been performed. [2]

## Local hardware discovery

On 2026-11-11, Windows and Ubuntu WSL2 both exposed an NVIDIA RTX 6000 Ada Generation with 49,140 MiB total VRAM and driver 610.88. Ubuntu reported Python 3.12.3; `uv` was not on PATH. Only 9,560 MiB VRAM was free at the WSL observation. These are hardware-discovery results, not training compatibility or performance measurements. The commands and observations are recorded in the [evidence journal](../article/evidence-journal.md).

## Primary sources

Reviewed during initial planning; pin the model/data/software revisions actually used at implementation time.

1. Unsloth vision-specific guide: https://unsloth.ai/docs/models/tutorials/lfm2.5.md
2. Selected model card: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
3. Official notebook with saved outputs, pinned: https://github.com/unslothai/notebooks/blob/c834e1741666c50e3b7a57f56e7f6855229ac2ac/nb/LFM2.5_VL_(1.6B)-Vision.ipynb
4. Liquid TRL guidance: https://docs.liquid.ai/lfm/fine-tuning/trl.md
5. Liquid vision library: https://docs.liquid.ai/lfm/models/vision-models.md
6. Liquid deprecations: https://docs.liquid.ai/lfm/help/deprecations.md
7. Author PlantVillage dataset card: https://huggingface.co/datasets/mohanty/PlantVillage
8. Original PlantVillage experiment: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full
9. Author PlantDoc classification release: https://github.com/pratikkayal/PlantDoc-Dataset
10. Author PlantDoc detection release: https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
11. Karpathy autoresearch: https://github.com/karpathy/autoresearch
12. Checkpoint license: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/LICENSE and license guide: https://docs.liquid.ai/lfm/help/model-license.md
