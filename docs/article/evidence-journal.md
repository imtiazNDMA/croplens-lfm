# CropLens evidence journal

Record actual decisions, commands, observations, failures, and artifact references here. This journal supports the [article outline](substack-outline.md); it is not a substitute for machine-readable training logs or evaluation results.

## 2026-11-11 — Repository bootstrap and compute discovery

### Question

Can we start a standalone, reproducible project with local GPU compute rather than immediately renting a GPU?

### Actions and observations

1. Verified the public GitHub repository `imtiazNDMA/croplens-lfm`; it had no commits or files.
2. Cloned it as a separate checkout beside the agriculture portal. No portal application files or Git history were copied.
3. Moved the living plan into this checkout and left a pointer in the original workspace.
4. Added a standalone README, feasibility memo, article outline, journal, and ignore rules.
5. Queried Windows and Ubuntu WSL2 GPU visibility and checked Python/tool availability.

### Commands actually run

From Windows PowerShell:

```powershell
gh repo view imtiazNDMA/croplens-lfm --json nameWithOwner,url,isEmpty,defaultBranchRef
gh repo clone imtiazNDMA/croplens-lfm "<parent-directory>\croplens-lfm"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
wsl --status
wsl --list --verbose
wsl -d Ubuntu -- nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
wsl -d Ubuntu -- python3 --version
wsl -d Ubuntu -- uv --version
```

The clone command above substitutes a portable parent-directory placeholder for the machine-specific local path. Run each command separately; a failed availability check should not be mistaken for a successful installation.

### Observed output

| Check | Observation |
| --- | --- |
| GPU model | NVIDIA RTX 6000 Ada Generation |
| Total VRAM | 49,140 MiB |
| NVIDIA driver | 610.88 |
| WSL distribution | Ubuntu, running, version 2 |
| GPU accessible inside Ubuntu | Yes, `nvidia-smi` succeeded |
| Free VRAM at WSL query | 9,560 MiB |
| Ubuntu Python | Python 3.12.3 |
| Ubuntu `uv` | Command not found on current PATH |

### Interpretation and decision

Local Ubuntu WSL2 is a promising first training environment. This observation does not prove PyTorch CUDA, Unsloth compatibility, training fit, or throughput. Most GPU memory was in use at the observation; confirm an available training window instead of assuming all 48 GB is free.

No packages, models, or datasets were installed/downloaded, and no other GPU workloads were stopped. No paid compute was booked. No CropLens results exist yet.

### Next evidence to capture

- User decision on a local GPU window and total compute/API spending ceiling.
- Reproduced isolated environment and exact package versions.
- Dataset rights/coverage audit and frozen initial taxonomy.
- Image-conditioned forward/backward and save/reload smoke-run artifacts.

## 2026-11-11 — Free GPU and isolated setup attempt

### Question

Can we reproduce the exact-checkpoint vision training path now that the local GPU is available?

### Observations

- The user confirmed the GPU was free. A fresh WSL query reported 46,493 MiB free of 49,140 MiB total; a final check reported 46,596 MiB free.
- Created `/home/user/croplens-venv` using Ubuntu's Python 3.12.3, without changing system Python or the portal environment.
- Installed pip 26.3 and uv 0.13.7 into that environment.
- The pip/uv bootstrap first exceeded a 120-second terminal timeout; its retry completed but included a temporary DNS resolution failure and slow download throughput.
- An explicit Unsloth/PyTorch dependency dry run resolved 108 packages successfully.
- A 900-second installation attempt and a 600-second cached retry did not complete. The first also reached the terminal's outer timeout. Large NVIDIA runtime downloads remained incomplete; many other wheels were cached.
- Afterward, `pip list` still showed only pip and uv. `pgrep -af "uv pip install"` found no remaining installer process.
- No model/dataset download, inference, optimizer step, or adapter save/reload was performed. This is a setup blocker, not a model-training failure or a crop-performance result.

### Commands and candidate versions

Environment creation/bootstrap, from PowerShell:

```powershell
wsl -d Ubuntu -- python3 -m venv /home/user/croplens-venv
wsl -d Ubuntu -- /home/user/croplens-venv/bin/python -m pip install --upgrade pip uv
```

The resolved candidate stack included torch 2.9.1+cu128, torchvision 0.24.1+cu128, Unsloth 2026.11.3, unsloth-zoo 2026.11.4, Transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2, Accelerate 1.15.0, and bitsandbytes 0.50.2. These are resolved versions, not a verified training lock.

The final bounded installation attempt was:

```powershell
wsl -d Ubuntu -- timeout --signal=TERM --kill-after=10s 600s env UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=1 UV_CONCURRENT_DOWNLOADS=4 /home/user/croplens-venv/bin/uv pip install --python /home/user/croplens-venv/bin/python "unsloth[huggingface]" bitsandbytes pillow "torch==2.9.1" "torchvision==0.24.1" --torch-backend=cu128
```

### Decision and next step

Stop downloads at the bounded setup limit; leave no installer running unattended. Preserve the cache, record a resumable command, and finish the setup documentation. See the [WSL setup report](../setup-wsl.md) for the continuation and pending CUDA/VLM checks. No paid compute was booked.

Teaching point: dependency resolution, installation, CUDA visibility, and a working multimodal backward pass are four different milestones. A green resolver output is not a successful fine-tune.

## 2026-11-11 — Installation complete and synthetic LoRA smoke test passed

### Setup outcome

A manually resumed 900-second dependency installation completed with exit status 0. Package preparation took approximately 6 minutes 33 seconds and installation of 108 packages took 556 ms, excluding initial resolution. `pip check` reported no broken requirements. Unsloth imported successfully, and CUDA tensor execution plus BF16 support passed on the RTX 6000 Ada.

The starting GPU query reported 46,511 MiB free. A first inline CUDA check lost quotation marks across PowerShell/WSL and raised a `NameError`; using `device=0` resolved that shell-quoting mistake. Do not present it as a CUDA failure.

### Model download and authentication

The first model download stalled before training and was interrupted. No process remained afterward; its local report was marked interrupted rather than left as running. A second bounded attempt used transient authentication and `HF_HUB_DISABLE_XET=1` and completed. Because those changes occurred together, the download improvement cannot be attributed to authentication alone. Credentials are not part of source, reports, examples, or this journal.

### Actual engineering test

`smoke_test.py` used two synthetic 224×224 solid-color images, a single identical user question, and the assistant targets `red` and `blue`. It used Unsloth's model/processor and vision collator with explicit assistant-only masking. A direct PyTorch AdamW loop performed 20 updates at learning rate 2e-4 with rank/alpha 8, seed 3407, native BF16 weights, and frozen non-adapter parameters.

| Measurement/check | Actual result |
| --- | --- |
| Model revision | `919fde3d022e3f90a4716006f993938ee8c2eb97` |
| Trainable parameters | 4,571,136; language-side LoRA only |
| Assistant-only label check | Passed on both expected answers |
| Image conditioning | Identical input text tokens, different pixels; max next-token logit difference 10.125 |
| Loss | 3.7499771118164062 → 0.00031462765764445066 |
| Gradients and actual adapter updates | Finite gradients and changed adapter weights verified |
| Training loop | 23.451 seconds |
| Whole successful run | 487.244 seconds, including download/loading/reload |
| PyTorch peak allocated / reserved | 3.224 / 3.299 GiB |
| Greedy red-image answer | `red` |
| Adapter and processor save/reload | Identical greedy generated token IDs for the tested prompt |
| Process exit | 0 |

The [retained JSON report](../../reports/smoke-002.json) contains the observed versions and every training loss. The [setup guide](../setup-wsl.md) includes the bounded reproduction command; the [environment snapshot](../../requirements-wsl-cu128.txt) captures exact package versions. Adapters and generated compile caches are ignored local artifacts. The model revision is pinned in the script for subsequent runs.

Follow-up verification: Python compilation and the smoke-test CLI help passed. A reinstall **dry run** from the version snapshot resolved the same 109 installed packages, including uv; it did not reinstall them. A source scan found no Hugging Face token patterns outside ignored outputs/caches. These checks do not substitute for a fresh-machine reinstall.

### Limits and next step

This is a tiny-batch pipeline check, not crop/disease evaluation. We did not verify SFTTrainer, measure field performance, compare baselines, or run multi-seed experiments. The runtime warned that it used a reference convolution because `causal_conv1d` was absent; it was correct and fast enough for this check, so no new optimized kernel was installed.

Next: freeze the crop/photo-check contract, audit the existing/public data for labels, provenance and split leakage, then build and verify agricultural training/evaluation before any autoresearch search. A fresh reinstall from the version snapshot remains untested.

## 2026-11-11 — Dataset discovery and interrupted full audit

Located the existing crop manifest/taxonomy and raw `dataset_filtered` corpus in the agriculture portal workspace. Source documents report 17,952 entries and 11,453 retained images, but those counts have not yet been independently confirmed by a completed CropLens audit. The source report lists no healthy-rice class. Sample filenames include augmentation-like names; missing parent-photo provenance is a leakage risk to investigate, not proven cross-split leakage.

Also located the existing photo-check evaluation manifest and builder. Its crop images originate from the crop manifest's validation/test splits; degraded images explicitly reference validation parents. Treat these as existing evaluation assets, not new training images. Inspect actual lineage/overlap before any reuse.

Implemented a read-only `audit_dataset.py` plus five focused unittest checks for path traversal, exact split leakage, near-hash thresholds, missing/corrupt/hash-mismatched files, and a full tiny-fixture audit without rewriting its input manifest. All five tests passed in 0.126 seconds.

The real-data audit was executed under `timeout --signal=TERM --kill-after=10s 600s`. It emitted Pillow warnings about palette transparency and corrupt EXIF metadata, then exited with status 124. Those warnings alone do not establish undecodable images. The script writes its aggregate report only after completing, so **no `reports/dataset-audit-001.json` exists**. No audit process remained at the follow-up check. No source images, manifests, or splits were rewritten.

Session paused at the user's request for a Claude handoff. Next: profile the audit's image-I/O and hash-comparison stages, add bounded progress/checkpointing or split metadata and image verification into resumable passes, then rerun. Do not claim audit counts, clean splits, or a proposed screening taxonomy have been validated yet. Dataset report, proposed taxonomy, and source-rights review are still pending.

## 2026-11-11 — Audit completed, and what it found

**Question:** is the existing corpus safe to train and score on?

**The timeout, diagnosed.** The 600-second audit had not hung: the second attempt, with stage timings added, completed in 594 s. Timing the stages separately showed the per-image work (hash, decode, dHash) takes 73 s for all 17,952 images with eight threads, and the near-hash comparison under a second. The cost was manifest validation (426 s) and the photo-check pass (55 s), both calling `image_path()`, which resolved the full path against the Windows-mounted drive three times per image. Resolving each directory once gave 160 s with a report identical to the first. Two new tests pin the cause (resolve count) and the safety property (symlinks cannot escape the image root). Lesson for the article: measure the stages before raising a timeout.

**Confirmed:** 17,952 manifest rows, 11,453 kept (8,013 train, 1,720 val, 1,720 test), 37 retained classes, 0 missing, undecodable or hash-mismatched files, 0 exact or manifest-group overlap across splits, 0 near-duplicates at the original 12-bit threshold. No healthy rice class.

**Found, and not visible in the headline numbers:**

| Finding | Evidence |
| --- | --- |
| Near-duplicates cross the splits beyond 12 bits | 40 / 28 / 6 train-val / train-test / val-test pairs at 16 bits; 105 / 87 / 22 at 24 bits; closest pair is 13 bits. All 192 pairs within 24 bits share a label; 151 are rice tungro. Two inspected pairs were the same photograph |
| Augmented photos have unrecorded parents | 430 kept augmentation-named images (307 train, 69 test, 54 val), all in four classes; dHash cannot match flips or rotations |
| Image size gives the crop away | Size alone names the crop for 81.7% of test photos (majority baseline 31.7%); 1,609 of 1,610 photos at 300x300 are rice |
| Provenance is not recorded | None of the six provenance fields exist; at least one image is watermarked "PhilRice" |

Full tables and the proposal: [dataset audit](../dataset-audit.md). Reports: `reports/dataset-audit-001.json`, `-002.json`, `dataset-shortcuts-001.json`.

**Not established:** how many of the near pairs are true duplicates (two inspected), whether augmented parents leak, label correctness, and rights to any source. Nothing was trained; no model metric exists yet.

**Next:** the user decides the split proposal; then rebuild the training pool, re-audit it, and run the baselines with the three-way scoring (existing test, shared-size subset, size-normalised copies).

## What to record in future sessions

For each substantive session, record the date and question; actual commands; code, model, data, and environment revisions; observed output; errors and fixes; decision and rationale; run/artifact links; timing and cost; and unresolved questions. Caption figures with dataset/split, units, run IDs, and the limitation they illustrate. Use only shareable images and sanitized logs.

## 2026-11-11 — Cleaned training pool

Decisions: accepted the audit's split proposal (existing val/test untouched; report full, no-shared-size and size-normalised scores) and the 18-class screening start set.

Built `prepare.py` (3 tests; 15 pass in total). It removes train images within 32 bits of any val/test image and train images with augmentation-style names. Result: 8,013 → 7,469 train images (240 near copies, 304 augmentation-named; the audit counted 307 augmentation-named, but 3 of those were already near copies). Re-check on the output: 0 pairs within 32 bits of val or test. Heaviest losses: `rice_tungro` 177 near copies (664 → 487), and the four augmentation-heavy classes, e.g. `cotton_aphid` 146 → 55 and `maize_armyworm` 172 → 75. Output: `data/pool-001/training_pool.csv` and `training_pool_report.json` (per-class removals). Lesson for the article: removing leakage costs the small classes most, so the thinnest classes need a support caveat.

Froze `configs/taxonomy.yaml` (taxonomy v1: the portal's 39 classes unchanged, plus the 18-class screening start set) and `contract.py` (photo-check contract v1: `is_plant`, `crop`, `other_plant`, four quality flags; strict parser with named failure reasons, e.g. a crop named for a non-plant is invalid). 5 new tests; 20 pass in total.

## 2026-11-11 — Baselines and trainer

Scored three systems on the 700 `tune` photos (details: `docs/baselines.md`). Untuned LFM2.5-VL-1.6B: crop identification 5.0%, 67.8% of crop photos wrongly stopped (it says "Other"), 23.3% invalid answers. Qwen (portal, tuned on these photos): 51.2%. ConvNeXt: 98.4% with the crop inferred, but the audit's size shortcut applies to it. First attempt to run the untuned model in the background silently died when the WSL launcher exited; the fix was to run the foreground command as a harness background task. First untuned prompt copied its placeholder text back, so the prompt now shows two worked example answers.
Built `sft_data.py` (8,091 examples from the pool; 622 degraded copies, strength randomised) and `train.py`. Trainer integration bumps: TRL renamed `tokenizer` to `processing_class`, and TRL wants a `datasets.Dataset`, so images are decoded lazily through `set_transform`. 4-step smoke run: loss 2.78 to 1.98, 44 s, 3.25 GiB peak. No negatives (other plants, non-plants) in the training data yet; waiting on the source/licence decision.

## 2026-11-11 — First fine-tuned reference run

`ref-001` (150 steps x 16, rank 16, lr 2e-4): trained in 177 s, 3.35 GiB. On the 700 `tune` photos: crop identification 88.6% (untuned 5.0%, Qwen 51.2%, ConvNeXt 98.4%), 1.0% invalid answers, quality flags 98% on poor photos and 0.2% on clean ones. Stops no other plants or non-plants (no negatives in the data), so those bars are not yet testable. Wheat (77) and rice (84) are the weak crops. Not yet checked against the image-size shortcut. Training was about 10x faster than the first 4-step profile suggested (that profile shared the GPU with another run), which makes the planned 6-trial search cheap.

## 2026-11-11 — Shortcut check on ref-001

Normalised copies (square centre crop, 512 px, JPEG 85, no metadata) and a size-ambiguous subset (179 of 500 crop photos). ref-001: 88.6% as is, 83.0% normalised, 78.2% on the ambiguous subset normalised. ConvNeXt: 98.4% to 98.0%. So the fine-tuned model did lean on size, aspect or compression cues; ConvNeXt did not. Wheat is weakest (70). Worth telling in the article: the audit's warning was real, and the check changed the number by 5.6 points.

## 2026-11-11 — Crop-only fine-tune completed after scope correction

The user asked to stop excessive preparation and train immediately. Preserved the newer structured photo-check work and ran a separate crop-only experiment with one fixed prompt, balanced 500 training / 100 validation images from the original retained manifest, rank-8 language-side LoRA, 200 AdamW steps, batch 4, learning rate 2e-4, seed 3407, BF16, and 384 px image cap. All 600 selected images decoded and matched their SHA-256; selected training hashes/groups were disjoint from existing validation/test assignments. No test images were decoded or scored.

First attempt failed during image preparation on malformed EXIF serialization. Applying only orientation then discarding metadata fixed it; four focused unit tests passed. The successful `crop-pilot-002` completed training in 86.736 seconds (147.283 seconds whole run). Accuracy improved **53% → 86%**, macro F1 **0.524455 → 0.861105**, with identical validation images/prompt/decoding. Peak PyTorch allocated memory was 3.315 GiB. Full answers, confusion matrices and losses: `reports/crop-pilot-002.json`.

Adapter and processor saved under `outputs/crop-pilot-002/adapter`. A separate process loaded them through `predict_crop.py` and reproduced `Cotton` on the first validation image. No new model downloads/authentication were needed; the run used cached weights offline.

This is closed-set crop classification, not disease screening or field accuracy. It used the original retained manifest rather than the later near-duplicate-cleaned pool, so augmented-parent/near-copy and source-shortcut caveats remain. The 100-image score is not directly comparable with the earlier 700-photo structured photo-check experiments. Details and inference command: [crop pilot](../crop-pilot.md). Next training should use the already prepared cleaned pool and extend supported crop–condition targets.
