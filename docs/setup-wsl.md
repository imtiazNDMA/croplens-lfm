# Local Ubuntu WSL2 setup

## Current status

**Installation, basic CUDA checks, and a bounded synthetic-image LoRA smoke test passed on 2026-11-11.** The isolated environment is `$HOME/croplens-venv`, using Python 3.12.3, pip 26.3, and uv 0.13.7. The [machine-readable report](../reports/smoke-002.json) records the actual result.

This proves the tested Unsloth model/processor/collator, assistant-only labels, direct PyTorch AdamW updates, and adapter/processor save/reload path. It does **not** prove crop accuracy, agricultural data preparation, an SFTTrainer integration, QLoRA, quantized export, or deployment performance.

## Hardware and environment

| Check | Observation |
| --- | --- |
| GPU | NVIDIA RTX 6000 Ada Generation |
| Total VRAM | 49,140 MiB |
| Free VRAM at start of successful setup session | 46,511 MiB |
| Driver | 610.88 |
| OS | Ubuntu WSL2 |
| Python | 3.12.3 |
| torch / CUDA runtime | 2.9.1+cu128 / 12.8 |
| Unsloth / unsloth-zoo | 2026.11.3 / 2026.11.4 |
| Transformers / TRL | 5.17.0 / 1.15.0 |
| PEFT / Accelerate | 0.21.2 / 1.15.0 |

Memory availability is a snapshot, not a reservation. No unrelated workloads were stopped and no paid compute was provisioned. The synthetic test peaked at 3.224 GiB allocated and 3.299 GiB reserved **by PyTorch**; these are not total system GPU usage or estimates for real crop batches.

## Install in an isolated environment

Run the following in **Ubuntu WSL2 Bash**, not Windows PowerShell. Clone/open this repository first and run from its root.

```bash
python3 -m venv "$HOME/croplens-venv"
timeout --signal=TERM --kill-after=10s 300s \
  "$HOME/croplens-venv/bin/python" -m pip install --upgrade pip uv \
  --timeout 30 --retries 1
```

The exact observed package versions are in [requirements-wsl-cu128.txt](../requirements-wsl-cu128.txt). Install them with CUDA 12.8 wheel selection:

```bash
timeout --signal=TERM --kill-after=10s 900s \
  env UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=1 UV_CONCURRENT_DOWNLOADS=4 \
  "$HOME/croplens-venv/bin/uv" pip install \
  --python "$HOME/croplens-venv/bin/python" \
  -r requirements-wsl-cu128.txt --torch-backend=cu128
```

The version snapshot is not a hash-verified or cross-platform lockfile. The initial environment was installed with `"unsloth[huggingface]" bitsandbytes pillow "torch==2.9.1" "torchvision==0.24.1" --torch-backend=cu128`, then frozen after the successful smoke run. A fresh-machine reinstall from the snapshot has not yet been tested.

Slow downloads previously exceeded bounded attempts; a subsequent cached attempt completed in about 6 minutes 33 seconds of package preparation plus resolution/installation. Resume manually if needed; do not set up infinite retries or overnight jobs. `timeout` status 124 means the command exceeded its budget, not that installation succeeded.

## Verify dependencies and basic CUDA

```bash
"$HOME/croplens-venv/bin/python" -m pip check
timeout --signal=TERM --kill-after=10s 120s \
  "$HOME/croplens-venv/bin/python" -c \
  'from unsloth import FastVisionModel; import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); x = torch.ones(16, device=0); assert x.sum().item() == 16; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Both checks passed. Import Unsloth first so its patches can apply. One initial inline PowerShell command lost a quoted `cuda` string across the PowerShell/WSL boundary; using `device=0` avoided that shell-quoting issue. It was not a CUDA failure.

## Run the multimodal smoke test

Authenticate locally if needed using Hugging Face's interactive `hf auth login` in WSL. The model is public, but authentication may improve download limits. Never paste a token into this repository, examples, logs, or screenshots.

```bash
timeout --signal=TERM --kill-after=10s 900s \
  env HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30 \
  "$HOME/croplens-venv/bin/python" -u smoke_test.py \
  --output outputs/my-smoke-run
```

Use a new output directory for every run; the script refuses to overwrite one. `outputs/` and Unsloth's generated compile cache are ignored by Git. Run normally, not with Python `-O`, because the assertions are the engineering checks. The external timeout is mandatory: the SIGTERM handler records a timeout where Python can handle it, and the supervisor force-kills an unresponsive process after the grace period.

The script pins model revision `919fde3d022e3f90a4716006f993938ee8c2eb97`. It creates two synthetic 224×224 red/blue images and verifies:

1. Only language-side LoRA parameters are trainable; the vision encoder is excluded.
2. The collator supplies image tensors and supervises only the expected assistant answer.
3. Identical text tokens with different images produce different next-token logits.
4. Twenty direct AdamW updates yield finite losses/gradients, changed adapters, and lower tiny-batch loss.
5. Saving/reloading the adapter and processor preserves greedy generation for the tested prompt.

Actual run `smoke-002`: 4,571,136 trainable parameters, loss 3.749977 → 0.000315, training loop 23.451 seconds, full run 487.244 seconds, and `red` generated before/after reload. The full duration includes downloading/loading and reload, not just training. This is a pipeline check on two artificial examples, not a generalization benchmark or a replicated multi-seed result.

The first unauthenticated download attempt stalled and was interrupted before training. The successful attempt used transient authentication and disabled Xet. Those changes happened together, so we cannot attribute the improvement to the token alone. No token was written to the project or report.

## Known limits and next gate

- The runtime used a correct but slower reference convolution because `causal_conv1d` was absent. Do not add an optimized kernel until throughput profiling justifies and verifies it.
- The test uses a direct AdamW loop, not TRL's SFTTrainer. The agricultural training entry point still needs its own integration checks.
- Match crop-image preprocessing, target schema, batch sizes, grouped splits, and evaluation behavior before real training.
- Save/reload equivalence was checked on one greedy prompt; broader regression tests remain future work.
- Freeze the crop taxonomy and data specification, audit sources and duplicates, and establish baselines before autoresearch.
