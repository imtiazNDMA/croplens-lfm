# Local Ubuntu WSL2 setup

## Current status

On 2026-11-11, the GPU was available and visible inside Ubuntu WSL2. A dedicated virtual environment was created at `$HOME/croplens-venv` with Python 3.12.3, pip 26.3, and uv 0.13.7.

**The training dependency installation is incomplete.** Two time-bounded installation attempts downloaded many wheels into uv's cache but did not finish. The environment still contains only pip and uv. No model or dataset has been downloaded, and no forward/backward or save/reload test has run.

The blocker is dependency download progress, not an observed CUDA/model failure. One bootstrap download also reported temporary DNS resolution failure. Do not treat the candidate versions below as a verified training environment or dependency lock.

## Hardware observations

| Check | Observation |
| --- | --- |
| GPU | NVIDIA RTX 6000 Ada Generation |
| Total VRAM | 49,140 MiB |
| Free VRAM at start of this session | 46,493 MiB |
| Free VRAM at final check | 46,596 MiB |
| Driver | 610.88 |
| OS environment | Ubuntu WSL2 |
| Python | 3.12.3 |

Memory availability is a snapshot, not a reservation. No unrelated workloads were stopped. No paid compute was provisioned.

## Continue the installation

Run these commands **in Ubuntu WSL2 Bash**, not Windows PowerShell. They use an isolated environment and do not replace system Python packages.

For a new machine, create the environment and bootstrap the installer:

```bash
python3 -m venv "$HOME/croplens-venv"
timeout --signal=TERM --kill-after=10s 300s \
  "$HOME/croplens-venv/bin/python" -m pip install --upgrade pip uv \
  --timeout 30 --retries 1
```

The environment and installer already exist on the project machine. Continue there with:

```bash
timeout --signal=TERM --kill-after=10s 900s \
  env UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=1 UV_CONCURRENT_DOWNLOADS=4 \
  "$HOME/croplens-venv/bin/uv" pip install \
  --python "$HOME/croplens-venv/bin/python" \
  "unsloth[huggingface]" bitsandbytes pillow \
  "torch==2.9.1" "torchvision==0.24.1" \
  --torch-backend=cu128
```

This is the candidate installation command, not yet a successfully reproduced training recipe. It reuses cached downloads. A normal successful exit must precede the checks below. `timeout` exit status 124 means the command exceeded its budget; a partial cache is not a completed installation. Do not automatically restart the command or schedule it overnight.

The dry run resolved 108 packages, including:

| Component | Candidate resolved version, not training-verified |
| --- | --- |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| unsloth | 2026.11.3 |
| unsloth-zoo | 2026.11.4 |
| transformers | 5.17.0 |
| trl | 1.15.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |

These are observations from resolution, not pins for all transitive packages. The official vision notebook uses older APIs and pins; verify the installed APIs rather than copying its trainer call unchanged. Save the full working environment only after the smoke run passes.

## Checks after installation succeeds

These are the **next checks**, not checks completed during bootstrap:

```bash
"$HOME/croplens-venv/bin/python" -m pip check
timeout --signal=TERM --kill-after=10s 120s \
  "$HOME/croplens-venv/bin/python" -c \
  'from unsloth import FastVisionModel; import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; x = torch.ones(16, device="cuda"); assert x.sum().item() == 16; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Import Unsloth before PyTorch/Transformers so its patches can apply. Passing this check establishes import and basic CUDA execution, not VLM training correctness.

Then implement and execute a bounded image-conditioned smoke test using the selected model and its processor. Before agricultural training it must establish:

1. Correct image–conversation pairing and image tokens.
2. Intended assistant-label masking.
3. Finite forward/backward loss and actual adapter updates.
4. Frozen vision parameters staying frozen.
5. Image-dependent behavior and a tiny-batch sanity check.
6. Adapter/processor save/reload and repeatable inference.

Use synthetic or explicitly shareable images for that engineering test. It must not be reported as a crop benchmark. Record the actual model revision, complete package environment, commands, loss, timing, memory, and outcomes in the [evidence journal](article/evidence-journal.md).

## Why not a lockfile yet?

A dependency resolver proves compatible version constraints, not working CUDA kernels, multimodal labels, or trained-adapter reload. A lockfile and executable training instructions should describe a reproduced working stack. Until then, the candidate command and its failures remain explicitly documented here.
