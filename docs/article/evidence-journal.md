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

## What to record in future sessions

For each substantive session, record the date and question; actual commands; code, model, data, and environment revisions; observed output; errors and fixes; decision and rationale; run/artifact links; timing and cost; and unresolved questions. Caption figures with dataset/split, units, run IDs, and the limitation they illustrate. Use only shareable images and sanitized logs.
