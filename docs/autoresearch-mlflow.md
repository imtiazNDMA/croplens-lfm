# Bounded autoresearch with MLflow: condition-search-001

Completed **2026-11-13**. Three GPU trials trained and evaluated successfully. **MLflow is running locally at http://localhost:5000**; select the `croplens-conditions` experiment.

## What actually ran

The AI coding assistant acted as the research agent, following [program.md](../program.md), an adaptation of [Karpathy's autoresearch instructions](https://github.com/karpathy/autoresearch/blob/master/program.md). It proposed hypotheses, executed them, inspected results, and kept/discarded candidates. [condition_search.py](../condition_search.py) is the deterministic execution/tracking supervisor; it is not another LLM or a stock upstream agent executable.

The original GPT trainer, validation bits/byte, indefinite overnight loop, automatic commits and Git resets were not used. This project uses the pinned Liquid VLM, Unsloth language-side LoRA, fixed training work, task-specific validation, local MLflow, and explicit stop limits. No new commits/resets were made during the experiments.

## Results and selection

| Run | One change from reference | Joint macro F1 | Exact crop–condition accuracy | Crop accuracy | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Historical reference | LR 2e-4, rank 8 | 0.878469 | 87.86% | 99.29% | Reference |
| lr-low | LR 1e-4 | 0.893298 | 89.29% | 98.93% | Keep initially |
| **lr-high** | **LR 4e-4** | **0.899777** | **90.00%** | **98.93%** | **Selected** |
| rank-16 | Rank/alpha 16, LR 2e-4 | 0.900639 | 90.00% | 98.57% | Discard for promotion; retained |

Keep required at least **0.005** improvement in joint macro F1 over the best-so-far, no invalid answers, and crop accuracy no more than 0.01 below the reference. The larger adapter had the highest raw F1, but its 0.000862 gain over lr-high was below the threshold. It doubles trainable LoRA parameters, so the simpler rank-8 model stays selected. "Discard" means not promoted, not deleted.

The selected model gets **252/280** exact crop–condition answers right versus **246/280** for the reference: six additional correct answers, +2.14 percentage points. Crop accuracy fell slightly, within the predeclared guardrail. This single-seed validation gain is promising, not proof of statistical significance or field improvement.

## Fairness and boundaries

All candidates started from the same pinned base—not a previous winner's adapter—and used the same seed, prompt, loader, 14 classes, 700 cleaned training images, 280 validation images, batch 4, and 400 optimizer steps. Rank and alpha changed together to keep alpha/rank equal to 1. Reference untuned answers were reused after manifest/pool hashes, task/config and validation-image checks instead of re-running the identical baseline.

Protected source/data fingerprints were checked before and after each trial. The supervisor recomputed scores from raw answers using the reference gold labels before selection. No final-test images were decoded/scored. These checks detect accidental changes; they are not an OS permission sandbox.

## Runtime

- Three candidates, sequentially on the RTX 6000 Ada.
- Training times: **118.735 s**, **112.250 s**, **120.315 s** (total 351.300 s).
- Peak PyTorch allocated memory: 3.387 GiB for rank 8; 3.416 GiB for rank 16.
- Per trial: 300-second training safety cap and 600-second total supervisor cap.
- Entire session: 1,800 seconds maximum, with no automatic restart and a two-consecutive-failure stop.
- Actual session span: **1,324.713 seconds**, including the UI interruption/resumption; all work stayed inside the original 30-minute window.

The supervisor was interrupted while the UI was being requested, but the higher-LR worker finished and logged its artifacts. Added resume support adopted its completed report/run, recomputed metrics, and then executed rank-16. It did not repeat training, overwrite reports, reset the deadline, or create duplicate trial runs. Incomplete/active workers are refused for adoption.

## MLflow records

- Tracking database: `/home/user/croplens-mlflow/mlflow.db`.
- Artifact root: `/home/user/croplens-mlflow/artifacts`.
- Experiment: `croplens-conditions` (ID 1 on this machine).
- Parent session run: `b8f7595623074c83ae16c0d64a25954d`.
- Selected run: `bd122d1d690f4309ac4db6f01f8c05aa`.
- Retained runs: one parent, one explicitly tagged historical reference, three candidates; all finished.

MLflow stores hyperparameters, input/model hashes, per-class recalls, accuracy/F1, losses by step, timing/memory, raw prediction/report artifacts, adapter/processor files, full package freeze, source snapshots, and Git revision/dirty-state tags. Source images and credentials are not logged. The historical reference is an imported prior result, not a freshly rerun experiment.

For this completed session, loss histories were uploaded at completion. The trainer was subsequently updated to publish parameters at startup and loss at every optimizer step for future live curves; final artifact logging avoids duplicate loss points. The completed workers' source snapshots are retained, and the resumed supervisor snapshot is saved separately.

MLflow 3.8.1 required SQLAlchemy 2.0.46: the resolver initially selected 2.1.8, which removed a pool class imported by MLflow. A real SQLite persistence test caught this; [requirements-mlflow.txt](../requirements-mlflow.txt) pins the compatible overlay. MLflow also resolves older compatible pandas/protobuf and related dependencies; exact post-install versions are in each MLflow run's `environment.txt`. The older base environment snapshot remains historical, not a simultaneous constraint with the overlay.

## View or restart the UI

The UI is already running on loopback port 5000 and its health endpoint returned HTTP 200. To restart it later, run in Ubuntu WSL2 (after stopping the old server):

```bash
"$HOME/croplens-venv/bin/mlflow" server \
  --backend-store-uri "sqlite:///$HOME/croplens-mlflow/mlflow.db" \
  --host 127.0.0.1 --port 5000 --workers 1 --no-serve-artifacts
```

Then open **http://localhost:5000** in Windows. The UI reads the same local database; no remote tracking account or paid service is used. The UI remains running by request, but no GPU search/training jobs remain active.

## Use the selected adapter

```bash
timeout --signal=TERM --kill-after=10s 120s \
  env HF_HUB_OFFLINE=1 \
  "$HOME/croplens-venv/bin/python" predict_condition.py \
  --adapter outputs/condition-search-001/lr-high/training/adapter \
  --image "/path/to/a/supported-crop-photo.jpg"
```

A separate-process reload returned `{"crop":"Cotton","condition":"aphid"}` on the checked validation photo. This checks one persisted prediction, not independent performance.

Final verification: **60 unit tests passed**, including a real local SQLite metric/artifact round trip, process timeout, completed-worker adoption without rewriting, metric integrity, and promotion guardrails. `pip check` found no broken requirements; MLflow's health endpoint remained HTTP 200.

The [search summary](../reports/condition-search-001.json) records hypotheses, decisions, metrics, run IDs, protected hashes, and the selected adapter. Raw candidate reports/adapters remain under `outputs/condition-search-001/` and in MLflow artifacts. The database, weights and transient logs are not tracked in Git.

## Deployment readiness

The saved model loads and produces the supported JSON schema, making it suitable for a **controlled experimental demo**. This repository has inference CLIs, not yet a deployed inference service. It is not established as a field-ready crop-health product.

Before broader deployment, evaluate the frozen selected adapter once on the held-out test data, assess field/unsupported inputs, and verify serving behavior and resource use on the actual target hardware. The current model always selects from 14 classes: it has no validated unknown/nonplant rejection or calibrated confidence. Source-data/adapter redistribution rights are also unresolved, as recorded in the living plan; keep weights local until that is settled.

Seed confirmation can test whether the small validation gain is stable. No field, calibration, rejection, or treatment claims follow from the present results. A demo should present outputs as experimental screening and state its supported crops/conditions.
