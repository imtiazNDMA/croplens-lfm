# Bounded CropLens autoresearch instructions

Adapted from [Karpathy's autoresearch program](https://github.com/karpathy/autoresearch/blob/master/program.md).
The AI coding assistant is the research agent. `condition_search.py` is its execution/tracking supervisor, not a second LLM or an upstream GPT trainer.

## Objective and immutable evaluation

Improve **joint crop–condition validation macro F1** over `reports/condition-pilot-001.json` (0.8784693925). Invalid outputs remain failures. Keep the same 14 classes, 700 cleaned training images, 280 validation images, seed, prompt, parsing, preprocessing, 400 optimizer steps and batch 4. Every candidate starts from the pinned base, never from the previous winner's adapter. The old untuned predictions may be reused only after data/config/image checks.

Read the condition pilot, trainer, existing results, and this file before proposing changes. The original loop edits GPT code, measures bits/byte, commits/resets, and runs indefinitely. This project instead uses fixed-work LoRA trials, integrity checks, local MLflow, and explicit daytime deadlines. Do not run the upstream GPT script or claim this is unchanged upstream execution.

## Allowed work this session

Change one independent knob from the reference: `learning_rate` or `rank`. LoRA alpha follows rank so alpha/rank stays 1. Do not change datasets, labels, validation/test access, trainer code, evaluator, deadlines, or dependencies during the search. The supervisor checks protected fingerprints before and after each trial. This detects accidental edits; it is not an OS security sandbox.

Initial hypotheses: lower learning rate 1e-4 may generalize better; higher rate 4e-4 may learn weak classes faster; rank 16 may give more capacity. Record rationale before execution. These are agent-authored proposals, not an external model-generating service or a generic optimizer.

## Budgets and decisions

- One GPU, one foreground trial at a time.
- At most 3 candidate trials, 600 seconds per complete trial, 300 seconds training safety cap.
- Entire session capped at 1,800 seconds, with no more than a 10-second termination grace. An external timeout is also required.
- Stop after two consecutive failures. No automatic restart or overnight runs.
- Select only complete trials, recomputing metrics from raw answers and the reference gold labels.
- Keep only if joint macro F1 improves the best-so-far by at least 0.005, invalid count is zero, and crop accuracy is no more than 0.01 below the reference.
- Otherwise discard for promotion but retain its report, adapter and MLflow history. Do not delete user work or reset Git. Do not commit or push unless requested.

Inspect the outcome, explain weak classes and failed hypotheses, and suggest the next bounded session based on evidence. A winning validation run is not a final-test, multi-seed, or field result. Do not touch locked tests during selection.
