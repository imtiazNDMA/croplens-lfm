# CropLens: Vision-Language Model Fine-Tuning Plan

> Living project plan, experiment roadmap, and publication brief.
> Created: 2026-11-11. Last updated: 2026-11-11.
> Status: isolated WSL environment created; training-dependency installation incomplete after bounded download attempts. No CropLens training or evaluation has been run.

## 1. Project summary

Build a standalone, portfolio-ready project that fine-tunes a compact Liquid AI vision-language model (VLM) with Unsloth for crop identification and photo usability. Investigate crop health screening as a separately evaluated extension. Use a bounded, human-supervised adaptation of Karpathy's autoresearch workflow, with no overnight runs. Add grounded treatment recommendations later through retrieval-augmented generation (RAG), after a market survey and source review.

**Working title:** CropLens: Efficient Crop Vision Fine-Tuning with Liquid LFM, Unsloth, and Budgeted Autoresearch.

**Research question:** Can a compact VLM identify supported crops, reject unsuitable images, and screen a limited set of crop conditions under a reproducible, tightly controlled compute budget?

**Initial checkpoint:** `LiquidAI/LFM2.5-VL-1.6B`.

**Initial training method:** image-conditioned supervised fine-tuning (SFT), using native 16-bit base weights and language-side LoRA adapters, with the vision encoder frozen.

**Deliverables:** a reproducible GitHub project, audited data manifests, baselines, bounded experiments, evaluation reports, a working demo, model/dataset documentation, and a complete beginner-friendly Substack article grounded in our actual work.

## 2. How to maintain this document

- Update this file whenever an accepted decision, scope, dependency, dataset, experiment policy, result, or next action changes.
- Keep proposed work distinct from completed work. Never turn a target into a measured result.
- Record material decisions with a date and rationale in the decision log below.
- Link detailed run artifacts rather than copying every log into this document.
- Update the milestone checklist and open questions at the end of each substantive work session.
- Preserve unsuccessful hypotheses and limitations for the final report and article.
- This repository's root `fine-tuning.md` is the canonical plan. The agriculture portal workspace keeps a pointer instead of a second editable copy.

## 3. Verified feasibility and remaining uncertainty

### 3.1 Why this model

Unsloth publishes a vision-specific LFM2.5-VL-1.6B recipe using `FastVisionModel`, `UnslothVisionDataCollator`, and SFTTrainer. Its official notebook includes saved training outputs. This is a credible starting point, not a locally reproduced training result. [S1-S3]

Start with the documented frozen-vision, non-4-bit path. Treat QLoRA, vision-encoder training, adapter merging, and quantized multimodal deployment as separately verified extensions.

Newer Liquid vision models exist, but the strongest exact-checkpoint Unsloth training evidence inspected for this plan is for LFM2.5-VL-1.6B. Older LFM2-VL checkpoints are deprecated by Liquid. Text-only LFM support does not establish vision training support. [S4-S5]

The published recipe and saved runtime output have dependency-version differences. We must create a working dependency lock from a reproduced compatibility run, not simply copy installation commands and assume they are current.

### 3.2 What autoresearch means here

Karpathy's original project trains a small GPT model and selects experiments using validation bits per byte. It does not directly fine-tune this VLM or evaluate agricultural images. [S6]

We will adapt its workflow:

**Hypothesis → change an allowlisted training setting → train → evaluate → keep/discard → record findings.**

Describe the project publicly as an adaptation of Karpathy's autoresearch workflow, with attribution. Do not imply that its original training code supports Liquid VLM SFT unchanged.

### 3.3 Existing project foundations

The Climate Smart Agriculture portal already has:

- A photo check for crop identity and image usability.
- Separate crop health screening classifiers.
- A historical 37-class, five-crop experiment using 11,453 unique images.
- A historical classifier report recording test macro F1 of 0.898.
- Previous autoresearch experience and a local-first AI architecture.

Those historical numbers are context, not a fresh LFM benchmark or evidence of Pakistan field performance. The existing report also records clock-dependent reproducibility problems, motivating fixed-work training comparisons with time limits only as safety caps.

In the portal's domain language, a **photo check never names diseases**. Giving LFM responsibility for **crop health screening** changes that boundary and requires an explicit architecture decision before portal integration.

Local references at plan creation:

- `CONTEXT.md`
- `backend/csa_portal/photo_check.py`
- `docs/evaluations/crop-disease-69-best-model.md`
- `docs/adr/0001-local-language-model-by-default.md`
- `docs/research/lfm-vision-feasibility.md`

These local references are provenance notes; the standalone project must contain enough independent documentation to reproduce its own results without private portal access.

## 4. Scope and terminology

| Capability | Initial scope |
| --- | --- |
| Identify the main crop in one image | Yes, within an explicit supported taxonomy |
| Recognize unsupported plants and nonplant images | Yes |
| Flag unusable or ambiguous photos | Yes |
| Crop health screening | Conditional extension after the core pipeline works |
| Arbitrary plant/species identification | Not part of the initial release |
| Bounding boxes or lesion segmentation | Not part of the initial release |
| Treatment or pesticide recommendations | Later RAG phase |
| Urdu explanations | Separate evaluation milestone |

Here, crop identification is image-level classification of the main subject. Object detection with bounding boxes needs different annotations and evaluation. The article must explain this distinction even if its introductory wording uses the familiar phrase "crop detection."

Crop health screening means identifying visually supported healthy, disease, or pest classes within the supported taxonomy. It is not confirmation of a biological cause from a photo alone. Unknown, unsupported, and insufficient-evidence outcomes must be available.

Default proposal: audit cotton, maize, rice, sugarcane, and wheat for photo checks; start condition screening with a smaller, well-supported subset rather than immediately promising all 37 historical classes.

## 5. Proposed system architecture

```text
Image upload
    │
    ▼
Upload validation and image normalization
    │
    ▼
Fine-tuned Liquid photo-check model
    ├── Plant / nonplant assessment
    ├── Supported crop / other / unclear
    └── Photo usability
    │
    ▼
Application validation and routing
    ├── Request a better photo
    ├── Ask the user to confirm the crop
    └── Run crop health screening
             ├── Existing specialist classifier baseline
             └── Experimental LFM screening path
                      │
                      ▼
              Structured screening result
                      │
                      ▼
              Later: treatment RAG
```

Begin with one base checkpoint and one photo-check adapter. Train a separate screening adapter if the extension is justified. This keeps objectives and failures measurable independently.

The final system may use LFM for photo checks and a specialist classifier for screening. Use LFM for both only if measured quality, efficiency, and maintainability justify it. A hybrid system winning is a valid project result.

Use a compact, validated output contract with explicit unknown/unclear states. Keep photo-check outputs separate from condition-screening outputs. Do not put free-form treatment advice into either training target.

## 6. Standalone GitHub project and portfolio strategy

**Repository:** [imtiazNDMA/croplens-lfm](https://github.com/imtiazNDMA/croplens-lfm).

Created and verified as public on 2026-11-11. A separate local checkout contains the documentation bootstrap and an honest setup report. Keep it independently reproducible rather than copying the complete agriculture portal or its Git history.

Local checkout: `G:\dr aamir\croplens-lfm`, alongside the original portal checkout. The complete plan now lives here. The user requested proceeding with the next setup step after the commit/push and local-compute questions; only reviewed project documentation is in the initial publication scope.

The bootstrap includes a README, this canonical plan, standalone feasibility research, article outline, evidence journal, and focused ignore rules. It does not copy application files, private datasets, credentials, or the portal's Git history.

### Proposed repository structure

Create each part when the corresponding phase needs it; this is a layout proposal, not a request to generate empty scaffolding now.

```text
croplens-lfm/
├── README.md                  # Project story, status, results, reproduction entry point
├── fine-tuning.md             # Canonical living plan after migration
├── pyproject.toml             # Reproduced training dependencies
├── uv.lock                    # Lock after compatibility verification
├── program.md                 # Bounded research-agent instructions
├── prepare.py                 # Manifest preparation and split validation
├── train.py                   # Unsloth SFT entry point
├── evaluate.py                # Frozen task evaluation entry point
├── run_experiments.py         # Sequential bounded runner, when needed
├── configs/                   # Versioned experiment settings
├── tests/                     # Data, metrics, contract, and budget checks
├── data/                      # Manifests and acquisition instructions, not private images
├── reports/                   # Experiment ledger, aggregate metrics, plots
├── docs/
│   ├── research/              # Source-backed feasibility notes
│   ├── model-card.md
│   ├── dataset-card.md
│   └── article/
│       ├── substack-draft.md
│       └── evidence/          # Publishable figures and supporting references
└── demo/                      # Minimal independent demo after model evaluation
```

### Recruiter-facing deliverables

- Concise README with the task, constraints, architecture, and actual status.
- Before/after baseline table and links to machine-readable metrics.
- Reproduction commands with hardware and dependency versions.
- Complete experiment history, including failed/discarded runs.
- Confusion matrices, field-versus-in-corpus results, and failure analysis.
- Model and dataset cards, plus attribution and license information.
- A short demo video and a curated image gallery.
- Link to the completed Substack article and, when available, a model/adapter release.

Select a license for original project code before publishing it. The selected code license does not replace the base model's license or grant rights to source datasets. Liquid's model license is not Apache-2.0; record the exact checkpoint terms and distribution notices. [S7]

## 7. Phase 0 — Freeze the experiment specification

**Estimated active effort:** half a day.

Decide and record:

1. Supported crops and conditions, including stable label identifiers.
2. Intended image types: close-up leaf, whole plant, pest image, or a defined subset.
3. Available hardware, deployment target, and total spending ceiling.
4. Primary evaluation metric for each task.
5. Rejection rules and acceptable false-acceptance tradeoffs.
6. Success criteria and release boundaries.

**Deliverable:** a fixed initial taxonomy and experiment specification.

**Exit gate:** unresolved labels and unsupported task claims are removed or explicitly deferred before data preparation.

## 8. Phase 1 — Dataset audit and preparation

**Estimated active effort:** 2–4 days, excluding new field collection and expert annotation.

### 8.1 Reuse and audit existing data

Audit the existing corpus before collecting replacements:

- Source provenance, acquisition method, consent where applicable, and redistribution rights.
- Crop and condition label consistency, including healthy examples.
- Exact and near duplicates within and across sources.
- Derived crops/views from the same original image or physical leaf.
- Existing training/evaluation overlap.
- Backgrounds, watermarks, annotations, and filenames that expose the answer.
- Class balance, ambiguous labels, and expert review requirements.
- Representation of the images users will actually upload.

Record exclusions with reasons. Synthetic paraphrases of an answer do not create new independent images or justify larger sample-count claims.

### 8.2 Supplement only where needed

PlantVillage supplies predominantly controlled-condition imagery; PlantDoc supplies non-laboratory variation. The inspected taxonomies cover maize/corn but do not cover all five target crops. Neither establishes Pakistan field performance. [S8-S10]

Do not adopt a mirror's license without tracing the original release. PlantVillage's author dataset declares CC-BY-SA-3.0; PlantDoc's repository declares CC-BY-4.0, with scraped-image provenance still requiring attention. Document the exact subset and revision used.

### 8.3 Include rejection and ambiguity examples

Include other plants, nonplant objects, severe blur/exposure problems, ambiguous grass-leaf close-ups, multiple unclear subjects, and insufficient visual evidence.

Do not generate unknown examples by randomly relabelling otherwise identifiable images. A crop-pest photo and a nonplant object also need clearly defined labels consistent with the chosen task.

### 8.4 Build grouped, immutable splits

Keep all related views within one split. Use physical leaf, original photo, collection session, or field grouping where metadata allows; document uncertainty where it does not.

Create:

1. Training split.
2. Validation split used for model selection and threshold setting.
3. Locked final test used only after selection.
4. Independent field holdout, if available.

Preserve valid existing holdouts. If a development team has repeatedly inspected a test set, do not describe it as untouched; obtain an additional final holdout for stronger claims.

Dataset manifest records should include image ID/hash, group ID, source/revision, license reference, crop, condition where known, usability labels where annotated, and split. Keep machine-specific paths out of public manifests.

**Deliverables:** manifest, taxonomy, source/license inventory, split hashes, dataset card, and label-quality report.

**Exit gate:** no known cross-split duplicate groups, documented target coverage, and enough examples to report class-level support honestly.

## 9. Phase 2 — Prove the training pipeline works

**Estimated active effort:** 1 day.

Use an isolated GPU training environment, initially Linux or WSL2. The portal currently requires Python 3.13+; do not assume its environment is compatible with the required CUDA/Unsloth stack.

Begin with the official exact-checkpoint recipe, then pin the versions that actually work on the selected hardware.

Required checks:

- Image-conditioned inference works with the correct processor and chat template.
- Forward/backward passes yield finite loss.
- Intended LoRA parameters receive updates and frozen parameters stay frozen.
- Assistant labels are masked as intended; padding and prompt/image content do not accidentally become the supervised target.
- Preprocessing and truncation preserve image tokens and useful visual detail.
- Distinct images can affect predictions; image/text targets are correctly paired.
- A tiny clean batch can be deliberately overfit as a pipeline sanity check.
- Adapters and processor save/reload correctly.
- Reloaded inference matches the saved run under fixed decoding settings.

Record GPU, driver, CUDA, PyTorch, Unsloth, Transformers, TRL, PEFT, model revision, and processor revision. Check data/metric and output-contract behavior with small automated tests; test budget enforcement before letting an agent run experiments.

**Deliverable:** compatibility report and working environment lock.

**Exit gate:** no automated search until these checks pass. Tiny-batch overfitting is an implementation check, not evidence of generalization.

## 10. Phase 3 — Establish fair baselines

**Estimated active effort:** 1 day.

| Baseline | Purpose |
| --- | --- |
| Untuned LFM with a fixed prompt | Measure the benefit of fine-tuning |
| Existing photo-check implementation | Measure application-level improvement |
| Existing specialist classifier on compatible labels | Measure screening quality and efficiency |
| Fine-tuned LFM candidate | Experimental comparison |

Use identical eligible images, compatible label mappings, equivalent output constraints, and fixed decoding settings. Document what each baseline saw during training; do not compare a model trained on test images as if it were a clean baseline.

For screening, report two settings separately:

- **Crop provided:** isolates condition recognition.
- **Crop inferred:** evaluates routing and screening end to end.

Do not hide crop-routing errors by always supplying the correct crop. If a private portal baseline cannot be released, keep the public untuned-LFM comparison independently reproducible and explain the private baseline's limitations.

**Deliverable:** baseline report before hyperparameter search.

## 11. Phase 4 — Bounded autoresearch, with no overnight runs

**Estimated active effort:** 1–2 working days, divided into explicitly started sessions.

### 11.1 Session policy

| Control | Initial proposal |
| --- | --- |
| Start | Explicit human command |
| Concurrent experiments | One |
| Trials per session | Maximum 6 |
| Training budget | Fixed optimizer-step/example budget chosen after profiling |
| Training safety cap | 10 minutes per trial |
| Complete-trial limit | 20 minutes, including loading, evaluation, and saving |
| Whole-session limit | 2 hours |
| Consecutive failed trials | Stop after 2 |
| Restart | Manual; no automatic session relaunch |
| Overnight behavior | No scheduled or open-ended unattended runs |

An external process supervisor must enforce deadlines; instructions in `program.md` alone are insufficient. Apply provider spending and idle-shutdown controls when available. Session deadlines take precedence over trial counts, and cleanup/checkpoint saving must not extend into an unbounded run.

**Comparison rule:** fixed training work for fair comparisons; wall-clock limits for overrun protection. Mark time-capped incomplete runs explicitly instead of ranking them as equivalent completed runs.

### 11.2 Initial search sequence

1. Reference LoRA configuration.
2. Lower learning rate.
3. Higher learning rate.
4. Lower LoRA rank.
5. Higher LoRA rank.
6. Confirmation of the most promising setting.

Choose actual learning rates, ranks, step counts, and batch sizes after Phase 2 profiling. Change one setting at a time relative to a recorded reference. Hold effective batch size, data order policy, preprocessing, and task contract fixed initially.

Study image-token/resolution tradeoffs in a separate bounded ablation. Record their throughput impact rather than silently giving a faster configuration more training examples.

Each comparable trial starts from the same pinned base checkpoint, not the preceding winner's adapter. Keep model selection on validation only. Short-budget results are search signals, not evidence that training has converged.

### 11.3 Agent edit boundary

The agent may change allowlisted training settings. It may not modify dataset splits, labels, taxonomy, evaluation code, selection criteria, test access, or budget enforcement. Protect these boundaries through the runner and file permissions/process separation where practical, not only prose.

### 11.4 Experiment record

Every trial records:

- Run ID, timestamp, hypothesis, and reference run.
- Code revision, dataset/split hashes, model/processor revisions, and environment.
- Seed, trainable parameter count, LoRA targets/rank/alpha, optimizer and schedule.
- Precision, batch size, accumulation, image processing, sequence length, and intended training work.
- Completed steps/examples, loss history, validation metrics, and output validity.
- Load/train/evaluate/save times, peak VRAM, and allocated cost where measurable.
- Completion/failure/timeout status, error reason, artifact paths, and keep/discard rationale.

Use one experiment ledger and the simplest suitable existing logging approach. A new hosted tracking platform is not required for this mini-project.

**Deliverable:** auditable trial ledger, including failed and discarded attempts.

## 12. Phase 5 — Confirm, evaluate, and analyze failures

**Estimated active effort:** 1–2 days.

Rerun the selected configuration with three seeds and the same fixed training budget. Complete confirmation runs within explicitly started daytime sessions. Lock the configuration and rejection thresholds before final testing.

### 12.1 Evaluation metrics

| Task | Measurements |
| --- | --- |
| Crop identity | Primary: validation macro F1 over a fixed label set; accuracy and per-crop recall alongside it |
| Photo usability | Per-flag precision/recall and rejection confusion matrix |
| Unsupported inputs | False acceptance rate and supported-input rejection rate |
| Condition screening | Primary: validation macro F1; per-condition precision, recall, F1, and support |
| End-to-end screening | Correct crop–condition outcome, including routing failures |
| Structured output | Valid-output rate; invalid answers count as task failures |
| Deployment | End-to-end latency, warm/cold distinction, peak memory, artifact size |
| Research efficiency | Training work, allocated GPU hours, and quality-versus-cost comparison |

For the teaching article, explain:

- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2 × precision × recall / (precision + recall).
- Macro F1 averages class F1 values equally; accuracy alone can hide minority-class failures.
- Coverage = accepted examples / all eligible examples.
- False acceptance rate = unsupported examples incorrectly accepted / all unsupported examples.

Define zero-division handling, fixed label sets, and how invalid/unknown outputs affect metrics in the evaluator. Use a standard metric library rather than inventing a second implementation.

Report performance on the full eligible set and on accepted cases, with coverage beside accepted-case metrics. A model must not appear better simply because it rejects difficult images. Report class counts, seed variation, and uncertainty where sample size permits; avoid precise reliability claims from tiny classes or field samples.

### 12.2 Confidence and calibration

Do not present a generated confidence number as a calibrated probability. If probabilities or candidate scores are available, validate their meaning and fit rejection thresholds using validation data. Otherwise expose categorical uncertainty rather than fabricated percentages.

### 12.3 Failure analysis

Review at least:

- Maize versus sugarcane confusion.
- Healthy leaves mistaken for disease.
- Disease versus nutrient/environmental stress.
- Unsupported plants incorrectly accepted.
- Ambiguous grass leaves and poor-quality images.
- Performance changes between controlled and field backgrounds.
- Invalid structured outputs and confident-looking wrong answers.

Annotate representative errors with IDs and expert-reviewed interpretation where available. Keep all figures traceable to the evaluation artifacts.

**Deliverables:** final report, baseline comparison, seed summary, confusion matrices, failure gallery, and model card.

## 13. Phase 6 — Portfolio demo and deployment

**Estimated active effort:** 1–2 days.

Build a minimal independent demo, reusing established FastAPI/frontend patterns where appropriate. Portal integration is optional follow-on work, not a prerequisite for public reproduction.

The demo should show:

- Image upload and validation.
- Crop identity and photo-quality outcome.
- Screening result only for supported conditions.
- Unknown, unsupported, and insufficient-evidence states.
- Model version and measured inference latency.
- Curated examples and a link to evaluation limitations.

For eventual portal integration, preserve the local-first architecture and existing photo-check contract. Keep upload handling, image metadata removal, and logging consistent with existing privacy behavior.

Start deployment with the verified base model, adapter, and processor. Only claim merged/quantized support after an image-conditioned save/reload and regression evaluation. Benchmark actual deployment hardware rather than transferring training-GPU timings to a laptop claim.

Publish only reviewed, shareable code and artifacts. Keep private photos, credentials, internal assets, and nonredistributable dataset images out of the public project. Supply acquisition scripts and manifests where image redistribution is not permitted.

## 14. Acceptance criteria and decision gates

Set numeric targets after baselines and sample sizes are known, but before search starts. Record them here when agreed.

| Criterion | Required evidence |
| --- | --- |
| Reproducible | Pinned environment, model/data revisions, commands, and save/reload checks |
| Useful | Improvement on the selected task over untuned LFM without unacceptable rejection regressions |
| Honest | Field results separated from in-corpus results; absent field data stated explicitly |
| Efficient | Quality reported with latency, memory, training work, and cost |
| Robust | Invalid answers and unsuitable images have tested outcomes |
| Bounded | Trial/session limits and failure stop conditions are enforced and tested |
| Publishable | Source attribution, artifact rights, and complete reproduction documentation |

If LFM screening does not beat the specialist classifier, retain the classifier and document why. If field data is unavailable, release a research demo with that limitation rather than claiming field readiness.

## 15. Later phase — Market-informed treatment RAG

Do not train pesticide/product knowledge into the initial visual adapter. Treatment guidance is a later, separately evaluated capability.

### 15.1 Market survey and source review

Investigate intended users, current workflows, common crops/problems, local language and connectivity needs, registered products, approved crop–pest uses, label availability, update frequency, formulations, availability, and market relevance.

Prioritize authoritative current Pakistan registration and product-label sources. Retailer availability can inform the market survey but does not establish approved use. Record document dates, versions, jurisdiction, and provenance.

### 15.2 Proposed RAG flow

```text
Screening result + confirmed crop + location/context
                         │
                         ▼
         Retrieve authoritative current sources
                         │
                         ▼
         Filter for applicable crop–pest–product use
                         │
                         ▼
       Generate source-cited management guidance
```

Distinguish a visually suspected condition from a confirmed cause, include nonchemical management where appropriate, and derive product-specific application instructions from applicable labels rather than model memory. Unsupported or conflicting evidence should produce an explicit insufficient-evidence outcome.

Evaluate retrieval relevance, citation support, crop/pest applicability, source freshness, and unsupported recommendation rates separately from image-model accuracy. A market survey is planned, not completed.

## 16. Compute, costs, and schedule

### Hardware proposal

Hardware discovery on 2026-11-11 found an **NVIDIA RTX 6000 Ada Generation**, **49,140 MiB total VRAM**, and NVIDIA driver **610.88**. The GPU is also visible from the running **Ubuntu WSL2** distribution, where `python3 --version` reports **Python 3.12.3**. The `uv` executable is not installed on that distribution's current PATH.

At the WSL check, only **9,560 MiB VRAM was free**. This is a point-in-time observation, not a reserved allocation. Confirm a suitable free-GPU window before the smoke run; do not terminate other workloads to make room without the owner's instruction.

The user confirmed GPU availability, and a fresh check found 46,493 MiB free. Local Ubuntu WSL2 is now the selected first environment. A dedicated Python 3.12.3 virtual environment at `$HOME/croplens-venv` contains pip 26.3 and uv 0.13.7.

The candidate Unsloth/PyTorch installation resolved successfully but did not complete within two bounded download attempts. Many dependency wheels are cached; the environment still has only pip and uv. The [WSL setup report](docs/setup-wsl.md) records the exact candidate command and next checks. No model download or training has occurred. No GPU time, provider rate, or paid cloud instance has been booked. Actual training fit and throughput remain unverified.

### Cost accounting

**GPU cost = hourly rental rate × total allocated hours.**

Include downloads, loading, compilation, evaluation, failed trials, confirmation runs, and idle allocation. Record agent/API charges and storage separately. No dollar estimate is a measured project cost until provider, rate, and actual allocation are known.

### Initial active-work estimate

| Phase | Active effort |
| --- | --- |
| Specification | Half a day |
| Dataset audit/preparation | 2–4 days |
| Compatibility pipeline | 1 day |
| Baselines | 1 day |
| Bounded search | 1–2 days |
| Confirmation/evaluation | 1–2 days |
| Demo | 1–2 days |

Plan for roughly 8–13 active working days, with a possible half-day overrun at the upper end. New field collection, expert annotation, and the complete article can extend the calendar. Treat the article as a parallel evidence-capture effort and a final writing/reproduction pass, not something automatically finished when training ends.

## 17. Substack publication plan

### 17.1 Audience and promise

Write one complete, long-form teaching article for beginner AI engineers and software engineers transitioning into AI engineering. It should explain the concepts, decisions, implementation, evaluations, mistakes, costs, and deployment with enough detail for readers to reproduce the public experiment.

**Working article title:** "Fine-Tuning a Vision-Language Model for Crop Identification and Disease Screening: A Complete Engineering Walkthrough."

**Working subtitle:** "From dataset audits and Unsloth LoRA to bounded autoresearch, honest evaluations, and a deployable demo."

The final title and claims must reflect what we actually complete. If disease screening remains experimental or loses to a specialist classifier, explain that result instead of implying success. The article is planned, not written as a completed retrospective or published now.

### 17.2 Reader prerequisites and teaching approach

Assume basic Python, APIs, Git, and software development experience. Explain tensors, GPU memory, train/validation/test splits, model weights, processors, SFT, LoRA, and evaluation before relying on them. Separate essential concepts from optional deeper dives.

For every practical section include:

1. What problem this step solves.
2. The concept in plain language.
3. The exact commands or minimal complete code we actually used.
4. Expected behavior and our observed output.
5. Common failure modes and how we diagnosed them.
6. The artifact or checkpoint readers can inspect to verify their progress.

Avoid unexplained jargon, giant unexplained notebook cells, invented outputs, or optimistic benchmark screenshots without context. Explain why fine-tuning was appropriate and when prompting or a specialist classifier would be enough.

### 17.3 Complete article outline

| Section | Teaching content and required evidence |
| --- | --- |
| 1. The problem and final outcome | Real use case, honest final demo, supported inputs, headline results with dataset context |
| 2. What a VLM does | Image encoder, language backbone, processor, image tokens, generation versus classification |
| 3. Why Liquid and Unsloth | Exact model choice, verified compatibility, alternatives considered, limitations and license |
| 4. Scope before training | Crop ID versus object detection; screening versus confirmed cause; unknown states; later RAG boundary |
| 5. Set up the environment | GPU choice, Python/CUDA/dependency versions, reproducible install, cost controls |
| 6. Audit the data | Sources, licenses, labels, balance, duplicates, shortcuts, field-domain gap |
| 7. Split without leakage | Source grouping, immutable manifests, validation use, locked tests, field holdout |
| 8. Build training examples | Image–conversation structure, chat template, target schema, assistant-label masking |
| 9. Measure before fine-tuning | Untuned prompt baseline, classifier comparison, same-image/class-mapping rules |
| 10. Understand and configure LoRA | Frozen weights, low-rank adapters, trainable parameters, rank, alpha, learning rate, memory |
| 11. Run the first training job | Collator, preprocessing, precision, batch/accumulation, loss, tiny-batch check, save/reload |
| 12. Adapt autoresearch responsibly | Original workflow, our modified objective, protected evaluator, fixed work, deadlines, stop conditions |
| 13. Walk through experiments | Hypotheses, one-setting changes, keep/discard decisions, failures, timeouts, reproducibility |
| 14. Understand evaluations | Confusion matrix, precision/recall/F1, macro averaging, class imbalance, invalid outputs, coverage |
| 15. Report actual results | Before/after tables, per-class support, three-seed results, field results or explicit absence, latency/memory/cost |
| 16. Inspect mistakes | Concrete failures, spurious cues, rejection tradeoffs, why generated confidence is not calibration |
| 17. Deploy the demo | Input handling, model+adapter loading, structured API response, UI, timing, deployment constraints |
| 18. What we would improve | Data gaps, classifier/VLM tradeoffs, quantization only if verified, future field validation |
| 19. Extend with RAG later | Market survey, authoritative label retrieval, citations, separate evaluation; no fabricated completed module |
| 20. Reproduce and continue | Exact repository revision, commands, downloadable permitted artifacts, total costs, limitations, references |

### 17.4 Capture evidence while building

Maintain an evidence journal alongside the implementation. Each substantive experiment/session should leave:

- Date, question, decision, rationale, and next step.
- Run ID and exact code/configuration/data revisions.
- Actual commands and useful sanitized logs.
- Debugging notes, including incorrect assumptions and fixes.
- Metrics linked to raw predictions and evaluation settings.
- Figures with captions, units, dataset/split names, and source artifact IDs.
- Shareable before/after predictions and permission/provenance notes for example images.
- Wall-clock timing, hardware, GPU allocation, and costs.
- Notes on explanations that a beginner will need.

Capture enough to reconstruct the narrative without reconstructing undocumented experiments later. Do not log credentials or upload private photos into the public evidence folder.

### 17.5 Required publication figures and tables

- System architecture and training/evaluation workflow diagrams.
- Dataset class distribution and split summary with grouping policy.
- Illustrated training example and label-masking explanation.
- Baseline-versus-fine-tuned metrics with identical evaluation scope.
- Experiment ledger summary, including unsuccessful trials.
- Training/validation curves where collected.
- Crop and condition confusion matrices and per-class support.
- Accepted-case quality versus coverage/rejection behavior.
- Three-seed confirmation summary.
- Field-versus-in-corpus comparison, or an explicit missing-field-data statement.
- Latency, memory, and cost table with hardware and measurement method.
- Annotated failure gallery and final demo screenshots.

### 17.6 Publication workflow

1. Create an outline and evidence journal when implementation starts.
2. Draft concept/setup sections as those steps are verified.
3. Draft experiment/evaluation sections only from recorded artifacts.
4. Complete the end-to-end article after final evaluation and demo.
5. Reproduce the public instructions from a clean environment and pin the article to a release/commit.
6. Check every numerical claim against its source artifact; separate targets, historical results, and our measurements.
7. Review source attribution, image permissions, accessibility of figures, and beginner readability.
8. Prepare Substack-ready text and figures; publish only when requested.

Use `docs/article/substack-draft.md` in the standalone repository as the working draft when writing begins. Keep this file as the project plan, not a mixture of unfinished prose and final results. A complete article may include optional expandable explanations or companion notebook links, but must remain coherent as one article.

## 18. Current milestone checklist

- [x] Research exact-checkpoint Liquid/Unsloth feasibility using primary sources.
- [x] Write the living project, standalone-repository, and Substack plan.
- [x] Create and verify the standalone GitHub repository.
- [x] Bootstrap documentation in a separate local checkout and make this plan canonical.
- [x] Inspect Windows and WSL GPU visibility and the available Python interpreter.
- [x] Confirm a free local GPU window and create an isolated WSL environment with pip/uv.
- [ ] Complete the cached dependency downloads and verify basic Unsloth/PyTorch CUDA execution.
- [x] Prepare and review the documentation-only bootstrap for initial publication; Git history records commit/push state.
- [ ] Confirm hardware, spend ceiling, crop scope, and deployment target.
- [ ] Audit datasets and freeze taxonomy/splits.
- [ ] Reproduce the Unsloth compatibility smoke run and lock dependencies.
- [ ] Establish baseline metrics and numeric acceptance targets.
- [ ] Implement and test bounded experiment supervision.
- [ ] Run initial daytime search sessions.
- [ ] Confirm the winner across three seeds and evaluate locked tests.
- [ ] Complete error analysis, model/dataset cards, and demo.
- [ ] Finish the complete article and verify reproduction instructions.
- [ ] Publish approved repository artifacts and Substack article.
- [ ] Conduct the later treatment-RAG market survey and source review.

## 19. Open decisions

| Decision | Current state |
| --- | --- |
| GPU, VRAM, OS/provider | RTX 6000 Ada, 49,140 MiB total; Ubuntu WSL2 selected; user confirmed availability and 46,493 MiB free was observed |
| Total GPU and agent/API budget | Unconfirmed |
| Initial crop coverage | Five-crop photo check proposed, subject to data audit |
| Screening class subset | Unconfirmed; start with well-supported classes |
| Independent field data and annotation access | Unconfirmed |
| Public code license | Unconfirmed; separate from model/dataset terms |
| Public data/adapter release permissions | Must be established per source/artifact |
| Final deployment hardware | Unconfirmed |
| Numeric quality/rejection/latency targets | Set after baselines, before tuning |
| Urdu output | Deferred pending separate evaluation |
| Substack publication date | After measured results and reproducibility review |
| Initial publication of repository files | Reviewed documentation only; consult repository history for publication state |
| Dependency installation | Blocked by incomplete large downloads within the bounded attempts; cache retained for resumption |

## 20. Decision and update log

| Date | Decision/update | Rationale |
| --- | --- | --- |
| 2026-11-11 | Plan LFM2.5-VL-1.6B with Unsloth language-side LoRA and frozen vision encoder | Strongest inspected exact-checkpoint training evidence; reproduction still required |
| 2026-11-11 | Start with photo checks and evaluate screening separately | Preserve task boundaries and make gains/failures measurable |
| 2026-11-11 | Use fixed training work with hard daytime session limits | Reproducibility and explicit no-overnight requirement |
| 2026-11-11 | Build an independent public portfolio repository | Reproducible showcase without copying the portal and its history |
| 2026-11-11 | Produce one complete beginner-focused Substack walkthrough after experiments | Teach the full workflow using real evidence, including failures and limitations |
| 2026-11-11 | Keep pesticide recommendations for later market-informed RAG | Visual SFT does not establish current treatment knowledge |
| 2026-11-11 | Bootstrap a sibling checkout and migrate the canonical plan | Keep the public project independent and avoid diverging plan copies |
| 2026-11-11 | Propose local Ubuntu WSL2 on the discovered RTX 6000 Ada | GPU access works in WSL; availability and training compatibility still need confirmation |
| 2026-11-11 | Select local WSL2 after the user confirmed GPU availability | Rechecked 46,493 MiB free; no cloud booking needed for setup |
| 2026-11-11 | Stop incomplete dependency downloads after bounded attempts and preserve the cache | Avoid unattended indefinite setup; document the actual blocker instead of claiming a smoke-test success |

## 21. Sources

Primary sources inspected during planning; pages and branches may change. Pin the revisions actually used during implementation.

- **S1:** [Unsloth Liquid LFM2.5 text and vision guide](https://unsloth.ai/docs/models/tutorials/lfm2.5.md).
- **S2:** [Liquid LFM2.5-VL-1.6B model card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B).
- **S3:** [Official Unsloth vision notebook with saved outputs, pinned revision](https://github.com/unslothai/notebooks/blob/c834e1741666c50e3b7a57f56e7f6855229ac2ac/nb/LFM2.5_VL_(1.6B)-Vision.ipynb).
- **S4:** [Liquid vision model library](https://docs.liquid.ai/lfm/models/vision-models.md).
- **S5:** [Liquid model deprecations](https://docs.liquid.ai/lfm/help/deprecations.md).
- **S6:** [Karpathy autoresearch](https://github.com/karpathy/autoresearch).
- **S7:** [Selected checkpoint license](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/LICENSE) and [Liquid license guide](https://docs.liquid.ai/lfm/help/model-license.md).
- **S8:** [PlantVillage author dataset card](https://huggingface.co/datasets/mohanty/PlantVillage).
- **S9:** [PlantDoc author repository](https://github.com/pratikkayal/PlantDoc-Dataset).
- **S10:** [Original PlantVillage experiment and external-image results](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full).
- **S11:** [Liquid TRL fine-tuning guidance](https://docs.liquid.ai/lfm/fine-tuning/trl.md), a documented fallback if exact Unsloth reproduction fails.

See the [standalone feasibility memo](docs/research/lfm-vision-feasibility.md), [article outline](docs/article/substack-outline.md), and [evidence journal](docs/article/evidence-journal.md). The original portal-specific research memo remains in the agriculture workspace as historical context.
