# Substack article: working outline

**Audience:** beginner AI engineers and software engineers transitioning into AI engineering.

**Working title:** Fine-Tuning a Vision-Language Model for Crop Identification and Disease Screening: A Complete Engineering Walkthrough.

**Status:** outline only. Training, evaluations, demo, and final retrospective have not been completed. The eventual title and conclusions must match the work actually completed.

The [living plan](../../fine-tuning.md) defines scope and milestones. The [evidence journal](evidence-journal.md) captures the actual process. This outline is the writing workspace, not a second project plan.

## Teaching pattern

For each practical step: explain the problem, introduce the concept in plain language, show the actual minimal complete commands/code, compare expected behavior to observed output, explain failures, and link a verifiable artifact.

Assume Python, Git, and API familiarity. Explain model weights, processors, image tokens, SFT, LoRA, GPU memory, grouped splits, and macro F1 before relying on those terms.

## Complete narrative

| Section | What the reader learns | Evidence to collect |
| --- | --- | --- |
| 1. Problem and honest outcome | What the project does and what it cannot claim | Final demo and results with evaluation context |
| 2. How a VLM works | Vision encoder, language backbone, image tokens, processor, generation | Architecture diagram and an inspected input example |
| 3. Why this model and framework | Compatibility, size/quality tradeoffs, licenses | Exact model revision and reproduced software versions |
| 4. Define the task | Crop identity versus object detection; screening and unknown states | Taxonomy and output contracts |
| 5. Set up compute | CUDA, environment isolation, memory, session limits, costs | Hardware discovery and verified install commands |
| 6. Audit the dataset | Provenance, labels, imbalance, duplicates, shortcuts | Source inventory, counts, exclusions, label audit |
| 7. Split without leakage | Grouping, validation selection, locked tests, field holdouts | Manifest revision, group checks, split summary |
| 8. Construct SFT examples | Image–conversation pairing, templates, masking, truncation | Actual collated example and masking checks |
| 9. Baselines before training | Why an improvement needs a fair comparison | Untuned-LFM and eligible classifier measurements |
| 10. LoRA explained | Low-rank updates, frozen weights, rank/alpha, learning rate | Actual trainable parameters and configuration |
| 11. First working run | Precision, batch/accumulation, loss, tiny-batch test, save/reload | Smoke-run logs and assertions |
| 12. Adapt autoresearch | Protected evaluation, fixed work, deadlines, keep/discard decisions | `program.md`, runner behavior, stop-condition tests |
| 13. Experiment walkthrough | Hypotheses, controlled changes, failures and recovery | Complete ledger, including discarded runs |
| 14. Understand metrics | Precision/recall/F1, macro averaging, rejection, invalid answers | Worked example and evaluator definitions |
| 15. Report results | Full-set versus accepted-case quality, seed variation, field gap | Machine-readable metrics, support, confusion matrices |
| 16. Inspect errors | Spurious cues, ambiguous leaves, domain shift, confidence limits | Annotated, shareable failure gallery |
| 17. Deploy a demo | Input validation, adapters/processors, structured API, timing | Verified serving commands and screenshots |
| 18. Account for costs and tradeoffs | GPU allocation, latency, memory, classifier/VLM choice | Measured cost and quality/efficiency table |
| 19. Treatment RAG roadmap | Retrieval grounding and why image SFT is insufficient | Clearly future work unless actually completed |
| 20. Reproduce and continue | Exact release/commit, commands, artifacts, limitations | Clean-environment reproduction and references |

## Publication rules

- Write setup sections after verification and result sections after measurement.
- Never turn historical portal metrics, published notebook outputs, or proposed targets into CropLens results.
- Show per-class sample counts and the evaluation split behind every performance claim.
- Include unsuccessful trials and what the final system chose not to use.
- Use captioned figures with units, source run IDs, and readable labels.
- Keep private data and credentials out of examples and screenshots.
- Complete one coherent article; optional deep dives and companion notebooks may support it.
- Verify the public reproduction path before publishing and link the article to a specific revision.

Create `substack-draft.md` when substantive drafting begins. Publication requires a separate user request.
