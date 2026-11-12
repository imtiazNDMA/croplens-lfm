# Dataset audit

A read-only audit of the existing crop disease corpus and of the photo-check evaluation set, run on 2026-11-11 before any CropLens training. Nothing in the source datasets, manifests or splits was changed.

**Verdict: the corpus is usable for crop identification and a subset of condition screening only after two problems are handled.** Near-duplicate photos cross the train/val/test boundary, and image size alone gives the crop away. Neither is visible in the headline numbers, and both would inflate any score measured on this data as it stands.

## What was run

| Run | Command | Result |
| --- | --- | --- |
| `reports/dataset-audit-001.json` | `audit_dataset.py` | Completed in 594 s, 6 s inside its 600 s cap |
| `reports/dataset-audit-002.json` | same, after the path-check fix below | Completed in 160 s; identical to 001 except for elapsed time |
| `reports/dataset-shortcuts-001.json` | `shortcut_checks.py` | 12 s; reads image headers and the stored hashes only |

The audit validates every manifest row (path stays inside the image root, SHA-256 and dHash format, status, split, taxonomy), reads and hashes every image, fully decodes it and recomputes its dHash, scans the image folder for unlisted files, compares dHashes across splits, and checks the photo-check set's files and the parents of its derived images. Ten unit tests cover the logic.

**Why the first run nearly timed out.** It was not hashing or decoding: the audit's own per-image check takes 73 s for all 17,952 images with eight threads. Progress output showed 426 of the 594 seconds went to validating the manifest, where `image_path()` resolved the full path (several filesystem lookups, each slow across WSL's Windows-drive mount) three times per image. Each directory is now resolved once, with tests that a symlinked directory or file cannot lead outside the image root and that 500 images in one directory cost at most four resolves.

## Confirmed

These match the source project's report, now independently checked.

| Check | Result |
| --- | --- |
| Manifest rows | 17,952 (kept 11,453; duplicate 6,398; conflict 55; rare class 46) |
| Images on disk | 17,952, none unlisted |
| Missing, undecodable or hash-mismatched files | 0 |
| Kept images by split | train 8,013; val 1,720; test 1,720 |
| Kept by crop | Maize 3,633; Rice 2,635; Sugarcane 1,980; Wheat 1,643; Cotton 1,562 |
| Kept by category | disease 7,789; healthy 2,198; pest 1,466 |
| Retained classes | 37 of 39 in the taxonomy; `cotton_boll_rot` and `cotton_pink_bollworm` have no kept images |
| Healthy rice | none: Rice has four disease classes and no healthy class |
| Exact (SHA-256) or manifest-group overlap across splits | 0 |
| Cross-split near-duplicates at the manifest's 12-bit threshold | 0 |
| Photo-check set | 1,400 files, no errors; every five-crop and poor-photo parent is in the manifest, in val (tune, poor) or test (final), none in train; no exact or near overlap with training |

## Found

### 1. Near-duplicates do cross the splits

The manifest's threshold counts two photos as copies when their 256-bit dHashes differ by 12 bits or fewer. Counting pairs between splits at looser thresholds:

| Bits | train vs val | train vs test | val vs test |
| --- | --- | --- | --- |
| 12 | 0 | 0 | 0 |
| 16 | 40 | 28 | 6 |
| 24 | 105 | 87 | 22 |
| 32 | 153 | 130 | 40 |

The closest cross-split pair sits at 13 bits, one past the threshold. There are 192 train-vs-val/test pairs within 24 bits, **all with the same label**, and 151 of them are rice tungro. I looked at two of the closest pairs: the rice blast pair (13 bits) is the same photograph with different JPEG compression, down to the "PhilRice" watermark, one copy in train and one in val; the sugarcane red rust pair is two near-identical frames. So the original threshold was too tight: the source project's note says grouping was stable from 8 to 16 bits and chained look-alike photos beyond that, but at least these two pairs just above 12 bits are real copies.

What this does not show: how many of the 192 pairs are true duplicates (I inspected two), and how much a model gains from them. 192 pairs is small against 3,440 evaluation photos, but it is concentrated in a few classes and the true figure could be higher, since dHash does not match flipped, rotated or cropped copies at all.

### 2. Augmented photos may have parents anywhere

635 filenames look augmented (`crop_`, `rotation_`, `translation_`, `flip_`, `zoom_`, `contrast_`); 430 of them are kept: 307 train, 69 test, 54 val. They fall in just four classes: `cotton_aphid` (131), `maize_armyworm` (128), `cotton_bacterial_blight` (93) and `cotton_healthy` (78). No manifest field records the original photo, and dHash does not match flips or rotations, so the near-duplicate check cannot see these parents. The risk is concentrated: those four classes' test scores are the least trustworthy. Cross-split leakage from them is **not proven**; it cannot be ruled out with the current metadata.

### 3. Image size gives the crop away

| | Result |
| --- | --- |
| Distinct image sizes among kept photos | 2,599 |
| Photos at 256x256 | 2,884, of which 2,865 Maize and 19 Cotton |
| Photos at 300x300 | 1,610, of which 1,609 Rice and 1 Maize |
| Photos at 1040x493 or 493x1040 | 1,134, all Sugarcane |
| Photos at 6000x4000 | 119, all Wheat |

Predicting the crop from exact image size alone (the most common training crop at that size) is right for **81.7% of test photos and 81.7% of val photos**, against 31.7% for always guessing the most common crop. For photos whose size was seen in training (82.2%) it is right 96% of the time. A model that resizes to 224x224 loses the size itself but not what comes with it: the source, the compression, the camera look. Whether the chosen Liquid checkpoint sees images at near-native resolution, which would make the size directly visible, is not yet confirmed from its model card. **A high crop-identification score on this data therefore cannot be read as "the model recognises the crop".**

### 4. Provenance and rights are not recorded

The manifest has no source URL, dataset revision, licence, original-photo ID, field or collection-session field. No README or licence file was found in the `dataset_filtered` folder (a filename search, not proof they exist elsewhere). The folder names and image names point to several origins, and at least one image carries a "PhilRice" watermark (a Philippine rice research institute), so the data is not all from Pakistan. **Nothing here can yet be redistributed or described as Pakistani field data.**

### 5. Coverage gaps

- No healthy rice. The photo check can still name rice; a condition screen cannot say "healthy" for it.
- Every class with fewer than 30 val photos has too few to report honestly: 19 of 37 classes. Examples: `cotton_american_bollworm` (8 val/test), `cotton_anthracnose` (4), `wheat_aphid` (29), `wheat_stem_fly` (23).
- Heavy class imbalance within crops: maize common rust has 1,318 kept photos and maize stem borer 95.
- Class-level photos are heavily clustered by source (finding 3), so even large classes may have few independent sources.

## Proposal

These are proposals. They change nothing until confirmed.

### Splits: keep the existing holdouts, clean the training pool

The existing val and test splits have been used (the portal's classifier was selected on val and scored once on test), so they stay as they are, and none of their images goes into training. To remove the leakage without regenerating anything:

1. Training pool = the existing train split, minus any train image within 32 bits of any val or test image (there are 283 train-vs-val/test pairs within 32 bits; the number of distinct train images is counted when the audit is re-run), minus the 307 augmentation-named train images, until their parents can be traced.
2. Re-run the audit on the pool and publish the removed count per class. Exit gate: zero cross-split pairs within 32 bits.
3. Evaluate on the existing val (selection) and test (scored once at the end), and report a second score on the **evaluation images that share no size with a different crop** and a third on **size-normalised copies** (centre crop to a common aspect ratio, resized to one size, recompressed at one quality) to show how much of the score is the shortcut. If the gap is large, say so.
4. Independent field photos (#18) remain the only honest evidence of Pakistan performance and are not yet available.

### Photo-check output contract (stage 1)

The same shape the portal's photo check uses, so results are comparable: `is_plant`; `crop` one of Rice, Maize, Cotton, Wheat, Sugarcane, Other, Unclear; `other_plant` text; and the quality flags blurred, too dark, not a close-up, several leaves. Invalid answers count as failures. Report accuracy and the share handed back as Unclear separately, as the portal's evaluation does.

### Condition screening (stage 2): a start set

Start with the 18 classes that have at least 30 val and 30 test photos: `cotton_aphid`, `cotton_bacterial_blight`, `cotton_healthy`, `cotton_wilt`, `maize_armyworm`, `maize_common_rust`, `maize_gray_leaf_spot`, `maize_healthy`, the four rice disease classes, `sugarcane_healthy`, `sugarcane_mosaic`, `sugarcane_red_rot`, `sugarcane_yellow_rust`, `wheat_healthy`, `wheat_powdery_mildew`. Report the other 19 as unsupported rather than hiding them in an "other" bucket. The four augmentation-heavy classes in this list (`cotton_aphid`, `cotton_bacterial_blight`, `cotton_healthy`, `maize_armyworm`) need the note in finding 2.

### Negatives and rejection examples

The photo-check set's other-plant and non-plant photos are the portal's evaluation assets and stay out of training. Training negatives must come from different images: other splits of PlantVillage (CC-BY-SA-3.0 on the author's release, to be confirmed for the mirror used), Imagenette's other images (derived from ImageNet, whose terms need reading before any public release), and PlantDoc for non-laboratory plants. Poor-photo examples can be made by degrading training photos only, never the evaluation parents.

## Not done

- No licence or provenance review of the main corpus beyond the findings above; the source of each folder is unknown.
- No physical-leaf, field or session grouping: the data does not carry it.
- No check that labels are correct, beyond what the taxonomy and the source project already did.
- Pairs beyond 24 bits and flipped, rotated or cropped copies are not searched. A stronger check (hashes of all eight flips and rotations, or image embeddings) would cost about an hour and would put numbers on finding 2.
- Nothing was trained.

## Decisions needed

1. Accept the split proposal (keep holdouts, clean the training pool at 32 bits, report three scores)?
2. Start with the 18-class set, or first look for healthy rice and more support for the rest?
3. Which sources may be used for negatives, once their terms are read?
