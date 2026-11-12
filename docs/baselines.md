# Baselines on the photo-check `tune` part

Measured 2026-11-11 on the 700 `tune` photos of the portal's photo-check set (#93): 500 five-crop photos, 100 other plants, 50 non-plants, 50 poor photos. The five-crop photos are a sample of the crop-disease **val** split. Nothing here touches the `final` part (a sample of the locked test split), which is scored once, for the chosen model, at the end.

| | Qwen3.8-27B (portal, tuned prompt) | LFM2.5-VL-1.6B untuned | ConvNeXt five-crop-001 |
| --- | --- | --- | --- |
| Crop named correctly, crop not given | 51.2% | 5.0% | 98.4% |
| Handed back as Unclear | 30.4% | 0.0% | cannot |
| Crop photos wrongly stopped | 6.8% | 67.8% | cannot |
| Non-plants stopped | 100.0% | 100.0% | cannot |
| Other plants stopped | 88.0% | 98.0% | cannot |
| Invalid answers | 3.0% | 23.3% (23.3% also under lenient reading) | n/a |
| Disease class right, crop given (500 photos) | n/a | n/a | 97.6% (macro F1 0.962) |
| Disease class right, crop not given | n/a | n/a | 96.6% (macro F1 0.951) |
| Seconds per photo | not timed here (27B model served by LM Studio) | 0.96 s | not timed (run on CPU) |
| Peak GPU memory | n/a | 3.288 GiB | none |

The #93 bars for letting the photo check run on Auto are crop identification at least 97%, false stops at most 2%, non-plants stopped at least 95%, other plants stopped at least 85%.

## What each column is, and is not

- **Qwen** is the portal's shipped model after three rounds of prompt tuning **on these same tune photos**, so its tune numbers flatter it. Its answers are the recorded #93 readings, rescored here by this project's scorer; they match the portal's report (crop identification 51.2%, false stops 6.8%).
- **Untuned LFM** got the portal's instructions plus a worked example of the answer format, one prompt, no tuning of it. Without training it answers "Other" for most plant photos (67.8% of crop photos stopped) and 23.3% of its answers break the format (it describes the photo in prose, drops the crop, or names a disease as the crop), so it is not usable for this job as it comes. A fence-stripping reading changes nothing here: none of its failures are code fences.
- **ConvNeXt** is the portal's disease classifier. "Crop not given" means the crop is the crop of its best class over all 37. It cannot say "not a plant", "other plant" or "unclear", so only the five-crop photos are scored for it.

## Cautions

- ConvNeXt's 98.4% would clear the crop-identification bar, but the audit found image size alone predicts the crop for 81.7% of photos, and each crop's photos come from different sources. A model can score high by recognising the source's look. Qwen and the untuned LFM have no training on this data and so cannot do that; ConvNeXt and any model fine-tuned here can. The size-normalised score must accompany every fine-tuned result.
- One prompt and one decoding setting were used for the untuned LFM. A different prompt could change its numbers, but a 5% crop identification is unlikely to be a prompt problem alone.
- 500 photos per crop-ID figure: roughly plus or minus 4 points at 95% confidence near 50%, tighter near 0 or 100.

Raw files: `reports/baselines/` (`lfm-untuned-tune.json` holds every answer and its time).

## First fine-tuned run (`ref-001`), on the same `tune` photos

Reference configuration, trained once: 150 steps of 16 examples (2,400 of the pool's 8,091 examples, about 0.3 of an epoch), LoRA rank 16 on the language side with the vision encoder frozen, learning rate 2e-4, seed 3407, 9,142,272 trainable parameters. Training took 177 s and 3.35 GiB at peak. The loss fell from 2.8 to about 0.01. Scored with the short prompt, photos capped at 512 px.

| | Untuned LFM | Qwen (portal) | ConvNeXt | **LFM ref-001** |
| --- | --- | --- | --- | --- |
| Crop named correctly | 5.0% | 51.2% | 98.4% | **88.6%** |
| Wrongly stopped | 67.8% | 6.8% | cannot | 0.0% |
| Non-plants stopped | 100% | 100% | cannot | **0.0%** |
| Other plants stopped | 98% | 88% | cannot | **0.0%** |
| Invalid answers | 23.3% | 3.0% | n/a | 1.0% |
| Quality flag raised on poor photos / on clean crop photos | n/a | 96% / 13.2% | n/a | 98% / 0.2% |
| Seconds per photo, peak GPU memory | 0.96 s, 3.29 GiB | n/a | n/a | 0.97 s, 3.05 GiB |

Per crop (right out of 100): Cotton 94, Maize 93, Sugarcane 95, Rice 84, Wheat 77. Wheat is mistaken for maize (13), rice (7) and cotton (3); rice for wheat (11).

**Reading it.** Three minutes of training took crop identification from 5% to 88.6%, far above Qwen, and cut invalid answers to 1%. It is still below ConvNeXt's 98.4% and the 97% bar, and the rest of the bars cannot be judged yet: the training data holds no other plants or non-plants, so the model never says "Other" or "not a plant" and stops 0% of the photos that should be stopped. That is the expected result of the data, not a finding about the model. Quality flags are the clearest gain: 98% of poor photos flagged against 0.2% of clean ones, against Qwen's 13.2% on clean ones. The poor photos are made with a recipe close to the training one, so that score is easier than a real-world one.

**Not yet shown.** The 88.6% has not been checked for the source-shortcut problem. Until the same model is scored on size-normalised copies of these photos, it cannot be read as "recognises the crop". One configuration, one seed, 0.3 of an epoch: no search has been done.

## Is it the crop, or the size? (`ref-001` and ConvNeXt, `tune` photos)

All 700 `tune` photos were also made size-normalised (`normalize.py`: centre-cropped square, 512 px, JPEG quality 85, metadata dropped) and both systems re-scored on those copies. The size-ambiguous subset is the 179 of 500 crop photos whose exact pixel size also occurs under a different crop (maize 80, rice 59, cotton 19, wheat 16, sugarcane 5).

| Crop named correctly | Photos as they are | Size-normalised | Size-ambiguous subset, as they are | Size-ambiguous subset, normalised |
| --- | --- | --- | --- | --- |
| LFM ref-001 | 88.6% | **83.0%** | 90.5% (of 179) | **78.2%** (of 179) |
| ConvNeXt five-crop-001 | 98.4% | 98.0% | 98.9% (of 179) | 98.9% (of 179) |

Per crop for ref-001 on normalised copies (right of 100): Cotton 89, Maize 84, Rice 81, Sugarcane 91, Wheat 70. Quality flags held (96% of poor photos, 1% of clean ones); invalid answers 0%.

**Reading it.** ConvNeXt does not move, so its high score does not rest on the pixel size, aspect ratio or compression of the photos; it may still lean on colour and texture that differ by source, which these copies keep. The fine-tuned LFM does move: it loses 5.6 points overall and 12.3 on the size-ambiguous subset (about plus or minus 6 points at that size), so part of its 88.6% came from cues the normalising removes. Centre-cropping also discards the edges of some photos, which could cost any model a little; ConvNeXt shows no such cost. **83.0% is the fairer figure for ref-001**, and it is 15 points under ConvNeXt and 14 under the 97% bar.

The deciding comparison for replacing ConvNeXt's crop reading, once the model is tuned, is the normalised score, not the as-is one.
