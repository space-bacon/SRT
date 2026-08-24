---
license: apache-2.0
tags:
  - srt
  - interpretability
  - cross-modal
  - substrate-invariance
  - artifacts
---

# SRT gemma-4 artifacts

Every number in the SRT program's cross-modal and invariance sections has a file
here. This repository is the evidence, not the model: encoded states, fitted
heads, result JSONs, and run logs.

Papers: [`arxiv_program/paper.md`](https://github.com/space-bacon/SRT/blob/main/arxiv_program/paper.md).
Code: [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT).

**Reproduction principle.** If a claim is in the paper and you cannot rebuild it
from the files here plus the scripts in the GitHub repo, that is a bug and we
want to hear about it. Two such bugs have been found by external review and
both are fixed: the cross-runtime validation that never loaded the head, and the
gemma/gemma v6 head that existed nowhere. See the ledger in §9 of the paper.

---

## Layout

### `procrustes/` — the cross-modal ladder on gemma-4-31B (layer 47)

| file | what it is |
|---|---|
| `train_pairs/chunk_000..023.pt` | 118,287 COCO train2017 image/caption state pairs, d=5376 |
| `encoded_L47_n5000.pt` | the 5,000-image evaluation pool |
| `mlp_align.json` | train-size ladder at n=1,000 / 2,000 / 3,500 |
| `mlp_align_full.json` | ladder at n=39k / 60k / 90k / 117k, no hard negatives |
| `mlp_align_scaled.json` | scaled variant |
| `procrustes_xmodal.{json,W.pt}` | the orthogonal-map rung, which fails; the modality gap is not a rotation |
| `karpathy_eval.json` | Karpathy-protocol retrieval |
| `chain.log` | run log |

**Reading `mlp_align_full.json` correctly.** `linear_n117000` i2t R@1 0.661 is the
**no-hard-negative** recipe. Heads trained with compositional hard negatives at
weight 1.0 land at 0.622 clean i2t by design, trading retrieval for
compositional accuracy. Comparing a hard-negative head against 0.661 understates
it by about four points.

### `q4/` — quantization, runtime drift, and the tower factorial

Heads (all linear, 1024-dim projection, tau 0.05, seed 0, trained on the 118k
pairs above with K=4 compositional negatives at weight 1.0):

| file | image tower | text tower |
|---|---|---|
| `gemma_v6_head.pt` | gemma-4-31B L47 (5376) | gemma-4-31B L47 (5376) |
| `mixed_v6_head.pt` | gemma-4-31B L47 (5376) | Qwen2.5-VL-3B L29 (2048) |
| `cell4_v6_head.pt` | Qwen2.5-VL-3B L29 (2048) | gemma-4-31B L47 (5376) |
| `qwen3b_v6_head.pt` | Qwen2.5-VL-3B L29 (2048) | Qwen2.5-VL-3B L29 (2048) |
| `qwen3b_q4_head.pt` | native-Q4 head from the quantization work | |

`gemma_v6_head.pt` is a **rebuild**. The original existed neither locally nor
here, which external review would have found; it was regenerated from
`procrustes/train_pairs` plus a seed-0 regeneration of the v6 negatives
(473,148 at K=4, `prep_replace` 75,786, `adj_transfer` 54,894, matching the
published counts to the item). It reproduces `baseline_centered` 0.288 and the
shuffled control 0.001 exactly, and `linear` 0.620 against 0.622.
Details in `mlp_align_gemma_v6_rebuild.json`.

SugarCrepe states, so the benchmark can be scored without a GPU:

| file | contents |
|---|---|
| `sugarcrepe_img_states_gemma_L47.npz` | 1,560 images, d=5376 |
| `sugarcrepe_txt_states_gemma_L47.npz` | 11,844 captions, d=5376 |
| `sugarcrepe_img_states_qwen3b_L29.npz` | 1,560 images, d=2048 |
| `sugarcrepe_txt_states_qwen3b_L29.npz` | 11,844 captions, d=2048 |

Together with the four heads these reproduce every cell of the tower factorial
on CPU in seconds via `scripts/sugarcrepe_controls.py`.

Controls and results:

| file | contents |
|---|---|
| `controls_*.json` (4 cells) | every null in one place: real, pairing-deranged, mean-image, foreign-split mean, random direction, blind bigram prior, typicality regression |
| `mean_image_control.json` | the round-13 ablation, later withdrawn (see below) |
| `blind_bigram_prior.json`, `prior_comparison.json`, `margin_over_blind.json` | the blind word-order prior and both independent rebuilds |
| `sugarcrepe_*.json`, `mlp_align_*_v6.json` | per-cell raw accuracy and fit records |
| `q4_drift.json`, `procrustes_q4.json`, `mlp_align_q4.json` | bf16 to 4-bit drift |

**Which null to trust.** Three of the four are blind to the thing they null: the
bigram prior, the mean-image ablation, and a random fixed direction never open
an image, and on `swap_obj` two of them sit 17.8 points apart, so the *sign* of
"does image content help" was being set by null choice rather than measurement.
The mean-image ablation is reproduced by *another* split's mean image and
defeated by a random direction, so it prices a learned generic-image direction
in text space, not image content. Its apparent win over real images was the
caption formatting and does not survive normalisation. Candidate position
exchange cannot arbitrate either: it is an identity for an independent-scoring
cosine.

Use `shuffle_pair` (real images, trained head, pairing deranged over 20 seeds).
It is the only null that varies per cell. Against it, on normalised captions,
the heads clear by +16.3 to +18.6 clean-five points.

**Normalise the captions first.** SugarCrepe pairs a human COCO caption with a
model-generated foil and the generator punctuates: the foil is the dotted one in
1,384 of the 1,384 pairs where the two differ, no counter-example in 7,511.
States cached under the raw caption carry that signature. Stripping it reverses
three conclusions we published: both swap splits go from at or below their
deranged control to about ten points above it, the mean-image ablation stops
outscoring real images, and the text-tower asymmetry inverts and shrinks to
under a point on raw accuracy. The un-normalised numbers are kept because they
are what other work quoting SugarCrepe is comparable to, not because they are
the better measurement. Normalised state caches are published here as
`q4/sugarcrepe_txt_norm_*.npz`, with `q4/sugarcrepe_txt_reenc_*.npz` as the
same-session re-encode control (floor 0.0042 on any arm).

### `scalefloor/` — 31B to 3B, no capability loss at matched data

`encoded_L29_n5000.pt`, `qwen3b_linear_head.pt`, `mlp_align_qwen3b.json`,
`procrustes_qwen3b.json`.

### Post-training modification (the abliteration axis)

| file | contents |
|---|---|
| `qwen38_abliteration_layer_sweep.json` | per-layer displacement, Qwen3.8-27B base vs abliterated |
| `qwen38_abliteration_capability.json` | cross-entropy by group, plus interpretation and limits |

Both models held resident on separate GPUs with the same batch through each, so
the comparison carries no re-encode floor. Headline: layers 0 to 14 displaced
*exactly* zero, onset at 15, peak rho 0.074 at layer 48, cosine never below
0.9949. For scale, cross-runtime drift in the same units is rho 1.334 (MLX-Q4)
and 0.749 (NF4), so a behavioural weight edit is roughly 10 to 18 times smaller
geometrically than a quantizer swap. Capability holds (+0.028 nats on held-out
text) while refusal-register text costs +0.479. The effect is register-local,
not topic-local: contestedness tiers are flat.

### `overnight/`, `base_anchors_L47.json`, `base_kcurve_ce.jsonl`

Run logs, anchor sets, and the base-model K-curve.

---

## Quick start

```python
from huggingface_hub import hf_hub_download
import numpy as np, torch

R = "RiverRider/srt-nla-gemma4-artifacts"
head = torch.load(hf_hub_download(R, "q4/gemma_v6_head.pt"), weights_only=True)
img  = np.load(hf_hub_download(R, "q4/sugarcrepe_img_states_gemma_L47.npz"), allow_pickle=True)
txt  = np.load(hf_hub_download(R, "q4/sugarcrepe_txt_states_gemma_L47.npz"), allow_pickle=True)

# head-space projection: centre by the head's own means, then project
zi = (img["states"] - head["mu_img"].numpy()) @ head["img"]["weight"].numpy().T + head["img"]["bias"].numpy()
```

Then score with `scripts/sugarcrepe_controls.py --cell gemma-img_x_gemma-txt`.

## Conventions that matter

- **Anisotropy.** Raw cosine on these states is not interpretable. Centre by the
  pool mean before any similarity claim, and report the anisotropy baseline.
- **Margins over blind priors as intervals**, never point estimates.
- **Per-cell controls that are not blind to either tower.**
- **`swap_obj` is underpowered** at n=245. Its 95% interval is about six points,
  wider than any tower effect measured anywhere in this program. Do not quote it
  as a floor.

Conventions two and three are due to Dipankar Sarkar, who reviewed this program
in public over fourteen rounds and reproduced our numbers from these files alone
at every step.

## Citation

Lancaster, J. B. (2026). *Train Once, Read Everywhere: Substrate Invariance of
the Linearly Readable Structure in Frozen Language Models.*
