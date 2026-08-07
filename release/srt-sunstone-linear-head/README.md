---
license: apache-2.0
base_model: google/gemma-4-31B-it
tags:
- cross-modal
- retrieval
- srt
- frozen-backbone
---

# SRT-Sunstone Linear Head

**Train once, read everywhere.** This head was trained once on
datacenter bf16 states and reads the same structure unchanged across a
10× host-scale reduction, 4-bit quantization (−0.01 R@1), and a change
of silicon (CUDA → Apple-Silicon MLX: 97 % head-space text agreement
against a 99.96 % same-runtime ceiling, and on-device image→text
retrieval within 3 R@1 points of the datacenter reference). One
artifact, every deployment tier from edge to fleet.

A single trained linear projection pair that turns frozen
**gemma-4-31B-it** into an image↔text retrieval engine. No backbone
weights are touched; the head reads layer-47 hidden states from the
forward pass you are already running.

| method | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 |
|---|---:|---:|---:|---:|
| centered cosine (zero training) | 0.288 | 0.523 | 0.648 | 0.173 |
| **this head** (117k COCO pairs, InfoNCE) | **0.661** | **0.911** | **0.967** | **0.506** |

Protocol: 1,000 held-out COCO val2017 images vs their 5,000 captions
(chance R@1 ≈ 0.001); training pairs from train2017 (cross-partition).
An MLP trained on the same data never beats this linear map at any
training size, and the gap widens with data: the cross-modal
correspondence in this substrate is linear (paper §11.6.4,
`space-bacon/SRT`).

## Contents

`sunstone_linear_head.pt` (`weights_only=True`-safe):

- `img`, `txt` — state dicts for two `nn.Linear(5376, 1024)` heads
- `mu_img`, `mu_txt` — per-modality centering means (mandatory; raw
  cosine is dominated by the anisotropy attractor)
- `meta` — training config + eval numbers

`sunstone_linear_head_v2_jitter30.pt` — same format, same protocol,
trained with mean-jitter augmentation (each batch perturbed by a
random constant displacement per modality, norm ~U(0, 30), covering
the ~22-unit mean displacement a runtime change actually produces).

`sunstone_linear_head_v3_drift.pt` — same format, trained with
**measured drift-family augmentation**: each training sample is
perturbed by a real per-vector cross-runtime residual (MLX-Q4 minus
CUDA-bf16, 3,000 measured pairs, scaled ~U(0, 1.5)). On held-out
cross-runtime data this internalizes the runtime drift: **no-recal
i2t R@1 0.636** (v1 needs the recalibration to reach 0.634), t2i
0.469 vs v1's 0.424 ceiling, and clean same-runtime performance
*improves* (i2t 0.679 vs 0.661). The residual augmentation acts as a
regularizer.

**Which to use.** The three heads form a ladder of target-runtime
knowledge:

| you have | use | cross-runtime behavior |
|---|---|---|
| nothing | v3 | i2t 0.636 / t2i 0.469 with zero calibration |
| ~200 unpaired states from your runtime (42KB mean recal) | v3 + recal | i2t 0.658, within 0.2 points of same-runtime |
| generic caution, no target runtime known | v2 | degrades most gracefully across arbitrary runtimes |
| exact v1 reproduction | v1 + recal | the paper's reference numbers |

The per-runtime mean recalibration (average ~200 states from your own
runtime, use instead of the shipped `mu_*`) is still recommended with
every head; with v1 it is load-bearing (skipping it costs ~24 i2t R@1
points, a failure mode invisible to same-transform agreement metrics
and only visible on end-task retrieval). The drift analysis behind all
of this is in `artifacts/nla/q4/*_2026080[56].json` in the repo,
developed through a public five-round review exchange on the release
thread.

## Usage

```python
import torch, torch.nn.functional as F

ck = torch.load("sunstone_linear_head.pt", weights_only=True)
head_i = torch.nn.Linear(5376, 1024); head_i.load_state_dict(ck["img"])
head_t = torch.nn.Linear(5376, 1024); head_t.load_state_dict(ck["txt"])

# h_img: mean L47 state over image soft-token positions
# h_txt: BOS-prefixed last-token L47 state
zi = F.normalize(head_i(h_img - ck["mu_img"]), dim=-1)
zt = F.normalize(head_t(h_txt - ck["mu_txt"]), dim=-1)
score = zi @ zt.T
```

## Calibration notes (read before deploying)

1. Center per-modality, always.
2. Anchor domain match beats anchor size: recalibrate `mu_img` on
   ~150+ images from your deployment's own query domain.
3. Prepend BOS on every text encode (gemma-4 is BOS-sensitive).
4. Retrieval readout needs a single prefill pass, not generation.

Training recipe, deployment tiers, and reproduction scripts:
`docs/CROSSMODAL_LINEAR_HEAD.md` in
[space-bacon/SRT](https://github.com/space-bacon/SRT). Pair encodings
for retraining: `RiverRider/srt-nla-gemma4-artifacts` (`procrustes/`).
Trained on COCO captions (CC-BY 4.0).
