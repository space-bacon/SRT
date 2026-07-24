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
