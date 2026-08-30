#!/usr/bin/env python3
"""Publish the fitted ChestX-ray14 probe as a model, with the card.

The weights are 344 KB. That is the point worth making: the radiology signal is
already in a general-purpose backbone, and what ships is the reading.

    python scripts/publish_cxr_model.py --probe artifacts/nla/cxr14/cxr14_probe.pt
"""
import argparse
import pathlib

import torch
from huggingface_hub import HfApi, get_token

REPO = "RiverRider/srt-cxr14-linear-probe"
DATA_REPO = "RiverRider/srt-cxr14-frozen-probe"

CARD = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - medical-imaging
  - chest-xray
  - linear-probe
  - frozen-features
base_model: google/gemma-4-31B-it
datasets:
  - RiverRider/srt-cxr14-frozen-probe
model-index:
  - name: srt-cxr14-linear-probe
    results:
      - task:
          type: image-classification
          name: Multi-label thoracic pathology detection
        dataset:
          name: NIH ChestX-ray14 (official test_list.txt)
          type: nih-chestxray14
        metrics:
          - type: auroc
            value: 0.7590
            name: Mean AUROC over 14 findings
---

# A 344 KB linear probe on frozen features, ahead of a fine-tuned ResNet-50

`Linear(5376, 14)` fitted on frozen `google/gemma-4-31B-it` hidden states. The
backbone is untouched: no fine-tuning, no radiology pretraining, no
augmentation. All 112,120 images of NIH ChestX-ray14, official `test_list.txt`.

| ChestX-ray14, official split | mean AUROC | trainable params |
|---|---:|---:|
| Wang et al. 2017 (dataset authors) | 0.7451 | ~25 M, fine-tuned end to end |
| **this probe** | **0.7590** | **5376 x 14 + 14** |
| view-position only | 0.5883 | shortcut baseline |
| shuffled labels | 0.5002 | refit floor |

Ahead on **12 of 14** findings. **14 of 14** clear the view-position baseline.

The whole result refits from the published states in **48 seconds on a MacBook
CPU**. No GPU is needed to reproduce it, only to encode the images once.

## Read the split before comparing

**CheXNet's 0.8414 and Yao's 0.8027 are on a different test set.** CheXNet
section 5: *"We randomly split the dataset into training (70%), validation
(10%), and test (20%) sets."* That is their own random partition, not the
official list.

Which split is harder is not established. Wang scored 0.7381 on a random
partition and 0.7451 on the official one, so for their model the official split
was marginally easier. The only supportable statement is that the two are not
comparable, so exactly one split-matched row appears above.

**Do not compare 0.7590 to 0.8414.**

## Per-finding

| finding | AUROC | 95% CI | shuffled | view-only | n_pos |
|---|---:|---|---:|---:|---:|
| Emphysema | 0.8650 | [0.849, 0.880] | 0.4340 | 0.5740 | 1093 |
| Pneumothorax | 0.8465 | [0.832, 0.859] | 0.5252 | 0.5870 | 2665 |
| Cardiomegaly | 0.8221 | [0.798, 0.844] | 0.5347 | 0.5185 | 1069 |
| Edema | 0.8170 | [0.798, 0.837] | 0.4975 | 0.7011 | 925 |
| Effusion | 0.7849 | [0.773, 0.796] | 0.4740 | 0.5274 | 4658 |
| Hernia | 0.7828 | [0.692, 0.869] | 0.5174 | 0.6564 | 86 |
| Fibrosis | 0.7538 | [0.726, 0.781] | 0.4836 | 0.6310 | 435 |
| Mass | 0.7423 | [0.718, 0.766] | 0.5159 | 0.5502 | 1748 |
| Pleural_Thickening | 0.7347 | [0.714, 0.754] | 0.5031 | 0.5920 | 1143 |
| Atelectasis | 0.7248 | [0.710, 0.738] | 0.4901 | 0.5140 | 3279 |
| Consolidation | 0.7107 | [0.695, 0.727] | 0.5032 | 0.6377 | 1815 |
| Nodule | 0.6956 | [0.674, 0.715] | 0.5137 | 0.5768 | 1623 |
| Infiltration | 0.6862 | [0.674, 0.696] | 0.5048 | 0.6034 | 6112 |
| Pneumonia | 0.6600 | [0.637, 0.686] | 0.5052 | 0.5853 | 555 |

Intervals are a **patient-level cluster bootstrap**. The test split is 25,596
films from 2,797 patients, about 9 each, and those films are not independent.
Resampling images instead of patients gives intervals roughly 1.5x too narrow.

The view-only baseline is **folded** (`max(vw, 1-vw)`). Hernia's raw view-only
AUROC is 0.3436, which is 0.6564 of shortcut once flipped, and reporting the raw
figure would have flattered the probe.

## Usage

The probe expects a **mean-pooled hidden state from the same backbone and
layer**, and the checkpoint carries the train-split `mu` and `sd` it needs.
Applied to any other features, or without that normalisation, the scores mean
nothing.

```python
import torch
from huggingface_hub import hf_hub_download

ck = torch.load(hf_hub_download("RiverRider/srt-cxr14-linear-probe",
                                "cxr14_probe.pt"), weights_only=True)
x = (state - ck["mu"]) / ck["sd"]          # state: (n, 5376) from gemma-4-31B-it
p = torch.sigmoid(x @ ck["W"] + ck["b"])   # (n, 14), order in ck["findings"]
```

Precomputed states for all 112,120 images, the manifest with the official split,
and the fitting script are in
[`RiverRider/srt-cxr14-frozen-probe`](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe).

## Scope, and what this is not

**Detection, not early detection.** These labels describe what is visible in the
image in front of you. Nothing here speaks to catching disease before it is
apparent, which needs longitudinal data with outcomes.

**Not a diagnostic device.** Research artifact. No clinical validation, no
prospective evaluation, no regulatory clearance.

**Labels are NLP-mined** from radiology reports by the dataset authors. Every
model on this benchmark inherits that ceiling.

**One backbone, and the number depends on which.** The probe is deliberately
linear, because anything stronger starts measuring the probe rather than the
representation. The identical probe, split and protocol on other frozen
backbones gives Qwen3-Omni-30B-A3B 0.7650 and Aria 0.7080, so the 0.7590 here is
a gemma-4 result and is labelled as one. Averaging three backbones' logits,
which adds no parameters, reaches 0.7774: see
`RiverRider/srt-cxr14-pooled-probe`.

## Banked negatives

| hypothesis | result |
|---|---|
| Attention-style pooling beats mean for focal findings | **Falsified.** Focal mean falls 0.0537 under max-pool and 0.0225 under top16, at every depth tested |
| Readout depth matters | **No.** 0.7600 to 0.7605 across 0.4 / 0.6 / 0.8 of backbone depth |

Part of the SRT program, <https://github.com/space-bacon/SRT>.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="artifacts/nla/cxr14/cxr14_probe.pt")
    ap.add_argument("--results", default="artifacts/nla/cxr14_probe_full112k.json")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.probe, weights_only=True)
    n = ck["W"].numel() + ck["b"].numel()
    print(f"  probe: {tuple(ck['W'].shape)} + bias = {n:,} trainable params")
    print(f"  mean AUROC recorded in checkpoint: {ck['mean_auroc']}")
    for k in ("mu", "sd", "findings", "d"):
        if k not in ck:
            raise SystemExit(f"checkpoint is missing {k}; card promises it")

    api = HfApi(token=get_token())
    api.create_repo(REPO, repo_type="model", exist_ok=True, private=a.private)

    def up(local, remote):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type="model")
        print(f"  {remote}")

    card = pathlib.Path("/tmp/cxr_model_card.md")
    card.write_text(CARD)
    up(card, "README.md")
    up(a.probe, "cxr14_probe.pt")
    if pathlib.Path(a.results).is_file():
        up(a.results, "eval/cxr14_probe_full112k.json")
    for s in ("scripts/cxr_probe.py", "scripts/get_cxr14.py"):
        if pathlib.Path(s).is_file():
            up(s, f"scripts/{pathlib.Path(s).name}")

    print(f"\nhttps://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
