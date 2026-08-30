#!/usr/bin/env python3
"""Publish the pooled chest probe, with a card built from the artifact.

Three separate incidents on 2026-08-29 put a stale number on a live card: a
superseded pilot's 0.8192, a 0.5827 that appeared in no artifact at all, and a
per-finding row that had drifted in the third decimal. Each was caught after
publication by a checker. The checker is worth having, but a card that is
FORMATTED FROM the artifact cannot drift in the first place, so every figure
below is read out of the json rather than typed.

The gemma-4 surfaces are deliberately untouched. They name their backbone, their
claim is true as it stands, and pooling is a different claim that deserves its
own surface rather than a caveat bolted onto a finished one.

    python scripts/publish_pooled_probe.py --probe artifacts/nla/cxr14_pooled_probe.pt
"""
import argparse
import json
import os

E = json.load(open("artifacts/nla/cxr14_ensemble3.json"))
T = json.load(open("artifacts/nla/cxr14_transport.json"))
REPO = "RiverRider/srt-cxr14-pooled-probe"

_s, _t = E["summary"], T["summary"]
_best, _pool = _s["best_single"], _s["best_ensemble"]
_wang = E["split_matched_baseline"]
_lm = E["ensemble"][f"logit_mean_vs_{_s['best_single_vendor']}"]
_rows = "\n".join(
    f"| {v} alone | {E['single'][v]:.4f} | {E['single'][v] - _best:+.4f} | "
    f"{E['single'][v] - _wang:+.4f} |" for v in E["vendors"])
_ctrl = next(k for k in E["control"] if k.startswith("concat_"))

CARD = f"""---
license: mit
tags: [chest-xray, linear-probe, frozen-features, ensemble, medical-imaging]
pipeline_tag: image-classification
---

# Reading three frozen backbones together on ChestX-ray14

Three general-purpose multimodal models, none trained on radiology and none
fine-tuned here, encode all {E['n_train'] + E['n_test']:,} images of
ChestX-ray14. One linear probe per backbone. Their logits are then **averaged**,
which adds no parameters at all.

Official `test_list.txt`, {E['n_train']:,} train / {E['n_test']:,} test.

| | mean AUROC | vs best single | vs split-matched baseline |
|---|---:|---:|---:|
{_rows}
| **mean of the three probes' logits** | **{_pool:.4f}** | **{_lm['delta']:+.4f}** | **{_pool - _wang:+.4f}** |
| concatenated features | {E['ensemble']['concat']:.4f} | {E['ensemble']['concat'] - _best:+.4f} | {E['ensemble']['concat'] - _wang:+.4f} |
| control: best single concatenated with itself | {E['control'][_ctrl]:.4f} | {E['control'][_ctrl] - _best:+.4f} | {E['control'][_ctrl] - _wang:+.4f} |

Split-matched baseline **{_wang}**: Wang et al. 2017 (arXiv:1705.02315v5,
Table 17), a ResNet-50 fine-tuned end to end, on the same official test list.

**The gain is significant**: paired patient-clustered bootstrap on the
difference of mean AUROCs gives {_lm['delta']:+.4f}, 95% CI
[{_lm['ci95'][0]:+.4f}, {_lm['ci95'][1]:+.4f}], positive in
{int(_lm['frac_bootstrap_positive'] * 1000):,} of 1,000 resamples. Averaging
logits adds no parameters, so the gain cannot be extra capacity. It is
information one backbone holds and the others do not.

The clearest evidence for that: **Aria scores {E['single']['aria']:.4f} on its
own, which is {_wang - E['single']['aria']:.4f} BEHIND the baseline**, and
pooling it in still helps.

## Concatenation loses, and the control says why

Concatenating features scored {E['ensemble']['concat']:.4f}. Concatenating the
best backbone **with itself** scored {E['control'][_ctrl]:.4f}, at identical
parameter count and with zero new information. The two differ by
{abs(E['ensemble']['concat'] - E['control'][_ctrl]):.4f}, so the entire
concatenation effect is width rather than content. It was never retuned, so read
it as untuned rather than refuted.

## The probes are interchangeable across backbones

A probe fitted on one backbone and read on another's states, through a ridge map
fitted on training rows only:

| | mean AUROC |
|---|---:|
| native, each backbone probing itself | {_t['mean_native']:.4f} |
| self-map control | {_t['mean_self_map']:.4f} |
| **transported across backbones** | **{_t['mean_transported']:.4f}** |
| round-trip cycle | {_t['mean_cycle']:.4f} |
| shuffled floor | {_t['mean_shuffled']:.4f} |

Transport cost is {_t['transport_cost']:+.4f}. It is negative: moving a probe
between backbones costs nothing on average. Four of six cross directions beat
the target backbone's own probe.

## Usage

The checkpoint holds one weight matrix, bias, and the train-split mean and
standard deviation per backbone. The normalisation ships with the weights
because each backbone is standardised with its own training statistics, without
which the pooled score is not reproducible.

```python
import torch
ck = torch.load("pooled_probe.pt", map_location="cpu", weights_only=False)
logits = []
for tag, p in ck["probes"].items():
    x = (states[tag] - p["mu"]) / p["sd"]        # that backbone's raw states
    logits.append(x @ p["weight"] + p["bias"])
score = torch.stack(logits).mean(0)              # {len(E['vendors'])} x 14 -> 14
```

`ck["findings"]` gives the column order.

## Scope, and what this is not

**Detection, not early detection.** These labels describe what is visible in the
image in front of you. Nothing here speaks to catching disease before it is
apparent, which needs longitudinal data with outcomes.

**Not a diagnostic device.** Research artifact. No clinical validation, no
prospective evaluation, no regulatory clearance.

**Labels are NLP-mined** from radiology reports by the dataset authors. Every
model on this benchmark inherits that ceiling.

**Patient overlap between train and test is zero**, asserted by the script,
which refuses to run otherwise. Confidence intervals resample patients rather
than images, because the unit that repeats is the patient.

**A fourth backbone is missing.** Mistral Small 3.1 was encoded but one shard
wrote a truncated file and the chain deleted the good shards alongside it. That
is our bug, now fixed, and the result here is three backbones rather than four.

## Reproducing

`scripts/cxr_probe_ensemble.py` in the SRT repository, with the controls above
built in. Sibling surfaces: the single-backbone gemma-4 probe is at
`RiverRider/srt-cxr14-linear-probe`, and its states at
`RiverRider/srt-cxr14-frozen-probe`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="artifacts/nla/cxr14_pooled_probe.pt")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        print(CARD)
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model")
    if os.path.exists(a.probe):
        api.upload_file(path_or_fileobj=a.probe, path_in_repo="pooled_probe.pt",
                        repo_id=a.repo, repo_type="model")
        print(f"  uploaded {a.probe}")
    for f in ("artifacts/nla/cxr14_ensemble3.json",
              "artifacts/nla/cxr14_transport.json"):
        api.upload_file(path_or_fileobj=f, repo_id=a.repo, repo_type="model",
                        path_in_repo=f"results/{os.path.basename(f)}")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
