#!/usr/bin/env python3
"""Publish the ChestX-ray14 probe: states, manifest, results, and the honest card.

The claim needs a public artifact because it is a split comparison, and split
comparisons are exactly where this literature goes wrong. Anyone should be able
to refit the probe and check the split themselves without renting four GPUs to
re-encode 112,120 radiographs.

Run this on the box that holds the states.

    python scripts/publish_cxr_probe.py --states /root/cxr14_gemma4.npz \
        --manifest /root/cxr14_manifest.json
"""
import argparse
import json
import pathlib

from huggingface_hub import HfApi, get_token

REPO = "RiverRider/srt-cxr14-frozen-probe"
SCRIPTS = ["get_cxr14.py", "cxr_probe.py", "cxr_encode_pool.py",
           "cxr_pool_sweep.py", "cxr_vendor_compare.py"]

CARD = """---
license: apache-2.0
task_categories:
  - image-classification
tags:
  - medical-imaging
  - chest-xray
  - linear-probe
  - frozen-features
  - representation-analysis
---

# Frozen general-purpose features beat a fine-tuned baseline on ChestX-ray14

A linear probe on frozen `google/gemma-4-31B-it` hidden states, on all 112,120
images of NIH ChestX-ray14, using the **official `test_list.txt`**. No
fine-tuning, no radiology pretraining, no augmentation. One `Linear(d, 14)`
under BCE, which is fourteen logistic regressions.

| ChestX-ray14, official split | mean AUROC | method |
|---|---:|---|
| Wang et al. 2017 (dataset authors) | 0.7451 | ResNet-50, fine-tuned end to end |
| **this probe** | **0.7590** | frozen backbone, linear probe |
| view-position only | 0.5883 | shortcut baseline |
| shuffled labels | 0.5002 | refit floor |

Ahead on **12 of 14** findings.

## Read the split before you compare anything

This is the part that matters and the part that is usually wrong.

The widely quoted numbers, **CheXNet 0.8414** and **Yao 2017 0.8027**, are on a
**different split**. CheXNet section 5: *"We randomly split the dataset into
training (70%), validation (10%), and test (20%) sets... We ensure that there is
no patient overlap between the splits."* That is their own random partition,
patient-disjoint but not the official list.

Which split is *harder* is not established, and we are not going to assert it.
Wang et al. scored 0.7381 on a random partition and 0.7451 on the official one,
so for their model the official split was very slightly easier. The honest
statement is only that the two are not comparable, which is why a single
split-matched row is the head-to-head above and the rest is context.

**Do not compare this 0.7590 to 0.8414.** Different test sets.

## Per-finding, split-matched

| finding | this probe | Wang (official split) | delta |
|---|---:|---:|---:|
| Pleural_Thickening | 0.7347 | 0.6835 | +0.0512 |
| Mass | 0.7423 | 0.6933 | +0.0490 |
| Pneumothorax | 0.8465 | 0.7993 | +0.0472 |
| Emphysema | 0.8650 | 0.8330 | +0.0320 |
| Nodule | 0.6956 | 0.6687 | +0.0269 |
| Effusion | 0.7849 | 0.7585 | +0.0264 |
| Infiltration | 0.6862 | 0.6614 | +0.0248 |
| Atelectasis | 0.7248 | 0.7003 | +0.0245 |
| Cardiomegaly | 0.8221 | 0.8100 | +0.0121 |
| Edema | 0.8170 | 0.8052 | +0.0118 |
| Consolidation | 0.7107 | 0.7032 | +0.0075 |
| Pneumonia | 0.6600 | 0.6580 | +0.0020 |
| Fibrosis | 0.7538 | 0.7859 | -0.0321 |
| Hernia | 0.7828 | 0.8717 | -0.0889 |

Hernia has 227 positives in the entire dataset and 86 in the test split, so that
column is thin for everyone and should not carry weight in either direction.

## Controls, and why each one is there

**Shuffled labels (0.5002).** Labels permuted within the training split and the
probe refit. Anything above 0.5 on held-out data is leakage or a bug.

**View position only (0.5883).** Portable AP films are taken of sicker, bedbound
patients, so view alone is a real route to a high AUROC that involves no
pathology. A finding that does not clear this baseline has not been detected.
The baseline is **folded** (`max(vw, 1-vw)`): Hernia's raw view-only AUROC is
0.3033, which is 0.6967 of shortcut once flipped, and reporting the raw figure
would have flattered the probe.

View position is a single binary feature, so every one of its AUROC comparisons
is a tie. Scoring it needs rank averaging within tied groups, or the answer
becomes an artefact of the sort order and moves between machines. An earlier
release of this card said 0.5896 for that reason.

**Patient-level cluster bootstrap.** Confidence intervals resample *patients*,
not images. The test split is 25,596 films from 2,797 patients, roughly 9 per
patient, and those films are anything but independent. Resampling rows treats
correlated images as fresh evidence and yields intervals about 1.5x too narrow.

**Patient overlap is asserted to be zero** and the script refuses to run
otherwise.

## Scope

**Detection, not early detection.** These labels describe what is visible in
the image in front of you. Nothing here speaks to catching disease before it is
apparent; that needs longitudinal data with outcomes.

**One backbone, one probe class, and the number depends on the backbone.** The
probe is deliberately linear because anything stronger measures the probe rather
than the representation. The identical probe and split on other frozen backbones
gives Qwen3-Omni-30B-A3B 0.7650 and Aria 0.7080, so the 0.7590 here is a gemma-4
result. Averaging three backbones' logits reaches 0.7774 at no added parameters:
see `RiverRider/srt-cxr14-pooled-probe`.

**Labels are NLP-mined** from radiology reports by the dataset authors, with
their own reported precision and recall. Every model on this dataset inherits
that ceiling.

## Banked negatives

Kept because they bound the claim.

| hypothesis | result |
|---|---|
| Attention-style pooling beats mean for focal findings | **Falsified.** Focal mean drops 0.0537 under max-pool and 0.0225 under top16-pool, at every depth tested |
| Readout depth matters | **No.** 0.7600 to 0.7605 across 0.4 / 0.6 / 0.8 of backbone depth |

Nine variants (3 depths x 3 poolings) are in `results/cxr14_pool_sweep.json`.

## Contents

| path | what |
|---|---|
| `states/cxr14_gemma4.npz` | frozen pooled hidden states, all 112,120 images |
| `manifests/cxr14_manifest.json` | rows with `key`, `labels`, `patient_id`, `view`, official `split` |
| `results/cxr14_probe_full112k.json` | per-finding AUROC, CIs, both controls, split-matched comparison |
| `results/cxr14_pool_sweep.json` | the nine pooling / depth variants |
| `scripts/` | dataset fetch, encoder, probe, sweep |

## Reproducing

```bash
python scripts/get_cxr14.py --shards 12
python scripts/cxr_probe.py --states states/cxr14_gemma4.npz \\
    --manifest manifests/cxr14_manifest.json --out probe.json
```

Align on the manifest `key`, never on array position: any row the encoder
dropped shifts every later row.

Part of the SRT program, <https://github.com/space-bacon/SRT>.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="/root/cxr14_gemma4.npz")
    ap.add_argument("--manifest", default="/root/cxr14_manifest.json")
    ap.add_argument("--results", nargs="*",
                    default=["artifacts/nla/cxr14_probe_full112k.json",
                             "artifacts/nla/cxr14_pool_sweep.json"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-states", action="store_true",
                    help="card and results only, for a fast card-only update")
    a = ap.parse_args()

    api = HfApi(token=get_token())
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=a.private)

    def up(local, remote):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset")
        print(f"  {remote}")

    card = pathlib.Path("/tmp/cxr_card.md")
    card.write_text(CARD)
    up(card, "README.md")

    for r in a.results:
        p = pathlib.Path(r)
        if p.is_file():
            up(p, f"results/{p.name}")
        else:
            print(f"  MISSING {p}")

    man = pathlib.Path(a.manifest)
    if man.is_file():
        # Sanity-check the split before publishing a split claim.
        rows = json.load(open(man))["rows"]
        tr = {r["patient_id"] for r in rows if r["split"] == "train"}
        te = {r["patient_id"] for r in rows if r["split"] == "test"}
        if tr & te:
            raise SystemExit(f"{len(tr & te)} patients on both sides; refusing")
        print(f"  split check ok: {len(rows)} rows, 0 patient overlap")
        up(man, f"manifests/{man.name}")

    for n in SCRIPTS:
        p = pathlib.Path("scripts") / n
        if p.is_file():
            up(p, f"scripts/{n}")

    st = pathlib.Path(a.states)
    if not a.skip_states and st.is_file():
        print(f"  uploading states ({st.stat().st_size / 1e9:.1f} GB)")
        up(st, f"states/{st.name}")

    print(f"\nhttps://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
