#!/usr/bin/env python3
"""Publish the omni / cross-vendor states, fitted towers and results to the Hub.

These states cost four Blackwell GPUs and four 24-31B hosts to produce, so the
point of publishing is that the fits and the retrieval analysis can be rerun
against them without any of that hardware.
"""
import pathlib

from huggingface_hub import HfApi, get_token

REPO = "RiverRider/srt-omni-crossvendor-states"
SRC = pathlib.Path("artifacts/nla/omni")
SCRIPTS = ["build_omni_manifest.py", "omni_encode_all.py", "omni_joint_fit.py",
           "xvendor_fit.py", "xvendor_fit_n.py", "omni_smoke.py"]

CARD = """---
license: apache-2.0
task_categories:
  - text-retrieval
  - image-to-text
tags:
  - multimodal
  - retrieval
  - representation-analysis
  - cross-model
---

# SRT omni / cross-vendor states

Frozen hidden states for the same 7,000-item manifest encoded through four
multimodal hosts from four vendors, plus the fitted retrieval towers and the
results computed from them.

A gallery encoded by one vendor is searchable by another vendor's text encoder
at a rate statistically indistinguishable from native, and one linear map places
image, audio and video in a single searchable space more effectively than one
map per modality.

## The two results

**One tower beats one-per-modality** (Qwen3-Omni-30B, 1,376 holdout items):

| | shared | per-modality | n |
|---|---|---|---|
| mixed gallery | **0.2885** | 0.2667 | 1376 |
| image | **0.2902** | 0.2687 | 1027 |
| audio | **0.4451** | 0.4207 | 164 |
| video | **0.2486** | 0.1730 | 185 |

Derangement floor 695 +/- 21 against an analytic 688.

**Cross-vendor retrieval matches within-vendor:**

| | 2 vendors | 4 vendors |
|---|---|---|
| modalities | image, video | image |
| holdout | 1200 | 1000 |
| cross / within r@1 | 0.2537 / 0.2621 | 0.2862 / 0.2898 |
| retention | 0.968, 95% CI [0.908, 1.030] | **0.988, 95% CI [0.955, 1.023]** |
| floors (analytic) | 593-601 (600) | 495-504 (500) |

Both intervals contain 1.0. The claim is indistinguishability, not a specific
retained fraction. Four of the twelve cross directions score above the
within-vendor baseline of the gallery they search.

## What retention does and does not tell you (added after publication)

**The ratio above is close to insensitive to the vendor boundary it is named
after, and later work established why.** Both of its terms are limited by the
caption head, not by the difference between vendors:

| | r@1 |
|---|---:|
| cross-vendor image-to-image agreement, direct ridge map | **0.8024** |
| within-vendor text-to-image retrieval | 0.1050 |
| shuffled floor | 0.0007 |

The encoders agree with each other about pictures roughly eight times better
than any one of them connects its own captions to its own pictures. A ratio of
two caption-head-limited numbers cannot report much about the encoders.

Tested directly by swapping the head. Replacing the caption tower with an
unrelated `all-MiniLM-L6-v2` on the radiology gallery moves both terms by
nearly the same factor: within scales 0.938, cross scales 0.973, gap 0.0347.
That is what a shared bottleneck predicts. Test proposed by Dipankar Sarkar.

**Use the 0.8024 image-agreement number for portability claims.** It is
measured without a shared space, against a floor of 0.0007, and it is not
throttled by the caption head.

## A probe trained on one backbone reads another (added 2026-08-29)

Retrieval says two vendors agree about *which picture*. It does not say they
agree about *what is in it*. On satellite imagery, a linear scene probe fitted
on one vendor's frozen states and read on a different vendor's states through a
ridge map fitted on train rows only, 17 land-use classes, 12 cross directions:

| | mean AUROC |
|---|---:|
| native, each vendor probing itself | 0.9507 |
| self-map control | 0.9517 |
| **transported across backbones** | **0.9484** |
| shuffled floor | 0.5014 |

Transport costs **0.0024**, and 4 of the 12 transported pairs beat the native
target. **The labels are weak** and the number should not be quoted without
that: scene classes are keyword-matched from the RSICD captions, the same shape
of supervision ChestX-ray14 uses for its findings, and they are coarser than
the medical case. The probe reads the image tower while the label comes from
the caption, so the text side cannot leak the answer.

## One frame for four backbones, and reading them together

Everything above is bilateral: one ridge map per ordered pair, twelve maps for
four vendors. Fitting a single shared frame from all four at once (MAXVAR
generalised CCA) gives eight maps instead of twelve and costs little:

| | satellite | radiology |
|---|---:|---:|
| direct, 12 pairwise maps | 0.8425 | 0.8817 |
| **via one shared frame, 8 maps** | **0.8164** | **0.8400** |

**Three vendors read together beat the best single vendor, on all four targets
in both domains, eight for eight.** Satellite +0.0345 mean (mistral 0.8310 to
0.8810), radiology +0.0165 (qwen3omni 0.9670 to 0.9940). The backbones are not
redundant; each carries something the others do not.

Routing every hop through the shared frame also flattens the iterated-loop
degradation to 1.0000 at 36 hops. Read that with the caveat recorded in the
artifact: the composed map collapses to exactly the joint width, so later hops
cannot lose dimensions that are already gone. Rank-matched pairwise maps rule
out the trivial bottleneck reading, but the result may only restate that the
joint route reuses one subspace per hop.

## Read this before using the states

**Centering is not optional.** Raw cosine between unrelated items on these
states is +0.869 and raw retrieval sits exactly at chance. Every number above is
per-modality centered, and the fitted towers ship with their centering vectors.
Skip the centering and you will measure the anisotropy.

**Pool content tokens only.** These states pool positions where the input id
equals the modality's content token. Averaging every position lets the shared
chat prompt dominate: unrelated items then sit at cosine 0.9987 and you are
measuring the template.

## Scope

- The four-vendor result is **image only**. Two of the four hosts carry no audio
  or video tower, so audio evidence is single-vendor and video is two-vendor.
- **Audio is the thinnest leg despite the highest score**: 164 holdout items,
  and 12.6% of the source clips downloaded empty because AudioCaps is
  YouTube-sourced. The surviving set is "clips still available", not a random
  sample.
- Each configuration is **one fit on one split**. Refitting moved the two-vendor
  retention from 0.960 to 0.968, so fit-to-fit variation is real; the bootstrap
  interval covers sampling only.
- All four hosts are trained on **overlapping web-scale corpora**. Convergence
  shows the structure survives a change of vendor, architecture and training
  run. The stronger reading, that independent minds would converge on it,
  remains untested and is not asserted here.

## Contents

| path | what |
|---|---|
| `states/{vendor}_states_s*.npz` | per-shard `item`, `text`, `modality`, `ok` |
| `manifests/` | shard manifests; rows carry a stable `key` |
| `maps/omni_shared_map.pt` | shared tower, single backbone |
| `maps/xvendor_map.pt`, `maps/xvendor4_map.pt` | cross-vendor towers |
| `results/*.json` | the numbers above, with floors and CIs |
| `scripts/` | manifest build, encode, and both fitters |

Vendors: `qwen3omni` (Qwen3-Omni-30B-A3B), `gemma4` (gemma-4-31B-it),
`mistral` (Mistral-Small-3.1-24B), `aria` (rhymes-ai/Aria).

## Reproducing

Align on the manifest `key`, never on array position. The vendors were sharded
differently and any row one encoder dropped shifts every later row, so
positional alignment is silently wrong the moment failure counts differ.

```bash
python scripts/xvendor_fit_n.py \\
  --vendor "qwen3omni:states/omni_states_s*.npz:manifests/omni_manifest_s*.json" \\
  --vendor "gemma4:states/gemma4_states_s*.npz:manifests/xv_manifest_s*.json" \\
  --vendor "mistral:states/mistral_states_s*.npz:manifests/img_s*.json" \\
  --vendor "aria:states/aria_states_s*.npz:manifests/img_s*.json" \\
  --out xvendor4.json
```

Sources: COCO val2017 images, AudioCaps clips, MSR-VTT videos.
Written up in `paper_nla.md` section 11.9.
"""


def main():
    api = HfApi(token=get_token())
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)

    def up(local, remote):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset")
        print(f"  {remote}")

    card = pathlib.Path("/tmp/omni_card.md")
    card.write_text(CARD)
    up(card, "README.md")

    for p in sorted(SRC.glob("*_states_s*.npz")):
        up(p, f"states/{p.name}")
    for p in sorted(SRC.glob("*.pt")):
        up(p, f"maps/{p.name}")
    for n in ["omni_joint.json", "xvendor.json", "xvendor4.json"]:
        up(SRC / n, f"results/{n}")
    for p in sorted((SRC / "manifests").glob("*.json")):
        up(p, f"manifests/{p.name}")
    for n in ["omni_manifest.json", "img_s0.json", "img_s1.json"]:
        up(SRC / n, f"manifests/{n}")
    for n in SCRIPTS:
        up(pathlib.Path("scripts") / n, f"scripts/{n}")

    print(f"\nhttps://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
