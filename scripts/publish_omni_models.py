#!/usr/bin/env python3
"""Publish the omni towers as model repos and the manifest as a benchmark.

The states dataset already carries the raw evidence. These add the two things it
does not: the fitted towers as usable models, and a manifest others can encode
with their own host to add a fifth vendor.
"""
import json
import pathlib

from huggingface_hub import HfApi, get_token

SRC = pathlib.Path("artifacts/nla/omni")
STATES = "RiverRider/srt-omni-crossvendor-states"

SHARED_CARD = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - multimodal
  - retrieval
  - cross-modal
---

# SRT omni shared tower

One linear tower that places image, audio and video beside text in a single
searchable space, read off the frozen hidden states of Qwen3-Omni-30B-A3B at 60%
depth. 8 MB.

One shared tower beats one tower per modality on every modality (1,376 holdout
items):

| | shared | per-modality | n |
|---|---|---|---|
| mixed gallery | **0.2885** | 0.2667 | 1376 |
| image | **0.2902** | 0.2687 | 1027 |
| audio | **0.4451** | 0.4207 | 164 |
| video | **0.2486** | 0.1730 | 185 |

Derangement floor 695 +/- 21 against an analytic 688. The widest margin is video
(+0.076), the modality with the fewest items, which is what a shared tower
borrowing structure from the larger modalities looks like.

## Contents

`omni_shared_map.pt` holds `Wi` (one 2048 -> 512 item tower, key `all`), `Wt`
(the matching text tower), `mu_img` (a separate centering vector per modality)
and `mu_txt`.

```python
import torch
m = torch.load("omni_shared_map.pt", map_location="cpu", weights_only=False)
Wi, Wt = m["Wi"]["all.weight"], m["Wt"]["weight"]
v = (item_state - m["mu_img"]["audio"]) @ Wi.T + m["Wi"]["all.bias"]
q = (text_state - m["mu_txt"])      @ Wt.T + m["Wt"]["bias"]
# cosine between normalised v and q
```

## Two things that will silently break this

**Centering is not optional.** Raw cosine between unrelated items on these states
is +0.869 and raw retrieval sits exactly at chance. Subtract the modality's `mu`
before projecting. The vectors ship with the tower for that reason.

**Pool content tokens only.** These states pool positions where the input id
equals the modality's content token. Average every position instead and the
shared chat prompt dominates: unrelated items land at cosine 0.9987 and you are
measuring the template.

## Scope

Fitted on one backbone, one manifest of 5,000 COCO images, 1,000 AudioCaps clips
and 1,000 MSR-VTT videos, one split. Audio carries the highest score on the
thinnest evidence: 164 holdout items, from a source where 12.6% of clips
downloaded empty because AudioCaps is YouTube-sourced, so the surviving set is
"clips still available" rather than a random sample.

States, scripts and results: [`{states}`](https://huggingface.co/datasets/{states}).
Written up in `paper_nla.md` section 11.9.
""".replace("{states}", STATES)

XV_CARD = """---
license: apache-2.0
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - multimodal
  - retrieval
  - cross-model
  - model-comparison
---

# SRT cross-vendor towers

Towers that put four multimodal hosts from four vendors into one retrieval
space, so a gallery encoded by one model is searchable by another model's text
encoder.

Cross-vendor retrieval is statistically indistinguishable from within-vendor
retrieval:

| | 2 vendors | 4 vendors |
|---|---|---|
| modalities | image, video | image |
| holdout | 1200 | 1000 |
| cross / within r@1 | 0.2537 / 0.2621 | 0.2862 / 0.2898 |
| retention | 0.968, 95% CI [0.908, 1.030] | **0.988, 95% CI [0.955, 1.023]** |
| floors (analytic) | 593-601 (600) | 495-504 (500) |

Both intervals contain 1.0, so the claim is indistinguishability rather than a
specific retained fraction. Four of the twelve cross directions score above the
within-vendor baseline of the gallery they search.

## Contents

| file | vendors | dim |
|---|---|---|
| `xvendor_map.pt` | qwen3omni, gemma4 | 512 |
| `xvendor4_map.pt` | qwen3omni, gemma4, mistral, aria | 512 |

Each holds `Wi` and `Wt` keyed by vendor, plus `mu` (per vendor, per modality)
and `mu_txt` (per vendor).

```python
import torch
m = torch.load("xvendor4_map.pt", map_location="cpu", weights_only=False)
gal = (img_gemma4 - m["mu"]["gemma4"]["image"]) @ m["Wi"]["gemma4.weight"].T \\
      + m["Wi"]["gemma4.bias"]
qry = (txt_aria   - m["mu_txt"]["aria"])        @ m["Wt"]["aria.weight"].T \\
      + m["Wt"]["aria.bias"]
```

**Every vendor needs its own tower.** Hidden sizes differ (2048 for qwen3omni,
5376 for gemma4, 2560 for aria), so a tower fitted on one vendor's text cannot be
applied to another's. The towers are trained on all vendor pairs at once, or the
result is two spaces sitting side by side rather than one shared space.

**Centering is not optional**, for the same reason as above: raw cosine on these
states is +0.869 and raw retrieval sits at chance.

## Scope

The four-vendor result covers **images only**. Two of the four hosts carry no
audio or video tower, so audio evidence is single-vendor and video is
two-vendor. Each configuration is one fit on one split; refitting moved the
two-vendor retention from 0.960 to 0.968, so fit-to-fit variation is real and
the bootstrap interval covers sampling only.

All four hosts are trained on overlapping web-scale corpora. Convergence shows
the readable structure survives a change of vendor, architecture and training
run. The stronger reading, that independent minds would converge on it, remains
untested and is not asserted here. Separating the two needs a host whose
training corpus does not overlap the others, which the current model ecosystem
does not offer.

Hosts: Qwen3-Omni-30B-A3B-Instruct, gemma-4-31B-it,
Mistral-Small-3.1-24B-Instruct-2503, rhymes-ai/Aria.

States, scripts and results: [`{states}`](https://huggingface.co/datasets/{states}).
Written up in `paper_nla.md` section 11.9.
""".replace("{states}", STATES)

BENCH_CARD = """---
license: apache-2.0
task_categories:
  - text-retrieval
tags:
  - multimodal
  - benchmark
  - cross-model
---

# SRT omni cross-vendor manifest

The item list behind the cross-vendor retrieval result, published on its own so
that a fifth vendor can be added without re-deriving the setup.

7,000 rows, each `{modality, key, caption}`:

| modality | n | source | `key` |
|---|---|---|---|
| image | 5000 | COCO val2017 | file name |
| audio | 1000 | AudioCaps | `{audiocap_id}.wav` |
| video | 1000 | MSR-VTT | video id |

## Adding your own model

Encode every row through your host, pool the content token positions, and fit
against the published states. `key` is the join column; align on it rather than
on array position, because any row your encoder drops shifts every later row and
positional alignment then pairs the wrong item with the wrong caption.

```bash
python omni_encode_all.py --model YOUR/MODEL --manifest omni_manifest.json \\
    --out yourvendor_states_s0.npz
python xvendor_fit_n.py \\
  --vendor "gemma4:gemma4_states_s*.npz:xv_manifest_s*.json" \\
  --vendor "yourvendor:yourvendor_states_s*.npz:omni_manifest.json"
```

Report `retention` with its bootstrap interval, and the derangement floor beside
it. Raw cosine on hidden states of this kind runs around +0.869 with retrieval at
chance, so an uncentered number is not interpretable.

## What the existing vendors reached

Four hosts from four vendors, cross-vendor r@1 0.2862 against 0.2898
within-vendor, retention 0.988 with a 95% interval of [0.955, 1.023]. The
interval contains 1.0.

## Caveats carried by the sources

12.6% of the AudioCaps clips downloaded empty because the source is
YouTube-hosted, so the usable audio set is "clips still available" rather than a
random sample. Two of the four published vendors carry no audio or video tower,
which is why the four-vendor comparison is image-only.

States and fitted towers: [`{states}`](https://huggingface.co/datasets/{states}).
""".replace("{states}", STATES)


def main():
    api = HfApi(token=get_token())

    def push(repo, kind, files, card):
        api.create_repo(repo, repo_type=kind, exist_ok=True)
        p = pathlib.Path("/tmp/_card.md")
        p.write_text(card)
        api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md",
                        repo_id=repo, repo_type=kind)
        for local, remote in files:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=repo, repo_type=kind)
        print(f"  {kind:8s} https://huggingface.co/"
              f"{'datasets/' if kind == 'dataset' else ''}{repo}")

    push("RiverRider/srt-omni-shared-tower", "model",
         [(SRC / "omni_shared_map.pt", "omni_shared_map.pt"),
          (SRC / "omni_joint.json", "results.json")], SHARED_CARD)

    push("RiverRider/srt-omni-xvendor-towers", "model",
         [(SRC / "xvendor_map.pt", "xvendor_map.pt"),
          (SRC / "xvendor4_map.pt", "xvendor4_map.pt"),
          (SRC / "xvendor.json", "results_2vendor.json"),
          (SRC / "xvendor4.json", "results_4vendor.json")], XV_CARD)

    push("RiverRider/srt-omni-manifest", "dataset",
         [(SRC / "omni_manifest.json", "omni_manifest.json"),
          (pathlib.Path("scripts/omni_encode_all.py"), "omni_encode_all.py"),
          (pathlib.Path("scripts/xvendor_fit_n.py"), "xvendor_fit_n.py"),
          (pathlib.Path("scripts/build_omni_manifest.py"), "build_omni_manifest.py")],
         BENCH_CARD)


if __name__ == "__main__":
    main()
