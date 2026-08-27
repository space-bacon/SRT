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

Two dtypes of the same head. `sunstone_linear_head_v3_drift.pt` is fp32 at
44.1 MB; `sunstone_linear_head_v3_drift_bf16.pt` is 22.0 MB and is what the
"22 MB head" figure in the papers refers to. The export is measured, not
assumed: on 5,000 L47 pairs the two agree to cosine 0.999997 (minimum
0.999992) and text-to-image R@1 moves from 0.3016 to 0.3018 with median rank
4 unchanged. Receipt in `bf16_head_equivalence.json`, produced by
`scripts/bf16_head_equivalence.py`. Scope: that pool is 5,000 images, well
short of the shipped 123,287 gallery, so those are agreement figures and not
deployment numbers.

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
developed through a public fourteen-round review exchange on the
release thread.

## The map and its reader

Three files here turn the head into an instrument you can stand inside.
It is served live at [lab.sunstonenorth.com](https://lab.sunstonenorth.com),
pane 04.

- `space_map.npz`: a two-dimensional layout of 40,000 photographs sampled
  from the served 123,287-image gallery and positioned by this head's
  1,024-d space.
- `space_map_labels.json`: 24 region names covering those 40,000 points,
  each written by the reader below rather than by us.
- `sunstone_verbalizer.pt`: a ~36M-parameter prefix (16 soft tokens,
  hidden 2048) that conditions a frozen Qwen3-0.6B on one 1,024-d point
  and has it write what is there. The 0.6B has no vision path and never
  sees a photograph. It reads the record the 31B left behind.

Because the reader takes a point rather than a picture, it answers
anywhere on the map, including the open water between clusters where no
photograph exists. The empty gap between the skiing and skateboarding
regions reads as "A man is doing a trick on a snowboard."

Three pools are in play and they are not interchangeable. The marker is
positioned among the **40,000** mapped points, a click averages the ~48
nearest of those, the eight neighbours shown beside the sentence are
retrieved from all **123,287** gallery images, and the evaluation below
scores against the **118,287** train2017 subset.

### Results (`sunstone_verb_eval.json`)

500 held-out points retrieved against the 118,287 train2017 images, so
chance is a median rank of 59,143:

| arm | R@1 | R@10 | median rank |
|---|---:|---:|---:|
| first COCO caption (**the head trained on this**) | 0.212 | 0.554 | 8 |
| a second COCO caption of the same image (unseen) | 0.074 | 0.254 | 45 |
| the reader | 0.042 | 0.194 | 64 |
| another image's point | 0.000 | 0.000 | 55,866 |
| the mean of all points | 0.000 | 0.002 | 59,135 |

**Do not read the first row as a human ceiling.** This head was fitted
on train2017 pairs built from each image's *first* caption, and the
gallery is train2017, so that arm scores a pair the head was trained to
align. Its job is to be a wiring control: it runs first and aborts the
evaluation if human captions cannot retrieve their own images, because a
reader pointed at the wrong space emits fluent sentences over a dead
harness and that failure reads like a result. The honest human reference
is the second caption, at median 45.

Counted per photograph instead of per median, the reader ranks the image
above the first human caption on 17.0% of images, where a second human
beats the first on 20.0%. Both control arms sit at chance, which is what
makes the reader's number mean anything.

### Re-measured with the evaluation images held out of head training

The arm above is a wiring control precisely because this head never held
the evaluation images out of its *own* training. To find what the reader
is actually worth, we refit the head from the same gemma L47 states with
the same 5,000 images held out of head training as well as reader
training, then retrained the reader on the resulting space. Nothing else
changed.

The contamination disappears exactly as predicted. The first caption
falls from median 8 to 57, and the two human references become
indistinguishable, which is what two people describing one photograph
should look like:

| arm | R@1 | median rank |
|---|---:|---:|
| first human caption | 0.100 | 57 |
| second human caption | 0.080 | 48 |
| **the reader** | 0.074 | **46** |
| another photograph's point | 0.000 | 57,494 |
| the mean point | 0.000 | 59,681 |

Counted per photograph against the first human caption, **the reader wins
49.6%** of images (tie 4.2%). A second human wins 48.0% (tie 4.0%), and
that human-to-human arm is symmetric to the decimal, which is the
validity check on the comparison. The reader sits inside the
human-to-human band rather than below a ceiling.

**These are not the deployed numbers.** The Lab serves the shipped head
and its reader, whose figures are the table above. The refit head is a
measurement, not a release.

### Caption coverage in head training

The refit made a second question cheap. This head was fitted on one
caption per image, the first of COCO's five, and a head only ever asked
to match one description of a scene is never required to keep the rest of
it. Refitting with all five, resampled per epoch, changing nothing else:

| head fitted on | unseen caption_0 | unseen caption_1 |
|---|---:|---:|
| one caption per image | median 60 | median 53 |
| all five, resampled per epoch | median **40** | median **34** |

About a third better at placing a caption it has never seen, on a
training loss that is *higher* (1.22 against 0.51). Matching any of five
descriptions is the harder objective and the one that generalises.

**It does not carry through to the reader, and that is the useful part.**
Retraining the reader on each space with the identical recipe:

| arm | cap0 head | all5 head | change |
|---|---:|---:|---:|
| first human caption | 57 | 39 | +32% |
| second human caption | 48 | 28 | +41% |
| the reader | 46 | 40 | **+12%** |

Everyone improves and the reader improves least, so its standing against
a human goes backwards: it beats the first human caption on 49.6% of
images under `cap0` but only 44.8% under `all5`, while human-versus-human
moves 48.0% to 50.4%. At `cap0` the reader is at parity; at `all5` it is
below.

So the head's caption coverage is **not** the reader's bottleneck. The
text-side organisation of the space and the reader's generative quality
are largely separable. The early tell was that reader cross-entropy was
near-identical across both arms at matched steps, ~1.9 to 2.0, while the
head-level retrieval gain was large.

For anyone deploying this: refitting to `all5` materially improves
text-to-image placement and neighbour retrieval, and does not much
improve the sentences a reader writes. Those are separate decisions.

Artifacts for both: `artifacts/nla/verbalizer/caption_coverage/` in
`space-bacon/SRT`, mirrored here under `caption_coverage/`. Scripts:
`scripts/encode_captions_l47.py`, `scripts/refit_sunstone_head.py`,
`scripts/run_caption_coverage_ab.sh`,
`scripts/eval_sunstone_verbalizer.py`.

## What the head reads, and what it does not

Measured directly against COCO annotations rather than against a competing
caption, so no text prior can reach it: per-category detection AUC **0.883**
across the 80 COCO categories, and per-image recovery of the full annotated
object list at **0.543** against a 0.038 chance floor, about 14x. The
worst-ranked of each image's five reference captions still lands at median rank
44 of 5,000, so the vector answers to every description of a scene rather than
to one dominant subject, and retrieval is flat as scenes get busy (r@5 0.906 at
one annotated category, 0.887 at six or more).

The clean characterisation, uniform across every backbone pairing we have
tested: **the linear read-out recovers a scene's inventory nearly in full and
its arrangement far less well.** The stronger version of that line, that
arrangement is not recovered at all, turned out to be a property of the
benchmark's caption formatting rather than of the read-out. See the note below.

Against the one benchmark null that is not blind to the model (real images, the
trained head, image-to-caption pairing deranged over 20 seeds), and on
normalised captions, image identity is worth **+16.3 to +18.6** points of
five-split SugarCrepe macro. Object identity dominates it by roughly three to
one (+34.7 on replace_obj against +10.2 and +9.6 on the two word-order swap
splits), but the swaps are no longer zero. On un-normalised captions the same
margins read +6.5 to +13.0 with the swaps at or just below zero, which is what
this card said before the formatting was measured.

## Compositional variants

Hard-negative training buys compositional accuracy and costs clean retrieval, on
a measured dial. Four settings, K=3 negatives, weight varying:

| weight | replace_obj | 7-split macro | clean i2t R@1 |
|---|---|---|---|
| 0.5 | 0.793 | 0.642 | 0.661 |
| 1.0 | 0.785 | 0.685 | 0.622 |
| 1.5 | 0.766 | 0.682 | 0.586 |
| 2.0 | 0.759 | 0.680 | 0.580 |

Retrieval-first is w=0.5; compositionality-first tops out near macro 0.705.
Four levers were eliminated: mixing negative families dilutes at fixed budget,
pressure saturates, projection width is a null (rank 2,048 and 4,096 reproduce
macro 0.705 exactly), and spatial pooling reallocates rather than lifts.
Compositional training also largely erases a drift-trained head's runtime
nulling, so **v3 and the compositional heads compete for the same 1,024
dimensions**; pick one axis per deployment. The four tower-factorial heads are
published in `RiverRider/srt-nla-gemma4-artifacts` under `q4/`.

## Reading SugarCrepe honestly

Three cautions, all found by external review, all costly to ignore:

1. **The two `add` splits are 99% solvable by caption length alone**, because the
   negative is longer by construction. Report the five length-clean splits.
2. **A blind bigram prior over the benchmark's own captions scores 0.658** on
   those five splits with no image, no head and no encoder. Quote margins over
   that prior, as intervals, not raw accuracy. Strip punctuation when building
   it: on whitespace tokens the trailing period rides inside the final bigram,
   which put our own prior at 0.663 and at exactly 140/245 on `swap_obj`, the
   same score a rule reading only punctuation obtains.
3. **Normalise the candidates before encoding.** The foil is model-generated and
   ends in a period; the human caption often does not, with no counter-example
   in 7,511 pairs. Stripping it moved three of our published conclusions,
   including the sign of the tower comparison.
4. **`swap_obj` is underpowered** at n=245: its 95% interval is about six points,
   wider than any tower effect measured anywhere in this program. Do not quote it
   as a floor.

And a caution about nulls themselves. A mean-image ablation looks like a control
but is reproduced by *another* split's mean image and defeated by a random
direction, so it prices a learned generic-image direction in text space rather
than image content. Its apparent win over real images was the caption
formatting and does not survive normalisation. Candidate position exchange
cannot arbitrate either: it is an identity for an independent-scoring cosine.

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
