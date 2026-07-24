# The Cross-Modal Linear Head — architecture, deployment tiers, and recipe

**Status:** validated 2026-07-24 on gemma-4-31B-it L47 (paper_nla.md §11.6.4).
**Artifacts:** `artifacts/nla/gemma4/procrustes/`, tensors on HF
[`RiverRider/srt-nla-gemma4-artifacts`](https://huggingface.co/RiverRider/srt-nla-gemma4-artifacts) under `procrustes/`.

This document is the engineering companion to the academic result. The
academic finding is: **after per-modality centering, the image↔text
modality gap inside a frozen multimodal LLM is anisotropic-linear.** A
rotation cannot express it (orthogonal Procrustes underperforms doing
nothing), a two-layer MLP finds nothing beyond it at any training size
tested, and a single trained linear map captures it. The engineering
consequence is that the cheapest possible cross-modal adapter is also
the correct one, and that fact is proven rather than assumed.

## The capability ladder (measured)

Protocol: 1,000 held-out COCO val2017 images vs their 5,000 captions
(hit = any of the image's 5 captions), chance R@1 ≈ 0.001. Training
pairs from COCO train2017 (cross-partition, no eval leakage). Image
state = mean L47 hidden state over image soft-token positions; caption
state = BOS-prefixed last-token L47.

| tier | added components | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 |
|---|---|---:|---:|---:|---:|
| raw cosine | nothing | (unusable: anisotropy attractor) | | | |
| **zero-training** | two mean vectors (μ_img, μ_txt) | 0.288 | 0.523 | 0.648 | 0.173 |
| orthogonal Procrustes | rotation W (refuted) | 0.226 | 0.472 | 0.628 | 0.164 |
| **trained linear head** | one matrix pair, InfoNCE, 39k pairs | **0.590** | **0.871** | **0.937** | **0.404** |
| two-layer MLP | (never beats linear) | 0.495 | 0.823 | 0.925 | 0.348 |

The linear curve was still rising at 39k pairs
(0.409 → 0.450 → 0.528 → 0.590 at 5k/10k/20k/39k); the full-118k sweep
locates the saturation point (`mlp_align_full.json` when complete).

## Component sizes (why this deploys anywhere)

| component | size | inference cost |
|---|---|---|
| linear head (2 × 5376→1024) | ~44MB fp32 / ~22MB bf16 | one matvec, sub-ms on CPU |
| anchors μ_img, μ_txt | ~42KB | one vector subtract |
| 10k-item retrieval index @1024-d | ~40MB fp32 | one 10k-row matvec, ~ms on CPU |
| one hidden state on the wire | 10–21KB | trivial bandwidth |
| the backbone | ~62GB bf16 | **the only heavy piece** |

## Deployment tiers

### Tier 1 — sidecar on an existing LLM deployment (available now)

The customer already runs the multimodal LLM. The head + anchors +
index attach to the serving stack and read `hidden_states[L]` from the
forward pass the customer is already paying for. Cross-modal search,
tagging, and routing become a near-zero-marginal-cost feature: no CLIP
service, no separate embedding model, no new GPU pool. The backbone is
byte-identical (zero regression risk to its primary function) and the
added component is a linear map (auditable; you can inspect what it
does and bound what it cannot do).

### Tier 2 — edge retrieval tier (e.g. Raspberry Pi) with a remote backbone

Everything except the backbone fits comfortably on Pi-class hardware.
Split: edge device sends the image to wherever the backbone runs, gets
back a 21KB state vector, and performs centering → projection → top-K
locally against an on-device index. A single Pi can serve as the
retrieval/routing tier for a fleet of cameras or a document pipeline.
This is the SRT sidecar pattern taken literally: the semiotic layer is
small enough for edge hardware while the substrate stays in the
datacenter.

### Tier 3 — fully local small backbone (open experiment, "scale floor")

Key fact: retrieval readout needs **one prefill pass, not generation**.
That changes edge arithmetic entirely: a 2–4B multimodal model at
4-bit quantization (1–2GB weights) fits a Raspberry Pi 5 (8/16GB), and
one prefill over a few hundred image tokens is tens of seconds to ~2
minutes on its CPU. Useless interactively; entirely usable for batch
tagging, event labeling, or offline photo indexing.

The open question is scientific: does a 2–4B host's state space have
the same linearly-alignable cross-modal structure? Substrate
generality is validated from 7B to 235B, but never below 7B and never
cross-modally below 31B. The experiment is one box-day with the
existing generic pipeline (all three scripts take `--backbone`). A
positive extends the substrate claim dramatically downward and creates
the "Sunstone on an $80 computer" tier; a negative gives the
capability a measured scale floor, which the paper currently cannot
speak to. Queued in FORWARD_PLAN.md.

### Tier 3+ — the instrumented appliance (chat + retrieval + monitoring on one Pi)

The most coherent version of Tier 3 is not "retrieval on a Pi" but a
general-purpose device where **the chat model and the retrieval
substrate are the same forward passes**. One small multimodal LLM
serves chat natively; the SRT layer reads its hidden states for free.
Nothing runs twice.

| function | mechanism | realistic Pi-5 (16GB) performance |
|---|---|---|
| general chat | 2–4B multimodal model, 4-bit (1.5–2.5GB) | ~3–6 tok/s generation — slow-typing speed; fine for short answers and voice-assistant turns |
| vision Q&A | same model, image prefill | ~30–90s per image on CPU — sluggish conversationally, fine for "what's this?" |
| cross-modal retrieval | linear head + anchors + index on `hidden_states[L]` | near-zero marginal cost — states already exist from whatever prefill just ran |
| SRT monitoring (regime / r̂ / divergence) | linear probes on the same states | near-zero — same tap pattern |

Memory budget: 2.5GB weights + ~1GB KV cache + 22MB head + 40MB index
+ OS fits in 8GB, comfortably in 16GB.

The compounding property: every chat turn produces states, and states
feed the index, so the device accumulates a searchable memory of
everything it has seen and said **as a byproduct of operating**.
"Show me the photo from last week with the dog" is a retrieval query
against an index the conversations built for free. This is the SRT
thesis in appliance form: conversation, perception, retrieval, and
self-monitoring as one substrate with cheap linear taps, not a model
zoo.

**Three gating items, in order (none are might-be-impossible risks):**

1. **Scale floor** (the experiment; queued top of FORWARD_PLAN): does
   linear alignment survive on a 2–4B host?
2. **Quantization drift** (measurement): heads were trained on bf16
   states; Q4 inference perturbs them. Expected fix is calibrating
   anchors + head on states from the quantized runtime itself — the
   recipe already supports this (encode pairs through the quantized
   stack) — but it must be measured, not assumed.
3. **State-tap in the edge runtime** (engineering): llama.cpp does not
   expose intermediate-layer hidden states through its normal API; the
   ggml eval-callback mechanism (or MLC/ONNX runtimes) can reach them.
   This is the edge equivalent of `output_hidden_states=True` and is a
   prerequisite for Tiers 3 and 3+ regardless of model choice.

## Calibration rules (hard-learned, mandatory)

1. **Always center per-modality before comparing.** Raw cosine is
   dominated by the anisotropy attractor (every image retrieves the
   most image-like text). This is the repo-wide anisotropy rule
   expressed in the visual channel.
2. **Anchor domain match beats anchor size.** A μ_img over 4,000
   natural COCO photos moved the synthetic white-heart probe from rank
   352 to rank 5 against a COCO caption pool, but *degraded* the demo
   gallery's CIFAR-style thumbnails (dog 0.778 → 0.705, several
   categories off-topic). Rule: calibrate anchors on a population from
   the deployment's own query domain; 150 in-domain images beat 4,000
   out-of-domain ones. This is a productizable onboarding step:
   "provide 150–4,000 representative images."
3. **BOS on every text encode** (gemma-4: bare re-encode collapses
   replay 0.9986 → 0.615).
4. **Objective matters when training the head: use a ranking loss
   (InfoNCE), not least-squares.** Procrustes' least-squares objective
   *raised* absolute paired similarity while *destroying* rank
   discrimination. The same trap exists for any regression-trained
   projection.

## Recipe (reproduce or port to a new backbone)

```bash
# 1. Encode paired data (resumable 5k chunks; ~6 img/s on RTX PRO 6000)
python scripts/gemma4_encode_pairs.py --backbone <HF_ID> \
    --work-dir /workspace/procrustes --n-images 40000

# 2. Eval-split encodings + baseline + Procrustes control
python scripts/gemma4_procrustes_xmodal.py --backbone <HF_ID> \
    --work-dir /workspace/procrustes --out procrustes_xmodal.json

# 3. Train + evaluate the ladder (minutes from cached tensors)
python scripts/gemma4_mlp_align.py \
    --encodings /workspace/procrustes/encoded_L47_n5000.pt \
    --train-encodings /workspace/procrustes/train_pairs/chunk_*.pt \
    --train-sizes 5000 10000 20000 39000 --n-val 1000 \
    --out mlp_align_scaled.json
```

The expensive step (1) is one-time per backbone and banked to HF; head
retraining from cached tensors takes minutes, so new domains/heads are
cheap forever after.

## Honest scoping (for any external pitch or comparison)

- The claim is **"no new model"**, never "beats CLIP". Dedicated dual
  encoders trained on 10⁸+ pairs remain the standalone-retrieval
  ceiling. This is comparable-band retrieval as a free rider on the
  LLM the customer already operates, from 39k pairs and minutes of
  training, with LLM-grade compositional caption semantics.
- Validated on one backbone (gemma-4-31B-it, L47). Porting cost per
  backbone is small but nonzero; each backbone needs its own head and
  anchors.
- Our protocol (val2017 eval, 5k pool) is not the literature-standard
  Karpathy split. Run that split before placing numbers next to
  published tables (queued).
