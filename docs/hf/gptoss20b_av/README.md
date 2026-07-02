---
license: apache-2.0
language:
- en
base_model:
- openai/gpt-oss-20b
base_model_relation: adapter
tags:
- interpretability
- activation-verbalization
- nla
- frozen-backbone
- prefix-tuning
- gpt-oss
- gpt-oss-20b
- mechanistic-interpretability
- hidden-states
library_name: pytorch
pipeline_tag: feature-extraction
---

# srt-nla-av-gptoss20b — Activation Verbalizer for gpt-oss-20b (multi-layer)

> 🔬 **Live demo:** [RiverRider/srt-nla-gptoss20b-trace](https://huggingface.co/spaces/RiverRider/srt-nla-gptoss20b-trace) — this AV running best-of-K decodes next to the (stronger) codebook retrieval decoder.

A prefix-adapter **Activation Verbalizer (AV)** over a fully frozen
`openai/gpt-oss-20b`: given a hidden state `v ∈ ℝ²⁸⁸⁰` from layer 6/12/18/24,
it generates text whose re-encoded hidden state approximates `v`. Trained
corpus-free on 200K `(v, prefix)` pairs sampled from the backbone's own
unconditional generations across all four layers (the *full input→output
trace* recipe), with a learned per-layer embedding on the inject slot.

**TL;DR — read this before using:** on gpt-oss-20b the AV **does not reach the
zero-training nearest-neighbour retrieval baseline at any practical K**. If you
want to decode gpt-oss hidden states, use the **VQ state codebook**
(`RiverRider/srt-nla-gptoss20b-artifacts`) — retrieval decoding is stronger and
O(1). This card publishes the negative result honestly because it is a real
cross-backbone finding: Qwen-2.5-7B's AV crosses its paraphrase ceiling at
best-of-64; gpt-oss-20b's does not even reach its NN baseline.

## Measured K-curve (centered fve, 50 held-out L18 targets)

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| best-of-K | 0.541 | 0.566 | 0.578 | 0.589 | 0.609 | 0.628 | **0.642** |

Anchors (same protocol): **random floor 0.500 · NN retrieval 0.744 · replay 0.999.**
Slope ≈ +0.017 per doubling of K (Qwen: +0.030); matching the NN baseline
extrapolates to K ≈ 4096.

Four training recipes were tried and all landed in a statistically tied
0.56–0.60 band at best-of-8: multi-layer CE (this checkpoint, 0.599),
+injection-scale normalization (0.572), +best-of-K checkpoint selection
(0.563), single-layer L18 × 3 epochs (0.572). The ceiling is a property of the
backbone (harmony-tuned reasoning model; L18 anisotropy ‖μ‖ ≈ 4438, ~80× Qwen's
L20), not a training artifact.

## Metric discipline (important on this backbone)

Raw `fve_nrm` is **uninterpretable** on gpt-oss-20b: two *unrelated* L18 states
already score ≈ 0.837 raw. Always report the **centered** metric
(`½(1+cos(h−μ, v−μ))` with μ the pool mean) plus a retrieval baseline.
`anchors_L18.json` in the artifacts repo carries the calibration.

## Card metadata

| | |
|---|---|
| Backbone (frozen) | `openai/gpt-oss-20b`, MXFP4/bf16 |
| Layers / targets | v = per-position hiddens at L6/12/18/24 of 64-token self-generations; gold text = the exact generating prefix |
| Trainable params | ~12.7M eq. class: proj(2880→2880) + 16 static prefix tokens + layer embedding (zero-init) |
| Objective | token CE on (v, gold-prefix) pairs (`ce_weight=1, act_weight=0`) |
| Training data | 200K pairs, balanced ~50K/layer (`trace_pairs.jsonl` in the artifacts repo) |
| Config | `num_prefix_tokens=16`, `use_layer_embed=True`, `inject_norm="none"` for this checkpoint |

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

bb = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b",
                                          dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
cfg = NLAConfig(backbone_id="openai/gpt-oss-20b", extraction_layer=18,
                num_prefix_tokens=16, use_layer_embed=True, max_new_tokens=64)
av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
sd = torch.load(hf_hub_download("RiverRider/srt-nla-av-gptoss20b", "best_av/best_av.pt"),
                map_location="cuda", weights_only=False)
av.load_state_dict(sd["trainable"], strict=False)

texts = av.verbalize(v, do_sample=True, temperature=1.0, layer=18)  # best-of-K: repeat v K times
```

Deploy-time decoding, when you must use the AV: sample K candidates, re-encode
each, keep the arg-max of centered fve against `v` (oracle rerank — `v` is
available by construction).

## Recommended alternative: codebook retrieval

```python
from srt.nla import StateIndex
si = StateIndex.load(hf_hub_download("RiverRider/srt-nla-gptoss20b-artifacts",
                                     "state_codebook_vq.pt", repo_type="dataset"))
text = si.decode(v)          # O(1)-ish nearest-of-4096-codes lookup
code = si.encode(v)          # the state's integer "magic number"
```

## Siblings

- `RiverRider/srt-adapter-gptoss20b` — the SRT read-out adapter (regime AUROC 0.974)
- `RiverRider/srt-nla-gptoss20b-artifacts` — pairs, codebook, anchors, K-curve
