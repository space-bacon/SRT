---
license: apache-2.0
language:
- en
base_model:
- openai/gpt-oss-20b
base_model_relation: adapter
tags:
- interpretability
- introspection
- monitoring
- frozen-backbone
- moe
- srt
- gpt-oss
- gpt-oss-20b
- mechanistic-interpretability
- hidden-states
- calibration
library_name: pytorch
pipeline_tag: feature-extraction
---

# srt-adapter-gptoss20b — SRT read-out + closed-loop adapter for gpt-oss-20b

> 🔬 **Live demo:** [RiverRider/srt-nla-gptoss20b-trace](https://huggingface.co/spaces/RiverRider/srt-nla-gptoss20b-trace) — full input→output hidden-state trace, magic-number state grid, A/B state-identity compare.

A **12.7M-parameter side-channel adapter** on a fully frozen `openai/gpt-oss-20b`
(MXFP4 MoE, 24 layers). It taps the residual stream between the frozen decoder
layers and exposes a calibrated introspection channel: discourse-community
assignment, regime classification, interpretant-divergence vectors, a
bifurcation order-parameter `r̂`, and a divergence-chain prediction. Trained in
combined **Phase A+B** mode: the read-out heads *and* the closed-loop RRM
injection (inject-CE through the frozen backbone) train in one run.

**TL;DR:** regime monitoring is essentially perfectly calibrated
(**ECE 0.0009**, **AUROC 0.974** on 510K held-out tokens). `r̂` tracks the
ground-truth order parameter at **Pearson 0.69** (under-predicts magnitude; an
affine rescale fixes the scale, as on every prior backbone). Community NMI 0.42
is the soft spot. Frame this as a **monitoring channel, not a capability lift**.

## Held-out probe (3,000 val passages, 510,777 tokens)

| Signal | Metric | Value |
|---|---|---|
| Regime | ECE | **0.0009** |
| Regime | Brier | 0.0152 |
| Regime | AUROC | **0.9742** (base rate 0.9455) |
| r̂ (bifurcation) | Pearson | **0.6894** (pred 0.569 / true 1.027 — affine-rescalable) |
| r̂ | MAE | 0.6156 |
| Community | NMI / ARI | 0.4226 / 0.1347 (n=3000, 35 ids, k=64) |
| Divergence norms | L6 / L12 / L18 | 20.7 / 30.0 / 29.0 (non-degenerate) |

Raw probe output ships as `phaseAB_probe.json`.

## Architecture / training

| | |
|---|---|
| Backbone (frozen) | `openai/gpt-oss-20b`, MXFP4, 24 layers, d=2880, 12/24 sliding-window layers |
| Taps | MAH @ L6, L12, L18 · community @ L3 · RRM inject @ L12, L18 |
| Trainable params | 12,713,923 (backbone: 1.8B frozen) |
| Mode | Phase A+B: read-out heads + inject-CE (gradient flows through frozen MXFP4 experts — verified differentiable) |
| Data | 1M labeled discourse passages (community ids, per-token `r_true`, chain labels). Corpus not redistributable (Reddit terms); schema + rebuild recipe in the SRT repo. |
| Recipe | bs=8, lr=1e-4, warmup 1000, val every 2000; best val at step ~18K |
| **Backbone-scale fix** | gpt-oss residual scale is ~10× Qwen's (div-std ≈ 26 vs 1–3). Divergence-magnitude losses were rescaled: `chain_weight 0.5→0.05`, `divergence_supcon 1.0→0.1`, `bif_weight 1.0→0.5`. Without this the val loss climbs after warmup. |

The adapter autodetects gpt-oss's alternating sliding-window/full attention
(`config.layer_types`) and builds explicit per-layer masks — forward is
**bit-exact** vs the HF reference (`max|diff| = 0.0` in the port smoke).

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from srt.config import SRTConfig
from srt.adapter import SRTAdapter   # pip install "srt-adapter @ git+https://github.com/space-bacon/SRT.git"

cfg = SRTConfig(backbone_id="openai/gpt-oss-20b", backbone_dtype="bfloat16")
adapter = SRTAdapter(cfg).to("cuda")   # resolves MAH@[6,12,18], inject@[12,18], community@3
sd = torch.load(hf_hub_download("RiverRider/srt-adapter-gptoss20b", "best_adapter.pt"),
                map_location="cuda", weights_only=False)
adapter.load_adapter_weights(sd) if hasattr(adapter, "load_adapter_weights") else \
    adapter.load_state_dict(sd, strict=False)
```

Requires `transformers>=4.55,<5` (gpt_oss support), `accelerate`, and — for
native MXFP4 — recent `triton` + the `kernels` package (otherwise transformers
dequantizes to bf16, ~40 GB).

## Honest caveat

SRT's validated value is **read-out / monitoring**. On prior backbones the
closed-loop injection did not improve task accuracy; the pre-registered,
replicated finding is that *low divergence predicts wrong answers*. Use this as
a calibrated introspection channel on a frozen open reasoning model, not as a
way to make gpt-oss answer better.

## Siblings

- `RiverRider/srt-nla-av-gptoss20b` — activation verbalizer (see its card for the honest K-curve)
- `RiverRider/srt-nla-gptoss20b-artifacts` — NLA pairs, VQ state codebook, anchors, K-curve
