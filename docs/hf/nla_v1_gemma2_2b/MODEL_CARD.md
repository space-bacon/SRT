---
license: apache-2.0
language:
- en
base_model:
- google/gemma-2-2b
tags:
- interpretability
- activation-verbalization
- prefix-tuning
- nla
- frozen-backbone
- cross-backbone-transfer
library_name: pytorch
pipeline_tag: feature-extraction
---

# srt-nla-av-gemma2-2b-v1 — Activation Verbalizer for Gemma-2-2B (L19)

**Read a single hidden activation as a sentence — on a third backbone family.**
A 5.31M-parameter prefix adapter over a fully frozen `google/gemma-2-2b`
that, given a layer-19 last-token hidden state `v ∈ ℝ²³⁰⁴`, generates text
whose own re-encoded L19 hidden state `h` maximizes the anisotropy-corrected
reconstruction `fve_nrm_cen(h, v) = ½(1 + cos(h − μ, v − μ))`.

This is the **third-backbone replication** of
[`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
(Qwen2.5-7B) and
[`RiverRider/srt-nla-av-llama32-3b`](https://huggingface.co/RiverRider/srt-nla-av-llama32-3b)
(Llama-3.2-3B). Same training pipeline, same hyperparameters, third
model family (Google's Gemma-2 lineage). Result: every qualitative
finding of the original paper reproduces. See `paper_nla.md` §11.

**TL;DR:** at best-of-64 sampling the AV exceeds the paraphrase ceiling
(`fve_nrm_cen = 0.631 > 0.598`, `ρ_norm = 1.33`). Gemma-2-2B has the
**highest anisotropy of the three backbones** (`‖μ‖ ≈ 156`, vs Qwen 55,
Llama 7.2), making this the cleanest case for the centring claim of
`paper_nla.md` §§4–5: raw greedy `fve_nrm` (0.664) is *below* the raw
random floor (0.675), so the centred metric is non-optional on this
backbone.

## Card metadata

| | |
|---|---|
| **Backbone (frozen)** | `google/gemma-2-2b`, bf16 |
| **Layer / target** | `ℓ = 19` (73% depth, mirrors Qwen and Llama at 71–73%), last-valid-token hidden of a 64-token Gemma continuation |
| **AV trainable params** | 5.31M (1 static prefix token + 1 inject slot + projection); smaller than Qwen/Llama AVs because `hidden_size = 2304` |
| **Training objective** | Token CE on (v, text) pairs, where text is a Gemma continuation |
| **Training data** | `srt-nla-targets-gemma2-2b-v1` (29,952 (v, text) pairs after 48 token-budget skips, seed=1) |
| **Headline metric** | best-of-64 `fve_nrm_cen = 0.631` (M=200) → exceeds paraphrase ceiling 0.598 → `ρ_norm = 1.33` |
| **License** | Apache-2.0 (weights). Backbone subject to Gemma 2 terms of use at load time. |

## Files

| File | Notes |
|---|---|
| `best_av.pt` | Best SFT checkpoint (val fve_nrm 0.3334 at step 4500/5337, 3 epochs on ~28,455 train pairs, ~10 min wall on RTX PRO 6000) |
| `config.json` | `NLAConfig` JSON; reproduces verbalizer geometry |
| `eval/centered_eval_M200_K64.json` | M=200, K=64 centered eval |
| `eval/rerank_eval_M200_K32.json` | M=200, K=32 K-curve + cheap-rerank diagnostics |
| `eval/oracle_ceiling_M200.json` | M=200 replay/random/NN/paraphrase ceilings |

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

repo = "RiverRider/srt-nla-av-gemma2-2b-v1"
cfg = NLAConfig.from_json(hf_hub_download(repo, "config.json"))

bb = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", torch_dtype=torch.bfloat16
).cuda().eval()
for p in bb.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
state = torch.load(hf_hub_download(repo, "best_av.pt"), map_location="cuda",
                   weights_only=False)
av.load_state_dict(state, strict=False)
```

To **verbalize** an activation vector `v ∈ ℝ²³⁰⁴` extracted from layer 19 of
the frozen backbone, draw a best-of-K rollout and score each candidate by
`fve_nrm_cen` (centred cosine vs `v`); pick argmax. See
`scripts/centered_eval.py` for the canonical eval loop.

## Evaluation

`fve_nrm_cen` = anisotropy-corrected (subtract pool μ before cosine).
Pool size 2,000 in all rows.

### M=200 oracle ceiling (`scripts/oracle_ceiling.py`)

| anchor | raw fve_nrm | centred fve_nrm |
|---|---|---|
| replay (sanity) | 0.799 | 0.713 |
| NN-in-pool | 0.781 | 0.653 |
| paraphrase best-of-8 (Gemma) | 0.720 | **0.598** ← used as ceiling |
| random floor | 0.675 | **0.498** ← floor |

Gemma-2-2B's centred ceiling–floor gap is `0.100`, smaller than Qwen's
and Llama's. This is a substantive cross-backbone fact: Gemma's
paraphrase distribution is sharper in the centred geometry, so the
normalized scale of `ρ_cen` is more compressed on this backbone.

### M=200 centered eval (K=64; `scripts/centered_eval.py`)

| condition | raw fve_nrm | centred fve_nrm | ρ_cen |
|---|---|---|---|
| greedy | 0.664 | 0.528 | 0.30 |
| sampled (mean) | 0.645 | 0.515 | 0.17 |
| **best-of-64** | **0.752** | **0.631** | **1.33** |
| NN-retrieval | 0.815 | 0.712 | 2.14 |
| random floor | 0.668 | 0.500 | 0.00 |

Note: raw greedy (0.664) sits **below** the raw random floor (0.675).
This is the strongest empirical case across the three backbones for the
non-optionality of the centring move — any uncentred reading on
Gemma-2-2B L19 reports the verbalizer as worse-than-random.

### M=200 K-curve (`scripts/rerank_eval.py`)

| K | centred fve_nrm |
|---|---|
| 1 | 0.511 |
| 2 | 0.534 |
| 4 | 0.555 |
| 8 | 0.572 |
| 16 | 0.593 |
| 32 | 0.618 |

Log-linear: ~+0.021 centred per doubling of K (shallower than Qwen's
+0.030 and Llama's +0.034, consistent with the smaller ceiling–floor
gap; the *shape* is preserved across all three backbones).

- **logp-rerank gives 0.512 centred** (-0.014 vs greedy 0.527, Spearman
  +0.030 with the oracle) — same death-of-logp-rerank result as Qwen
  and Llama. Third backbone, same finding: the AV's own confidence is
  uncorrelated with how well the candidate re-encodes.
- **NN-anchor rerank gives 0.600 centred**, well above greedy.

## Known limitations

- **Highest-anisotropy backbone tested.** `‖μ‖ ≈ 156` is roughly `3×`
  Qwen-2.5-7B's and `22×` Llama-3.2-3B's. Any uncentred metric on
  this backbone is dominated by the rotation-into-`μ` component.
- **Compressed centred scale.** The `0.100` ceiling–floor gap on Gemma
  vs `0.258` on Llama means `ρ_cen` is more sensitive to absolute
  changes; small differences in centred fve correspond to large
  differences in `ρ_cen`. Compare in centred fve directly across
  backbones, not in `ρ_cen`.
- **Greedy gap is the open problem here too.** Best-of-64 oracle rerank
  is required to beat the paraphrase ceiling.
- **Same-layer transfer only.** The release uses `ℓ=19` (73% depth).
  Other layers were not evaluated.

## Recommended deployment

Best-of-K oracle rerank (sample K, score each by `fve_nrm_cen`, return
argmax). At K=64 this delivers `fve_nrm_cen ≈ 0.63` (centred) /
`0.75` (raw), exceeding the paraphrase ceiling.

## Citation

```bibtex
@misc{lancaster2026nlareframe,
  title  = {Natural-Language Activation Verbalization:
            Probing the Decodability of Frozen Hidden States via Prefix-Tuned Generation},
  author = {Lancaster, Burton},
  year   = {2026},
  note   = {Draft; see github.com/space-bacon/SRT/blob/main/paper_nla.md (§11 cross-backbone Gemma)},
}
```

## Related

- Code: <https://github.com/space-bacon/SRT> (`nla` branch)
- Targets dataset: [`RiverRider/srt-nla-targets-gemma2-2b-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-gemma2-2b-v1)
- Qwen sibling: [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
- Llama sibling: [`RiverRider/srt-nla-av-llama32-3b`](https://huggingface.co/RiverRider/srt-nla-av-llama32-3b)
