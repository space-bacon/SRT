---
license: apache-2.0
language:
- en
base_model:
- meta-llama/Llama-3.2-3B
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

# srt-nla-av-llama32-3b — Activation Verbalizer for Llama-3.2-3B (L20)

**Read a single hidden activation as a sentence — on a different backbone family.**
A 9.44M-parameter prefix adapter over a fully frozen `meta-llama/Llama-3.2-3B`
that, given a layer-20 last-token hidden state `v ∈ ℝ³⁰⁷²`, generates text
whose own re-encoded L20 hidden state `h` maximizes the anisotropy-corrected
reconstruction `fve_nrm_cen(h, v) = ½(1 + cos(h − μ, v − μ))`.

This is the **cross-backbone replication** of
[`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
(Qwen2.5-7B). Same training pipeline, same hyperparameters, different
model family and different size. Result: every qualitative finding of the
original paper (saturating ceiling at best-of-K, log-linear K-curve, death
of logp-rerank) reproduces. See `paper_nla.md` §10.

**TL;DR:** at best-of-64 sampling the AV exceeds the NN-retrieval ceiling
(`fve_nrm_cen = 0.858 > 0.756`). Greedy decoding remains the open problem
(`0.633` centered), still below the retrieval baseline.

## Card metadata

| | |
|---|---|
| **Backbone (frozen)** | `meta-llama/Llama-3.2-3B`, bf16 |
| **Layer / target** | `ℓ = 20` (71% depth, mirrors Qwen's L20/28), last-valid-token hidden of a 64-token Llama continuation |
| **AV trainable params** | 9.44M (1 static prefix token + 1 inject slot + projection); smaller than the Qwen AV due to `hidden_size = 3072` and tied 128k-vocab lm_head |
| **Training objective** | Token CE on (v, text) pairs, where text is a Llama continuation |
| **Training data** | `srt-nla-targets-llama32-3b-v1` (30K (v, text) pairs, seed=1) |
| **Headline metric** | best-of-64 `fve_nrm_cen = 0.858` (M=32) → exceeds NN ceiling 0.756 (M=200) → `ρ_norm ≈ 1.40` |
| **License** | Apache-2.0 (weights). Backbone subject to Llama 3.2 community license at load time. |

## Files

| File | Notes |
|---|---|
| `best_av.pt` | Best SFT checkpoint (val fve_nrm 0.332 at step 5000/5337, 3 epochs on 28,465 train pairs) |
| `config.json` | `NLAConfig` JSON; reproduces verbalizer geometry |
| `eval/centered_eval.json` | M=32, K=64 centered eval |
| `eval/rerank_eval.json` | M=200, K=32 K-curve + cheap-rerank diagnostics |
| `eval/oracle_ceiling.json` | M=200 replay/random/NN/paraphrase ceilings |

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

repo = "RiverRider/srt-nla-av-llama32-3b"
cfg = NLAConfig.from_json(hf_hub_download(repo, "config.json"))

bb = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16
).cuda().eval()
for p in bb.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
state = torch.load(hf_hub_download(repo, "best_av.pt"), map_location="cuda",
                   weights_only=False)
av.load_state_dict(state, strict=False)
```

To **verbalize** an activation vector `v ∈ ℝ³⁰⁷²` extracted from layer 20 of
the frozen backbone, draw a best-of-K rollout and score each candidate by
`fve_nrm_cen` (centered cosine vs `v`); pick argmax. See
`scripts/centered_eval.py` for the canonical eval loop.

## Evaluation

`fve_nrm_cen` = anisotropy-corrected (subtract pool μ before cosine).
Pool size 2,000 in all rows.

### M=200 oracle ceiling (`scripts/oracle_ceiling.py`)

| anchor | raw fve_nrm | centered fve_nrm |
|---|---|---|
| replay (sanity) | 0.904 | 0.881 |
| paraphrase best-of-8 (Llama) | 0.764 | 0.720 |
| NN-in-pool | 0.785 | **0.756** ← used as ceiling |
| random floor | 0.569 | 0.498 |

Note: on Llama-3.2-3B base, the bare paraphrase prompt
underperforms NN-retrieval — the "paraphrase ceiling" is an
instruction-following ceiling of the base model, not a property of the
verbalization problem. We use **NN-in-pool as the headline ceiling** for
this release.

### M=32 centered eval (K=64; `scripts/centered_eval.py`)

| condition | raw fve_nrm | centered fve_nrm |
|---|---|---|
| greedy | 0.672 | 0.633 |
| sampled (mean) | 0.684 | 0.637 |
| **best-of-64** | **0.873** | **0.858** |
| NN-retrieval | 0.837 | 0.820 |
| random floor | 0.569 | 0.500 |

### M=200 K-curve (`scripts/rerank_eval.py`)

| K | centered fve_nrm |
|---|---|
| 1 | 0.636 |
| 2 | 0.678 |
| 4 | 0.716 |
| 8 | 0.748 |
| 16 | 0.780 |
| 32 | 0.809 |

Log-linear: ~+0.034 centered per doubling of K (within sampling noise of
Qwen's +0.030).

- **logp-rerank gives 0.624 centered** (+0.005 vs greedy 0.619, Spearman 0.055
  with the oracle) — same death-of-logp-rerank result as Qwen.
- **NN-anchor rerank gives 0.783 centered**, well above greedy.

## Known limitations

- **Llama-3.2-3B base paraphrase prompt is a weaker ceiling than Qwen's.**
  The bare instruction `"Paraphrase the following text using different
  words but the same meaning."` zero-shots cleanly on Qwen-2.5-7B base
  but underperforms NN-retrieval on Llama-3.2-3B base. Comparisons across
  the two releases should use centered fve_nrm directly, not normalize to
  a backbone-specific paraphrase ceiling.
- **Greedy gap is the open problem here too.** Without K-way sampling, the
  AV under-performs a 1-line numpy NN-lookup against the same pool — same
  shape as Qwen v1.
- **Same-layer transfer only.** The release uses ℓ=20 (71% depth, mirrors
  Qwen's L20/28). Other layers were not evaluated.

## Recommended deployment

Best-of-K oracle rerank (sample K, score each by `fve_nrm_cen`, return
argmax). At K=64 this delivers `fve_nrm_cen ≈ 0.86`, exceeding the
NN-retrieval baseline.

## Citation

```bibtex
@misc{lancaster2026nlareframe,
  title  = {Natural-Language Activation Verbalization:
            Probing the Decodability of Frozen Hidden States via Prefix-Tuned Generation},
  author = {Lancaster, Burton},
  year   = {2026},
  note   = {Draft; see github.com/space-bacon/SRT/blob/main/paper_nla.md (§10 cross-backbone)},
}
```

## Related

- Code: <https://github.com/space-bacon/SRT> (`nla-v1.1.0` tag)
- Targets dataset: [`RiverRider/srt-nla-targets-llama32-3b-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-llama32-3b-v1)
- Qwen sibling: [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
