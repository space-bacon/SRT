---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-7B
tags:
- interpretability
- activation-verbalization
- prefix-tuning
- nla
- frozen-backbone
library_name: pytorch
pipeline_tag: feature-extraction
---

# srt-nla-av-v1 — Activation Verbalizer for Qwen2.5-7B (L20)

**Read a single hidden activation as a sentence.** A 12.7M-parameter prefix
adapter over a fully frozen `Qwen/Qwen2.5-7B` that, given a layer-20 last-token
hidden state `v ∈ ℝ³⁵⁸⁴`, generates text whose own re-encoded L20 hidden state
`h` maximizes the anisotropy-corrected reconstruction
`fve_nrm_cen(h, v) = ½(1 + cos(h−μ, v−μ))`.

**TL;DR:** at best-of-64 sampling the AV **saturates the Qwen paraphrase
ceiling** (`ρ_norm ≈ 0.92`). Greedy decoding remains the open problem
(`ρ_norm ≈ 0.26`), still below a zero-training nearest-neighbour baseline.

## Card metadata

| | |
|---|---|
| **Backbone (frozen)** | `Qwen/Qwen2.5-7B`, bf16 |
| **Layer / target** | `ℓ = 20`, last-valid-token hidden of a 64-token Qwen continuation |
| **AV trainable params** | 12.7M (16 static prefix tokens + 1 inject slot + projection) |
| **Training objective** | Token CE on (v, text) pairs, where text is a Qwen continuation |
| **Training data** | `srt-nla-targets-v1` (30K (v, text) pairs, seed=1) |
| **Headline metric** | best-of-64 `fve_nrm_cen = 0.777` → `ρ_norm = 0.92` (M=200 held-out) |
| **License** | Apache-2.0 (weights). Backbone subject to Qwen license at load time. |

## Files

| File | Notes |
|---|---|
| `best_av.pt` | Warm-start AV checkpoint (`ce_seq64_np16` lineage, 30k pairs) |
| `config.json` | `NLAConfig` JSON; reproduces verbalizer geometry |
| `eval_results.json` | Triangulated numbers from `centered_eval.py` and `rerank_eval.py` |

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

repo = "RiverRider/srt-nla-av-v1"
cfg = NLAConfig.from_json(hf_hub_download(repo, "config.json"))

bb = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16
).cuda().eval()
for p in bb.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
state = torch.load(hf_hub_download(repo, "best_av.pt"), map_location="cuda",
                   weights_only=False)
av.load_state_dict(state, strict=False)
```

To **verbalize** an activation vector `v ∈ ℝ³⁵⁸⁴` extracted from layer 20 of
the frozen backbone, draw a best-of-K rollout:

```python
texts, _ = av.generate(v[None], do_sample=True, temperature=1.0,
                       max_new_tokens=64, num_return_sequences=64)
# Score each candidate by fve_nrm_cen vs v, pick argmax. See
# scripts/centered_eval.py for the canonical eval loop.
```

## Evaluation (200-target held-out slice, pool=2000)

`fve_nrm_cen` = anisotropy-corrected (subtract pool μ before cosine).
`ρ_norm = (cen − 0.510) / 0.289` ∈ [0, 1] where 1 ≡ Qwen paraphrase ceiling.

| condition | raw fve_nrm | centered | ρ_norm |
|---|---|---|---|
| greedy (T=0) | 0.687 | 0.586 | **0.26** |
| sampled (T=1) mean | 0.686 | 0.582 | 0.25 |
| best-of-8 | 0.768 | 0.686 | 0.61 |
| best-of-16 | 0.791 | 0.716 | 0.71 |
| best-of-32 | 0.814 | 0.747 | 0.82 |
| **best-of-64** | **0.834** | **0.777** | **0.92** |
| logp-rerank | 0.653 | 0.561 | 0.18 *(hurts greedy)* |
| NN-anchor rerank | 0.741 | 0.722 | 0.73 |
| NN-retrieval baseline (pool=2000) | 0.795 | 0.715 | 0.71 |
| random floor | 0.622 | 0.510 | 0.00 |
| paraphrase ceiling | 0.799 | 0.799 | 1.00 |

K-curve is log-linear: ~+0.10 `ρ_norm` per doubling of K. Extrapolation
suggests K ≈ 256 to saturate the ceiling.

## Known limitations

- **Single backbone, single layer, single target type.** All numbers above
  are Qwen2.5-7B L20 last-token of 64-token continuations. The anisotropy
  magnitude (`‖μ‖ ≈ 55`) is backbone-specific.
- **logp-rerank is dead.** Spearman(mean-logp, oracle-cen) ≈ 0.04.
  Any reranker that consumes only the policy's own sequence log-prob will
  not beat greedy. See [`paper_nla.md`](https://github.com/space-bacon/SRT/blob/nla/paper_nla.md) §3.
- **Greedy gap is the open problem.** Without K-way sampling, the AV
  under-performs a 1-line numpy NN-lookup against the same pool.

## Recommended deployment

`v` is provided at inference time, so scoring is free: do **best-of-K
oracle rerank** (sample K, score each by `fve_nrm_cen`, return argmax).
At K=64 this delivers `ρ_norm = 0.92`. No retraining required.

## Citation

```bibtex
@misc{lancaster2026nlareframe,
  title  = {Natural-Language Activation Verbalization:
            Probing the Decodability of Frozen Hidden States via Prefix-Tuned Generation},
  author = {Lancaster, Burton},
  year   = {2026},
  note   = {Draft; see github.com/space-bacon/SRT/blob/nla/paper_nla.md},
}
```

## Related

- Code: <https://github.com/space-bacon/SRT> (branch `nla`)
- Targets dataset: [`RiverRider/srt-nla-targets-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-v1)
- Companion product: [`RiverRider/srt-adapter-v1.0`](https://huggingface.co/RiverRider/srt-adapter-v1.0) (different codepath, semiotic awareness)
