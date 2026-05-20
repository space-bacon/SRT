---
license: apache-2.0
language:
- en
tags:
- activation-vectors
- interpretability
- nla
- qwen
- frozen-backbone
size_categories:
- 10K<n<100K
task_categories:
- feature-extraction
pretty_name: SRT-NLA Targets v1 — Qwen2.5-7B L20 (seq=64, seed=1)
---

# srt-nla-targets-v1 — 30K (activation, text) pairs from Qwen2.5-7B L20

The training and evaluation corpus for [`srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1).
Each example is a (hidden activation, source text) pair where the activation
is the **last valid token's L20 hidden state** of a Qwen2.5-7B continuation
of length 64.

## Card metadata

| | |
|---|---|
| **Backbone** | `Qwen/Qwen2.5-7B` |
| **Layer** | 20 |
| **Token position** | Last valid token (attention-mask determined) |
| **Sequence length** | 64 tokens |
| **Dtype** | bf16 (activation), str (text) |
| **Activation dim** | 3584 |
| **N targets** | 30,000 (seed=1) |
| **Pool size (paper anchors)** | 2,000 |
| **Anisotropy `‖μ‖`** | ≈ 55.18 |

## Files

| File | Size | Notes |
|---|---|---|
| `targets_q7b_L20_seq64_30k_seed1.pt` | ~26 GB | Full (activations, texts, attn_masks). Use with `weights_only=False`. |
| `targets_q7b_L20_seq64_30k_seed1_pool.pt` | ~155 MB | Just the (last-token) pool tensor — sufficient for centered_eval / oracle_ceiling. |
| `data_card.md` | — | This file. |

## Schema

```python
obj = torch.load(path, weights_only=False)
# obj["activations"]: list[Tensor(seq_len, 3584)] of len N
# obj["texts"]:       list[str] of len N
# obj["attn_masks"]:  list[Tensor(seq_len,)] of len N
# obj["meta"]:        {"backbone": "Qwen/Qwen2.5-7B", "layer": 20, "seed": 1, ...}
```

The "pool" file contains only `torch.stack([a[last_valid_idx] for a in activations])`,
shape `(N, 3584)`, which is what `centered_eval.py` and `oracle_ceiling.py`
actually consume.

## Reproduction

```bash
python scripts/sample_targets.py \
  --backbone Qwen/Qwen2.5-7B \
  --layer 20 --seq-len 64 \
  --n-targets 30000 --seed 1 \
  --out artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt
```

**Critical**: `scripts/sample_targets.py` was patched on 2026-05-16
(commit `902b746`) to guard against Qwen2.5's `bos_token_id ==
eos_token_id == 151643`, which previously caused every target to collapse
to a single constant activation. The targets file above was regenerated
*after* that fix. Validate any newly-produced targets file with
`python -m srt.nla.targets_check <path>` (asserts `targets.std(0).mean() > 0.1`).

## Anchors derived from this dataset

Computed on a 200-target held-out slice (see `artifacts/nla/oracle_ceiling_30k_v2.json`):

| anchor | raw fve_nrm | centered |
|---|---|---|
| replay (re-encode) | 0.973 | 0.968 |
| paraphrase best-of-8 (Qwen) | 0.848 | **0.799** ← ρ_norm = 1 |
| NN in-pool (pool=200) | 0.750 | 0.663 |
| NN-retrieval (pool=2000) | 0.795 | 0.714 |
| random floor (off-diagonal) | 0.622 | **0.510** ← ρ_norm = 0 |

## Intended use

- Training and evaluating activation verbalizers / decoders for Qwen2.5-7B L20.
- Building hard-negative pools for InfoNCE-style activation→text losses.
- Anisotropy / dynamic-range studies of mid-layer Qwen representations.

## Out-of-scope

- Numbers do **not** transfer to other backbones or layers without recomputing μ.
- Texts are model-generated continuations of seed prompts, not human-written;
  do not treat as a natural-language corpus for unrelated NLP work.

## Citation

See [`srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1) model card.
