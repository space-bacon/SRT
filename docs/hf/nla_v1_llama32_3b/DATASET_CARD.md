---
license: apache-2.0
language:
- en
tags:
- activation-vectors
- interpretability
- nla
- llama
- frozen-backbone
- cross-backbone-transfer
size_categories:
- 10K<n<100K
task_categories:
- feature-extraction
pretty_name: SRT-NLA Targets — Llama-3.2-3B L20 (seq=64, seed=1)
---

# srt-nla-targets-llama32-3b-v1 — 30K (activation, text) pairs from Llama-3.2-3B L20

The training and evaluation corpus for
[`srt-nla-av-llama32-3b`](https://huggingface.co/RiverRider/srt-nla-av-llama32-3b),
the cross-backbone replication of
[`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
on Llama-3.2-3B. Each example is a (hidden activation, source text) pair
where the activation is the **last valid token's L20 hidden state** of a
Llama-3.2-3B continuation of length 64.

## Card metadata

| | |
|---|---|
| **Backbone** | `meta-llama/Llama-3.2-3B` |
| **Layer** | 20 (71% depth, mirrors Qwen's L20/28) |
| **Token position** | Last valid token (attention-mask determined) |
| **Sequence length** | 64 tokens |
| **Dtype** | bf16 (activation), str (text) |
| **Activation dim** | 3072 |
| **N targets** | 30,000 (seed=1) |
| **Pool size (paper anchors)** | 2,000 |
| **Anisotropy `‖μ‖`** | ≈ 7.21 (~7.6× smaller than Qwen-2.5-7B's 55) |
| **SHA-256 (full file)** | `db5c9d22…1981fa` |

## Files

| File | Size | Notes |
|---|---|---|
| `targets_L20_seq64_30k_seed1.pt` | ~22.7 GB | Full (sequences, activations, attn). Use with `weights_only=False`. |

## Schema

```python
obj = torch.load(path, weights_only=False)
# obj["sequences"]:   Tensor(N, T)    — generated token ids
# obj["activations"]: Tensor(N, T, d) — bf16, L20 hidden states
# obj["d"]:           int             — 3072
# obj["bos_token_id"]: int            — 128000 (Llama-3.2)
# obj["meta"]:        {"backbone": "meta-llama/Llama-3.2-3B", "layer": 20, "seed": 1, ...}
```

## Reproduction

```bash
python scripts/sample_targets.py \
  --backbone meta-llama/Llama-3.2-3B \
  --layer 20 --seq-len 64 \
  --num-sequences 30000 --batch-size 16 \
  --dtype bfloat16 --seed 1 \
  --out artifacts/nla/llama32_3B/targets_L20_seq64_30k_seed1.pt
```

The activation extractor (`scripts/sample_targets.py`) is backbone-agnostic;
no code changes were needed to switch from Qwen to Llama. The companion
gold-pair builder is similarly parametric: `scripts/build_gold_pairs.py
--backbone meta-llama/Llama-3.2-3B --max-len 64`.

## Anchors derived from this dataset

Computed on a 200-target held-out slice (see `oracle_ceiling.json` released
alongside the model card), pool=2000:

| anchor | raw fve_nrm | centered |
|---|---|---|
| replay (re-encode) | 0.904 | 0.881 |
| NN-retrieval (pool=2000) | 0.785 | **0.756** ← headline ceiling |
| paraphrase best-of-8 (Llama) | 0.764 | 0.720 |
| random floor (off-diagonal) | 0.569 | **0.498** ← floor |

Centering subtracts the per-coordinate pool mean before cosine; on
Llama-3.2-3B `‖μ‖ ≈ 7.21`. Note that paraphrase best-of-8 underperforms
NN-retrieval on Llama-3.2-3B base because the bare paraphrase prompt
zero-shots less well on it than on Qwen-2.5-7B base — the "paraphrase
ceiling" is an instruction-following ceiling of the base model, not a
property of the verbalization problem. NN-in-pool is the headline ceiling
for this release.

## Intended use

- Training and evaluating activation verbalizers / decoders for Llama-3.2-3B L20.
- Building hard-negative pools for InfoNCE-style activation→text losses.
- Cross-backbone studies of mid-layer representation geometry (Qwen vs Llama).

## Out-of-scope

- Numbers do **not** transfer to other backbones, sizes, or layers without
  recomputing μ, the floor, and the ceiling.
- Texts are model-generated continuations of seed prompts, not human-written;
  do not treat as a natural-language corpus for unrelated NLP work.

## Citation

See [`srt-nla-av-llama32-3b`](https://huggingface.co/RiverRider/srt-nla-av-llama32-3b) model card.
