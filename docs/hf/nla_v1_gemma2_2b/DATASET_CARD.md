---
license: apache-2.0
language:
- en
tags:
- activation-vectors
- interpretability
- nla
- gemma
- frozen-backbone
- cross-backbone-transfer
size_categories:
- 10K<n<100K
task_categories:
- feature-extraction
pretty_name: SRT-NLA Targets — Gemma-2-2B L19 (seq=64, seed=1)
---

# srt-nla-targets-gemma2-2b-v1 — 30K (activation, text) pairs from Gemma-2-2B L19

The training and evaluation corpus for
[`srt-nla-av-gemma2-2b-v1`](https://huggingface.co/RiverRider/srt-nla-av-gemma2-2b-v1),
the third-backbone replication of
[`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
on Gemma-2-2B. Each example is a (hidden activation, source text) pair
where the activation is the **last valid token's L19 hidden state** of a
Gemma-2-2B continuation of length 64.

## Card metadata

| | |
|---|---|
| **Backbone** | `google/gemma-2-2b` |
| **Layer** | 19 (73% depth, mirrors Qwen's L20/28 and Llama's L20/28) |
| **Token position** | Last valid token (attention-mask determined) |
| **Sequence length** | 64 tokens |
| **Dtype** | fp32 (activation), str (text) |
| **Activation dim** | 2304 |
| **N targets** | 30,000 (seed=1) |
| **Pool size (paper anchors)** | 2,000 |
| **Anisotropy `‖μ‖`** | ≈ 156.3 (~3× Qwen-2.5-7B, ~22× Llama-3.2-3B) |
| **SHA-256 (full file)** | `129f1322…fba9d` |

## Files

| File | Size | Notes |
|---|---|---|
| `targets_L19_seq64_30k_seed1.pt` | ~17 GB | Full (sequences, activations, meta). Use with `weights_only=False`. |

## Schema

```python
obj = torch.load(path, weights_only=False)
# obj["sequences"]:   list[Tensor]    — generated token id tensors
# obj["activations"]: list[Tensor]    — fp32, L19 hidden states (per sample, shape (T_i, 2304))
# obj["meta"]:        {"backbone_id": "google/gemma-2-2b", "extraction_layer": 19, "d": 2304, "seed": 1, "temperature": 1.0, "top_p": 1.0}
```

Note: unlike the Qwen and Llama target packs which store `(N, T)` /
`(N, T, d)` dense tensors, the Gemma pack stores per-sample lists. This
follows the post-`902b746` sampler's preferred format and avoids
padding-mask ambiguity at evaluation time.

## Reproduction

```bash
python scripts/sample_targets.py \
  --backbone google/gemma-2-2b \
  --layer 19 --seq-len 64 \
  --num-sequences 30000 --batch-size 16 \
  --dtype bfloat16 --seed 1 \
  --out artifacts/nla/gemma2_2B/targets_L19_seq64_30k_seed1.pt
```

The activation extractor (`scripts/sample_targets.py`) is backbone-agnostic;
no code changes were needed to switch from Llama to Gemma. Gemma-2-2B has
**distinct `bos_token_id=2` and `eos_token_id=1`** — explicitly *not* the
Qwen-style trap that produced the constant-target bug fixed at commit
`902b746`. The companion gold-pair builder is similarly parametric:
`scripts/build_gold_pairs.py --backbone google/gemma-2-2b --max-len 64`,
which yields 29,952 pairs (48 token-budget skips at this length).

## Anchors derived from this dataset

Computed on a 200-target held-out slice (see `oracle_ceiling_M200.json`
released alongside the model card), pool=2000:

| anchor | raw fve_nrm | centred |
|---|---|---|
| replay (re-encode) | 0.799 | 0.713 |
| NN-retrieval (pool=2000) | 0.781 | 0.653 |
| paraphrase best-of-8 (Gemma) | 0.720 | **0.598** ← headline ceiling |
| random floor (off-diagonal) | 0.675 | **0.498** ← floor |

Centring subtracts the per-coordinate pool mean before cosine; on
Gemma-2-2B `‖μ‖ ≈ 156`. Note that raw greedy verbalizer fve (0.664) sits
**below** the raw random floor (0.675) — the strongest empirical case
across the three backbones for the non-optionality of centring on
high-anisotropy backbones.

## Intended use

- Training and evaluating activation verbalizers / decoders for Gemma-2-2B L19.
- Building hard-negative pools for InfoNCE-style activation→text losses.
- Cross-backbone studies of mid-layer representation geometry (Qwen vs Llama vs Gemma — three labs, three architectures, three anisotropy regimes).

## Out-of-scope

- Numbers do **not** transfer to other backbones, sizes, or layers without
  recomputing μ, the floor, and the ceiling.
- Texts are model-generated continuations of seed prompts, not human-written;
  do not treat as a natural-language corpus for unrelated NLP work.

## Citation

See [`srt-nla-av-gemma2-2b-v1`](https://huggingface.co/RiverRider/srt-nla-av-gemma2-2b-v1) model card.
