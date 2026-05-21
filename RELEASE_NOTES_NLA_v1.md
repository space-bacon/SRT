# SRT-NLA v1 — Release Notes

**Date:** 2026-05-18 (shipped to HF same day)
**Branch:** `nla`
**Last commit at release:** `7b53130`

> **2026-05-20 update.** v1 is the Qwen2.5-7B L20 anchor of what is
> now the **Stage 4** workstream of the SRT program (see
> [`paper_nla.md`](paper_nla.md) §0 for staging and §1.5 / §12 for the
> theoretical framing). Cross-backbone replications on Llama-3.2-3B
> (`paper_nla.md` §10, HF: `RiverRider/srt-nla-av-llama32-3b` and
> `RiverRider/srt-nla-targets-llama32-3b-v1`) and Gemma-2-2B
> (`paper_nla.md` §11, in progress) live on top of this v1 anchor.
> "arxiv/" in this repo is *staged source for a planned arXiv
> submission of the SRT-Adapter manuscript*; the SRT-Adapter paper is
> not yet on arXiv. The only currently-posted Lancaster preprints are
> SSRN 5987495 and SSRN 6349978.

**HF artifacts (live):**
- Model: [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1)
  — `best_av.pt` (51.6 MB), `config.json`, `README.md` (= model card),
  `eval/{centered_eval_30k_M200, oracle_ceiling_30k_v2, rerank_eval_ce_seq64_np16_v2}.json`.
- Dataset: [`RiverRider/srt-nla-targets-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-v1)
  — `targets_q7b_L20_seq64_30k_seed1.pt` (27.6 GB), `README.md` (= dataset card).

## What this release is

The first public release of an **anisotropy-corrected Activation Verbalizer
(AV)** for `Qwen/Qwen2.5-7B`, layer 20, trained with token CE on 30K
(activation, text) pairs. Total trainable parameters: 12.7M, over a fully
frozen 7B backbone.

The release ships:

- A warm-start AV checkpoint (`best_av.pt`, ~50 MB).
- The 30K-pair targets file used to train and evaluate it (~26 GB raw,
  ~155 MB pool-only).
- A triangulated `eval_results.json` covering greedy, sampled, best-of-K
  ∈ {1, 2, 4, 8, 16, 32, 64}, logp-rerank, NN-anchor rerank, and the
  NN-retrieval baseline.

## Headline numbers (200-target held-out slice, pool=2000)

| condition | raw fve_nrm | centered | ρ_norm |
|---|---|---|---|
| greedy | 0.687 | 0.586 | **0.26** |
| **best-of-64 (oracle)** | **0.834** | **0.777** | **0.92** |
| NN-retrieval baseline | 0.795 | 0.715 | 0.71 |
| paraphrase ceiling | 0.799 | 0.799 | 1.00 |

K-curve is log-linear: ~+0.10 ρ_norm per doubling of K.

## What is new vs prior work in this repo

1. **Anisotropy correction is mandatory.** Pre-2026-05-16 NLA numbers (any
   ρ ≈ 0.62 result) were dominated by a shared mean `‖μ‖ ≈ 55` in
   Qwen2.5-7B L20 activations. The "0.689 plateau" reported by the now-
   archived `scripts/_archive/probe_bestofn.py` was a measurement
   artifact. See `paper_nla.md` §2.
2. **logp-rerank has zero ranking power.** Spearman(mean-logp,
   oracle-cen) per-target mean 0.04, p50 0.05. Reranking by mean-logp
   *hurts* greedy by −0.025 cen. This kills any value-head reranker that
   only consumes logp features.
3. **Best-of-K oracle rerank is the recommended decoding method.**
   `v` is provided at inference, so scoring K rollouts costs one batched
   backbone forward and delivers ρ_norm = 0.92 at K=64. No retraining.

## Reproduction recipe

```bash
git clone https://github.com/space-bacon/SRT.git && cd SRT && git checkout nla
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Pull artifacts from HF
huggingface-cli download RiverRider/srt-nla-av-v1 best_av.pt \
  --local-dir artifacts/nla/ce_seq64_np16_30k
huggingface-cli download RiverRider/srt-nla-targets-v1 \
  targets_q7b_L20_seq64_30k_seed1.pt --local-dir artifacts/nla/ --repo-type dataset

# Reproduce the headline table
python scripts/centered_eval.py \
  --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \
  --av-ckpt artifacts/nla/ce_seq64_np16_30k/best_av.pt \
  --backbone Qwen/Qwen2.5-7B --layer 20 \
  --num-prefix-tokens 16 --num-inject-slots 1 \
  --num-vectors 200 --K 64 --pool-size 2000 \
  --out artifacts/nla/centered_eval_30k_M200.json

# Reproduce the K-curve + logp-Spearman diagnostic
python scripts/rerank_eval.py \
  --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \
  --av-ckpt artifacts/nla/ce_seq64_np16_30k/best_av.pt \
  --backbone Qwen/Qwen2.5-7B --layer 20 \
  --num-prefix-tokens 16 --num-inject-slots 1 \
  --num-vectors 200 --K 64 --pool-size 2000 \
  --out artifacts/nla/rerank_eval_ce_seq64_np16_v2.json
```

Approximate runtime on an RTX PRO 6000 Blackwell 96GB: 5 min for
`centered_eval.py`, 9 min for `rerank_eval.py`.

## What is **not** in this release

- Other layers (only L20).
- Other backbones (only Qwen2.5-7B). The anisotropy magnitude is
  backbone-specific; centering is required everywhere but the size of
  the correction is not universal.
- A greedy-competitive AV. The greedy gap (ρ_norm 0.26 → 0.92) is the
  acknowledged open problem.
- BoK distillation checkpoints. The K=4 smoke run regressed; K≥16 is
  required and has not yet been launched on Blackwell.

## Reproducibility caveats

- Targets seed=1 is fixed; reseeding will produce a different anchor
  table. The paper-reported anchors (random_floor_cen=0.510,
  paraphrase_ceiling_cen=0.799) hold for this specific shard.
- `torch.load(..., weights_only=False)` is required to load both the AV
  ckpt and the targets file. Both ship from this repo and are not
  considered an untrusted-source security risk for downstream users
  loading them from the official HF mirror.
- Validate any *regenerated* targets file with the BOS-as-EOS guard:
  `python -m srt.nla.targets_check <path>` (asserts non-collapsed std).

## Changelog from prior (unreleased) attempts

- 2026-05-16 (commit `902b746`): BOS-as-EOS bug fix in
  `scripts/sample_targets.py`. **Invalidates all NLA results predating
  this commit.** The released targets file was regenerated after this fix.
- 2026-05-17 (commit `573f51e`): Reframe — `centered_eval.py`,
  `oracle_ceiling.py`, `paper_nla.md` draft. First public anchored
  evaluation.
- 2026-05-17 (commit `8cfb357`): `rerank_eval.py` with prefix-free
  h_last extraction (avoids leaking `proj(v)` into the scored activation).
- 2026-05-18 (this release): Code cleanup, archive of dead REINFORCE
  scripts, canonical `srt/nla/metrics.py`, HF release.

## License

Apache-2.0 (code, AV weights, targets dataset). The `Qwen/Qwen2.5-7B`
backbone retains its own Qwen license at load time.
