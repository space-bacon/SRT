# NLA expert-iteration log — 2026-05-17

Goal: `fve_nrm ≥ 0.85` on SRT-NLA (greedy decode against the 10K Q7B L20
last-token target pool, sha `8f64c7dc20cc5852`).

Strategy: BoN curation + SFT distillation (Option C, expert iteration).

## Recipe (per cycle)

1. **BoN curation** (`scripts/curate_bon.py`):
   - K=32 samples per target (temp 1.2, top-p 0.95, max-new-tokens 64)
   - Score `fve_best` per target → JSONL of (target_idx, gold_ids, fve_best)
   - Runtime: ~2 h on RTX PRO 6000 96GB for N=10000
2. **Top-K filter**: keep top-30% by `fve_best` (≈3000 pairs, mean fve ≈ 0.94)
3. **SFT distillation** (`scripts/train_nla_sft.py`):
   - Teacher-forced CE on (gold_ids | injected_vector)
   - Cosine LR 1e-4, warmup 50, batch 16, 10 epochs, patience 4
   - `--init-from` previous best AV ckpt
   - Runtime: ~6 min
4. **Unbiased eval** (`scripts/eval_av_full.py`):
   - Greedy fve_nrm on 1000 random targets + bucketed (bot/mid/top by BoN ceiling)

## iter-1 sweep (data-quality ablation)

All from warm-start `artifacts/nla/n1i_v2_best/av_step002500.pt`,
trained on `bon_iter1.jsonl` (mean fve_best=0.44).

| run | data filter | LR schedule | best val (biased) |
|---|---|---|---|
| iter-1 | all 10K | 3e-5 const | 0.2631 |
| iter-1b | all 10K | 3e-4 const | 0.3053 |
| iter-1c | all 10K | 1e-4 cosine | 0.2943 |
| iter-1d | top-50% (3149) | 1e-4 cosine | 0.4344 |
| **iter-1e** | **top-30% (3000)** | **1e-4 cosine** | **0.4628** |

Key insight: data quality (top-K filter) >> LR tuning for SFT distillation.

## iter-2 (single cycle)

- BoN iter-2 from iter-1e: fve_best mean **0.5838** (vs iter-1 0.44, +0.14)
- Distribution highly bimodal: p50=0.48, p70=0.79, p90=0.98
- Top-30 pool: N=3000, mean=0.938, min=0.789 (expert-grade demos)
- SFT iter-2 from iter-1e ckpt: best val (biased on top-30 holdout) = **0.8958**
- Unbiased eval revealed overfit to easy subdistribution:

| bucket | greedy mean | BoN ceiling | frac ≥ 0.85 |
|---|---|---|---|
| **OVERALL (random 1000)** | **0.448** | — | **25.8%** |
| top-33 (easy) | 0.850 | 0.921 | 75.0% |
| mid-33 (medium) | 0.271 | 0.493 | 0.3% |
| bot-33 (hard) | 0.207 | 0.351 | 0.0% |

## Current artifacts (remote `/workspace/srt-adapter`)

- Targets: `artifacts/nla/targets_q7b_L20_10k.pt` (155 MB, sha `8f64c7dc20cc5852`)
- BoN pairs: `artifacts/nla/curated/bon_iter{1,2}.jsonl` + `_top{15,30,50}.jsonl` filters
- AV ckpts:
  - `artifacts/nla/sft_iter1e/best_av.pt` (val 0.4628 biased)
  - `artifacts/nla/sft_iter2/best_av.pt` (val 0.8958 biased / 0.448 unbiased)
- Logs: `artifacts/nla/logs/`

## In-flight (started 15:34 UTC)

- BoN iter-3 from `sft_iter2/best_av.pt` (PID 19068, ETA ~2 h)
- Early signal at step 250: fve_best=0.602, **fve_mean=0.395** (vs iter-2 0.315
  at same step — mean jumped +0.08, meaning policy now finds good demos on
  previously-hard targets)

## Next actions

1. Wait for BoN iter-3 completion (~14:50 UTC + 2h)
2. Filter top-30 → SFT iter-3 from `sft_iter2/best_av.pt`
3. Unbiased eval — expect overall mean to climb past 0.55
4. If hard bucket still stuck after iter-3, add weighted-CE variant
   (sample weight = `fve_best`) to `train_nla_sft.py` so all 10K pairs
   contribute, weighted by demo quality
5. Continue expert iteration until greedy ≥ 0.85 on unbiased eval, or until
   BoN ceiling stops climbing (then pivot to architecture: more prefix tokens,
   attention pool, wider proj)

## Reliability notes

- **Always pass `--max-val-samples`** equivalent (here `--val-vectors 256`)
  to avoid full-pool val
- Never trust subagent SSH output for status — verify via direct `pgrep` and
  log tail
- `train_nla_sft.py` reports val on a *filtered* subset (val_fraction of the
  loaded pairs), which is biased upward; always re-run `scripts/eval_av_full.py`
  on the full pool for honest numbers
