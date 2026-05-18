# SRT-NLA: Plan to reach `fve_nrm ≥ 0.85`

## State (2026-05-17)
- Backbone: frozen `Qwen/Qwen2.5-7B`, layer 20, last-token pool. d=3584.
- Targets: `artifacts/nla/targets_q7b_L20_10k.pt` — 10,000 activations produced by
  Qwen self-sampling. **The file also stores the gold source `sequences` that
  produced each activation.** Every target is therefore reachable on the
  manifold by at least one known token sequence.
- Best biased eval: 0.896 (SFT iter-3b, `n_pref=8`).
- Best unbiased eval: mean 0.459, `frac ≥ 0.85` = 0.288, bot-tertile mean 0.215.

## Diagnostic conclusion (BoN K=256 on hardest-300)
- K=32 → K=256 lifted `fve_best` from 0.201 → ~0.27 mean on the hard cohort.
- Sample-efficiency curve has clearly saturated.
- Combined with the fact that **gold prefixes exist for every target** (verified
  in the artifact), the bottleneck is **inversion / search**, not the manifold.
  Current SFT used BoN pseudo-labels and never used the true gold labels.

## Plan in priority order

### P0 — Gold-teacher SFT (highest expected ROI)
Train AV with teacher-forced CE on the actual gold `sequences[i]` paired with
each target activation. Replaces BoN-curated pairs with `(target_act,
gold_prefix_tokens)`.

- Reuses `scripts/train_nla_sft.py`; only the dataset source changes.
- Warm-start from `artifacts/nla/sft_iter3b_npref8/best_av.pt` (keeps `n_pref=8`).
- Expected outcome: large jump on biased (gold leakage) and material jump on
  unbiased held-out (generalisation of the inverse map).

### P1 — Test-time guided decoding (stacks on P0 ckpt)
For held-out targets, replace stochastic AV sampling with AR-guided decoding:
beam search whose value function is `fve_nrm(partial_pool, target)`. Optional
soft-embed gradient refinement of the conditioning prefix per target.

- No retraining; bolts onto whichever checkpoint wins.
- Run on the hardest tertile only (cheap that way).

### P2 — Reconstruction-loss fine-tune (RFT) on top of P0
Close the residual gap by directly optimising `fve_nrm` via Gumbel-softmax or
REINFORCE through generation. P0 gives a strong init that removes cold-start.

### P3 — Architectural surface (only if P0+P1+P2 plateau short of 0.85)
- Multi-token AR pool (mean of generated tokens).
- Longer `max_new_tokens` (256/512).
- Larger `n_pref` (32) and deeper AV `proj`.

### P4 — Reframe (escape hatch)
If 0.85 is still out of reach, report the K-scaling curve and the gold-vs-inverse
gap as a geometric characterization of the inverse problem.

## Acceptance
Target: `fve_nrm ≥ 0.85` on the held-out unbiased eval mean, with `frac ≥ 0.85`
moving substantially above 0.288.
