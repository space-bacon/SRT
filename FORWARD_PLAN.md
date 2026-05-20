# Forward Plan — SRT / SRT-NLA

**Date:** 2026-05-18
**Author of last edit:** end-of-cleanup pass after NLA v1 release.

This is the single source of truth for "what is the next thing to work on"
across both workstreams in this repo. Supersedes the per-day handoff in
`SESSION_HANDOFF.md` (which remains a snapshot of 2026-05-17).

---

## Two workstreams, one repo

| Workstream | Current SOTA | Branch | Status |
|---|---|---|---|
| **SRT-Adapter** (semiotic awareness for frozen LLMs) | `v22c_a050`, mean MTEB-STS 0.3744 (soup of v18+v21a) | `main` | Shipping. v1.0 on HF for downstream pinning. |
| **SRT-NLA** (activation verbalization, frozen backbone) | `srt-nla-av-v1`, best-of-64 ρ_norm = 0.92, greedy ρ_norm = 0.26 | `nla` | First public release 2026-05-18. Greedy gap is the open problem. |

---

## NLA — next push, in priority order

### 1. Ship the paper *(this week)*

Status: `paper_nla.md` is at v0.2 (post-cleanup). Action items:

- [ ] Add a §3.5 subsection summarising the K-curve and the
      Spearman(logp, oracle) ≈ 0.04 finding from `rerank_eval.py`. *(in
      this release)*
- [ ] Cross-link the HF model + dataset cards from §6 (Artifacts).
- [ ] One pass for unit consistency: every "0.28 / 0.99" claim should be
      `ρ_norm = 0.26 / 0.92` (post-rerank_eval triangulation).
- [ ] Decide venue: short methods note (≤8 pages) vs blog-post-with-arxiv.
      The contribution is the **reframe + the K-curve + the logp-death
      result**, not a new model.

**Why now**: the result is publishable as-is. Gating on closing the
greedy gap means gating on a separate research project.

### 2. Best-of-K oracle rerank as the deployable decode *(zero work)*

Already done — it ships as the recommended path in the model card and
release notes. Nothing more to do on this option; just make sure the
paper frames it as the headline decoding method.

### 3. BoK distillation at K=32 *(the policy-improvement experiment)*

From the released K-curve, win-of-32 ≈ 0.747 cen vs greedy 0.586 cen —
a real teacher gap, unlike the K=4 smoke (which regressed val).

- [ ] Profile memory on Blackwell: K=32 × seq_len=64 × batch=16 = 32k
      rollout tokens/step. May need `--batch=8 --samples-per-v=32`.
- [ ] Temperature anneal 1.5 → 0.7 over training.
- [ ] Keep hard-neg InfoNCE (J=8 NN from pool), α_bok=1.0, β_ctr=0.3,
      γ_act=0.
- [ ] Success criterion: greedy ρ_norm > 0.40 (currently 0.26) without
      regressing best-of-64 ρ_norm < 0.85 (currently 0.92).

**Risk**: even at K=32, teacher distribution may still concentrate on
greedy mode → no signal. Mitigation: monitor `win_top1_vs_greedy_rate`
during warmup; abort if < 0.2 after 500 steps.

### 4. Different-backbone sanity check *(one-script experiment)*

All current numbers are Qwen2.5-7B L20 (`‖μ‖ ≈ 55`). The greedy/best-of-K
gap may be largely an anisotropy story. Quick prior-art check:

- [ ] Run `oracle_ceiling.py` on LLaMA-3-8B L20 — measure `‖μ‖` and
      random_floor_cen.
- [ ] If `‖μ‖_LLaMA ≪ ‖μ‖_Qwen`, raw fve_nrm may already give a useful
      signal there → greedy gap may close *for free* on a less
      anisotropic backbone.
- [ ] One Blackwell day total; either confirms backbone-agnosticism of
      the centering claim or surfaces a new lever.

### 5. Things to NOT do *(based on negative results)*

- Don't build a logp-only reranker. Spearman 0.04.
- Don't rerun K=4 BoK. Teacher = noise.
- Don't add more prefix tokens past np=16. np=32 returned +0.003 raw.
- Don't add more multi-inject slots (M=4 returned 0).
- Don't retrain greedy-only with a higher CE weight — confirmed
  ρ_norm 0.26 ceiling on the warm-start.

---

## SRT-Adapter — next push, in priority order

### 1. Souping is the cheap Pareto move

Confirmed twice (v21b, v22c). Default behaviour for any new checkpoint
pair should be `soup_adapters.py` over `--alpha {0.3, 0.5, 0.7}` before
declaring a winner.

### 2. Acknowledged ceiling on Qwen2.5-7B-NLI

Mean STS 0.366 → 0.371 → 0.374 over 4 souping iterations suggests we
are within sampling noise of the InfoNCE-on-Qwen2.5-7B-NLI ceiling.
Further gains likely require one of:

- a different corpus (multilingual hard NLI, code-switched STS),
- a different backbone (LLaMA-3, Mistral, Phi-3.5),
- supervised distillation from a much larger embedder (e.g. e5-mistral-7b).

None of these are scheduled. SRT-Adapter is in maintenance mode pending
a new corpus or backbone delivery.

---

## Repo hygiene

Done in 2026-05-18 cleanup pass:

- ✓ Canonical metrics in [srt/nla/metrics.py](srt/nla/metrics.py).
- ✓ Dead REINFORCE arc archived to `scripts/_archive/` with explanation.
- ✓ README updated to mention NLA workstream + paper.
- ✓ `docs/nla_mission.md` banner pointing to `paper_nla.md`'s
  centered-unit conventions.
- ✓ HF model card, dataset card, release notes drafted.

Still to do (deferred):

- [ ] Backfill integration tests for `srt/data/dataset.py` (history of
      contrastive-collision bug).
- [ ] Promote `srt/nla/targets_check.py` into a pytest fixture that runs
      automatically on any new targets file.
- [ ] Consolidate `_last_token_h()` / `_last_h()` helpers (also
      duplicated across the surviving train_nla_*.py scripts).
- [ ] Replace `weights_only=False` calls with a centralized
      `srt.io.safe_load()` helper that documents the assumption.

---

## "If I am picking this up after a break" checklist

1. `git fetch origin && git checkout nla && git pull --ff-only`.
2. Read this file. Then [`RELEASE_NOTES_NLA_v1.md`](RELEASE_NOTES_NLA_v1.md)
   for what shipped, then [`paper_nla.md`](paper_nla.md) for the framing.
3. If the targets file is missing locally, pull from HF
   (`huggingface-cli download RiverRider/srt-nla-targets-v1 ...`).
4. If the AV ckpt is missing locally, pull from HF
   (`huggingface-cli download RiverRider/srt-nla-av-v1 best_av.pt ...`).
5. Smoke test: `pytest tests/test_nla_smoke.py -q`.
6. To resume, pick item 1 / 3 / 4 from the NLA list above based on
   available compute (paper = local; BoK = Blackwell day; backbone-check
   = Blackwell day).
