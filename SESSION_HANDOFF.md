# SRT — Session Handoff (2026-07-24)

## What got done

1. **§12.5 nla-law battery complete: all 4 tests now run.** T3 (causal
   U_com via community-forcing) landed as an informative NEGATIVE:
   zooL4nD3r-v0.1's divergence channel does not track contestedness
   (wrong channel, rho=+0.05); v1.0's does (rho=+0.58) but forcing its
   24 elicited community anchors moves the interpretant a near-constant
   0.086±0.001 — comm_proj is a uniform style bias, not a per-sign
   disambiguator. Coupling is representational, localized to the sign.
   T4 (diachronic, gemma-4 over 11,876 AmericanStories articles
   1770-1964) PASSES: contested-vs-control DiD +0.048, perm p=0.007,
   decade crossover ~1900-1940.
2. **Project-review hygiene pass** (full-repo code review, top-5 fixes):
   - README Quick Start fixed (`SRTConfig.from_json`, `out.ben_output.r_hat`);
     `srt.__version__` synced to 1.0.0.
   - Both `torch.load(weights_only=False)` holes closed:
     `srt_introspect/trace.py` now loads AV ckpts with
     `weights_only=True` + `safe_globals([PosixPath])`;
     `StateIndex.load` uses `weights_only=True`.
   - `rho_norm` + anchor constants deduplicated — `srt/nla/metrics.py`
     is the single source; `decoding.py` re-exports for back-compat.
     Verified `_rollout`'s EOS-inclusive scoring convention matches
     `centered_eval.py` (anchors consistent); documented in the docstring.
   - New unit tests: `tests/test_losses.py` + `tests/test_dataset.py`
     (44 tests) covering the SupCon-collapse bug class; dataset now
     counts + warns on the `_stable_hash("")` fallback and on silent
     r_true alignment failures. Full fast suite: 68 passed.
   - `chain_loss([])` empty-list guard fixed (was IndexError).
   - WIRED pitch references removed from tracked files (handled outside
     the repo).
3. Local test venv created at `.venv/` (gitignored), torch 2.x CPU —
   fast suite now runnable locally, not just on remote boxes.
4. **Cross-modal alignment arc (§11.6.4, one box-day, ssh5:28621):**
   - Procrustes rotation hypothesis REFUTED (all variants below the
     0.288 centred baseline; mu_cos 0.979).
   - Zero-training COCO benchmark banked: i2t R@1 0.288/R@5 0.523/R@10
     0.648, t2i 0.173.
   - Anchor rule refined: natural-photo mean fixes the white-heart
     boundary (352 → 5) but domain beats size (COCO mean degraded the
     CIFAR-thumb gallery; demo Space kept its in-domain mean).
   - **Modality gap is anisotropic-LINEAR**: trained linear + InfoNCE
     at 39k train2017 pairs → R@1 0.590/R@5 0.871/R@10 0.937, median
     rank 1; MLP never overtakes. Full 118k sweep RUNNING (auto-backup).
   - Product/engineering write-up: `docs/CROSSMODAL_LINEAR_HEAD.md`
     (three deployment tiers incl. the Raspberry-Pi retrieval tier and
     the small-backbone scale-floor experiment). Strategy:
     `leverage.md` 2026-07-24 addendum. Queue: `FORWARD_PLAN.md`.

## Open items

- NOTE: the "gemma-4 base checkpoint run" queue item from 2026-07-08 is
  DONE (paper_nla.md §11.7: base-vs-IT conjecture refuted; K-curves
  indistinguishable). Remaining carry-overs: Sunstone Procrustes
  image→text projection, polyseme minimal pairs, state-switchboard
  pilot, MTEB engv2 check, ginigen leaderboard.
- New box (2026-07-24): `ssh -p 28621 root@ssh5.vast.ai -L 8080:localhost:8080`.

---

# SRT — Session Handoff (2026-07-08, night)

All work committed and pushed; SRT and the SRT-Sunstone mirror in sync.
Box ssh2:24453 is SAFE TO DESTROY: no jobs, all data on HF/git/local
(compact L47 retrieval indexes + AV ckpts on HF; raw target .pt files
regenerable from `data/{corpus_targets,caption_pool}.jsonl` in ~8 min).

## Tonight's results (gemma-4-31B-it, L47)

1. **Greedy-gap campaign complete** (`paper_nla.md` §11.7): anchors
   replay .994 / NN .695 / floor .494; CE verbalizer mode-collapses at
   the floor under argmax; draft-conditioning REFUTED via 4-way CE
   decomposition (NN draft adds 0.03 nats); K-curve +0.017/doubling
   patterns with gpt-oss. Hypothesis: base models verbalize,
   instruction-tuned hosts do not. Cheap test: gemma-4 base checkpoint.
2. **Open-vocab caption retrieval validated** (§11.6.3): 5/5 CIFAR
   images retrieve on-topic COCO captions at rank 1 from a 10k pool,
   zero training. Boundary: synthetic graphics rank low (heart caption
   352/10088).
3. **Sunstone demo v2 LIVE**: third read-out panel "The caption it
   retrieves" (per-category, up to 0.778). Space RUNNING.
4. Backbone gotchas banked: gemma-4 needs corpus targets (bare-BOS
   degenerates), BOS-prepended re-encodes, Gemma4ForConditionalGeneration
   loader. Units rule: train_nla_act val logs are COSINE, anchors are fve.

## Open items

- gemma-4 **base** checkpoint run of the same pipeline (tests the
  tuned-vs-base hypothesis; highest-value next GPU job).
- Vision boundary levers: natural-photo image-mean, Procrustes
  image→text projection.
- Carry-overs: MTEB engv2 status; ginigen leaderboard;
  arXiv posting (leverage.md 1).

---

# SRT — Session Handoff (2026-07-08)

All work committed and pushed; `space-bacon/SRT` and the
`space-bacon/SRT-Sunstone` mirror are both at `7da4691`. A vast.ai box
(ssh4:20759, 1× RTX PRO 6000 Blackwell) may still be billing with no
active jobs — all data is safe on HF/git/local, spin down at will.

## Current state (what shipped recently)

1. **SRT-Sunstone live.** Text-trained community read-out (12.3M head,
   step-2250 ckpt) on frozen `google/gemma-4-31B-it` reads images
   zero-shot. Space: `RiverRider/srt-sunstone` (RUNNING, gradio 6.19.0),
   model: `RiverRider/Gemma-4-31B-it-SRT-Sunstone`. Source:
   `demo/cross_modal_space/`. Mobile overflow fixed (root cause:
   Gradio 6 `.grid-container` fixed-px columns; layout now
   single-column + auto-fill grids).
2. **Autostereogram study** in `paper_nla.md` §11.6.2: read-out honestly
   reports texture on a disparity-only figure; simulated binocular
   fusion (`scripts/stereo_decode.py`) recovers the heart and both the
   caption head and read-out then name it. Composite figure at
   `artifacts/nla/gemma4/stereo/stereo_figure.png`.
3. **Repo duplicated** to `github.com/space-bacon/SRT-Sunstone` (mirror
   push, all branches + tags) and cloned locally at
   `/Users/burtron/development/SRT-Sunstone`, synced to `7da4691`.
4. **gptoss20b trace_pairs_L18.jsonl** committed to git (13MB).
5. (Press/outreach item moved out of the repo.)

## Open items

- Uncommitted: `demo/cross_modal_space/promo/sunstone_mobile.png`
  (untracked screenshot; commit or discard).
- Dependabot: resolved 2026-07-08 — gradio bumped to 6.19.0
  (CVE-2026-48545); transformers CVE-2026-4372 dismissed tolerable-risk
  (5.3.0 fix breaks KV-cached generation; rationale at each pin).
- MTEB(eng, v2) 41-task run for `v22c_a050`: launched 2026-07-03 on a
  since-retired box; verify whether results were pulled, relaunch if not.
- ginigen Metacognition leaderboard: 4 backbones queued, still unscored.
- Queued GPU experiments: multi-position spoof test; Qwen2.5-7B
  replication of the L24-surface/L18-semantic split.
- See `FORWARD_PLAN.md` (2026-07-08 addendum) for the prioritized queue.

---

# SRT-Adapter NLA — Session Handoff (2026-05-20, end of day)

Vast.ai Blackwell instance still up at `ssh -p 37853 root@ssh8.vast.ai`
(decide whether to spin down — no active jobs). All v2b artifacts pulled
to local `artifacts/nla/bok_v2b_seq64_np16/`. Lever A remains the
deployable headline; Lever B is now a documented negative result.

## What got done today

1. **Lever A validation pass** confirmed greedy `ρ=0.290`, oracle K=64
   `ρ=1.000` on the warm-start `ce_seq64_np16/best_av.pt`. This is the
   paper headline.
2. **Lever B v1 (hot hyperparams)**: temperature `1.5→0.7`, β_ctr=0.3,
   lr=3e-5. Launched as a full run; **collapsed**. Over ~2.4k steps the
   5-gram duplication on rollouts climbed `0.003 → 0.045` while training
   losses fell, and *both* greedy and oracle `ρ_cen` regressed past their
   warm-start values. Killed at step 2450.
3. **Lever B v2b (gentle hyperparams)**: temperature `1.5→1.2`,
   β_ctr=0.1, lr=1e-5, warmup=100, val-every=500, val-vectors=200,
   val-K=32, patience=3, batch=8, samples-per-v=32, hard-negs=8,
   `expandable_segments:True`, `out=artifacts/nla/bok_v2b_seq64_np16`.
   Two vals before kill:
   - step 500: greedy `ρ=0.321`, oracle `ρ=0.854`, 5gram_dup=0.002 →
     locked as `best_av.pt`.
   - step 1000: greedy `ρ=0.312`, oracle `ρ=0.803` (no improvement, 1/3).
   Killed at ~step 1040 to lock step-500 best. Plateau, no collapse.
4. **Negative result documented** in:
   - `paper_nla.md` §6 Implications (last bullet) — writes up both v1 and
     v2b regimes and the "winner-CE on K rollouts ≠ paraphrase manifold"
     reading.
   - `paper_nla.md` §8 Artifacts — adds bok_v2b paths, notes no separate
     HF revision is being released.
   - `README.md` plain-English block — short Lever B addendum.
   - `docs/hf/nla_v1_demo/README.md` plain-English block — same addendum.
5. **Paper §7 "Related work and positioning" added** — Patchscopes/SelfIE,
   vec2text, MBR best-of-N, STaR/ReST/RFT/seq-KD, probing/mech-interp,
   computational semiotics. Defensible narrow claim: first system to
   commit Peircean primitives to measurement on a frozen production-scale
   LLM with a calibrated round-trip metric. Old §7/§8 renumbered to §8/§9.
6. **Local artifacts pulled** via scp from Blackwell:
   `artifacts/nla/bok_v2b_seq64_np16/{best_av.pt (51.6MB), train_log.jsonl,
   val_text_step000500.jsonl, val_text_step001000.jsonl}`.

## Open items

- **HF release decision**: skip a v2b revision. `RiverRider/srt-nla-av-v1`
  (warm-start) remains canonical. v2b's `best_av.pt` is within sampling
  noise of v1.
- **Instance teardown**: Blackwell box can be killed any time now. No
  jobs running. (`pgrep -f train_nla` returns nothing as of kill.)
- **Branch state**: all changes from today are local working-tree edits,
  not yet committed. Files modified:
  - `paper_nla.md` (§6 Lever B, §7 added, §8/§9 renumbered)
  - `README.md` (plain-English block extended)
  - `docs/hf/nla_v1_demo/README.md` (plain-English block extended)
  - `SESSION_HANDOFF.md` (this file)
  - `artifacts/nla/bok_v2b_seq64_np16/*` (new, untracked)
- **Merge `nla → main`**: hold until paper finalization.

## Numbers (current, post-Lever-B)

| Decoding | greedy ρ_norm | oracle K=32 ρ_norm | oracle K=64 ρ_norm |
| --- | --- | --- | --- |
| CE-only warm-start (v1) | 0.29 | ~0.85 | **1.00** |
| BoK v1 (hot, collapsed) | regressed | regressed | n/a |
| BoK v2b step 500 (best) | 0.32 | 0.85 | n/a |
| BoK v2b step 1000 | 0.31 | 0.80 | n/a |
| Paraphrase ceiling | 1.00 | 1.00 | 1.00 |

Lever A (oracle K=64) stays the deployable headline. Lever B yields
+0.03 greedy ρ over the warm-start at best, no oracle benefit.

---

# SRT-Adapter NLA — Session Handoff (2026-05-17, end of day)

Vast.ai A6000/Blackwell instance is being **spun down** tonight. All committed
code is on `origin/nla`. All eval artifacts/logs from today are mirrored to
local `artifacts/nla/` and `logs/`. Large checkpoints (`.pt`) remain only on
remote — see "If you spin up a fresh instance" below.

---

## What got done today

1. **BoK distillation trainer (`scripts/train_nla_bok.py`)** — written, smoke-
   ran, **diagnosed as inadequate at K=4**. Teacher signal (win-of-4) ≈ greedy
   baseline → CE on teacher = noise → val regressed (greedy_cen 0.28 → 0.17 in
   normalized units). Committed `f8e34a4`, device-fix `892b6b9`.
2. **K-curve / cheap-rerank eval (`scripts/rerank_eval.py`)** — built and ran
   on warm-start `ce_seq64_np16/best_av.pt`, M=200, K=64. Committed `8cfb357`
   (with h_last extraction fix).
3. **Methodology reconciled** with `scripts/centered_eval.py`. Both scripts now
   agree to within sampling noise on the same ckpt:
   `greedy_cen ≈ 0.59`, `best-of-64_cen ≈ 0.78`, `nn_baseline_cen ≈ 0.72`.

## Today's results in normalized `ρ_norm` units (the paper's units)

`ρ_norm = (cen − 0.510) / 0.289`, where 0.510 = random_cen floor and
0.799 = paraphrase_cen ceiling.

| quantity | raw `cen` | `ρ_norm` |
| --- | --- | --- |
| greedy (T=0) | 0.586 | **0.26** |
| sampled (T=1) | 0.582 | 0.25 |
| best-of-2 (oracle) | 0.617 | 0.37 |
| best-of-4 | 0.652 | 0.49 |
| best-of-8 | 0.686 | 0.61 |
| best-of-16 | 0.716 | 0.71 |
| best-of-32 | 0.747 | 0.82 |
| best-of-64 | 0.777 | **0.92** |
| NN-retrieval baseline | 0.715 | 0.71 |
| logp-rerank (cheap) | 0.561 | 0.18  (HURTS greedy by −0.08) |
| nn-anchor-rerank | 0.722 | 0.73 |
| paraphrase ceiling | 0.799 | 1.00 |

**Key diagnostics:**

- **K-curve is log-linear**, +0.030 raw / +0.10 norm per doubling of K. K=64
  reaches `ρ_norm = 0.92`; extrapolation suggests K≈256 needed to hit ceiling.
- **Spearman(mean-logp, oracle-cen) per target: mean 0.04, p50 0.05.**
  The policy's own sequence log-prob has essentially **zero ranking power**
  over centered-fve. Logp-rerank actively hurts greedy. *Any value-head
  reranker using logp features is dead on arrival.*

## Tomorrow's open question

**How do we close the greedy → best-of-K gap (`ρ_norm` 0.26 → 0.92)?**
The four levers that survive today's evidence, ranked by promise:

### A. Best-of-K with oracle scoring at deploy time *(the trivial answer)*
- `v` is provided at inference, so we can compute centered-fve cheaply
  (one batched backbone forward over K candidate rollouts, same compute as
  K-way sampling). This **already gives `ρ_norm` 0.92** today, no retraining.
- Cost: K× sampling + 1× batched scoring forward.
- **Recommendation**: report this in the paper as the deployable decoding
  method. The "greedy" number is a misleading metric when oracle reranking
  is free in our setup.

### B. Policy improvement via BoK distillation — **needs K≫4**
- Today's smoke ran K=4 → teacher = noise. We need K such that
  `E[best-of-K_cen] ≫ greedy_cen` by enough to be a real teacher.
- From the K-curve: K=8 gives +0.10 raw over greedy, K=16 gives +0.13. So
  **launch at K=16 minimum, K=32 preferred**.
- Memory budget: K=32 × seq_len=64 × batch=16 = 32k rollout tokens per
  step. On Blackwell 96GB with bf16 frozen 7B this is borderline tight; may
  need `--batch=8` and `--samples-per-v=32`. **Profile first.**
- Add temp annealing 1.5 → 0.7 over training.
- Hard-neg InfoNCE term (J=8 NN from pool) is cheap to keep; α_bok=1.0,
  β_ctr=0.3, γ_act=0 to start.

### C. Process-reward / value-guided beam search
- Train a head that predicts `final ρ_cen` given a partial prefix +
  `v`. Beam-search with `score = logp + λ · V(prefix, v)`.
- **Caveat**: cheap features (logp trajectory only) won't work — Spearman
  ≈ 0.04 already proves this. Head MUST consume the prefix's hidden state
  at layer L, which is the same compute path as just sampling + scoring.
  So this is no cheaper than (A). Skip unless we find a feature with real
  signal.

### D. Direct activation matching on greedy rollouts
- Currently γ_act=0 (no L2 between rollout h_last and v). Cranking
  γ_act ≈ 0.5 with greedy decoding (no sampling) might pull greedy h_last
  toward v directly. Cheap to try; risks collapse.

**My recommendation for the first run tomorrow**: pure Option A as the paper's
headline decoding result, then launch Option B with K=32 as the policy
improvement experiment.

---

## If you spin up a fresh instance

Remote state to recreate on a new vast.ai box (`/workspace/srt-adapter`):

1. `git clone https://github.com/space-bacon/SRT.git && cd SRT && git checkout nla`
2. `python -m venv .venv && source .venv/bin/activate && pip install -e .`
3. Re-generate (or pull from S3 / re-upload from another box) the targets file:
   `artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt` (~2GB).
   Producer: `scripts/build_targets.py` (or whichever; check repo).
4. Warm-start checkpoint to use: `artifacts/nla/ce_seq64_np16/best_av.pt`
   (12.7M params). **This was only on the spun-down box.** Either:
   - retrain from scratch via `scripts/train_nla_ce.py --num-prefix-tokens 16
     --seq-len 64 --steps ~30k` (~6h on Blackwell), OR
   - if you saved it to S3 / scp'd locally before spindown, restore.
5. Standing flag rule: **always pass `--max-val-samples 5000`** to any
   `scripts/train.py` invocation (5k is fine; 100k wastes hours/pass).

## Files of interest

- `scripts/rerank_eval.py` — today's K-curve / logp-rerank / NN-rerank script.
- `scripts/train_nla_bok.py` — BoK trainer (untuned; relaunch with K≥16).
- `scripts/centered_eval.py` — canonical eval (greedy / best-of-K / NN /
  random, raw + centered).
- `paper_nla.md` — current paper draft. Numbers in §3-§4 are correct in
  `ρ_norm` units; check that no claim implies "ceiling reached at greedy".
- `artifacts/nla/rerank_eval_ce_seq64_np16_v2.json` — today's full result.
- `artifacts/nla/centered_eval_30k_M200.json` — sibling result on `_30k` ckpt,
  matches within noise.
- `logs/rerank_eval_v2.log`, `logs/bok_smoke.log` — today's run logs.

## What NOT to do tomorrow

- **Don't** rerun K=4 BoK. It demonstrably noises out the teacher.
- **Don't** build a value head whose features are logp only. Spearman 0.04.
- **Don't** retrain just to re-measure greedy on the same ckpt — the
  rerank_eval ≡ centered_eval numbers are now triangulated and trustworthy.

---

## Last commits on `nla`

```
8cfb357 nla: rerank_eval — fix h_last extraction (prefix-free forward...)
05c3b4c nla: rerank_eval.py — K-curve + logp-rerank + NN-anchor-rerank...
892b6b9 nla: fix build_hardneg_index device mismatch
f8e34a4 nla: train_nla_bok.py — best-of-K self-distill + NN hard-neg...
573f51e nla: reframe — centered metric, oracle baselines, paper draft
```
