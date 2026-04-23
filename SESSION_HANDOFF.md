# SRT-Adapter — Session Handoff (April 23, 2026)

> **Purpose:** Single document handing off all live state to a fresh VS Code
> window opened on `/Users/burtron/development/srt-adapter`. Read this top to
> bottom before doing anything.

---

## 1. Where we are

**v5 training is COMPLETE.** Finished 08:02 local, 23 April 2026 after
20K steps / 12.2 h on vast.ai A6000. Warm-started from v4 step-94K with
`--reset-community` to drop the collapsed community head, then trained
with the SupCon fix described in §4.

**v5 is the first checkpoint with non-degenerate community geometry**:

| Metric | v3 baseline | v4_step6000 | **v5_step17000** | random |
|---|---|---|---|---|
| within/between cosine ratio | 1.0009 | 1.0002 | **1.0050** | — |
| 35-class recall@1 | 0.0495 | 0.0535 | **0.3595** | 0.0286 |
| probe diversity (20 inputs) | 1 class | 1 class | **9 classes** | — |
| val CE | ~2.74 | ~2.74 | **2.734** | — |

recall@1 is **7.3× v3 baseline and 12.6× random**. CE val is unchanged —
SupCon pulled the community head out of its degenerate basin without
hurting language modeling.

**The actual root cause of the v3/v4 community collapse was a data
pipeline bug**, not the architectural diagnosis we acted on first. See
§4.4 and §12.

---

## 2. Remote GPU connection

```bash
# vast.ai A6000 48GB
ssh -p 30761 root@209.137.198.14

# Remote layout
/root/srt-adapter/                              # working tree (NOT a git repo)
  scripts/                                      # train.py, launch_v5.sh, instrument_eval.py, render_heatmap.py, probe_text.py
  srt/                                          # mirror of local /Users/burtron/development/srt-adapter/srt/
  data/
    all_train.jsonl                             # 1M samples
    all_val.jsonl                               # 100K samples
    curated_passages.jsonl                      # 100 passages for G3a 3.1.5
  probes/inputs.txt                             # 20 contested-topic passages
  checkpoints/
    adapter_v3/best_adapter.pt                  # historical baseline
    adapter_v4/best_adapter.pt                  # warm-start source for v5
    adapter_v5/                                 # v5 final state
      best_adapter.pt                           # step 17000 (val total 6.4679)
      adapter_epoch1.pt                         # step 20000 epoch 1 final
      final_adapter.pt                          # === adapter_epoch1.pt
      training_checkpoint.pt                    # full opt/sched state
      train_log.jsonl                           # per-step + per-val
      adapter_v5_stdout.log
      config.json
  artifacts/
    instrument/v5_step17000/{community_metrics.json,community_vectors.npz,traces.jsonl}
    probes/v5_step17000.html
```

**Sync code from local → remote** (remote has no git):
```bash
cd /Users/burtron/development/srt-adapter
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  scripts/<file>.py \
  root@209.137.198.14:/root/srt-adapter/scripts/
```

---

## 3. Two repos, easy to confuse

| Local path | Git remote | What it is |
|---|---|---|
| `/Users/burtron/development/srt-adapter` | `space-bacon/SRT` | Adapter, modules, training loop. **You are here.** |
| `/Users/burtron/development/SRT` | `space-bacon/Semiotic-Reflexive-Transformer` | Original SRT — paper, roadmap, theory docs, substack drafts. |

**Heads-up:** `/Users/burtron/development/SRT/src/srt/` contains a separate
pydantic-based config with the same module names as our `srt/` package.
Pylance from the SRT workspace can resolve adapter imports against the
wrong package. Open the srt-adapter folder directly to avoid.

---

## 4. v5 architectural changes (vs v4)

### 4.1 SupCon on encoder output, not mixed vector — `srt/modules/community.py`, `srt/training/losses.py`
- `CommunityOutput` now carries an `encoded` field — the bijective
  pre-mixing image of the input.
- `compute_total_loss` passes `output.community_output.encoded` (not
  `.vector`) to `community_supcon_loss`.
- **Why:** `vector = weights @ prototypes` becomes constant across the
  batch when `weights` collapses to one prototype, killing SupCon
  gradients by symmetry. The encoder output is per-sample and always
  varies.

### 4.2 SupCon weight bump — `srt/config.py`
- `community_supcon_weight: 0.5 → 2.0`. With the encoder fix landing,
  the loss now has signal worth amplifying.

### 4.3 Community-head reset on warm-start — `scripts/train.py`
- New CLI flag `--reset-community`. When set, the warm-start
  `drop_prefixes` filter additionally drops `community_head.*`.
- v5 launched with this flag, so the v4 collapsed community-head weights
  were discarded and reinitialized.

### 4.4 Data pipeline field-name fix — `srt/data/dataset.py` ⚠ **the actual bug**
- v3/v4 `dataset.py` read `community_str = row.get("community", "") or ""`,
  but the field is named `community_id` (already an int) — the
  `community` key didn't exist.
- 100% of training rows hashed empty-string → same id → SupCon saw
  one unique class per batch → the loss converged analytically to
  `log(B-1) = 2.71` with zero gradient.
- Fix:
  ```python
  if "community_id" in row and row["community_id"] is not None:
      community_id = int(row["community_id"])
  else:
      for fld in ("community_label", "source", "community"):
          community_str = row.get(fld) or ""
          if community_str:
              break
      community_id = _stable_hash(community_str)
  ```
- Diagnostic confirmation post-fix: `comm_supcon_unique_classes ≈ 13.0`
  per batch of 16, `pos_pairs ≈ 6.8`. Healthy distribution every step.

### 4.5 Diagnostic counters — `srt/training/losses.py`
- `community_supcon_loss` now returns `tuple[Tensor, dict]` with
  `pos_pairs` and `unique_classes` per batch.
- Logged as `comm_supcon_pos_pairs`, `comm_supcon_unique_classes`.
- **These counters are what caught the data bug.** Always log them for
  any future grouping/contrastive loss.

### 4.6 Launch hardening — `scripts/launch_v4.sh`, `scripts/launch_v5.sh`
- Both now `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  before invoking train.py. Avoids first-backward OOM on relaunch
  (memory fragmentation on A6000 + Qwen-7B + bf16 at batch=16/seq=512).
- Both pass `--max-val-samples 5000`. Without this, validation runs over
  the full 100K val set — ~2 h per pass, days of wasted wall-clock.

---

## 5. v5 launch command (reference)

```bash
# scripts/launch_v5.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data data/all_train.jsonl \
    --val-data data/all_val.jsonl \
    --max-val-samples 5000 \
    --output-dir checkpoints/adapter_v5 \
    --warm-start checkpoints/adapter_v4/best_adapter.pt \
    --reset-community \
    --batch-size 16 --epochs 1 \
    --lr 1e-4 --warmup-steps 500 --max-seq-len 512 \
    --val-every 1000 --log-every 100 --grad-clip 1.0 --dtype bfloat16
```
- 20K steps, ~0.5 step/s, ~12 h wall on A6000.
- Cosine LR anneal to ~0 by step 20000.

---

## 6. v5 results

### 6.1 Val curve (every 1000 steps, smoothed across 5K val samples)

| step | total | ce | comm_supcon |
|---|---|---|---|
| 1000 | 7.685 | 2.733 | 2.346 |
| 3000 | 6.900 | 2.738 | 1.946 |
| 5000 | 6.868 | 2.766 | 1.881 |
| 7000 | 6.610 | 2.738 | 1.806 |
| 10000 | 6.521 | 2.737 | 1.764 |
| 13000 | 6.478 | 2.734 | 1.747 |
| **17000** | **6.468** | 2.734 | 1.742 |
| 20000 | 6.468 | 2.734 | 1.742 |

- **comm_supcon dropped 25.7% over training** (2.346 → 1.742). v4 was
  pinned at log(B-1)=2.71 for 6300 steps.
- **CE held steady at 2.734.** No language-modeling regression from the
  contrastive pressure.
- **Convergence by ~step 14000.** Last 6K steps moved val total by 0.006.
  Best checkpoint (used for all evals): step 17000.

### 6.2 Instrument eval (`scripts/instrument_eval.py`)

Headline numbers in §1 table. Files at:
```
artifacts/instrument/v5_step17000/community_metrics.json
artifacts/instrument/v5_step17000/community_vectors.npz
artifacts/instrument/v5_step17000/traces.jsonl
artifacts/instrument/v5_step17000/heatmap.html       # rendered locally
artifacts/instrument/v3/heatmap.html                  # baseline for comparison
```

### 6.3 Probes (`scripts/probe_text.py`)

20 contested-topic passages (vaccine, border, trans, freedom, climate,
CRT, election, gender, Israel, Fed × factual/charged registers) at
`probes/inputs.txt`. v5 predicts **9 distinct community ids** across
the 20 inputs. v4 predicted id=16 for all 20.

```
artifacts/probes/v5_step17000.html
```

---

## 7. Tier-1 differentiator status

Roadmap in `/Users/burtron/development/SRT/docs/RESEARCH_ROADMAP.md` §13.

**Tier 1**
- A. Token-level discourse-phase monitoring — *not started*
- B. Counterfactual community decoding — *unblocked by v5 separation*
- C. Hallucination ≡ supercritical token — *not started*

**Tier 2**
- D. r̂ heatmap visualizer — ✓ shipped, v3+v5 heatmaps rendered
- E. Community-vector retrieval eval — ✓ shipped, v5 passes (recall@1=0.36)
- F. Calibration curve for regime head — *unblocked*
- G. RRM `gamma_proj` std init sweep — *queued*

**Tier 3 (research-grade)**
- H. Multi-perspective RAG — *unblocked by B*
- I. MAH as mixture-of-readers
- J. Cross-community paraphrase detection benchmark

**Three loss-design wins still open (from v4 review)**
1. Expose `chain_residual_per_token` as a 4th uncertainty channel.
2. Apply SupCon to MAH divergence (mirror of community SupCon, with the
   v5 "use the encoder output not the mixed vector" lesson baked in).
3. Add listwise ranking loss to BEN (ListNet on r̂ within batch).

---

## 8. Execution sequence (next steps)

1. **Commit v5 changes** to space-bacon/SRT main (this session, queued).
2. **Tier-1 B** (counterfactual community decoding): with the community
   space now meaningfully discriminative, decode with each of the 35
   community vectors as a soft prefix and measure perspective drift on
   contested prompts.
3. **Tier-1 C** (hallucination probe): regress hallucination labels on
   max-`r̂` and max-`chain_residual` over TruthfulQA / HaluEval / SimpleQA.
4. **Tier-2 win #2** (SupCon on MAH divergence): reuse the
   `community_supcon_loss` pattern. Apply to encoder output, not
   the mixed vector — same lesson as v5.

---

## 9. Why we cannot run anything heavy locally

- M2 Ultra has no Qwen 2.5-7B cached (15 GB download).
- v3/v4/v5 adapter weights are tiny (~25 MB) but useless without the
  backbone.
- Decision: all training + instrument runs on the GPU box. Local
  machine is for code editing, smoke testing, and rendering HTML.

---

## 10. Recent commits (both repos)

`space-bacon/SRT` (the adapter):
- `1e1c861` tier-1 D+E: instrument_eval + render_heatmap
- `f67a4be` v4: fix BEN tanh saturation, RRM dead inject, community collapse
- `eb0a7cb` (v3 docs)
- *uncommitted at handoff:* v5 = dataset.py field bug fix + SupCon on
  encoded + reset community head + diagnostic counters + launch
  hardening + probe_text.py + probes/inputs.txt

`space-bacon/Semiotic-Reflexive-Transformer` (paper/roadmap):
- `64b81df` roadmap: v4 differentiator section
- `8aa32e1` v3 colleague email + substack note
- `c77031c` (prior substack draft)

---

## 11. Useful one-liners

```bash
# GPU status
ssh -p 30761 root@209.137.198.14 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# Tail a training log
ssh -p 30761 root@209.137.198.14 'tail -F /root/srt-adapter/checkpoints/adapter_v5/adapter_v5_stdout.log'

# Sync a single edited file to remote
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  srt/training/losses.py \
  root@209.137.198.14:/root/srt-adapter/srt/training/

# Pull back artifacts from a remote eval run
rsync -avz -e "ssh -p 30761" \
  root@209.137.198.14:/root/srt-adapter/artifacts/instrument/ \
  artifacts/instrument/

# Render heatmap locally (no GPU needed)
python3 scripts/render_heatmap.py \
  --traces artifacts/instrument/v5_step17000/traces.jsonl \
  --out artifacts/instrument/v5_step17000/heatmap.html \
  --title "SRT-Adapter v5 step 17000"

# Find live training PIDs on the box
ssh -p 30761 root@209.137.198.14 'pgrep -af "scripts/train.p[y]"'

# Safely kill remote training (avoid pkill self-kill — see §12)
ssh -p 30761 root@209.137.198.14 'kill -TERM $(pgrep -f "scripts/train.p[y]" | head -1)'
```

---

## 12. Things that have bitten us (avoid repeating)

- **`row.get("community", "")` silently masked a missing field for 3
  model versions.** Spent ~6300 v4 training steps and an architectural
  refactor diagnosing what was actually a one-line dataset bug. Lesson:
  for any grouping/contrastive loss, log per-batch counts (`pos_pairs`,
  `unique_classes`) from day one. A degenerate label source looks
  identical in aggregate metrics to a healthy-but-slow-learning loss
  until you count.
- **SupCon on a convex combination is dead when the assignment
  collapses.** `vector = weights @ prototypes` becomes constant across
  the batch → gradient zero by symmetry. Apply contrastive losses to
  the encoder output (pre-mixing), which is bijective wrt the input.
- **Warm-starting from a collapsed module locks you in.** SupCon cannot
  escape a basin where every input maps to the same output. Drop and
  reinit the offending state-dict prefix at warm-start.
- **`pkill -f "scripts/train.py"` from inside an SSH command can kill
  its own shell** (pattern matches the SSH'd shell). Use
  `kill -TERM $(pgrep -f "scripts/train.p[y]" | head -1)`. The `[y]`
  trick prevents grep from matching itself.
- **CUDA fragmentation.** A6000 + Qwen-7B + bf16 at batch=16/seq=512
  occasionally OOMs on first backward after relaunch. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in launch script.
- **Always pass `--max-val-samples 5000`.** Without it, validation runs
  over the full 100K val set (~2 h per pass).
- `0 * -inf = NaN` in SupCon mask; mask to 0 *before* multiplying by
  `pos_mask`. Already fixed.
- `nn.Tanh()` on a regression head whose targets exceed ±1 silently caps
  predictions at the rail. (BEN v3.) Always plot pred-vs-target scatter
  early.
- Norm-regularizers (`inject_reg`, `divergence_alive`) that force a
  vector's magnitude without selecting its direction will satisfy
  themselves with arbitrary directions and contribute zero useful
  signal. Prefer contrastive/ranking losses over magnitude losses.
- Pylance from the `/Users/burtron/development/SRT` workspace resolves
  adapter imports against the wrong (pydantic) `srt/` package and
  produces spurious errors in `scripts/train.py`. Open the srt-adapter
  folder directly.
- Warm-starting requires `strict=False` and explicit prefix-drop for
  removed/reset keys. See `scripts/train.py` `drop_prefixes` logic
  around line 240.
# SRT-Adapter — Session Handoff (April 22, 2026)

> **Purpose:** Single document handing off all live state to a fresh VS Code
> window opened on `/Users/burtron/development/srt-adapter`. Read this top to
> bottom before doing anything.

---

## 1. Where we are

**v4 training is LIVE on vast.ai A6000.** Launched 13:35 local time, 22 April
2026. Warm-started from v3 step-94K checkpoint with three architectural
fixes (BEN tanh, RRM FiLM, community SupCon). At step 300 it was already
showing the fixes were working (r̂ range [-0.09, 3.00] vs v3's
tanh-clipped [-0.21, 0.77], `inj_norms` [10.3, 22.3] vs v3's [0.23, 0.65]).

Current focus: **Tier-1 differentiator demonstrations.** D + E shipped
(see §6); A / B / C / F still queued. Roadmap in
`/Users/burtron/development/SRT/docs/RESEARCH_ROADMAP.md` §13.

---

## 2. Remote GPU connection

```bash
# vast.ai A6000 48GB, idle when not training
ssh -p 30761 root@209.137.198.14

# Remote layout
/root/srt-adapter/                              # working tree (NOT a git repo)
  scripts/                                      # train.py, launch_v4.sh, instrument_eval.py, render_heatmap.py
  srt/                                          # the package (mirror of local /Users/burtron/development/srt-adapter/srt/)
  data/
    all_train.jsonl                             # 320K samples
    all_val.jsonl                               # 100K samples
    curated_passages.jsonl                      # 100 passages for G3a 3.1.5
  checkpoints/
    adapter_v3/best_adapter.pt                  # warm-start source (step 94K)
    adapter_v4/                                 # current run output
      training_checkpoint.pt                    # full state (model+opt+sched)
      best_adapter.pt                           # best by val
      adapter_v4_stdout.log                     # tail this
      training_log.jsonl                        # per-step metrics
```

**Sync code from local → remote** (pattern used throughout this session;
remote has no git):
```bash
cd /Users/burtron/development/srt-adapter
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  scripts/<file>.py \
  root@209.137.198.14:/root/srt-adapter/scripts/
# Repeat for srt/<subdir>/<file>.py against the matching remote dir.
```

**Tail the live training log:**
```bash
ssh -p 30761 root@209.137.198.14 'tail -f /root/srt-adapter/checkpoints/adapter_v4/adapter_v4_stdout.log'
```

---

## 3. Two repos, easy to confuse

| Local path | Git remote | What it is |
|---|---|---|
| `/Users/burtron/development/srt-adapter` | `space-bacon/SRT` (yes, that's the name) | The actual training code — adapter, modules, training loop. **You are here.** |
| `/Users/burtron/development/SRT` | `space-bacon/Semiotic-Reflexive-Transformer` | The original SRT project — paper, roadmap, theory docs, substack drafts. The adapter was **transformed out of** this repo. |

**Heads-up:** `/Users/burtron/development/SRT/src/srt/` contains a *separate*
pydantic-based config with the same module names as our `srt/` package.
Pylance from the SRT workspace can resolve our adapter imports against the
wrong package and produce confusing false errors (`SRTConfig has no attribute
'loss'`, etc.). Working from a clean srt-adapter window avoids this.

---

## 4. v4 architectural changes (vs v3)

All in commit `f67a4be` on `space-bacon/SRT` main.

### 4.1 BEN — `srt/modules/ben.py`
- Removed final `nn.Tanh()` from `r_head` Sequential.
- Added explicit `r_out: nn.Linear = self.r_head[-1]` reference and
  initialized `weight ~ N(0, 0.02²)`, `bias = 0`.
- **Why:** v3 r̂ was clamped to [-1, 1] but log-compressed targets reach
  2.55. 32% curated / 17% val tokens were pinned at +1.

### 4.2 RRM — `srt/modules/rrm.py`
Old (v3, dead):
```python
self.inject_proj = nn.Linear(d_meta, d_backbone, bias=False)  # zero-init
self.inject_gate = nn.Sequential(nn.Linear(d_meta, 1), nn.Sigmoid())
# inject() returned inject_gate(meta) * inject_proj(meta), starts identically 0
```
New (v4, FiLM):
```python
self.gamma_proj = nn.Linear(d_meta, d_backbone)   # weight std=0.02, bias=0
self.beta_proj  = nn.Linear(d_meta, d_backbone)   # weight=0,        bias=0
# inject() returns (h * (1 + gamma) + beta) * inject_scale
```
- Identity in expectation at init; gradient flows from step 0.
- `inject_scale: 0.1 → 1.0`
- `inject_reg_weight: 0.5 → 0.0` (the v3 norm regularizer was driving
  arbitrary directions orthogonal to gradient signal — proven dead by
  ablation that showed zero perplexity change when injection was zeroed).

### 4.3 Community SupCon — `srt/training/losses.py`
- New function `community_supcon_loss(community_vectors, community_ids, temperature=0.1)`
  implementing Khosla 2020 supervised contrastive on per-sample community
  vectors. Includes the `0 * -inf = NaN` fix on the diagonal mask.
- Wired into `compute_total_loss` behind `community_ids is not None`.
- New `LossConfig` fields: `community_supcon_weight=0.5`,
  `community_supcon_temperature=0.1`.
- `srt/data/dataset.py`: emits `community_id` per sample via FNV-1a 32-bit
  hash of source-subreddit string mod 100003 (function `_stable_hash`).
- `scripts/train.py`: passes `community_ids=batch.get("community_id")` into
  `compute_total_loss` in both validate() and main loop.

### 4.4 Warm-start — `scripts/train.py`
- New CLI arg `--warm-start <path>` (separate from `--resume`).
- Drops keys matching `("rrm.inject_proj", "rrm.inject_gate")`,
  loads with `strict=False`.
- Verified live at v4 launch:
  > loaded 37 v3 tensors, dropped 3 v3-only, reinitialized 4 v4-only
  > (`rrm.gamma_proj.{weight,bias}`, `rrm.beta_proj.{weight,bias}`)

---

## 5. v4 launch (running right now)

```bash
# scripts/launch_v4.sh
python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data data/all_train.jsonl \
    --val-data data/all_val.jsonl \
    --output-dir checkpoints/adapter_v4 \
    --warm-start checkpoints/adapter_v3/best_adapter.pt \
    --batch-size 16 --epochs 1 --max-train-samples 320000 \
    --lr 1e-4 --warmup-steps 500 --max-seq-len 512 \
    --val-every 1000 --log-every 100 --grad-clip 1.0 --dtype bfloat16
```
- 20K steps total (1 epoch × 20K batches).
- LR is **lower** than v3's 3e-4 because most weights are warm.
- ~0.5 step/s → ~11 h wall time.
- Watch: if `inj_norms` blow past ~50, may need to revisit
  `inject_scale=1.0`. At step 300 they were 22; sub-50 is fine.

---

## 6. Tier-1 D + E (just shipped, commit `1e1c861`)

Two scripts in `scripts/` — both import-tested on the GPU box:

### 6.1 `scripts/instrument_eval.py`
Single inference pass per checkpoint emits:
- `artifacts/instrument/<tag>/traces.jsonl` — one passage per line
  (tokens, r_hat, r_true, r_mask, regime_pred, community_id_true,
  community_pred). Drives the renderer.
- `artifacts/instrument/<tag>/community_vectors.npz` — raw per-sample
  vectors + true ids.
- `artifacts/instrument/<tag>/community_metrics.json` — within/between
  cosine, ratio, k-NN recall@1/5/10. The single number to compare v3 vs v4.

Smoke-tested in isolation with synthetic vectors:
| Mock setup | within/between ratio | recall@1 |
|---|---|---|
| Collapsed (v3-like)   | 1.000 | 0.10 (random over 10 cls) |
| Separated (v4 target) | 116   | 1.00 |

Run pattern (planned, not yet executed):
```bash
# When v4 hits step 1000, on GPU box:
python scripts/instrument_eval.py \
    --adapter checkpoints/adapter_v3/best_adapter.pt \
    --val-data data/all_val.jsonl --tag v3 \
    --max-samples 2000 --trace-samples 60

python scripts/instrument_eval.py \
    --adapter checkpoints/adapter_v4/best_adapter.pt \
    --val-data data/all_val.jsonl --tag v4_step1000 \
    --max-samples 2000 --trace-samples 60
```

### 6.2 `scripts/render_heatmap.py`
Self-contained HTML, no JS, no external CSS. Per-token color = r̂ on
blue→white→red (R_MIN=-1.5, R_MAX=3.0). Hover any token for
r̂/r_true/regime tooltip.
```bash
python scripts/render_heatmap.py \
    --traces artifacts/instrument/v4_step1000/traces.jsonl \
    --out    artifacts/instrument/v4_step1000/heatmap.html \
    --title  "SRT-Adapter v4 (step 1000) — per-token r̂"
```

---

## 7. The roadmap (v4 differentiators)

Full version in
`/Users/burtron/development/SRT/docs/RESEARCH_ROADMAP.md` §13. TL;DR
ranked by **(novelty × demonstrability) ÷ cost**:

**Tier 1 — bigger wins, modest cost**
- A. Token-level discourse-phase monitoring (`r̂` of "freedom", "vaccine"
  tracked weekly across longitudinal news; show subcritical →
  near-critical → supercritical around documented external events).
- B. Counterfactual community decoding (community vector as soft prefix at
  generation time).
- C. Hallucination ≡ supercritical token (regress hallucination labels on
  max-`r̂` and max-`chain_residual` over TruthfulQA / HaluEval / SimpleQA;
  AUROC > 0.7 = SOTA training-free hallucination detector).

**Tier 2 — easier wins**
- D. r̂ heatmap visualizer ✓ shipped
- E. Community-vector retrieval eval ✓ shipped
- F. Calibration curve for the regime head (was impossible in v3 due to
  tanh ceiling; v4 enables it).
- G. RRM `gamma_proj` std init sweep (0.02 → 0.05 → 0.1).

**Tier 3 — research-grade**
- H. Multi-perspective RAG (community-stratified evidence sets when query
  token has high r̂).
- I. MAH as mixture-of-readers (community = expert).
- J. Cross-community paraphrase detection benchmark.

**Three loss-design wins discovered during v4 review**
1. Expose `chain_residual_per_token` as a 4th uncertainty channel in
   `SRTAdapterOutput`. The `chain_loss` already computes it; we average
   to a scalar and discard. **No model in the world has a per-token
   Peirce-chain-break detector** because no model has the chain.
2. **Apply SupCon to MAH divergence** (mirror `community_supcon_loss`
   for divergence vectors, positives = same-token-same-community pairs).
   Highest-EV code change available — directly attacks the longest-failing
   G3a test (3.1.2/3.1.4: norms 1.05× contested vs neutral, need 2.0×).
   `divergence_alive_loss` is the same hack we just retired in RRM.
3. Add listwise ranking loss to BEN (e.g. ListNet on r̂ within batch).
   The dataset's signal is rank-shaped (~99% of `r_true` near zero);
   listwise loss converts dense calibration into the sparse "which tokens
   are more contested than which" question.

---

## 8. Execution sequence (next steps)

1. **At v4 step 1000 (~30 min after launch):** SSH in, run
   `instrument_eval.py` on v3 baseline + v4 step-1000 checkpoint, scp
   `community_metrics.json` files back. Apples-to-apples numbers for
   substack and as evidence the architectural fixes landed.
2. **At v4 step 5000:** Implement Tier-2 win #2 above (SupCon on MAH
   divergence). Pattern is identical to `community_supcon_loss`. New CLI
   flag, no warm-start invalidation.
3. **At v4 finish (~midnight):** Run hallucination probe (Tier-1 C).
   Negative result is still a paper.
4. **Paper-shaped pivot:** Reframe the academic paper around "the
   instrument and its three orthogonal channels" — `r̂`, divergence,
   community — with A/B/C as external-ground-truth validations of each.

---

## 9. Why we cannot run instrument_eval locally

- M2 Ultra has no Qwen 2.5-7B cached. Download is 15 GB.
- v3 weights are local at `artifacts/checkpoints/step94k/best_adapter.pt`
  (25 MB) but are useless without the backbone.
- Decision: do all instrument runs on the GPU box. Local machine is for
  code editing, smoke testing, and rendering HTML.

---

## 10. Recent commits (both repos)

`space-bacon/SRT` (the adapter, this folder):
- `1e1c861` tier-1 D+E: instrument_eval + render_heatmap
- `f67a4be` v4: fix BEN tanh saturation, RRM dead inject, community collapse
- `eb0a7cb` (v3 docs)

`space-bacon/Semiotic-Reflexive-Transformer` (paper/roadmap):
- `64b81df` roadmap: add v4 differentiator section (Tier 1/2/3 wins)
- `8aa32e1` docs: v3 colleague email + substack note; curated r_true regen script
- `c77031c` (prior substack draft)

---

## 11. Useful one-liners

```bash
# GPU status
ssh -p 30761 root@209.137.198.14 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# Tail v4 log
ssh -p 30761 root@209.137.198.14 'tail -f /root/srt-adapter/checkpoints/adapter_v4/adapter_v4_stdout.log'

# Sync a single edited file to remote
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  srt/training/losses.py \
  root@209.137.198.14:/root/srt-adapter/srt/training/

# Pull back artifacts from a remote eval run
rsync -avz -e "ssh -p 30761" \
  root@209.137.198.14:/root/srt-adapter/artifacts/instrument/ \
  artifacts/instrument/

# Verify no stray .mypy_cache (causes phantom errors in train.py)
find . -type d -name '.mypy_cache' -exec rm -rf {} +
```

---

## 12. Things that have bitten us (avoid repeating)

- `0 * -inf = NaN` in SupCon mask; always mask to 0 *before* multiplying
  by `pos_mask`. Already fixed in `community_supcon_loss`.
- `nn.Tanh()` on a regression head whose targets exceed ±1 silently caps
  predictions at the rail. (BEN v3.) **Lesson: always plot
  pred-vs-target scatter early.**
- Norm-regularizers (`inject_reg`, `divergence_alive`) that force a
  vector's magnitude without selecting its direction will satisfy
  themselves with arbitrary directions and contribute zero useful signal.
  **Lesson: prefer contrastive/ranking losses over magnitude losses.**
- Pylance from the `/Users/burtron/development/SRT` workspace resolves
  adapter imports against the wrong (pydantic) `srt/` package and produces
  spurious errors in `scripts/train.py`. Open the srt-adapter folder
  directly to avoid.
- Warm-starting requires `strict=False` and explicit prefix-drop for
  removed keys. Filter `("rrm.inject_proj", "rrm.inject_gate")` is in
  `scripts/train.py` lines ~239–265.
