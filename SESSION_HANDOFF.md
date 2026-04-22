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
