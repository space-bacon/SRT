# SRT-Adapter — Session Handoff (April 25, 2026)

> Single document handing off live state to a fresh VS Code window opened
> on `/Users/burtron/development/srt-adapter`. Read top-to-bottom before
> doing anything.

---

## 1. Where we are

**v8a is COMPLETE and shipped.** v8b is **TRAINING NOW** on vast.ai A6000
(launched 15:32 UTC, PID 292290 remote). ETA ~5h.

The headline arc since v7:

| | v6 | v7 | **v8a** | **v8b (target)** |
|---|---|---|---|---|
| VAL CE | 2.738 | 2.739 | 2.739 | preserve |
| Best val total | 9.117 | 9.0044 | 9.0040 | ≤ 9.004 |
| Reddit retrieval recall@1 (35 cls) | 0.395 | 0.413 | **0.484** | ≥ 0.484 |
| within/between cos ratio | 1.012 | 1.006 | **2.016** | ≥ 2.0 |
| Archetype recall@1 (33 cls, vs 0.030 chance) | 0.168 | 0.149 | **0.230** | ≥ 0.23 |
| Archetype centroid off-diag cos | 0.999 | 0.999 | **0.873** | < 0.85 |
| Trajectory anisotropy (λ_max/λ_min) | 52 | 72 | **23,333** | n/a |
| Hallucination AUROC mean_r̂ | — | 0.578 | 0.577 | preserve |

**v8a takeaway:** removing the discrete prototype basis (`use_prototypes=False`)
preserved CE while doubling Reddit retrieval ratio and lifting archetype
recall 54%. The §5.8 PCA finding (prototypes barely move from random init)
explained why the bottleneck was the binding constraint.

**v8b hypothesis:** community_supcon_weight 2.0→4.0, temperature 0.1→0.05
with everything else identical to v8a. Goal: orthogonalize archetype
centroids further (off-diag cos < 0.85) without disturbing token CE.

---

## 2. Remote GPU connection

```bash
# vast.ai A6000 48GB
ssh -p 30761 root@209.137.198.14
```

Remote layout (NOT a git repo — use rsync to sync):

```
/root/srt-adapter/
  scripts/train.py, launch_v8a.sh, launch_v8b.sh, instrument_eval.py,
          archetype_probe.py, trajectory_eval.py, eval_v8a.sh, ...
  srt/                                    # mirror of local srt/
  data/all_train.jsonl (1M), all_val.jsonl (100K), archetypes.json
  probes/inputs.txt                       # 20 contested-topic probes
  checkpoints/
    adapter_v6/best_adapter.pt            # baseline
    adapter_v7/best_adapter.pt            # warm-start source for v8a
    adapter_v8a/best_adapter.pt           # warm-start source for v8b
    adapter_v8b/                          # CURRENT TRAINING RUN
  artifacts/
    instrument/{v7_step6000,v8a_step10000}/community_metrics.json
    archetype_probe/{v7_step6000,v8a_step10000}/results.json
    archetype_probe/generations.jsonl     # 986 archetype-conditioned sentences
    trajectory/{v6_step12000,v7_step6000,v8a_step10000}/results.json
    hallucination/, regime_calibration/, context_conditional/, counterfactual/
    eval_v8a.log
```

Sync code local → remote:
```bash
cd /Users/burtron/development/srt-adapter
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  srt/<file>.py scripts/<file>.py \
  root@209.137.198.14:/root/srt-adapter/<dest>/
```

Tail v8b training:
```bash
ssh -p 30761 root@209.137.198.14 \
  'tail -f /root/srt-adapter/checkpoints/adapter_v8b/adapter_v8b_stdout.log'
```

---

## 3. Two repos, easy to confuse

| Local path | Git remote | What it is |
|---|---|---|
| `/Users/burtron/development/srt-adapter` | `space-bacon/SRT` | Adapter code, training loop, paper.md. **You are here.** |
| `/Users/burtron/development/SRT` | `space-bacon/Semiotic-Reflexive-Transformer` | Original SRT — theory docs, longer-form drafts. |

Pylance from the SRT workspace resolves `srt/` against a *different*
pydantic-based package with the same module names. Always open
srt-adapter directly.

---

## 4. v8a architectural change

`srt/config.py`, `srt/modules/community.py`, `srt/training/losses.py`,
`scripts/train.py` (committed in `2413838`):

- `CommunityConfig.use_prototypes: bool = True` (default unchanged for
  back-compat with v5/v6/v7 checkpoints).
- When `False`: `CommunityDiscoveryHead` skips the K-prototype
  embedding entirely. `CommunityOutput.weights` and `.logits` become
  `None`; `.vector == .encoded`.
- `community_entropy_loss` is guarded out (no soft assignment to
  entropize). `community_supcon_loss` is unchanged — it always operated
  on `.encoded`, which is the loss that was doing the work.
- `--no-prototypes` CLI flag in `scripts/train.py`.
- **`SRT_USE_PROTOTYPES=0` env override** (committed in `032e155`):
  any newly-constructed `SRTConfig` flips `use_prototypes=False`. Lets
  probe scripts run against v8a checkpoints without per-script flag
  plumbing.

Patches applied in `386e724` for trajectory-mode probes:
- `instrument_eval.py`: guards `community_pred` when `weights is None`.
- `counterfactual_decode.py`: explicit skip in trajectory mode (no
  discrete communities to enumerate; writes a marker html).

## 5. v8b launch (currently training)

```bash
# scripts/launch_v8b.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data data/all_train.jsonl --val-data data/all_val.jsonl \
    --output-dir checkpoints/adapter_v8b \
    --warm-start checkpoints/adapter_v8a/best_adapter.pt \
    --no-prototypes \
    --batch-size 16 --epochs 1 --max-train-samples 160000 \
    --lr 5e-5 --warmup-steps 250 --max-seq-len 512 \
    --val-every 1000 --max-val-samples 5000 --log-every 100 \
    --grad-clip 1.0 --dtype bfloat16 \
    --divergence-supcon-weight 0.3 --listnet-weight 0.5 \
    --chain-residual-aux-weight 0.05 \
    --community-supcon-weight 4.0 \
    --community-supcon-temperature 0.05
```

10K steps, ~0.5 step/s, ~5h on A6000. Checkpoints land in
`/root/srt-adapter/checkpoints/adapter_v8b/`.

## 6. v8b eval workflow (when training finishes)

The full v8a eval suite lives at `/root/srt-adapter/scripts/eval_v8a.sh`.
Adapt the same pattern for v8b — change `ADAPTER` and `TAG`, keep
`SRT_USE_PROTOTYPES=0`. The 7 stages are:

1. instrument_eval (Reddit retrieval recall@k, within/between cos)
2. counterfactual_decode (skipped in trajectory mode — writes marker)
3. hallucination_probe (TruthfulQA AUROCs)
4. regime_calibration (ECE, Brier)
5. context_conditional_r (per-token Δr̂)
6. archetype_probe (33-archetype recall@k, off-diag cos)
7. trajectory_eval (path length, log-det cov, anisotropy)

Already-existing comparison artifacts for v6/v7/v8a let v8b results
drop straight into the §5.9 paper table.

---

## 7. Repository state

Branch: `main`. Latest commits:

```
725a31d v8b: sharper community SupCon (warm-start from v8a)
386e724 v8a results: trajectory-mode probes + paper §5.9
032e155 config: SRT_USE_PROTOTYPES env override for CommunityConfig
2413838 v8a: continuous-trajectory community head (no prototypes)
02c6675 paper: §5.7 v6+v7 results, §5.8 archetype convergence + PCA finding
```

Latest GitHub release: `v7.0.0`. v8a and v8b deliberately NOT released
yet — release the winner of the v8 family (likely v8a or v8b depending
on v8b's archetype off-diag delta).

Untracked working-tree files (intentionally not in git): `artifacts/`,
`data/val_200.jsonl`, `scripts/benchmark.py`, `scripts/plot_artifacts.py`,
`srt/training/# Code Citations.md`.

---

## 8. Useful one-liners

```bash
# GPU status
ssh -p 30761 root@209.137.198.14 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# Sync edited file to remote
rsync -avz --no-perms --no-times -e "ssh -p 30761" \
  srt/training/losses.py root@209.137.198.14:/root/srt-adapter/srt/training/

# Pull artifacts back
rsync -avz -e "ssh -p 30761" \
  root@209.137.198.14:/root/srt-adapter/artifacts/ artifacts/

# Find live training PIDs
ssh -p 30761 root@209.137.198.14 'pgrep -af "scripts/train.p[y]"'

# Safely kill remote training (avoid pkill self-kill — see §9)
ssh -p 30761 root@209.137.198.14 \
  'kill -TERM $(pgrep -f "scripts/train.p[y]" | head -1)'

# Run a probe in trajectory mode without code changes
ssh -p 30761 root@209.137.198.14 \
  'export SRT_USE_PROTOTYPES=0 && python3 scripts/<probe>.py ...'
```

---

## 9. Things that have bitten us (avoid repeating)

- **Prototypes barely train** (the v7 PCA bombshell). Across v5/v6/v7
  the 32×64 prototype matrix moved by mean abs delta ~3e-5 against
  magnitudes of 0.5–1.5 — essentially still random init. The encoder
  was doing all the discriminative work and the soft-argmax readout
  was throwing it away. v8a removed the bottleneck; v8b is testing
  whether stronger SupCon completes the de-collapse.
- **SupCon on a convex combination is dead** when assignments collapse.
  Always apply contrastive losses to encoder output (pre-mixing).
- **`row.get("community", "")` silently masked a missing field for 3
  model versions** (v3/v4). Fixed in v5. Lesson: log per-batch counts
  (`pos_pairs`, `unique_classes`) for any grouping/contrastive loss.
- **`pkill -f "scripts/train.py"` from inside SSH can kill its own
  shell.** Use `kill -TERM $(pgrep -f "scripts/train.p[y]" | head -1)`
  — the `[y]` trick prevents grep self-match.
- **CUDA fragmentation** OOMs on first backward at relaunch. Always
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Always pass `--max-val-samples 5000`.** Without it, validation runs
  over the full 100K val set (~2h per pass).
- `0 * -inf = NaN` in SupCon mask; mask to 0 *before* multiplying.
- `nn.Tanh()` on a regression head whose targets exceed ±1 silently
  caps predictions at the rail (BEN v3 bug).
- Norm regularizers without direction selection contribute zero useful
  signal — prefer contrastive / ranking losses.
- Warm-starting requires `strict=False` and explicit prefix-drop for
  removed/reset keys. See `scripts/train.py` `drop_prefixes` logic.
- **Trajectory-mode probe scripts**: `instrument_eval`, `archetype_probe`,
  `counterfactual_decode`, `context_conditional_r` all need to handle
  `community_output.weights is None`. The first three are patched. Any
  new probe must guard.
