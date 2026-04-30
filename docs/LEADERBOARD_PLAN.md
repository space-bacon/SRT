# Leaderboard Plan: SRT-Adapter → MTEB

Last updated: 2026-04-30 (post-v12)

## Headline result (v12, 2026-04-30 17:04 UTC)

One epoch of InfoNCE on 396K public pairs (NLI + Quora), warm-started
from v8a. Adapter unchanged: frozen Qwen/Qwen2.5-7B + 14.5M trainable.

| Cohort | Mean Spearman (40 STS splits) | Best split | Notes |
|---|---:|---:|---|
| **v8a** (no contrastive) | **+0.210** | STS22.v2 +0.642 | Self-supervised reflexivity target only |
| **v12** (1ep contrastive) | **+0.346** | HUMESICK-R **+0.792** | warm-start from v8a |

Per-region readout (v12):

* **English STS:** HUMESICK-R 0.79, STS17 0.72, SemRel24 0.71,
  STSBenchmark 0.53, SICK-R 0.61, STS22 family 0.49–0.66.
* **English-only mean ≈ 0.55** (excluding the 12 IndicCrosslingualSTS
  splits we never trained on; those carry near-0 prior and drag the
  full mean down ~0.10).
* **Indic crosslingual:** -0.10 to +0.11 — expected, no Indic data in
  training set; would require multilingual mix.

Training trajectory was monotonically improving recall@1 every 1000
steps (0.111 → 0.206) with no plateau, indicating significant headroom.

Result artifacts: `artifacts/mteb/v12/summary.json` and
`artifacts/mteb/v8a/summary.json` on the A6000.

## TL;DR

Convert the existing SRT-Adapter (frozen Qwen/Qwen2.5-7B + ~12.7M
trainable params) into a **sentence embedding model** and submit to
[MTEB / MMTEB](https://huggingface.co/spaces/mteb/leaderboard).

This is the single highest-ROI use of the science we have:

* `community_output.encoded` is **already a per-sample dense vector**
  trained with supervised contrastive losses (community + archetype
  supcon). That is exactly what MTEB scores.
* The adapter is small, runs at 7B-class inference cost, and we can
  iterate in 1–3 day cycles on a single A6000.
* The novel losses (archetype supcon, divergence supcon, separatrix
  geometry) are unusual relative to the standard E5/BGE/GTE recipe and
  give us a defensible scientific story regardless of where we land in
  the rankings.

What we are **not** doing here:

* Chatbot Arena (frozen 7B base + 12.7M params has no realistic chat
  ceiling).
* Open LLM Leaderboard generative tasks (same reason — would require
  unfreezing into LoRA-FT or full SFT, which is a separate project).

## Why MTEB, why now

| Leaderboard | Fit to current adapter | Cost to compete | Decision |
| --- | --- | --- | --- |
| MTEB (English, 56 tasks) | **Direct** — embedding model is the artifact | 1× A6000, days | **GO** |
| MMTEB (multilingual) | Same as above, larger eval footprint | 1× A6000, ~week | Stretch goal |
| BEIR / BRIGHT | Subset of MTEB retrieval, narrower story | Free w/ MTEB run | Auto-included |
| Open LLM Leaderboard v2 | Poor — needs SFT/instruct + capability gains | Multiple GPU-weeks | Defer |
| Chatbot Arena | Poor — needs chat-tuned generative model | Months + human prefs | Defer |

## Pipeline

```
                 ┌─────────────────────────────┐
   v8a (current) │ frozen Qwen2.5-7B + adapter │
                 │ community_output.encoded    │ ← already an embedding
                 └──────────────┬──────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                                    ▼
   1. mteb_eval.py (baseline)       2. train_contrastive.py (v12)
      - run v8a, v9, v10, v11           - InfoNCE on E5/BGE pair mix
      - write artifacts/mteb/<ver>/     - keep archetype supcon as aux
      - report 56-task average          - warm-start from best of {v8a..v11}
              │                                    │
              └──────────────┬─────────────────────┘
                             ▼
                  3. mteb_eval.py (v12)
                     - confirm gain
                     - submit best checkpoint to MTEB leaderboard
```

## Concrete steps (next 2 weeks)

### Week 1 — measure where we stand

1. Install `mteb`, `sentence-transformers` on the A6000 venv.
2. Run [scripts/mteb_eval.py](../scripts/mteb_eval.py) against
   `checkpoints/adapter_v8a/best_adapter.pt`,
   `checkpoints/adapter_v9/best_adapter.pt`,
   `checkpoints/adapter_v10/best_adapter.pt`,
   `checkpoints/adapter_v11/best_adapter.pt`.
3. Subset first: `--task-types Classification,Clustering,STS` (these
   come almost free for our embedding shape and tell us if the
   geometry transfers at all). Full 56-task run only after that.
4. Compare the four checkpoints across MTEB and pick the best
   warm-start base for v12. Almost certainly v8a or v10.

Expected shape of the output (one row per task):

```
artifacts/mteb/v8a/results.json
artifacts/mteb/v8a/per_task.csv
artifacts/mteb/v8a/summary.json   # main_score per task, average
```

### Week 2 — train v12 contrastive

1. Pull the standard MTEB pretraining mix (start small, scale up):
   * `sentence-transformers/all-nli` (NLI triplets)
   * `sentence-transformers/msmarco-hard-negatives` (retrieval w/
     mined hard negatives)
   * `sentence-transformers/quora-duplicates` (paraphrase)
   * Optionally: `intfloat/query2doc_msmarco`,
     `BAAI/bge-large-en-v1.5-instruct` mix.
2. Run [scripts/launch_v12_mteb.sh](../scripts/launch_v12_mteb.sh)
   which calls [scripts/train_contrastive.py](../scripts/train_contrastive.py).
   Default config:
   * Warm-start from the week-1 winner.
   * InfoNCE on `(query, positive, [negatives])`, in-batch negatives
     + 7 mined hard negatives per row.
   * Batch size 256 effective (gradient accumulation if needed),
     temperature 0.02.
   * Keep `--archetype-supcon-weight 0.5` and
     `--community-supcon-weight 0.5` as aux losses to preserve the
     interpretability geometry that is our story.
   * 1 epoch over ~1M pairs first; expand if val improves.
3. Re-run [scripts/mteb_eval.py](../scripts/mteb_eval.py) on v12.
4. If MTEB average ≥ v8a + 1.0 pt, submit to the leaderboard via
   `mteb` CLI; otherwise iterate (lr, temperature, mix ratio).

### Week 3+ — push and harden

* Increase batch size with gradient cache (sentence-transformers
  CachedMultipleNegativesRankingLoss equivalent).
* Add `instructor`-style task instructions in the encoder prompt
  (`"Represent this sentence for retrieval: ..."`) — these are worth
  several points on MTEB.
* Try a heads-only fine-tune that **unfreezes the last 2 backbone
  layers** (LoRA r=16). This pushes us out of the 12.7M bucket but is
  still a small, novel artifact.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| `community_output.encoded` dim too small (<256) for retrieval | Add a configurable projection head in `train_contrastive.py`; eval script auto-detects dim |
| Backbone CE loss collapses when we train pure contrastive | Keep backbone frozen as today; only adapter params move. CE is monitored but not in the loss |
| MTEB eval takes longer than expected on 7B | Stage: Classification + STS first (~1h), retrieval last (~6h) |
| We are not competitive on retrieval (most MTEB weight) | Pivot to MTEB classification/clustering subset, where supcon training is over-represented in our objective |

## Success criteria

* **Floor**: v8a baseline on MTEB published to model card so the
  artifact is reproducible. **DONE** (mean Spearman 0.210, 40 STS splits).
* **First milestone**: v12 demonstrates contrastive training transfers
  through the SRT adapter geometry. **DONE** (+0.136 mean Spearman vs
  v8a; +0.79 on HUMESICK-R; trajectory monotone with no plateau).
* **Target**: v12 / v13 in MTEB top-25 on the English overall
  leaderboard among ≤7B models. **In progress** — v13 (msmarco hard
  negatives + nli + quora, warm-start from v12) launched 2026-04-30 17:12 UTC.
* **Stretch**: top-10 on MTEB classification subset (this is the
  cheapest path to a real leaderboard rank given our supcon-heavy
  training objective).

## Roadmap after v13

1. **v14: gradient-cache + larger effective batch** (eff. batch ≥ 256
   negatives via CachedMultipleNegativesRankingLoss-style accumulation).
   Standard 5–15 pt STS lift in the literature.
2. **v15: instruction prefixes** (`"Represent this sentence for
   retrieval: ..."`) per the E5/Instructor recipe. Several points on
   MTEB without retraining.
3. **v16: full MTEB English** (Classification + Clustering +
   Retrieval, not just STS). Submit best checkpoint to the leaderboard
   via `mteb` CLI.
4. **MMTEB stretch**: train on a multilingual mix to fix the
   Indic-crosslingual splits.

## Files added in this plan

* [scripts/mteb_eval.py](../scripts/mteb_eval.py) — MTEB harness
  wrapping `SRTAdapter` as a sentence encoder.
* [scripts/train_contrastive.py](../scripts/train_contrastive.py) —
  InfoNCE training loop with hard negatives, keeps adapter aux losses
  as regularizers.
* [scripts/launch_v12_mteb.sh](../scripts/launch_v12_mteb.sh) —
  reference launcher for the v12 run.
