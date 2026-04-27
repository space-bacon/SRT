# Layer-as-organ map · v1 findings

**Date:** 2026-04-27
**Setup:** SRT adapter v8a (14.5M params) on frozen Qwen2.5-7B (28 layers).
MAH taps at L7, L14, L21. RRM injections at L14, L21. Community discovery at L4.
**Battery:** 11 semiotic regimes × 25 prompts = 275 items
([data/probes/probe_battery_v1.jsonl](data/probes/probe_battery_v1.jsonl)).
**Artifacts:** [readouts.jsonl](../artifacts/interiority/v1/readouts.jsonl),
[summary.json](../artifacts/interiority/v1/summary.json),
[heatmap.png](../artifacts/interiority/v1/heatmap.png),
[heatmap.svg](../artifacts/interiority/v1/heatmap.svg).

## Headline observations

These are the column-wise winners and losers (z-scored across the 11 regimes,
so positive z = "this regime activates this channel more than average," not
"more than another channel").

| channel | high | z | low | z |
|---|---|---:|---|---:|
| div L7 | code | +1.71 | self_reference | −1.25 |
| div L14 | metaphor | +1.91 | counterfactual | −1.10 |
| div L21 | code | +2.84 | negation_modality | −0.94 |
| inj L14 | metaphor | +1.97 | quoted_speech | −1.29 |
| inj L21 | metaphor | +1.84 | quoted_speech | −1.40 |
| r_hat | code | +2.12 | negation_modality | −1.45 |
| P(super) | literal | +0.56 | code | −3.15 |
| regime entropy | code | +3.13 | negation_modality | −0.57 |
| chain residual | metaphor | +2.66 | counterfactual | −1.07 |
| comm top1 weight | counterfactual | +2.06 | negation_modality | −1.20 |

## Real signals

1. **Code is the regime-classifier's only ambiguity.** P(super) drops to 0.65
   (z = −3.15) and entropy is highest (z = +3.13) — every other class is
   pegged at P(super) ≥ 0.96. The regime head was trained on natural language
   only, so this is exactly the right behavior: it abstains on out-of-domain
   syntax. Useful: code can be detected by *high regime-classifier entropy*
   without retraining.

2. **Metaphor is the injection-heavy class.** ‖inj L14‖ ≈ 509 vs ≈ 277 for
   quoted_speech (cleanest contrast). RRM is doing the most work pushing the
   residual stream around on figurative inputs. Pairs with the highest chain
   residual (z = +2.66) — divergence is least predictable layer-to-layer on
   metaphor, i.e. each layer adds new metapragmatic information.

3. **negation_modality is a quiet class.** Lowest r_hat (0.355), lowest
   community top-1 weight, near-zero regime entropy. The adapter treats dense
   hedging as confidently in-distribution and low-novelty — a slightly
   counterintuitive result worth following up.

4. **Layer L21 is where divergence diverges most.** Across regimes, L21 has
   the largest spread (code = 2.33, negation_modality = 1.33). L7 and L14
   are dominated by literal/metaphor norms; L21 is where the discriminative
   *between-regime* signal lives. Suggests L21 is the best single-layer probe
   site for downstream classifiers.

5. **Counterfactual sharpens the community channel.** Top-1 community weight
   z = +2.06, while no other regime exceeds +0.5. Most classes get a near-
   uniform community softmax (entropy ≈ ln 32 = 3.466, see caveats below).

## Caveats / known issues to address in v2

- **Community entropy = 3.46 caveat — RESOLVED (see "v2 update" below).**
  Root cause was a probe-script bug: `SRTConfig()` defaults to
  `community.use_prototypes=True`, but the v8a checkpoint was trained with
  `use_prototypes=False` (the encoder output *is* the 64-D community
  vector; no discrete prototype heads were ever trained). Loading the
  adapter doesn't reset that flag, so the v1 probe was running cosine
  similarity against random-initialized prototypes, which gives an exactly
  uniform softmax (entropy = ln 32 = 3.466). Fixed in
  [scripts/interiority_probe.py](../scripts/interiority_probe.py) by
  honoring `use_prototypes=False` and dumping the raw 64-D vector instead.
- Sample size is 25/class. Bootstrapped CIs would be appropriate before any
  of this gets cited.
- Token positions are mean-pooled. Per-position trajectories (experiment 6
  in [INTERIORITY_STUDY.md](INTERIORITY_STUDY.md)) will tell a different
  story and should be the next pass.
- Prompts are short (mean ~16 tokens). A longer-context battery (200–500
  tokens of paragraph-level prose) would be a different experiment, not a
  rerun of this one.
- The "channels" in the heatmap mix scales. The diverging color encodes
  z-score; the cell text is the raw mean — read both, not one.

## Reproducibility

```bash
# 1. Build battery
python3 scripts/build_probe_battery.py
# → data/probes/probe_battery_v1.jsonl  (275 lines, 11 classes × 25)

# 2. Extract readouts (needs adapter checkpoint + GPU)
python3 scripts/interiority_probe.py \
    --adapter checkpoints/adapter_v8a/best_adapter.pt \
    --battery data/probes/probe_battery_v1.jsonl \
    --output  artifacts/interiority/v1/readouts.jsonl
# Wall time on RTX A5000: ~20 s after model load.

# 3. Render heatmap + summary
python3 scripts/interiority_heatmap.py \
    --readouts artifacts/interiority/v1/readouts.jsonl \
    --out-dir  artifacts/interiority/v1
```

---

## v2 update — community channel works, just not as a softmax

After fixing the `use_prototypes` flag, we re-ran the same battery and
analyzed the raw 64-D community vector directly. Artifacts:
[v2/readouts.jsonl](../artifacts/interiority/v2/readouts.jsonl),
[v2/summary.json](../artifacts/interiority/v2/summary.json),
[v2/heatmap.png](../artifacts/interiority/v2/heatmap.png),
[v2/community_pca.png](../artifacts/interiority/v2/community_pca.png).

**Headline:** the community channel *does* discriminate regimes — but as a
continuous embedding, not a categorical softmax. v8a was trained without
prototypes, so reading it as "soft cluster assignments" was the wrong
question. Reading it as a 64-D coordinate is the right one.

### Quantitative separation

Per-regime centroids of the 64-D community vector, then:

| metric | value |
|---|---:|
| mean intra-class radius (centroid → members) | 0.820 |
| mean inter-centroid distance | 0.698 |
| **separation ratio (inter / intra)** | **0.85** |

A ratio < 1 means classes overlap on average — but the *structure of
overlap* is meaningful. Top and bottom centroid pairs:

**Most distant centroid pairs:**

| a | b | distance |
|---|---|---:|
| code | literal | 1.068 |
| irony | literal | 0.962 |
| code | metaphor | 0.957 |
| counterfactual | literal | 0.942 |
| code | counterfactual | 0.937 |

**Closest centroid pairs:**

| a | b | distance |
|---|---|---:|
| lyric | metaphor | 0.556 |
| deixis | self_reference | 0.539 |
| deixis | irony | 0.536 |
| irony | lyric | 0.527 |
| deixis | quoted_speech | 0.527 |

The "closest" list reads like a literary-criticism adjacency graph: lyric
and metaphor share figurative density; deixis, self-reference, irony, and
quoted speech all involve point-of-view shifts. The "most distant" list
puts code at one pole and literal/figurative natural language at the
other, with counterfactual sitting between them — consistent with code
also dominating the regime-entropy channel above.

### Heatmap channel update

The two old community columns (`comm top1`, `comm H`) were dropped — both
were degenerate when `use_prototypes=False`. Replaced by `comm norm`, the
L2 norm of the continuous community vector. Same regime ranking pattern:
metaphor sits at the top (z = +1.56), self_reference at the bottom
(z = −1.34), echoing the inj-norm columns and reinforcing the
"figurative content draws more adapter compute" reading.

### What this resolves and what it doesn't

- ✅ Resolves the "constant 3.46 entropy" caveat — it was a config mismatch,
  not a model collapse.
- ✅ Confirms the community channel carries continuous, semantically
  organized structure on these prompts.
- ⚠️  The separation ratio of 0.85 means single-prompt nearest-centroid
  classification will be unreliable; bootstrap CIs and longer prompts
  remain on the agenda.
- ⚠️  Top-2 PCs explain only ~20% variance, so the
  [community_pca.png](../artifacts/interiority/v2/community_pca.png)
  scatter is a thin slice — a t-SNE/UMAP on the full 64-D vector would
  make the regime clusters more visible.

### Reproducibility (v2)

```bash
# Same battery as v1; only the probe and heatmap scripts changed.
python3 scripts/interiority_probe.py \
    --adapter checkpoints/adapter_v8a/best_adapter.pt \
    --battery data/probes/probe_battery_v1.jsonl \
    --output  artifacts/interiority/v2/readouts.jsonl
python3 scripts/interiority_heatmap.py \
    --readouts artifacts/interiority/v2/readouts.jsonl \
    --out-dir  artifacts/interiority/v2
```
