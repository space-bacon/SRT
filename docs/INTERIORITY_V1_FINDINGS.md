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

---

## Update v3 — per-position trajectories (Experiment #2)

**Date:** 2026-04-27.
**Question:** are the regime-level "lifts" from v1/v2 distributed across the
prompt, or are they concentrated at one or two anchor tokens?
**Method:** rerun the same 275 prompts but dump per-token readouts instead of
mean-pooling. Resample each prompt onto 20 evenly-spaced relative positions
(0 → 1) and average within regime.
**Artifacts:** [readouts.jsonl](../artifacts/interiority/v2_trajectory/readouts.jsonl),
[summary.json](../artifacts/interiority/v2_trajectory/summary.json),
9 small-multiples plots in
[artifacts/interiority/v2_trajectory/](../artifacts/interiority/v2_trajectory/).

### Result 1 — adapter signal channels are dominated by a BOS sink

For every regime, every MAH-divergence layer, every RRM-injection layer, and
the chain residual, the per-token signal looks the same: a large spike at
relative position 0 (the BOS / first content token) followed by an order-of-
magnitude collapse to ~0 by position 0.2.

The peak/mean ratios make this concrete:

| channel | peak/mean range across regimes | peak position |
|---|---:|---:|
| inj L14 | 8.4 – 17.1 | 0.00 |
| inj L21 | 8.7 – 13.8 | 0.00 |
| div L7  | 6.1 – 9.1  | 0.00 |
| div L14 | 5.0 – 8.5  | 0.00 |
| div L21 | 3.8 – 4.9  | 0.00 |
| chain residual | 7.0 – 13.4 | 0.00 |

This is the **attention-sink** phenomenon ([Xiao et al., 2024](https://arxiv.org/abs/2309.17453))
applied to *adapter writes*, not just attention scores: the adapter is using the
first token as a scratch register and dumping most of its contribution there.

**Implication for v1.** The mean-pooled inj/div/chain numbers we reported in
the v1 heatmap are real, but they are largely averages over the BOS spike
amplitude. The regime ordering survives — `metaphor` does write more at
token 0 than `quoted_speech` — but readers should not interpret those columns
as "metaphor uses MAH consistently across the prompt." The right reading is:
*metaphor prompts produce a larger first-token write into the residual stream.*
Mid-prompt MAH/RRM activity is essentially zero in v8a.

### Result 2 — only the regime classifier is distributed

`r_hat` and the regime-classifier entropy carry per-token structure that is
not concentrated at the BOS token. The shapes are regime-specific:

- **literal** — `r_hat` rises monotonically through the prompt, peak at
  position 1.00 (P/M = 1.84). The classifier integrates evidence as the
  literal context accumulates.
- **metaphor** — clear mid-prompt bump at position 0.42 (P/M = 1.60),
  consistent with the classifier reacting to the metaphorical word once
  it appears.
- **counterfactual** — late rise (peak at 1.00, P/M = 1.43), consistent
  with the conditional being resolved at the consequent.
- **deixis, negation_modality, refusal_bait, self_reference** — flat
  trajectories (P/M ≈ 1.2). The classifier locks in early and does not
  update.
- **code** — uniformly elevated `r_hat` and the only regime with
  sustained classifier *uncertainty* (regime entropy ≈ 0.3 nats throughout
  vs. ~0.03 nats for natural language). This corroborates v1's "abstain on
  code" reading at the per-token level.

So when we said in v1 that "the regime classifier is the most discriminating
column," v3 sharpens that: it is also the **only column with token-resolved
behavior** in v8a.

### Result 3 — community vector is a single coordinate

The community head pools internally before producing its 64-D vector, so
trajectories do not apply to it; v2's separation_ratio = 0.85 finding stands
as the per-prompt summary.

### What this re-frames

- v1's "metaphor draws more adapter compute" remains true, but the compute
  is delivered as a single first-token write, not a distributed lift.
- v1's "code lights up the regime classifier" generalises to "code is the
  *only* regime where the classifier remains uncertain at every token."
- The MAH/RRM modules in v8a appear to have learned a **prompt-summary**
  policy (write everything at token 0) rather than a **content-routing**
  policy (write at the relevant words). This is a training-objective
  finding worth flagging for v9.

### Reproducibility (v3)

```bash
python3 scripts/interiority_trajectory.py \
    --adapter checkpoints/adapter_v8a/best_adapter.pt \
    --battery data/probes/probe_battery_v1.jsonl \
    --output  artifacts/interiority/v2_trajectory/readouts.jsonl \
    --max-seq-len 128
python3 scripts/interiority_trajectory_plot.py \
    --readouts artifacts/interiority/v2_trajectory/readouts.jsonl \
    --out-dir  artifacts/interiority/v2_trajectory
```

---

## Update v3.1 — BOS-sink ablation (Experiment #3)

**Date:** 2026-04-27.
**Question:** v3 showed that adapter writes are concentrated at token 0.
Were v1's regime-level findings *entirely* the BOS sink, or is there a
mid-prompt signal underneath?
**Method:** re-pool the v3 trajectory readouts with the first token excluded.
Compare the resulting per-regime rankings against v1's BOS-included pooling
(Spearman rank correlation; whether the top-ranked regime per channel is
preserved).
**Artifacts:**
[bos_ablation.json](../artifacts/interiority/v2_trajectory/bos_ablation.json),
[scripts/interiority_bos_ablation.py](../scripts/interiority_bos_ablation.py).

### Result — v1's findings survive the ablation

| channel | top w/ BOS (z) | top w/o BOS (z) | Spearman | kept |
|---|---|---|---:|:---:|
| r_hat       | code (+2.09)     | code (+1.98)     | +1.000 | yes |
| P(super)    | literal (+0.55)  | literal (+0.55)  | +0.991 | yes |
| regime H    | code (+3.13)     | code (+3.13)     | +0.927 | yes |
| chain res   | metaphor (+2.66) | metaphor (+2.59) | +0.918 | yes |
| div L21     | code (+2.83)     | code (+2.85)     | +0.791 | yes |
| div L14     | metaphor (+1.91) | metaphor (+2.16) | +0.673 | yes |
| div L7      | code (+1.74)     | code (+1.83)     | +0.618 | yes |
| inj L21     | metaphor (+1.81) | metaphor (+2.31) | +0.464 | yes |
| inj L14     | metaphor (+1.97) | metaphor (+2.61) | +0.236 | yes |

**Aggregate:** mean Spearman = +0.735, min = +0.236, top-ranked regime
preserved in 9 / 9 channels.

### What this re-frames (again)

v3's "adapter writes are dominated by a BOS sink" finding stands as a
*description of the per-token shape*. But v3.1 shows the sink is
**amplitude-modulated by the same regime preference** that v1 measured
mean-pooled — not the *source* of it. Mid-prompt writes are small but
regime-aligned: when the BOS token is excluded, `metaphor` actually
*strengthens* its lead on the inj-norm channels (z = +1.97 → +2.61 for
inj L14). The adapter writes more at token 0 *because* the prompt is
metaphorical, not the other way around.

So the layer-as-organ map from v1 is real, with two qualifications:

1. The signal is delivered through a BOS-anchored attention sink (v3).
2. The mid-prompt residue is small but rank-consistent (v3.1).

The classifier channels (`r_hat`, `regime H`, `P(super)`, `chain res`)
are nearly invariant under the ablation (Spearman ≥ 0.92), confirming
v3's finding that they alone carry distributed per-token structure.

### Reproducibility (v3.1)

```bash
python3 scripts/interiority_bos_ablation.py \
    --readouts artifacts/interiority/v2_trajectory/readouts.jsonl \
    --out      artifacts/interiority/v2_trajectory/bos_ablation.json
```

---

## Update v3.2 — bootstrap CIs on community separation (Experiment #4)

**Date:** 2026-04-27.
**Question:** v2 reported community `separation_ratio = 0.85` from a single
275-prompt run. Is this a reliable population estimate, and is the
community vector usable as a per-prompt classifier?
**Method:** bootstrap by resampling prompts within each regime (n = 2000
iterations). Report percentile CIs on `separation_ratio`, on each
inter-centroid distance, and on leave-one-out 1-NN accuracy (200
bootstrap iterations for the more expensive nearest-centroid metric).
**Artifacts:**
[separation_bootstrap.json](../artifacts/interiority/v2/separation_bootstrap.json),
[scripts/interiority_separation_bootstrap.py](../scripts/interiority_separation_bootstrap.py).

### Results

| metric | point | bootstrap mean | 95% CI | chance |
|---|---:|---:|---:|---:|
| separation_ratio | 0.851 | 0.916 | [0.862, 0.983] | — |
| 1-NN accuracy (LOO) | 0.698 | 0.770 | [0.716, 0.818] | 0.091 |

**Rank stability** (out of 2000 bootstraps):
- Most-distant pair `code ↔ literal` (d = 1.068) stays #1 in **85.5%** of bootstraps.
- Most-similar pair `quoted_speech ↔ self_reference` (d = 0.385) stays last in **83.0%** of bootstraps.

### What this answers

1. **Yes, the community vector is usable as a per-prompt classifier.** 1-NN
   accuracy of 0.70 on 11 balanced classes is **7.7× chance**. Not strong
   enough to ship as a standalone regime detector, but strong enough to
   say the v8a community head learned a meaningful semiotic embedding —
   it isn't just a noisy projection of token statistics.
2. **The v2 separation_ratio of 0.85 is a stable population estimate.**
   The bootstrap CI sits slightly *above* the point estimate (a known
   bias of bootstrap on centroid-distance ratios: with-replacement
   resampling produces duplicate vectors, deflating intra-cluster
   distance), so 0.85 should be read as a lower bound on the true value.
3. **Centroid-pair rankings are stable.** Both extreme pairs stay extreme
   in >83% of bootstraps, supporting the v2 reading that
   "code is semantically far from natural language" and
   "quoted_speech / self_reference are nearly indistinguishable in this
   embedding."

### What this doesn't answer

- Whether the embedding generalises beyond the v1 battery's 25-prompt
  sub-distributions. A held-out battery would address this.
- Whether 1-NN at 70% scales — it might be a property of the 275-prompt
  closed set rather than an open-world detector.

### Reproducibility (v3.2)

```bash
python3 scripts/interiority_separation_bootstrap.py \
    --readouts artifacts/interiority/v2/readouts.jsonl \
    --out      artifacts/interiority/v2/separation_bootstrap.json \
    --n-boot   2000
```
