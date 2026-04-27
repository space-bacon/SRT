# Interiority Study — what to mine from a queryable SRT

Living plan for the empirical follow-on to v8a. The adapter exposes per-layer
divergences, injections, meta-state, BEN (`r_hat`, `regime_logits`), community
weights, and chain residual — which makes the model itself an instrument we
can point at language. This document is the menu of experiments and the order
we attack them in.

## Inventory of readouts (what's actually queryable)

For each forward pass over `(B, T)` tokens the adapter returns:

| Field | Shape | Interpretation |
|---|---|---|
| `divergences[l]` | `(B, T, d_div=256)` | metapragmatic deviation at MAH layer `l` |
| `injections[l]` | `(B, T, d_back)` | RRM correction added back at layer `l` |
| `meta_state` | `(B, T, d_meta=512)` | RRM GRU state after final MAH tap |
| `ben_output.r_hat` | `(B, T)` | scalar reflexivity (Lyapunov-like) per token |
| `ben_output.regime_logits` | `(B, T, 2)` | sub- vs super-critical classifier |
| `community_output.weights` | `(B, K=32)` | soft community assignment |
| `community_output.encoded` | `(B, d_comm=64)` | continuous community coordinate |
| `chain_residual_per_token` | `(B, T)` | how predictable layer `l+1`'s divergence is from `l`'s |

For a 28-layer Qwen2.5-7B the MAH taps default to layers `[7, 14, 21]`.

## Experiment menu (priority order)

### 1. Layer-as-organ heatmap **— in progress**
Run a fixed probe battery (10 semiotic regimes × 50 prompts) and compute, per
MAH layer, per regime: mean ‖divergence‖, variance, mean ‖injection‖, mean
`r_hat`, mean regime-classifier logit, mean chain residual. Plot a
regime × (layer, channel) heatmap. **Deliverable:** one figure that says
"Qwen2.5-7B's irony locus sits at L_x, its self-reference locus at L_y." The
artifacts land in `artifacts/interiority/v1/`.

### 2. Regime classifier as labeler
The regime head hits AUROC 0.99 / ECE 9e-4 on the held-out val set. Sweep a
real corpus (Gutenberg + arXiv + Reddit, 50K sentences) and tag each with
`P(supercritical)`. Mine: paragraph-level transition rhythms, maximally
ambiguous sentences (entropy ≈ ln 2).

### 3. Steering slider in the demo
At inference, scale `injections[l] *= α` for one module. Add a slider to the
gradio app; ship 20 samples per α on a fixed seed prompt. Turns the demo from
"description of interiority" into "demonstration of interiority."

### 4. Quote-depth probe (single falsifiable claim)
MAH activations vs nesting depth in `She said "He said 'I am tired'"`. If MAH
activations track quote depth monotonically, that's a one-figure paper.

### 5. Module ablation matrix (causal)
Zero-ablate one of {community, MAH-l, RRM-l, BEN} at a time. Measure delta
in (a) base-model perplexity on held-out text, (b) regime AUROC, (c) human
rated weirdness on 50 generations. Isolates *what each module contributes*.

### 6. Trajectory geometry of single utterances
For one prompt dump `divergences` as a `(T, L, d_div)` tensor. PCA per layer.
Compute path length and curvature. Hypothesis: figurative inputs have longer,
more curved trajectories at interior layers than literal inputs. Align across
paraphrases — distance becomes a semantic-not-lexical similarity score.

### 7. Cross-prompt invariants — model "personality"
Pool readouts across 10K prompts. Find prompt-invariant directions in
adapter space — those are the model's tonic priors. Repeat with a second
base (Llama-3-8B with the adapter retrained) → which interiority structure
is universal vs model-specific?

### 8. Calibration → behavior bridge
Correlate regime-classifier confidence with: (a) base-model surprisal,
(b) human-judged truthfulness on TruthfulQA-style items, (c) hedging marker
frequency in the output. If high-uncertainty regimes → more hedging, the
adapter is reading something the base model behaviorally expresses.

### 9. Community-module dictionary
Dump top-activating ngrams per cluster across 1M tokens. Some clusters will
be obvious (negation, numbers); the interesting ones are the unnamed ones —
empirical handles on hitherto-unnamed semiotic categories.

### 10. Developmental time-series
If intermediate checkpoints survive, plot when each interiority structure
crystallizes during training. "Regime separation appears at step ~20k;
metaphor curvature emerges at step ~50k" — a developmental story.

## Rules of engagement
- All readouts dumped to `artifacts/interiority/<exp>/` as `.jsonl` + `.npz`.
- Every figure is reproducible from the script that produced it.
- Honest framing: report negative results when the adapter sees nothing.
- No new capability claims unless an ablation supports them.
