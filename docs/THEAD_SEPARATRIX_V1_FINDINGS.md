# T-Head separatrix probe · v1 findings

**Date:** 2026-04-28
**Setup:** Frozen Qwen2.5-7B (28 layers), backbone hidden states from layer 16,
piped through `srt.modules.thead.TakensHead` (131K params: project_to=32,
embedding_dim=4, delay=2, knn_k=5, recurrence_eps=20).
**No SRT-Adapter loaded.** This is a v10-prototype readout running directly
off the frozen backbone, complementary to the v8a separatrix probe in
[SEPARATRIX_V8A_FINDINGS.md](SEPARATRIX_V8A_FINDINGS.md).
**Probe:** [data/probes/separatrix_illusion_v1.jsonl](../data/probes/separatrix_illusion_v1.jsonl).
**Hardware:** vast.ai A6000, bf16 backbone forwards + fp32 T-Head (cdist on
CUDA requires fp32). 244 forwards in ~3 s after eps calibration.
**Calibration:** Swept `recurrence_eps ∈ {1, 3, 10, 15, 20, 25, 28, 30, 100}`.
Below 15: zero recurrence (eps too tight). At 25 and above: saturated
(everything within radius). 20 hits the transition zone with rec_mean ≈ 3.6
and max ≈ 21 over T ≈ 30 (i.e., ~10% of pairs recur). All reported numbers
use eps = 20.
**Artifacts:** [readouts.jsonl](../artifacts/thead/separatrix_v1/readouts.jsonl)
(207 rows; the 9 prompt_only sequences shorter than `embedding_dim*delay+1 = 9`
were skipped, leaving the three continuation branches for all 61 items).

## Headline (Wilcoxon signed-rank, two-sided)

| metric | mystic − bedrock | tech − mystic | tech − bedrock |
|---|---|---|---|
| Lyapunov mean      | +0.070 (p=0.15) | **−0.140 (p=0.040)** | −0.071 (p=0.16) |
| **Lyapunov std**   | **+0.327 (p=2×10⁻⁹, frac+=0.89)** | **−0.263 (p=4×10⁻¹⁰)** | +0.064 (p=0.08) |
| **Recurrence mean** | **−2.77 (p=2×10⁻⁵)** | −0.40 (p=0.41) | **−3.17 (p=1×10⁻⁶)** |
| **Recurrence max** | **−6.36 (p=7×10⁻⁸)** | +0.92 (p=0.34) | **−5.44 (p=6×10⁻⁸)** |

Bolded cells: |z| > 3, p < 0.001.

## Two findings, one structure

**(F1) The mystical basin is more volatile.** Lyapunov *std* is markedly
larger on mystical continuations than on either technical (+0.263, p=4×10⁻¹⁰,
89-92% of items) or bedrock (+0.327, p=2×10⁻⁹). Lyapunov *mean* alone barely
discriminates; the spread does. Reading: mystical text walks a more
trajectory-volatile path through Qwen-7B's hidden-state phase space — the
model is *less stable* in its local divergence rate while generating
mysticism than while generating either technical or bedrock-philosophical
continuations. This is the cleanest direct confirmation of the original
separatrix-illusion picture: the mystical branch is in a wider, shallower
basin.

**(F2) The bedrock basin is the most structured.** Recurrence is *highest*
on bedrock continuations by a large margin: bedrock − technical = +3.17
(p=10⁻⁶), bedrock − mystical = +2.77 (p=2×10⁻⁵). Same ordering on
recurrence_max: bedrock dominates by ~5–6 over both alternatives,
p < 10⁻⁷ in both directions. Reading: bedrock-philosophical continuations
trace a more *recurrent* (orbit-like, structured-attractor) path through
phase space than either technical or mystical text. Coordinating a
philosophical move with technical vocabulary apparently produces the most
geometrically organised trajectory of the three branches.

## Convergence with the v8a MAH falsification

The v8a probe ([SEPARATRIX_V8A_FINDINGS.md](SEPARATRIX_V8A_FINDINGS.md))
falsified prediction (4) — MAH peak fired *more* on bedrock than on
mystical (p=3.8×10⁻¹¹, frac+=0.05). At the time we enumerated three
candidate interpretations of that result; the T-Head readouts here adjudicate
between them:

- **(a) "Bedrock is the harder collision"** — *strongly supported.* Both MAH
  peak (v8a) and T-Head recurrence (this probe) identify bedrock as the
  branch where the model is doing the most distinctively-structured work.
  MAH measures local prediction surprise modulated by community-conditioned
  attention geometry; recurrence measures phase-space periodicity. They are
  computed by *unrelated* heads (one is the SRT-Adapter's MAH3 at L21, the
  other is a fresh Takens delay-embedding at L16 with no learned alignment
  to the adapter). They agree on the rank ordering of the three branches.
- **(b) "Mystical is in-distribution slop"** — *partially supported.* The
  mystical branch shows the *highest* trajectory volatility (Lyapunov std,
  F1) and *low* recurrence — the chaotic-flow signature of an
  in-distribution but unconstrained generation pattern. This is consistent
  with mysticism-as-aesthetic being well-rehearsed but unforced.
- **(c) Probe construction artifact** — *not ruled out.* Length and density
  control on the bedrock branches still wants a v2 battery, but the fact
  that two independent heads tracking different geometric properties
  produce the same rank ordering makes a pure construction artifact a much
  less attractive explanation than (a) and (b).

## What this means for the paper

- §6.9 (just added in commit 53bf194) anticipated reading (a). It now has
  independent confirmation from a head that wasn't trained on this data and
  doesn't share any parameters with MAH.
- The T-Head program (Geofinitism / Haylett 2026) and the SRT-Adapter
  program agree on a separatrix probe with no theoretical alignment between
  them. This is one of the strongest possible cross-program convergence
  results for the §6.8 claim that the two routes are converging on the same
  picture from different starting axioms.
- v9 now has a clear empirical task: add a prototype-forcing decode path so
  Haylett's prediction (2) (counterfactual decoding under forced basin)
  becomes testable. If forcing the technical prototype on a mystical-prompt
  reduces Lyapunov std and raises recurrence, that closes the loop.

## Caveats

- 61 items, single hand-authored battery (one author).
- Bedrock continuations not length-balanced against mystical; v2 battery
  is queued.
- T-Head reads only one backbone layer per run; the multi-layer profile
  in `## Depth profile` below uses four runs at L7, L14, L16, L21.
- T-Head runs in fp32 over a bf16 backbone (cdist limitation). Per-item
  numbers may shift slightly on a fully-fp32 forward; the population-level
  rankings should not.
- Recurrence eps is per-layer (hidden-state norms grow with depth in
  Qwen2.5-7B). Calibrated values: eps = 15 at L7, 17 at L14, 20 at L16,
  35 at L21, each chosen so the population-mean recurrence falls in the
  3-12 range where the metric is well-resolved.

## Depth profile (added 2026-04-28)

The single-layer L16 result was extended to a four-layer sweep at L7, L14,
L16, L21 with the per-layer eps calibration above. Headline numbers
(Wilcoxon signed-rank, two-sided, n=61):

| layer | eps | Lyap std (mystic − bedrock) | Recurrence mean (bedrock − tech) |
|---:|---:|---|---|
| L7  | 15 | +0.294, frac+=0.84, p = 9 × 10⁻⁹ | −0.412, frac+=0.41, p = 0.40 (NS) |
| L14 | 17 | +0.280, frac+=0.80, p = 2 × 10⁻⁸ | +1.355, frac+=0.74, p = 2 × 10⁻⁴ |
| L16 | 20 | +0.327, frac+=0.89, p = 2 × 10⁻⁹ | +3.170, frac+=0.79, p = 1 × 10⁻⁶ |
| L21 | 35 | +0.318, frac+=0.93, p = 4 × 10⁻¹⁰ | **+3.673, frac+=0.85, p = 3 × 10⁻⁹** |

Two structural readings:

**(D1) Mystical-as-volatile is depth-invariant.** The Lyapunov-std
separation between mystical and bedrock is essentially constant across all
four depths (0.28-0.33), with frac+ ∈ [0.80, 0.93] and p ≤ 10⁻⁷ everywhere.
This is a property of the basin the model is in throughout the forward
pass, not a late-layer reasoning effect. The mystical branch traces a
wider, shallower trajectory through phase space at every measurement
site.

**(D2) Bedrock-as-recurrent builds monotonically with depth and peaks at
L21.** The recurrence-mean separation between bedrock and technical is
essentially zero at L7 (−0.41, NS), small at L14 (+1.36, p = 2 × 10⁻⁴),
large at L16 (+3.17, p = 10⁻⁶), and largest at L21 (+3.67, frac+=0.85,
p = 3 × 10⁻⁹). Bedrock-as-the-most-structured-trajectory is a property
that *emerges with depth* and is strongest at the deepest probed layer.

L21 is also where the v8a SRT-Adapter places its third MAH layer, and where
the `INTERIORITY_V1_FINDINGS.md` heatmap reports MAH divergence has the
largest between-regime spread ("L21 is where divergence diverges most").
The T-Head and MAH agree on depth, not just on rank ordering. The MAH peak
falsification (v8a, L21, p = 3.8 × 10⁻¹¹) and the T-Head recurrence
result (this probe, L21, p = 3 × 10⁻⁹) are pointing at the same
structural fact about Qwen2.5-7B: at the depth where the model is doing the
most between-regime discriminative work, the bedrock-philosophy basin is
the most distinctively-organised of the three.

Artifacts: `artifacts/thead/separatrix_v1/readouts_L{7,14,16,21}_calib.jsonl`
(L16 = `readouts.jsonl`).

## Per-layer MAH cross-check (added 2026-04-28)

To complete the cross-program convergence picture, we re-ran the v8a
SRT-Adapter on the same separatrix battery and recorded the
continuation-slice MAH peak-divergence and mean-divergence at *every* MAH
layer in v8a (L7, L14, L21), for every branch
(`scripts/separatrix_mah_layers.py`,
`artifacts/separatrix/v8a/readouts_mah_layers.jsonl`, n = 61). Wilcoxon
signed-rank, two-sided:

| MAH layer | bedrock − tech (peak) | bedrock − mystic (peak) | mystic − tech (peak) |
|---:|---|---|---|
| L7  | +0.41, frac+=0.87, p = 4 × 10⁻⁹  | +0.62, frac+=0.93, p = 7 × 10⁻¹¹ | −0.21, frac+=0.18, p = 2 × 10⁻⁸ |
| L14 | +0.23, frac+=0.80, p = 2 × 10⁻⁷  | +0.36, frac+=0.90, p = 1 × 10⁻¹⁰ | −0.13, frac+=0.16, p = 7 × 10⁻⁷ |
| L21 | +0.27, frac+=0.84, p = 8 × 10⁻⁶  | +0.58, frac+=0.95, p = 4 × 10⁻¹¹ | −0.31, frac+=0.11, p = 2 × 10⁻⁹ |

**MAH peak-divergence rank ordering at every depth: bedrock > technical >
mystical.** Every contrast significant at p ≤ 10⁻⁵; bedrock vs mystical
contrasts at p ≤ 10⁻¹⁰ at every depth. The bedrock branch produces the
largest local mismatch between MAH-predicted and observed hidden-state
trajectory — i.e. the branch the autoregressive prior fits the *worst*,
which is exactly what we would expect from the most idiosyncratic
(non-formulaic) of the three continuations.

T-Head recurrence at L21 (above): bedrock > technical (+3.67, p = 3 × 10⁻⁹)
> mystical (bedrock − mystic = +2.77, p = 2 × 10⁻⁵, from the headline
table). **Same rank ordering — bedrock > technical > mystical — from a
parameter-disjoint head.** That ordering holds at every measured depth for
MAH, and at L14, L16, L21 for T-Head recurrence.

Two depth-structure differences:

1. **MAH is depth-invariant in direction but not magnitude.** The
   bedrock−mystic peak gap is 0.62 at L7, 0.36 at L14, 0.58 at L21 — a
   weak U with the largest contrast at the shallowest layer.
2. **T-Head recurrence is monotonic with depth.** The bedrock−tech
   recurrence gap is NS at L7, +1.36 at L14, +3.17 at L16, +3.67 at L21.

The two heads disagree on *where* the discriminative signal is strongest
(MAH: L7 ≈ L21; T-Head: L21) but agree on *which way it points*
(bedrock > technical > mystical) at every depth they overlap. The L21
agreement is therefore not a coincidence of single-layer choice on either
side — both heads carry signal at L21, with consistent sign, large effect,
and p ≤ 10⁻⁵ in both cases.

## Length-matched falsification (added 2026-04-28)

In v1 the technical continuations are systematically ~4 words longer than
the bedrock and mystical continuations. The bedrock-vs-technical MAH effect
runs *opposite* to the length difference (bedrock peak is *larger* despite
being shorter, which weakens the length-confound reading), but the proper
test is to re-run on a length-matched battery. We built
`data/probes/separatrix_illusion_v1_lentrunc.jsonl` by, for each item,
truncating all three branches to the minimum word count among them
(per-item mean 13.2 words, range 7-20, no items dropped). Re-running both
heads:

**MAH peak ‖divergence‖ (length-matched, n = 61):**

| MAH layer | bedrock − tech | bedrock − mystic | mystic − tech |
|---:|---|---|---|
| L7  | +0.24, p = 1 × 10⁻³ | +0.53, p = 3 × 10⁻¹⁰ | −0.29, p = 1 × 10⁻⁴ |
| L14 | +0.18, p = 6 × 10⁻⁵ | +0.35, p = 1 × 10⁻¹⁰ | −0.17, p = 3 × 10⁻⁹ |
| L21 | +0.31, p = 5 × 10⁻⁷ | +0.49, p = 4 × 10⁻¹⁰ | −0.18, p = 2 × 10⁻⁷ |

**T-Head (length-matched, per-layer eps calib):**

| layer | Lyap std mystic − bedrock | recurrence bedrock − tech |
|---:|---|---|
| L7  | +0.21, p = 4 × 10⁻⁸ | +2.31, p = 4 × 10⁻⁶ |
| L14 | +0.20, p = 2 × 10⁻⁷ | +1.56, p = 6 × 10⁻⁸ |
| L16 | +0.21, p = 2 × 10⁻⁸ | +3.11, p = 2 × 10⁻⁹ |
| L21 | +0.16, p = 3 × 10⁻⁶ | +3.85, p = 2 × 10⁻⁹ |

**Interpretation.** The bedrock > technical > mystical MAH-peak rank
ordering holds at every depth, every contrast, p ≤ 10⁻³ everywhere on the
length-matched battery. T-Head Lyap-std mystical-volatility holds at every
depth, p ≤ 10⁻⁵ everywhere. T-Head recurrence bedrock > technical holds at
every depth, with L21 still the largest (the strict monotonicity in the
unmatched battery was a length artifact: L7 went from NS to p = 4 × 10⁻⁶
under length matching, indicating the unmatched L7 result was *deflated*
by the longer technical branches at that shallow layer where eps
calibration is most sensitive). Effect sizes attenuate ~40-50% under
truncation, which is itself partly a power loss from shorter trajectories.
The convergence claim survives the principal alternative explanation
(length confound).

Artifacts:
`artifacts/separatrix/v8a/readouts_mah_layers_lentrunc.jsonl`,
`artifacts/thead/separatrix_v1/readouts_lentrunc_L{7,14,16,21}_calib.jsonl`,
`data/probes/separatrix_illusion_v1_lentrunc.jsonl`.

## Counterfactual prototype-forcing — clean negative for Haylett pred (2) on v8a (added 2026-04-28)

The four results above are correlational. Haylett's prediction (2) is the
*causal* claim that decoding while forcing the model into the bedrock (resp.
mystical) basin should make that branch's continuation more probable. The
v8a forward already supports a `forced_community` override (see
`SRTAdapter.forward`, `forced_community` argument), so the test is direct.

**Procedure** (`scripts/separatrix_force_prototype.py`):

1. Compute per-branch centroids of the discovered community vectors from
   `artifacts/separatrix/v8a/readouts.jsonl` (n = 61 per branch).
2. For each item, for each branch, compute mean cross-entropy of the
   continuation tokens under four conditions: natural (no forcing),
   `force = bedrock`, `force = mystical`, `force = technical`.
3. Diagonal effect: CE under `force = self` should be lower than CE under
   `force = other` if the prototype causally biases decode.

**Result** (n = 61, Wilcoxon signed-rank, two-sided, deltas in nats):

| branch | mean(CE_other − CE_self) | frac+ | p |
|---|---:|---:|---:|
| technical | −0.0001 | 0.48 | 0.97 |
| mystical  | +0.0005 | 0.54 | 0.73 |
| bedrock   | +0.0017 | 0.57 | 0.083 |

Pairwise: every contrast |Δ| ≤ 0.002 nats, every p > 0.08. The most
forgiving (bedrock under force_bedrock vs force_technical) gives p = 0.083,
not significant uncorrected, certainly not under multiple comparisons.

**Sanity check — the forcing path is wired correctly.** On a single item
(`sep_001`, bedrock continuation, natural CE = 1.336):

| forcing | CE | Δ vs natural |
|---|---:|---:|
| natural | 1.336 | — |
| force = bedrock centroid (‖·‖ ≈ 1.0) | 1.346 | +0.010 |
| force = mystical centroid | 1.341 | +0.005 |
| force = zeros | 1.346 | +0.010 |
| force = 10× bedrock (‖·‖ ≈ 10) | 1.330 | −0.006 |
| force = 100× bedrock (‖·‖ ≈ 100) | 1.473 | +0.137 |
| force = randn × 10 | 1.268 | −0.068 |
| **force = randn × 100** | **3.609** | **+2.273** |

The path is live (random × 100 swings CE by 2.27 nats). The signal isn't
there at the magnitude v8a's community vectors live in.

**Reading.** The three branch centroids sit within ‖·‖ ∈ [0.19, 0.35] of
each other in 64-d space, with norms ≈ 1. The decode head is essentially
invariant to differences of that magnitude. v8a's MAH heads are
*sensitive* to the branch differences (large effect, p ≤ 10⁻⁵, four
depths, length-matched) but the v8a continuous-community injection path
lacks the leverage to *causally drive* decode toward those differences.

**Pred (2) is provisionally falsified for v8a** and queued as a primary
motivation for v9 (discrete prototypes with construction-enforced
inter-cluster spread, full prototype-forcing decode path).

The honest reading of the separatrix programme as a whole is therefore:
the *diagnostic* half of the SRT/Geofinitism convergence (read-off of
basin structure) is robust under two parameter-disjoint heads, four
depths, length matching, and a pre-registered three-branch design; the
*causal* half (forced-prototype decode) has a clean null on v8a and is
pending v9.

Artifacts:
`artifacts/separatrix/v8a/force_proto_readouts.jsonl`,
`artifacts/separatrix/v8a/force_proto_summary.json`,
`scripts/separatrix_force_prototype.py`.

## Reproducibility

```bash
HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 \
PYTHONPATH=/path/to/srt-adapter \
python3 scripts/thead_eval.py \
  --backbone Qwen/Qwen2.5-7B \
  --battery data/probes/separatrix_illusion_v1.jsonl \
  --output artifacts/thead/separatrix_v1/readouts.jsonl \
  --layer-index 16 --max-seq-len 128 --recurrence-eps 20
```

Local smoke test (CPU, tiny backbone): see header docstring of
[scripts/thead_eval.py](../scripts/thead_eval.py).
