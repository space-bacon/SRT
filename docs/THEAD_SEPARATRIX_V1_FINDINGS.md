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
- T-Head reads only one backbone layer (L16). A multi-layer sweep (L7,
  L14, L16, L21) would tell us whether trajectory volatility / recurrence
  has the same depth structure as MAH divergence.
- T-Head runs in fp32 over a bf16 backbone (cdist limitation). Per-item
  numbers may shift slightly on a fully-fp32 forward; the population-level
  rankings should not.
- Recurrence eps was hand-calibrated (=20). A principled per-sequence
  scaling (e.g., eps as a fixed quantile of pairwise distances) would
  remove this knob.

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
