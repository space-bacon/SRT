# Format as an ordering field: susceptibility, not a scaling law

Working note, 2026-09-02. Status: a fixed-endpoint result measured per problem
inside one benchmark, plus a cross-benchmark correlation it explains. The
mechanical artifact that would fake it has been measured and removed. Not yet a
bifurcation: we have an attractor and its basin, not a control-parameter sweep.

## What we set out to check

`paper_hivemind.md` section 5.2 claims the chat-template format effect **grows
with scale**, at +0.0441 intra-model similarity per decade of parameters across
the Qwen2.5-Coder ladder, and reads that as a scaling law for homogeneity.

Every arm behind that number was generated at `max_new=192`. HumanEval solutions
do not fit in 192 tokens, and the budget did not bind equally across arms, so we
regenerated the full 36-arm matrix at 1024 tokens and ran the identical analysis.
We also ran the same 36 arms on MBPP, which has 425 problems against HumanEval's
164.

## What came back

HumanEval replicates almost exactly. MBPP does not replicate at all.

| domain | format slope per decade | format range | tuning slope |
|---|---|---|---|
| HumanEval (164) | **+0.0433** (published: +0.0441) | 0.088 to 0.163 | −0.0402 |
| MBPP (425) | **−0.0070** | 0.012 to 0.050 | +0.0585 |

Both slopes invert between domains, including the tuning term. The paper's line
"Two families, two domains, two labs, same sign" does not hold for this pair.

Per rung:

| size | HE base raw | HE inst raw | HE inst chat | HE format | MBPP inst raw | MBPP format |
|---|---|---|---|---|---|---|
| 0.5B | 0.5593 | 0.7879 | 0.8757 | +0.0878 | 0.8588 | +0.0226 |
| 1.5B | 0.6076 | 0.7776 | 0.8744 | +0.0968 | 0.8196 | +0.0279 |
| 3B | 0.6226 | 0.7406 | 0.8896 | +0.1490 | 0.8589 | +0.0503 |
| 7B | 0.6160 | 0.7409 | 0.8765 | +0.1356 | 0.9122 | +0.0263 |
| 14B | 0.5861 | 0.7493 | 0.9125 | +0.1632 | 0.9507 | +0.0123 |
| 32B | 0.6454 | 0.7825 | 0.9414 | +0.1589 | 0.9576 | +0.0179 |

The published claim that the effect grows **monotonically across all six rungs**
is also gone. At a real budget the HumanEval format column dips at 7B and again
at 32B. It grows with scale. It does not grow monotonically.

## The variable that actually predicts the gain

Not parameter count. How disordered the raw arm already is.

    corr(inst_raw baseline similarity, format gain) = -0.870   pooled, n = 12
    slope                                           = -0.661
    within HumanEval alone  r = -0.561
    within MBPP alone       r = -0.590

    mean baseline order:  HumanEval 0.7631   MBPP 0.8930
    mean format gain:     HumanEval 0.1319   MBPP 0.0262

Scale was standing in for distance-from-order. On HumanEval the ladder happens to
get more disordered in the middle rungs, which produces a positive slope against
parameters. On MBPP the baseline climbs steadily with size, so the same
underlying relation produces a flat-to-negative slope. One relation, two apparent
scaling laws, opposite signs.

## Why this is the pitchfork, and what kind of evidence it is

Lancaster (2025) treats semiotic mediation as a pitchfork: a control parameter
carries a system from one shared interpretive attractor to two divergent
branches. The natural way to write that normal form includes an external field
`h` that breaks the symmetry and selects a branch, and the standard fact about
that form is that the susceptibility to `h` is largest near the critical point
and falls away on either side.

A prompt format is exactly such a field. It is a low-dimensional intervention
applied from outside that tells the model which branch to take, and it changes no
weights. So the prediction is that the format matters most where the system is
least committed, and stops mattering once the task statement has already
committed it.

That is the relation we measure, at r = −0.870. HumanEval hands the model a bare
function signature, which underdetermines the continuation, so the system sits
near the transition and the field does a lot of work. MBPP hands it a
natural-language task statement, which already supplies the field, so the
template adds almost nothing on top.

The termination data says the same thing from the other side. Measured by the
generator's own stop reason at a 1024-token budget, raw arms hit the cap 21.8% of
the time and never settle; chat arms terminate at 0.2%. That is the difference
between a system with no attractor and a system with a strong one, and it is not
a budget artifact: raising the budget does not fix it, because a base model given
a bare signature has no correct place to stop.

**What this evidence is.** A susceptibility signature consistent with the normal
form, over twelve points, plus a mechanism that predicts the sign. **What it is
not.** We have not located a critical point, not shown a branch structure, and
not varied a control parameter continuously. Two benchmarks are two values of a
coarse proxy, not a sweep.

## The per-problem test, and what it found

Two benchmarks are two points, so the relation above could be a between-domain
artifact. The sharp version holds model, family, tokenizer and budget fixed and
asks the question inside a single benchmark, per problem.

Two traps had to be handled first. Naive `gain = chat - raw` is mechanically
anti-correlated with `raw`, so measurement noise alone manufactures the result:
we measure that artifact at r = **-0.929**, which is most of what a careless
version of this test would report. Every number below is instead split-half, with
the eight raw samples cut into halves A and B, half A supplying the stratifying
variable and half B the outcome, so the two carry independent noise. Separately,
raw cosine on an anisotropic encoder is not interpretable, so every similarity is
quoted against a different-problem floor computed through the same path.

    pooled over 984 problem-model pairs, 164 problems x 6 rungs
      split-half  corr(raw_A, chat - raw_B) = -0.411
      naive       corr(raw_B, chat - raw_B) = -0.929   (the artifact, for scale)
      direct      corr(raw_A, chat)         = +0.080

The relation survives inside a single benchmark at -0.411. But the third line is
the one that matters, and it is not what we went looking for.

**The chat arm's per-problem order is very nearly independent of the raw arm's,
at r = +0.080.** Stratifying by half-A raw order into four equal bins:

| bin | raw_A | raw_B | chat | gain |
|---|---|---|---|---|
| 0, most disordered | 0.6062 | 0.6826 | **0.8909** | +0.2083 |
| 1 | 0.7482 | 0.7526 | **0.8950** | +0.1424 |
| 2 | 0.8226 | 0.7877 | **0.8909** | +0.1032 |
| 3, most ordered | 0.8880 | 0.8206 | **0.9032** | +0.0826 |

The raw column spans 0.282. The chat column spans 0.012.

So the format is not adding a fixed increment of order, and it is not adding an
increment that shrinks with baseline either. **It is pulling every problem to the
same level of order, near 0.89, regardless of where that problem started.** The
gain varies across bins only because the starting point varies. The destination
does not move.

That is an attractor with a basin, not a susceptibility gradient. It is a
stronger claim than the one the two-benchmark correlation supported, and it
explains that correlation as a consequence: if the endpoint is fixed, then
`gain = endpoint - start` is necessarily a decreasing function of the start, both
within a benchmark and across benchmarks. MBPP shows a small format effect
because MBPP's raw arm already sits near the same endpoint, not because MBPP is
insensitive to framing.

Per-rung, with floors, showing the effect is not an anisotropy artifact:

| size | raw intra (floor) | chat intra (floor) | split-half r |
|---|---|---|---|
| 0.5B | 0.7886 (0.3881) | 0.8757 (0.3006) | -0.427 |
| 1.5B | 0.7847 (0.3903) | 0.8744 (0.3054) | -0.594 |
| 3B | 0.7349 (0.3571) | 0.8896 (0.3026) | -0.308 |
| 7B | 0.7384 (0.3353) | 0.8765 (0.3012) | -0.294 |
| 14B | 0.7432 (0.3326) | 0.9125 (0.3109) | -0.423 |
| 32B | 0.7754 (0.3480) | 0.9414 (0.3075) | -0.505 |

Worth noting the floors themselves: the raw arm's different-problem floor is
consistently **higher** than the chat arm's, 0.33 to 0.39 against 0.30 to 0.31.
Raw outputs resemble each other more even across unrelated problems, which is
what generic continuation looks like. Against its own floor the chat arm is
further above baseline than the raw arm is, so the ordering is not an artifact of
the encoder's geometry.

## The ceiling objection, tested

A high baseline leaves less room, so a smaller gain is expected mechanically and
the correlation could be arithmetic rather than physical.

Normalising by the headroom actually available, `gain / (1 - inst_raw)`:

| | 0.5B | 1.5B | 3B | 7B | 14B | 32B | mean |
|---|---|---|---|---|---|---|---|
| HumanEval | 0.414 | 0.435 | 0.574 | 0.523 | 0.651 | 0.731 | **0.555** |
| MBPP | 0.160 | 0.155 | 0.356 | 0.300 | 0.249 | 0.422 | **0.274** |

The domain gap survives at 2.03x, so it is not only a ceiling. Caveat that
matters: MBPP's 32B point has 0.042 of headroom left, so that cell is fragile and
should not carry the argument alone.

## What settles it

1. **Done, and it changed the claim.** The per-problem test above holds the
   relation inside one benchmark and shows the endpoint is fixed. See that
   section. What remains is to check whether the endpoint is the same constant on
   MBPP: if the chat arm lands near 0.89 there too, the attractor is a property of
   the template rather than of the benchmark, which is the strong version.

2. **A continuous control parameter.** The five arms we have (raw, persona,
   shared, shared_persona, chat) are an ordinal ladder of framing strength, not a
   dial. Interpolating framing strength, for instance by varying how much of the
   template is present, turns four points into a curve. This is now the single
   most valuable experiment, because a fixed endpoint reached from many starts is
   exactly what should show a threshold as the field is weakened.

3. **Where the basin ends.** Bin 0 starts at 0.606 and still lands at 0.891. We
   have not found a problem disordered enough to escape. Deliberately
   underdetermined prompts would locate the edge of the basin, if there is one.

4. **A third domain picked in advance.** State the predicted chat-arm endpoint
   before running it. That is the difference between fitting the data and
   predicting it.

## Consequences for the papers

- Section 5.2's headline survives on HumanEval and must stop being stated as a
  general law. It is domain-conditional.
- The word "monotonically" has to go.
- The published tuning slope is wrong: −0.0220 becomes −0.0402 per decade.
- The cross-domain claim needs the MBPP counterexample stated in the same
  paragraph as the HumanEval result, not in limitations.
- One thing gets stronger, and it should be said. Truncation completely inverted
  between the two budgets: at 192 tokens the chat arm was the more truncated one,
  at 1024 the raw arm is, 21.8% against 0.2%. The effect kept its sign, magnitude
  and slope across that reversal. A differential-truncation artifact would have
  flipped or died. It did neither.

## Artifacts

    artifacts/nla/coder_matrix1024/scaling_curve.json    HumanEval, 36 arms, 1024 tok
    artifacts/nla/mbpp_matrix1024/scaling_curve.json     MBPP, 36 arms, 1024 tok
    artifacts/nla/coder_matrix1024/meta/                 per-arm finish_reason truncation
    artifacts/nla/format_susceptibility_perproblem.json  per-problem split-half test
    scripts/coder_matrix_vllm.py                         generation
    scripts/coder_ladder_analyze.py                      aggregate similarity
    scripts/format_susceptibility_perproblem.py          per-problem test

Both matrices: 47,232 generations each on HumanEval, same encoder, same
different-prompt floors, bfloat16, K = 8, top_p 0.9, temperature 1.0.
