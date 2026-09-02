# Format as an ordering field: susceptibility, not a scaling law

Working note, 2026-09-02. Status: the template acts as a one-sided floor on
output order, measured per problem on two benchmarks with the mechanical artifact
that would fake it quantified and removed. Under a continuous field the response
is a saturating curve with no branch structure; a claimed coexistence window was
pre-registered, tested, and retracted (see "Outcome"). This is a susceptibility
result, not a bifurcation, and the pitchfork remains an analogy here.

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
increment that shrinks with baseline either. On HumanEval it pulls every problem
to roughly 0.89 regardless of where that problem started. The gain varies across
bins only because the starting point varies.

## MBPP corrects that reading: a floor, not a fixed point

Read on HumanEval alone, the above says the template has a fixed destination.
Running the identical test on MBPP shows that conclusion was too strong, and
shows why.

| | direct corr(raw_A, chat) | chat column span |
|---|---|---|
| HumanEval | +0.080 | 0.012 |
| MBPP | **+0.654** | **0.100** |

On MBPP the chat arm's order is not independent of the raw arm's at all, and the
chat column is not flat. The difference is that MBPP's raw arm is already more
ordered than HumanEval's, and in two of its four bins it is already more ordered
than the level HumanEval's chat arm converges to.

Putting all eight bins from both benchmarks on one axis and splitting at 0.88:

| bench | bin | raw_A | chat | delta | |
|---|---|---|---|---|---|
| HumanEval | 0 | 0.6062 | 0.8909 | **+0.2846** | below |
| HumanEval | 1 | 0.7482 | 0.8950 | +0.1468 | below |
| HumanEval | 2 | 0.8226 | 0.8909 | +0.0683 | below |
| MBPP | 0 | 0.7907 | 0.8699 | +0.0792 | below |
| MBPP | 1 | 0.8725 | 0.9021 | +0.0295 | below |
| HumanEval | 3 | 0.8880 | 0.9032 | +0.0152 | above |
| MBPP | 2 | 0.9266 | 0.9348 | +0.0082 | above |
| MBPP | 3 | 0.9815 | **0.9700** | **−0.0115** | above |

    starting below 0.88 (5 bins):  0.7681 -> 0.8898   +0.1217
    starting at or above (3 bins): 0.9320 -> 0.9360   +0.0040

**The template imposes a floor on order, not a destination.** Below roughly 0.88
it lifts hard, and lands the system on the floor. At or above it, the template
does essentially nothing, thirty times less, and in the most ordered bin it
slightly reduces order.

The HumanEval-only picture looked like a fixed point because no HumanEval bin
ever starts above the floor. That is a good illustration of why a second domain
was worth the compute: the single-domain reading was not wrong about HumanEval,
it was wrong about what HumanEval was showing.

This still explains the cross-benchmark correlation, and better. With a one-sided
floor, `gain = max(floor, start) - start` falls to zero as `start` rises past the
floor, which is what the -0.870 was measuring. MBPP's format effect is small
because most of MBPP already sits at or above the floor, and its most ordered
problems are actively made slightly less orderly by the template.

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

## A continuous field, and a two-branch window

Everything above uses four discrete arms, which is a ladder rather than a control
parameter, so it cannot show a threshold. To get a dial we build the templated
sequence as prefix + stem + suffix, so template positions are known exactly, then
replace the embedding at each template position with

    e' = neutral + alpha * (e - neutral)

where `neutral` is the mean of the embedding matrix. At alpha = 1 this is exactly
the chat arm. At alpha = 0 the template positions are still present, at the same
length and the same positions, carrying the average token instead of turn
structure. That is a better control than the raw arm, which confounds the
template's presence with its content. Here only the content moves.

**The dial validates against the arm it replaces.** At alpha = 0 intra-model
order is 0.7372; the raw arm in the 36-arm matrix, with no template tokens at
all, is 0.7406. They agree to 0.003. Neutralising the template's meaning
reproduces not having a template, which is what the construction was for.

Qwen2.5-Coder-3B, 164 HumanEval problems, K = 8, 1024 tokens, truncation 0.0% to
0.5% across all eleven points:

| alpha | mean | sd | BC | BIC1−BIC2 | minority w | branches |
|---|---|---|---|---|---|---|
| 0.00 | 0.7372 | **0.1865** | 0.506 | 3.5 | — | one |
| 0.10 | 0.8345 | 0.0963 | **0.690** | **133.2** | 0.241 | **TWO** |
| 0.20 | 0.7989 | 0.1143 | **0.685** | **100.5** | 0.267 | **TWO** |
| 0.30 | 0.8759 | 0.0395 | 0.486 | 20.7 | — | one |
| 0.40 | 0.8922 | 0.0360 | 0.398 | 0.8 | — | one |
| 0.50 | 0.8918 | 0.0359 | 0.426 | −5.5 | — | one |
| 0.60 | 0.8908 | 0.0336 | 0.327 | −9.9 | — | one |
| 0.70 | 0.8876 | 0.0346 | 0.372 | −4.9 | — | one |
| 0.80 | 0.8920 | 0.0376 | 0.432 | 28.1 | — | one |
| 0.90 | 0.8910 | 0.0373 | 0.431 | 20.7 | — | one |
| 1.00 | 0.8897 | 0.0351 | 0.336 | −11.0 | — | one |

**In the mean, this is a saturating field.** Order climbs from 0.7372 to 0.8922
between alpha 0 and 0.4, then is flat to alpha 1 within 0.005 across six points.
Forty percent of the template's embedding magnitude buys the whole effect. There
is also no dead zone: alpha = 0.1 already gains +0.097. Read as a mean, that is
saturation, not a threshold, and it is what we first concluded.

**The mean is the wrong instrument, and we said so before checking.** A pitchfork
does not predict a shifted mean. It predicts two branches with a moving mixture
weight, and a population splitting between a low and a high branch produces a
perfectly smooth mean. Testing the distribution instead, by bimodality
coefficient (flag at 0.555), Hartigan-style dip, and one- versus two-component
GMM BIC (10 is conventionally very strong):

**Bimodality appears only at alpha = 0.10 and 0.20**, in the steep region below
the knee, on two independent readings at once: BC 0.690 and 0.685, BIC gaps of
133 and 100. Every other field strength is single-moded.

**Fluctuations collapse through the window.** The spread of the order parameter
runs 0.187, 0.096, 0.114, 0.040, then holds near 0.035. A fivefold collapse as
the system commits to one branch.

**The anomaly we dismissed was the signal.** We called alpha = 0.20's mean of
0.7989 noise because it broke monotonicity. It is not noise: alpha = 0.20 carries
more weight in the low branch than alpha = 0.10 does, 0.267 against 0.241, which
is exactly why its mean sits lower. A mixture-weight difference, not measurement
error.

So there is a coexistence window, and it sits where a transition should sit. That
is the branch structure the mean could not see.

**What this is not.** Two flagged points out of eleven, 164 problems each, one
model, one benchmark, one seed. The minority weight moves 0.241 to 0.267, which
is not a clean monotone sweep and is not meaningful at this n. A window this
narrow is barely resolved by a grid of 0.1. What would settle it: a dense grid
across alpha 0.02 to 0.30, repeat seeds to show the branches are not a sampling
artifact, and a second model to show the window is not a quirk of this one.

## Outcome of the NLA test: no branches at any K

Run 2026-09-02 on a fresh 2400-target sample (Qwen2.5-7B, layer 20, seq 64,
per-dim std 1.84), the canonical `RiverRider/srt-nla-av-v1` checkpoint, M = 200,
K = 64. The means reproduce the banked run to within noise: greedy 0.5878 against
0.5860, K-curve 0.585 / 0.621 / 0.652 / 0.686 / 0.724 / 0.758 / 0.793 against
0.573 / 0.617 / 0.652 / 0.686 / 0.716 / 0.747 / 0.777. So this is the same system
the paper describes, and now with the per-target matrix kept.

| K | mean | sd | dip p | BC | BIC gap | branches |
|---|---|---|---|---|---|---|
| 1 | 0.5846 | 0.0892 | 0.955 | 0.463 | 45.0 | one |
| 2 | 0.6208 | 0.0894 | 0.875 | 0.509 | 40.6 | one |
| 4 | 0.6524 | 0.0943 | 0.911 | 0.537 | 48.7 | one |
| 8 | 0.6862 | 0.0974 | 0.769 | 0.508 | 31.2 | one |
| 16 | 0.7237 | 0.1024 | 0.742 | 0.553 | 30.5 | one |
| 32 | 0.7580 | 0.0996 | 0.950 | 0.510 | 10.9 | one |
| 64 | 0.7931 | 0.0982 | 0.846 | 0.443 | −8.0 | one |

| | registered | result |
|---|---|---|
| N1 | bimodal at two contiguous K in {2..32} | **FAIL**, bimodal at none |
| N2 | modes near 0.586 and 0.799 | **FAIL**, nothing to locate |
| N3 | upper-mode weight monotone in K | **FAIL**, no upper mode |
| N4 | unimodal at K = 1 | pass, dip p = 0.955 |

The per-target distribution of best-of-$K$ centred fve is unimodal at every $K$
from 1 to 64, with dip p between 0.74 and 0.96 throughout. The spread barely
moves, 0.089 to 0.102. The whole population slides rightward together as $K$
grows, which is what a monotone max over exchangeable samples does by
construction and is evidence of nothing.

Worth noting what the two broken detectors would have said. BC sits at 0.44 to
0.55, under the flag but only just, and the GMM BIC gap is 31 to 49 at low $K$,
three to five times the threshold we were using this morning. Without the real
dip test gating the call, $K$ = 1 through 16 would have been flagged as two
branches, and we would have reported a pitchfork in the NLA sampling
distribution. It is a skewed unimodal distribution against a hard ceiling, the
same shape that fooled us on the format sweep.

**The section 12 pitchfork reading is unsupported by its own data.** There is no
greedy branch and no paraphrase-manifold branch in the per-target distribution.
There is one population whose max improves smoothly with $K$. The qualitative
observations that motivated the reading remain true (the greedy gap is real,
logp cannot rank candidates, cosine to an anchor can), but they do not need a
bifurcation to explain them and the distribution does not show one.

## Where the framework stands after two closed loops

Two pre-registered tests on two systems, one of which the framework was not
fitted to. Eight predictions. Six fail, and the two that pass are the null
predictions (unimodal at zero field, unimodal at $K$ = 1), which the alternative
explanation also predicts. Nothing the pitchfork reading uniquely predicted
happened.

What survives is what was measured without the framework: the template is a
saturating one-sided field on output order, the effect is domain-conditional,
and best-of-$K$ selection on HumanEval saturates by $K$ = 8. Those results
stand on their own. The bifurcation is an analogy, and on these two systems the
data do not support promoting it past that.

## Outcome: the coexistence window is not supported

Scored against the predictions registered in commit c8e5a454, before the
analysis.

| | registered | result |
|---|---|---|
| P1 | bimodality in a contiguous run inside [0.08, 0.22], at least two of 0.12/0.14/0.16/0.18 | **FAIL**, zero of four |
| P2 | minority weight decreases monotonically across the window | **FAIL** |
| P3 | spread peaks inside the window | **FAIL**, spread declines monotonically, max at alpha 0.02 |
| P4 | the 1.5B shows a window in [0.05, 0.30] | **pass**, but see below |

**The detector was broken and it was flattering the hypothesis.** The dip
function in the first version of `format_field_bimodality.py` returned 0.0030 for
every input. Validated afterwards against known distributions, it gives the same
0.0030 for a unimodal normal, a clean bimodal mixture and a skewed exponential.
It is a constant in $n$ and it never tested anything, while the script's own
docstring called it one of "three independent readings".

The two remaining detectors are not independent either. Both fire on
non-Gaussianity rather than on two modes. A **unimodal** skewed exponential
scores a BIC gap of +43.8 against the >10 threshold used here. At low field the
per-problem order distribution has sd near 0.19 against a hard ceiling at 1.0,
which is exactly the geometry that produces left skew.

With a real Hartigan dip gating the call, the two "TWO branches" points that
motivated this section evaporate: alpha 0.10 and 0.20 give dip p = 0.946 and
0.736. Their BC and BIC values are unchanged and still look impressive. They were
measuring skew.

On the dense 25-point grid only alpha 0.04 and 0.06 survive, and **25 dip tests at
p < 0.05 expect 1.25 false positives by chance**. Two is the false-positive rate.

**P4 passes as written, and the wording was the flaw.** On the 1.5B the dip test
finds genuine bimodality at alpha 0.10, 0.15 and 0.20, all at p = 0.000 and all
inside the registered interval. But alpha = 0.00 is bimodal too, at p = 0.023.
Two branches with no field at all. A transition has to *create* the split; here
the split pre-exists and the field destroys it, with sd collapsing from 0.244 to
0.064 between alpha 0.20 and 0.25. P4 should have required unimodality at zero
field, and did not. That is the population-heterogeneity confound we registered
against the NLA test and failed to guard against in our own.

**The two models also disagree.** Same test, same interval: the 3B flags none of
0.12/0.14/0.16/0.18, the 1.5B flags 0.10/0.15/0.20 at p = 0.000. Whatever the
1.5B is showing is not a property of the template.

**The standing reading is the one from before the bimodality detour: a saturating
field.** Order rises from 0.737 to 0.892 and flattens past alpha ~= 0.3, spread
collapses monotonically from 0.19 to 0.035, and no field strength on the 3B shows
two branches. Roughly 40% of the template's embedding magnitude buys the whole
effect.

Per the pre-registration we are not relocating the window, switching order
parameter, or dropping the 3B as a different regime.

## Pre-registration, written before the dense grid was analysed

This note has changed its mechanism six times in one day, and every version fitted
the data it was shown:

1. the format effect grows with scale (published), killed by MBPP
2. it is susceptibility, gain tracks baseline disorder, r = -0.870
3. it is a fixed endpoint near 0.89, killed by MBPP the same afternoon
4. it is a one-sided floor at 0.88
5. it is a saturating field and not a bifurcation, from the mean
6. there is a two-branch coexistence window, from the distribution

A frame that accommodates every result is not being tested by any of them. Some
of those turns were genuine corrections forced by new data, but the pattern is
also exactly what motivated reasoning looks like from the inside, and we cannot
tell the two apart from where we are standing. So the predictions below are
recorded before the analysis, and we are bound by them.

**Provenance, stated exactly.** At the time of writing, generations exist on disk
for 5 of the 14 new alpha values. None have been embedded, scored or inspected.
This is pre-registration of the analysis, not of the data collection, and it is
weaker than the real thing. Say so when reporting.

**If the coexistence window is real:**

- **P1, contiguity.** Bimodality (BC > 0.555 and BIC1−BIC2 > 10) appears in a
  contiguous run of alphas inside [0.08, 0.22], including at least two of
  0.12, 0.14, 0.16, 0.18.
- **P2, mixture sweep.** Minority weight *decreases* monotonically as alpha rises
  across the window, since raising the field should move mass into the ordered
  branch.
- **P3, fluctuation peak.** The standard deviation of per-problem order peaks
  *inside* the window and falls on both sides.
- **P4, replication.** The 1.5B shows a window somewhere in [0.05, 0.30].

**Two of these already look bad, and we are saying so first.** On the coarse grid
the minority weight ran 0.241 at alpha 0.10 and 0.267 at alpha 0.20, which is the
wrong direction for P2. And the largest spread was 0.1865 at alpha = 0.00, at the
edge rather than inside the window, which is the wrong shape for P3. We described
both as "not meaningful at this n" when we first saw them. Under P2 and P3 they
are predictions, and if the dense grid reproduces them the window is not a
mixture sweep.

**Falsifiers, any one of which we will report as such:**

- no bimodal points at all in 0.12 to 0.18: the original two were flukes
- bimodal points scattered outside [0.08, 0.22]: the flag is tracking noise, not a
  transition
- minority weight not monotone across the window: not a mixture sweep
- the 1.5B shows no window: an artifact of one model rather than a property of the
  template

**What we will not do.** We will not rescue the hypothesis by relocating the
window after seeing where the flags land, nor by switching order parameter, nor
by dropping the 1.5B as "a different regime". If the result is mixed we will
report it as mixed and leave the claim unsupported.

## Second pre-registration: closing the loop on NLA best-of-K

The framework above was built on the format data and has been fitted to it six
times. The only way to make it pay rent is to predict something on a system it
was not built from, and state the prediction first.

`paper_nla.md` section 12 already reads best-of-$K$ as a pitchfork control
parameter, with a below-threshold branch at the greedy/modal $\hat{x}$ and an
above-threshold branch on the paraphrase manifold. That is a bimodality claim.
It has never been tested distributionally. If the coexistence-window reading is
real rather than a story we tell about smooth curves, it should predict one here
too.

**A units error in Figure 12 that has to be fixed either way.** The caption gives
the low branch at "centred cosine 0.59 to 0.63" and the high branch at "0.99".
Those are different units. In centred fve on the same backbone the published
values are: random floor 0.510, greedy 0.586, paraphrase-all 0.625, NN-in-pool
0.663, paraphrase-best 0.799, replay 0.968. The 0.99 is $\rho_{\text{cen}}$, the
fraction of the paraphrase ceiling, not centred cosine. **The high branch in
centred units is 0.799, not 0.99.** The banked K-curve, oracle top-1 centred:
K=1 0.573, K=2 0.617, K=4 0.652, K=8 0.686, K=16 0.716, K=32 0.747, K=64 0.777.

**What we cannot do, and it is our own fault.** None of the banked NLA artifacts
carry per-target arrays. `rerank_eval_*.json` and `oracle_ceiling_*.json` hold
means and a few percentiles only. We saved exactly the statistic that cannot see
branch structure, which is the same error we nearly made this morning. Testing
this requires regenerating per-target centred cosines.

**Predictions, before any per-target NLA data is generated or inspected:**

- **N1.** The per-target distribution of best-of-$K$ centred fve is bimodal, by
  the same criteria used for the format sweep (BC > 0.555 and BIC1−BIC2 > 10), for
  at least two contiguous values of $K$ in {2, 4, 8, 16, 32}.
- **N2.** The two modes sit near 0.586 (greedy) and near 0.799 (paraphrase-best),
  within 0.05. Not near 0.99.
- **N3.** The weight in the upper mode increases monotonically with $K$.
- **N4.** At $K$ = 1 the distribution is unimodal, since with one sample there is
  nothing to select across.

**The alternative explanation we are obliged to rule out, and expect to lose to.**
Bimodality across targets can simply mean some targets are easy to verbalise and
some are not, which is heterogeneity in the population and not a transition in a
control parameter. This is a serious confound and it is the more likely
explanation on priors. The discriminating test is N4 plus the shape of the change:
target heterogeneity predicts bimodality already present at $K$ = 1 and merely
translating rightward as $K$ grows, whereas a transition predicts unimodality at
$K$ = 1 and a window that opens and then closes.

**Also note best-of-$K$ max is monotone in $K$ by construction**, so the
distribution will shift right no matter what. A rightward shift is not evidence.
Only the splitting is.

**Falsifiers.** Unimodal at every $K$; or bimodal at $K$ = 1 (heterogeneity, not
transition); or modes not where N2 says; or upper-mode weight not monotone. Any of
these and we report that the section 12 pitchfork reading is unsupported by its
own data, and say so in the paper rather than in a footnote.

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

1. **Done twice, and the second run corrected the first.** The per-problem test
   holds the relation inside one benchmark; the MBPP repeat showed the endpoint is
   a floor rather than a destination. Both are above.

2. **A continuous control parameter. Done, and it did not show a threshold.**
   The field sweep above interpolates the template's embedding contribution from
   0 to 1. The response is a saturating curve, flat past alpha ~= 0.3, with no
   two-branch structure on the 3B once a valid unimodality test is used. The
   1.5B shows a pre-existing split that the field erases, which is heterogeneity
   and not a transition. If the pitchfork is to be more than an analogy for this
   system, the control parameter is not template strength.

3. **Where the floor sits, and whether it is one number.** The floor reads at
   roughly 0.88 on both benchmarks, but eight bins cannot separate "one constant"
   from "two similar constants". A third domain would.

4. **The negative cell.** MBPP's most ordered bin goes 0.9815 to 0.9700 under the
   template. If the template genuinely reduces order in already-committed cases,
   that is a second, opposite effect and worth isolating rather than rounding to
   zero.

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
    artifacts/nla/format_susceptibility_perproblem.json       HumanEval per-problem
    artifacts/nla/format_susceptibility_perproblem_mbpp.json  MBPP per-problem
    scripts/coder_matrix_vllm.py                         generation
    scripts/coder_ladder_analyze.py                      aggregate similarity
    scripts/format_susceptibility_perproblem.py          per-problem test

Both matrices: 47,232 generations each on HumanEval, same encoder, same
different-prompt floors, bfloat16, K = 8, top_p 0.9, temperature 1.0.
