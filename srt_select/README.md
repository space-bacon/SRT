# srt_select

Pick one of K model replies to a coding request, using only the replies.

No reference solution, no test suite, no scoring model, no training. Run every
candidate on the same synthesised inputs, group them by what they compute, and
return a member of the largest group.

```python
from srt_select import select, choose

best = select(user_message, replies)          # the chosen reply

pick = choose(user_message, replies)          # or the decision itself
pick.index, pick.entry, pick.ran, pick.clusters, pick.agreed
```

```
echo '{"prompt": "write is_even(n)", "replies": ["...", "..."]}' | python -m srt_select
```

## What it is worth

Against a random-single-draw floor and an any-candidate-passes oracle, over the
generation pools in `artifacts/nla/`:

| | problems | arms | floor | selected | oracle |
|---|---:|---:|---:|---:|---:|
| HumanEval | 164 | 36 | 0.4679 | **0.6301** | 0.7346 |
| MBPP | 425 | 10 | 0.7185 | **0.8174** | 0.8887 |

That is +0.1622 and +0.0989 over the floor, which is 60.8% and 58.1% of the
headroom an oracle would capture. HumanEval pools are generated at 1024 new
tokens (`artifacts/nla/coder_matrix1024/`); an earlier 192-token run put the
selected score at 0.4426 against a 0.1868 floor, and that 82.9% share was
inflated by truncated candidates that could not execute.

Those two rows are handed the benchmark's entry point and signature. Recovering
both from the chat turn alone is the deployable case, and it costs coverage
rather than accuracy:

| | resolved | selected, unresolved scored as an arbitrary pick | floor |
|---|---:|---:|---:|
| HumanEval | 62.3% | 0.5476 | 0.4679 |
| MBPP | 98.6% | 0.7991 | 0.7185 |

HumanEval resolves on under two thirds of its problems because its prompts are
docstrings that often state no example and carry no annotations, so there is
nothing to synthesise arguments from. MBPP states an example call for almost
every problem, which is why it resolves on 98.6%. Coverage is the axis to
improve, not the ranking.

## What it is not

**It is not a correctness check.** It returns the pool's majority opinion. When
the pool is confidently wrong together, so is this. `pick.cluster_size` is the
number of candidates that agreed, and a cluster of one is a coin flip wearing a
result's clothes.

**It does not help a model that is already good.** Selection value decays with
scale at −0.0704 per decade: +0.1538 at 3B, +0.0097 at 32B. This buys the most
where the pool is weakest and diverse, and nearly nothing at the top.

**Pooling several models into one pool buys nothing.** Over 48 candidates from
six frontier models across five labs, selection scores 0.9390 on HumanEval,
solving the same 154 of 164 problems as the best single member's own eight. The
gain over that member plus its own selector is +0.0000. A diverse pool does hold
more correct answers, 162 of 164 by oracle, and agreement cannot find them,
because agreement returns the majority and extra members mostly add votes for
what the pool already believed. Point this at one model's samples, not at an
ensemble.

**More probe inputs do not help.** `n_cases` defaults to 2 because that is what
the sweep supports: from 2 to 48 inputs, accuracy does not move at any K, while
cluster purity climbs 0.7076 to 0.8909. The sharper probe genuinely does split
clusters that mix passing and failing candidates, and the candidates it
separates are not the ones that pass. Raising it only costs coverage, since
every extra input is another chance for a candidate to crash.

**A learned verifier did worse.** Fitting a classifier on 47,232 execution
labels reached 0.5102 on HumanEval against this method's 0.6301, and it did not
transfer to MBPP. Executing the pool beat learning to score it, on both
benchmarks.

**Execution-guided filtering leads where it applies; agreement covers the rest.**
Filtering by the prompt's stated examples reaches 0.6496 on HumanEval and 0.8492
on MBPP against this method's 0.6301 and 0.8174. It applies to 128 of 164
HumanEval problems and 425 of 425 MBPP problems; agreement applies to every
problem, which is what it buys. Use the filter when the prompt states an
example and fall back to agreement when it does not. That combination is not
measured here; the filter plus a learned verifier reaches 0.6585 on HumanEval.

## What K actually costs

K samples do **not** cost K times one answer, and misreading this is what makes
selection look undeployable when it is not.

Autoregressive decode is memory-bandwidth bound. Every decode step streams the
weights out of HBM once, and that cost is identical whether the step advances one
sequence or thirty-two. Prefill is paid once and shared across the K samples. So
K samples in a single batch cost far less than K separate answers, and the
marginal sample is close to free until K spills past one batch.

The curve has a knee, not a slope, and where the knee sits depends on the card
and the model. Measure it on your own hardware with `scripts/k_latency.py` before
choosing K.

Two caveats. Under concurrency, batching K samples for one user eats batch
capacity that would otherwise serve others, so it is nearly free for a single
user and a real throughput cost under load. And because selection needs the
samples before it can choose, it cannot move a published `pass@1` figure, which
is by definition single-sample.

## Running untrusted code

Selecting requires executing every candidate, so this is a remote code execution
primitive if you point it at the wrong host. `srt_select.sandbox` documents the
confinement in full: a fresh `-I` interpreter, clamped address-space, CPU and
file-size limits, a stripped environment and a wall clock.

Read that module before deploying. It is confinement, not a security boundary.
Run it somewhere you are willing to lose.

## Relationship to `scripts/`

`scripts/chat_consensus.py` and `scripts/consensus_select.py` are the harness
the published numbers came from and stay as the reproduction record. This
package is the deployable extraction of it.
`tests/test_srt_select.py::test_package_agrees_with_the_harness_it_was_extracted_from`
pins the two together so they cannot drift apart silently.

One deliberate difference: the package applies the resource guard to the probe
step, which the harness applied only when executing test suites. A candidate
killed by the guard is counted as one that did not run, so the guard can lower
coverage and can never raise the score.

## Provenance

Generations, pass matrices and result files:
[`RiverRider/srt-hivemind`](https://huggingface.co/datasets/RiverRider/srt-hivemind).
Method and controls: `paper_hivemind.md` §5.4.
