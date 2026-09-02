---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - interpretability
  - model-convergence
  - homogeneity
  - code-generation
  - evidence
  - reproducibility
size_categories:
  - 100K<n<1M
---

# SRT hivemind artifacts

Every number in *Where the Hivemind Comes From* has a file here. This repository is
the evidence, not a model: sampled continuations, hidden states, fitted-map scores,
execution outcomes and run logs.

Paper: [`paper_hivemind.md`](https://github.com/space-bacon/SRT/blob/main/paper_hivemind.md).
Code: [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT).

**Reproduction principle.** If a claim is in the paper and you cannot rebuild it from
the files here plus the scripts in the GitHub repo, that is a bug and we want to hear
about it. Four claims were corrected during the work by exactly that process, and each
correction is recorded below rather than quietly overwritten.

**What the paper argues.** Jiang et al. (arXiv:2510.22954) measured 70+ closed models
writing alike. Open weights let us take that apart: the convergence is not inherited
from pretrained geometry, is not mostly bought by instruction tuning, and largely
arrives with the deployment format. It is also suppressible, by a lever nobody pulls.

---

## Layout

### `atlas/` — representation transport across 12 models from 8 labs

| file | what it is |
|---|---|
| `openweight_transport_atlas.json` | the 12x12 ridge-map retrieval matrix, COCO captions |
| `states/*.npz` | mean-pooled hidden states at 60% depth, one per model |
| `hivemind_sampled_protocol.json` | base models under the original study's sampling settings |
| `hivemind_posttraining_isolation.json` | six matched base/instruct pairs, raw and chat arms |
| `hivemind_template_decomp.json` | the five-arm template decomposition |
| `hivemind_suppression.json` | persona arms, exotic and deployed |
| `hivemind_mechanism_link.json` | transport against output similarity |
| `samples/`, `samples_instruct/` | the continuations behind all of the above |

Held-out retrieval through a fitted map reaches **0.9181** across lab boundaries against
a shuffled floor of **0.00101** and a self-map ceiling of **0.999**. Transport cost is
**0.0787**. Shared corporate lineage is worth only 0.0357 of that.

### `coder_matrix1024/` — the scale test, Qwen2.5-Coder 0.5B to 32B

36 arms, 164 HumanEval prompts, K=8, 1024 new tokens. `scaling_curve.json` holds the
per-rung table.

The format effect **grows** with scale, from 0.0878 at 0.5B to **0.1589** at 32B, a
slope of **+0.0433 per decade** over a 66x range. The tuning term moves the other way
at **−0.0402**.

`coder_ladder/` is the same matrix at 192 new tokens, where 43% to 80% of candidates
were cut off mid-function. Its similarity slope (+0.0441) matches the clean run; its
pass rates do not, and every pass-rate figure below reads from the 1024-token pools.

### `ministral_ladder/` — independent replication, Mistral 3B/8B/14B

18 arms of prose stems. Role structure alone is worth **+0.3809 to +0.4354**, persona
alone **+0.0612 to +0.0821**, and instruction tuning alone is slightly *negative*. The
format effect rises at **+0.0348 per decade**, reproducing the Qwen sign independently.

### `ministral_suppression/`, `crosslab_suppression/` — can it be undone

Eight genuinely distinct personas take inter-model similarity from **0.6464** to
**0.3817**. Eight production-style personas, each a rewording of "helpful assistant",
leave it at **0.6441**. Across three labs at 14B-31B the deployed arm is worse than
inert: inter-model similarity *rises*, an inter_drop of **−0.0111**.

### `mbpp_ladder/`, `ensemble/`, `verifier_1024/`, `code_select/` — what the pool is worth

Candidate pools and execution outcomes. `verifier_1024/` holds the HumanEval read
comparison on the 1024-token pools and copies of the MBPP reads; `verifier/` is the
superseded 192-token HumanEval run; `ensemble/` holds six frontier models from five labs.

| read | HumanEval | MBPP |
|---|---:|---:|
| random single | 0.4679 | 0.7185 |
| learned verifier | 0.5102 | 0.7292 |
| agreement among candidates | 0.5854 | 0.8094 |
| filter by the prompt's examples | **0.6496** | **0.8492** |
| that filter plus the verifier | 0.6585 | 0.8485 |
| oracle | 0.7346 | 0.8887 |

### `holonomy/` — a retracted selector, kept

The transport-defect selector and the sweep that killed it. Retained because the
retraction is the useful part.

---

## Using the selector rather than reading about it

The winning read is packaged as `srt_select` in the source repository. It picks one
of K replies using only the replies: no reference solution, no test suite, no scoring
model, no training.

```python
from srt_select import select
best = select(user_message, replies)
```

Handed the benchmark's entry point and signature it scores 0.5854 on HumanEval and
0.8094 on MBPP, against floors of 0.4679 and 0.7185. Recovering the entry point and
the argument shapes from the chat turn alone is the deployable case, and it costs
coverage rather than accuracy: 62.3% of HumanEval problems resolve and 98.6% of MBPP
ones do, giving 0.5476 and 0.7991 when unresolved problems are scored as an arbitrary
pick. `verifier_1024/chat_consensus.json` and `verifier/chat_consensus_mbpp.json` are
those two runs.

Selecting means executing every candidate, so it is a remote code execution primitive
if pointed at the wrong host. `srt_select.sandbox` states what the confinement does
and does not cover. Run it somewhere you are willing to lose.

---

## Reading these files correctly

**`ensemble/union_ceiling.json` is superseded and carries a `SUPERSEDED` key.** Use
`union_ceiling_fixed.json`. In the first run GLM-4.7-Flash was rendered with its
`<think>` block open, so it spent its whole budget on analysis and never emitted code.
Its pass@1 there (0.0526) is a render bug, not a capability. Corrected it is **0.7904**,
the union moves to **0.9878** against a best single model of **0.9634**, and GLM turns
out to contribute the only 2 problems no other model solves. The first run reported
zero unique solves for every model, which was an artifact of that one broken member.

**The union ceiling is an oracle, and a selector does not reach it.** `pooled_select.json`
runs real selection over all 48 pooled candidates from the six frontier members. It scores
0.9390, solving the same 154 of 164 problems as the best member's own eight candidates,
for a gain of **+0.0000**. Recovering the target from the replies alone finds one more
problem, which is noise at this size. Read 0.9878 as a bound on what better selection
could win, never as an ensembling result: pooling models buys nothing that one good
model plus a selector does not already give.

**Example-filtering leads on both benchmarks, and agreement is the coverage play.**
Filtering by the prompt's stated examples recovers 68.1% of the gap on HumanEval and
76.8% on MBPP; agreement recovers 44.1% and 53.4%. MBPP states an example for 425 of
425 problems; HumanEval states one for 128 of 164, and agreement covers the other 36.
On the superseded 192-token HumanEval pools agreement appeared to lead at 82.9%. That
ordering was a truncation artifact: agreement was discarding code that could not run,
against a floor that included it.

**`consensus*_covered_only_superseded.json` carry a `SUPERSEDED` key.** Those runs
averaged the agreement read over the problems it could run while the floor and oracle
were averaged over all of them, so 12 of 36 HumanEval arms appeared to beat their own
oracle. The files without the key score every problem: `consensus` falls back to the
first reply where nothing runs, as `srt_select.select()` does, and `consensus_strict`
charges those problems as failures. On the 192-token pools the correction is 0.4426 to
0.3762; on the 1024-token pools 0.6301 to 0.5854; on MBPP 0.8174 to 0.8094. Found by
Dipankar Sarkar from the published artifact.

**The learned verifier does not transfer.** It captures 15.9% of the gap on HumanEval
and 6.3% on MBPP, and on MBPP appending it to example-filtering makes that baseline
slightly worse. Its AUROC is 0.778 on both budgets; a length-only control that scored
0.581 on the 192-token pools scores 0.526 on the clean ones. It is reported as a
negative result.

**Selection value decays with scale**, at −0.0704 per decade on MBPP, from +0.1538 at
3B to +0.0097 at 32B, and at −0.0676 per decade on the 1024-token HumanEval matrix. It
is a weak-model amplifier and does not push a strong model past its own ceiling. The
MBPP slope is the chat-native read; agreement gives −0.0830 and execution-guided
filtering −0.1192, so the direction is read-independent and the quoted figure is the
most conservative of the three. See `hivemind_census.json`.

**Ladder magnitudes are not comparable across domains.** Code prompts constrain output
far more than open-ended stems, so instruct-raw already sits at 0.74-0.79 on HumanEval
against 0.4435 on prose. Compare slopes across domains, not levels.

**The Ministral chat arm was regenerated.** Its first version is kept under
`corrupt_chat_pre_fix/`. `apply_chat_template(tokenize=False)` does not round-trip
special tokens through Mistral's tokenizer backend, so the model read its own control
tokens as prose. Anything in that directory is a record of the bug, not data.

---

## Scale

110,704 generations across 97 arms, 12 models with saved states, 6 frontier models in
the ensemble, two code benchmarks with execution outcomes from a harness validated at
164/164 and 425/425 on canonical solutions.

## Citation

```bibtex
@misc{srt_hivemind_2026,
  title  = {Where the Hivemind Comes From: Geometry, Tuning and Format,
            Separated on Open Weights},
  author = {SRT},
  year   = {2026},
  url    = {https://github.com/space-bacon/SRT}
}
```
