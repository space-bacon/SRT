# Where the Hivemind Comes From: Geometry, Tuning and Format, Separated on Open Weights

## Abstract

Jiang et al. (arXiv:2510.22954) measured 70 or more language models at the output
layer and found them writing alike: different models answering the same
open-ended query score 0.71 to 0.82 in pairwise similarity, and 79% of queries
produce intra-model similarity above 0.8 across 50 resamples. Almost every model
in that study is a closed endpoint, so text is the only observable. The result
establishes that models agree without being able to say why.

Open weights let us take the question apart. We report three measurements.

First, representations are mutually recoverable. On 12 open-weight models from 8
labs, a ridge map from one model's hidden states to another's retrieves the right
held-out item 0.9181 of the time across lab boundaries, against a shuffled floor
of 0.00101 and a self-map ceiling of 0.999. Shared corporate lineage is worth only
0.0357 of that.

Second, base models do not reproduce the reported level. Under the original
study's own sampling settings, our base models reach intra-model 0.3644 and
inter-model 0.3401 on a floor of 0.0993 that matches theirs, and zero of 720
model-prompt cells clear 0.8. The floors agree while the signal differs by more
than a factor of two, so this is not a scale artifact.

Third, and decisively, we recover their level and isolate its cause. Using six
matched base/instruct pairs, holding pretrained weights, prompts, decoding and
scorer fixed, instruction tuning alone raises intra-model similarity by 0.0786.
The same tuned weights prompted through the model's own chat template raise it by
0.3623, reaching 0.7272, with four of six models exceeding 0.80 and reproducing
the band reported for frontier systems from models of 0.6B to 2B. **The prompt
format does roughly 4.6 times the work of the tuning.**

Decomposing that template gives a fourth result. The system persona contributes
nothing on its own, moving similarity by −0.0120. Generic role markers that no
model was ever trained on reproduce 66% of the full template effect, and each
model's own tuned tokens supply the remaining 34%. What drives the convergence is
the assistant turn structure, not what the system prompt says.

Two extensions close the obvious objections. Across the Qwen2.5-Coder ladder from
0.5B to 14B, one lab and one recipe held fixed, the format effect **grows** with
scale at +0.0473 per decade of parameters while the tuning term declines at
−0.0285, so the split tilts further toward format as models get bigger. On
Ministral-3 from an eighth lab, at 3B, 8B and 14B, the decomposition reproduces
with role structure worth +0.3809 to +0.4354, persona worth +0.0612 to +0.0821,
instruction tuning alone slightly negative, and the format effect rising at +0.0348
per decade. In that family generic markers recover 94% to 104% of the full template
effect, so the 66/34 split between framing and native tokens is a property of the
original six models rather than a constant.

A final measurement bounds what any of this licenses. Scoring the same code
generations with an execution harness, an arm at 0.8638 intra-model similarity
passes only 0.5404 of the time per sample while its 8-sample pool contains a
passing solution 0.9024 of the time. Convergence in phrasing is not convergence in
correctness.

Finally we test whether the effect can be undone. Eight genuinely distinct personas
cut inter-model similarity from 0.6464 to 0.3817 and intra-model from 0.7272 to
0.4533, so the convergence is not locked into the weights. Eight personas of the
kind actually deployed, each a rewording of "helpful assistant", move inter-model
similarity by 0.0023. **The lever works and nothing shipped pulls it.**

The homogeneity is not inherited from pretrained geometry and is not mostly bought
by instruction tuning. It largely arrives with the deployment format, and most of
that is reachable by prompt convention alone. We also report negatives, including
a published null that later reversed and a selector result that did not survive
our own replication.

---

## 1. The question

"Artificial Hivemind: The Open-Ended Homogeneity of Language Models" is a NeurIPS
2025 Datasets and Benchmarks best paper by Liwei Jiang, Yuanjun Chai, Margaret Li,
Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak
and Yejin Choi. It introduces Infinity-Chat, 26,070 genuinely open-ended queries
mined from real user traffic in WildChat, and measures similarity within and
between model outputs.

The measurements are careful. Similarity is cosine over sentence embeddings from
`text-embedding-3-small`. Crucially they report a floor: randomly paired responses
from the global pool fall 100% within 0.1 to 0.2. Their limitations section
concedes that embedding similarity "may lack sufficient expressiveness," and their
Table 13 shows two visibly different 30-word essays scoring 0.737.

What the study cannot do is explain the agreement. Output cosine sits downstream
of the thing it characterises. If two models write alike, a sentence embedding of
their text cannot distinguish "these systems compute similar internal
representations" from "these systems were tuned toward similar behaviour" from
"these systems were prompted into the same role." Almost every model they study is
served behind an API, so this is a limit of access, not an oversight.

Open weights remove that limit, and the three candidate explanations turn out to
have very different answers.

## 2. Setup

Twelve models, eight labs, run locally on one desktop machine.

| tag | model | lab | dim | tap layer |
|---|---|---|---:|---:|
| qwen25_05b | Qwen/Qwen2.5-0.5B | Alibaba | 896 | 14 |
| qwen3_06b | Qwen/Qwen3-0.6B-Base | Alibaba | 1024 | 17 |
| gemma2_2b | google/gemma-2-2b | Google | 2304 | 16 |
| llama32_1b | meta-llama/Llama-3.2-1B | Meta | 2048 | 10 |
| llama32_3b | meta-llama/Llama-3.2-3B | Meta | 3072 | 17 |
| tinyllama | TinyLlama-1.1B-Chat-v1.0 | TinyLlama | 2048 | 13 |
| olmo2_1b | allenai/OLMo-2-0425-1B | AI2 | 2048 | 10 |
| smollm2_360m | HuggingFaceTB/SmolLM2-360M | HuggingFace | 960 | 19 |
| pythia_410m | EleutherAI/pythia-410m | EleutherAI | 1024 | 14 |
| qwen25_7b | Qwen/Qwen2.5-7B | Alibaba | 3584 | 17 |
| gptoss_20b | openai/gpt-oss-20b | OpenAI | 2880 | 14 |
| qwen25_7b_f32 | Qwen2.5-7B at fp32 | Alibaba (control) | 3584 | 17 |

States are mean-pooled hidden states at 60% relative depth, over 5,000 COCO
val2017 captions split 4,000 train and 1,000 held out.

The last row is a control, not a model: the same Qwen2.5-7B weights at fp32
instead of bf16. Identical weights must top every axis we measure, or the
instrument is broken.

### 2.1 Measurement discipline

Every claim carries a floor computed through the same machinery that produced the
effect. Shuffled-label floors, not analytic chance.

States are centered on the training mean before every ridge solve and every
cosine. Section 9 shows why that is not optional.

Transport is scored by held-out retrieval rank, which is scale-free, rather than
by cosine, which is not.

Where we compare against the original study we report our floor beside theirs. A
number is comparable to another number only if the floors agree, and Sections 4
and 5 both turn on that point.

## 3. Representations transport regardless of lineage

For every ordered pair, fit a ridge map from A's states to B's on 4,000 training
captions, then push each of 1,000 held-out states through the map and retrieve the
matching item among B's states. Chance is 0.0010.

| | value |
|---|---:|
| self-map diagonal | 0.9990 |
| transported, off-diagonal | 0.9203 |
| transport cost | 0.0787 |
| shuffled floor | 0.00101 |
| within-lab | 0.9539 |
| cross-lab | 0.9181 |
| within minus cross | 0.0357 |

The best off-diagonal pair is Qwen2.5-0.5B to OLMo-2-1B at 0.999, indistinguishable
from the self-map ceiling. Alibaba to AI2, different architecture, tokenizer and
corpus, and a linear map recovers the right caption 999 times in 1000. The worst
pair is gpt-oss-20b to Qwen2.5-7B at 0.735, still 735 times the floor.

Shared lineage is worth 0.0357. That is the entire premium for being the same
company with the same architecture family and tokenizer, against a cross-lab level
of 0.9181. Whatever these models share, they did not get it from each other.

The precision control returns 0.999 in both directions, so numerics cost nothing
at this resolution.

## 4. Base models do not reach the reported level

Sampling settings are theirs: top-p 0.9, temperature 1.0, K = 8 samples per prompt
per model, 60 shared open-ended prompts, 5,760 samples. Outputs are scored with
`all-MiniLM-L6-v2`, chosen because its different-prompt floor of 0.0993 sits
inside the 0.1 to 0.2 band their encoder produces. Matching the floor is what
makes the levels comparable.

| arm | ours | theirs |
|---|---:|---|
| floor, different prompts | 0.0993 | 0.1 to 0.2 |
| intra-model, own resamples | 0.3644 | above 0.8 on 79% of queries |
| inter-model, same prompt | 0.3401 | 0.71 to 0.82 |
| intra minus inter | +0.0244 | overlapping |

Two things happen, and they must not be run together.

**The structure replicates.** Their most striking observation is that a model's
own resamples are barely more similar to each other than to a different model's
output. We see the same shape, 0.3644 against 0.3401. Resampling explores almost
nothing that distinguishes a model from its competitors.

**The level does not.** Our floor is 0.0993 and theirs is 0.1 to 0.2, so the
measurement scales agree, yet our intra-model similarity is less than half theirs.
Per-model intra runs from 0.3087 for Pythia-410m to 0.3998 for TinyLlama.
**Zero of 720 model-prompt cells clear 0.8.**

If the encoder or the scale were responsible the floors would differ. They do not.
The gap is in the signal, and Section 5 finds where it comes from.

## 5. The level is reproduced, and the format does most of the work

Six matched pairs. Every instruct model is the sibling of a base model already
measured, so pretrained weights, prompts, decoding and scorer are all fixed and
only post-training and prompt format move.

| tag | base | instruct | lab |
|---|---|---|---|
| qwen25_05b | Qwen2.5-0.5B | Qwen2.5-0.5B-Instruct | Alibaba |
| qwen3_06b | Qwen3-0.6B-Base | Qwen3-0.6B | Alibaba |
| gemma2_2b | gemma-2-2b | gemma-2-2b-it | Google |
| llama32_1b | Llama-3.2-1B | Llama-3.2-1B-Instruct | Meta |
| smollm2_360m | SmolLM2-360M | SmolLM2-360M-Instruct | HuggingFace |
| olmo2_1b | OLMo-2-1B | OLMo-2-1B-Instruct | AI2 |

Two arms, because instruction tuning and chat formatting are different things.
The `raw` arm feeds each stem byte-identically to the base run, so the only
difference is the weights. The `chat` arm routes the same stem through the model's
own template, which is how these models are deployed and how the original study
prompts its subjects.

| arm | intra | inter | intra − inter | floor | models with any prompt above 0.8 |
|---|---:|---:|---:|---:|---:|
| base, 6 matched | 0.3649 | 0.3473 | 0.0175 | 0.0972 | 0 of 6 |
| instruct, raw prompt | 0.4435 | 0.4045 | 0.0390 | 0.1125 | 0 of 6 |
| **instruct, chat template** | **0.7272** | **0.6464** | 0.0808 | 0.1028 | **6 of 6** |
| theirs | above 0.8 | 0.71 to 0.82 | overlapping | 0.1 to 0.2 | 79% of queries |

Deltas against the matched base arm:

| | intra | inter | floor |
|---|---:|---:|---:|
| instruction tuning alone | +0.0786 | +0.0572 | +0.0153 |
| tuning through the chat template | **+0.3623** | +0.2991 | +0.0056 |

**The format does roughly 4.6 times the work of the tuning.** Identical weights
sit in both instruct rows. The only difference is whether the prompt passes
through the chat template, and that single change moves intra-model similarity
from 0.4435 to 0.7272.

The floor moves +0.0056 between those two arms, so this is not a scale artifact.

Per model, base to raw to chat:

| model | base | raw | chat | share of prompts above 0.8, chat |
|---|---:|---:|---:|---:|
| OLMo-2-1B | 0.3921 | 0.5125 | **0.8253** | 0.73 |
| Llama-3.2-1B | 0.3750 | 0.4967 | **0.8196** | 0.75 |
| gemma-2-2b | 0.3485 | 0.4812 | **0.8101** | 0.60 |
| Qwen3-0.6B | 0.3422 | 0.3666 | **0.8064** | 0.58 |
| Qwen2.5-0.5B | 0.3385 | 0.3597 | 0.6577 | 0.15 |
| SmolLM2-360M | 0.3928 | 0.4442 | 0.4442 | 0.02 |

Four of six exceed 0.80 mean intra-model similarity, reproducing the band the
original study reports for frontier systems, using models between 0.6B and 2B
parameters. Whatever produces the hivemind does not require frontier scale.

SmolLM2-360M is the exception and worth stating rather than hiding. Its chat arm
matches its raw arm to four decimals, with only 2% of prompts above 0.8. We
checked this was not a template failure: all six raw and chat generation sets
differ by hash, and all six tokenizers carry genuine templates. At 360M it appears
to lack the capacity to hold the assistant role the template asks for. The effect
therefore has a capacity threshold somewhere below 0.5B.

Both arms were necessary. A chat-only design would have concluded that
post-training causes the homogeneity. A raw-only design would have concluded that
post-training barely matters. Both conclusions are wrong.

### 5.1 Which part of the template does it

A chat template bundles a system persona, role structure and native special tokens
trained during tuning. Section 5 cannot say which one carries the effect.

We separate them with five arms, each a plain string transformation of the same
stem. Editing six different Jinja templates would have been fragile, and any
difference here is attributable to the transformation. The `shared` arms use
markers no model was ever trained on, identical across all six.

| arm | what the model sees | intra | inter | floor | gain over raw |
|---|---|---:|---:|---:|---:|
| raw | the bare stem | 0.4435 | 0.4045 | 0.1125 | |
| persona | `You are a helpful assistant.` then the stem | 0.4315 | 0.3958 | 0.1076 | **−0.0120** |
| shared | `### User:` / `### Assistant:` | 0.6018 | 0.5186 | 0.0976 | **+0.1583** |
| shared_persona | persona plus those markers | 0.6310 | 0.5775 | 0.0961 | **+0.1875** |
| chat | the model's own template | 0.7272 | 0.6464 | 0.1028 | **+0.2837** |

The total effect of +0.2837 decomposes as:

| component | intra gain |
|---|---:|
| persona text alone | −0.0120 |
| role structure alone | +0.1583 |
| persona added within a role structure | +0.0292 |
| native tuned tokens | +0.0962 |

**The persona is not the cause.** Telling a model it is a helpful assistant, as
plain text, moves intra-model similarity by −0.0120. That is nothing, and slightly
the wrong way. This is worth stating plainly because the intuitive explanation for
the hivemind is that every lab writes the same system prompt. Measured, that
explanation is false. The persona earns its +0.0292 only once a role frame exists
for it to attach to.

**The turn structure is the largest single component**, at +0.1583, and it works
with plain-text markers that no model saw in training.

**Generic framing reproduces 66% of the full template effect.** The model's own
tuned tokens supply the remaining 34%, which is real and a minority. So the
convergence is mostly the assistant frame, reachable by prompt convention alone,
with a genuine but secondary contribution from the tuned format.

The floor stays between 0.0961 and 0.1125 across all five arms, so none of this is
a scale artifact.

### 5.2 The effect does not shrink with scale

The standing objection to Section 5 is that 0.36B to 2B models are not the systems
anyone deploys, and that convergence at that size could be a small-model artifact.

We ran the same four arms across the Qwen2.5-Coder ladder, one lab, one recipe and
one tokenizer held fixed from 0.5B to 14B, on 164 HumanEval prompts with K = 8.
Holding the family fixed removes lineage as a confound, so the only thing varying
across rungs is parameter count.

| size | params (B) | base raw | inst raw | inst chat | shared | tuning | format |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 0.5 | 0.5966 | 0.7863 | 0.8650 | 0.8508 | +0.1897 | +0.0787 |
| 1.5B | 1.5 | 0.6371 | 0.7723 | 0.8526 | 0.8414 | +0.1352 | +0.0803 |
| 3B | 3.1 | 0.6416 | 0.7499 | 0.8764 | 0.8752 | +0.1083 | +0.1265 |
| 7B | 7.6 | 0.6215 | 0.7315 | 0.8638 | 0.8648 | +0.1100 | +0.1323 |
| 14B | 14.8 | 0.6085 | 0.7653 | 0.9055 | 0.8974 | +0.1568 | +0.1402 |

**The format effect grows with scale**, from +0.0787 at 0.5B to +0.1402 at 14B, a
slope of +0.0473 per decade of parameters. The small-model objection is answered in
the direction that matters: whatever this is, it does not wash out as models get
bigger over the 30x range we can afford. The tuning term moves the other way, at
−0.0285 per decade, so the split between weights and format tilts further toward
format as scale increases.

**Generic markers reach parity with native tokens in this domain.** The `shared`
arm recovers 82% of the format gain at 0.5B and 94% to 101% from 3B upward, against
66% in the prose setting of Section 5.1. At 7B the untrained markers slightly
exceed the model's own template, 0.8648 against 0.8638. Whatever the tuned tokens
buy in prose, on code prompts a plain-text turn structure is sufficient.

Absolute levels are much higher here than in Section 5.1, with instruct raw already
at 0.73 to 0.79 before any framing, because a function signature and docstring
constrain the output far more than an open-ended stem. The gains are therefore
measured against a compressed ceiling, and the magnitudes are not comparable across
the two domains. The slope is.

### 5.3 An independent lab reproduces the decomposition

Section 5.1 rests on one set of six models. We repeated it on Ministral-3, from
Mistral AI, a lab absent from that set, with matched base and instruct weights at
3B, 8B and 14B on the same 60 prose stems.

| | 3B | 8B | 14B |
|---|---:|---:|---:|
| base raw | 0.4614 | 0.4473 | 0.4449 |
| instruct raw | 0.4307 | 0.4251 | 0.4301 |
| instruct chat | 0.8350 | 0.8380 | 0.8541 |
| tuning alone | −0.0307 | −0.0222 | −0.0148 |
| persona alone | +0.0612 | +0.0759 | +0.0821 |
| role structure alone | **+0.3809** | **+0.4307** | **+0.4354** |
| persona within the frame | +0.0268 | +0.0006 | +0.0068 |
| full format effect | **+0.4043** | **+0.4129** | **+0.4240** |

The ordering reproduces exactly. Role structure dominates, persona text alone is a
minor term, and persona contributes almost nothing once a frame already exists. The
`shared` arm reaches 0.8116 to 0.8655 intra against floors of 0.1329 to 0.1391,
using markers Mistral never trained on.

Two things are sharper here than in Section 5.1. **Instruction tuning on its own is
slightly negative**, between −0.0148 and −0.0307, so for this family the tuned
weights supply none of the convergence and the frame supplies all of it. And the
format effect is larger, +0.4043 to +0.4240 against +0.2837, on the same kind of
prompt.

**The native-token component does not replicate.** In Section 5.1 the model's own
tuned tokens were worth +0.0962, a real 34% of the effect. Here generic markers
recover 94% of the full template effect at 3B and then exceed it, at 104% for 8B and
103% for 14B, so the native tokens are worth +0.0234, −0.0178 and −0.0114. The 66/34
split is a property of those six models, not a constant. In this family the
assistant frame is the whole mechanism.

The format effect also **rises with scale here**, +0.0348 per decade across the three
rungs, independently reproducing the direction found on the Qwen ladder in Section
5.2 at +0.0473. Two families, two domains, two labs, same sign.

### 5.4 Homogeneous text, divergent correctness

High intra-model similarity is a claim about text, not about being right. On code
we can measure both on the same samples.

Using a sandboxed executor validated at 164/164 on HumanEval canonical solutions,
we scored every candidate behind the arms above. `coder7B_inst__chat` sits at 0.8638
intra, comfortably inside hivemind territory, while a single sample from it passes
0.5404 of the time and the K = 8 pool contains a passing solution 0.9024 of the
time.

Across all 30 arms the gap between the pool and a single sample averages **+0.3151**
and reaches +0.4649. For comparison the same quantity on GSM8K is +0.060. Samples
that an encoder scores as 0.86 similar disagree about correctness on roughly a third
of problems.

**Embedding homogeneity does not imply outcome homogeneity.** This bounds what the
hivemind result licenses: convergence in phrasing is real and we reproduced it, but
it should not be read as convergence in what the models actually get right.

### 5.5 It can be suppressed, but not by anything anyone ships

Everything above adds framing and watches similarity rise. That is an explanation,
not a remedy. The question a deployer would ask is the reverse one: if the shared
frame causes the convergence, does breaking the frame undo it?

We vary the persona in two directions against the same six models and the native
chat arm as baseline. In `persona_model` each model gets a different voice and all
eight of its samples share it, which is the deployer's lever for differentiating a
product. In `persona_sample` each of the eight samples gets a different voice within
one model, which is the diversity lever for making resampling useful again. We run
both with eight genuinely distinct voices, and again with eight personas of the kind
actually shipped, every one a rewording of "helpful assistant".

| arm | intra | inter | cross-lab | floor | models any prompt > 0.8 |
|---|---:|---:|---:|---:|---:|
| chat (baseline) | 0.7272 | 0.6464 | 0.6490 | 0.1028 | 1.0000 |
| exotic, per model | 0.5701 | **0.3817** | 0.3708 | 0.1143 | 0.6667 |
| exotic, per sample | **0.4533** | 0.4130 | 0.4083 | 0.1114 | 0.3333 |
| deployed, per model | 0.7207 | 0.6441 | 0.6455 | 0.0998 | 0.8333 |
| deployed, per sample | 0.6799 | 0.6226 | 0.6224 | 0.1061 | 0.8333 |

**The convergence is suppressible.** Genuinely different voices take inter-model
similarity down by 0.2647, from 0.6464 to 0.3817, and per-sample variation takes
intra-model similarity down by 0.2739, from 0.7272 to 0.4533. The fraction of models
with any prompt above 0.8 falls from all of them to a third. Nothing about the
homogeneity is locked in by the weights.

**The personas the industry actually ships do essentially nothing.** Eight different
production-style system prompts move inter-model similarity by 0.0023 and intra by
0.0065. That is indistinguishable from not intervening at all. Varying them per
sample buys 0.0473 of intra, still leaving 83% of models above the 0.8 line.

This is the practical finding. The lever exists and it is a single string, but it
only works if the string stops being a variation on "helpful assistant". Eight
different ways of saying helpful assistant are, to the model, one persona. The
industry has standardised on a register, and the register is the mechanism.

Two supporting observations. The floor stays between 0.0998 and 0.1143 across all
five arms, so none of the movement is a scale artifact. And default templates do not
explain the baseline: only Qwen2.5 and SmolLM2 inject a branded persona at all,
while Qwen3, gemma-2 and OLMo-2 inject no system prompt and Llama-3.2 injects only
date metadata. They converge regardless.

## 6. What transport predicts

Within any given level, transport still carries information about which pairs
converge. Across the 66 unordered pairs of the 12-model set, using symmetrized
transport against inter-model output similarity:

| | n | Spearman | p |
|---|---:|---:|---:|
| all pairs | 66 | +0.3600 | 0.0020 |
| minus the identical-weights anchor | 65 | +0.3304 | 0.0055 |
| cross-lab only | 59 | +0.3098 | 0.0139 |

The effect survives removing the anchor, which is the same weights twice, and
survives restricting to pairs that cross a lab boundary. It is roughly a tenth of
the variance and we report it as such.

Read with Section 5 this gives a clean division. Representation geometry predicts
**which** models agree more than others. Instruction tuning contributes a little to
**how much**. The chat template contributes most of it.

## 7. The decoding negative, and why it reversed

We first ran Section 6 with greedy decoding, reasoning that greedy removes
sampling luck. That was the wrong instrument and produced a null:

| greedy | n | Spearman | p | |
|---|---:|---:|---:|---|
| all pairs | 66 | +0.2645 | 0.0282 | |
| minus anchor | 65 | +0.2308 | 0.0598 | n.s. |
| cross-lab only | 59 | +0.2213 | 0.0866 | n.s. |

We banked that as a negative before establishing that the decode was wrong. Greedy
returns the mode of a distribution; the original study measured the distribution.
Greedy also drove base models into degenerate loops, with gpt-oss-20b emitting "So
the total number of servings is 1" repeatedly, which then gets scored as content.
Under the correct protocol nothing changed except the decode, and cross-lab moved
from +0.2213 at p = 0.0866 to +0.3098 at p = 0.0139.

The greedy run also produced an anecdote that misled us. On the prompt "The best
way to learn a new language is", seven models from seven labs continued with the
identical phrase "to immerse yourself in it", and gpt-oss-20b and Qwen2.5-7B
agreed for 110 consecutive characters.

Measured across all 60 prompts it is far weaker. The mean number of models
producing an identical 40-character continuation is 2.2 of 12, the maximum on any
prompt is 5, **no prompt** has 6 or more agreeing, and 48 of 60 have 2 or fewer.
Across all 3,960 pair-prompt comparisons the median shared prefix is 5 characters
and only 2.1% reach 50. The vivid case was real and sits in a thin tail. One
prompt is not a measurement.

## 8. A baseline the original study does not have

Their random-pair floor answers "is there any signal." It draws pairs answering
*different* queries. The comparison that bears on convergence is independent
sources answering the *same* query, and the study has no such arm: its 31,250
human annotations rate model output rather than responding to the prompts.

COCO supplies the missing structure, with five captions per image each written by
a different annotator. We scored 2,000 images across four encoders.

On MiniLM, whose different-image floor of 0.0898 matches their encoder's, two
independent humans describing the same photograph reach **0.6121**, below the 0.71
to 0.82 they report between models. On a floor-matched scale their models are more
alike than independent humans are.

**This goes their way.** We built the control expecting it to complicate their
claim and it strengthened it. Captioning is narrower than open-ended generation,
so it calibrates rather than settles, and the arm they need is independent human
responses to their own prompts.

## 9. Raw cosine is not portable between encoders

| encoder | same image | different image | raw gap | centered gap |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.6121 | 0.0898 | 0.5224 | 0.5743 |
| bge-small-en-v1.5 | 0.7610 | 0.4412 | 0.3198 | 0.5687 |
| gte-base | 0.8876 | 0.7361 | 0.1514 | 0.5695 |
| e5-base-v2 | 0.8784 | 0.7241 | 0.1542 | 0.5598 |

The raw gap between same-item and different-item pairs ranges from 0.1514 to
0.5224, a factor of 3.5, depending only on encoder choice. After centering on the
pool mean, all four collapse to between 0.5598 and 0.5743.

The underlying quantity is constant. The raw spread is each encoder's anisotropy.
No raw cosine figure is transferable between encoders, and every one needs its
floor quoted beside it. The original study's numbers hold because
`text-embedding-3-small` is a low-anisotropy encoder, which its 0.1 to 0.2 floor
demonstrates. Had they used gte-base, "same query" and "different query" would have
been separated by 0.15 instead of 0.52.

## 10. Negative results

**The mechanism link failed under greedy decoding**, cross-lab +0.2213 at
p = 0.0866, and we published that null before establishing the decode was wrong.
Section 7.

**That design could not have succeeded.** Transport is at or above 0.83 for all 66
pairs and spans only 0.173, leaving almost no variance to correlate against.
Underpowered by construction, not by bad luck.

**Our anisotropy critique of the original study failed.** We predicted their raw
cosines were inflated by anisotropy; their reported floor of 0.1 to 0.2
demonstrates a low-anisotropy encoder and pre-empts the objection.

**Our human baseline strengthened their claim rather than weakening it.**
Section 8.

**Seven-lab agreement was an artifact of reading one prompt.** Section 7.

**Centered semantic voting does not select better code.** Anisotropy among code
candidates is severe, 0.7929 mean pairwise cosine across 30 arms, so we expected
centering by the pool mean to be necessary before consensus voting. It changes
nothing: centered minus raw is −0.0016, with 8 wins, 12 losses and 10 ties. Medoid
voting beats a random single sample by +0.0064, which is 2.0% of the +0.3151
headroom. Centering corrects similarity magnitudes and leaves the argmax alone.

**A cross-model transport selector did not survive replication, and we published
the claim before checking.** Scoring candidates by how far a hidden state fails to
return after transport around a closed cross-lab loop gave +0.0727 over the floor
at p < 0.0001, passing a length control, a reversed-direction control and 10,000
permutations. Refitting the ridge maps on a different set of arms, with the
evaluated candidates and ground truth byte-identical, moved the same measurement to
+0.0158 at p = 0.1986, and at 14B it went to −0.0208 with the discriminative
direction reversed. The signal belonged to the particular map fit, not to the
candidates. We record it because the three controls we ran all passed on the
configuration that was wrong, and none of them could have caught it. Varying an
arbitrary knob would have.

**The Ministral chat arm was silently corrupted by a tokenizer round-trip.**
Rendering a chat template to a string and re-tokenizing it is standard practice and
works for most families. Mistral's `MistralCommonBackend` does not survive it: the
rendered `<s>[INST]` came back as the literal characters `<`, `s`, `>[`, `IN`,
`ST`, `]` rather than the special ids 1 and 3, so the model read its own control
tokens as prose. It responded by emitting template markup in 336 of 480
continuations at 3B, 315 at 14B and 102 at 8B, with a format gain that oscillated
in sign across rungs. Passing token ids directly fixes it, and the repaired arm is
what Section 5.3 reports. The pre-fix generations are kept under
`corrupt_chat_pre_fix/`. We record this because nothing about the failure looked
like a tokenizer problem from the metric alone.

## 11. Limitations

Section 5.2 takes the ladder to 14B within one family and finds the format effect
growing at +0.0473 per decade, which answers the small-model objection over a 30x
range. It does not reach frontier scale, and it does not show that a 500B model
reaches the reported level by this route.

Our prompts are 60 hand-written open-ended stems, not 26,070 mined from real
traffic, and our continuations are 48 tokens rather than full responses. The chat
arm gives instruct models a format their base siblings never had, which is the
point of the comparison, but it also means the two arms differ in effective task
as well as in weights.

The scale and replication ladders use 192 and 128 new tokens respectively. Chat
arms spend part of that budget on preamble before content, so cross-arm comparisons
of anything other than similarity, capability in particular, are confounded by
truncation. We make no capability claim across arms for that reason.

Section 5.3 covers three rungs of one additional family. It replicates the ordering
of the decomposition and reverses the native-token term, which suggests that term
is family-specific rather than general.

We use a different scorer than they do. We matched the floor rather than the
encoder because we do not have theirs, and Section 9 argues the floor is what
matters. That argument could be wrong.

Transport is measured on COCO captions, a narrow domain, and may not generalize to
reasoning or code.

The correlation in Section 6 is roughly a tenth of the variance. It orders pairs;
it does not explain them.

Section 5.1 uses one persona sentence and one generic marker format. A different
persona, or markers closer to a specific model's native format, could shift the
66/34 split. We have shown that generic framing recovers most of the effect, not
that this particular framing is optimal.

Section 5.5 shows the effect is suppressible with eight distinct personas, on 60
stems and six models. It does not establish how far the register can be varied
before a product stops being usable as an assistant, which is the constraint that
actually binds a deployer.

## 12. Artifacts

Everything reproduces from committed JSON. We publish the states, not just the
scores.

- `scripts/openweight_transport_atlas.py`, `artifacts/nla/atlas/openweight_transport_atlas.json`
- `scripts/hivemind_sampled_protocol.py`, `artifacts/nla/atlas/hivemind_sampled_protocol.json`
- `scripts/hivemind_posttraining_isolation.py`, `artifacts/nla/atlas/hivemind_posttraining_isolation.json`
- `scripts/hivemind_template_decomp.py`, `artifacts/nla/atlas/hivemind_template_decomp.json`
- `scripts/hivemind_suppression.py`, `artifacts/nla/atlas/hivemind_suppression.json`
- `scripts/hivemind_mechanism_link.py`, `artifacts/nla/atlas/hivemind_mechanism_link.json`
- `scripts/hivemind_human_baseline.py`, `artifacts/nla/hivemind_human_baseline.json`
- `scripts/coder_ladder.py` and `scripts/coder_ladder_analyze.py`, `artifacts/nla/coder_ladder/`
- `scripts/ministral_ladder.py`, `artifacts/nla/ministral_ladder/`
- `scripts/code_select.py`, `artifacts/nla/code_select/results.json`
- `scripts/holonomy_select.py`, `artifacts/nla/holonomy/`, including the retraction sweep
- `artifacts/nla/atlas/samples/` and `samples_instruct/`, sampled continuations
- `scripts/bench_device.py`, same-task benchmark for cross-device comparison

Sections 1 through 5.1 and 6 through 10 ran on a single Apple M2 Ultra with 64 GB
of unified memory, concurrently with that machine serving a 31B multimodal model to
a public endpoint. Sections 5.2 through 5.4 ran on four RTX PRO 6000 Blackwell
cards, one model per card, because 39,360 generations at 14B do not fit in the
former.

## 13. Conclusion

The original study established that language models write alike, carefully, with a
floor, on real user traffic, at a scale we cannot match. Its limit is access:
behind an API, text is all there is.

With open weights the question decomposes, and the three parts have different
answers.

**Representations are mutually recoverable and lineage barely matters.** A ridge
map moves one model's state into another's well enough to retrieve the right item
0.9181 of the time across lab boundaries, against a floor of 0.00101, and being
the same company is worth 0.0357. This appears to be a property of learning
language at all.

**Base models do not write alike.** Under the original protocol they reach 0.3644
intra-model on a matched floor, with zero of 720 cells clearing 0.8. Near-ceiling
transportability coexists with low output convergence, so shared geometry is not
sufficient.

**The level arrives with the chat template.** Instruction tuning alone buys
+0.0786. The same weights prompted through their own template buy +0.3623,
reaching 0.7272, with four of six models above 0.80 at parameter counts three
orders of magnitude below the systems originally measured.

**And it is the turn structure, not the persona.** Telling a model it is a helpful
assistant does nothing by itself, −0.0120. Generic role markers no model was
trained on recover 66% of the template's effect, with the tuned tokens supplying
the rest.

The homogeneity that matters is therefore not inherited from pretraining and is
not mostly purchased by alignment. It is largely a consequence of how models are
addressed at inference: not the words in the system prompt, but the act of putting
the model in an assistant's turn and asking it to speak there.

That is a more optimistic finding than a hivemind living in the weights, because
two-thirds of it is a prompt convention rather than a property of any model, and
conventions can be changed. It is also a warning, because a homogeneity that comes
from the interface will not be fixed by training more diverse models, and every
lab converging on the same turn format is enough to produce it without anyone
choosing it.

Establishing it required opening the models, which is the one move the original
design could not make.

## Acknowledgments

To Jiang, Chai, Li, Liu, Fok, Dziri, Tsvetkov, Sap, Albalak and Choi for
Infinity-Chat and for reporting a floor. The random-pair baseline is why our first
critique failed, and a paper that defeats an attempted rebuttal on its own
reported controls is a paper doing its job.

Three results here exist because a reader pushed back on our method: on whether
identical outputs indicated a bug, which exposed that our agreement metric was
wrong; on greedy versus sampled decoding, which reversed a published null; and on
running the matched-pair follow-up, which produced Section 5.

## References

Jiang, L., Chai, Y., Li, M., Liu, M., Fok, R., Dziri, N., Tsvetkov, Y., Sap, M.,
Albalak, A., Choi, Y. *Artificial Hivemind: The Open-Ended Homogeneity of Language
Models*. arXiv:2510.22954. NeurIPS 2025 Datasets and Benchmarks, Oral.
Code: github.com/liweijiang/artificial-hivemind

Ethayarajh, K. *How Contextual are Contextualized Word Representations?* EMNLP 2019.

Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.

Zhao, W. et al. *WildChat: 1M ChatGPT Interaction Logs in the Wild*. ICLR 2024.
