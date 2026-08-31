# Convergence Has Two Sources: Separating Representation Geometry from Post-Training on Open Weights

## Abstract

Jiang et al. (arXiv:2510.22954) measured 70 or more language models at the output
layer and found them writing alike: responses to the same open-ended query score
0.71 to 0.82 in pairwise similarity between different models, and 79% of queries
produce intra-model similarity above 0.8 across 50 resamples. Almost every model
in that study is a closed endpoint, so text is the only observable available. The
result establishes that models agree without being able to say why.

We open the box. On 12 open-weight models from 8 labs, spanning 360M to 20B
parameters, we fit ridge maps between the models' hidden states and measure how
much of one model's representation is a linear function of another's. Held-out
retrieval reaches 0.9181 across lab boundaries against a shuffled floor of
0.00101, with a self-map ceiling of 0.999. Crossing a vendor boundary costs
0.0787, and shared corporate lineage buys only 0.0357 of that.

We then run their sampling protocol on the same 12 models. Two things happen. The
structure they report replicates: a model's own resamples are barely more similar
to each other than to a different model's output, 0.3644 against 0.3401. The level
does not replicate, and the floors prove this is not a scale artifact. Our
different-prompt floor is 0.0993 against their 0.1 to 0.2, yet our intra-model
similarity is 0.3644 against their 0.8, and zero of 720 model-prompt cells clear
0.8.

Base-model states transport at near ceiling while base-model text converges at
roughly half the instruction-tuned level. Within that lower level, transport
predicts which pairs converge (cross-lab Spearman +0.3098, p = 0.0139). We read
this as two separable sources: representation geometry orders which models agree,
and post-training sets how much. Only open weights can separate them.

We also report a negative that reversed under scrutiny, and a baseline the
original study lacks.

---

## 1. The question

"Artificial Hivemind: The Open-Ended Homogeneity of Language Models" is a
NeurIPS 2025 Datasets and Benchmarks best paper by Liwei Jiang, Yuanjun Chai,
Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap,
Alon Albalak and Yejin Choi. It introduces Infinity-Chat, 26,070 genuinely
open-ended queries mined from real user traffic in WildChat, and measures
similarity within and between model outputs.

The measurements are careful. Similarity is cosine over sentence embeddings from
`text-embedding-3-small`. Crucially, they report a floor: randomly paired
responses drawn from the global pool fall 100% within 0.1 to 0.2. That is the
discipline we insist on in our own work, and it is present here. Their limitations
section concedes that embedding similarity "may lack sufficient expressiveness,"
and their Table 13 shows two visibly different 30-word essays scoring 0.737.

What the study cannot do is explain the agreement. Output cosine sits downstream
of the thing it is trying to characterise. If two models write alike, the sentence
embedding of their text cannot distinguish "these systems compute similar internal
representations" from "these systems were tuned toward similar behaviour" from
"this prompt admits few good answers." Almost every model they study is served
behind an API, so this is not an oversight. It is a hard limit of the access they
had.

Open weights remove that limit. The question we can ask, and they could not, is
whether the agreement is already present in the representations.

## 2. Setup

Twelve models, eight labs, all base models except where noted, all run locally on
one desktop machine.

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

States are mean-pooled hidden states at 60% relative depth. Text is 5,000 COCO
val2017 captions, one per image, split 4,000 train and 1,000 held out.

The last row is a control, not a model. It is the same Qwen2.5-7B weights loaded
at fp32 instead of bf16. Identical weights must land at the top of every axis we
measure. If they do not, the instrument is broken and no result below survives.

### 2.1 Measurement discipline

Every claim in this paper carries a floor computed through the same machinery that
produced the effect. Shuffled-label floors, not analytic chance.

States are centered on the training mean before every ridge solve and every
cosine. Section 8 shows why this is not optional.

Transport is scored by held-out retrieval rank, which is scale-free, rather than
by cosine, which is not.

Where we compare against the original study, we report our floor beside theirs. A
number is only comparable to another number if the floors agree, and Section 4
turns on exactly that point.

## 3. The transport atlas

For every ordered pair of models, fit a ridge map from A's states to B's on the
4,000 training captions, then take the 1,000 held-out items, push A's state
through the map, and retrieve the matching item among B's states. Chance is
0.0010.

The diagonal is the self-map: the same ridge machinery from a model back to
itself. It is the ceiling the retrieval task allows, and it comes in at 0.999.

| | value |
|---|---:|
| self-map diagonal | 0.9990 |
| transported, off-diagonal | 0.9203 |
| transport cost | 0.0787 |
| shuffled floor | 0.00101 |
| within-lab | 0.9539 |
| cross-lab | 0.9181 |
| within minus cross | 0.0357 |

The best off-diagonal pair is Qwen2.5-0.5B to OLMo-2-1B at 0.999, which is
indistinguishable from the self-map ceiling. Alibaba to AI2, different
architecture, different tokenizer, different corpus, and a linear map recovers
the right caption 999 times in 1000.

The worst pair is gpt-oss-20b to Qwen2.5-7B at 0.735, which is still 735 times
the floor.

Shared lineage is worth 0.0357. That is the entire premium for being the same
company, the same architecture family and the same tokenizer, measured against a
cross-lab level of 0.9181. Whatever these models share, they did not get it from
each other.

The precision control returns 0.999 in both directions, against a cross-lab mean
of 0.9181. Numerics cost nothing at this resolution, which also clears the way for
mixing quantized and full-precision rows in one matrix.

## 4. Their protocol, our models

The atlas says representations are mutually recoverable. It does not say the text
converges. To connect the two we ran their protocol.

Sampling settings are theirs: top-p 0.9, temperature 1.0, K = 8 samples per
prompt per model, 60 shared open-ended prompts, 5,760 samples total. Outputs are
scored with `all-MiniLM-L6-v2`, chosen because its different-prompt floor of
0.0993 sits inside the 0.1 to 0.2 band their encoder produces. Matching the floor
is what makes the levels comparable.

| arm | ours | theirs |
|---|---:|---|
| floor, different prompts | 0.0993 | 0.1 to 0.2 |
| intra-model, own resamples | **0.3644** | above 0.8 on 79% of queries |
| inter-model, same prompt | **0.3401** | 0.71 to 0.82 |
| intra minus inter | +0.0244 | overlapping |

Two separate things happen here and they must not be run together.

**The structure replicates.** Their most striking observation is that a model's
own resamples are barely more similar to each other than to a different model's
output. We see the same shape: 0.3644 against 0.3401, a gap of 0.0244. Resampling
a model explores almost nothing that distinguishes it from its competitors. That
finding survives translation to open weights at a hundredth of the scale.

**The level does not replicate, and the floors prove it is real.** Our floor is
0.0993 and theirs is 0.1 to 0.2. The measurement scales agree. Yet our intra-model
similarity is 0.3644 where theirs exceeds 0.8, and our inter-model similarity is
0.3401 where theirs is 0.71 to 0.82. Per-model intra-similarity ranges from 0.3087
for Pythia-410m to 0.3998 for TinyLlama. **Zero of 720 model-prompt cells clear
0.8.** Not one, on any of the twelve, including gpt-oss-20b.

If the encoder or the scale were responsible, the floors would differ. They do
not. The gap is in the signal.

So: these models' states transport at 0.9181, near the 0.999 ceiling, while their
text converges at roughly half the level reported for instruction-tuned models.
High representational transportability coexists with low output convergence. That
combination rules out the simple story in which shared representation geometry is
sufficient for the hivemind.

## 5. What transport does predict

Within the lower level, transport still carries information about which pairs
converge. For each of the 66 unordered model pairs we have a transport score,
symmetrized across both directions, and an inter-model output similarity.

| | n | Spearman | p |
|---|---:|---:|---:|
| all pairs | 66 | +0.3600 | 0.0020 |
| minus the identical-weights anchor | 65 | +0.3304 | 0.0055 |
| cross-lab only | 59 | +0.3098 | 0.0139 |

The effect survives removing the anchor, which is the same weights twice and says
nothing about labs, and it survives restricting to pairs that cross a lab
boundary. It is a modest effect, roughly a tenth of the variance, and we report
it as such.

Read together with Section 4, this gives two separable sources. Representation
geometry predicts **which** models agree more than others. Post-training sets
**how much** any of them agree. The first is visible only in the states, the
second only by comparing base against tuned. Neither is reachable from output
cosine on closed endpoints alone.

## 6. The decoding negative, and why it reversed

We first ran Section 5 with greedy decoding, reasoning that greedy removes
sampling luck and any agreement would therefore be the models genuinely agreeing.

That was the wrong instrument, and the result was a null:

| greedy | n | Spearman | p | |
|---|---:|---:|---:|---|
| all pairs | 66 | +0.2645 | 0.0282 | |
| minus anchor | 65 | +0.2308 | 0.0598 | n.s. |
| cross-lab only | 59 | +0.2213 | 0.0866 | n.s. |

We banked that as a negative, concluding the mechanism link did not hold. It was
an artifact of the decode. Greedy returns the mode of a distribution; the original
study measured the distribution. Comparing the two conflates decoding with
everything else. Greedy also drove base models into degenerate loops, with
gpt-oss-20b emitting "So the total number of servings is 1" repeatedly, which then
gets scored as though it were content.

Under the correct protocol the same models and the same transport matrix produce
cross-lab +0.3098 at p = 0.0139, where greedy gave +0.2213 at p = 0.0866. Nothing
changed except the decode.

The greedy run also produced an anecdote worth reporting precisely because it
misled us. On the prompt "The best way to learn a new language is", seven models
from seven labs continued with the identical phrase "to immerse yourself in it",
and gpt-oss-20b and Qwen2.5-7B agreed for 110 consecutive characters. Read from a
log, that looks like overwhelming convergence.

Measured across all 60 prompts, it is not. The mean number of models producing an
identical 40-character continuation is 2.2 of 12, the maximum on any prompt is 5,
**no prompt** has 6 or more agreeing, and 48 of 60 prompts have 2 or fewer. Across
all 3,960 pair-prompt comparisons the median shared prefix is 5 characters and
only 2.1% reach 50 characters. The vivid case was real and sits in a thin tail.
One prompt is not a measurement.

## 7. A baseline the original study does not have

Their random-pair floor answers "is there any signal at all." It draws pairs that
answer *different* queries. The comparison that bears on convergence is
independent sources answering the *same* query, and the study has no such arm.
Its 31,250 human annotations are ratings of model output, not human responses to
the prompts.

Without that arm, 0.75 could mean the models converged, or it could mean the
prompt constrains the answer space and the encoder is scoring topical overlap.

COCO supplies the missing structure: five captions per image, each written by a
different annotator, all describing the same thing. Independent sources, same
item. We scored 2,000 images across four encoders.

On MiniLM, whose different-image floor of 0.0898 matches their encoder's, two
independent humans describing the same photograph reach **0.6121**, below the 0.71
to 0.82 they report between models. On a floor-matched scale, their models are
more alike than independent humans are.

**This goes their way.** We built the control expecting it to complicate their
claim and it strengthened it. Captioning a photograph is a narrower task than
open-ended generation, so this calibrates the question rather than settling it,
and the arm they actually need is independent human responses to their own
prompts. But the critique we set out to make did not survive contact with the
measurement.

## 8. Raw cosine is not portable between encoders

The same human-baseline run produces a methodological result that applies to every
similarity number in this literature, including ours.

| encoder | same image | different image | raw gap | centered gap |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.6121 | 0.0898 | 0.5224 | 0.5743 |
| bge-small-en-v1.5 | 0.7610 | 0.4412 | 0.3198 | 0.5687 |
| gte-base | 0.8876 | 0.7361 | 0.1514 | 0.5695 |
| e5-base-v2 | 0.8784 | 0.7241 | 0.1542 | 0.5598 |

The raw gap between same-item and different-item pairs ranges from 0.1514 to
0.5224, a factor of 3.5, depending only on which encoder you happen to pick. After
centering on the pool mean, all four collapse to 0.5598 through 0.5743.

The underlying quantity is constant. The raw spread is each encoder's anisotropy.

This means no raw cosine similarity figure is transferable between encoders, and
every such figure needs its floor quoted beside it. The original study's numbers
hold up because `text-embedding-3-small` happens to be a low-anisotropy encoder,
which their 0.1 to 0.2 floor demonstrates. That is a property of their encoder
choice, not a robustness of the metric. Had they used gte-base, "same query" and
"different query" would have been separated by 0.15 instead of 0.52 and the paper
would have read very differently.

## 9. Negative results

Banked, in the order we found them.

**The mechanism link failed under greedy decoding.** Cross-lab Spearman +0.2213
at p = 0.0866. We published this as a negative before establishing that the decode
was wrong. Section 6.

**That design could not have succeeded.** Transport is at or above 0.83 for all
66 pairs and spans only 0.173. There is almost no variance to correlate against.
The test was underpowered by construction, not by bad luck, and we should have
seen that before running it.

**Our anisotropy critique of the original study failed.** We predicted their raw
cosines would be inflated by anisotropy. They report a random-pair floor of 0.1 to
0.2, which demonstrates a low-anisotropy encoder and largely pre-empts the
objection.

**Our human baseline strengthened their claim rather than weakening it.**
Section 7.

**Seven-lab agreement was an artifact of reading one prompt.** Section 6.

## 10. Limitations

Our models are small. The largest is 20B against frontier systems in the original
study. The level gap in Section 4 could in principle be a scale effect rather than
a post-training effect. The decisive experiment is cheap and we have not run it:
the identical protocol on *instruct variants of these same models*, holding
pretrained representations and the transport matrix fixed while changing only
post-training. If intra-model similarity moves from 0.3644 toward 0.8, the cause
is isolated.

Our prompts are 60 hand-written open-ended stems, not their 26,070 mined from real
traffic. Our continuations are 48 tokens, not full responses.

We use a different scorer than they do. We matched the floor rather than the
encoder because we do not have their encoder, and Section 8 argues the floor is
what matters. That argument could be wrong.

Transport is measured on COCO captions, a narrow and homogeneous text domain. The
atlas may not generalize to reasoning or code.

The correlation in Section 5 is roughly a tenth of the variance. It orders pairs;
it does not explain them.

## 11. Artifacts

Everything below reproduces from committed JSON. We publish the states, not just
the scores.

- `scripts/openweight_transport_atlas.py`, `artifacts/nla/atlas/openweight_transport_atlas.json`
- `scripts/hivemind_sampled_protocol.py`, `artifacts/nla/atlas/hivemind_sampled_protocol.json`
- `scripts/hivemind_mechanism_link.py`, `artifacts/nla/atlas/hivemind_mechanism_link.json`
- `scripts/hivemind_human_baseline.py`, `artifacts/nla/hivemind_human_baseline.json`
- `artifacts/nla/atlas/samples/`, 5,760 sampled continuations
- `artifacts/nla/atlas/generations/`, 720 greedy continuations
- `scripts/bench_device.py`, same-task benchmark for cross-device comparison

Every measurement in this paper ran on a single Apple M2 Ultra with 64 GB of
unified memory, concurrently with that machine serving a 31B multimodal model to
a public endpoint.

## 12. Conclusion

The original study established that language models write alike and did so
carefully, with a floor, on real user traffic, at a scale we cannot match. Its
limit is one of access rather than method: behind an API, text is all there is.

With open weights the question splits in two, and the halves have different
answers.

Representations are mutually recoverable to a degree that does not depend on
lineage. A ridge map moves one model's state into another's well enough to
retrieve the right item 0.9181 of the time across lab boundaries, against a floor
of 0.00101, and being the same company is worth only 0.0357 of that. Pythia-410m
and gemma-2-2b share no architecture, tokenizer, corpus or organisation, and one
reads the other at 0.9820.

Output convergence is not the same story. Base models reproduce the *structure*
the original study reports, with resamples barely more distinctive than a rival's
output, but at *half the level*, on a matched floor, with zero of 720 cells
clearing the threshold that 79% of their queries clear. Transport predicts which
pairs converge, modestly and significantly. It does not deliver the level.

The most likely account is that both sources are real and separable.
Representation geometry, which appears to be a property of learning language at
all, sets the ordering. Post-training sets the magnitude. If that is right, the
homogeneity worth worrying about is one the field is choosing rather than
inheriting, which makes it far more tractable than a hivemind that lives in the
weights.

Testing it requires opening models, which is the one move the original design
could not make.

## Acknowledgments

To Jiang, Chai, Li, Liu, Fok, Dziri, Tsvetkov, Sap, Albalak and Choi for
Infinity-Chat and for reporting a floor. The random-pair baseline is the reason
our first critique failed, and a paper that can defeat an attempted rebuttal on
its own reported controls is a paper doing its job.

Two of the results here exist because a reader pushed back on our method twice in
one session: once on whether identical outputs indicated a bug, which exposed that
our agreement metric was wrong, and once on greedy versus sampled decoding, which
reversed a published negative.

## References

Jiang, L., Chai, Y., Li, M., Liu, M., Fok, R., Dziri, N., Tsvetkov, Y., Sap, M.,
Albalak, A., Choi, Y. *Artificial Hivemind: The Open-Ended Homogeneity of Language
Models*. arXiv:2510.22954. NeurIPS 2025 Datasets and Benchmarks, Oral.
Code: github.com/liweijiang/artificial-hivemind

Ethayarajh, K. *How Contextual are Contextualized Word Representations?* EMNLP 2019.

Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.

Zhao, W. et al. *WildChat: 1M ChatGPT Interaction Logs in the Wild*. ICLR 2024.
