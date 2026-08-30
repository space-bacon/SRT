# Frozen Backbones Read Each Other

## Cross-vendor probe transport in photographs, radiology and satellite imagery

Burton Lancaster, SRT program. 2026-08-30.

---

## Abstract

Four multimodal models from four companies encode the same images. Every model is
frozen and nothing is fine-tuned anywhere in this work. Their hidden states do
not even share a width, running to 5376, 5120, 2560 and 2048 dimensions, and
nothing ties one model's coordinates to another's.

We fit a linear probe, a single weight matrix, on one model's states to predict
image labels. We then read that same probe, unchanged, on a **different** model's
states, bridged by a ridge map estimated on training rows only. Neither model was
built with reference to the other, so nothing in their construction gives that
bridge a reason to work.

It works. Across three unrelated visual domains the probe survives the move, and
on the hardest of them it scores better than a probe fitted on the target model
directly.

On satellite imagery, a 17-class land-use probe scores 0.9507 mean AUROC natively
and 0.9484 transported across backbones, a cost of 0.0024. The shuffled floor on
the same holdout is 0.5014. AUROC is the area under the ROC curve, one
per class and averaged; 0.5 is chance and 1.0 is perfect ranking. On chest radiographs, across three of the
four backbones, using the official ChestX-ray14 test list and 25,596 held-out
films from patients never seen in training, the same procedure scores 0.7440
natively and **0.7511 transported**. The transport cost is negative. Four of six
cross directions beat the target backbone's own probe, and the single best
reading in the study belongs to a probe that was never fitted on the states it is
reading.

Because the backbones are not redundant, reading them together beats reading any
one of them. The task is the fourteen ChestX-ray14 findings, scored as mean AUROC
across those fourteen labels on the official test list. A single linear probe per
backbone, with their output logits averaged, adds no parameters beyond the three
probes themselves and reaches **0.7774**. The dataset's own authors, fine-tuning
a ResNet-50 end to end on that same split, report **0.7451**, so three frozen
backbones read together finish **0.0323** ahead of a network trained for the
task. Measured against the best single backbone rather than against the
literature, the gain is **+0.0124**, 95% CI [+0.0082, +0.0168] under a
patient-clustered bootstrap.

That the gain is information rather than capacity rests on one observation:
Aria scores 0.7080 by itself, which is 0.0371 *behind* the baseline, and pooling
it in still raises the average.

We also report what did not work: feature concatenation gains exactly 0.0000 once
a duplicate-vendor control absorbs the capacity effect, a routing recipe that
held on photographs failed on both other domains, and an earlier reading of
iterated-composition geometry was confounded and is withdrawn.

---

## 1. The question

Four companies each trained a large multimodal model. Different data, different
architectures, different objectives, no coordination between them. Show all four
the same chest radiograph and each produces a few thousand numbers.

Do those numbers mean the same thing?

There are two ways the answer could be no, and they are opposites. The models
might encode privately, each in coordinates the others cannot be read in, so that
a readout built on one is worthless on the next. Or they might encode
identically, so that once you have read one there is nothing left for the second
to add. The first would make every probe a captive of its backbone. The second
would make three models an expensive way to own one.

Each has a direct test. Fit a linear probe on one model and read it on another:
if the encodings are private, this fails. Then read the models together: if the
encodings are identical, this gains nothing. Accuracy alone tests neither, since
two models can score the same while encoding differently.

Retrieval does not settle it either. Two encoders agreeing about *which* picture
is weaker than agreeing about *what is in it*, since retrieval is satisfied by
any injective encoding, meaning any scheme that gives distinct pictures distinct
codes, however arbitrary those codes are. We therefore run both tests three
times, with increasing semantic commitment: retrieval on photographs,
weak-labelled scene classes on satellite imagery, and expert-labelled pathology
on chest radiographs.

---

## 2. Setup

**Backbones.** Qwen3-Omni-30B-A3B-Instruct, Gemma-4-31B-it, Mistral-Small-3.1-24B
and Aria. Two dense, two mixture-of-experts, four vendors, four pretraining
corpora. Every model is frozen. Nothing is fine-tuned anywhere in this paper.

All four encode every domain. The photograph, satellite and shared-frame results
use all four. **The ChestX-ray14 results use three**, Qwen3-Omni, Gemma-4 and
Aria, and every clinical figure in this paper is a three-backbone figure.

**States.** Hidden states are pooled over content-token positions at a fixed
fraction of depth. Nothing is read from a position the input did not occupy.

**Domains.** COCO photographs, ROCO radiology, RSICD satellite imagery, and
ChestX-ray14 chest radiographs.

**Metrics and controls.** Labelled tasks are scored by **AUROC**, the area under
the ROC curve, computed per class and averaged; 0.5 is chance. Retrieval is
scored by **r@1**, the fraction of queries whose correct match is ranked first.
Two controls recur. A **shuffled floor** refits or rescores with the labels or
pairings permuted, giving the value the procedure returns when no signal is
present. A **self-map control** transports a probe through a ridge map fitted
from a backbone to *itself*, which measures what the fitting step costs on its
own, separately from crossing a vendor boundary.

**Transport.** To read vendor A's probe on vendor C's states we fit a ridge map
C to A on training rows only, apply it to held-out states, then apply A's probe
unchanged. The map is never fitted on a test row, and the probe is never fitted
on vendor C at all.

### 2.1 Measurement discipline

Three habits do most of the work, and each of them changed a conclusion at least
once in this program.

**Anisotropy is corrected everywhere.** Raw mean pairwise cosine between
unrelated items on these image states runs from 0.873 to 0.998 depending on the
backbone. At 0.998 every item is nominally similar to every other and raw cosine
carries almost no information. Centering on the training mean brings all four
vendors to 0.005 or below. Every similarity in this paper is centered. We regard
an uncentered cosine on these states as uninterpretable rather than merely noisy.

**Every claim carries a floor.** Shuffled-label refits, shuffled-pair retrieval
and analytic chance are reported alongside results, not in an appendix.

**Controls are designed to absorb the boring explanation.** The clearest example
is in §6: a concatenation result that looks like evidence for complementary
information is fully explained by model width, and only a control that
concatenates a backbone *with itself* can show that.

---

## 3. Photographs: vendors agree about which picture

Four backbones encode a shared gallery. A ridge map fitted between two vendors'
image states moves a picture from one vendor's space into another's, where it is
matched against a 1,000-image pool.

| | r@1 |
|---|---:|
| **cross-vendor image agreement, direct map** | **0.8024** |
| routed through a third vendor | 0.7864 |
| shuffled floor | 0.0007 |

Across all 12 ordered cross-vendor pairs, cross-vendor retrieval is
indistinguishable from within-vendor retrieval: 0.2862 against 0.2898 on the
mixed-modality task, a retention ratio of 0.9876 with a 95% CI of [0.9550,
1.0229] over 2,000 bootstrap resamples. The interval contains 1.0. The claim is
indistinguishability, not a specific retained fraction.

### 3.1 A correction: the ratio was the wrong instrument

We initially reported that retention ratio as the headline. It was close to
insensitive to the vendor boundary it was named after, and it refused to move
under isotropic noise, spectral truncation and spectral complement. The reason is
that both of its terms are throttled by the same component. Both terms are r@1 on
the same holdout: within-vendor text-to-image retrieval reaches 0.1050 while
vendor-to-vendor image agreement reaches 0.8024. The encoders agree with each
other about pictures roughly eight times better than any one of them connects its
own captions to its own pictures, so a ratio of two caption-head-limited numbers
cannot report much about encoders.

Dipankar Sarkar proposed the test that settles it: hold the image side fixed and
swap the caption tower. Replacing it with an unrelated `all-MiniLM-L6-v2` on the
radiology gallery moves both terms by nearly the same factor.

| | native head | swapped head | ratio |
|---|---:|---:|---:|
| within r@1 | 0.0887 | 0.0833 | 0.938 |
| cross r@1 | 0.0853 | 0.0830 | 0.973 |

That is what a shared bottleneck predicts, and it is the test designed to break
the reframe rather than confirm it. **Report the legs separately.** The
image-agreement number is the one that bears on the encoders.

He raised a second objection we record without having resolved: the 8x gap
contains a task-ceiling term that the comparison does not net out, because
image-to-image matching has a unique target by construction while text-to-image
is many-to-many on the same holdout. The gap is therefore partly two tasks scored
against different achievable maxima, and the caption-head reading is not the
whole of it.

---

## 4. Satellite: a probe fitted on one backbone reads another

Retrieval establishes that two vendors agree about which picture. It does not
establish that they agree about what is in it. The labelled version of the
question, on RSICD:

17 land-use classes, keyword-matched from the captions, each with at least 100
positives. Linear probe per backbone, then transported across all 12 cross
directions.

| | mean AUROC |
|---|---:|
| native, each backbone probing itself | 0.9507 (spread across vendors 0.0277) |
| self-map control | 0.9517 |
| **transported across backbones** | **0.9484** |
| shuffled floor | 0.5014 |

Scoring is mean AUROC across the 17 classes. **Transport costs 0.0024 mean
AUROC**, and 5 of the 12 transported pairs beat the native target outright.

**The labels are weak and this number should not be quoted without that
sentence.** Scene classes are keyword-matched from captions, which is the same
shape of supervision ChestX-ray14 uses for its fourteen findings, and they are
coarser than clinical labels, which is part of why 0.95 is reachable at all. The
probe reads the image tower while the label derives from the caption, so the text
side cannot leak the answer.

---

## 5. Radiology: transport costs less than nothing

The strongest version of the test uses expert labels, a standard benchmark and a
published split. This section and the two after it use three backbones,
Qwen3-Omni, Gemma-4 and Aria.

**Setup.** All 112,120 images of ChestX-ray14, the **official `test_list.txt`**,
86,524 train and 25,596 test across 30,805 patients, patient overlap zero and
asserted by the script, which refuses to run otherwise. Confidence intervals use
a patient-level cluster bootstrap because the unit that repeats is the patient
rather than the film.

Single backbones first, on the identical probe and split:

| backbone | mean AUROC | vs split-matched baseline |
|---|---:|---:|
| Qwen3-Omni-30B-A3B | 0.7650 | +0.0199 |
| Gemma-4-31B-it | 0.7590 | +0.0139 |
| Aria | 0.7080 | −0.0371 |

Every figure in this section is mean AUROC across the fourteen ChestX-ray14
findings on that test list. The split-matched baseline is 0.7451, from Wang et
al. 2017 (arXiv:1705.02315v5, Table 17), an ImageNet-pretrained ResNet-50
fine-tuned end to end on the same official list. Reference numbers from CheXNet
(0.8414) and Yao (0.8027) are on a different, random 70/10/20 partition and are
**not** comparable to any number here. We assert only that the splits differ,
since Wang scored 0.7381 random against 0.7451 official.

The spread across backbones is 0.057 mean AUROC, which is a real limit on the
single-model claim: linear presence of pathology replicates on all three, but
beating the baseline does not, because Aria does not.

**Now transport.** Probe fitted on one backbone, read on another's states through
a train-only ridge map. The native row is the mean of the three single-backbone
scores above:

| | mean AUROC |
|---|---:|
| native, each backbone probing itself | 0.7440 |
| self-map control | 0.7450 |
| **transported across backbones** | **0.7511** |
| round-trip cycle | 0.7426 |
| shuffled floor | 0.5020 |

**Transport cost is −0.0071.** Moving a probe between backbones costs nothing on
average and usually gains. Per direction:

| probe from | read on | mean AUROC | vs target's own probe |
|---|---|---:|---:|
| gemma4 | qwen3omni | **0.7708** | +0.0058 |
| qwen3omni | gemma4 | **0.7707** | +0.0117 |
| aria | qwen3omni | 0.7522 | −0.0128 |
| aria | gemma4 | 0.7480 | −0.0110 |
| qwen3omni | aria | 0.7325 | +0.0245 |
| gemma4 | aria | 0.7321 | +0.0241 |

Four of six beat the target's native probe. The best single reading in the study,
0.7708, comes from a probe that was never fitted on the states it is reading.

![Every cross direction, measured relative to the target backbone's own probe.
Bars above zero are directions where a borrowed probe outscores the one fitted
natively.](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe/resolve/main/figs/fig1_transport.png)

The same comparison in both labelled domains, with the two controls that bound
it:

![Native, self-map control, transported and shuffled floor, on satellite scene
classes and on chest findings.](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe/resolve/main/figs/fig3_transport_cost.png)

This was a prediction on record. The satellite result implied pathology should
transport if scene content did; the observed cost is smaller still.

---

## 6. Pooling: they are not redundant

If the backbones carried the same information in a shared frame and nothing more,
pooling them would gain nothing. It gains. Scoring is again mean AUROC across the
fourteen findings on the official test list.

| | mean AUROC | vs best single | vs baseline 0.7451 |
|---|---:|---:|---:|
| Qwen3-Omni alone | 0.7650 | | +0.0199 |
| Gemma-4 alone | 0.7590 | −0.0060 | +0.0139 |
| Aria alone | 0.7080 | −0.0570 | −0.0371 |
| **mean of three probes' logits** | **0.7774** | **+0.0124** | **+0.0323** |
| concatenated features | 0.7627 | −0.0023 | +0.0176 |
| control: best single concatenated with itself | 0.7626 | −0.0024 | +0.0175 |

The pooled gain is significant: **+0.0124, 95% CI [+0.0082, +0.0168]**, positive
in 1,000 of 1,000 patient-clustered resamples.

Two things make this hard to explain away. Averaging logits adds **no
parameters**, so the gain cannot be capacity. And **Aria is 0.0371 behind the
baseline on its own** and still improves the average, which means it holds
something the two better backbones do not.

![Each backbone alone, the mean of their logits, feature concatenation, and the
duplicate-vendor control, shown with the split-matched baseline.](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe/resolve/main/figs/fig2_pooling.png)

### 6.1 Concatenation gains exactly nothing, and the control is why we know

Feature concatenation scored 0.7627 mean AUROC, below the best single backbone.
Read alone that is ambiguous: it could mean the extra features are useless, or
that a wider probe is simply worse-conditioned at fixed hyperparameters.

Concatenating the best backbone **with itself** gives identical parameter count
and zero new information. It scored 0.7626. The difference between real
concatenation and the duplicate is **0.0000**. The entire concatenation effect is
width. Concatenation was never retuned, so we bank it as untuned rather than
refuted, but no part of it is evidence about cross-backbone information.

---

## 7. One frame for four backbones

Everything above is bilateral: one ridge map per ordered pair, twelve maps for
four vendors. Fitting a *single* shared frame from all four at once gives four
encoders and four decoders, eight maps instead of twelve. The frame is found by
MAXVAR generalised CCA, which searches for the directions on which all four
backbones agree most strongly and uses those as a common coordinate system.

| | satellite | radiology |
|---|---:|---:|
| direct, 12 pairwise maps | 0.8425 | 0.8817 |
| **via one shared frame, 8 maps** | **0.8164** | **0.8400** |
| cost | 0.0261 | 0.0417 |

Consensus in that frame reproduces the pooling result on a second task: **three
vendors read together beat the best single vendor on all four targets in both
domains, eight for eight.** Mean gain +0.0345 on satellite and +0.0165 on
radiology.

---

## 8. Iterated composition, and a withdrawn reading

A single pass through a cross-vendor map is nearly free, which makes single-pass
tests weak instruments. Iterating the same maps separates them. Carrying a vector
out along a closed route of maps and back, then measuring how far it lands from
where it started, is what the geometry literature calls **holonomy**, and the
script for this section is named for it. Hop counts are matched at 12, 24 and 36,
where every route's period divides evenly.

Holding enclosed area at zero and retracing fixed, and varying only how many
distinct vendor boundaries a route crosses. Scoring is round-trip **r@1**: the
fraction of held-out items that come back to their own starting vector as
nearest neighbour once the route closes.

| domain | distinct edges | 12 hops | 24 hops | 36 hops |
|---|---:|---:|---:|---:|
| radiology | 1 | 0.9470 | 0.8190 | 0.7220 |
| radiology | 2 | 0.9090 | 0.7170 | 0.5580 |
| radiology | 3 | 0.8130 | 0.5640 | 0.3880 |
| satellite | 1 | 0.9650 | 0.7760 | 0.5840 |
| satellite | 2 | 0.9170 | 0.5800 | 0.3770 |
| satellite | 3 | 0.7820 | 0.4230 | 0.2510 |

Fewer distinct boundaries scores higher at every hop count in both domains, six
of six.

![Round-trip r@1 by hop count, for routes crossing one, two and three
distinct vendor boundaries, in radiology and satellite.](https://huggingface.co/datasets/RiverRider/srt-cxr14-frozen-probe/resolve/main/figs/fig4_ladder.png)

**The first reading of this was wrong and is withdrawn.** We originally reported
a three-way ordering and read it as evidence about *enclosed area*. Dipankar
Sarkar pointed out that there-and-back also differs by composing a map with its
own approximate inverse, so errors cancel pairwise, and proposed the control:
`A → B → C → D → C → B → A`, four vendors, every edge retraced, zero enclosed
area. The palindrome tracks the four-cycle, so area is not the variable.

**Caveat carried on the face of the claim.** Applying any linear map over and
over pulls vectors toward whichever direction that map stretches most, until
little else survives. Every route pays that cost whatever it means, so the
degradation itself is not the finding. Hop counts are matched across routes,
which is what makes the *ordering* between them readable. Read the ordering
rather than the size of the split.

**A second reading, and why we do not make it.** Send the same routes through the
shared frame of §7 instead of through the pairwise maps, and the ladder stops
degrading altogether. It sits at 1.0000 at every hop count, including 36.

The tempting conclusion is that the shared frame preserves what the pairwise maps
lose. There is a duller possibility. The joint route projects onto a fixed
128-dimensional subspace on its very first hop, and after that there is nothing
further to discard, so it would read 1.0000 whether or not the frame meant
anything at all.

Those two are separable. We truncated every pairwise map to rank 128 as well, so
both routes pass through a bottleneck of identical width. Rank-matched pairwise
degrades exactly as full-rank pairwise does. The flat ladder is therefore not a
consequence of the bottleneck's width.

What we cannot separate is subtler, and it is why the claim stops here. The joint
route returns to the **same** subspace at every hop. Each pairwise map has its
own preferred subspace, so a route through several of them keeps changing basis.
The flat ladder may say only that reusing one subspace avoids compounding
mismatch, which is a weaker statement than the shared frame carrying more
information. We claim the weaker one.

---

## 9. Negative results

Kept because they bound the claims above.

| hypothesis | result |
|---|---|
| Concatenating backbones' features beats either alone | **Falsified as run.** 0.7627 against 0.7650 best single, and the duplicate-vendor control scored 0.7626. Net of capacity the gain is 0.0000 |
| The chest result holds for any frozen backbone | **Scoped.** Spread of 0.057 across three. Linear presence replicates; beating the baseline does not, since Aria is 0.0371 behind it |
| Vendor-first routing beats a directly fitted cross-vendor map | **Photographs only.** 12/12 on COCO at p=0.0002, but 8/12 on radiology (p=0.19) and 5/12 on satellite (p=0.81). Published as general, withdrawn to one domain |
| The four-cycle penalty is about enclosed area | **Withdrawn.** A palindrome route with zero area tracks the four-cycle |
| A shared frame buys back loop degradation | **Not established.** Rank-matching rules out the trivial reading; what remains may only restate subspace reuse |
| A two-vendor mapping needs a third vendor to mediate it | **Negative.** Routing through a third costs 0.0160 beyond the extra fitted hop it adds, so at one pass a direct pair is enough |
| `retention` measures the vendor boundary | **Withdrawn.** Both terms are caption-head limited; report the legs separately |
| Attention-style pooling beats mean for focal findings | **Falsified.** max −0.0537, top16 −0.0225 on focal findings |
| Readout depth matters | **No.** 0.7600 to 0.7605 across 0.4 / 0.6 / 0.8 of depth |

---

## 10. Limitations

**Detection, not early detection.** ChestX-ray14 labels describe what is visible
in the film. Nothing here bears on catching disease before it is apparent, which
needs longitudinal data with outcomes.

**Not a diagnostic device.** No clinical validation, no prospective evaluation,
no regulatory clearance.

**Labels are NLP-mined** from radiology reports by the dataset authors, and the
satellite labels are keyword-matched by us. Every model on these benchmarks
inherits those ceilings.

**Three backbones on the clinical task.** Every ChestX-ray14 figure here comes
from Qwen3-Omni, Gemma-4 and Aria. The photograph, satellite and shared-frame
results use all four. Whether a fourth backbone would extend the pooled margin
or dilute it is untested.

**Linear probes throughout.** Deliberate: anything stronger begins measuring the
probe rather than the representation.

**The transport maps are linear and fitted per pair.** We have not tested whether
a single map generalises to a backbone held out entirely from map fitting, which
is the natural next experiment and the one that would turn this from a
measurement into a method.

---

## 11. Artifacts

All results are reproducible from published states.

| what | where |
|---|---|
| states, all backbones, 112,120 chest films | `RiverRider/srt-cxr14-frozen-probe` |
| pooled probe weights and normalisation | `RiverRider/srt-cxr14-pooled-probe` |
| single-backbone probe | `RiverRider/srt-cxr14-linear-probe` |
| cross-vendor states, photographs and radiology and satellite | `RiverRider/srt-omni-crossvendor-states` |
| live: one film, three backbones, pooled | `RiverRider/srt-cxr14-probe` |

Scripts: `cxr_probe.py`, `cxr_probe_transport.py`, `cxr_probe_ensemble.py`,
`rsicd_scene_probe.py`, `joint_frame.py`, `holonomy_palindrome.py`,
`head_swap.py`.

Figures: `make_crossvendor_figures.py`, which reads every value from the result
files below rather than carrying its own copy, so a regenerated artifact
regenerates the plot.

Result files: `cxr14_probe_full112k.json`, `cxr14_probe_{qwen3omni,aria}.json`,
`cxr14_transport.json`, `cxr14_ensemble3.json`, `cxr14_vendor_compare.json`,
`rsicd_scene_probe.json`, `joint_frame_{roco,rsicd}.json`,
`holonomy_palindrome_{roco,rsicd}.json`, `geometry_compare_roco.json`,
`head_swap_roco.json`, `xvendor4.json`, `triadic_composition_roco.json`.

---

## 12. Conclusion

We asked whether four independently trained models encode a picture in ways that
can be read interchangeably, and whether reading a second one adds anything.
A probe transports at no cost, and on the official ChestX-ray14 split pooling
still beats the best single backbone.

**A probe moves between them.** Four models, four companies, no shared training
and no fine-tuning. A linear probe fitted on one reads another through a map
estimated on training rows alone, at a cost of 0.0024 on satellite scene classes
and at a **negative** cost on chest pathology, where four of six cross directions
beat the target backbone's own probe.

**They still are not saying the same thing.** Averaging three backbones' logits,
at zero added parameters, beats the dataset authors' fine-tuned ResNet-50 by
0.0323 on the official split, and the backbone that helps most conspicuously is
the one that is worst on its own, 0.0371 behind that baseline unaided.

Held together those findings are narrower than either alone. The models describe
the same picture in compatible coordinates and still register different things
about it. Being readable in the same way is not the same as having read the same
thing, and an evaluation that reports only accuracy separates neither.

The practical reading: a probe is not bound to the backbone it was fitted on, and
a second backbone is worth more as a co-reader than as a replacement.

## Acknowledgments

Dipankar Sarkar proposed the head-swap control in §3.1 and the palindrome control
in §8, and raised the task-ceiling objection we record as unresolved. Two of our
published readings did not survive his tests. Both are withdrawn above.

## References

Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?
Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. EMNLP 2019.
arXiv:1909.00512.

Kettenring, J. R. (1971). Canonical analysis of several sets of variables.
*Biometrika* 58(3), 433–451.

Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J.,
Perona, P., Ramanan, D., Zitnick, C. L., Dollár, P. (2014). Microsoft COCO:
Common Objects in Context. arXiv:1405.0312.

Lu, X., Wang, B., Zheng, X., Li, X. (2017). Exploring Models and Data for Remote
Sensing Image Caption Generation. *IEEE Transactions on Geoscience and Remote
Sensing*. arXiv:1712.07835. (RSICD)

Pelka, O., Koitka, S., Rückert, J., Nensa, F., Friedrich, C. M. (2018). Radiology
Objects in COntext (ROCO): A Multimodal Image Dataset. MICCAI LABELS workshop.

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D.,
Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., Ng, A. Y. (2017).
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep
Learning. arXiv:1711.05225.

Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks. EMNLP 2019. arXiv:1908.10084. (`all-MiniLM-L6-v2`)

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., Summers, R. M. (2017).
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
Weakly-Supervised Classification and Localization of Common Thorax Diseases.
arXiv:1705.02315v5. (ChestX-ray14; Table 17 is the split-matched baseline used
throughout §5 and §6)

Yao, L., Poblenz, E., Dagunts, D., Covington, B., Bernard, D., Lyman, K. (2017).
Learning to diagnose from scratch by exploiting dependencies among labels.
arXiv:1710.10501.
