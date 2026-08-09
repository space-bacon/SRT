# Train Once, Read Everywhere: Substrate Invariance of the Linearly Readable Structure in Frozen Language Models

**James Burton Lancaster**

*The SRT Research Program, consolidated manuscript. July 2026.*

Draft v1. Companion documents: the Stage-3 SRT-Adapter manuscript
([arxiv/paper.md](arxiv/paper.md)), the Stage-4 NLA manuscript
([paper_nla.md](paper_nla.md)), and the engineering guide
([docs/CROSSMODAL_LINEAR_HEAD.md](docs/CROSSMODAL_LINEAR_HEAD.md)).
Every number in this paper has a public artifact and a control behind
it; the artifact index is §14.

---

## Abstract

We report the consolidated findings of the SRT (Semiotic-Reflexive
Transformer) research program: a multi-year effort to treat frozen,
production-scale language models not as black boxes to be fine-tuned
but as *substrates* whose internal states carry structure that small,
inspectable instruments can read. The program's individual results
include a ~12M-parameter adapter that exposes per-token semiotic
signals from a frozen 7B backbone with zero cross-entropy degradation;
an activation verbalizer that recovers text from single hidden states
up to a calibrated paraphrase ceiling; read-out ports spanning
architectures from dense 3B to 94-layer 235B-parameter
mixture-of-experts models; and a 22MB linear head that gives a frozen
multimodal chat model image-to-text retrieval at the level of
fully-trained 2018 dual encoders on the standard COCO benchmark.

The headline finding, established in the program's final arc, unifies
these results: **the readable structure is a stable property of the
model class, invariant along every axis a deployment can vary.** The
cross-modal correspondence inside a frozen multimodal LLM is
anisotropic-linear (a rotation cannot express it; a nonlinear model
finds nothing beyond it at any data scale tested, and falls further
behind as data grows). A head trained once on one host reads, with no
retraining and at most a 42KB recalibration: a host ten times smaller
(31B to 3B, no capability loss at matched data), the same host at
4-bit weight precision (a 0.011 loss in R@1 with unchanged head
weights), and the same computation on entirely different silicon and
kernels (CUDA/bf16 datacenter to Apple-Silicon/MLX-Q4, 97.0%
head-space retrieval agreement over the full 5,000-caption pool
against a 99.96% same-runtime reproducibility ceiling, and
consumer-tier image-to-text retrieval within three R@1 points of the
datacenter reference). Deployment
tiers, from Raspberry-Pi-class edge
devices to datacenter fleets, differ in latency and cost, never in
capability. Train once, read everywhere.

We present the instrument stack, the measurement discipline that made
the numbers trustworthy (anisotropy-corrected metrics, anchored
reference frames, controls at every rung), the invariance evidence,
and an honest ledger of the program's negative results, several of
which are load-bearing for the headline claim.

---

## 1. Introduction

### 1.1 The claim

A frozen language model computes more than its output distribution. At
every layer, for every token, it produces intermediate states, and the
SRT program's founding bet was that these states carry *readable*
structure: signals about meaning, divergence, reflexive awareness, and
discourse position that a small instrument can extract without
touching a single backbone weight.

The program's accumulated experiments support the bet, but the result
that organizes all the others arrived last, and it is stronger than
the bet itself. It is not merely that the structure is readable. It is
that **what the instruments read does not depend on where or how the
substrate runs.** The structure survives a ten-fold change in host
scale. It survives quantization of the weights to four bits. It
survives a change of hardware, numerical kernels, and quantization
scheme simultaneously. And within a multimodal host it is shared
across modalities, connected by a transformation so simple (a single
linear map after per-modality centering) that its entire deployed form
fits in 22 megabytes and can be audited by inspection.

We call this property **substrate invariance**, and we summarize it
operationally: *train once, read everywhere.* An artifact calibrated
against one instance of the model class reads every instance the
deployment world produces, from an edge device performing a single
prefill pass to a datacenter serving a fleet. The tiers differ in
latency and cost. They do not differ in capability.

### 1.2 The program in six stages

1. **Stages 1–2 (theory and pretraining architecture).** The semiotic
   framework: signs, interpretants, attractor basins of meaning, and a
   proposed architecture with these structures built into the
   embedding layer (Lancaster 2025, SSRN 5987495; Lancaster 2026a,
   SSRN 6349978).
2. **Stage 3 (the adapter).** The realization that honest
   instrumentation beats reconstruction: freeze a strong existing
   model and bolt on ~12M trainable parameters that read what is
   already latent ([arxiv/paper.md](arxiv/paper.md)).
3. **Stage 4 (activation verbalization).** Reading a single hidden
   state back as text, with the anisotropy-corrected, anchored
   measurement frame that the rest of the program inherited
   ([paper_nla.md](paper_nla.md)).
4. **Stage 5 (scaling the read-out up).** Ports of the read-only
   adapter to gpt-oss-20b and Qwen3-235B-A22B, establishing
   architecture- and scale-generality upward.
5. **Stage 6 (crossing modalities: SRT-Sunstone).** The discovery that
   a text-trained read-out interprets images zero-shot on a frozen
   multimodal host, and the boundary studies that made the claim
   precise.
6. **The invariance arc (this paper's contribution).** The controlled
   ladder that identified the cross-modal gap as anisotropic-linear,
   benchmarked it, and then deliberately varied scale, precision, and
   hardware to establish that the readable structure is invariant.

### 1.3 How to read this paper

§2 describes the instruments. §3 describes the measurement discipline;
we consider it a contribution in its own right, because nearly every
number in this field is uninterpretable without it. §4 through §7
present the evidence, organized by invariance axis. §8 assembles the
headline claim. §9 is the ledger of negative results. §10 gives the
semiotic interpretation. §11–§13 situate, scope, and conclude.

---

## 2. The instrument stack

All instruments share one design rule: **the backbone is frozen,
byte-identical, and runs natively.** Instruments read hidden states
from forward passes the host is already performing; a deployment gains
capabilities without any risk of regression to the host's primary
function.

### 2.1 The SRT-Adapter (Stage 3)

A ~12M-parameter module (about 0.17% of a 7B backbone) that taps the
residual stream of frozen Qwen2.5-7B at three layers. Metapragmatic
Attention Heads (MAH) read divergence; a GRU-based Reflexive Recurrent
Module (RRM) integrates the divergence stream into a meta-state; a
Bifurcation Estimation Network (BEN) emits per-token reflexivity `r̂`
and a regime label; a community head discovers discourse-trajectory
structure without labels; and an optional FiLM correction
(`h ← h·(1+γ) + β`) is injected at two layers. Because the backbone's
embeddings and LM head are untouched, cross-entropy starts at
pretrained quality; there is nothing to "recover."

Two consumer artifacts ship from this stage: `srt-adapter-v1.0`
(semantic embeddings; the research series behind it culminated in a
model-soup checkpoint, mean MTEB-STS 0.3744) and `zooL4nD3r-v0.1`
(961 discourse communities). A separate diagnostic established the
substrate's raw capacity for one downstream task: frozen-backbone
features plus gradient boosting reach TruthfulQA-MC2 AUC 0.8656 ±
0.011 on Qwen2.5-7B (0.8563 on Gemma-2-2B, 0.8475 on Llama-3.2-3B),
at the top of the published hidden-state-detector band (SAPLMA ≈ 0.72;
INSIDE ≈ 0.78–0.85; EigenScore ≈ 0.80–0.85).

### 2.2 The Activation Verbalizer (Stage 4)

A ~12.7M-parameter head trained so that, given a mid-layer hidden
vector `v` from the frozen backbone, it generates text whose own
re-encoded activation matches `v`. Stage 4's lasting contributions are
the measurement frame (§3) and the decoding result: with oracle
best-of-K scoring (legitimate at deploy time, since `v` is by
construction available), the verbalizer saturates the paraphrase
ceiling at K=64 on Qwen2.5-7B. Its honest limits are equally
important and appear in §9.

### 2.3 The linear cross-modal head (the invariance arc)

The simplest instrument in the program and the one that carries the
headline: two linear maps (5376→1024 each, ~22MB in bf16) trained with
a symmetric InfoNCE objective on frozen-backbone states of paired
images and captions, plus two per-modality centering vectors (~42KB).
Nothing else. §6 and §7 show what it does and what it survives.

---

## 3. Measurement discipline

The program's numbers are trustworthy because of four standing rules,
each learned the hard way.

**Rule 1: always correct for anisotropy.** Transformer hidden states
share a dominant direction; unrelated Qwen2.5-7B L20 states already
have cosine ≈ 0.24 (‖μ‖ ≈ 55), and raw cosine metrics are therefore
uninterpretable. Every similarity in this program is computed after
subtracting the relevant pool mean, and the raw anisotropy is reported
alongside. In the visual channel the same rule reappears with a
refinement (§6.3): the anchor population must be drawn from the
query's own domain, and domain match beats population size.

**Rule 2: anchor every scale.** Stage 4 metrics are reported as
ρ_norm ∈ [0,1], normalized between a measured random-text floor
(centered fve 0.510 on Qwen L20) and a measured same-source paraphrase
ceiling (0.799), with replay (0.968) and nearest-neighbour retrieval
(0.714) as reference points. A number without an anchored frame is not
a result.

**Rule 3: every ladder carries controls.** Shuffled-pairs fits
(capacity floors), train-size curves (memorization tests), held-out
splits that are never touched by model selection, and leakage audits.
The Karpathy benchmark run (§6.2) dropped 8,799 training pairs whose
images belong to the Karpathy test and validation splits but had been
moved into train2017 by COCO's 2017 re-partition, and retrained the
head before evaluating.

**Rule 4: respect encoding conventions.** BOS-sensitivity (a bare
re-encode drops gemma-4 replay from 0.9986 to 0.615), EOS-inclusive
scoring matched between evaluation and calibration, and per-modality
centering are all mandatory and all documented in the released code.

---

## 4. What the taps read on a single substrate

Before invariance can mean anything, the readings themselves must be
established on one substrate. On frozen Qwen2.5-7B the program
established, with full details in the companion manuscripts:

- **Semantic structure**: the adapter's discourse/embedding vector
  supports competitive semantic similarity (the v1.0 product line and
  its research series).
- **Reflexivity and regime**: per-token `r̂` and a two-class regime
  signal, calibrated well enough that downstream ports report
  expected-calibration errors in the fourth decimal place (§5).
- **Recoverability of the state itself**: a single L20 hidden state is
  recoverable as text up to the paraphrase ceiling of the backbone
  under best-of-64 oracle decoding (ρ_cen ≈ 0.99 against the anchored
  frame), with zero-training nearest-neighbour retrieval (ρ ≈ 0.71)
  as the deploy-time fallback that needs no generation at all.
- **Cross-architecture replication of the substrate claim**: the
  Stage-3 read-off survives 1-NN probes on Qwen3-8B and
  Mistral-7B-v0.3, and the NLA pipeline replicated on Llama-3.2-3B
  and Gemma-2-2B.

---

## 5. Invariance axis one: architecture and scale, upward

The read-only adapter ports cleanly to radically different hosts, with
only the ~16M side-channel heads trained and the backbone verified
bit-exact against its reference forward pass in each case.

| host | architecture | regime ECE | regime AUROC | community NMI |
|---|---|---:|---:|---:|
| Qwen3-235B-A22B-FP8 | 94-layer MoE, 22B active | 0.0005 | 0.986 | 0.62 |
| gpt-oss-20b (MXFP4) | sliding-window MoE | 0.0009 | 0.974 | 0.42 |

The 235B port required sharding across eight GPUs and a
sliding-window-mask implementation verified byte-identical to the
reference; the signals survived unchanged. Reflexivity correlation on
gpt-oss-20b: Pearson 0.689. These ports established upward scale- and
architecture-generality well before the question of *downward*
invariance (§7) was posed.

---

## 6. Invariance axis two: modality

### 6.1 The zero-shot discovery (SRT-Sunstone)

A 12.3M community read-out trained on **text only**, attached to
frozen multimodal gemma-4-31B-it, interprets **images** with zero
image training: CIFAR-10 image-to-word retrieval@1 of 0.93 against a
chance rate of 0.10, and open-vocabulary retrieval of full sentences
from a pool of ten thousand COCO captions it was never paired with.
Cross-modal alignment peaks at roughly 80% of backbone depth and
collapses at the final layer, replicating the late-is-surface
signature seen on gpt-oss-20b.

The boundary study made the claim precise. On a random-dot
autostereogram, whose figure exists only in binocular disparity, the
read-out honestly reports texture; a simulated binocular-fusion
front-end recovers the figure from the same pixels, and the read-out
then names it, matching the true-silhouette profile. The capability
gap was in the sensor, not the semiotics. A second boundary, the
apparent inability to retrieve captions for synthetic graphics, was
later shown to be substantially a *calibration artifact*: re-anchoring
the image-side mean on a large natural-photo population moved the
white-heart control's exact caption from rank 352 to rank 5 out of
roughly ten thousand (§6.3).

### 6.2 The modality gap is anisotropic-linear

The question the discovery begged: what transformation connects the
image states and the text states? We answered it with a controlled
ladder on COCO pairs (gemma-4-31B-it, layer 47; evaluation split of
1,000 held-out val2017 images against their 5,000 captions, untouched
by any model selection; shuffled-pairs control at every rung).

| method | i2t R@1 | R@5 | R@10 | t2i R@1 |
|---|---:|---:|---:|---:|
| centered cosine (zero training) | 0.288 | 0.523 | 0.648 | 0.173 |
| orthogonal Procrustes (best variant) | 0.247 | 0.506 | 0.656 | 0.164 |
| **trained linear, 117k pairs, InfoNCE** | **0.661** | **0.911** | **0.967** | **0.506** |
| two-layer MLP, 117k pairs | 0.567 | 0.887 | 0.943 | 0.448 |
| shuffled-pairs control | 0.001 | 0.008 | 0.013 | 0.002 |

![**Figure 1. The method ladder.** A rotation (orthogonal Procrustes)
scores below doing nothing beyond per-modality centering; a single
trained linear map captures the gap; a two-layer MLP with the same
data never reaches it. Shuffled-pairs control at the floor.](arxiv_program/figs/fig1_ladder.png)

Three findings. First, **a rotation cannot express the gap**: every
orthogonal-Procrustes variant scores below doing nothing beyond
centering, because the two modality means are nearly parallel
(cos(μ_img, μ_txt) = 0.979) and the residual difference is a
scale-and-shear that a norm-preserving map cannot fix. Procrustes'
least-squares objective raises absolute paired similarity while
destroying rank discrimination; the objective must optimize ranking.
Second, **a single linear map captures the gap**, and its ceiling is
still rising at the full COCO supervision (gains decelerate from
roughly +0.10 to +0.06 R@1 per doubling; 0.661 is a lower bound).
Third, **there is nothing beyond linear to find**: the MLP trails the
linear map at every training size, and the gap *widens* with data
(0.047 at 39k pairs to 0.094 at 117k). Thirty-three times more data
gives the nonlinear model every chance to reveal structure a linear
map cannot express; it instead falls further behind.

![**Figure 2. Data scaling and the scale floor in one frame.** The
linear map absorbs data faster than the MLP at every size on both
hosts, and the 3B host's curve tracks the 31B host's: a ten-fold
reduction in host scale costs nothing at matched training budget. The
two 39k markers on the 31B linear curve show early-stopping fold
variance on identical training pairs.](arxiv_program/figs/fig2_scaling.png)

On the literature-standard **Karpathy 5k test**, leakage-controlled as
described in §3, the head scores i2t R@1/R@5/R@10 = 0.416 / 0.710 /
0.818 (median rank 2) and t2i 0.292 / 0.567 / 0.685. This matches
fully-trained 2018 dual encoders essentially digit for digit (VSE++:
0.413 / 0.711 / 0.812) from a linear map over a frozen chat model,
using roughly three thousand times less pair data than CLIP-class
systems. The claim is not "beats CLIP" (CLIP-class zero-shot sits near
0.58 at R@1); the claim is **no new model**: retrieval as a free rider
on a host the deployment already runs.

![**Figure 3. Karpathy 5k placement (leakage-controlled).** The linear
head over frozen gemma-4-31B-it matches VSE++, a fully-trained 2018
dual encoder, essentially digit for digit, and sits below CLIP-class
zero-shot, which trained 400M parameters on 400M
pairs.](arxiv_program/figs/fig4_karpathy.png)

### 6.3 The anchor rule, refined

The white-heart revision above generalizes Rule 1 to the visual
channel with a twist a control forced on us: re-scoring the demo
gallery's CIFAR-style thumbnails with the 4,000-photo COCO mean
*degraded* their retrievals, while the same mean rescued the synthetic
probes queried against a COCO pool. The anchor population must match
the query's domain, and when domain and size conflict, domain wins:
150 in-domain images beat 4,000 out-of-domain ones. Practically this
makes anchoring a calibration step, not a constant.

### 6.4 What the head keeps is chosen by the objective, and capacity
is a budget

Compositionality benchmarks probe whether a retrieval system
distinguishes "a horse eating grass" from "grass eating a horse."
CLIP-class dual encoders famously sit near chance on the word-order
splits of SugarCrepe. The shipped head inherits the same failure
(swap splits at 0.50 to 0.57), which is at first surprising, because a
diagnostic on the raw layer-47 states shows the substrate separates
word-order swaps *more* sharply than object replacements (mean
positive-negative cosine 0.396 vs 0.582). The information survives to
the tap; the head discards it, because InfoNCE with random in-batch
negatives never charges for discarding syntax. The failure belongs to
the objective, not the encoder.

Fixing the objective recovers most of it. Adding rule-based
compositional perturbations of each training caption as explicit
hard negatives (noun swaps, then vocabulary replaces, then spatial
preposition replaces and dependency-parsed adjective transfers, up to
K=7 per caption, full-pool in the InfoNCE denominator) moves
SugarCrepe macro accuracy from 0.631 to 0.705 across four retraining
rounds, with gains landing mostly on the trained axes (relation
negatives moved replace_rel +4.7 and attribute negatives moved
add_att +5.1, neither reachable by reweighting the previous mix; the
swap_att gain, by contrast, was 95% available from the weight dial
alone, so per-axis attribution requires the dial as a control). A
weight sweep gives the trade a measured frontier: from retrieval-first
(clean i2t 0.661, macro 0.642) to compositionality-first (clean
0.600, macro 0.705), with intermediate weights dominated on both axes
by w=1.0.

The arc ends at a wall that is itself a finding, twice over. The
union of both negative families does not compose: per split it lands
at approximately max of the parents, not the sum of their exclusive
gains (macro 0.705 vs 0.703/0.685). Because the union run raised the
per-batch negative count 1.75x alongside the mix change, mix and
pressure were deconfounded with a pressure-matched control: the union
subsampled to K=4 (same pool size as either parent, both families
present) lands at macro 0.689, below the specialized parent. Mixing
at fixed budget dilutes (replace_rel 0.758 to 0.735, add_att 0.695 to
0.659), and the union's small edge over the specialized parent was
pressure buying back that dilution, not families composing. The two
levers separate cleanly: the mix chooses which axes move, and the
count pays for coverage, with the count lever cheaper in clean
retrieval per unit of macro (the pressure-matched union also keeps
clean i2t at 0.628 versus the full union's 0.600). The
natural reading of the wall, that the
1,024 projection dimensions are a capacity budget, was then tested
directly and refuted: retraining at proj_dim 2,048 and 4,096 with the
same union negatives reproduces macro 0.705 exactly (0.698/0.705),
so quadrupling the rank of the bilinear similarity buys nothing. The
competition among trained properties (drift-nulling versus
compositional margins versus clean retrieval) is a property of the
objective and the data, not of width: a caption and its perturbation
share one image-side target, and no projection of any rank can
separate what the pooled image state does not distinguish. The last
candidate, pooling itself, was then tested and also eliminated. A
diagnostic on raw states was encouraging: scoring SugarCrepe by the
maximum over four contiguous image-token bands instead of the global
mean lifted exactly the order-sensitive splits (swap_obj +6.5 points
from chance, swap_att +3.2, replace_rel +2.3), suggesting the mean
was destroying real spatial signal. But the trained version of that
hypothesis, a five-slot multi-vector head (four token bands plus the
global mean per image, per-slot centering, max-over-slots InfoNCE,
the same union negatives, re-encoding all 118,287 training images)
lands at macro 0.702, inside the 0.703-0.705 band. Whatever spatial
signal max-over-bands exposes on raw states, the trained linear
readout already extracts an equivalent amount from the pooled state.
A fifth lever closed the within-pair elimination: re-encoding the
decomposition as 2x2 spatial quadrants (calibration first established
that the backbone's image tokens form a row-major, aspect-matched
grid of variable dimensions) reproduces swap_obj at 0.645 to the
third digit, and retraining on the swap-family negatives alone
regresses (macro 0.682, swap_obj 0.633): concentrating the objective
on the permutation axis extracts nothing further.

External review then exposed two flaws in the aggregate itself, and
repairing them reversed the attribution. First, the two add splits
are 99% solvable by caption length alone (the negative is longer by
construction; SugarCrepe's adversarial refinement controlled
plausibility and fluency, not length), so the seven-split macro
blends a text-side artifact into the measurement. Restated over the
five length-clean splits the band is unchanged (0.69-0.71), and the
trained heads score 0.66-0.71 on the add splits rather than 0.99,
confirming that InfoNCE deletes length as a nuisance feature rather
than exploiting it. Second, and decisively: every head in the
elimination read gemma-4 states on both sides, so nothing in it could
separate an image-side limit from a text-side one. A 2x2 tower
factorial with Qwen2.5-VL-3B (layer 29, d=2048) as the second
backbone, identical recipe in every cell (117,787 pairs, the same
negative families re-encoded in whichever space the text tower lives
in), settles the question. Clean five-split macro: gemma/gemma 0.704,
gemma-image with qwen-text 0.665, qwen-image with gemma-text 0.690,
qwen/qwen 0.661. Swapping the text tower costs 2.9 to 3.9 points;
swapping the image tower costs 0.4 to 1.4. The band the elimination
attributed to the image representation is set roughly 3.5x more
strongly by the text representation, and a 10x smaller image tower
read through gemma's text tower loses only 1.4 points. Two structures
survive every cell: swap_att follows the text tower (0.69 and 0.66
under gemma text against 0.61 under qwen text), the clearest
single-split tower fingerprint, while swap_obj moves in no cell at
all (0.600, 0.604, 0.620, 0.604), an object-permutation floor
invariant to which model sits on either side.

The correct summary of the wall is therefore not an image-side
ceiling but a division of labor that is uniform across towers: the
linear readout recovers the scene's inventory nearly in full and its
arrangement hardly at all. Three direct measurements support that
characterization. Scoring 80 COCO category prompts against 1,560
annotated images through the head gives mean per-category detection
AUC 0.883 and per-image R-precision 0.543 against a 0.038 chance
floor, so most object types present in a scene are recoverable from
the single 1,024-dimensional vector. The worst-ranked of each image's
five reference captions still lands at median rank 44 of 5,000, so
the vector answers to every description of the scene rather than to a
dominant subject. And retrieval is flat in scene complexity (r@5
0.906 at one annotated category, 0.887 at six or more). A 22 MB
linear head on a frozen chat model closes 75% of the gap to CLIP
ViT-B/32 on SugarCrepe by objective repair alone; the remaining gap
is a property of the frozen pair, dominated by the text side, with an
arrangement floor that no tested combination of towers, objectives,
or decompositions moves (`artifacts/nla/q4/sugarcrepe_*.json`,
`artifacts/nla/q4/w05_verdict_20260806.json`,
`artifacts/nla/q4/width_null_20260807.json`,
`artifacts/nla/q4/slot_pool_verdict_20260807.json`,
`artifacts/nla/q4/sugarcrepe_mixed_v6.json`,
`artifacts/nla/q4/sugarcrepe_qwen3b_v6.json`,
`artifacts/nla/q4/sugarcrepe_cell4_v6.json`,
`artifacts/nla/q4/inventory_A_multilabel.json`).

---

## 7. Invariance axis three: scale downward, precision, and hardware

The final arc varied the three parameters a deployment actually
controls and measured what the head noticed.

### 7.1 Host scale: 31B → 3B, no loss

The full ladder was re-run, identical code paths, on frozen
Qwen2.5-VL-3B-Instruct (layer 29 of 36, d = 2048, the same 80%-depth
heuristic). Every element of the fingerprint reproduced: the
zero-training baseline works (R@1 0.180, 180× chance); Procrustes
hurts (0.147); the trained linear map dominates (0.577 R@1, 0.870 R@5,
0.955 R@10 at 39k pairs, median rank 1); the MLP trails at every size;
the shuffled control sits at 0.000. The decisive comparison is at
matched training budget: the 3B host's 0.577 against the 31B host's
0.553–0.590 (the range reflecting early-stopping fold variance).
Within noise, **a ten-fold reduction in host size costs nothing.** The
linearly readable cross-modal correspondence is not an emergent luxury
of scale; it is generic to the model class down to at least 3B, and
because the retrieval read-out needs a single prefill pass rather than
generation, 3B at 4-bit is Raspberry-Pi-class hardware.

### 7.2 Weight precision: bf16 → 4-bit, −0.011 R@1

The 3B head, trained on bf16 states, was applied **unchanged** to
states from the same model loaded in 4-bit NF4:

| configuration | i2t R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| head on bf16 states (reference) | 0.577 | 0.870 | 0.955 |
| same head, unchanged, on 4-bit states | 0.566 | 0.857 | 0.941 |
| plus 42KB mean recalibration | 0.569 | 0.868 | 0.948 |

The zero-training baselines are statistically identical across
precisions (0.185 vs 0.180): four-bit quantization barely perturbs the
representation geometry the head reads. A head retrained natively on
quantized states lands on the same data-scaling curve as bf16
training, confirming that quantized states are fully usable substrate;
but the cross-application result shows retraining is unnecessary.
Data budget matters; precision does not.

### 7.3 Hardware and runtime: datacenter CUDA → consumer Apple Silicon

Finally, the original 31B head (trained on CUDA bf16 states in a
datacenter) was tested against states produced by a different
quantization of the same model (MLX 4-bit), running under different
kernels on a consumer machine (Apple M2 Ultra). The protocol is
cross-runtime self-retrieval over the full 5,000-caption calibration
pool, both directions, six arms including two subspace controls, with
the reference re-encoded fresh on CUDA to establish a reproducibility
ceiling
(`artifacts/nla/q4/head_space_validation_v2_20260805.json`).

The ceiling first: re-encoding the identical captions on the identical
CUDA runtime reproduces the stored reference at 99.96% R@1, once
duplicate captions are scored correctly (17 of the 5,000 rows are
string-identical to another row, and returning an identical twin is
not an error). Kernel nondeterminism alone therefore costs about
0.04%, and no cross-runtime number should be read against 100%.
Across the runtime boundary, raw states retrieve their datacenter
twins at 93.3% R@1 as-is. Mean-centering alone is inert (93.4%).
**Through the head**, agreement rises to 96.8%, and to **97.0% with
the 42KB per-runtime mean recalibration**.

Two control arms pin down where the text-side drift lives. Projecting
both sides onto the top-1024 PCA basis of the reference, the
high-variance subspace at the head's own output dimension, lands at
92.2%, below raw. A random orthonormal 1024-dimensional projection
lands at raw (93.2%). Keeping the top tenth of the variance keeps the
damage; the head's row space, which lives in the low-variance tail,
is selectively clear of it. The drift is not a mean shift and not
isotropic: it is concentrated in the high-variance subspace that the
head's 5,376-to-1,024 projection discards.

A correction is owed here. An earlier draft of this section reported
98.4%/100% and attributed them to head space; a reader's code review
established that the validation script never applied the head (it
measured mean-centered raw states at a smaller pool, where the task
saturates). The numbers above are from the corrected protocol, with
the head applied, duplicate-aware scoring, and the subspace controls
the same reader's null experiment called for.

The image side of the same boundary closes the remaining scope, and a
further reader observation sharpened how to read it: the agreement
metric transforms both sides identically, so a shared-mean frame error
cancels there, while an end task scores queries against an external
gallery frame and cannot hide it. The end tasks are therefore the
instrument of record. The 1,000 evaluation images were encoded on both
runtimes (mean of the layer-47 states over image-token positions). On
image-to-text retrieval against the 5,000-caption gallery, the
consumer runtime reaches **R@1 0.640 / R@5 0.903** against the
datacenter reference's 0.670 / 0.919, through the unchanged head plus
the 42KB recalibration; with the shipped training mean instead of the
local one, R@1 drops to 0.401, so the image-side mean correction is
worth 24 points. In the reverse direction, text-to-image with the
same external gallery frame, the consumer runtime reaches **R@1 0.444
/ R@5 0.740** against the reference's 0.503 / 0.808; the per-runtime
mean buys +2.6 points there (and is neutral in a same-runtime
control), with a roughly 6-point structured residual that no mean
correction recovers, consistent with the subspace-control finding
above. Both sides of the boundary carry a mean-frame component that
the 42KB recalibration fixes; the text side additionally carries the
high-variance structured drift, and pooled image states carry almost
none. The claim's "at most a 42KB recalibration" clause does real
work on both sides, and the honest tier summary is end-task: i2t at
95% of the datacenter reference, t2i at 88%
(`artifacts/nla/q4/{image_head_space_validation,t2i_external_frame}_20260805.json`).

A final refinement, again reader-supplied, fixes the units for all of
this. A constant displacement competes with the query it is added to,
so the meaningful knob is not its raw norm but rho, the displacement's
head-space gain divided by the median head-space query norm. In those
units the two branches respond essentially identically to matched
displacement, and the apparent image/text asymmetry in end-task damage
is explained by the query-norm gap alone: the same cross-runtime mean
delta (raw norms 22.7 and 21.5) lands at rho 1.33 on the image branch
and 0.35 on the text branch. There is one displacement, arriving
unequally, not two drift geometries. The same instrument also
characterized a jitter-augmented retraining of the head: training
against random-direction displacements bought 3.0x rho headroom
against random directions but only 1.47x against the real, structured
direction, which is measured evidence that augmentation priors for
runtime drift should be drawn from measured drift families, not
isotropic noise
(`artifacts/nla/q4/round5_v2_direction_20260806.json`).

Acting on that evidence closes the loop. A head retrained with the
measured drift family itself, each training sample perturbed by a real
per-vector cross-runtime residual (3,000 measured MLX-minus-CUDA
pairs, scaled U(0, 1.5)), internalizes the drift outright. On held-out
data whose residuals were never seen in training: image-to-text with
**no calibration at all reaches R@1 0.636**, matching what the
original head needed the recalibration to achieve (0.634);
text-to-image reaches 0.469 against the original head's 0.424 ceiling,
the first crack in the structured residual after post-hoc affine
correction and isotropic jitter both failed; the cross-runtime gap
narrows to 0.2 R@1 points on image-to-text; and there is no
measurable clean-performance tax (leakage-controlled same-runtime
reference 0.660 vs the original head's 0.668, inside noise at this
sample size). The residual is learnable
structure that generalizes across inputs. On the drift-trained head
the 42KB recalibration adds +2.2 i2t and nothing on t2i, against
+24.4 and +2.5 on the original head: the recalibration and
drift-trained retraining are substitutes for the mean component, not
steps of a ladder, and the choice between them is a data-availability
question (200 unpaired states versus 3,000 paired encodes from the
target runtime)
(`artifacts/nla/q4/v3_drift_head_eval_20260806.json`;
`sunstone_linear_head_v3_drift.pt` is the shipped artifact and now
serves the public demo).

An engineering note with strategic weight: on this consumer runtime
the "state tap," the one piece of edge engineering the program had
scoped as remaining work, turned out to require no work at all (the
runtime exposes intermediate states natively). The full stack, a 17GB
quantized 31B host plus the 44MB head, runs on a home computer with no
GPU server involved.

![**Figure 4. Precision and hardware invariance.** Left: the
bf16-trained head applied unchanged to 4-bit states loses 0.011 R@1;
a 42KB mean recalibration recovers half. Right: across a simultaneous
change of hardware, kernels, and quantizer, raw text states agree at
93.3% R@1 over the full 5,000 pool, centering and the subspace
controls do not rescue, and the head lifts agreement to 97.0%,
against a 99.96% same-runtime reproducibility
ceiling.](arxiv_program/figs/fig3_invariance.png)

---

## 8. The invariance claim, assembled

| axis | variation | result |
|---|---|---|
| architecture | dense 7B → sliding-window MoE 20B → 94-layer MoE 235B | read-out signals survive; calibration in the fourth decimal |
| modality | text → images, within one frozen host | shared space; gap is one linear map |
| host scale (down) | 31B → 3B | identical fingerprint; no loss at matched data |
| weight precision | bf16 → 4-bit NF4 | −0.011 R@1, head unchanged |
| hardware / runtime / quantizer | CUDA + bnb → Apple Silicon + MLX | text: 97.0% head-space agreement (n=5,000) vs a 99.96% ceiling, drift pinned to the high-variance subspace the head discards; end tasks on-device: i2t R@1 0.640 vs 0.670, t2i 0.444 vs 0.503, 42KB recal load-bearing on both sides |

The unification: what the SRT instruments read is not a property of a
checkpoint, a precision, a device, or a scale. It is a property of the
**model class**, stable enough that one small artifact, trained once,
reads it everywhere the class is instantiated. The deployment
consequences follow immediately. A single trained head serves an edge
device batch-tagging photographs overnight, a laptop doing interactive
local search, and a datacenter serving a fleet; every tier runs the
*same artifact* against the *same structure*; and because the artifact
is linear, it can be audited by inspection, and because the backbone
is frozen, the host's primary function carries zero regression risk.
The tiers differ in latency and cost. They do not differ in
capability.

We note the claim's shape: it is an *invariance* claim, and its
strongest evidence is a set of results that would individually be
reported as negatives. Rotation fails. Depth finds nothing. Data does
not rescue the nonlinear model. Quantization does not matter. Scale
does not matter. Each null removes a way the structure could have been
fragile.

---

## 9. The negative-results ledger

The program treats documented negatives as first-class findings; the
headline could not be trusted without them.

1. **The greedy gap.** On every host, greedy decoding from the
   activation verbalizer collapses far below retrieval; on
   instruction-tuned hosts the best-of-K curve never reaches the
   retrieval baseline at any practical K (+0.017 per doubling on
   gpt-oss-20b and gemma-4-31B-it, against +0.10 on base Qwen).
   Retrieval, not generation, is the honest decode on tuned hosts.
2. **The base-versus-tuned conjecture, refuted by its own control.**
   An earlier draft conjectured that instruction tuning collapses the
   manifold sampling must traverse. A within-family control (identical
   pipeline on the gemma-4 base checkpoint) produced an
   indistinguishable K-curve (+0.013 vs +0.015 per doubling). Whatever
   sets verbalizability is in the pretrained substrate, not the
   alignment stage.
3. **Draft-conditioning, refuted with a mechanism.** Conditioning the
   verbalizer on the retrieval neighbour's text adds 0.03 nats of
   predictive value for the gold text; cross-entropy training
   therefore has no gradient toward using it.
4. **Causal injection nulls.** Forcing community prototypes into the
   decode path produces near-zero causal movement on v8a (every
   diagonal effect within ±0.002 nats; the pathway verified live by a
   100×-magnified probe moving CE by +2.27 nats), and the released
   adapters' community-conditioning acts as a uniform style bias
   rather than a per-sign disambiguator. The diagnostic half of the
   program is robust; the causal-steering half is scoped out.
5. **The sensor boundary.** The autostereogram result: no read-out can
   report structure the encoder cannot compute. Supply the missing
   computation (simulated binocular fusion) and the boundary moves.
6. **The anchor artifact.** A published boundary claim (synthetic
   graphics "outside the domain") was substantially a calibration
   error, corrected and re-scoped in §6.3. We keep the original claim
   visible in the record because the correction is itself a finding
   about measurement discipline.
7. **Negative families do not compose, and width does not rescue
   them.** The union of two hard-negative families that individually
   moved their own SugarCrepe axes lands at approximately the
   per-split maximum of its parents, and compositional training erases
   a drift-trained head's family-nulling. The obvious explanation,
   1,024 dimensions as a capacity budget, was tested and refuted:
   projections of rank 2,048 and 4,096 reproduce macro 0.705 exactly.
   The competition among trained properties lives in the objective and
   the data, not the width (§6.4, §7.3).
8. **The wall was attributed to the wrong tower.** Five levers
   eliminated within a single backbone pair supported "the wall is
   the layer-47 image representation." A 2x2 tower factorial with a
   second backbone reversed the attribution: the text tower sets the
   band roughly 3.5x more strongly than the image tower, the
   seven-split macro carried a caption-length artifact on its two add
   splits, and the object-permutation floor moves in no cell. An
   elimination inside one pair cannot attribute a limit to either
   side of the pair; both correction and method are due to external
   review (§6.4).

---

## 10. Semiotic interpretation

The program began in Peircean semiotics, and the invariance result
gives the theory its sharpest empirical footing to date. The
*interpretant*, the effect a sign produces in an interpreting system,
was hypothesized in Stages 1–2 to be a real, structured object in
representational space rather than a theoretical convenience. The
evidence now reads: the interpretant-bearing structure (i) exists in
frozen substrates that were never trained to expose it, (ii) is shared
across the modality in which a sign arrives (word, photograph, or
decoded depth map complete the same interpretant, and name the
recovered figure identically to the real one), (iii) is *linearly*
situated, so that reading it never requires reconstructing the system
that produced it, and (iv) is invariant to the physical realization of
the substrate, which is exactly what one would demand of a structural
property of meaning as opposed to an artifact of a particular network.

The program's law-like battery on contestedness (dissociation,
coupling, causal forcing, and a diachronic test over 11,876 dated
articles spanning 1770–1964, where contested political signs' measured
divergence drifts above concrete controls with a
difference-in-differences of +0.048, permutation p = 0.007, crossing
over in the early twentieth century) locates the semiotic load in the
sign's representation itself rather than in any injectable community
signal, consistent with the causal nulls of §9.4.

---

## 11. Related work

The measurement frame descends from the anisotropy literature
(Ethayarajh 2019). The cross-modal ladder engages the cross-lingual
embedding alignment tradition (orthogonal maps between separately
trained spaces) and finds its central tool inapplicable here, for a
reason worth stating: separately trained embedding spaces differ by
rotation-like transforms, but two modalities inside one jointly
trained substrate differ by translation and anisotropic scale. The
retrieval baselines are the standard COCO lineage (Karpathy &
Fei-Fei 2015; VSE++, Faghri et al. 2018; CLIP, Radford et al. 2021).
The hidden-state diagnostic band is SAPLMA (Azaria & Mitchell 2023),
SAR (Duan et al. 2024), and INSIDE/EigenScore (Chen et al. 2024). The
adapter's contrastive components use supervised contrastive learning
(Khosla et al. 2020). The theoretical frame is Peirce, read through
Kockelman's semiotic stance and Silverstein's metapragmatics
(Lancaster 2025, 2026a).

## 12. Limitations

The invariance evidence spans 3B to 235B, three architectures, two
quantization schemes, and two hardware stacks; it does not yet span
model *families* trained by different organizations on the
cross-modal axis (the linear-gap ladder is validated on gemma-4 and
Qwen2.5-VL; the ceiling numbers are gemma-4's). The linear ceiling at
full COCO supervision is a lower bound, not a measured plateau.
Sub-3B hosts are untested. The 4-bit hardware result uses NF4 and
MLX-Q4; other quantizers should track but are unmeasured. Retrieval
quality inherits pool coverage, and anchoring is a genuine calibration
step (§6.3) that deployments must perform. Finally, the causal
(injection/steering) half of the original program remains an open
negative: these are reading instruments.

## 13. Conclusion

The SRT program set out to show that frozen language models are
instrumentable substrates. It ends with something stronger: the
structure the instruments read is invariant across the entire
deployment envelope, scale, precision, and silicon, and cross-modally
linear within a host. The practical corollary is a new class of
artifact, small enough to ship as a configuration file, honest enough
to audit by inspection, that grants a capability wherever its model
class runs. The theoretical corollary is that meaning, as these
substrates organize it, is a stable, linearly readable,
realization-independent structure. Train once, read everywhere.

---

## 14. Artifact index

**Models (Hugging Face, `RiverRider/`):** `srt-adapter-v1.0`,
`srt-adapter-v8a`, `zooL4nD3r-v0.1`, `srt-adapter-qwen3-235b`,
`srt-adapter-gptoss20b`, `Gemma-4-31B-it-SRT-Sunstone`,
`srt-sunstone-linear-head` (the 22MB head + anchors),
`srt-nla-av-v1`, `srt-nla-av-gemma4`, `srt-nla-av-llama32-3b`.

**Data and results (`RiverRider/srt-nla-gemma4-artifacts`):**
`procrustes/` (ladder results, fitted maps, 118k pair encodings,
Karpathy eval), `scalefloor/` (3B replication + head), `q4/`
(quantization-drift results + native-Q4 head), `overnight/` (run logs
and summaries).

**Code (github.com/space-bacon/SRT):** the adapter and NLA packages
(`srt/`, `srt_introspect/`), the experiment scripts
(`scripts/gemma4_procrustes_xmodal.py`, `gemma4_encode_pairs.py`,
`gemma4_mlp_align.py`, `gemma4_karpathy_eval.py`, `q4_drift_eval.py`,
`local_sunstone.py`), and the engineering guide
(`docs/CROSSMODAL_LINEAR_HEAD.md`).

**Demos:** `RiverRider/srt-sunstone`, `RiverRider/srt-showcase`,
`RiverRider/srt-nla-gptoss20b-trace`, `srt-adapter-v1.0-demo`.

## Acknowledgments

The hardware-and-runtime section (7.3) owes its final form to Dipankar
Sarkar, who reviewed the invariance evidence in public over five
rounds: a code review that established the original validation never
applied the head; an isotropic-noise null and subspace controls that
turned "the head avoids the drift" from asserted into measured; the
observation that same-transform agreement metrics structurally cancel
frame errors that end tasks expose, which also surfaced a
mean-calibration bug in the deployed system; an analytic decomposition
of the mean-swap arm; and the rho normalization that put both modality
branches in common units. Every correction was accompanied by
reproductions run from the published artifacts alone. The section's
remaining errors are ours.

## References

- Azaria, A., & Mitchell, T. (2023). The internal state of an LLM
  knows when it's lying. *Findings of EMNLP*.
- Chen, C., et al. (2024). INSIDE: LLMs' internal states retain the
  power of hallucination detection. *ICLR*. (Source of the INSIDE and
  EigenScore reference band.)
- Duan, J., et al. (2024). Shifting attention to relevance: Towards
  the predictive uncertainty quantification of free-form large
  language models. *ACL*. (SAR.)
- Ethayarajh, K. (2019). How contextual are contextualized word
  representations? Comparing the geometry of BERT, ELMo, and GPT-2
  embeddings. *EMNLP-IJCNLP*.
- Faghri, F., Fleet, D. J., Kiros, J. R., & Fidler, S. (2018). VSE++:
  Improving visual-semantic embeddings with hard negatives. *BMVC*.
- Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic alignments
  for generating image descriptions. *CVPR*.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
  Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised
  contrastive learning. *NeurIPS*.
- Kockelman, P. (2005). The semiotic stance. *Semiotica*, 157(1/4),
  233–304.
- Lancaster, J. B. (2025). The treachery of signs: Semiotic mediation,
  pitchfork bifurcation, and political polarization in algorithmically
  curated societies. SSRN 5987495.
  https://papers.ssrn.com/abstract=5987495
- Lancaster, J. B. (2026a). The Semiotic-Reflexive Transformer: A
  neural architecture for detecting and modulating meaning divergence
  across interpretive communities. SSRN 6349978.
  https://papers.ssrn.com/abstract=6349978
- Lancaster, J. B. (2026b). The SRT-Adapter. Repository manuscript,
  [arxiv/paper.md](arxiv/paper.md).
- Lancaster, J. B. (2026c). Natural-Language Activations. Repository
  manuscript, [paper_nla.md](paper_nla.md).
- Peirce, C. S. (1931–1958). *Collected Papers of Charles Sanders
  Peirce* (C. Hartshorne, P. Weiss, & A. W. Burks, Eds.). Harvard
  University Press.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning
  transferable visual models from natural language supervision.
  *ICML*.
- Silverstein, M. (1976). Shifters, linguistic categories, and
  cultural description. In K. H. Basso & H. A. Selby (Eds.), *Meaning
  in Anthropology* (pp. 11–55). University of New Mexico Press.
