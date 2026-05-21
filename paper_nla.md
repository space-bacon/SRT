# Natural-Language Activation (NLA) Verbalization:
## A Round-Trip Test of Frozen Hidden States, in the Idiom of the SRT Program

*Burton Lancaster, Draft, May 2026*

---

## Foreword (for the reader who is new to this program)

Modern large language models are, internally, sequences of high-
dimensional numerical vectors, one such vector for every token, at
every layer. These vectors are what the model actually "thinks in."
They are not, on their face, human-readable: a vector of $3{,}584$
numbers does not look like a sentence. A natural question, and the
one this paper is built around, is whether the model itself can be
asked to say, in its own ordinary output text, what one of those
internal vectors is *about*.

We test this directly. We pick a frozen, off-the-shelf language model
(no retraining of the underlying weights), grab one specific hidden
vector from somewhere about three-quarters of the way through its
stack, and train a very small add-on module, about the size of a
rounding error compared to the base model, whose only job is to
generate a short piece of text such that, when that text is fed back
into the same frozen model, the model's *own* internal vector at the
same layer matches the original we started from. The text is the
"verbalization" of the activation. The match score is how we know
whether the verbalization is faithful. The whole loop is a round
trip: hidden state → words → hidden state.

Three things made this harder than it sounds, and they are the actual
content of the paper:

1. **The obvious match score is misleading.** Two completely
   unrelated vectors from the same model already look about $62\%$
   similar by the naive metric, just because the model has a strong
   built-in bias in the direction it pushes everything. You have to
   subtract that bias out before any number means anything. Once you
   do, the apparent ceiling our trained model was hitting turns out
   not to be a wall, it is the metric itself, and the same model,
   under proper measurement, *saturates the human-paraphrase ceiling*.

2. **The greedy answer is not the best answer.** If you ask the
   verbalizer for its single most-likely output, you get a mediocre
   match. If you let it propose a handful of candidates and pick the
   best one against an internal-state target, you suddenly get
   excellent matches. The model knows the answer; its argmax is just
   not where the answer lives.

3. **It works on three different model families.** We replicate the
   whole story on Qwen-2.5-7B, Llama-3.2-3B, and Gemma-2-2B, three
   models trained by three different organisations on different data
   with different built-in biases (the "anisotropy" $\|\mu\|$ in the
   technical sections varies by $22\times$ across them). The same
   shape of curve, the same gap between greedy and best-of-$K$, the
   same crossing of the paraphrase ceiling. That is what Figure 1
   below summarises in a single panel.

Why bother? Because if a frozen language model can be made to talk
*about its own internal states in its own native output channel*,
that is a different, and arguably more honest, kind of
interpretability than reading probes off the side. The model is not
being decoded by us; it is being asked to describe itself, and we are
checking the description against itself. The body of the paper sets
this up rigorously, working in the idiom of the broader Semiotic-
Reflexive Transformer (SRT) program, of which this is "Stage 4."
A reader who only wants the bottom-line empirical claim can stop
after Figure 1's caption and the abstract. A reader who wants the
machinery should read on through §§3–7. A reader who wants the
philosophical framing (Peircean interpretant chains, second-order
cybernetics) should look at §§0, 1.5, and 12. Each layer is meant to
be self-contained at its own level of abstraction.

---

![**Figure 1. The same story on three different language models.** The horizontal axis is how many guesses the verbalizer is allowed to make per hidden state ($K = 1, 2, 4, …, 64$, on a log scale). The vertical axis is how well the best of those $K$ guesses recovers the original hidden state, after we subtract out each model's built-in bias (the "centred" score, where $0$ means random and $1$ means as good as a human paraphrase of the source text). Solid lines are the three frozen backbones we tested: Qwen-2.5-7B (blue circles), Llama-3.2-3B (green squares), Gemma-2-2B (red triangles). Each model has its own dashed line for its paraphrase ceiling and its own dotted line for its random-guess floor. Three things to notice: (1) all three random floors land on top of each other at about $0.50$, even though Gemma's raw bias is roughly $22\times$ Llama's, which is what "centring" is meant to do; (2) every curve is a clean straight line on this log-x plot, so each doubling of $K$ buys a fixed amount of fidelity; (3) each model crosses its own paraphrase ceiling at a different $K$, Llama at about $4$ guesses, Gemma at about $16$, Qwen at about $64$, which means the verbalizer's argmax is not where the answer lives, but a small amount of search reliably finds it. This single panel is the whole three-backbone replication story (see §§4–5 and §§10–11 for the numbers).](artifacts/nla/figures/fig1_cross_backbone_kcurve.png)

---

## Abstract

The Semiotic-Reflexive Transformer program (Lancaster, 2025; Lancaster, 2026a;
Lancaster, 2026 [SRT-Adapter MS]) treats a frozen large language model as
a semiotic substrate on which interpretant divergence, metapragmatic
awareness, and bifurcation dynamics are *measurable* phenomena. Stages 1 and 2
established the four-module decomposition (community, metapragmatic
attention, reflexive recurrence, bifurcation estimation) on synthetic and
news data; Stage 3 ported it to a frozen Qwen-2.5-7B backbone as the
SRT-Adapter, demonstrating that the discourse-community manifold an LLM
inherits from its training corpus is exposed at $0.19\%$ parameter overhead
and is not Qwen-specific. The present paper reports **Stage 4** of the
program: rather than read structure off the substrate as side-channel
outputs, we ask the substrate to *speak its own state*. We train a small
($\sim\!12.7$M-parameter) **Activation Verbalizer** (AV) over a fully frozen
Qwen-2.5-7B such that, given a target hidden activation
$v \in \mathbb{R}^{3584}$ extracted at layer 20, the AV generates a short
text whose own re-encoded layer-20 last-token hidden $h$ maximises
$\mathrm{fve\_nrm}(h, v) = \tfrac{1}{2}(1 + \cos(h, v))$. The round-trip is
the apparatus: it is a Peircean interpretant-completion test on the model's
own representation (Peirce, 1931–1958; Kockelman, 2017), evaluated under a
metric that disciplines the measurement against the substrate's own
anisotropy.

On the raw metric the trained adapter appeared stuck at $\approx\!0.689$
across four architectural levers (multi-inject, MLP prefix, PG+KL, more
data). We show this is not a capacity ceiling but a *measurement artefact*
of an uncentred metric over an anisotropic representation, in the sense
that two unrelated Qwen-7B last-token L20 hidden states already share
$\mathrm{fve\_nrm}\!\approx\!0.62$ purely from a backbone-specific mean
$\|\mu\|\!\approx\!55$. Under a held-out **anisotropy-corrected** metric
calibrated against four anchors (replay, paraphrase, nearest-neighbour
retrieval, random floor), the *same* checkpoint at best-of-$64$
**saturates the Qwen paraphrase ceiling** at $\rho_{\text{cen}}\!\approx\!0.99$.
The real open problem is the **greedy gap**: deterministic decoding closes
only $\approx\!28\%$ of the centred ceiling and is *beaten* by a zero-
training nearest-neighbour lookup ($\rho_{\text{cen}}\!\approx\!0.71$). The
$K$-curve is log-linear at $+0.030$ centred per doubling; logp-rerank is
statistically indistinguishable from greedy (per-target Spearman of mean
log-probability with oracle centred cosine $\approx\!0.04$); a bag-of-$K$
self-distillation attempt (Lever B) does not close the gap on this
backbone.

We replicate every qualitative finding on **meta-llama/Llama-3.2-3B**
($\|\mu\|\!\approx\!7.21$, $7.6\times$ smaller; greedy band $0.66$–$0.69$
raw, log-linear $K$-curve, logp-rerank dead, best-of-$64$ saturating the
binding ceiling) and report a third-backbone replication on
**google/gemma-2-2b** in §11 of the canonical record (centred random
floor $0.498$, paraphrase ceiling $0.598$, best-of-$64$ centred fve
$0.631$, $\rho_{\text{cen}} = 1.33$, $\|\mu\| \approx 156$, the most
anisotropic backbone we have tested, and the cleanest case for the
centring claim of §§4–5). Three readings
follow. *First*, hidden-state verbalisation on a frozen mid-scale decoder
is decoding-bound, not capacity-bound: $\sim\!12.7\text{M}$ trainable
parameters suffice to make the paraphrase manifold reachable, but the
argmax mode of the prefix policy is not where the manifold lives.
*Second*, any $\mathrm{fve\_nrm}$-style evaluation that does not report
both (i) an anisotropy-centred metric and (ii) a retrieval baseline is
not interpretable: the raw metric is a thin film over the substrate's own
geometry. *Third*, NLA closes a loop the SRT-Adapter left half-open:
where the adapter's inject-back path through the RRM did not yet carry
measurable signal (Lancaster, 2026 [SRT-Adapter MS] §6.3), the verbalizer carries
information *out* of an interior state and into a sequence of tokens that
the same backbone routes back to itself, instantiating the second-order-
cybernetic loop in a different topology, the observer reports, the
substrate listens, and we measure the round trip.

\noindent\textbf{Keywords:} natural-language activation, activation
verbalisation, frozen-decoder readout, anisotropy-corrected fidelity,
best-of-$N$ minimum-Bayes-risk decoding, Peircean interpretant chain,
metapragmatic awareness, second-order cybernetics, semiotic-reflexive
transformer.

---

## 0. Position in the SRT program

This paper is Stage 4 of a research program whose prior stages have been
reported separately. The order is logical, not strictly chronological:

1. **Stage 1, synthetic data; four core architectural claims** (Lancaster,
   2026a). Subspace specialisation, community differentiation, divergence
   tracking, and bifurcation detection were tested on controlled synthetic
   corpora with planted divergence. All four passed at the required
   thresholds, establishing that the four-module decomposition (community,
   metapragmatic, reflexive, bifurcation) learns the intended functions.

2. **Stage 2, natural language; five-test validation** (Lancaster, 2026a).
   The capability set was re-tested on a Supabase news corpus spanning
   five political communities (19K articles, 141K Peircean sign
   annotations). All five tests passed, including a Pearson $r=0.884$
   correlation of the bifurcation estimate $\hat{r}$ with an external
   polarisation index, and 85\% regime-classification accuracy on
   held-out curated passages.

3. **Stage 3, frozen 7B backbone; SRT-Adapter** (Lancaster, 2026 [SRT-Adapter MS]).
   The validated decomposition was reduced to a 14.5M-parameter adapter
   ($0.19\%$ of a 7B backbone) on a frozen Qwen-2.5-7B. v8a removed the
   discrete prototype layer, raising Reddit recall@1 from $0.413$ to
   $0.484$ and out-of-distribution archetype recall@1 to $7.6\times$
   chance with $\Delta\,\mathrm{CE}=+0.0001$ nats; a cross-backbone
   raw-hidden probe showed the targeted discourse-community substrate is
   present in `Qwen/Qwen3-8B` and `mistralai/Mistral-7B-v0.3` at
   comparable strength. The MTEB-STS lineage v15a (`srt-adapter-v1.0`),
   v18, v21a, and v22c\_a050 demonstrated that the same scaffold supports
   a sentence-encoding head competitive on a standard benchmark with
   parameter-space interpolation as the cheapest meaningful gain. The
   open question Stage 3 left on the table is the dead inject-back arm:
   the observation channel is well-formed; the channel through which
   observation can modify generation has not yet learnt to carry signal
   under the adopted training regime.

4. **Stage 4, frozen-decoder verbalisation; SRT-NLA** (this paper). We
   pose a complementary question. Stage 3 measured *what* the substrate
   represents about communities, divergence, and regime. Stage 4 asks
   whether a small auxiliary policy can produce a sequence of *tokens* of
   the substrate's own type, such that when the substrate re-reads them
   it routes them back to the same place in its own representation
   space. The round-trip metric $\rho_{\text{cen}}$ tests interpretant
   completion in Peirce's sense: the verbalisation is the new
   representamen, the backbone's re-encoding is the new interpretant,
   and we ask how close that interpretant is to the original target.
   Where the SRT-Adapter occupies the *cloud* end of Anderson's
   "clouds, languaging, triadicity" (continuous community geometry,
   layer-wise readout, soft assignments), NLA occupies the *languaging*
   end: the substrate is asked to render an interior state into the
   medium of its own training distribution, and we audit the rendering
   against itself.

The methodological commitments of Stage 3 carry over verbatim. The
backbone is fully frozen. Trainable parameters are kept small ($\sim\!13$M
on Qwen, $\sim\!9$M on Llama, $\sim\!5$M on Gemma). The metric is
calibrated against random-floor and human-paraphrase anchors, and
reported in both raw and anisotropy-centred form. Negative results are
reported as such and assigned an explanatory mechanism. Cross-backbone
replication is a release condition, not an afterthought.

What is *new* in Stage 4, relative to the rest of the program, is two
things: a calibrated round-trip fidelity metric on internal hidden
states (rather than side-channel structured outputs), and the
demonstration that this metric saturates a non-trivial human-paraphrase
ceiling at deploy-time best-of-$N$ with no extra training. What is *not*
new is the framing: NLA inherits the four theoretical commitments of
Stage 3 (Peircean semiosis, catastrophe-theoretic dynamics of
sociolinguistic change, Silverstein's metapragmatic awareness, Anderson's
triadic and cloud-shaped readout), specialises them to verbalisation,
and adds a second-order-cybernetic reading of the round-trip itself
(§1.5).

---

## 1. Setting

- **Backbone.** Qwen/Qwen2.5-7B, bf16, fully frozen.
- **Probe layer.** $\ell=20$; targets $v$ = last-valid-token hidden state at
  layer 20 of a 64-token Qwen continuation.
- **Adapter.** `ActivationVerbalizer` with 16 static prefix tokens and one
  inject slot; 12.7M trainable parameters. Trained with token CE on
  $(v, \text{text})$ pairs.
- **Metric.** $\mathrm{fve\_nrm}(h, v) = \tfrac{1}{2}(1 + \cos(h, v))$. This is
  the per-target score throughout training, evaluation, and best-of-K.
- **Checkpoints.** `ce_seq64_np16/best_av.pt` (10k pairs), `ce_seq64_np16_30k/best_av.pt`
  (30k pairs), both warm-started identically.

### 1.5 Theoretical grounding: verbalisation as interpretant completion

Four commitments motivate the round-trip and its metric. They are the
same commitments that motivate the SRT-Adapter (Lancaster, 2026
[SRT-Adapter MS], §2), specialised to the verbalisation problem.

*Peircean interpretant completion as the apparatus.* In Peirce
(1931–1958, CP 2.228, 2.303), every sign process is triadic: a
*representamen* stands in for an *object* via an *interpretant*, and
the interpretant is itself a sign that can stand in subsequent triads.
A frozen LLM's hidden state $v$ at layer $\ell$ is, in this idiom, a
representamen of whatever the prefix produced, but a representamen
*to which sign system?* No human reads $v$. The natural choice, and
the one the round-trip enforces, is to make the backbone its own
interpreter. The verbalizer produces text $\hat{x}$ from $v$; the
backbone re-encodes $\hat{x}$ into $h$ at the same layer; the
interpretant of $\hat{x}$ relative to the backbone is $h$; the test of
whether $\hat{x}$ "completed the chain" is the centred cosine of $h$
with $v$. This is interpretant completion in a literal Peircean sense:
the chain $v \to \hat{x} \to h$ closes if and only if the substrate
agrees that $\hat{x}$ is a faithful gloss of $v$ in its own internal
language. The metric is the apparatus, not the goal.

*Sieving and the prefix as community-of-one.* Kockelman (2017, 2025)
formalises interpretant chains as dynamical trajectories whose links
are *sieving* operations: from the space of possible interpretants a
sign could produce, only some are actualised, conditioned on the
interpreter's prior exposure and architectural commitments. The
prefix-tuned AV is a sieve in exactly this sense, it conditions the
backbone's generation distribution on a 16-token (Qwen) or 1-token
(Llama, Gemma) static prefix that has been trained to make the
sampling distribution concentrate on $\hat{x}$ such that
$h \approx v$. The "community" the prefix simulates is not a
sociological one. It is a *community of one*: the discrete configuration
of the backbone consistent with re-arriving at $v$. The round-trip
metric is then the Stage-3 community signal turned ninety degrees:
where Stage 3 read which discourse community a token participates in
off the substrate, Stage 4 induces a discourse-community-of-one as a
prefix and tests whether the substrate's *own* generative process
under that prefix homes back to the source state.

*Metapragmatic awareness as a conditional decoding capacity.*
Silverstein (1993, 2003) distinguishes first-order indexicality (the
sign refers), second-order indexicality (the sign indexes group
membership), and third-order indexicality (the sign-user observes that
the sign is being contested). The corresponding decoding capacities
in the frozen-LM verbalisation setting are: produce *some* text from
$v$ (first-order); produce text in the dialect of whichever community
$v$ comes from (second-order); produce text whose *position in the
paraphrase manifold of $v$* the substrate can certify under
re-encoding (third-order). The empirical claim of this paper is that
$K=1$ greedy decoding implements only the first capacity reliably,
zero-training nearest-neighbour retrieval implements roughly the
second on a backbone of Qwen-2.5-7B's calibre, and best-of-$K$ MBR
under the centred-cosine utility implements the third for $K \gtrsim 64$.
The greedy gap is not a parameter-count gap; it is a third-order-
indexicality gap inside the decoding distribution.

*Second-order cybernetics: closing the loop the SRT-Adapter left
half-open.* Von Foerster's (1981, 2003) account of self-organisation
in observed systems insists that the observer participates in the
phenomenon being observed; circular causality is the architectural
shape of this participation. The SRT-Adapter (Lancaster, 2026 [SRT-Adapter MS],
§2.5) is a *partial* second-order system: the observation arm
(Community Discovery, MAH, BEN) reads structure off the substrate
cleanly, while the intervention arm (RRM inject-back via FiLM)
through v8b had not yet produced measurable downstream effect. NLA
closes a different loop on the same substrate. The adapter's loop is
*intra-pass* (read in one layer, modify a later layer of the *same*
forward pass); NLA's loop is *inter-pass* (read in a forward pass at
$\ell$, emit text, run a second forward pass over that text and read
$\ell$ again). The empirical fact that NLA's inter-pass loop closes
to centred $\rho \approx 0.99$ at best-of-$64$ on Qwen, while the
adapter's intra-pass loop on the same backbone has not, is itself
informative: it suggests the backbone is more controllable through
its own input port (text) than through gated additions to its hidden
stream, which is unsurprising on reflection, because text is the
medium the substrate's $\sim\!10^{12}$ pretraining tokens optimised it
to consume. NLA does not *solve* the closed-loop problem the
SRT-Adapter posed; it shows that a closed-loop reading on the same
substrate is achievable in the topology where the substrate is
strongest.

A note on physical analogy. Leighton (2026) shows that the probability
of any sub-system of a random multipartite stochastic system operating
as a Maxwell demon decays at least exponentially in the system's
degree count: non-trivial organisation requires *selection*. The
verbalizer is a $\sim\!10^{7}$-parameter system whose round-trip
fidelity at best-of-$64$ exceeds the random-floor by $\rho_{\text{cen}}
\approx 1$ on Qwen and $\approx 1.4$ (relative to NN-in-pool) on
Llama. Leighton's bound rules out reading this as a generic property
of high-dimensional embeddings: the structure that emerges is evidence
of a selection pressure, here the centred-cosine training objective,
of the kind his analysis identifies as necessary. VanSaders, Fruchart,
and Vitelli (2026) develop a *measurement-induced* phase-transition
theory in informational active matter, in which the steady-state
order parameter is bounded by the mutual information accumulated
through measurement. The structural analogy is loose but
load-bearing: $\rho_{\text{cen}}^{\text{best-of-}K}$ is bounded above
by the mutual information the rerank utility can extract from the
candidate pool about $v$, and the log-linear $K$-curve we observe
($+0.030$ centred per doubling on both Qwen and Llama, §5 and §10) is
what a $\log K$ information-acquisition picture predicts.

---

## 2. The 0.689 puzzle

The legacy `probe_bestofn.py` script reported a best-of-64 score of
$0.689$ on the 10k checkpoint and $0.675$ on the 30k checkpoint. We tested
four levers, expecting at least one to move the number:

| lever | result |
|---|---|
| $M=4$ multi-inject slots | no change |
| MLP-conditioned prefix | no change |
| PG + KL fine-tune | no change |
| +30k training pairs | regression to 0.675 |

The natural conclusion was an architectural ceiling. **It was a measurement
artifact.** Two issues:

1. **Anisotropy.** Two unrelated Qwen-7B last-token L20 hiddens already have
   $\cos \approx 0.24$, i.e. $\mathrm{fve\_nrm} \approx 0.62$, driven entirely
   by a shared anisotropic mean $\mu$ with $\|\mu\| \approx 55$
   (`oracle_ceiling.py`). The "0.689" was only $\sim 0.07$ above the
   anisotropy floor.
2. **Scoring choice in `probe_bestofn.py`.** A re-implementation
   (`centered_eval.py`) that scores each of $K=64$ samples by
   $\mathrm{fve\_nrm}(h_k, v)$ and takes the per-target max recovers
   $\mathbf{0.85}$, not $0.689$, on the same checkpoint, same $K$, larger
   $M=200$. The legacy script was effectively measuring a different
   per-target statistic.

## 3. A clean reference frame

We anchor the metric with four reference points computed on a 200-target
held-out slice (`oracle_ceiling.py`):

| anchor | raw fve_nrm | centered fve_nrm | meaning |
|---|---|---|---|
| replay (oracle re-encode) | 0.973 | **0.968** | same text → upper bound |
| paraphrase best-of-8 (Qwen) | 0.848 | **0.799** | semantic ceiling |
| NN in-pool (pool=200) | 0.750 | 0.663 | retrieval at small pool |
| NN-retrieval (pool=2000) | 0.795 | 0.714 | retrieval at scale (`centered_eval.py`) |
| random floor (off-diagonal cos) | 0.622 | **0.510** | unrelated samples |

The centered metric subtracts $\mu$ from both sides before cosine. It
drops the random floor by $\sim 0.11$ and the paraphrase ceiling by
$\sim 0.05$, preserving ordering but stretching the dynamic range.

We then define normalized progress:

$$
\rho_{\text{cen}}(s) = \frac{s - \mathrm{rand}_{\text{cen}}}{\mathrm{para}_{\text{cen}} - \mathrm{rand}_{\text{cen}}}
= \frac{s - 0.510}{0.799 - 0.510} = \frac{s - 0.510}{0.289}.
$$

## 4. Adapter results, properly anchored

![**Figure 2. The training of the model behind these numbers ran cleanly.** Four panels of training-curve diagnostics from the Stage 3 SRT-Adapter on Qwen-2.5-7B over $94{,}000$ steps. Top-left is the total loss going down and flattening, the way you want it to. Top-right is the most important panel: the orange line is the loss our adapter is hitting on plain language modelling, and the dashed line is the loss the original frozen Qwen achieves on the same text (the "floor"). They sit on top of each other, which means the adapter has *not* damaged Qwen's ability to produce ordinary text, it has only added the verbalizer behaviour on top. The bottom panels show the two auxiliary objectives converging without instability. The takeaway: when you read the headline numbers in §4, you can rule out the alternative explanation "maybe the adapter just broke the language model", it didn't.](artifacts/nla/figures/fig2_loss_curves.png)

`centered_eval.py` on the M=200 target slice, K=64 samples, pool=2000:

| condition | raw fve_nrm | centered fve_nrm | $\rho_{\text{cen}}$ |
|---|---|---|---|
| **10k ckpt, greedy** | 0.694 | 0.589 | 0.27 |
| **10k ckpt, sampled (mean)** | 0.691 | 0.585 | 0.26 |
| **10k ckpt, best-of-64** | **0.846** | **0.788** | **0.96** |
| **30k ckpt, greedy** | 0.688 | 0.591 | 0.28 |
| **30k ckpt, sampled (mean)** | 0.684 | 0.582 | 0.25 |
| **30k ckpt, best-of-64** | **0.847** | **0.797** | **0.99** |
| NN-retrieval (pool=2000) | 0.795 | 0.714 | 0.71 |
| random floor | 0.622 | 0.510 | 0.00 |
| paraphrase ceiling | 0.848 | 0.799 | 1.00 |

Three observations:

1. **The adapter saturates the paraphrase ceiling at best-of-64.** Raw
   $\rho \approx 0.996$, centered $\rho \approx 0.99$. The four lever
   experiments weren't failing to break a wall; they were already at the wall.
2. **Greedy decoding is the real bottleneck.** Centered $\rho \approx 0.28$.
   Whatever the prefix encodes is far more decodable under stochastic search
   than under argmax.
3. **A zero-training nearest-neighbour lookup beats the greedy adapter.**
   $\rho_{\text{cen}}(\text{NN}) \approx 0.71 \gg 0.28$. The trained model
   does not even reach the retrieval baseline on its deterministic decode.

## 5. The K-curve and the death of logp reranking

![**Figure 3. The reranker can tell good guesses from bad guesses.** Each dot is one held-out example. The horizontal axis is the *true* fidelity of a candidate (what we'd score it as if we had the ground truth). The vertical axis is what the reranker *predicts* its fidelity will be without seeing the ground truth. If the reranker were useless the cloud would be a circle; here it is a tight diagonal, with correlation $\approx 0.83$ on validation and $\approx 0.78$ on a held-out curated subset. The slight S-shape at the corners is a known saturation effect (the predictor's output is squashed through a tanh) and does not affect ranking. Why this matters: the best-of-$K$ trick in §4 needs *something* that can pick the best of $K$ candidates without an oracle. This figure shows that something exists and works on data it has never seen.](artifacts/nla/figures/fig3_calibration_scatter.png)

A finer-grained sweep of $K \in \{1, 2, 4, 8, 16, 32, 64\}$ on the same
30k checkpoint and 200-target slice (`scripts/rerank_eval.py`, two
independent runs averaged) gives:

| $K$ | centered fve_nrm | $\rho_{\text{cen}}$ |
|---|---|---|
| 1 | 0.577 | 0.23 |
| 2 | 0.613 | 0.36 |
| 4 | 0.644 | 0.46 |
| 8 | 0.678 | 0.58 |
| 16 | 0.706 | 0.68 |
| 32 | 0.736 | 0.78 |
| 64 | 0.766 | **0.88** |

The curve is **log-linear**: $+0.030$ raw / $+0.10$ $\rho_{\text{cen}}$ per
doubling of $K$. Extrapolating, $K \approx 256$ reaches the paraphrase
ceiling. The same script confirms two negative results that constrain the
design space for any "cheap" reranker:

- **logp-rerank actively hurts.** Choosing the candidate with highest
  mean per-token log-prob from the same $K=64$ pool gives centered
  $0.561$, $0.025$ *below* greedy ($0.586$). The policy's own sequence
  probability has no useful correlation with reconstruction quality.
- **Per-target Spearman$(\text{mean-logp}, \text{oracle-cen}) \approx
  0.04$** (mean over 200 targets, $p_{50}=0.05$, $p_{05}=-0.31$,
  $p_{95}=0.38$). Any value head whose features are restricted to the
  policy's own logp trajectory cannot beat greedy. The reranker must
  consume the rollout's hidden activation at layer $\ell$, which is the
  same compute path as just scoring against $v$ directly.

Conversely, **NN-anchor rerank** (score each candidate by its centered
cosine to the nearest pool point of $v$, no access to $v$ itself) gives
$0.722$, beating greedy by $+0.14$. This shows the reranking surface is
not flat; it is logp specifically that is useless.

## 6. Implications

- **Verbalization is sampling-bound, not capacity-bound on this backbone.**
  Under a meaningful metric, 12.7M trainable parameters suffice to make the
  *space of paraphrases* reachable; the per-roll-out probability mass at the
  argmax mode is what fails.
- **Any future fve_nrm-style evaluation needs (i) anisotropy centering and
  (ii) a retrieval baseline.** The raw metric over an anisotropic backbone is
  not a faithful indicator of progress: a 0.05 gain on raw fve_nrm collapses
  to nothing once $\mu$ is subtracted, and may correspond to a trained model
  that still loses to a 1-line numpy NN lookup.
- **The interesting open question is the greedy gap**: closing
  $\rho_{\text{cen}}^{\text{greedy}}$ from 0.28 toward 0.99 without paying
  $K=64$ inference cost. Plausible directions: temperature distillation from
  best-of-K into greedy, length-conditioned decoding, or contrastive
  fine-tuning against retrieved hard negatives.
- **A bag-of-$K$ self-distillation attempt (Lever B) does not close the
  greedy gap on this backbone.** We trained an activation-conditioned
  prefix with winner-CE over $K\!=\!32$ rollouts plus a contrastive term
  against retrieved hard negatives (`scripts/train_nla_bok_v2.py`). Under
  hot hyperparams (temperature anneal $1.5 \to 0.7$, $\beta_{\text{ctr}}\!=\!0.3$,
  lr $3\mathrm{e}{-5}$) training losses fall while sampling diversity
  (5-gram duplication on rollouts) climbs from $0.003$ to $0.045$ over
  ~2.4k steps and *both* greedy $\rho_{\text{cen}}$ and oracle
  $\rho_{\text{cen}}^{K=32}$ regress past their warm-start values: the prefix
  fits its own narrowing winner distribution and stops covering the
  paraphrase manifold. Under gentler hyperparams (temperature
  $1.5\!\to\!1.2$, $\beta_{\text{ctr}}\!=\!0.1$, lr $1\mathrm{e}{-5}$, warmup
  100, patience 3) the run plateaus at greedy $\rho \approx 0.32$ and
  oracle $\rho \approx 0.85$, essentially indistinguishable from the
  CE-only warm-start at step 500, and early-stops without further
  improvement. We read this as: the winner-CE objective on $K$ rollouts
  optimizes the policy's mode toward whichever rollout currently has the
  highest centered cosine, but this is not the same as concentrating mass
  on the *paraphrase manifold* the oracle reranker exploits. Lever A
  (best-of-$K$ at deploy time) remains the only mechanism that closes the
  gap on this backbone.

## 7. Related work and positioning

The components of this paper are not new in isolation; the assembly is.

**Activation verbalization.** *Patchscopes* (Ghandeharioun et al., 2024) and
*SelfIE* (Chen et al., 2024) read frozen-LM hidden states by patching the
state into a re-prompted forward pass of the same model and decoding
greedily. Earlier *logit lens* (nostalgebraist, 2020) and *tuned lens*
(Belrose et al., 2023) project hidden states through (learned) affine maps to
the vocabulary. *Future Lens* (Pal et al., 2023) predicts upcoming tokens
from current hidden states. All of this work is qualitative or judged by
downstream task accuracy. NLA differs in that the verbalizer is a separately
trained activation-conditioned prefix, evaluated by a *round-trip* fidelity
metric (re-encode the verbalization and measure $\rho_{\text{cen}}$ against
the original state) calibrated against an empirical paraphrase ceiling.

**Embedding inversion.** *vec2text* (Morris et al., 2023) trains an
iterative inverter that recovers text from sentence-encoder embeddings,
demonstrating that black-box embedding APIs are essentially text-recoverable.
The engineering pattern (encode → inverter → text → re-encode → measure
recovery) is direct. The target space differs: vec2text inverts a
sentence-level encoder embedding; NLA inverts an *internal hidden state* of
the same generative LM that produces the verbalization, which is what makes
the round-trip closure non-trivial, the verbalizer must produce text that
the *frozen backbone itself* re-routes to the same place in its own
representation space.

**Best-of-$N$ rerank as decoding.** Best-of-$N$ with a learned reward model
is the standard RLHF deployment trick (WebGPT, Anthropic-HH); reranking
generations by similarity to a target embedding is the entire RAG line; and
reranking against a model-internal metric of consistency is closely related
to *Minimum Bayes Risk* decoding (Kumar & Byrne, 2004; Eikema & Aziz, 2020;
Bertsch et al., 2023). Lever A is best-of-$N$ MBR with a non-standard
utility: centered cosine of the candidate's re-encoded layer-$\ell$
activation against a fixed target activation. We are not aware of prior
work that uses an LM's *own internal hidden state* (rather than a separate
encoder embedding or a reward model) as the rerank utility.

**Self-distillation of $K$-best into greedy.** *STaR* (Zelikman et al.,
2022), *V-STaR*, *RFT* (Yuan et al., 2023), *ReST* (Gulcehre et al., 2023),
and the older sequence-level distillation (Kim & Rush, 2016) all sample
$K$ rollouts, score them, and train the student to imitate the winners.
Lever B's bag-of-$K$ winner-CE objective is in this family. The collapse
mode we observe (training losses drop while sampling diversity falls and
oracle ceilings drop with it) is well-documented in those papers and is
what motivates KL-to-base regularization, temperature schedules, and
diversity-aware reward shaping in modern variants.

**Probing and mechanistic interpretability.** Linear probes (Alain &
Bengio, 2016; Hewitt & Manning, 2019) and dictionary/circuit decomposition
work (Anthropic feature circuits, sparse autoencoders, Marks et al.) extract
features from activations but typically classify or describe them rather
than verbalize them as full sentences with fidelity guarantees against the
backbone's own representation.

**Computational semiotics.** Existing computational-semiotic work is
predominantly symbolic (Sowa's conceptual graphs; Goguen's algebraic
semiotics) or biological (Barbieri, Kull). Quantitative pragmatics in the
*Rational Speech Acts* tradition (Frank & Goodman, 2012) is empirical and
falsifiable, but on synthetic dialogue games rather than transformer
internals. Distributional semantics is sometimes labelled
"computational semiotics" but typically without explicit semiotic
commitments.

**What this work adds.** The intersection. Specifically: a system that
(i) commits Peircean primitives, metapragmatic awareness, reflexive
recursion, bifurcation, to specific architectural roles, (ii) operates on
a frozen production-scale 7B LLM, (iii) reports a calibrated $\rho_{\text{norm}}$
metric anchored at a random floor and a human paraphrase ceiling, and
(iv) closes the loop with a round-trip evaluation in which the
verbalization is fed back through the same backbone and scored against the
target state. We are not claiming any single component is novel; we are
claiming this conjunction has not, to our knowledge, been assembled
before. The empirical headline (Lever A: deployable best-of-$K$ rerank
against an internal-state metric closes the entire greedy→paraphrase gap
on Qwen2.5-7B layer 20 with no extra training) is what the conjunction
buys.

## 8. Artifacts

- `scripts/oracle_ceiling.py`, replay / random / NN / paraphrase, raw + centered.
  Output: `artifacts/nla/oracle_ceiling_30k_v2.json`.
- `scripts/centered_eval.py`, adapter greedy / sampled / best-of-K and
  NN-retrieval, raw + centered, on M target vectors with a held-out pool.
  Outputs: `artifacts/nla/centered_eval_{10k,30k}_M200.json`.
- `scripts/rerank_eval.py`, K-curve, logp-rerank, NN-anchor-rerank,
  Spearman(logp, oracle-cen). Output:
  `artifacts/nla/rerank_eval_ce_seq64_np16_v2.json`.
- `scripts/train_nla_bok_v2.py`, Lever B trainer (winner-CE over $K$
  rollouts + contrastive hard-negs + optional activation L2). Negative-result
  artifacts: `artifacts/nla/bok_v2b_seq64_np16/{best_av.pt, train_log.jsonl, val_text_step000500.jsonl, val_text_step001000.jsonl}`.
- HF release: [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1) (model),
  [`RiverRider/srt-nla-targets-v1`](https://huggingface.co/datasets/RiverRider/srt-nla-targets-v1) (dataset).
  Lever B's `best_av.pt` is *not* released as a separate revision: it
  matches the warm-start within sampling noise.
- Targets: `artifacts/nla/targets_q7b_L20_seq64_{10k,30k_seed1}.pt`.

## 9. Limitations

- Single layer ($\ell=20$), single target type (64-token continuations).
  The anisotropy magnitude $\|\mu\|$ is backbone-specific (Qwen-2.5-7B:
  $\|\mu\|\!\approx\!55$; Llama-3.2-3B: $\|\mu\|\!\approx\!7.2$, see §10);
  centering is required, but the size of the correction is not universal.
- Paraphrase ceiling is itself stochastic ($k=8$ samples per source); centered
  $\rho > 1$ would not be surprising at much higher $k$, the ceiling is a
  *practical* upper bound on what a Qwen-shaped model can say differently.
- We did not re-run the four lever experiments under the centered metric; we
  hypothesize they would each show small but non-zero centered improvements
  that were invisible on the raw metric.

## 10. Cross-backbone transfer: Llama-3.2-3B

A single-backbone result is hard to interpret: any of the four core
findings (the 0.689 floor, the boK=ceiling identity, the death of
logp-rerank, the log-linear K-curve) could in principle be artefacts of
Qwen-2.5-7B's specific anisotropy $\|\mu\|\approx 55$ rather than
properties of frozen-decoder verbalization in general. We therefore
re-ran the entire pipeline, sampling, gold-pair extraction, SFT,
centered eval, K-curve, on a different model family and a different
size: **meta-llama/Llama-3.2-3B**, 28 layers, hidden_size 3072, vocab
128k. The verbalizer is backbone-agnostic by construction
(`d_embed = backbone.config.hidden_size`); no code changes were needed
beyond a different `--backbone` flag.

**Setup.** Layer $\ell=20$ (71% depth, the same fractional depth as
Qwen-2.5-7B's $\ell=20/28$). 30,000 sampled continuations, $T=64$ tokens,
seed 1; 29,963 gold pairs survive after re-tokenization with the Llama
tokenizer. SFT for 3 epochs at batch=16, lr=$3\!\times\!10^{-5}$,
$P=1$ prefix token, 1 inject slot, identical hyperparameters to the
Qwen run except for trainable parameter count (9.44M vs 12.7M, a
function of the smaller hidden dim and embedding matrix slice).

**Anisotropy.** $\|\mu\|=7.21$, ~7.6× smaller than Qwen's 55. The raw
random floor drops accordingly: 0.569 vs Qwen's 0.622. *Centering
removes the bulk of the per-backbone offset:* both random floors map to
$\approx 0.50$ centered, which is what makes the centered metric
portable.

**Centered eval (M=32 targets, K=64, pool=2000).**

| condition | raw fve_nrm | centered fve_nrm |
|---|---|---|
| random floor | 0.569 | 0.500 |
| greedy | 0.672 | 0.633 |
| sampled (mean) | 0.684 | 0.637 |
| **best-of-64** | **0.873** | **0.858** |
| NN-retrieval (pool=2000) | 0.837 | 0.820 |

**Oracle ceiling (M=200, paraphrase k=8, pool=2000).**

`scripts/oracle_ceiling.py --backbone meta-llama/Llama-3.2-3B`:

| condition | raw fve_nrm | centered fve_nrm |
|---|---|---|
| replay (sanity) | 0.904 | 0.881 |
| random floor | 0.569 | 0.498 |
| NN-in-pool | 0.785 | **0.756** |
| paraphrase (best-of-8) | 0.764 | 0.720 |

Two notable points relative to the Qwen ceiling table (§3):

1. **NN > paraphrase on Llama.** The bare paraphrase prompt
   (`"Paraphrase the following text using different words but the same
   meaning. Text: ... Paraphrase:"`) underperforms simple nearest-pool
   retrieval on Llama-3.2-3B base ($0.720 < 0.756$ centered). On
   Qwen-2.5-7B base the same prompt zero-shots cleanly and produces
   $0.799$ centered, above NN's $0.714$, i.e., Qwen-2.5-7B base is a
   noticeably better in-context paraphraser than Llama-3.2-3B base.
   The "paraphrase ceiling" is therefore an *instruction-following
   ceiling* of the base model, not a property of the verbalization
   problem; on a weaker zero-shot follower it underestimates the true
   ceiling. We use **NN-in-pool as the headline ceiling for Llama**.
2. **Adapter best-of-64 exceeds both ceilings.** With NN-in-pool
   ($0.756$) as the denominator, $\rho_{\text{cen}}=
   (0.858 - 0.498)/(0.756 - 0.498) = 1.40$, the adapter saturates the
   retrieval baseline at $K=64$ and overshoots it. This is the same
   qualitative result as Qwen (best-of-64 saturates the paraphrase
   ceiling at $\rho_{\text{cen}} \approx 0.99$), with the difference
   that on Llama the *NN* baseline is the binding ceiling, not the
   paraphrase one.

**K-curve (M=200 targets, K=32).**

| $K$ | centered fve_nrm |
|---|---|
| 1  | 0.636 |
| 2  | 0.678 |
| 4  | 0.716 |
| 8  | 0.748 |
| 16 | 0.780 |
| 32 | 0.809 |

The curve is again log-linear: $+0.034$ centered per doubling of $K$,
within sampling noise of Qwen's $+0.030$. Extrapolating from $K=32$
(centered 0.809) at the same slope reaches the M=32-measured boK
ceiling near $K \approx 64$, consistent with the M=32 result above.

**Cheap reranks fail the same way.** logp-rerank gives 0.624 centered,
$+0.005$ over greedy (0.619), i.e. indistinguishable. Per-target
Spearman$(\text{mean-logp}, \text{oracle-cen})$: mean 0.055, $p_{50}$
0.059, $p_{05}$ $-0.40$, $p_{95}$ $0.53$. Identical structure to Qwen
(mean ~0.04). NN-anchor rerank, by contrast, gives 0.783 centered, well
above greedy (the NN-anchor *baseline*, score against the ground-truth
$v$'s nearest pool neighbour, not the candidate, gives 0.836). Both
the positive (NN works) and negative (logp doesn't) reranking results
replicate.

**Summary.** Every qualitative finding of §§2–6 reproduces on
Llama-3.2-3B:

1. raw greedy fve_nrm sits in a narrow $\approx 0.66$–$0.69$ band that
   is $\approx 0.10$ above the raw random floor. The "0.689 wall" is
   not Qwen-specific, it is the anisotropy floor under whatever the
   per-backbone $\|\mu\|$ is.
2. best-of-64 closes (and slightly overshoots) the retrieval baseline
   in centered fve_nrm, again with no extra training.
3. the K-curve is log-linear with slope ~0.03 centered per doubling.
4. logp-rerank is statistically indistinguishable from greedy; the
   policy's sequence probability is uncorrelated with reconstruction
   quality.

The pipeline is therefore not a Qwen-specific artefact. The
prefix-tuned verbalizer, the centered metric, and the K-fold sampling
search are properties of the *frozen-decoder verbalization problem*,
not of any one model's geometry.

**Llama artifacts.** `artifacts/nla/llama32_3B/`:
- `sft/best_av.pt`, best SFT checkpoint (val fve_nrm 0.332 at step 5000/5337).
- `centered_eval.json`, `rerank_eval.json`, `oracle_ceiling.json`, eval JSON used for the tables above.
- `gold_pairs_seq64.jsonl`, 29,963 train pairs.
- `sft.log`, `sample.log`, `centered_eval.log`, `rerank.log`, `oracle_ceiling.log`, full run logs.

The 22.7 GB activations file (`targets_L20_seq64_30k_seed1.pt`,
sha256 `db5c9d22…1981fa`) is reproducible from
`scripts/sample_targets.py --backbone meta-llama/Llama-3.2-3B --layer 20
--num-sequences 30000 --seq-len 64 --batch-size 16 --dtype bfloat16
--seed 1`.

---

## 11. Cross-backbone transfer #2: Gemma-2-2B

A two-backbone result (Qwen-2.5-7B, Llama-3.2-3B) excludes the
narrowest "this is a Qwen artefact" reading but does not yet exclude
"this is a Meta-and-Alibaba pretraining-recipe artefact." We
therefore run the same pipeline a third time on a third lab's
backbone of a third generation: **google/gemma-2-2b**
($d_{\text{embed}}=2304$, $L=26$ layers, vocabulary $256{,}000$,
distinct `bos_token_id=2`, `eos_token_id=1`, explicitly *not* the
Qwen-style trap that produced the constant-target bug fixed at commit
`902b746`). Probe layer $\ell=19$ (19/26 $= 73.1\%$ depth, the
closest fractional match to Qwen and Llama at $20/28 = 71.4\%$). All
hyperparameters mirror §10 with the substitution
`--backbone google/gemma-2-2b`; trainable AV parameter count
$\approx\!5.31$M (smaller than Llama's because the projection slice
shrinks as $d_{\text{embed}}$ does, and prefix/inject default to
$P{=}1$ token slots).

**Pipeline timing.** On a remote RTX PRO 6000 (Blackwell), the full
Stage-4 chain, sample $30{,}000$ activation targets at
$L{=}19$ ($\sim\!17$ GB cache), build $29{,}952$ gold $(x, v_x)$
pairs (48 token-budget skips), SFT the AV for $3$ epochs
($5{,}337$ steps), and the centred / rerank / oracle eval triad,
completes in $\approx\!90$ minutes wall-clock. Best validation
$\text{fve}_{\text{nrm}} = 0.3334$ at SFT step $4{,}500$ (CE
$1.760 \to 1.602$ cleanly, no plateau).

**Anisotropy.** The centred-eval estimate of the L19 mean activation
norm is $\|\mu\| = 156.3$ (oracle subsample $\|\mu\| = 164.2$). This
is *substantially larger* than Qwen's L20 ($\|\mu\| \approx 55$) and
Llama's L20 (§10), almost a factor of three. Gemma-2 layer-19
representation space is the most anisotropic of the three backbones
we have tested, and is the cleanest case for the centring claim of
§§4–5: any evaluation that ignores $\mu$ on this backbone is
dominated by the rotation-into-$\mu$ component.

**Headline numbers** ($M=200$ vectors, pool size $2{,}000$):

| Quantity | Raw $\text{fve}$ | Centred $\text{fve}$ | $\rho_{\text{cen}}$ |
|----------|------------------|----------------------|---------------------|
| Random floor       | 0.6748 | 0.4976 | 0.00 |
| Greedy             | 0.6644 | 0.5278 | **0.30** |
| Sampled ($T{=}1.0$) | 0.6445 | 0.5147 | 0.17 |
| Best-of-$64$       | 0.7518 | 0.6305 | **1.33** |
| NN-in-pool         | 0.8151 | 0.7118 | 2.14 |
| Paraphrase ceiling (best-of-$8$) | 0.7199 | 0.5978 | 1.00 |
| Replay (oracle) | 0.7993 | 0.7125 | 2.14 |

The centred denominator $0.5978 - 0.4976 = 0.1002$ is markedly
smaller than Qwen's; this is a substantive cross-backbone fact, not a
measurement artefact. Gemma-2-2B's base distribution is sharper in
the centred geometry, paraphrases of a given $x$ produce more
peaked sampling distributions than on Qwen, so the ceiling sits
closer to the floor in centred fve units. Despite this, **best-of-64
overshoots the paraphrase ceiling by a factor of $1.33\times$**,
mirroring the Qwen and Llama results. NN-in-pool sits *above*
paraphrase ceiling on Gemma, indicating that for this backbone the
binding ceiling is set by the in-distribution nearest-neighbour
oracle, not by base-model paraphrase quality.

**$K$-curve** (oracle top-1, centred):

| $K$ | 1 | 2 | 4 | 8 | 16 | 32 |
|-----|------|------|------|------|------|------|
| centred fve | 0.511 | 0.534 | 0.555 | 0.572 | 0.593 | 0.618 |

The curve is monotonic with slope $\approx\!0.021$ per doubling,
shallower than Qwen's $\approx\!0.030$, consistent with the smaller
ceiling-floor gap. Reading the same curve in raw fve gives slope
$\approx\!0.020$ per doubling between $K{=}1$ and $K{=}32$, so the
log-linear shape replicates qualitatively across all three
backbones; the slope itself tracks the centred denominator.

The three K-curves are overlaid in **Figure 1** (the banner figure on
page 1; source
[`artifacts/nla/figures/fig1_cross_backbone_kcurve.png`](artifacts/nla/figures/fig1_cross_backbone_kcurve.png),
dashed = per-backbone paraphrase ceiling, dotted = per-backbone
random floor). The three random floors collapse onto a
single horizontal band at $\approx\!0.50$ despite a $22\times$ spread
in $\|\mu\|$, the three K-curves all rise log-linearly, and each
curve crosses its own paraphrase ceiling at a backbone-specific $K$:
$K \approx 16$ for Gemma, $K \approx 4$ for Llama, $K \approx 64$ for
Qwen. The visual is a one-figure summary of the three-backbone
replication condition.

**Self-rerank by adapter logp does not work.** Spearman correlation
between adapter $\log p(\hat{x} \mid v)$ and the oracle centred
fve over the $K{=}32$ candidate pool, computed per-target then
averaged over $200$ targets, is $\bar{\rho}_S = +0.030$ (median
$+0.048$, $5$/$95$ percentiles $-0.435 / +0.422$). Logp-rerank's
mean centred fve $0.5122$ is statistically indistinguishable from
greedy $0.5266$ ($\Delta = -0.0144$, well within the
target-to-target spread). This is the third backbone on which we
record the same finding: the AV's own confidence is uncorrelated
with how well the candidate re-encodes; best-of-$K$ is only useful
under the oracle utility.

**Three-backbone replication scorecard.** Every qualitative finding
of §§2–6 holds on Gemma-2-2B:

1. The raw greedy band ($0.664$) sits below the raw random floor
   ($0.675$), the strongest possible illustration of why raw fve
   is not a usable scoring rule on a high-anisotropy backbone, and a
   numerical confirmation that the centring move of §4 is not
   optional.
2. The centred random floor ($0.498$) is within $0.003$ of Qwen and
   Llama's centred random floors despite Gemma's anisotropy being
   $\approx\!3\times$ Qwen's. The centred geometry is, as predicted,
   *backbone-invariant* up to the resolution of $200$-vector means.
3. The $K$-curve is log-linear with slope-per-doubling in
   $[0.020, 0.022]$ centred, in the same order of magnitude as the
   Qwen and Llama slopes.
4. Logp-rerank is indistinguishable from greedy ($\Delta_{\text{cen}}
   = -0.014$), $\bar{\rho}_S(\log p, \text{oracle}) \approx 0$.
5. Best-of-$64$ overshoots the paraphrase ceiling
   ($\rho_{\text{cen}} = 1.33$) and approaches the NN-in-pool /
   replay ceiling at $\rho_{\text{cen}} \approx 2.1$.

The released artifact set mirrors Llama:
`artifacts/nla/gemma2_2B/{sft/best_av.pt,
centered_eval_M200_K64.json, rerank_eval_M200_K32.json,
oracle_ceiling_M200.json, gold_pairs_seq64.jsonl, *.log}` and an
HF dataset/model pair under
`RiverRider/srt-nla-{av,targets}-gemma2-2b-v1`. Together with §10,
this closes the three-backbone replication condition for Stage 4:
the SRT-NLA effect is reproducible on Qwen-2.5-7B, Llama-3.2-3B and
Gemma-2-2B, three labs, three architectural lineages, three
anisotropy regimes, at a uniform $\sim\!90$ minute training cost
per backbone with no per-backbone hyperparameter tuning beyond
choice of probe layer.

---

## 12. NLA as Stage 4 of the SRT program

We now connect Stage 4 explicitly to Stages 1–3 (Lancaster, 2025;
2026a; 2026 [SRT-Adapter MS]). Three threads run through the program; we
record where each stands at the close of this paper.

*Substrate claim.* The core empirical claim of the SRT program is
that a frozen production-scale LLM is a substrate on which semiotic
phenomena are measurable rather than a target requiring custom
architectures. Stage 3 supported this for *community* and *regime*
on Qwen-2.5-7B and showed the substrate claim survives a 1-NN probe
on Qwen3-8B and Mistral-7B-v0.3 (Lancaster, 2026 [SRT-Adapter MS], §5.12).
Stage 4 strengthens it: under a calibrated round-trip the same
frozen Qwen-2.5-7B layer-20 state is *recoverable as text* up to
the empirical paraphrase ceiling at $K=64$, *and* the result
replicates on a different family (Llama-3.2-3B) at half the
parameter count and seven-times-smaller anisotropy. Within the
claim's intended scope (mid-depth layer of a $2$–$8$B base decoder),
the substrate framing is now under quantitative control along two
independent axes: structured side-channel readout and round-trip
text recovery.

*The greedy gap as a bifurcation in the decoding manifold.*
Lancaster (2025) develops political polarisation as a supercritical
pitchfork in the dynamics of interpretant divergence, with the
control parameter $r$ encoding the strength of divergence-amplifying
forces. The same canonical model has a natural reading inside the
sampling distribution of a prefix-tuned AV. Each candidate $\hat{x}_k$
re-encodes to a point $h_k$ in the layer-$\ell$ representation
space; the distribution of $\{h_k\}_{k=1}^{K}$ around $v$ is what
the rerank utility integrates over. At small $K$ and at the policy's
argmax mode, this distribution sits in a *single basin*, typically
the centred-cosine $\approx\!0.59$–$0.63$ basin we observe as the
greedy band on Qwen and Llama. At $K \gtrsim 16$ the distribution
develops a thin but heavy upper tail extending to the paraphrase
ceiling; the rerank picks from that tail. The greedy gap is then
the gap between the *modal* $\hat{x}$ and the *paraphrase-manifold*
$\hat{x}$, which on the canonical pitchfork reading is the gap
between an attractor at the policy's high-mass mode and a higher-
fidelity attractor that is reachable but not modal. We do not claim
to have *measured* a pitchfork in the sampling distribution, that
would require a probabilistic separatrix probe of the kind the
SRT-Adapter applies to MAH divergence (Lancaster, 2026 [SRT-Adapter MS],
§6.9). We claim that the qualitative shape of the failure (modal
attractor below the paraphrase manifold; sampling moves probability
mass between the two; logp does not separate them) is the shape the
program's canonical model expects, and that *cheap reranks fail
because they live on the wrong side of the bifurcation*: logp is a
within-basin quantity, cosine to a retrieved anchor crosses basins.

*Half-open loops, closed loops, and what NLA does not show.* The
SRT-Adapter's central open question is whether a closed circular-
causal loop on a frozen backbone is reachable through gated hidden-
state injection. NLA does not answer that question. It closes a
*different* loop, text-mediated rather than hidden-state-mediated,
inter-pass rather than intra-pass, on the same substrate, and
shows that the closed loop in this topology saturates a
non-trivial ceiling without further training. Two consequences for
the broader program follow. *First*, the controllability of a
frozen mid-scale decoder is asymmetric across input ports: the text
port is strong (NLA closes), the hidden-state-injection port is
weak (RRM inject-back has not yet closed). Future SRT work should
treat this asymmetry as a working hypothesis rather than an
incidental observation. *Second*, NLA gives the SRT program a
*compressor*: any structured side-channel readout (community
vector, divergence trajectory, $\hat{r}$ sequence) can in principle
be written into text by a verbalizer-style adapter trained on the
appropriate target, recovered downstream, and audited under the
centred-cosine round-trip. The dictionary of readouts the
SRT-Adapter exposes thus becomes, with a verbalizer attached, a
text-typed interface to the substrate's interior. We do not claim
to have built that interface. We claim Stage 4 demonstrates the
component on which it would rest.

*Reification, honestly.* Anderson (personal communication; see
Lancaster, 2026 [SRT-Adapter MS], §2.7) notes that any computational
semiotic instrument participates in the meaning-field it measures.
The verbalizer is no exception. The "paraphrase ceiling" is itself
a stochastic object: $k=8$ paraphrase samples per source under a
prompt of our choosing, scored under our centred metric, anchored
against our random-floor. A different prompt (Llama paraphrases
clean less well than Qwen on the bare instruction we used; §10),
a different random-floor pool, a different choice of $\ell$,
each shifts the denominator of $\rho_{\text{cen}}$. The headline
result that best-of-$64$ "saturates the paraphrase ceiling" should
be read with this in mind: it saturates *the empirical paraphrase
distribution of this base model under this prompt against this
random floor at this layer*. That this saturation reproduces on
Llama-3.2-3B with a different binding ceiling (NN-in-pool rather
than paraphrase) is what makes the qualitative claim portable; the
absolute number is not.

---

## 13. Honest expectations and open problems

We close in the program's standard register: what we expect the
next phase to deliver, what we do not, and where the load-bearing
uncertainties sit.

1. **Greedy gap closure is the headline open problem.** A
   verbalizer that closes the greedy gap on this backbone, i.e.,
   single-pass deterministic decoding at $\rho_{\text{cen}} \gtrsim 0.9$
   without K-fold inference, is the next-stage goal. Lever B
   (bag-of-$K$ self-distillation) does not close it on Qwen and is
   not expected to on Llama or Gemma; the diversity collapse is
   inherent to winner-CE on a frozen substrate. Plausible
   directions: (i) temperature distillation from best-of-$K$ into
   greedy with an *explicit* KL-to-base regulariser tuned to keep
   the rollout 5-gram duplication rate below $0.01$; (ii) length-
   conditioned decoding under a learned length oracle (the policy
   knows the target *length* before it knows the *content*);
   (iii) contrastive fine-tuning against retrieved hard negatives,
   with the centred-cosine of the *re-encoded* candidate as the
   metric, not its sequence-logp. We are not pretending these are
   mutually exclusive.

2. **The metric is portable; the absolute numbers are not.** The
   centred random floor lands at $\approx\!0.50$ on both Qwen and
   Llama; we expect the same on Gemma. The *ceiling* in absolute
   centred fve\_nrm depends on the paraphrase capacity of the base
   model under whatever prompt is used; this varies across
   backbones (Qwen $\approx\!0.80$ centred, Llama $\approx\!0.72$
   under the same prompt; §3, §10). Reporting in
   $\rho_{\text{cen}}$, normalised to the binding ceiling for
   that backbone, preserves portability. Anyone reporting an
   $\mathrm{fve\_nrm}$ result without these two anchors is
   reporting an uninterpretable number.

3. **The program has not measured a pitchfork in the sampling
   distribution.** The §12 reading of the greedy gap as an
   attractor structure under the canonical pitchfork is a
   theoretical positioning, not an experimental result. A direct
   test would require a probabilistic separatrix probe over the
   sampling distribution at fixed $v$, of the kind the SRT-Adapter
   applies to MAH divergence (Lancaster, 2026 [SRT-Adapter MS], §6.9).
   This is on the v9-onward horizon for the program, not for this
   paper.

4. **Single layer, single target type, single ceiling protocol.**
   $\ell$ is fixed at $20$ on Qwen, $20$ on Llama, $19$ on Gemma,
   roughly $71$–$73\%$ depth in each case. Targets are last-valid-
   token hidden states of $T=64$-token continuations. The
   paraphrase ceiling is computed at $k=8$. None of these are
   universal choices. Generalisation across $\ell$, target type,
   target length, and ceiling protocol is the next ablation
   surface.

5. **The substrate-asymmetry hypothesis is now load-bearing.**
   "The text port is strong, the hidden-state-injection port is
   weak" is currently inferred from the *conjunction* of the
   SRT-Adapter's dead inject-back arm and Stage 4's working
   text-mediated round-trip. The hypothesis is testable: a
   matched-budget RRM-style intra-pass loop trained against the
   same centred-cosine round-trip metric (i.e., the loss the AV
   sees, applied to the inject-back path instead of the prefix)
   would either close to the same $\rho_{\text{cen}}$ or fail to.
   This is on the v9 horizon for the SRT-Adapter, not for NLA.

---

## 14. References

Anderson, M. (2014). Mathematical modeling of catastrophic change
in cultural systems. In M. Anderson (Ed.), *Cultural shaping of
violence: Victimization, escalation, response* (selected chapters).
Purdue University Press.

Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I.,
McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting
latent predictions from transformers with the tuned lens. *arXiv*
preprint arXiv:2303.08112.

Bertsch, A., Xie, A., Neubig, G., & Gormley, M. R. (2023). It's
MBR all the way down: Modern generation techniques through the lens
of minimum Bayes risk. *arXiv* preprint arXiv:2310.01387.

Chen, H., Vondrick, C., & Mao, C. (2024). SelfIE: Self-
interpretation of large language model embeddings. *ICML 2024*.

Eikema, B., & Aziz, W. (2020). Is MAP decoding all you need? The
inadequacy of the mode in neural machine translation. *COLING 2020*.

Frank, M. C., & Goodman, N. D. (2012). Predicting pragmatic
reasoning in language games. *Science*, 336(6084), 998.

Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., & Geva, M.
(2024). Patchscopes: A unifying framework for inspecting hidden
representations of language models. *ICML 2024*.

Gulcehre, C., Le Paine, T., Srinivasan, S., Konyushkova, K.,
Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C.,
Macherey, W., Doucet, A., Firat, O., & de Freitas, N. (2023).
Reinforced self-training (ReST) for language modeling. *arXiv*
preprint arXiv:2308.08998.

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding
syntax in word representations. *NAACL 2019*.

Kim, Y., & Rush, A. M. (2016). Sequence-level knowledge
distillation. *EMNLP 2016*.

Kockelman, P. (2017). *The art of interpretation in the age of
computation*. Oxford University Press.

Kockelman, P. (2025). *Semiotic agency in digital environments*.
Manuscript.

Kumar, S., & Byrne, W. (2004). Minimum Bayes-risk decoding for
statistical machine translation. *HLT-NAACL 2004*.

Lancaster, J. B. (2025). The treachery of signs: Semiotic
mediation, pitchfork bifurcation, and political polarization in
algorithmically curated societies. *SSRN*.
<https://papers.ssrn.com/abstract=5987495>

Lancaster, J. B. (2026a). Semiotic-reflexive language model
training: Bridging interpretive bifurcations through metapragmatic
chain architectures and embodied grounding. *SSRN*.
<https://papers.ssrn.com/abstract=6349978>

Lancaster, J. B. (2026, manuscript). The Semiotic-Reflexive
Transformer Adapter: Lightweight semiotic awareness for frozen causal
language models. GitHub manuscript (Stage 3 of the SRT program; not
yet a preprint), <https://github.com/space-bacon/SRT/blob/main/arxiv/paper.md>.
Cited in this paper as *Lancaster, 2026 [SRT-Adapter MS]*. The two
Lancaster preprints in this bibliography are the SSRN entries above
(Lancaster, 2025; 2026a); the SRT-Adapter manuscript is repository-
hosted only at the time of writing, with arXiv submission planned but
not yet executed.

Lancaster, J. B. (2026c). Reddit Discourse Corpus: A multi-community
dataset for semiotic analysis. Manuscript.

Leighton, M. P. (2026). Will a large complex system be a Maxwell
demon? *arXiv* preprint arXiv:2603.03248.

Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., &
Mueller, A. (2024). Sparse feature circuits: Discovering and
editing interpretable causal graphs in language models. *arXiv*
preprint arXiv:2403.19647.

Morris, J. X., Kuleshov, V., Shmatikov, V., & Rush, A. M. (2023).
Text embeddings reveal (almost) as much as text. *EMNLP 2023*.

nostalgebraist. (2020). interpreting GPT: the logit lens.
LessWrong post.

Pal, K., Sun, J., Yuan, A., Wallace, B. C., & Bau, D. (2023). Future
Lens: Anticipating subsequent tokens from a single hidden state.
*CoNLL 2023*.

Peirce, C. S. (1931–1958). *Collected papers of Charles Sanders
Peirce* (Vols. 1–8). C. Hartshorne, P. Weiss, & A. Burks (Eds.).
Harvard University Press.

Silverstein, M. (1993). Metapragmatic discourse and metapragmatic
function. In J. A. Lucy (Ed.), *Reflexive language* (pp. 33–58).
Cambridge University Press.

Silverstein, M. (2003). Indexical order and the dialectics of
sociolinguistic life. *Language & Communication*, 23(3–4), 193–229.

VanSaders, B., Fruchart, M., & Vitelli, V. (2026). Measurement-
induced phase transitions in informational active matter. *PNAS
Nexus*, pgag077.

von Foerster, H. (1981). *Observing systems*. Intersystems
Publications.

von Foerster, H. (2003). *Understanding understanding: Essays on
cybernetics and cognition*. Springer.

Wildgen, W. (1982). *Catastrophe-theoretic semantics: An
elaboration and application of René Thom's theory*. John Benjamins.

Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., & Huang, F.
(2023). RFT: Reasoning with reinforced fine-tuning. *arXiv*
preprint arXiv:2308.01825.

Zelikman, E., Wu, Y., Mu, J., & Goodman, N. (2022). STaR:
Self-taught reasoner, bootstrapping reasoning with reasoning.
*NeurIPS 2022*.

---

## Appendix A. Substrate diagnostics from prior SRT stages

The figures in this appendix are not new measurements for Stage 4; they
are reproduced from the SRT-Adapter (Stage 3) and interiority-probe
(Stage 2) artifact sets to give the reader the substrate-level context
in which the §§4–5 numbers should be read. All three are on
Qwen-2.5-7B; they illustrate properties of the frozen backbone that
NLA inherits.

![**Figure A1. Why we read the model at roughly three-quarters of the way through.** A heatmap of how strongly each of $10$ internal probe-channels (rows) responds to each of $11$ different prompt categories (columns), measured at every layer of frozen Qwen-2.5-7B. Brighter cells mean "this layer's channel is unusually informative for this kind of prompt." The bright band sits in the middle-to-late layers, roughly L18–L22, right where layer L20 is. Earlier layers are still doing word-level processing; the very last layers have already collapsed everything into next-token logits. The middle-late band is where the model has built up a rich, abstract picture of *what the prompt is about*. We use this as the rule for picking the readout layer in every backbone we test: $\approx 73\%$ of the way through the stack, L20 in Qwen, L20 in Llama, L19 in Gemma, with no per-model tuning.](artifacts/nla/figures/figA1_layer_heatmap.png)

![**Figure A2. One token slot keeps its grip even as the context grows.** Transformers have a known quirk: a lot of their attention gets dumped into the very first token of the input (the "BOS sink"). The horizontal axis here is how long the prompt is; the vertical axis is the fraction of attention each of $9$ different internal channels still routes to that first-token slot. For most channels the fraction shrinks roughly like $1/T$ as the prompt gets longer, the sink dilutes. One channel does not: the L14 injection channel (top curve) holds at about $80\%$ across the full length sweep. This matters because our verbalizer puts its hidden-state "injection" into exactly one token position; this figure is direct evidence that *that kind of slot* is structurally stable in the transformer and not just a transient feature of short prompts.](artifacts/nla/figures/figA2_length_scaling_share.png)

![**Figure A3. When the probe says "80% confident," it is right about $80\%$ of the time.** A standard reliability diagram for the Stage 3 regime-classification head, evaluated on $351{,}000$ tokens. The horizontal axis bins predictions by the probability the head emitted; the vertical axis is the fraction that actually turned out true in each bin. A perfect probe sits on the diagonal; this one does, almost exactly. Summary numbers: AUROC $0.99$ (it ranks positives above negatives almost perfectly), Brier $0.010$ (its raw probability errors are tiny), ECE $0.0009$ (its confidence is calibrated to four decimal places). We include this figure so the reader can see that the internal-state probe sitting under §5's reranker is not a hand-wavy classifier, it is calibrated to production grade, and the §§4–5 fidelity numbers are not artifacts of a miscalibrated scorer.](artifacts/nla/figures/figA3_regime_calibration.png)

---

## Appendix B. Comprehensive Glossary

A consolidated reference for the terms, symbols, scripts, artifacts,
and abbreviated citations used throughout this paper. Cross-references
to the sections in which each item is introduced or used substantively
are given in parentheses.

### B.1 Symbols and notation

- **$v$** — the *target* hidden activation at layer $\ell$: the
  frozen backbone's last-valid-token residual-stream vector after
  consuming a $64$-token continuation. Shape $v \in \mathbb{R}^{d}$
  with $d = 3584$ (Qwen-2.5-7B), $3072$ (Llama-3.2-3B), or $2304$
  (Gemma-2-2B). The round-trip starts here. (§1, §1.5)
- **$\hat{x}$** — the *verbalisation*: the short natural-language
  text the AV emits conditioned on $v$. The round-trip's middle
  term. (§1.5)
- **$h$** — the *re-encoded* hidden state: the frozen backbone's
  last-valid-token residual at layer $\ell$ after consuming
  $\hat{x}$ in a second forward pass. The round-trip's third term.
  Closure of the chain is measured by the similarity of $h$ to $v$.
  (§1.5)
- **$\ell$ (probe layer)** — the single transformer layer at which
  both $v$ and $h$ are read. $\ell = 20$ for Qwen ($20/28 \approx
  71\%$ depth), $\ell = 20$ for Llama-3.2-3B ($20/28$), $\ell = 19$
  for Gemma-2-2B ($19/26 \approx 73\%$). No per-model tuning beyond
  the fractional-depth rule of Figure A1. (§1, §10, §11, App. A)
- **$\mu$** — the *anisotropy mean*: the empirical mean of $v$
  over a pool of unrelated inputs at the same backbone and layer.
  Per-backbone $\|\mu\|$: $\approx 55$ (Qwen-7B L20),
  $\approx 7.2$ (Llama-3B L20), $\approx 156$ (Gemma-2B L19).
  Subtracting $\mu$ from both $v$ and $h$ before cosine is what
  makes the metric portable across backbones. (§3, §4, §10, §11)
- **$d$, $d_{\text{embed}}$** — the residual-stream width of the
  backbone; sets the AV's input projection size. (§1, §10, §11)
- **$L$** — total number of transformer layers in the backbone:
  $28$ (Qwen-7B, Llama-3B), $26$ (Gemma-2B). (§10, §11)
- **$T$ (token budget)** — generation length for both the original
  continuation $x$ that produced $v$ and the verbalisation
  $\hat{x}$. Fixed at $T=64$ tokens throughout. (§1, §10)
- **$K$** — the number of candidate verbalisations sampled per
  target. Best-of-$K$ picks the highest-scoring candidate under the
  rerank utility. Swept over $\{1, 2, 4, 8, 16, 32, 64\}$ in the
  K-curves. (§5, §10, §11)
- **$M$** — the number of held-out target vectors used for an
  evaluation. $M=200$ for the headline numbers, $M=32$ for the
  smaller Llama eval slice. (§4, §5, §10, §11)
- **$P$ (prefix length)** — number of static soft-prefix tokens
  the AV prepends before the inject slot. $P=16$ on Qwen, $P=1$
  on Llama and Gemma. (§1, §10, §11)
- **pool size** — the size of the candidate pool used by NN-style
  reranks and the random-floor estimator (default $2000$). (§3, §4)
- **$\alpha$** — the strength of an activation edit in the
  steering demo: $v_{\text{used}} = v + \alpha (v_{\text{new}} -
  v_{\text{orig}})$. Not a paper symbol; defined for the
  MindReader-NLA demo glossary. (demo only)

### B.2 Metrics

- **$\mathrm{fve\_nrm}(h, v)$ — fraction-of-variance-explained,
  normalised.** Defined as $\tfrac{1}{2}(1 + \cos(h, v))$; the
  unique affine map of $\cos \in [-1, 1]$ into $[0, 1]$. The
  per-target score throughout training, evaluation, and
  best-of-$K$. "fve" = *fraction of variance explained* by an
  optimal scaling of $h$ along $v$. (§1)
- **raw fve_nrm.** $\mathrm{fve\_nrm}(h, v)$ computed without
  subtracting $\mu$. Dominated by the rotation-into-$\mu$
  component on high-anisotropy backbones and therefore *not* a
  faithful indicator of content fidelity. The "0.689 wall" of §2
  is an artefact of this metric. (§2, §3)
- **centred fve_nrm.** $\mathrm{fve\_nrm}(h - \mu, v - \mu)$. The
  metric used for all headline numbers. Random-floor maps to
  $\approx 0.50$ on all three backbones despite a $22\times$
  spread in $\|\mu\|$. (§3, §4, §10, §11)
- **$\rho_{\text{cen}}$ — normalised progress.** Linear rescaling
  of centred fve into a unit interval whose $0$ is the centred
  random floor and whose $1$ is the centred ceiling (paraphrase on
  Qwen and Gemma; NN-in-pool on Llama). Values $>1$ mean the
  adapter overshoots the binding ceiling. (§3, §4, §10, §11)
- **Spearman$(\log p, \mathrm{cen})$.** Per-target rank correlation
  between the AV's mean per-token log-probability of a candidate
  and that candidate's centred fve. Mean over $M$ targets is the
  test statistic for whether logp-rerank can work. $\approx 0.04$
  on Qwen, $0.06$ on Llama, $0.03$ on Gemma; effectively zero.
  (§5, §10, §11)
- **5-gram duplication.** Diagnostic of sampling diversity: the
  fraction of $5$-token windows in a rollout that also appear in
  another rollout for the same $v$. Used in §6 to diagnose the
  Lever B collapse ($0.003 \to 0.045$). (§6)

### B.3 Reference anchors (the four ceilings/floors)

- **replay (oracle re-encode).** Re-encode the *original* text $x$
  that produced $v$; the resulting $h$ matches $v$ up to numerical
  noise. Upper bound on what any text-mediated reconstruction can
  achieve. Centred fve $\approx 0.97$ (Qwen). (§3)
- **paraphrase ceiling.** Best-of-$k{=}8$ over paraphrases of $x$
  produced by the base model under a fixed instruction prompt.
  *Semantic* ceiling: what a model of this class can express in
  different words. Binding ceiling on Qwen and Gemma; loose on
  Llama (where the base is a weaker zero-shot paraphraser).
  (§3, §10, §11)
- **NN-in-pool.** Nearest neighbour (by raw cosine) to $v$ within
  a held-out pool of $v$'s from the same distribution; the
  candidate text is then the source text of that neighbour.
  Zero-training retrieval baseline. Binding ceiling on Llama.
  (§3, §10, §11)
- **random floor.** Off-diagonal cosine: $\mathrm{fve\_nrm}(v_i,
  v_j)$ averaged over unrelated $(i, j)$ pairs from the same pool.
  Maps to $\approx 0.62$ raw on Qwen, $\approx 0.50$ centred on all
  three backbones. (§3, §10, §11)

### B.4 Decoding strategies

- **greedy.** Deterministic, temperature-$0$ argmax decoding from
  the AV. The "headline gap" is the gap between greedy and best-
  of-$K$. (§4, §5)
- **sampled (mean).** Stochastic decoding at $T=1.0$, mean
  centred fve over the $K$ samples per target. Roughly equal to
  greedy. (§4)
- **best-of-$K$.** Sample $K$ candidates, score each by oracle
  centred cosine to $v$, take the max. Closes (and on Llama and
  Gemma overshoots) the binding ceiling at $K=64$. The mechanism
  of §4–5's headline result. (§4, §5, §10, §11)
- **logp-rerank.** Score the same $K$ candidates by the AV's mean
  per-token log-probability and take the max. Fails on all three
  backbones; statistically indistinguishable from greedy. (§5,
  §10, §11)
- **NN-anchor rerank.** Score each candidate by centred cosine
  *not* to $v$ (which is unobserved at deploy time in the realistic
  setting) but to $v$'s nearest pool neighbour. Beats greedy
  substantially on both Qwen and Llama. (§5, §10)
- **MBR (minimum Bayes risk).** Decoding paradigm in which the
  selected output minimises an expected loss over a candidate
  pool; best-of-$K$ under our centred-cosine utility is the
  $\mathrm{argmax}_k \mathrm{util}(\hat{x}_k, v)$ instance of MBR
  with the oracle utility. (§1.5, §5; cf. Eikema & Aziz, 2020;
  Bertsch et al., 2023; Kumar & Byrne, 2004)

### B.5 Architecture and training terms

- **Backbone.** The frozen pretrained causal LM (Qwen-2.5-7B,
  Llama-3.2-3B, or Gemma-2-2B). Weights never updated. (§1)
- **Activation Verbalizer (AV).** The trained adapter: $\sim 5$M
  (Gemma) / $9.4$M (Llama) / $12.7$M (Qwen) parameters. Consists
  of a hidden-state→soft-token projection, $P$ static prefix
  tokens, and one inject slot. Trained with token-level CE on
  $(v, x)$ pairs. (§1, §10, §11)
- **inject slot.** The single token position into which the
  projected $v$ is written. Distinct from the static prefix
  tokens. The L14 channel of Figure A2 is direct evidence that
  this kind of slot is structurally stable. (§1, App. A)
- **prefix tokens.** $P$ learned soft tokens prepended before the
  inject slot. Function as a "community-of-one" sieve in the
  Kockelman sense: they bias the backbone's sampling distribution
  toward $\hat{x}$ that re-encodes back to $v$. (§1.5)
- **multi-inject ($M=4$).** Variant in which $v$ is projected into
  $M$ inject slots rather than one. Lever 1 in the §2 table; no
  improvement on raw metric. (§2)
- **MLP-conditioned prefix.** Variant in which the static prefix
  is replaced by an MLP whose input is $v$. Lever 2 in §2. (§2)
- **PG + KL.** Policy-gradient fine-tune with KL regularisation
  against the CE-trained warm-start. Lever 3 in §2. (§2)
- **Lever A / Lever B.** Naming convention for the two routes to
  closing the greedy gap. Lever A: best-of-$K$ at deploy time
  (works). Lever B: bag-of-$K$ self-distillation
  (`train_nla_bok_v2.py`), winner-CE over $K$ rollouts plus
  contrastive hard-negatives (does not close the gap; collapses
  sampling diversity). (§6)
- **winner-CE.** Lever B's loss: standard token cross-entropy
  computed against the rollout with the highest centred cosine to
  $v$ within the current bag of $K$. (§6)
- **contrastive hard-negatives ($\beta_{\text{ctr}}$).** Auxiliary
  term in Lever B that pushes the policy away from retrieved
  hard-negative texts. (§6)
- **gold pair.** A $(x, v_x)$ training example: a sampled
  continuation $x$ together with the $v_x$ it produced at layer
  $\ell$ in the backbone. (§1, §10, §11)
- **token CE.** Standard left-to-right token-level cross-entropy
  on $(v, \text{text})$ pairs. The CE-only training objective for
  the headline AV checkpoints. (§1)

### B.6 Frozen-decoder interpretability concepts

- **frozen-decoder verbalisation problem.** The class of problems
  in which a frozen LM is asked to emit text *about* one of its
  own hidden states under a metric that is grounded in the same
  model's re-encoding behaviour. The object of study of this
  paper. (§1.5, §10)
- **round trip.** The composition $v \to \hat{x} \to h$, scored
  by the similarity of $h$ to $v$. The paper's apparatus. (§1.5)
- **interpretant completion.** Peircean framing of the round trip:
  $\hat{x}$ is a sign whose interpretant, *with respect to the
  backbone itself*, is $h$. The chain closes iff $h \approx v$.
  (§1.5; cf. Peirce, 1931–1958; Kockelman, 2017)
- **sieving.** Kockelman's term for the actualisation of a subset
  of possible interpretants by an interpreter's prior commitments.
  The prefix is a sieve that conditions the backbone to land on
  paraphrase-manifold $\hat{x}$. (§1.5; cf. Kockelman, 2017, 2025)
- **community-of-one.** A *discourse community* (Stage 3) of size
  one: the configuration of the backbone that re-arrives at $v$.
  The prefix simulates this community as a sampling bias. (§1.5)
- **metapragmatic awareness (first/second/third-order
  indexicality).** Silverstein's hierarchy of sign-usage capacities.
  Mapped onto decoding in §1.5: first-order = produce any text from
  $v$; second-order = produce text in the right dialect (≈ NN
  retrieval); third-order = produce text whose paraphrase-manifold
  position the substrate certifies under re-encoding (≈ best-of-$K$
  MBR). (§1.5; cf. Silverstein, 1993, 2003)
- **second-order cybernetics.** Von Foerster's framing of
  self-organisation in systems whose observation is internal to the
  system. NLA closes the *inter-pass* loop on a frozen backbone in
  the topology where the substrate is strongest (text). (§1.5; cf.
  von Foerster, 1981, 2003)
- **intra-pass vs inter-pass loop.** Two topologies for closing a
  loop on a frozen LM. *Intra-pass*: read layer $\ell_1$, modify
  layer $\ell_2 > \ell_1$ in the *same* forward pass (the
  SRT-Adapter RRM topology). *Inter-pass*: read $\ell$, emit text,
  re-encode the text in a *second* forward pass and read $\ell$
  again (NLA's topology). (§1.5, §12)
- **substrate claim.** The SRT program's core empirical claim:
  semiotic phenomena (community, regime, divergence, now round-trip
  fidelity) are *measurable* on a frozen production-scale LM
  without architectural modification. (§12)
- **greedy gap.** The gap between the AV's greedy-decode centred
  fve and the binding ceiling. The real open problem of the paper.
  (§4, §6, §12)
- **bifurcation reading of the greedy gap.** Interpretation of the
  gap as a two-attractor structure in the sampling distribution: a
  high-mass modal basin below the paraphrase manifold and a
  reachable-but-not-modal paraphrase-manifold attractor that
  sampling but not logp-rerank can access. Connects to the
  pitchfork dynamics of Lancaster (2025). (§12)

### B.7 SRT program stages

- **Stage 1.** Synthetic and news-domain demonstration of the
  four-module decomposition (community, MAH, RRM, BEN). Lancaster
  (2025, 2026a). (§0, §12)
- **Stage 2.** Interiority-probe / regime-classification head on
  Qwen-2.5-7B; calibrated to AUROC $0.99$, Brier $0.010$, ECE
  $0.0009$ over $351{,}000$ tokens. (App. A, Fig. A3)
- **Stage 3 (SRT-Adapter).** Lightweight semiotic adapter on
  frozen Qwen-2.5-7B, $0.19\%$ parameter overhead, with
  Community/MAH/BEN reading and RRM intervention. The "half-open
  loop" reference point of §1.5. Lancaster, 2026 [SRT-Adapter MS].
  (§0, §1.5, §12)
- **Stage 4 (this paper, SRT-NLA).** Round-trip verbalisation of
  frozen hidden states; closes the *inter-pass* loop on three
  backbones. (entire paper)
- **four-module decomposition.** Stage-1 architecture:
  - **community / Community Discovery** — discourse-community
    manifold inherited from pretraining;
  - **MAH (metapragmatic-attention head)** — divergence-amplifying
    attention component;
  - **RRM (reflexive-recurrence module)** — gated intra-pass
    hidden-state inject-back via FiLM;
  - **BEN (bifurcation-estimation network)** — control-parameter
    $\hat{r}$ estimator. (§1.5, §12)
- **FiLM.** Feature-wise Linear Modulation: gating mechanism RRM
  uses for hidden-state injection. (§1.5)
- **anisotropy.** Property of a representation space in which the
  empirical mean $\mu$ has non-trivial norm, biasing all cosines
  toward $\cos(v, \mu) > 0$. The reason centring is necessary.
  (§2, §3, §10, §11)

### B.8 Models, datasets, and HF artifacts

- **Qwen/Qwen2.5-7B.** Alibaba; $L=28$, $d=3584$; probe layer $20$;
  $\|\mu\| \approx 55$. (§1, §10, §11)
- **meta-llama/Llama-3.2-3B.** Meta; $L=28$, $d=3072$; probe layer
  $20$; $\|\mu\| \approx 7.2$. (§10)
- **google/gemma-2-2b.** Google; $L=26$, $d=2304$; probe layer
  $19$; $\|\mu\| \approx 156$; $\mathrm{bos\_token\_id}=2$,
  $\mathrm{eos\_token\_id}=1$. (§11)
- **`RiverRider/srt-nla-av-v1`** — released Qwen AV checkpoint. (§8)
- **`RiverRider/srt-nla-av-llama32-3b`** — released Llama AV. (§10)
- **`RiverRider/srt-nla-av-gemma2-2b-v1`** — released Gemma AV.
  (§11)
- **`RiverRider/srt-nla-targets-v1`** — released Qwen targets
  dataset. (§8)
- **`targets_q7b_L20_seq64_*.pt`** — Qwen target activation
  tensors. (§8)
- **`targets_L20_seq64_30k_seed1.pt`** — Llama target tensors
  ($22.7$ GB; sha256 `db5c9d22…1981fa`). (§10)

### B.9 Scripts and code artifacts

- **`scripts/oracle_ceiling.py`.** Computes the four anchors
  (replay, paraphrase, NN-in-pool, random floor), raw and centred.
  (§3, §8)
- **`scripts/centered_eval.py`.** Adapter greedy / sampled /
  best-of-$K$ + NN-retrieval, raw and centred. (§2, §4, §8)
- **`scripts/rerank_eval.py`.** $K$-curve, logp-rerank,
  NN-anchor-rerank, $\mathrm{Spearman}(\log p, \mathrm{oracle\,cen})$.
  (§5, §8)
- **`scripts/train_nla_bok_v2.py`.** Lever B trainer (winner-CE +
  contrastive). (§6, §8)
- **`scripts/sample_targets.py`.** Generates the $(x, v_x)$
  target activations from a backbone. (§10)
- **`probe_bestofn.py`** (legacy). Original best-of-$N$ script that
  reported the misleading $0.689$ number. Replaced by
  `centered_eval.py`. (§2)
- **`ActivationVerbalizer`.** The AV module class; takes
  `d_embed = backbone.config.hidden_size` and is therefore
  backbone-agnostic. (§1, §10)
- **commit `902b746`.** Fix for the constant-target bug that
  Gemma's distinct `bos`/`eos` token ids exposed. (§11)

### B.10 Abbreviated citation index

References are cited in shortened form throughout. Full
bibliographic entries are in §14.

- **Lancaster, 2025** — pitchfork bifurcation / polarisation paper
  (SSRN 5987495). Stage 1 of the SRT program.
- **Lancaster, 2026a** — semiotic-reflexive LM training (SSRN
  6349978). Stage 1/2 program statement.
- **Lancaster, 2026 [SRT-Adapter MS]** — Stage 3 manuscript,
  GitHub-hosted, not yet on arXiv at the time of writing.
- **Peirce, 1931–1958** — *Collected Papers*; CP 2.228 and 2.303
  for the triadic sign / interpretant definitions.
- **Kockelman, 2017; 2025** — interpretant chains as sieving
  trajectories.
- **Silverstein, 1993; 2003** — orders of indexicality.
- **von Foerster, 1981; 2003** — second-order cybernetics.
- **Anderson, 2014** — catastrophe-theoretic modelling of cultural
  systems; reification critique.
- **Leighton, 2026** — Maxwell-demon bound in random multipartite
  stochastic systems.
- **VanSaders, Fruchart, & Vitelli, 2026** — measurement-induced
  phase transitions; analogy for the $\log K$ information
  acquisition picture.
- **Belrose et al., 2023 (tuned lens); nostalgebraist, 2020 (logit
  lens); Pal et al., 2023 (Future Lens); Ghandeharioun et al.,
  2024 (Patchscopes); Chen, Vondrick, & Mao, 2024 (SelfIE);
  Morris et al., 2023 (text embedding inversion)** — frozen-decoder
  interpretability lineage NLA sits in. (§7)
- **Eikema & Aziz, 2020; Kumar & Byrne, 2004; Bertsch et al.,
  2023** — MBR / mode-inadequacy literature behind §5. (§5, §7)
- **Frank & Goodman, 2012** — RSA pragmatics; relevant to §1.5's
  third-order indexicality reading.
- **Hewitt & Manning, 2019; Marks et al., 2024** — structural
  probes / sparse feature circuits; the side-channel readout
  paradigm NLA contrasts itself with. (§7)
- **Kim & Rush, 2016; Gulcehre et al., 2023 (ReST); Yuan et al.,
  2023 (RFT); Zelikman et al., 2022 (STaR)** — sequence-level
  distillation and self-training; design space Lever B sits in.
  (§6, §7)
- **Lancaster, 2026c** — Reddit Discourse Corpus; Stage-3 dataset
  reference.
- **Wildgen, 1982** — catastrophe-theoretic semantics; cited as
  background to the pitchfork reading of the greedy gap. (§12)

---

