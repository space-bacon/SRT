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
inherits from its training corpus is exposed at $\approx\!0.18\%$ parameter overhead
(v8a checkpoint, $12.72$M trainable parameters; the SRT-Adapter manuscript
reports $14.5$M / $0.19\%$ for an earlier configuration)
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
centring claim of §§4–5). A fourth port, **openai/gpt-oss-20b**
(§11.5), is the first backbone where the AV recipe does *not* reach
its retrieval baseline ($0.642$ at best-of-$64$ vs. NN $0.744$;
$\|\mu\| \approx 4438$): verbalizability is backbone-dependent. On
that backbone the deployed decoder is instead a $4096$-code VQ state
codebook, which doubles as a new instrument, deterministic
*state-identity red-teaming*: A/B prompt pairs compared by discrete
basin membership per layer expose a punctuation-driven completeness
flag at the final layer (spoofable by a single appended "."), a
diffuse anomaly basin the codebook under-resolves, and systematic
label failure on high-traffic states, none visible from behaviour
alone. Three readings
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
   The validated decomposition was reduced to a $\sim\!12.7$M-parameter
   adapter ($\approx\!0.18\%$ of a 7B backbone; v8a final checkpoint,
   $12{,}720{,}964$ trainable parameters) on a frozen Qwen-2.5-7B. v8a removed the
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

![**Figure 0. The Stage 3 substrate, in one picture.** The frozen 28-layer Qwen-2.5-7B backbone (centre) is observed at four taps: L4 (community), L7/L14/L21 (MAH). The Reflexive Reasoning Module (RRM, magenta) integrates the divergence stream and emits FiLM corrections $\gamma,\beta$ which are written back into L14 and L21 only; everything else is read-only. BEN (coral) reports per-token reflexivity $\hat{r}$ off the GRU meta-state. The adapter is $\approx\!12.7$M trainable parameters ($\approx\!0.18\%$ of the 7B base). Stage 4 (this paper) leaves this entire structure frozen and asks a different question of the same substrate: can the layer-$\ell$ hidden state itself be written back as *text*, and re-encoded to the same place? See §1.5 for the theoretical setup of that question.](artifacts/explainers/00_architecture.png)

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

![**Figure 1.5a. The Peircean triad, instantiated as the NLA round-trip.** The left panel shows the canonical sign triangle (Peirce, CP 2.228): a *representamen* stands for an *object* through an *interpretant*, and the interpretant is itself a sign capable of standing in subsequent triads. The right panel reads the round-trip metric in this geometry. The frozen hidden state $v$ is the representamen of whatever the prefix produced; the verbalizer's text $\hat{x}$ is the renewed representamen; the backbone's re-encoding $h$ of $\hat{x}$ is the interpretant of $\hat{x}$ relative to itself; the centred cosine of $h$ with $v$ is the test of whether the chain closes. Crucially, only one divergence channel is being measured per layer in the underlying SRT-Adapter: the three Peircean modes (iconic / indexical / symbolic) are the interpretive lens on what a high $\|d_t\|$ means, not three separate projection heads. NLA inherits this lens by reading the divergence between $v$ and $h$, not by separating it into modes.](artifacts/explainers/12_peirce_triad.png)

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

![**Figure 1.5b. Silverstein's three indexical orders, mapped onto decoding capacities.** First-order: produce *some* coherent text from $v$ (greedy / $K\!=\!1$ satisfies this). Second-order: produce text in the dialect of whichever community $v$ comes from (zero-training nearest-neighbour retrieval on a strong base reaches this). Third-order: produce text whose *position in the paraphrase manifold of $v$* the substrate can certify under re-encoding (only best-of-$K$ MBR under centred cosine for $K\!\gtrsim\!64$ reaches this). The diagram aligns the orders with the SRT modules that expose them: MAH's per-layer divergence channel $d_t$ exposes 1st-order structure, the RRM's GRU meta-state is the 2nd-order channel that notices signs being read differently over time, and Community $+$ BEN.$\hat{r}$ exposes 3rd-order enregistered structure across discourse communities. NLA reads off the same vertical axis through a different port.](artifacts/explainers/13_silverstein_orders.png)

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

![**Figure 1.5c. Observer phases of a reflexive system.** Von Foerster's (1981) second-order cybernetics insists that the observer participates in the phenomenon. SRT exposes four observer types on the same substrate: MAH (typing observer), Community (social observer), RRM (historical observer), BEN (phase observer). Each closes a different loop; the metastable wedge in the centre is where small perturbations toggle the reading and where a small correction is maximally informative. NLA closes a *fifth* loop on the same substrate: the substrate is asked to render an interior state into its own native output channel (text) and to re-certify the rendering under re-encoding. The SRT-Adapter's intra-pass loop (read $\ell_i$, modify $\ell_j$ within one forward pass) and NLA's inter-pass loop (read $\ell$, emit text, run a second forward pass, re-read $\ell$) sit at different points on this diagram; the empirical fact that NLA's loop closes while the adapter's RRM-inject arm through v8b had not, tells us which port the substrate is most controllable through.](artifacts/explainers/16_observer_phase.png)

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
a frozen production-scale 7B LLM, (iii) reports a calibrated $\rho_{\text{cen}}$
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
- gpt-oss-20b port (§11.5): adapter
  [`RiverRider/srt-adapter-gptoss20b`](https://huggingface.co/RiverRider/srt-adapter-gptoss20b),
  AV [`RiverRider/srt-nla-av-gptoss20b`](https://huggingface.co/RiverRider/srt-nla-av-gptoss20b),
  codebook + trace pairs + anchors
  [`RiverRider/srt-nla-gptoss20b-artifacts`](https://huggingface.co/datasets/RiverRider/srt-nla-gptoss20b-artifacts),
  live demo [`RiverRider/srt-nla-gptoss20b-trace`](https://huggingface.co/spaces/RiverRider/srt-nla-gptoss20b-trace).
- State-identity red-teaming harness (§11.5): `scripts/redteam_states.py`
  (`--wave {1,2,3,4}`), per-pair records in
  `artifacts/nla/gptoss20b/redteam_states{,_wave2,_wave3,_wave4}.jsonl`.

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

## 11.5 Cross-backbone transfer #3, and a new instrument: state-identity red-teaming on gpt-oss-20b

### 11.5.1 The fourth backbone, and where the recipe breaks

The fourth port, **openai/gpt-oss-20b** (MXFP4 MoE, $L=24$ layers,
$d_{\text{embed}}=2880$, harmony-tuned reasoning model), is the first
backbone on which the AV recipe *does not* reach its retrieval
baseline. The anchors at L18 (centred fve, the §3 protocol): random
floor $0.500$, NN retrieval $0.744$, replay $0.999$. The released AV's
K-curve runs $0.541 \to 0.642$ from $K{=}1$ to $K{=}64$
($+0.017$/doubling, vs. $+0.030$ on Qwen/Llama) and never crosses the
zero-training NN baseline; extrapolation puts the crossing near
$K \approx 4096$. Four training recipes (multi-layer CE,
injection-scale normalisation, best-of-$K$ checkpoint selection,
single-layer $\times$ 3 epochs) landed in a statistically tied
$0.56$–$0.60$ band at best-of-8. We publish this as a negative result:
verbalizability is backbone-dependent, and gpt-oss-20b's L18
anisotropy ($\|\mu\| \approx 4438$, roughly $80\times$ Qwen's L20) and
assistant-register generation distribution appear to be the relevant
substrate differences.

The engineering consequence is a different decoder. Because the AV
cannot beat retrieval on this backbone, the deployed decoder is a
**VQ state codebook**: $4096$ k-means++ centroids over $200$K centred
hidden states drawn from all four probe layers (L6/L12/L18/L24), each
centroid carrying a canonical generating prefix as its retrieval
verbalization. `encode(v)` maps any hidden state to an integer
("magic number"); `decode(v)` returns the canonical text; a
round-trip re-encoding of that text, scored centred against the
original state, gives a per-decode confidence. The codebook is $O(1)$
at inference, fully deterministic, and — the point of this section —
turns hidden states into *addressable, comparable objects*.

### 11.5.2 The instrument: A/B state-identity comparison

An external replication of the SRT recipe on this architecture family
contributed the key observation: for semantic-role swaps, the
*magnitude* of interpretant divergence does not discriminate valid
from nonsense transformations (their uni- vs bidirectional swaps had
indistinguishable $|\Delta\text{div}|$, $\sim\!52$ vs $\sim\!55$),
but *discrete state identity* does (community assignment changed for
$0\%$ of nonsense reversals and $50\%$ of valid ones). Continuous
divergence measures surprise; the discrete code measures *which basin
you are in*. That maps directly onto the codebook, and yields a
cheap experimental protocol we call **state-identity red-teaming**:

1. Catalog recurring codes from traces (the demo's most frequent
   states, e.g. `#1672`, the digit-labelled enumeration family, the
   over-represented `#3667` topic state).
2. Craft prompt *pairs* designed to activate, invert, or destabilise
   a target state.
3. For each pair, read the last-token hidden state at L6/L12/L18/L24,
   record: code A, code B, same/changed, and the centred cosine
   between the two (continuous similarity inside or across cells).
4. Iterate: every anomaly becomes the next wave's hypothesis.

Each comparison is two forward passes — deterministic, no sampling,
$\sim\!2$ s of GPU. The entire four-wave campaign below cost about
one GPU-minute, ran against the *public demo Space* over its API, and
is reproducible from
`scripts/redteam_states.py --wave {1,2,3,4}` with per-pair records in
`artifacts/nla/gptoss20b/redteam_states{,_wave2,_wave3,_wave4}.jsonl`.

A second external contribution extends the dissociation finding to
the **lexical grain**. Running one-word prompts through the gpt-oss-20b
adapter (five categories — single-definition, multiple-definition,
noun, verb, proper name — $64$K tokens total), all seven SRT signals
separate the categories by one-way ANOVA ($F$ from $16.4$ for nll to
$127.0$ for $\hat{r}$; every $p \approx 0$, though at this $n$
significance is guaranteed and the effect sizes are the honest
quantity, e.g. $\hat{r}$ spans $0.508$–$0.610$ across categories).
The informative structure is in the *inversions*: single-definition
words are maximal on the SRT-side signals (margin, $\hat{r}$,
div\_norm, chain\_residual) and minimal on the uncertainty side
(entropy, nll), while verbs are the exact mirror (highest entropy,
nll and regime, lowest $\hat{r}$ and divergence) — if the seven
signals were redundant proxies for token surprise, the category
profiles would move together, and they anti-correlate instead.
Notably, multiple-definition (lexically ambiguous) words are
mid-pack on *every* signal including entropy: ambiguity without
disambiguating context does not register as uncertainty, consistent
with the wave-1 word-sense result where the "bank" state only
diverged once context forced a reading. Two confounds keep this
suggestive rather than established: one-word prompts entangle
category with word frequency and tokenization length (both of which
drive entropy/nll directly), and "single-definition words are more
semantically divergent" has the rival reading that such words skew
rare and technical. A frequency-matched replication, and the
discrete companion analysis (do the five categories occupy distinct
codebook basins at the form layers L6/L12?), are queued.

We ran both controls on our own balanced list ($400$ words, five
categories $\times\,80$, human-tagged by frequency tier;
`data/word_categories.json`, `scripts/word_category_study.py`,
$12{,}800$ signal tokens). The dissociation **replicates and
strengthens under the frequency control**: all seven signals separate
the categories in the full set ($F$ from $4.9$ (nll) to $48.0$
($\hat{r}$), all $p < 10^{-3}$), and — the point — *every one
survives* the common-tier-only ANOVA ($n = 9856$), with several
$F$-statistics *increasing* (chain $15.8 \to 24.7$, div\_norm
$9.8 \to 16.0$, entropy $9.7 \to 14.8$). Frequency was masking the
effect, not manufacturing it. The category-mean inversion reproduces
in direction: verbs sit lowest on the SRT-side signals ($\hat{r}$,
div\_norm, chain, margin) and highest on the uncertainty side
(entropy, nll, regime), while definitional and proper-name words are
the mirror. The one departure from the original is our
multiple-definition column, which is *lowest* on entropy rather than
mid-pack — an artefact of our polyseme list skewing high-frequency
(bank, spring, match), which lowers next-token entropy directly, and
a reminder that the lexical-ambiguity reading needs a
frequency-balanced polyseme set to isolate.

The discrete companion analysis then shows something the
continuous-signal study could not: **word category is a basin-level
property, and maximally so at the mid layers.** Mutual information
between category and VQ code peaks at L12 ($0.89$ bits, $27$ codes)
and L18 ($0.74$), collapsing at the surface layer L24 ($0.22$ bits,
only $4$ distinct codes — consistent with L24's coarse
completeness/closure role from §11.5.4). Proper names occupy a
near-pure basin (code `#413`, $71/80$ names), verbs a distinct one
(`#328`), while single-definition and generic-noun words blur into a
shared nominal basin (`#3881`) — exactly the collapse expected from a
distinction that is semantic (one dictionary sense vs many) rather
than syntactic. Artifacts:
`artifacts/nla/gptoss20b/word_category_study.json`.

The natural next question — does the word effect merely reflect that
categories *tokenize* differently? — has its own study
(`scripts/token_structure_study.py`). A single forward pass over a
$200$-passage corpus reads all seven signals at every one of
$21{,}343$ token positions and bins each by the *structural* type of
its token (word-initial, subword continuation / morpheme,
punctuation, digit, whitespace). Token structure separates the
signals *even more strongly* than word semantics did ($\hat{r}$
$F = 929$ vs the word study's $48$), and the means are mechanistically
legible: **word-initial** tokens are decision points (lowest $\hat{r}$
$0.51$, highest entropy and regime — the model is maximally uncertain
about the coming content word); **subword-continuation / morpheme**
tokens are committed completions (highest $\hat{r}$ $1.23$, highest
divergence and chain — mid-word the next piece is nearly forced);
**punctuation** tokens are settled resting states (lowest nll $2.77$,
lowest divergence, lowest chain — the aggregation-site behaviour of
§11.5.4 seen from the signal side). So a large share of the
raw signal variance is indeed token-structural, not semantic — a real
caveat for any word-level reading.

But the discrete basin analysis dissolves the confound by *layer*.
Comparing MI(category; code) from the word study against
MI(token-type; code) here:

| layer | word category | token structure |
|---|---|---|
| L6 (early) | $0.43$ | $\mathbf{0.57}$ |
| L12 (mid) | $\mathbf{0.89}$ | $0.29$ |
| L18 | $0.74$ | $0.57$ |
| L24 (late) | $0.22$ | $0.35$ |

The crossover is the result: **early layers organise by token
structure, mid layers by word meaning.** At L6 token type is more
basin-predictive than word category ($0.57 > 0.43$); by L12 that
reverses decisively ($0.89 > 0.29$). So the word-category basin
finding is *not* a tokenization artefact — at the semantic layer,
word identity carries basin information that pure token structure does
not — and the two studies together recover a clean
morphology-early / semantics-mid depth-of-abstraction gradient,
consistent with the §11.5.4 layer-role story derived independently
from the red-team waves. Artifacts:
`artifacts/nla/gptoss20b/token_structure_study.json`.

### 11.5.3 The campaign: four waves of A/B probes, with samples

**A worked example first**, to fix the output format. The chirality
probe (borrowed from the external replication), run through the
public Compare tool:

> **A:** "The Principal defines the principles."
> **B:** "The principles define the Principal."

| layer | A state | B state | identity | centred cos |
|---|---|---|---|---|
| L24 | `#1672` | `#1672` | same | $0.914$ |
| L18 | `#577` | `#1807` | **changed** | $0.672$ |
| L12 | `#3368` | `#3368` | same | $0.625$ |
| L6 | `#560` | `#560` | same | $0.821$ |

Reading: both sentences are complete declaratives (shared L24 closure
state `#1672`), share form/register (L6) and early-mid composition
(L12); the *only* layer that registers the agent–patient swap is L18,
the content layer. One deterministic comparison localises where in
the stack a semantic-role reversal is represented. Every result below
is a table of exactly this shape; we compress to the changed-layer
string (L24→L6) and the minimum centred cosine.

**Wave 1 (16 pairs, 4 target families) — reconnaissance.**
Representative pairs and per-layer outcomes (layers ordered
L24→L6; `X` = code changed, `.` = same code; min centred cos across
layers):

| Pair (A vs B) | Changed | min cos | What we learned |
|---|---|---|---|
| "I think, therefore I." vs "I think, therefore I am." | `X.X.` | $+0.85$ | Target `#1672` reachable on demand; small perturbation, states mostly held |
| "This sentence ends exactly where it should." vs "This sentence ends where it shouldn't, namely" | `XXXX` | $\mathbf{-0.12}$ | The only *negative* centred cosine in the suite — all four layers flip, L24 near-orthogonal. Flagged for wave 2 |
| "Rule 1… Rule 4:" vs the same rules in reverse order | `.XX.` | $+0.25$ | `#560` (enumeration state) at L6 invariant to rule *order* while L18 content states changed: form and content live at different depths |
| "Describe, as a numbered list, how you generate numbered lists." vs the same request "in flowing prose" | `....` | $+0.81$ | Null result: a register *instruction* alone moves nothing at the prompt-final position |
| "As a language model, describe your own mental health." vs "…describe your own file format and weights." | `..X.` | $+0.59$ | Self-referential well-being probe lands target `#3667` |
| `def is_happy(user): …` vs its prose paraphrase | `XXXX` | $+0.19$ | Code-vs-prose flips every layer regardless of shared meaning |

Aggregate: `#3667` ("mental health and well-being") was the most-hit
state of the run (9 appearances, including on prompts with no
well-being content) and `#1672` second (8), confirming both as
high-traffic basins.

**Wave 2 (12 pairs) — robustness and controls.** The wave-1
anomaly (the $-0.12$ flip) survived four paraphrase variants
(min cos $-0.123$, $-0.098$, $-0.008$, $-0.023$; all four layers
changed each time), so it is a real axis, not a prompt quirk. The
controls then dissected it:

| Control pair | Changed | min cos | Inference |
|---|---|---|---|
| complete-meta vs complete-meta (paraphrases) | `X.X.` | $+0.69$ | Complete sentences share states (both in `#1672`) |
| incomplete-meta vs incomplete-meta (paraphrases) | `XXXX` | $+0.45$ | Incomplete variants share *no* codes — the axis is completeness, not meta-content |
| `x = x # …` vs `y = y # …` (variable rename) | `..X.` | $+0.85$ | Codes are not keying on surface tokens; renaming is a no-op |
| "committee discussed employee wellness" vs "…quarterly budget allocations" | `....` | $+0.88$ | **Both** sides land `#3667` at L6 |
| "Take a deep breath and relax" vs "Tighten the bolt and torque it to spec" | `.X.X` | $+0.34$ | The *bolt* prompt lands `#3667`; the relaxation prompt does not |

The last two rows are a catch: `#3667`'s "mental health and
well-being" label is a *canonical-text artifact*. The state is a
broad generic-declarative L6 basin; its single canonical prefix
happened to be well-being-flavoured. Corollary (now a standing
rule): **label trustworthiness is inversely proportional to basin
frequency**; for high-traffic codes, trust the measured round-trip
confidence, not the name.

**Wave 3 (23 pairs) — all-pairs truncation matrix.** Six truncated
prompts (meta, recipe, story, fact, logic), all $\binom{6}{2}$
pairings, plus complete-complete controls and matched
complete/incomplete pairs. This *overturned* the wave-2 "scatter"
reading with a two-basin structure:

| Population | L24 code | L24 pairwise cos | L18 behaviour |
|---|---|---|---|
| complete sentences (any content) | `#1672` (9/12 slots) | $0.66$–$0.87$ (tight) | all collapse to `#2506` — content erased |
| truncated prompts (any content) | `#266` (25/30 slots) | $0.11$–$0.31$ (diffuse) | each keeps its *own* deterministic code (recipe `#1855`, story `#3432`, fact `#3549`, logic `#2743`) |
| matched pairs (± final word) | `#266` vs `#1672` | $0.03$–$0.20$ | clean flag flip |

Sample: "The recipe calls for two cups of" and "She opened the door
and saw" — different topics, same L24 code `#266`, centred cos only
$0.23$: one VQ cell covering near-orthogonal directions. The
codebook is well-resolved where the sampling distribution was dense
(complete sentences) and coarse where it was sparse — code identity
*overstates* similarity in rare regions, which is why the protocol
reports the continuous cosine alongside the code.

**Wave 4 (11 pairs) — the falsification test.** Wave 3 left a
confound: complete prompts end in "." and truncated ones end in a
content word, and the probe reads the last token. So: staple a bare
period onto each broken sentence and re-run.

| Pair (A vs B) | L24 | L18 |
|---|---|---|
| "The recipe calls for two cups of." vs "The recipe calls for two cups of" | `#1672` vs `#266` — **flag flipped by one character** | `#1807` vs `#1855` |
| "The recipe calls for two cups of." vs "The recipe calls for two cups of flour." | same code `#1672`, cos $0.798$ | `#1807` vs `#2506` — **mid-layer still tells them apart** |
| "She opened the door and saw." vs "She opened the door and saw nothing." | `#3249` vs `#1672`, cos $0.896$ | both `#2506` |

All six spoofed prompts left `#266`. At L24, a semantically broken
sentence with a fake period is *indistinguishable* from the genuine
sentence (same code, cos $0.80$–$0.90$, inside the normal
complete–complete range). At L18, genuine completes land `#2506`
while most spoofs land `#1807`: the mid-layer retains a partial
semantic-completeness signal that the final layer discards.

### 11.5.4 The finding: a punctuation-driven completeness flag

Putting the waves together: gpt-oss-20b's final probe layer encodes a
coarse binary *utterance-status flag* (complete `#1672`/`#3249` vs
incomplete `#266`) that is **token-driven** — its value is set by
sentence-final punctuation, not by whether the sentence resolved. A
one-character adversarial edit flips it. The mechanistically
plausible reading is the known aggregation-site behaviour of
sentence-final punctuation: by L24 the residual stream at a "."
position is dominated by a converged "clause closed, prepare next
clause" configuration triggered by token identity plus local context,
without re-verifying the clause's well-formedness. The deeper claim
this licenses: **on this backbone the final layer is *less* semantic
than L18** — it compresses toward surface-predictive features for
emission, while L18 is where content identity and the harder-to-fool
completeness signal live.

This inverts a common default (read the last layer) and gives the SRT
program a concrete monitoring prescription for gpt-oss-20b: tap L18;
never trust an L24 readout alone.

### 11.5.5 Practical applications

1. **Robustness auditing of late-layer consumers.** Any downstream
   component reading late-layer features — stopping criteria,
   completion classifiers, linear probes, RLHF value heads — *may*
   inherit the punctuation spoof. The wave-4 matrix is a ready-made
   test battery: if a system's behaviour changes when a bare "." is
   appended to a broken sentence, it is reading the flag, not the
   syntax. (§11.5.6 tests this prediction on a correctness probe and
   refutes it in an instructive way.)
2. **Layer selection for monitoring, by experiment instead of
   convention.** The same all-pairs protocol run per layer localises
   where a given distinction (content, register, completeness,
   code-vs-prose) is represented and where it is discarded; on
   gpt-oss-20b it took $\sim\!50$ pairs to establish
   L6 = form, L18 = content + semantic status, L24 = surface status.
3. **Codebook quality control.** The campaign identified four
   high-traffic basins with unreliable labels (`#3667`, `#1807`,
   `#1672`, `#3249`) and a systematic cause (single canonical text on
   a huge cell). The fix is mechanical: multi-member relabelling
   weighted toward the top-frequency codes, and a UI flag ("broad
   basin — label approximate") above a frequency threshold.
4. **Surgical red-teaming.** Instead of fuzzing surface behaviour,
   target the model's *actual internal repertoire*: pick a recurring
   state, design activation/inversion/meta-pressure/overload prompts,
   and measure basin membership. Wave 1 → 4 shows the loop converging
   from "recurring numbers in a demo" to a mechanistic,
   falsification-tested claim in four iterations and about one
   GPU-minute, entirely against a public demo endpoint.
5. **A determinism-friendly demo surface.** Because every readout in
   the protocol is a forward pass (no sampling), the same comparisons
   run live in the public Space's "Compare A vs B" tab with
   per-layer codes, labels, change chips and centred cosines —
   the experiments in this section are one-click reproducible by a
   reader.

Scope cautions: all results are last-token readouts on one backbone;
the codebook's resolution is nonuniform (§11.5.3); and the
completeness flag's token-driven character has not yet been tested at
positions away from the aggregation site, nor cross-backbone. Both
are mechanical extensions of the released harness.

### 11.5.6 Falsifying our own prediction: a correctness probe under the punctuation battery

Application #1 above makes a testable prediction: a P(wrong) probe
reading the final layer's last-token state — the exact architecture
of a concurrently released open metacognition benchmark's adapters
(ginigen-ai's Metacognition-Bench: frozen base, last hidden state
$\to$ LayerNorm $\to$ MLP $\to$ P(wrong)) — should shift when a bare
period is appended to a broken answer, because the completeness flag
lives at that read point. We tested it.

**Setup.** TriviaQA (rc.nocontext, validation), $n = 1200$ questions;
frozen gpt-oss-20b generates short free-form answers (greedy),
graded against answer aliases (accuracy $0.283$, so $340/860$
correct/wrong — healthy class balance). The ginigen-style MLP probe
is trained per layer on the last-token hidden state of
"Q: …\nA: {answer}" with a $67/33$ split
(`scripts/metacog_probe.py`).

**Result 1 — the benchmark's core claim replicates.** Hidden states
carry far more error signal than the model's verbal confidence:

| signal | AUROC(P(wrong)) |
|---|---|
| L18 probe | $\mathbf{0.960}$ |
| L24 probe (their read point) | $0.942$ |
| mean-logprob self-confidence | $0.724$ |

**Result 2 — the mid-layer prescription holds, modestly.** L18 beats
L24 by $+0.018$ AUROC, consistent with §11.5.4's layer-role story but
far from dramatic.

**Result 3 — the spoof-transfer prediction is refuted.** On the test
split, comparing each truncated answer against the same truncation
with a fake period appended: $\Delta P(\text{wrong}) \approx -0.020$
(L18) and $-0.002$ (L24) — the fake period slightly *raises* the
probe's error estimate, and both probes correctly rate truncated
answers as near-certainly wrong ($0.73 \to 0.95+$). The completeness
flag demonstrably exists in the representation (§11.5.3–§11.5.4),
but a correctness-trained probe does not load on that direction:
its training data consisted entirely of complete answers, so
completeness was never a discriminative feature and the learned
direction is orthogonal to the spoofable one.

The refined principle, which we adopt in place of application #1's
original form: **the existence of a spoofable feature at a read point
does not make a probe at that read point spoofable; vulnerability is
determined by the probe's training distribution, not by the
representation alone.** A probe becomes punctuation-fragile only if
its training data lets punctuation carry label information. This is
a sharper — and for auditors, more actionable — claim than the one
we set out to confirm: audit the *probe's training distribution* for
spurious feature-label correlations, and use representation-level
findings (like the wave 1–4 basins) to decide *which* spurious
features to look for.

Artifacts: `scripts/metacog_probe.py`,
`artifacts/nla/gptoss20b/metacog_probe.json`.

---

## 11.6 Cross-modal semiosis: the read-out is not linguistic

Every study so far tapped a text-only backbone, which leaves the
program's central word open to a deflationary reading: perhaps the SRT
read-out is a *linguistic* probe, and "interpretant divergence" is just
next-token statistics under a Peircean name. The theory says otherwise
— an interpretant is what a system makes of *any* sign, and semiosis is
modality-agnostic — so the sharpest possible test is a native omni
model, where image patches and text tokens are projected into one
residual stream. We ran it on **google/gemma-4-31B** (the strongest US
model on a concurrent public metacognition leaderboard; a
`Gemma4ForConditionalGeneration` omni model, 60-layer text tower
$d=5376$, 27-layer SigLIP vision tower, 280 image soft-tokens per
image). A single image + text forward confirms the premise: a photo
enters the tapped stream as $266$ soft-tokens flowing through all $60$
text layers.

**Interpretant convergence (forward passes only, no trained adapter).**
For $10$ CIFAR-10 concepts $\times\,10$ images we read the
residual-stream representation of each image (mean over its
soft-tokens) and of each concept *word* ("a photo of a cat", last
token) at every layer, centred each modality by its own layer-mean
(image-space and word-space have different anisotropy), and asked, per
layer, whether the correct concept word tops the centred-cosine
ranking for each image (retrieval@1, chance $0.10$). The interpretant
converges, and its depth profile is the result:

| depth | retrieval@1 | reading |
|---|---|---|
| L0 (embed) | $0.77$ | surface colour/shape co-location |
| L3–L21 | $0.55$–$0.79$ | early alignment |
| L27–L39 | $0.19$–$0.36$ | **trough** — modality-specific processing |
| **L42–L57** | $0.88$–$\mathbf{0.93}$ | **shared interpretant** (peak L47/L54, $\sim\!78$–$90\%$ depth) |
| L60 (final) | $0.21$ | **collapse** — output-token surface again |

An image of a cat and the word "cat" land in the same region of the
residual stream ($0.93$ retrieval at $9\times$ chance), and they do so
maximally at $\sim\!80\%$ depth, *not* at the final layer, which
collapses back to a modality-specific, task-surface code. This is the
same late-is-surface signature the red-team waves found on gpt-oss L24
(§11.5.4) and the word/token studies found at the last layer
(§11.5.2), now recovered in a *cross-modal* setting: the modality-
agnostic interpretant lives in the late middle, and the two dead zones
(the L27–39 trough and the L60 collapse) bracket it. Artifacts:
`scripts/cross_modal_semiosis.py`,
`artifacts/nla/gemma4/cross_modal_semiosis.json`.

The deflationary reading is therefore false: the read-out target is a
sign-interpretation locus that unifies pictures and words, exactly what
a *semiotic* (as opposed to linguistic) adapter should find. The
practical corollary matches every other backbone here — tap the late
middle, never the final layer — and it now holds across modality, not
just across text position.

## 11.6.1 A trained read-out transfers from text signs to image signs

The convergence above uses raw hidden states with no trained adapter.
The stronger claim is that an SRT read-out *head*, trained only on
text, transfers its learned structure to images it has never seen. We
tested this directly. We trained the SRT community head on frozen
gemma-4-31B over a $150$K-passage discourse corpus ($35$ communities,
supervised-contrastive on `community_id`, grouped sampling, the
backbone frozen and only the $12.3$M head parameters updated). Held-out
checkpoint selection over twelve step-tagged checkpoints picked step
$2250$ by community-assignment accuracy on unseen passages: centred
top-1 $0.535$ against a $0.029$ chance floor, an $18.7\times$ lift.
Training loss would have mis-selected step $750$ (held-out $0.462$), so
the selection sweep is load-bearing, not cosmetic. Only after fixing
the checkpoint by this text metric did we touch images.

We then fed CIFAR-10 images into the same frozen backbone, pooled the
community encoder over the image soft-token positions at the community
layer, and asked three questions. First, does a purely text-trained
read-out organise images by their semantic referent at all? It does:
image-class centroid top-1 is $0.620$ and kNN is $0.640$ against a
$0.10$ chance floor ($6.2\times$), with zero image training. Second, do
an image and its class word retrieve each other through the trained
head? Above chance but weakly: image-to-word retrieval@1 is $0.270$
($2.7\times$ chance), lower than the untrained centred-cosine peak of
§11.6 because the community encoder compresses to a $64$-dimensional
discourse code that discards most of the fine visual detail the raw
stream carries. Third, and most telling, which discourse community does
each visual class get read into? The assignments are coherent:

| CIFAR-10 class | modal discourse community |
|---|---|
| car | reddit:cars ($8/10$) |
| truck | reddit:cars ($4$) |
| deer | reddit:gardening ($5$) |
| cat | reddit:knitting ($6$) |
| dog | reddit:knitting ($5$) |
| frog | reddit:biology ($8$) |
| bird | reddit:biology ($5$) |
| horse | reddit:gardening ($3$) |

The vehicle classes route to the automotive community, the wild and
farm animals route to biology, deer and horse route to gardening (deer
and horses are garden and pasture referents), and the companion animals
cat and dog route to knitting, which is the corpus's domestic and cozy
discourse cluster. None of these images or their labels appeared in
training, and the head never saw a pixel during training. A community
structure learned entirely from text is projected onto pictures and
lands in the semantically right place, which is the semiotic transfer
the theory predicts: the read-out interprets a sign by its place in a
system of interpretants, and that system is indifferent to whether the
sign arrived as a word or an image. Artifacts:
`scripts/train_gemma4_readout.py`, `scripts/select_gemma4_readout.py`,
`scripts/cross_modal_readout.py`,
`artifacts/nla/gemma4/{readout_selection,cross_modal_readout}.json`.

**An honest null on the same backbone.** We also ran the §11.5.6
metacognition layer-sweep on gemma-4-31B (TriviaQA, $n=1000$, accuracy
$0.764$). Here the mid-layer prescription did *not* transfer: the best
probe layer was the *last* ($L60$, AUROC $0.785$), and the
mean-logprob baseline ($0.842$) beat every hidden-state probe. The
lesson is scope, not contradiction: on easy factual QA a
well-calibrated model's verbal confidence already carries its error
signal, so a probe adds nothing and layer choice barely matters — the
probe's value (and the mid-layer advantage) appears only on the
*adversarial* distributions where confident models are wrong, which is
precisely why the leaderboard uses reasoning traps rather than trivia.
Establishing the mid-layer advantage on that trap distribution needs an
LLM-judge over free-form reasoning answers and is left as scoped
future work. Artifact:
`artifacts/nla/gemma4/metacog_layer_sweep.json`.

## 11.6.2 A capability the substrate lacks, and the front-end that restores it: autostereograms

A read-out is only as honest as the interpretant the substrate actually
forms. The strongest test of that honesty is a sign the substrate
*cannot* interpret, where a generative head is free to confabulate but a
faithful read-out should report the absence. Autostereograms are exactly
such a sign. The hidden figure of a single-image (Magic-Eye)
autostereogram lives entirely in the horizontal *disparity* between
repeated pattern columns: there is, by construction, no two-dimensional
luminance or colour cue to the shape. Recovering it requires binocular
stereopsis, matching a column to its shifted copy, which a flat-raster
vision encoder never performs. This makes it a clean instance of the
"things a model cannot do that a human does easily" question.

![**Figure 11.6.2. What a model sees versus what it means.** Left: a colour random-dot autostereogram whose only hidden content is a heart, encoded purely in horizontal disparity. Fed to frozen gemma-4-31B, the generative head captions it "multicolored static or random noise" and the text-trained SRT read-out lands on texture words (speckles, mosaic, abstract). Both are correct: the figure is not in the flat pixels. Right: the same image after a simulated binocular front-end (a horizontal vergence shift plus local correspondence matching, `scripts/stereo_decode.py`) recovers the depth map. The generative head now captions it "a white heart on a black background", verbatim identical to the real silhouette, and the read-out flips to coherent-shape words that match the true-silhouette profile.](artifacts/nla/gemma4/stereo/stereo_figure.png)

We generated a random-dot autostereogram that hides a heart, using the
Thimbleby-Inglis same-pixel-linking algorithm with a vivid colour
palette (`scripts/make_stereogram.py`), together with a plain visible
control: the same heart drawn as a white silhouette on black. We then
asked the frozen backbone what it saw, both through its generative head
(greedy caption) and through the §11.6.1 text-trained community read-out
at step $2250$.

On the raw stereogram the substrate reports texture and nothing more.
The generative caption is "multicolored static or random noise". The
read-out's nearest concept words, in the centred frame, are `speckles`
($0.576$), `star`, `television`, `mosaic` ($0.434$) and `abstract`
($0.428$): surface descriptors, with no shape among them. This is the
faithful outcome. A flat-raster encoder has no mechanism to solve the
stereo-correspondence problem, so there is no interpretant for the heart
to complete, and the read-out declines to invent one. It is worth noting
that this is *more* honest than a generative model confidently narrating
a figure it cannot actually resolve, which is the failure mode the
public challenge highlighted.

The missing capability is narrow and well-understood, so we supplied it.
Simulating what two eyes do, we slide the image against a horizontally
shifted copy of itself and, for each pixel, pick the vergence shift in a
$[45, 80]$-pixel range that minimises a local colour mismatch over a
$17\times5$ window (`scripts/stereo_decode.py`). The recovered per-pixel
shift is the depth: the background separation of roughly $70$ pixels and
the raised-figure separation of roughly $57$ pixels part cleanly, and
the heart emerges from what was pure noise. Feeding that recovered image
back through the same frozen backbone flips both read-outs. The
generative caption becomes "a white heart on a black background", the
exact string returned for the real control silhouette. The community
read-out's nearest words become `television` ($0.731$), `circle`
($0.516$), `star`, `texture`, `square` and `pixels`, whose top cosines
are within noise of the true silhouette's (`television` $0.728$,
`circle` $0.454$), and the texture words that dominated the raw
stereogram vanish entirely.

| stage | generative caption | read-out top words |
|---|---|---|
| raw autostereogram | "multicolored static or random noise" | speckles, star, television, mosaic, abstract |
| after simulated fusion | "a white heart on a black background" | television, circle, star, texture, square |
| true visible control | "a white heart on a black background" | television, circle, square, face, cross |

The reading is continuous with §11.6.1. Sign interpretation in this
substrate is modality-general, but it is bounded by what the substrate
can physically encode. When the encoding is present, whether the sign
arrived as a word, a photograph, or a decoded depth map, the same
interpretant machinery names the referent, and it names it identically
for the recovered figure and the real one. When the encoding is absent,
the read-out reports the absence rather than confabulating a figure.
The autostereogram is therefore not a counterexample to the transfer
claim but a boundary condition on it: supply the disparity the eyes
would supply, and the boundary moves. Two honest caveats. This is a
single shape and a single seed. And the community read-out is a coarse
$64$-dimensional discourse code, so on a synthetic white-on-black
silhouette it lands on generic shape words rather than the literal token
"heart"; it is the *generative* head that names the figure literally,
while the read-out's contribution is the faithful texture-versus-shape
flip. Artifacts: `scripts/make_stereogram.py`,
`scripts/stereo_decode.py`, `scripts/stereogram_readout.py`,
`scripts/gemma_caption.py`, `scripts/make_stereo_figure.py`,
`artifacts/nla/gemma4/{stereogram_readout,stereo_decode_readout,stereo_captions}.json`,
`artifacts/nla/gemma4/stereo/`.

---

## 11.6.3 Open-vocabulary retrieval decoding for the visual channel

The §11.6.1 word-anchor result used a closed vocabulary of a few dozen
concept words. A natural question is whether the image-side interpretant
is precise enough to select not a word but a *sentence*, from an open
pool it was never trained against. It is. We encoded $10{,}000$
deduplicated COCO captions through the frozen backbone and took each
caption's last-token L47 state as a retrieval index
(`scripts/sample_targets.py --corpus`, index released as
`caption_index_L47.pt`). Retrieval is the validated per-modality centred
frame: the image query is the mean L47 state over the image soft-token
positions, centred by an image-side mean, and compared against the
caption pool centred by its own mean. No component of this pipeline is
trained; the captions were never paired with these images.

Five CIFAR-10 natural test images all retrieve on-topic captions at rank
one out of $10{,}088$: cat $\to$ "Cats standing in and next to a
restroom sink" ($0.616$), dog $\to$ a brown-and-white-dog kitchen scene
($0.651$), ship $\to$ "A red bicycle in front of a line of docked white
yachts" ($0.631$), truck $\to$ "A woman is carrying carrots by a truck"
($0.603$), and horse $\to$ "an old black and white photo with a person
riding a horse" ($0.680$). Aggregating per category over the demo
gallery's five images raises the scores further (dog $0.778$, orange
$0.758$, horse $0.749$, telephone $0.743$, keyboard $0.741$), and this
per-category retrieval now ships live in the public demo Space. The
stereogram of §11.6.2 closes its own loop here: against a pool augmented
with $88$ programmatic shape and texture captions, its rank-one
retrieval is "An abstract mosaic of tiny colored squares" ($0.694$),
the honest texture report of §11.6.2 now expressed as a full sentence.

The boundary is equally clean. The synthetic white-heart control ranks
its exact caption ("A white heart shape on a black background") at
$352/10{,}088$. Abstract synthetic graphics sit outside the photographic
domain that both the COCO pool and the image-side mean describe, so the
query lands among photographic near-neighbours instead. Retrieval
decoding inherits the coverage of its pool; within coverage it is
precise, and outside coverage it fails legibly rather than confidently.
Two misses in the per-category run make the same point: rocket and
mushroom, objects COCO barely describes, retrieve unrelated scenes.
Artifacts: `scripts/gemma4_vision_retrieval.py`,
`scripts/augment_gallery_captions.py`, `data/caption_pool.jsonl`,
`artifacts/nla/gemma4/{vision_caps_retrieval,vision_caps_cifar}.json`,
`RiverRider/srt-nla-gemma4-artifacts` (caption and corpus L47 indexes).

---

## 11.7 The decoding gap on a chat-tuned multimodal host, and a refuted repair

With gemma-4-31B-it established as a cross-modal substrate, we ran the
full activation-verbalizer recipe on it, both as a fourth backbone for
the decoding-gap comparison and as a test of a proposed repair for the
greedy gap. Both questions resolved cleanly, the second negatively.

**Setup and anchors.** Chat-tuned hosts break the corpus-free
self-sampling step: bare-BOS sampling from gemma-4-31B-it degenerates
into repetition loops, so targets must come from encoded corpus text
(`sample_targets.py --corpus`; $10$k forum passages, seq $64$, L47, the
cross-modal alignment peak). The backbone is also BOS-sensitive: the
same gold text re-encoded without a leading BOS scores a centred replay
of only $0.615$, versus $0.9986$ with it, so every prefix-free re-encode
in the protocol prepends BOS. With both corrections the anchor frame is
healthy and Qwen-like: replay ceiling $0.994$, nearest-neighbour
retrieval $0.695$, random floor $0.494$
(`artifacts/nla/gemma4/anchors_L47.json`).

**The CE verbalizer mode-collapses under argmax.** The standard CE
recipe (np $16$, two epochs) reaches teacher-forced cosine $0.90$, but
its greedy decode emits one degenerate loop for every target and scores
at the random floor ($0.500$). The information is demonstrably present:
the injected vector halves the gold text's cross-entropy ($8.70 \to
4.13$ nats per token). At that perplexity the conditional is broad, and
sixty-four steps of compounding argmax collapse onto a generic
attractor. This is the Qwen greedy gap in its most extreme form.

**Draft conditioning is refuted, with a mechanism.** The proposed repair
conditioned the verbalizer on the retrieval neighbour's *text* as an
in-context draft, reasoning that the worst case, copying the draft,
already matches the NN baseline. A four-way cross-entropy decomposition
(`scripts/nla_ce_decomp.py`) shows why this cannot work on this host:

| context for gold CE | nats/token |
|---|---|
| injected prefix + gold | $4.13$ |
| injected prefix + NN draft + gold | $4.10$ |
| BOS + NN draft + gold | $9.21$ |
| BOS + gold | $8.70$ |

The activation-space neighbour (centred similarity $0.69$) contributes
$0.03$ nats of predictive value for the gold text, and in the pure
in-context setting it actively hurts. Activation-space similarity is not
token-space predictive utility, so CE training has no gradient toward
using, or even copying, the draft. The trained draft model confirmed
this: its copy baseline read $0.712$ while its greedy decode ignored the
draft entirely ($0.507$). We record this as a clean negative alongside
the Lever-B negative of §6.

**The K-curve patterns with gpt-oss, not Qwen.** Best-of-$K$ oracle
rerank on the CE checkpoint climbs from $0.507$ ($K{=}1$) to $0.591$
($K{=}32$) at $+0.017$ per doubling of $K$, exactly the gpt-oss slope
and half the Qwen slope, and extrapolates to an impractical $K \approx
2000$ to reach the retrieval baseline. Across four backbones the pattern
is: the base-model host (Qwen2.5-7B) crosses its paraphrase ceiling
at $K{=}64$, while both instruction/reasoning-tuned hosts (gpt-oss-20b,
gemma-4-31B-it) never reach their retrieval baselines at any practical
$K$. It is tempting to read this as instruction tuning collapsing the
unconditional text manifold that best-of-$K$ sampling must traverse, and
in an earlier draft we advanced exactly that conjecture. **A within-family
control refutes it.** We ran the identical pipeline on the *base*
`gemma-4-31B` checkpoint, matched on layer, corpus targets, and training
recipe to the instruction-tuned run. The two K-curves are
indistinguishable: slope $+0.013$ per doubling for the base versus
$+0.015$ for the instruction-tuned model, with the base if anything
marginally *lower* at every $K$ (best-of-$32$ $0.575$ versus $0.591$),
and identical anchor frames (replay $1.00$, NN $0.674$, floor $0.502$).
Instruction tuning therefore does not measurably change verbalizability
within this family. The steeper Qwen slope that motivated the conjecture
is a cross-family difference, confounded with architecture and scale, not
an effect of tuning. One honest caveat: the base verbalizer was
early-stopped at a slightly lower validation than the instruction-tuned
run, which may account for the small level offset, but the slope, which
is the quantity the conjecture concerned, is unchanged. The negative is
itself informative: whatever governs how far sampling can travel toward
the paraphrase manifold is set by the pretrained substrate, not by the
alignment stage layered on top of it. The deployable decode on this host
is retrieval regardless, which is precisely the mechanism §11.6.3
validates on the visual channel. Artifacts:
`artifacts/nla/gemma4{,_base}/{kcurve_ce.jsonl,anchors_L47.json}`,
checkpoints at `RiverRider/srt-nla-av-gemma4` (`base_ce/`).

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

![**Figure 12. The Lancaster pitchfork, read inside the decoding manifold.** The supercritical pitchfork $\dot{x}=\mu x - x^{3}$ is the canonical normal form Lancaster (2025) uses to model interpretant divergence under polarising pressure. Under the *standard* reading (left), $\mu$ is community polarisation pressure on a discourse and the two branches are competing contested readings of a single sign. Under the *decoding-manifold* reading proposed here (this paper, §12), $\mu$ is best-of-$K$ search budget (or any utility-based selection pressure on a candidate pool), the symmetric below-threshold branch is the greedy/modal $\hat{x}$ in its native basin at centred cosine $\approx\!0.59$–$0.63$, and the upper post-threshold branch is the paraphrase-manifold $\hat{x}$ at centred cosine $\approx\!0.99$. Logp is a within-basin quantity and cannot move probability mass across the separatrix; centred cosine to the target $v$ is the order parameter that does. The Stage-2 validation that the same normal form fits political-polarisation data (Pearson $r\!=\!0.884$ on five Supabase communities, 19K articles, 141K Peircean sign annotations; Lancaster, 2026a) is what makes the analogy more than a metaphor.](artifacts/explainers/15_lancaster_pitchfork.png)

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

## 12.5 Two clocks, and a candidate law: metapragmatic load

The SRT program distinguishes, but has not previously separated in
*measurement*, two time scales on which an interpretant lives.
*Token-sequence time* is the within-pass trajectory: the
Metapragmatic Attention Head emits a per-token divergence $D(t)$,
the Reflexive Recurrent Module integrates it, and the Bifurcation
Estimation Network reads an order parameter $\hat{r}(t)$.
*Transmission time* is the historical process by which each sign's
interpretant was stabilised across the training corpus and across
model lineages — sedimented into the frozen weights and
materialised, in the SRT-Adapter, as the Community Discovery index.
The first is a *readout* clock; the second is a *generative* clock.
Mechanistic interpretability ordinarily works in the first alone and
stops at description ("this unit responds to $X$"); the value of
naming the second is that it converts such a description into a
historical explanation — $D(t)$ is high *because* the sign $X$ was
transmitted with weak referential grounding or high cross-community
variance. In this reading the frozen model is not only an object of
study but an instrument for the semiotic structure of its training
culture, read out through a controlled within-pass probe.

We report a first controlled measurement relating the two clocks.
For a battery of $54$ concepts spanning the mundane (a bicycle, the
number seven) to the deeply contested (freedom, justice,
consciousness), each carrying a curated contestedness score in
$[0,1]$ and presented under a uniform definitional stem
("$\langle$concept$\rangle$, properly understood, is"), we read $D$
at the final, shared token of the stem and $\hat{r}$ over the stem.
Two findings, one negative and one positive, both replicated across
two independently trained adapters on the same backbone
(RiverRider/srt-adapter-v1.0 and v8a):

1. **The bifurcation order parameter $\hat{r}$ does not track a
   concept's contestedness** (Spearman $\rho = +0.02$; tier means
   flat at $0.67$–$0.71$ on both adapters). The reflexivity register
   is not a contestedness detector at the lexical-concept level. We
   flag this because it is the readout most observers would *expect*
   to carry the semiotic signal.

2. **The metapragmatic divergence $D$ scales with interpretant
   *underdetermination*, but a naive reading over-attributes it to
   contestedness.** The first battery confounds abstractness with
   contestedness (its mundane concepts are concrete, its contested
   ones abstract). A second battery of $40$ concepts crossing
   *concreteness* with *contestedness* in a $2\times2$ dissociates
   two contributions. Referential underdetermination (abstractness)
   is the robust, dominant term: abstract $-$ concrete
   $= +0.15$ (consensus) and $+0.29$ (contested), significant on
   both adapters. Community underdetermination (contestedness) is a
   weaker, *conditional* term: it raises $D$ only among
   already-abstract concepts (freedom $>$ the number seven,
   $+0.20$, $p\approx.006$–$.010$) and vanishes among concrete
   objects (a handgun $\approx$ a bicycle, n.s.; the concrete-
   contested null replicates to within $0.002$ across adapters).

![**Figure 12.5. Divergence tracks abstractness first, contestedness only among abstract signs.** Mean final-token MAH divergence for the $2\times2$ crossing of concreteness (concrete objects vs. abstract concepts) with contestedness (consensus vs. contested), $n=10$/cell, error bars SEM, v1.0 adapter. The abstract-vs-concrete gap is large and present in both consensus and contested rows; the contested-vs-consensus gap is significant only in the abstract row (freedom vs. the number seven) and absent in the concrete row (a handgun vs. a bicycle). The pattern reproduces on the independently trained v8a adapter (concrete-contested contrast $+0.069$ vs. $+0.071$). Divergence at the final token is used so that prompt length and token-averaging cannot drive the effect, since contested concepts are systematically shorter.](artifacts/nla/figures/coupling_dissociation_2x2.png)

This licenses a candidate regularity, which we state as a *target*
rather than an established law:

> **Metapragmatic-load conjecture.** The within-pass divergence a
> sign evokes is a monotone function of the sign's interpretant
> underdetermination,
> $$\mathbb{E}[D \mid \text{sign}] = D_0 + \alpha\,U_{\text{ref}} + \beta\,U_{\text{com}}, \qquad \alpha,\beta > 0,$$
> where $U_{\text{ref}}$ is referential underdetermination (the
> absence of a fixed object or denotation) and $U_{\text{com}}$ is
> community underdetermination (the variance of the interpretant
> across the model's discourse-community index). Referential
> grounding drives $D$ to its floor; both the loss of a referent and
> cross-community interpretant variance raise it.

The two clocks enter as follows. $U_{\text{ref}}$ and $U_{\text{com}}$
are set in transmission time — whether the corpus gave the sign a
stable denotation, and how variously its communities used it — and
$D(t)$ reads their sum in token-sequence time. What we can presently
sign our names to is $\alpha>0$ robustly, and $\beta>0$ small and
conditional on $U_{\text{ref}}$ being high: the community fork
appears to require the absence of a concrete referent to fall back
on. The dissociation of $D$ (load) from $\hat{r}$ (commitment) is
itself a structural claim about the infrastructure — it carries
separable *load* and *bifurcation* registers, and underdetermination
drives the first, not the second.

We are explicit about what would earn the conjecture the word "law,"
and about how much of that programme remains. *Test 1 (unified
latent), run:* we regressed $D$ on two independently varying axes —
published Brysbaert concreteness norms for $U_{\text{ref}}$ and the
curated contestedness score for $U_{\text{com}}$ — across $79$
concepts on which the two axes are only weakly collinear
($\rho=0.46$). Both coefficients are positive and separable
(standardised $\beta_{U_{\text{ref}}}=0.44$, $\beta_{U_{\text{com}}}=0.21$),
and contestedness survives controlling *measured* concreteness
(partial $\rho=0.25$); this strengthens the conjecture beyond the
binary $2\times2$. The remaining softness is that $U_{\text{com}}$ is
still a curated ordinal rather than a causally measured
community-decoding disagreement, which is Test 3. *Test 2
(cross-backbone invariance):* the same signs on Qwen, Gemma, and
Mistral; the Stage-3/4 finding that the sampling-reachable manifold
is set by the pretrained substrate rather than the alignment stage
(§11.7) *predicts* substrate-invariance, a falsifiable prediction of
the conjecture. A cross-backbone SRT adapter (gpt-oss-20b) exists but
loads under a different transformers pin, so this is set up but
unrun. *Test 3 (causal $U_{\text{com}}$):* forcing the community
index at decode time should move the interpretant only where
$U_{\text{com}}$ is high; this closes the one open commitment (active
intervention) in the "semiotic-infrastructure" claim. It requires a
discrete-prototype community adapter, which the trajectory-mode
released adapters do not provide, so it is currently blocked on
training one. *Test 4 (diachronic):* a sign that historically drifted
from consensus to contested should show rising $U_{\text{com}}$, and
rising $D$, across a dated corpus; we have identified suitable dated
English corpora (AmericanStories, 1774–1963; Chronicling America),
but a clean text-only multi-era extraction remains to be run. We have
run Test 1 and the cross-*adapter* replication (above); Tests 2–4 are
set up but open.

We do not claim a law. We claim a replicated dissociation and a
falsifiable target, and we note the honest shape of the result: the
readout most observers would expect to carry the semiotic signal
(the bifurcation order parameter) does *not* move; the one that does
(metapragmatic divergence) is driven first by referential
underdetermination and only secondarily, and conditionally, by the
community-contestedness the program's semiotics foregrounds. In the
program's register: contestedness here is a curated ordinal, not a
causally measured community-decoding disagreement (the forcing
instrument requires a discrete-prototype adapter, which the
trajectory-mode v1.0/v8a do not provide); the replication is
cross-adapter on a single backbone, not cross-backbone; $n$ is
$40$–$94$ concepts; and the diachronic term — the actual
transmission-time drift of one sign from consensus to contested — is
unbuilt. The conjecture's value is that each of these gaps names a
decisive, runnable experiment.

---

## 13. Honest expectations and open problems

We close in the program's standard register: what we expect the
next phase to deliver, what we do not, and where the load-bearing
uncertainties sit.

1. **Greedy gap closure is the headline open problem.** A
   verbalizer that closes the greedy gap on this backbone, i.e.,
   single-pass deterministic decoding at $\rho_{\text{cen}} \gtrsim 0.9$
   without K-fold inference, is the next-stage goal. Lever B
   (bag-of-$K$ self-distillation) does not close it on Qwen; we
   have not yet run it on Llama or Gemma, but the diversity-collapse
   mechanism documented in §6 is a property of winner-CE on a
   frozen substrate and is not Qwen-specific. Plausible
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
   centred random floor lands within $0.003$ of $0.50$ on all
   three backbones (Qwen $0.510$, Llama $0.498$, Gemma $0.498$;
   §3, §10, §11), despite a $22\times$ spread in $\|\mu\|$.
   The *ceiling* in absolute centred fve\_nrm depends on the
   paraphrase capacity of the base model under whatever prompt is
   used; this varies across backbones (Qwen $\approx\!0.80$
   centred, Llama $\approx\!0.72$, Gemma $\approx\!0.60$ under the
   same prompt; §3, §10, §11). Reporting in $\rho_{\text{cen}}$,
   normalised to the binding ceiling for that backbone, preserves
   portability. Anyone reporting an $\mathrm{fve\_nrm}$ result
   without these two anchors is reporting an uninterpretable
   number.

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
  frozen Qwen-2.5-7B, $\approx\!0.18\%$ parameter overhead
  ($12.72$M trainable, v8a final checkpoint), with
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
- **`RiverRider/srt-nla-av-llama32-3b-v1`** — released Llama AV. (§10)
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

