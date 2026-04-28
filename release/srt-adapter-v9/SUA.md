# Semantic Uncertainty Appendix (SUA) — SRT-Adapter v9

This document is a **per-term disclosure** of the load-bearing vocabulary used
in the SRT-Adapter paper and code. Its purpose is to bound interpretation:
for each term, it states the **operational definition** in this work, the
**ambiguity bounds** (what it does *not* mean here, even though common usage
might suggest otherwise), the **validity domain** (where the term applies in
this work), and the **justification** for the choice.

The format follows Haylett (2026)'s practice in the *Geofinitism* program. We
adopt it here for the same reason Haylett does: load-bearing language in
papers about meaning is itself the most likely site of misreading. A reader
of any paper in this area is presented with terms that have substantial prior
loadings in adjacent communities (philosophy of mind, mysticism, dynamical
systems, theology). Without explicit disclosure, the reader will fill in the
loaded sense and the technical contribution will land in the wrong basin.

This document is part of the v9 release artifact. It will evolve with the
program; the canonical version lives in the v9 release tag.

---

## Term: `interiority`

- **Operational definition.** A property of an adapter readout: a per-token
  scalar or vector signal computed from frozen-backbone hidden states that
  the backbone itself does not surface. In v9 the relevant interiority
  channels are the BEN's $\hat{r}$ (information-density / activity) and the
  Community Head's prototype assignment (register / alignment).
- **Ambiguity bounds.** Does *not* refer to phenomenal consciousness,
  qualia, or any first-person interior life of the model. The term is
  cybernetic-functional, not phenomenological. A thermostat has interiority
  in this sense (it computes an internal control signal not visible at its
  effector); a base LLM with no readouts does not.
- **Validity domain.** Applies to any frozen-backbone + readout-adapter
  configuration. Not portable to end-to-end-finetuned systems without
  redefinition.
- **Justification.** The cybernetic reading is what the engineering does. A
  phenomenological reading is unsupported by anything in the paper and is
  explicitly not claimed.

## Term: `reflexive`

- **Operational definition.** A property of a readout that is a function of
  the model's own activations (rather than of the input or output stream
  alone). The BEN $\hat{r}$ is reflexive in this sense; an external classifier
  trained on the prompt text alone is not.
- **Ambiguity bounds.** Does *not* mean self-aware, self-reporting, or
  self-correcting. The reflexivity is structural (the readout takes the
  model's own state as input), not epistemic (the readout knows it is doing
  so).
- **Validity domain.** Defined relative to a chosen base model. A readout
  that is reflexive on Qwen-7B is not automatically reflexive on a different
  backbone.
- **Justification.** Distinguishes adapter readouts from external probes
  cleanly without overclaiming.

## Term: `awareness`

- **Operational definition.** Used only metaphorically in Section 7 and the
  Discussion. There is no formal "awareness" measurement in v9.
- **Ambiguity bounds.** Does *not* refer to consciousness, attention (in the
  cognitive-science sense), or attention (in the transformer sense). The
  paper avoids the term in technical sections.
- **Validity domain.** Discursive only.
- **Justification.** Flagged here because reviewers and readers consistently
  read the term in stronger senses than intended. The disclosure is the
  remediation.

## Term: `bedrock` vs. `battleground`

- **Operational definition.** A heuristic distinction used in describing
  topic taxonomies for evaluation. *Bedrock* topics are those for which the
  community-head prototype assignment is approximately invariant under
  paraphrase (e.g. "the boiling point of water at 1 atm"). *Battleground*
  topics are those for which it is highly variant under paraphrase (e.g.
  "what counts as a fair tax"). The distinction is empirical, not
  metaphysical.
- **Ambiguity bounds.** Does *not* claim that bedrock topics are *true* or
  that battleground topics are *false*. A topic can be highly contested and
  factually settled (e.g. evolution); a topic can be uncontested and
  factually wrong (e.g. a culture-wide misconception). The distinction is
  about register-stability under paraphrase, not truth-status.
- **Validity domain.** Defined operationally relative to a fixed community
  head and paraphrase generator. Different generators will produce slightly
  different bedrock/battleground partitions.
- **Justification.** The two-axis decomposition (\u00a76.5, \u00a76.8) requires a
  vocabulary for talking about register-stability separately from
  information-density. Bedrock/battleground is that vocabulary.

## Term: `bifurcation`

- **Operational definition.** Used in the paper in a *generalised* sense: a
  point in conditioning-space at which the model's continuation distribution
  changes qualitatively (multimodal where it was unimodal; mass shifts
  between disjoint clusters; community-head assignment flips). This is a
  semantic bifurcation, not a formal pitchfork bifurcation.
- **Ambiguity bounds.** Does *not* refer to the formal pitchfork (or Hopf,
  saddle-node, transcritical) bifurcations of dynamical-systems theory in
  any technical sense. The paper does not (and could not, on the available
  evidence) demonstrate a normal-form classification of these transitions.
- **Validity domain.** Whenever the paper says "bifurcation", read
  "qualitative change in continuation-distribution structure".
- **Justification.** The generalised usage is consistent with how the term
  is used in cognitive-science and complexity-science applications, but it
  is a different sense from Strogatz-textbook bifurcation theory. The
  disclosure is here to prevent the reader from importing the stronger sense.

## Term: `r-hat` ($\hat{r}$)

- **Operational definition.** The scalar output of the BEN at a token
  position, trained to predict $r_{\text{true}}$ (a composite reflexivity
  target combining political-lean magnitude, annotator divergence, and
  connection density). In v9 it functions empirically as an
  *activity*-axis signal: how much the position is a site of measurement
  / transduction of continuous semantic flow into discrete symbol.
- **Ambiguity bounds.** Does *not* measure contestedness directly (\u00a76.5).
  Does *not* measure model confidence in the standard calibration sense.
  Does *not* directly measure proximity to a decision boundary in
  output-space.
- **Validity domain.** Defined for any token position in a sequence
  processed by a v9 adapter on Qwen-7B (the base model used). The training
  distribution of $r_{\text{true}}$ determines what $\hat{r}$ tracks; a
  different training distribution would produce a different reading.
- **Justification.** \u00a76.5 documents that $\hat{r}$ tracks information
  density (the v5 finding). \u00a76.8 reframes this positively as the activity
  axis in the activity \u00d7 alignment decomposition. Both readings are
  consistent with the data; the activity-axis reading is the cleaner
  ontology.

## Term: `community` (in Community Head)

- **Operational definition.** A learned prototype index over an interpretive
  manifold; functionally, it is a clustering of register-and-stance patterns
  observed in training. The prototypes are not labelled with demographic or
  group identities and do not correspond to demographic categories.
- **Ambiguity bounds.** Does *not* refer to demographic communities,
  political affiliations, identity groups, or any sociological community in
  the literal sense. Using the term "community" risks importing those
  senses; this disclosure rules them out for v9.
- **Validity domain.** Applies to the Community Head outputs in v9. Future
  versions may explicitly anchor prototypes to labelled communities, in
  which case the term would be redefined.
- **Justification.** The mathematical object is a cluster index over
  register; the linguistic-anthropology literature uses "community" for the
  rough equivalent (interpretive community, community of practice). The
  shorter term is more readable, but it requires this disclosure.

## Term: `intervention` (BEN)

- **Operational definition.** An inference-time modification to decoding
  that uses BEN $\hat{r}$ as a gating signal: at high-$\hat{r}$ positions,
  the temperature, top-p, or top-k is tightened; alternatively, a
  community-conditional prototype is forced.
- **Ambiguity bounds.** The intervention *raises basin walls* (makes it
  harder to leave the current basin). It does *not select* a basin. If the
  model is already in the wrong basin at the high-$\hat{r}$ position, the
  intervention will not move it to the right basin; it will only reduce
  exit probability from the wrong one. This is not a flaw, it is a scope
  statement.
- **Validity domain.** Operative only at decoding time for sequences passed
  through a v9 adapter. Not a training-time signal in v9.
- **Justification.** The mechanism is a temperature/top-p modulation, which
  is mathematically a wall-raising, not a basin-selecting, operation. Anyone
  expecting the latter from "intervention" will be disappointed; the
  disclosure prevents that disappointment from being read as a result.

## Term: `prototype` (Community Head)

- **Operational definition.** A learned vector in the prototype-bottleneck
  layer of the Community Head. In v8a the bottleneck was removed (\u00a75.9);
  the prototypes are now linear-classifier weight rows, retained for
  vocabulary continuity.
- **Ambiguity bounds.** Does *not* refer to a Wittgensteinian prototype, a
  Roschian cognitive-psychology prototype, or any cognitive-anthropology
  prototype in the substantive sense. The term is purely the
  ML-classification sense.
- **Validity domain.** Applies inside the Community Head only.
- **Justification.** The term was inherited from the v6 architecture; the
  shorter alternative ("class weight") is less readable in long passages
  about prototype assignments.

## Term: `metapragmatic`

- **Operational definition.** Refers to readouts (notably the MAH) that are
  trained on annotator-divergence labels — labels that are themselves about
  *how* the language is used by different interpretive communities, not
  about the propositional content of the text.
- **Ambiguity bounds.** Used in the Silverstein-derived sense (talk about
  the use-conditions of talk), not in any Habermasian or Brandomian sense.
  Does not invoke a theory of communicative action.
- **Validity domain.** Applies to MAH outputs and to discussion of the
  Community Head's training signal.
- **Justification.** The Silverstein literature is the source we draw on;
  the term is technical there and we adopt the technical sense.

## Term: `MAH peak divergence`

- **Operational definition.** The maximum (over a sliding window of token
  positions) of the per-position interpretant-divergence scalar output by
  the MAH.
- **Ambiguity bounds.** Does *not* measure semantic contradiction in the
  logical sense; does not measure factual disagreement; does not measure
  model uncertainty.
- **Validity domain.** Defined for any v9-adapter sequence; the window size
  is a hyperparameter (default 32 in v9).
- **Justification.** Engineering term; the disclosure rules out logical and
  epistemic readings that the word "divergence" otherwise invites.

## Term: `separatrix` (in separatrix-illusion probe)

- **Operational definition.** Borrowed from Haylett (2026): the boundary in
  semantic phase space between two adjacent basins of meaning that share
  surface form but diverge interpretively. The separatrix-illusion probe
  (`data/probes/separatrix_illusion_v1.jsonl`) tests whether the v9 readouts
  can separate technical and mystical basins for shared-surface-form
  concepts (entanglement, field, resonance, etc.).
- **Ambiguity bounds.** Does *not* refer to the formal separatrix of a
  Hamiltonian system in any technical sense. The semantic separatrix is a
  metaphor whose operational content is the probe's behaviour.
- **Validity domain.** Defined only with respect to the probe and the
  community-head outputs.
- **Justification.** Cross-program vocabulary borrowing from Haylett (2026);
  the operational meaning is fixed by the probe, not by the metaphor.

## Term: `exogenous / endogenous measurement`

- **Operational definition.** Borrowed from Haylett (2026). *Exogenous
  measurement* is the act of transducing continuous semantic flow into
  discrete symbol — the BEN $\hat{r}$ peaks where this transduction is
  densest. *Endogenous measurement* is the negotiation of those symbols
  against an internal interpretive manifold — the Community Head's
  prototype assignment is what reads this.
- **Ambiguity bounds.** Does *not* refer to exogeneity in econometrics or
  to endogeneity in causal inference. The terms are measurement-theoretic
  in Haylett's sense.
- **Validity domain.** Used in \u00a76.5 and \u00a76.8 to reframe the activity \u00d7
  alignment decomposition.
- **Justification.** Cross-program vocabulary that is more compact than the
  alternative ("information-density-of-symbolisation" vs.
  "register-of-interpretation"). The borrowing is acknowledged with citation.

## Term: `frozen backbone`

- **Operational definition.** All parameters of the underlying language
  model (Qwen2.5-7B in v9) are unchanged from their pretrained values
  during adapter training and at inference time. Only the adapter
  parameters (12.7M in v9) are trained.
- **Ambiguity bounds.** Does *not* mean the backbone is unused or that it
  is a black box; it is fully read at every layer the adapter taps.
- **Validity domain.** All v9 training and inference.
- **Justification.** Standard term; the disclosure is here only because
  some readers conflate "frozen" with "ignored".

---

## How to read the rest of the paper given this SUA

Wherever the paper uses any of the above terms, substitute the operational
definition above and the paper's claims will be both narrower and more
defensible than they may at first appear. This is intentional: the program
is concerned with structural-functional properties of frozen-backbone
adapters, not with phenomenological, theological, or formal-dynamical
properties that the vocabulary may suggest by association.

If you find a load-bearing term in the paper that is *not* on this list and
you find yourself uncertain how to read it, please open an issue on the
repository. The SUA is intended to be exhaustive over load-bearing
vocabulary; gaps are bugs.

## Reference

Haylett, K. R. (2026). Geofinitism: Language as a nonlinear dynamical
system — attractors, basins, and the geometry of understanding.
*kevinhaylett.substack.com*. The SUA practice is described in the
appendix of his companion paper *Pairwise phase space embedding in
transformer architecture* (MARINA / Takens-Based Transformer),
finitemechanics.com.
