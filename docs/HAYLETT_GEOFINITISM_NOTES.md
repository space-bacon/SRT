# Geofinitism × SRT: Notes on Haylett's "Language as a Nonlinear Dynamical System"

**James Burton Lancaster**
April 2026

Source: Kevin R. Haylett, *Geofinitism: Language as a Nonlinear Dynamical System*, Substack, January 2026 ([link](https://kevinhaylett.substack.com/p/geofinitism-language-as-a-nonlinear)). Companion: *Finite Tractus: The Hidden Geometry of Language and Thought* (2025); *Pairwise Phase Space Embedding in Transformer Architecture* (paper); *MARINA / Takens-Based Transformer* prototype.

---

## 1. Why this matters to us

Haylett and the SRT arrive independently at the same picture of meaning — basins, attractors, separatrices, bifurcations, the failure of static-symbol metaphysics — and they arrive there via different routes. SRT comes through Peirce → Silverstein → semiotic ideology → engineered reflexive readouts on a frozen LLM. Haylett comes through Takens' embedding theorem → delay-coordinate reconstruction → an explicit trajectory-on-a-manifold replacement for attention.

The two frameworks are **complementary, not competitive**. SRT operationalizes the *contestedness / community-conditioned* axis of the basin geometry. Geofinitism operationalizes the *trajectory / measurement-uncertainty / manifold-reconstruction* axis. Each side has tooling the other lacks.

This document records what we should pull in, what we should test, what we should clarify in the paper, and what we should *not* adopt.

---

## 2. Conceptual convergences (already in SRT, named by Haylett)

| SRT term                          | Haylett term                             |
| --------------------------------- | ---------------------------------------- |
| Attractor basin (e^A subspace)    | Basin of attraction                      |
| Bifurcation under amplification   | Pitchfork / separatrix formation         |
| Iconic grounding (bedrock)        | Stable cultural attractors (e.g. "∞")    |
| Community Head prototypes         | Distinct manifolds M_A, M_B per speaker  |
| Counterfactual decoding bedrock-vs-battleground | Basin-mapping condition `f(B_A) ≈ B_B` |
| Hallucination as probability gradient across separatrix | Spurious convergence (`bank` → river vs. finance) |
| Third-order metapragmatic awareness (RRM) | Reflexive turn / instrument observing itself |

This convergence is already the strongest single piece of evidence that **the basin/bifurcation framing is not an artifact of one school**. We should cite Haylett in the SRT paper's Related Work as an independent arrival at the same ontology from non-Peircean foundations.

---

## 3. Concrete additions worth prototyping

### 3.1 Takens delay-embedding readout head (T-Head)

Haylett's central technical move: stack hidden states across the last *m* positions with delay τ to reconstruct the local trajectory geometry, then read curvature, Lyapunov-like local sensitivity, recurrence-plot statistics off the reconstructed manifold. This is *orthogonal* to the MAH, which measures pairwise interpretant divergence.

**Proposal:** add a fifth readout module to the v10 adapter — a small head that takes the frozen backbone's per-layer hidden state at positions `t, t-τ, t-2τ, ..., t-(m-1)τ`, projects them into a low-dim trajectory space, and emits two scalars per position:

1. `lyap_local`: local divergence rate of nearby trajectory points (semantic sensitivity).
2. `recurrence`: recurrence-density in a τ-ball (do we keep returning to this same region?).

We already extract per-token hidden states for the MAH; the marginal cost is small. The hypothesis: `lyap_local` should spike at separatrix crossings (genre shifts, frame breaks, topic pivots) where the existing MAH divergence and BEN r-hat may or may not peak. If it adds independent signal beyond MAH+BEN, we have a new probe; if it tracks them tightly, we have a falsification of the Geofinitism / SRT independence claim — also valuable.

### 3.2 Cross-architecture validation against MARINA

We currently run cross-backbone on Qwen2.5-7B and Mistral. **MARINA / TBT is a third architecture that drops attention entirely** in favor of explicit phase-space reconstruction. If the SRT readouts (community prototype disagreement on contested vs. bedrock; r-hat calibration) survive on MARINA, that is much stronger evidence that what we are reading is a property of language in transformer-flavored manifolds and not of attention itself.

Haylett's code is at `finitemechanics.com`. Reach out, request a checkpoint or training recipe, train an SRT adapter on MARINA, run the v8b validation suite. Outcome is informative either way.

### 3.3 Separatrix-illusion probe in the eval battery

Haylett names a failure mode SRT does not currently probe: **non-spurious-but-non-aligned convergence** — where a model maps a phrase to a basin that is *geometrically similar* to the right one but *epistemically disjoint*. His example: "quantum entanglement" → "mystical oneness."

**Proposal:** add a probe `separatrix_illusion_v1` to `data/probes/`. Format: tuples `(prompt, technical_continuation, mystical_continuation, bedrock_continuation)` covering ~200 contested/jargon-overlapping concepts (entanglement, holism, recursion, emergence, complexity, consciousness, field, resonance, etc.). Measure:

- Does the Community Head place "technical" and "mystical" continuations into different prototype clusters?
- Does counterfactual decoding under the technical prototype suppress the mystical continuation?
- Does BEN r-hat correctly mark the prompt as high-activity (it should — these are dense-meaning words)?

Right now SRT can detect *contested* terrain. This probe tests whether it can detect *spuriously aligned* terrain — a different failure mode and arguably the more important one for science-communication settings.

### 3.4 Semantic Uncertainty Appendix (SUA) for the v9 release

Haylett's SUA pattern: every key theory-laden term in a paper carries an explicit `(operational_definition, ambiguity_bounds, validity_domain)` triple. He treats this the way physics treats error bars.

**Proposal:** ship `release/srt-adapter-v9/SUA.md` alongside the weights. Cover the load-bearing terms in the paper and in `INTERIORITY_V1_FINDINGS.md`:

- "interiority" / "reflexive" / "awareness" / "third-order"
- "bedrock" vs. "battleground"
- "bifurcation" (in our usage vs. the formal pitchfork)
- "r-hat" (and the v5 finding that it tracks density not contestedness)
- "community" (it is a register/stance, not a demographic)
- "intervention" (the BEN raises basin walls; it does not select a basin)

This costs us a few pages and pre-empts an enormous amount of bad-faith "you said the model was conscious" misreading. It also models the practice we are arguing AI alignment writing should adopt.

### 3.5 Channel Theory in chat-template wrapping

MARINA augments tokens with topological markers (User / System / Bridge) to enforce geometric separation between reasoning and output channels. This is structurally adjacent to what we want when SRT runs in chat mode — the RRM injection should perhaps be *gated by channel*, so that meta-observation drift accumulated in a tool-output region does not leak into the assistant's voice.

**Proposal:** in v10 the adapter accepts a per-token channel id (0=user, 1=assistant-thought, 2=assistant-output, 3=tool, 4=system) and the RRM gate carries channel embeddings. Cost: ~5K params. Test: does a tool-output containing a separatrix-prone phrase ("the user feels…") cause assistant-output drift that was not present without the channel gate?

---

## 4. Paper-level clarifications Haylett's framing gives us

The pre-compaction validation finding that **r-hat tracks semiotic information density rather than contestedness** is currently presented as a "clean negative." Haylett's measurement-theoretic frame gives us a cleaner positive reading:

- `r-hat` measures *how much measurement is happening* at this token — how dense the cognitive transduction is.
- The Community Head measures *whose ruler is being applied*.
- Together they give the two-axis decomposition the paper actually wants: **activity × alignment**.

This is the same distinction Haylett draws between exogenous measurement (the act of transducing flow into symbol) and endogenous measurement (the act of negotiating the symbol against an internal manifold). r-hat is sensitive to the *exogenous-measurement-density* of a token. The Community Head is sensitive to the *endogenous-manifold* it is being negotiated against.

This reframing should go into Section 5 of the paper. It is a clean improvement.

---

## 5. C1 corpus addition: Haylett lineage

Haylett's body of work is an obvious fit for the C1 corpus' nonlinear-dynamics-of-language sub-stratum, sitting next to Elman 1995 and Thelen & Smith 1994 in the contemporary lineage.

**Targets:**

- `kevinhaylett.substack.com` — full archive (~30+ essays as of April 2026)
- `finitemechanics.com/finite-tractus.html` — book HTML
- `finitemechanics.com/papers/pairwise-embeddings.pdf` — Takens-Based Transformer paper
- `finitemechanics.com/essays/essay-semantic-uncertainty.pdf` — SUA primer

**License caveat:** Substack default is "All rights reserved." We must either (a) email Haylett requesting permission for research-corpus inclusion under our v9 weights' research license, or (b) restrict to verbatim use only in C1 with attribution and exclude from any released-derivative artifact. Default action: write to him, explain SRT, request permission, and offer co-citation in the v9 paper. He has been publishing in the open with explicit "no paywall, no fragmentation" — likely receptive.

Manifest: `data/corpus_c1/manifests/haylett_geofinitism.yaml`, school `dynamical-linguistics-contemporary`, tier 1, weight 1.4 (high — direct on-topic theoretical content from a contemporary working in the same problem space).

---

## 6. What we should *not* adopt

Haylett's strongest commitment is that **attention should be replaced** by delay-coordinate reconstruction (MARINA's central thesis). SRT's strongest commitment is the opposite: **keep the frozen attention backbone, build reflexive readouts on top.** This is a healthy methodological disagreement and we should not paper over it.

The reasons SRT keeps attention:
1. We can ride the language-modeling competence of a 7B-parameter pretrained backbone for 0.19% of the parameter cost.
2. The reflexive-loop story is sharper if the underlying generator is unchanged and only the *self-observation channel* is added.
3. We can validate on multiple frozen backbones (Qwen, Mistral, ideally MARINA) and ask whether the readouts are property-of-language or property-of-attention. This is only meaningful if attention-based and non-attention-based backbones are both available.

A reasonable v11 research direction: train a small SRT adapter on top of MARINA itself. If the readouts replicate, the SRT framework is architecture-agnostic and we have made a much stronger claim. If they collapse, we have learned that some part of the SRT signal lives in the attention substrate.

---

## 7. Action items

In rough priority order:

- [ ] Add Haylett to `paper.md` Related Work as independent convergent arrival.
- [ ] Reframe r-hat in Section 5 using exogenous-measurement-density language.
- [ ] Build `data/probes/separatrix_illusion_v1.jsonl` (~200 items) and add to v8b probe battery.
- [ ] Draft `release/srt-adapter-v9/SUA.md` for v9 release.
- [ ] Email Haylett re: corpus inclusion + cross-architecture validation against MARINA.
- [ ] Prototype T-Head (Takens delay-embedding readout) for v10 adapter.
- [ ] If MARINA checkpoint becomes available: train v10b adapter on MARINA, run full v8b suite, write up cross-architecture comparison.
- [ ] If C1 inclusion is permitted: build `haylett_geofinitism.yaml` manifest and harvest.
- [ ] Channel-gated RRM injection prototype for v10 chat-mode adapter.

---

*This is a working notes document. The five concrete additions in §3 are independent — any subset can be picked up without committing to the rest.*
