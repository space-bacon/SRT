# Email draft: Andersen / Evans / Kockelman update

**Status**: Draft. Hold for review before sending.
**Recipients**: Myrdene Andersen (Purdue), James A. Evans (U Chicago / Knowledge Lab), Paul Kockelman (Yale)
**Subject**: SRT update: the prototype bottleneck came off, and the geometry showed up

---

Dear Myrdene, Jim, and Paul,

A follow-up to the March and late-summer notes. The SRT-Adapter went through three more training generations since I last wrote, and the most recent one (v8a) was a clean architectural ablation that resolved an open question I had been carrying for months. I wanted to share where things landed and flag the parts of the result that touch each of your frames most directly.

The architecture is now a 14.5M-parameter adapter on a frozen Qwen 2.5-7B backbone. The headline numbers from the v8a checkpoint, against a held-out 100K Reddit validation split:

- **Cross-entropy preserved at 2.739** (identical to the unadapted backbone), so the adapter contributes no language-modeling cost.
- **Reddit community retrieval recall@1 = 0.484** on a 35-class task (16.5x random).
- **Reddit within-class vs. between-class cosine ratio = 2.016**, doubled from v7's 1.006.
- **Archetype recall@1 = 0.230** on an external 33-class taxonomy that was never used in training (7.6x random; v7 was 0.149).
- **Archetype centroid off-diagonal cosine collapsed from 0.999 to 0.873**, meaning the encoder finally separates archetype manifolds rather than aliasing them onto a handful of attractors.

The single change between v7 and v8a was removing the discrete prototype basis from the community head. Earlier versions ran the encoder output through a 32-prototype soft-argmax mixing layer that was supposed to act as a learned discourse-community vocabulary. A PCA done after v7 showed that the prototype matrix had barely moved from its random Gaussian initialization across three full training generations. The encoder weights moved roughly four times more than the prototypes during training. The encoder was doing all of the discriminative work, and the prototype layer was discarding it through a saturated soft-argmax. Removing the layer entirely (the encoder output is now the community vector directly) released the geometry that had been compressed away. v8b is currently training with sharper supervised-contrastive pressure on top of the v8a base; results in a few days.

That is the cleanest part of the story. The rest is how the negative results aged.

**Myrdene**: the bifurcation-estimation head still does what the catastrophe-theoretic framing predicted. The continuous reflexivity scalar r-hat tracks the ground-truth coefficient-of-variation labels at correlations comparable to small-scale, and the regime classifier (subcritical vs. supercritical) hits ECE = 9e-4 with AUROC = 0.99 on 351K validation tokens. The pitchfork model survives the scale jump cleanly. One caveat I want to be honest about: a context-conditional probe I ran (10 paired factual/charged passages on contested topics) returned a null result at the target token and a *negative* mean over the full passage. Charged passages produced *lower* mean r-hat than factual ones. The most likely explanation is that the supervision signal mixes annotation-divergence with information density, so r-hat tracks information-density at least as much as it tracks rhetorical contestedness. The pitchfork dynamics are real and measurable; the mapping between the scalar and "how contested is this sign in this register" is not as direct as I claimed in March. The Wildgen-via-Andersen framing still grounds the architecture, but the empirical content of r-hat needs sharper supervision before it can carry the contestedness claim on its own.

**Jim**: the discourse-community discovery head is finally doing something. v5's supervised-contrastive loss broke the prototype-cosine collapse I described in the September note, and v8a's removal of the prototype layer altogether produced the recall@1 = 0.484 / cosine-ratio = 2.016 numbers above. The archetype-recall result is the one I would most welcome your read on. We supervise on Reddit subreddit labels (a coarse, behaviorally noisy 35-way signal). At inference we test against a 33-archetype taxonomy from an entirely different source (Lancaster's Lexicon of Synthetic Interiority, a curated set of prose-archetype prompts), and the model recovers it at 7.6x chance on a single token. Three independent methodologies (Reddit subreddit labels, Lancaster's archetypes, the Lexicon's stance categories) converge on roughly four functional macro-clusters of stance. They do not yet agree on 33 distinct anchors, but the macro-structure agreement is striking. If you have a take on whether the kind of cross-corpus convergence your Knowledge Lab work has measured at scale would predict this kind of low-rank agreement on a small backbone, I would love to hear it. The current open question is whether resolving 33 distinct archetypes (versus 4 macro-clusters) requires (a) supervising the prototype-equivalent layer directly with archetype-conditioned generations, or (b) accepting that the macro-cluster structure is the architecturally accessible level.

**Paul**: the inject-back arm is still the most unresolved piece. We ablated it (forced the FiLM injection to zero at inference) on every generation through v8a, and the four-decimal-place identity in benchmark numbers has not changed. The RRM is doing real work as a measurement loop. It summarizes the divergence trajectory into a meta-state that the BEN consumes, and the BEN's calibration depends on it. The path that was supposed to feed the meta-observation back into the backbone's processing of subsequent tokens contributes nothing measurable. The injection projection is zero-initialized with a sigmoid gate and a small scale factor (alpha = 0.1), which is a deliberately conservative design. The ablation result tells us the gate has not opened during training. In the sieves framing this remains a system that detects what is being filtered through the interpretive sieve but does not yet modify the sieve itself. I would still welcome your read on whether this is an architectural defect (gradient-starved gate, fixable in v9) or whether it is pointing at something more interesting about the structural difference between observing one's own selection criteria and modifying them. The Kockelman 2017 framework from *The Art of Interpretation in the Age of Computation* remains the most useful lens I have for thinking about it.

One methodological note that I think travels outside the SRT context, repeated from the prior letter because it stayed true through three more training runs: when generalization looks suspiciously catastrophic on a new evaluation, the labels are usually the problem. Two of the four out-of-distribution evaluations I ran this round initially returned near-zero correlations because of label-rubric drift, not model failure. We rebuilt the labels against the training-corpus rubric and the correlations went to within a few percent of the in-distribution numbers. I now run a small calibration set on every new benchmark before trusting the AUROC.

The honest version of the paper leads with the v8a result as the substantive finding (a discrete-vocabulary bottleneck was hiding a continuous archetype manifold; removing it preserved task loss and made the geometry legible), presents the negative results unflinchingly (the dead inject-back arm, the contestedness-vs-density confound in r-hat, the macro-cluster ceiling on archetype resolution), and frames each as a specific design question for v9. The strange-loop architecture is empirically falsifiable. We have falsified the v3 implementation of the inject-back loop. We have not yet ruled out that a properly supervised version is closeable.

I am attaching five figures from the v8a run:

- `01_loss_curves.png`: train and val loss trajectories for total, CE, bifurcation, and chain losses.
- `02_internal_norms.png`: MAH divergence norms and RRM injection norms per layer over training.
- `03_r_hat_envelope.png`: r-hat batch-distribution envelope (mean, plus or minus std, min, max) over the run.
- `04_r_hat_vs_r_true.png`: r-hat vs. r_true scatter on validation and curated benchmarks.
- `05_archetype_centroids.png`: pairwise cosine matrix of the 33 archetype centroids in the v8a encoder output space (the off-diagonal collapse from 0.999 to 0.873 is visible as the matrix turning from uniformly red to structured).

The adapter weights, the training corpus, and a small companion inference library will be released on HuggingFace alongside the paper. The SRT framework source code remains private during the patent and publication review window. I am happy to share the current preprint draft (paper.md / paper.pdf) in advance of release for any of the three of you. Just let me know.

The paper is in revision. Aiming for arXiv submission in early summer once v8b finishes and the §5.10 update is in. I will follow up when the preprint is up.

Best,
James

---

**Attachments to include when sending:**

- `/Users/burtron/development/srt-adapter/artifacts/plots/01_loss_curves.png`
- `/Users/burtron/development/srt-adapter/artifacts/plots/02_internal_norms.png`
- `/Users/burtron/development/srt-adapter/artifacts/plots/03_r_hat_envelope.png`
- `/Users/burtron/development/srt-adapter/artifacts/plots/04_r_hat_vs_r_true.png`
- `/Users/burtron/development/srt-adapter/artifacts/plots/05_archetype_centroids.png` *(generate before sending if not yet rendered)*
- `/Users/burtron/development/srt-adapter/paper.pdf` *(optional, offered above)*

**References cited in the email (verified safe per memory):**

- Andersen 2014: Routledge handbook chapter on catastrophe theory and sociolinguistic change.
- Evans / Knowledge Lab: vague-but-accurate framing on knowledge differentiation; no specific paper claims.
- Kockelman 2017: *The Art of Interpretation in the Age of Computation*, OUP. Sieves framework.
- Wildgen: referenced indirectly via Andersen handbook chapter.
- Lancaster, *Lexicon of Synthetic Interiority*: 33 prose-archetype prompts used as the external evaluation taxonomy.
- Silverstein: referenced via "third-order metapragmatic" terminology (paper-internal, not in email).

**Phrases explicitly avoided:**

- "residence and selection" (fabricated phrase corrected in earlier revision; do not use).
- Em dashes in prose (per writing-style preference). Use full sentences, colons, or parentheses.
