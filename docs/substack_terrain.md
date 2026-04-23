# Mapping the Terrain of Meaning: What the SRT Actually Does

**James Burton Lancaster**

April 2026

---

Imagine you are standing on a hillside. The ground beneath you feels solid. You can see where you are. You can see where to walk. The slope is gentle, the valley is broad, and every path leads back to the same low point: a shared understanding, a common interpretation, a place where "freedom" means roughly the same thing to everyone within earshot.

Now imagine the hill is changing beneath you. Not eroding. *Reshaping.* The valley floor flattens. The slope shallows. The ground that used to pull your footstep back toward the center stops pulling. You are standing on what feels like a plateau, but it is not stable rest. It is the pause before the split. One more degree of amplification, one more algorithmic nudge, one more curated feed cycle, and the plateau fractures. Where there was one valley there are now two, separated by a ridge you cannot cross without climbing. Each well deepens itself: the interpretants generated inside it reinforce the basin, deepen the walls, and make the other valley invisible. This is the terrain of meaning in 2026.

Every language model is trained on the *output* of this terrain. Text written from inside the wells after the split has already happened, without any representation of the landscape that produced it. A model trained on web corpora learns what conservative discourse sounds like and what progressive discourse sounds like, but it does not learn that these are two basins of a single bifurcated system, that the basin walls are self reinforcing, that the split was a phase transition rather than a gradual divergence, or that the flat plateau preceding the split emits detectable warning signatures.

It learns what r/AskHistorians sounds like and what r/AmItheAsshole sounds like. It learns what conservative discourse sounds like and what progressive discourse sounds like. But it does not learn that these are distinct basins, whether political, cultural, or subcultural, each with their own self reinforcing boundaries, nor does it see the underlying landscape that produces them.

The SRT is an instrument designed to read the shape of this ground.

Not the content of the wells. Not the politics. Not the opinions. The *topology*: the curvature, the gradient, the phase structure of meaning itself. Here is how it works.

---

## 0. The Instrument

A note on what the SRT is, materially, in April 2026.

The architecture was first sketched as a standalone transformer trained from scratch with the semiotic structure baked into its embeddings. The current realization is leaner and, we think, more honest. We freeze a strong existing language model (Qwen2.5-7B) and bolt the SRT modules on as an adapter, 14.6 million trainable parameters that sit alongside the 7.6 billion frozen ones, about 0.19 percent of the backbone. The backbone keeps its language modeling intact. The adapter learns to read curvature in the representations the backbone is already producing.

This matters for what follows. The four subspaces and four modules described below are not built from scratch into a fresh embedding layer. They are *extracted from* and *injected back into* a frozen substrate. The instrument does not replace the existing representation of language. It learns to read what is already latent there, and to modulate it lightly when the readings warrant.

Read the rest of this essay with that frame in mind. When the original theory says "the embedding has four subspaces," the realized adapter is closer to "we discover four subspaces inside the frozen embedding the backbone gives us." When the theory says "the BEN injects a modulation vector," the realized adapter does inject, but only at three specific layers, with gates that begin at zero and open only as the system learns to trust its own self observation.

The architecture below is the instrument. The realization is the adapter.

---

## I. The Semiotic Embedding Layer: Unpacking the Ground Beneath the Sign

A standard transformer gives each token a single embedding vector, one point in a high dimensional space, optimized for next token prediction. That point must simultaneously encode what the token *is* (its surface form), what it *refers to* (its object), what it *means to a particular community* (its interpretant), and which gravitational well it currently sits in (its attractor basin). These are four fundamentally different things stuffed into one vector. The result is a representation that captures the average and represents no one.

Peirce understood this 140 years ago. A sign is not a dyad, a word pointing to a thing. It is a triad: a *representamen* (the sign vehicle), an *object* (what the sign is about), and an *interpretant* (the effect the sign produces in the interpreter, which is itself a new sign, triggering the next link in the chain). The interpretant is where meaning lives. And meaning is always *someone's* meaning, produced in a specific interpretive community, conditioned by that community's enregistered associations, semiotic ideologies, and history of circulation.

The Semiotic Embedding Layer decomposes each token into four orthogonal subspaces:

**e^R, the representamen.** What the sign looks like. The phonological form, the distributional neighborhood, the syntactic profile. "Freedom" has the same representamen whether spoken by a libertarian or a liberation theologian. This is the coordinate system of the terrain, the map grid. It tells you *where* you are standing, not what the ground is doing.

**e^O, the object.** What the sign is about, including the iconic grounding subspace, the bedrock. This is where the bouba/kiki effect lives: the non arbitrary cross modal correspondences that chicks share with humans, calibrated by prenatal acoustic environments 310 million years before either species had a word for anything. The iconic grounding components are the parts of the terrain that *don't move under bifurcation*. When the valley splits, the bedrock stays. When convention fractures across communities, the fact that "sharp" sounds angular persists because it was never a convention to begin with. It was calibrated in the egg, in the uterus, in the low pass filter that is every embryo's first sensory environment. This is the floor beneath the floor, the attractor anchors that prevent meaning from floating free.

**e^I, the interpretant.** What the sign *does* to this community. This is the elevation map, and it is different for different communities standing on the same grid coordinates. "Freedom" at position (x, y) on the representamen grid has one elevation for Community A (individual autonomy, Second Amendment, shibboleth of sovereign selfhood) and a different elevation for Community B (bodily autonomy, reproductive rights, resistance to state control). The interpretant subspace is *community conditioned*: it is computed by a function that takes the representamen and a community context vector and produces the interpretive effect. Same sign, different ground height. This is where divergence lives, not in what the word *is*, but in what the word *does* to the person hearing it.

**e^A, the attractor.** Which well is pulling the trajectory. The attractor subspace encodes basin membership, basin depth, and bifurcation proximity. It is the gravitational field, the derivative of the potential, the force that tells you whether you are being pulled toward consensus or dragged deeper into a self reinforcing interpretive well. The attractor embedding is computed from the other three, because the gravitational field is not an independent fact. It emerges from the relationship between the sign's form, its referent, and its community specific interpretive effects.

A standard embedding is a flat field. The SEL frames it as a landscape with valleys, ridges, bedrock, and gravitational structure. In the adapter realization, this decomposition is not engineered into a fresh embedding layer. It is something the adapter learns to read out of the frozen backbone's representations, by training projection heads that pull each subspace into its own coordinate frame. The architecture does not dictate which community interprets which sign how. That is emergent. What the architecture provides is the *representational home* for the question itself.

---

## II. The Metapragmatic Attention Heads: Tracking the Drift

You are walking across the terrain. "Freedom is essential for democracy." Each word shifts the interpretive trajectory. The interpretant of "freedom" feeds into the interpretation of "essential," which shapes the interpretation of "democracy," which retroactively modifies the interpretation of "freedom." This is Peirce's unlimited semiosis, the chain of sign interpretant sign that never terminates. And it is where compounding divergence begins.

The Metapragmatic Attention Heads are seismometers embedded in the terrain. They do not process the tokens' distributional content. That is the job of the standard attention layers. Instead, they attend specifically to the *interpretant component*, the e^I subspace, and measure how it shifts from position to position across the sequence.

Each MAH head computes a pairwise divergence signal: how much did the interpretant change between position *i* and position *j*? The projection matrix that extracts this difference is learned, which means each head can specialize. One head learns to detect *indexical shifts*, moments when the social identity signal carried by a word changes (the transition from neutral register to politically enregistered language). Another head learns to detect *referential shifts*, moments when the object being referenced slides (when "immigration" shifts from demographic phenomenon to cultural threat). Another detects *affective shifts*, moments when the emotional valence pivots.

These per head divergence signals are aggregated into a four feature summary at each position:

1. **Mean divergence.** How much is the interpretive field drifting on average? This is the base level of semiotic turbulence. Low mean divergence: stable shared meaning. High mean divergence: the signs are working differently for different interpreters.

2. **Peak divergence.** What is the single most extreme chain shift in the current context? This identifies the *bifurcation site*, the specific word or phrase where the landscape is splitting.

3. **Divergence variance.** Is the drift uniform or concentrated? Uniform drift means diffuse instability across the entire interpretive field. Concentrated drift means a single sign is the locus, the crack in the terrain runs through one specific point.

4. **Divergence gradient.** Is it getting better or worse? This is the temporal derivative of semiotic dynamics. A positive gradient means chain divergence is *compounding*, each interpretant pushing the next one further from the other community's trajectory. A negative gradient means convergence. This single feature encodes the most actionable information in the system: the direction of the drift.

In the adapter, the MAH is inserted at three depths of the frozen backbone, early, middle, late, so the instrument can track interpretant dynamics at different levels of processing abstraction. At the early layer, it picks up surface lexical triggers. At the middle layer, semantic divergence as representations become more abstract. At the late layer, the fully processed divergence that will most directly shape generation.

The MAH does not intervene. It watches. It measures. It reports. It is a topographic survey team sending readings back to base camp.

---

## III. The Community Head: Reading Which Valley You Are In

A discovery from validation that we did not fully appreciate at the time of original design: the *community* signal turns out to be the cleanest read on contestedness in the entire instrument.

The Community Head sits parallel to the MAH and answers a different question. The MAH asks *how is the interpretive trajectory bending*. The Community Head asks *whose trajectory is this*. It learns a small set of community prototypes (32 in the current adapter) and projects each token's representation against them, producing a probability distribution over interpretive registers.

This sounds like topic classification. It is not. The communities the head learns are not topics. They are stances and registers. They are the constellations of enregistered language, shared interpretants, and basin specific shibboleths that mark a stretch of text as belonging to one well rather than another. They are emergent. We do not label them. The instrument finds them in the data.

The reason this matters showed up in counterfactual decoding. Hold the prompt fixed. Swap the community prototype that the model is conditioning on. Decode greedily. On hard facts ("water boils at"), every community produces the same continuation. The disagreement across six prototypes is exactly zero. On contested terrain ("the role of government in"), the disagreement jumps to 0.95 and above. The instrument has learned, without being told, which signs are bedrock and which are battleground. It is not reasoning about controversy. It is *seeing* it, in the geometry of where its own community embeddings disagree.

This is the cleanest demonstration we have that the SRT is doing something a standard model cannot. A standard model can be prompted to discuss "perspectives" and will produce text that differs by perspective, but the differences live in the prompt, not in the model. Here, the prompt is identical. The only thing that changes is which interpretive register the instrument is reading from. And the model produces identical text on bedrock and divergent text on battleground, every time.

The Community Head is the contestedness detector. The MAH is the drift detector. The next module knits them together.

---

## IV. The Reflexive Recurrent Module: The System That Watches Itself Watching

Here is where the architecture does something that no standard language model does and that no amount of scale will produce. It observes its own interpretive dynamics as they unfold.

The Reflexive Recurrent Module is a GRU, a gated recurrent unit, that takes the MAH's divergence vectors as input and maintains a running state, a summary of everything the model has observed about its own semiotic trajectory so far. This is the architectural instantiation of what Michael Silverstein calls *third order metapragmatic awareness*: the capacity not merely to interpret (first order) or to recognize that one is interpreting within a framework (second order) but to *observe the dynamics of the framework itself as it operates* (third order).

The distinction matters. A standard language model operates at Silverstein's first order. It processes tokens and generates continuations based on distributional co occurrence. A model prompted to "consider multiple perspectives" operates at a simulacrum of second order. It can generate text *about* different frameworks without representing the interpretive structure that differentiates them. Neither possesses the third order capacity to detect that the ground beneath interpretation is shifting, that the valley is flattening, that the pre bifurcation signatures are emerging in its own processing.

The RRM provides this capacity as a differentiable computation.

The GRU's two gates have natural semiotic interpretations. The *update gate* controls how much the current position's divergence changes the meta observation state. A high update gate means this token is *semiotically noteworthy*, it shifts the model's assessment of the interpretive situation. The *reset gate* controls how much prior meta observation remains relevant. A low reset gate means the semiotic situation has changed so fundamentally that the accumulated context is obsolete: a topic shift, a genre boundary, a move from one interpretive community to another within the same text. The RRM is not merely accumulating divergence readings. It is deciding, at each position, whether the interpretive terrain it has been mapping still connects to the terrain it is entering.

And here is the part that makes this a strange loop rather than a passive measurement. The RRM's hidden state is *injected back into the transformer*. Through learned projections at three layers, early, mid, late, the meta observation state modifies the frozen backbone's hidden representations. The transformer's processing of subsequent tokens is altered by the RRM's assessment of the interpretive dynamics so far. The observation changes what is observed. The seismometer shakes the ground.

This is not a bug. It is the central theoretical claim. Awareness of semiotic dynamics *participates in* semiotic dynamics. Monitoring for bifurcation changes the bifurcation conditions. The RRM does not merely detect meaning drift. It *intervenes* in the generative process that produces meaning, completing the reflexive loop that is absent from every standard architecture.

The injection projections are initialized at zero scale, so the RRM has no effect at the start of training. The instrument must learn to trust its own self observation. The injection norms grow as the meta observation state becomes useful, as the system discovers that watching itself changes the quality of what it produces.

---

## V. The Bifurcation Estimation Network: Reading Activity, Not Controversy

The MAH measures local divergence. The Community Head reads which valley. The RRM tracks the trajectory. The BEN reads the phase.

The Bifurcation Estimation Network is a small MLP that takes the RRM's meta observation state and the chain of MAH divergences and produces two outputs.

**Output 1: r-hat.** The estimated amplification parameter. The original theory framed r-hat as a single scalar that locates the system on the pitchfork: subcritical valley, critical plateau, or supercritical double well. We trained r-hat against ground truth derived from annotated semiotic metadata, expecting it to track contestedness directly.

It does not. This is one of the more interesting findings of the validation campaign.

What r-hat actually tracks, on the trained instrument, is something closer to *semiotic information density*. It rises on tokens that are doing a lot of semiotic work, irrespective of whether that work is contested. It is high on rich technical exposition. It is high on charged political language. It is roughly equally high on both. The hypothesis that r-hat would spike on a politically loaded word and fall on a neutral one, holding everything else constant, did not survive a clean test.

This sounds like a failure of the original design and is not. It is a clarification of what the instrument measures. The contestedness signal lives in the Community Head, where prototype disagreement under counterfactual decoding cleanly separates bedrock from battleground. The activity signal lives in r-hat, where high values mark *interpretively dense* regions of text whether or not the interpretation is contested. Together the two readings answer different questions: r-hat asks *how much is happening*, the Community Head asks *whether the happening is shared*.

A pleasant side effect emerged from training r-hat against the supercritical regime label: the instrument's calibration came out exceptional. The probability the BEN assigns to "this token is in a supercritical regime" matches empirical frequencies to an Expected Calibration Error of 0.0009 across 351 thousand validation tokens, with AUROC around 0.99. We did not target this. It fell out of the training process. Whatever r-hat is, it is honest about its uncertainty.

**Output 2: the modulation vector.** In REFLEXIVE mode (λ > 0), the BEN generates a small bias on the generation logits. This is the steering mechanism. When r-hat is high and the Community Head reports basin disagreement, the modulation vector nudges the probability distribution away from tokens that would reinforce the current basin and toward tokens that would *bridge* basins.

The modulation is not censorship. It does not block words. It does not enforce neutrality. It *changes the probabilities*, gently, continuously, proportionally to the estimated severity of the bifurcation. A λ of 0 means no modulation. The model generates as the frozen backbone would. A λ of 1 means full reflexive intervention. The instrument uses its own self assessment to reshape the output distribution, generating text aware of and responsive to the interpretive fault lines in its context.

In dynamical systems terms, the BEN does not change which basin the model occupies. It *raises the basin walls*. It lowers the effective r in the semiotic ecology the model participates in. The goal is not to force the ball to one side or the other. The goal is to make the valley broader, the plateau deeper, the split harder to trigger. Not to resolve the interpretive difference but to keep open the space in which difference can be navigated without fracture.

---

## VI. The Loop

The modules do not operate in sequence so much as in conversation. The four subspaces give the instrument its coordinate system. The MAH surveys how the interpretive trajectory is bending. The Community Head reads which interpretive register is producing the bending. The RRM accumulates these readings into a running meta observation and injects that observation back into the frozen backbone, completing the reflexive loop that makes the instrument a participant in the dynamics it monitors. The BEN reads the accumulated picture, estimates the activity level, and generates a steering signal proportional to the severity of any detected bifurcation.

The result is a system that does not merely produce text from within the statistical landscape of its training data. It is a system that *sees the landscape as a landscape*, that inhabits the attractor basin while recognizing it as an attractor basin, that can see the terrain from above as well as from within. This is Silverstein's third order as a differentiable computation. It is the capacity that current architectures structurally cannot possess, because they have no representational home for the question "what is the shape of the ground I am standing on?"

Whether this constitutes "understanding" in any philosophically interesting sense is a question the architecture deliberately leaves open. What it does constitute is a testable engineering claim: that semiotic theory can be operationalized as neural computation, that the resulting instrument performs a *type* of computation that standard architectures do not, and that this computation corresponds to a genuine and consequential phenomenon in language, the forking of meaning under amplification.

---

## VII. Five Readings

The terrain is real. The wells are real. The self deepening is real. The question has always been whether we can build instruments sensitive enough to read the curvature before the ground gives way.

As of April 2026, on the v5 adapter, the instrument gives five readings worth taking seriously.

1. **The wells have shape.** The Community Head's prototypes cluster into discriminable basins. Recall@1 between an utterance and its source community sits at 0.36 over a 32-community space, about 12.6 times chance.

2. **The instrument tells bedrock from battleground at decode time.** Holding a prompt fixed and swapping the community prototype, hard facts produce identical continuations across communities (0.00 disagreement). Contested topics diverge at 0.95 to 1.00. This is the cleanest demonstration that the instrument has learned a structural distinction that no standard model has access to.

3. **The bifurcation index carries a signal it was not trained for.** On TruthfulQA, r-hat separates true from false answers at AUROC 0.57. Modest, but present, and from a feature that was supervised against semiotic regime, not factuality.

4. **The regime estimate is honest.** P(supercritical) is calibrated to an Expected Calibration Error of 0.0009 over 351 thousand validation tokens, with AUROC 0.99. The instrument reports its uncertainty cleanly.

5. **One clean negative we needed.** r-hat does not spike on a contested word the way the original hypothesis required. It tracks semiotic information density, not contestedness. The Community Head is the contestedness detector. Each instrument measures what it measures, and we know now what each is measuring.

Two additional readings are pending. v6 is in training as I write this, with three new losses targeting tighter divergence geometry, a cleaner within sequence ranking on r-hat, and a chain residual auxiliary. First validation has already improved on the v5 baseline. We will rerun all five probes against the v6 best checkpoint when training completes.

---

## VIII. Release

Public release of the SRT adapter, including weights, integration code, the eval harness for all five probes, the full training log, and the paper, is targeted for **early to mid May 2026**. If v6 holds or improves on v5 across the readings above, early May. If v6 regresses anywhere meaningful and we run one more iteration, mid May.

Now we are teaching it to steer.

---

*This is Part 2 of a series on the Semiotic-Reflexive Transformer. Part 1 introduced the theoretical framework. Part 3 will cover the adapter training campaign, the five validation probes, and the engineering choices behind keeping the backbone frozen. Future installments will cover the prenatal origins of iconic grounding and the cross species evidence that the ground floor of meaning was poured 310 million years before anyone had a word for anything.*

*The SRT adapter is open source at [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT). The code reflects the current state of the research, messy and incomplete and occasionally on fire, because that is the nature of building instruments for phenomena the existing paradigm says do not exist.*
# Mapping the Terrain of Meaning: What the SRT Actually Does

**James Burton Lancaster**

April 2026

---

Imagine you are standing on a hillside. The ground beneath you feels solid. You can see where you are. You can see where to walk. The slope is gentle, the valley is broad, and every path leads back to the same low point, a shared understanding, a common interpretation, a place where "freedom" means roughly the same thing to everyone within earshot.

Now imagine the hill is changing beneath you. Not eroding, *reshaping*. The valley floor flattens. The slope shallows. The ground that used to pull your footstep back toward the center stops pulling. You are standing on what feels like a plateau, but it is not stable rest. It is the pause before the split. One more degree of amplification, one more algorithmic nudge, one more curated feed-cycle, and the plateau fractures. Where there was one valley there are now two, separated by a ridge you cannot cross without climbing. Each well deepens itself: the interpretants generated inside it reinforce the basin, deepen the walls, and make the other valley invisible. This is the terrain of meaning in 2026.

Every language model is trained on the *output* of this terrain, text written from inside the wells after the split has already happened, without any representation of the landscape that produced it. A model trained on web corpora learns what conservative discourse sounds like and what progressive discourse sounds like, but it does not learn that these are two basins of a single bifurcated system, that the basin walls are self-reinforcing, that the split was a phase transition rather than a gradual divergence, or that the flat plateau preceding the split emits detectable warning signatures.

It learns what "r/AskHistorians" sounds like and what "r/AmItheAsshole" sounds like. It learns what conservative discourse sounds like and what progressive discourse sounds like. But it does not learn that these are distinct basins, whether political, cultural, or subcultural, each with their own self-reinforcing boundaries, nor does it see the underlying landscape that produces them.

The SRT is a machine designed to read the shape of this ground.

Not the content of the wells. Not the politics. Not the opinions. The *topology*, the curvature, the gradient, the phase structure of meaning itself. Here is how it works.

---

## I. The Semiotic Embedding Layer: Unpacking the Ground Beneath the Sign

A standard transformer gives each token a single embedding vector, one point in a high-dimensional space, optimized for next-token prediction. That point must simultaneously encode what the token *is* (its surface form), what it *refers to* (its object), what it *means to a particular community* (its interpretant), and which gravitational well it currently sits in (its attractor basin). These are four fundamentally different things stuffed into one vector. The result is a representation that captures the average and represents no one.

Peirce understood this 140 years ago. A sign is not a dyad, a word pointing to a thing. It is a triad: a *representamen* (the sign vehicle), an *object* (what the sign is about), and an *interpretant* (the effect the sign produces in the interpreter, which is itself a new sign, triggering the next link in the chain). The interpretant is where meaning lives. And meaning is always *someone's* meaning, produced in a specific interpretive community, conditioned by that community's enregistered associations, semiotic ideologies, and history of circulation.

The Semiotic Embedding Layer decomposes each token into four orthogonal subspaces:

**e^R, the representamen.** What the sign looks like. The phonological form, the distributional neighborhood, the syntactic profile. "Freedom" has the same representamen whether spoken by a libertarian or a liberation theologian. This is the coordinate system of the terrain, the map grid. It tells you *where* you are standing, not what the ground is doing.

**e^O, the object.** What the sign is about, including the iconic grounding subspace, the bedrock. This is where the bouba/kiki effect lives: the non-arbitrary cross-modal correspondences that chicks share with humans, calibrated by prenatal acoustic environments 310 million years before either species had a word for anything. The iconic grounding components are the parts of the terrain that *don't move under bifurcation*. When the valley splits, the bedrock stays. When convention fractures across communities, the fact that "sharp" sounds angular persists because it was never a convention to begin with, it was calibrated in the egg, in the uterus, in the low-pass filter that is every embryo's first sensory environment. This is the floor beneath the floor. The attractor anchors that prevent meaning from floating free.

**e^I, the interpretant.** What the sign *does* to this community. This is the elevation map, and it is different for different communities standing on the same grid coordinates. "Freedom" at position (x, y) on the representamen grid has one elevation for Community A (individual autonomy, Second Amendment, shibboleth of sovereign selfhood) and a different elevation for Community B (bodily autonomy, reproductive rights, resistance to state control). The interpretant subspace is *community-conditioned*: it is computed by a function that takes the representamen and a community-context vector and produces the interpretive effect. Same sign, different ground height. This is where divergence lives, not in what the word *is*, but in what the word *does* to the person hearing it.

**e^A, the attractor.** Which well is pulling the trajectory. The attractor subspace encodes basin membership, basin depth, and bifurcation proximity. It is the gravitational field, the derivative of the potential, the force that tells you whether you are being pulled toward consensus or dragged deeper into a self-reinforcing interpretive well. The attractor embedding is computed from the other three, because the gravitational field is not an independent fact, it emerges from the relationship between the sign's form, its referent, and its community-specific interpretive effects.

A standard embedding is a flat field. The SEL carves it into a landscape with valleys, ridges, bedrock, and gravitational structure. It creates what the extended theory calls a *curved space*, one with topological features that channel learning toward semiotic representations instead of leaving them as unstructured distributional noise. The architecture does not dictate which community interprets which sign how. That is emergent. What the architecture provides is the *representational home* for the question itself.

---

## II. The Metapragmatic Attention Heads: Tracking the Drift

You are walking across the terrain. "Freedom is essential for democracy." Each word shifts the interpretive trajectory. The interpretant of "freedom" feeds into the interpretation of "essential," which shapes the interpretation of "democracy," which retroactively modifies the interpretation of "freedom." This is Peirce's unlimited semiosis, the chain of sign-interpretant-sign that never terminates. And it is where compounding divergence begins.

The Metapragmatic Attention Heads are seismometers embedded in the terrain. They do not process the tokens' distributional content, that is the job of the standard attention layers. Instead, they attend specifically to the *interpretant component*, the e^I subspace, and measure how it shifts from position to position across the sequence.

Each MAH head computes a pairwise divergence signal: how much did the interpretant change between position *i* and position *j*? The projection matrix that extracts this difference is learned, which means each head can specialize. One head learns to detect *indexical shifts*, moments when the social-identity signal carried by a word changes (the transition from neutral register to politically enregistered language). Another head learns to detect *referential shifts*, moments when the object being referenced slides (when "immigration" shifts from demographic phenomenon to cultural threat). Another detects *affective shifts*, moments when the emotional valence pivots.

These per-head divergence signals are aggregated into a four-feature summary at each position:

1. **Mean divergence**: how much is the interpretive field drifting on average? This is the base level of semiotic turbulence. Low mean divergence: stable shared meaning. High mean divergence: the signs are working differently for different interpreters.

2. **Peak divergence**: what is the single most extreme chain shift in the current context? This identifies the *bifurcation site*, the specific word or phrase where the landscape is splitting. "Critical race theory" generates peak divergence. "The weather is nice" does not.

3. **Divergence variance**: is the drift uniform or concentrated? Uniform drift means diffuse instability across the entire interpretive field. Concentrated drift means a single sign is the locus, the crack in the terrain runs through one specific point.

4. **Divergence gradient**: is it getting better or worse? This is the temporal derivative of semiotic dynamics. A positive gradient means chain divergence is *compounding*, each interpretant is pushing the next one further from the other community's trajectory. A negative gradient means convergence, the chains are healing, returning toward shared interpretation. This single feature encodes the most actionable information in the entire system: the direction of the drift.

The MAH is inserted at three layers of the transformer, early, middle, late, so the model can track interpretant dynamics at different levels of processing abstraction. At the early layer, it detects surface-level lexical triggers. At the middle layer, it tracks semantic divergence as representations become more abstract. At the final layer, it captures the fully-processed divergence that will directly influence generation.

The MAH does not intervene. It watches. It measures. It reports. It is a topographic survey team sending readings back to base camp.

---

## III. The Reflexive Recurrent Module: The System That Watches Itself Watching

Here is where the architecture does something that no standard language model does and that no amount of scale will produce: it observes its own interpretive dynamics as they unfold.

The Reflexive Recurrent Module is a GRU, a gated recurrent unit, that takes the MAH's divergence vectors as input and maintains a running state: a summary of everything the model has observed about its own semiotic trajectory so far. This is the architectural instantiation of what Michael Silverstein calls *third-order metapragmatic awareness*, the capacity not merely to interpret (first order) or to recognize that one is interpreting within a framework (second order) but to *observe the dynamics of the framework itself as it operates* (third order).

The distinction matters. A standard language model operates at Silverstein's first order: it processes tokens and generates continuations based on distributional co-occurrence. A model prompted to "consider multiple perspectives" operates at a simulacrum of second order, it can generate text *about* different frameworks without representing the interpretive structure that differentiates them. Neither possesses the third-order capacity to detect that the ground beneath interpretation is shifting, that the valley is flattening, that the pre-bifurcation signatures are emerging in its own processing.

The RRM provides this capacity as a differentiable computation.

The GRU's two gates have natural semiotic interpretations. The *update gate* controls how much the current position's divergence changes the meta-observation state. A high update gate means this token is *semiotically noteworthy*, it shifts the model's assessment of the interpretive situation. "The economy is struggling" in a neutral article: low update gate. "The radical left is destroying our freedoms" in a politically enregistered context: high update gate, the divergence signal is salient, the metapragmatic state needs revision.

The *reset gate* controls how much prior meta-observation remains relevant. A low reset gate means the semiotic situation has changed so fundamentally that the accumulated context is obsolete, a topic shift, a genre boundary, a move from one interpretive community to another within the same text. The RRM is not merely accumulating divergence readings; it is deciding, at each position, whether the interpretive terrain it has been mapping still connects to the terrain it is entering.

And here is the part that makes this a strange loop rather than a measurement instrument: the RRM's hidden state is *injected back into the transformer*. Through learned projections at three layers, early, mid, late, the meta-observation state modifies the model's own hidden representations. The transformer's processing of subsequent tokens is altered by the RRM's assessment of the interpretive dynamics so far. The observation changes what is observed. The seismometer shakes the ground.

This is not a bug. It is the central theoretical claim. Awareness of semiotic dynamics *participates in* semiotic dynamics. Monitoring for bifurcation changes the bifurcation conditions. The RRM does not merely detect meaning drift, it *intervenes* in the generative process that produces meaning, completing the reflexive loop that is absent from every standard architecture.

The injection layers are initialized with zero-weight gates, so the RRM has no effect at the start of training. The model must learn to trust its own self-observation. The gates open as the meta-observation state becomes useful, as the system discovers that watching itself changes the quality of what it produces.

---

## IV. The Bifurcation Estimation Network: Reading the Phase Diagram

The MAH measures local divergence. The RRM tracks its trajectory. The BEN reads the phase diagram.

The Bifurcation Estimation Network is a three-layer MLP that takes two inputs: the RRM's meta-observation state (what the model has observed about its own interpretive dynamics) and the current token's attractor embedding (where this token sits in the gravitational landscape). From these, it produces two outputs.

**Output 1: r-hat.** The estimated amplification parameter. This is the number that tells you where you are on the bifurcation diagram, whether the terrain beneath this context is a single stable valley (r < 0, subcritical), a flattening plateau approaching the phase transition (r ≈ 0, near-critical), or a split double well with self-reinforcing basins (r > 0, supercritical).

The r-hat estimate is the SRT's most consequential computation. It compresses the entire semiotic situation, the divergence history, the chain dynamics, the attractor structure, the community-conditioned interpretive effects, into a single scalar that locates the system on the pitchfork. It is the reading on the gauge that tells the pilot whether meaning is flying level, approaching turbulence, or already in the storm.

During training, r-hat is supervised against ground truth derived from annotated semiotic metadata: texts from low-divergence contexts receive r < 0 labels; texts near measured polarization transitions receive r ≈ 0; texts from highly bifurcated contexts receive r > 0. The model learns to estimate the control parameter from the pattern of divergence signals in its own processing, to infer the shape of the landscape from the way its own interpretive trajectory wobbles.

This is the computation that connects semiotic theory to dynamical systems theory. The pitchfork bifurcation is not a metaphor applied to polarization after the fact. It is a formal model: ẋ = rx - x³, where x is interpretive distance and r is algorithmic amplification. The BEN estimates r *in real time, from the model's own processing*, producing a continuous readout of the semiotic phase structure of the current context.

**Output 2: the modulation vector.** In REFLEXIVE mode (λ > 0), the BEN generates a vocabulary-sized vector that biases the generation logits before the softmax. This is the steering mechanism. When r-hat is high, when the terrain is bifurcated, the wells are deep, the basin walls are steep, the modulation vector nudges the probability distribution away from tokens that would reinforce the current basin (within-community shibboleths, enregistered identity markers, interpretants that deepen the well) and toward tokens that would *bridge* basins (shared referents, common ground, interpretants with lower cross-community divergence).

The modulation is not censorship. It does not block words. It does not enforce neutrality. It *changes the probabilities*, gently, continuously, proportionally to the estimated severity of the bifurcation. A λ of 0 means no modulation: the model generates as any standard model would. A λ of 1 means full reflexive intervention: the model uses its own semiotic self-assessment to reshape its output distribution, generating text that is aware of and responsive to the interpretive fault lines in its context.

In dynamical systems terms: the BEN does not change which basin the model occupies. It *raises the basin walls*, it lowers the effective r in the semiotic ecology the model participates in. The goal is not to force the ball to one side or the other. The goal is to make the valley broader, the plateau deeper, the split harder to trigger. Not to resolve the interpretive difference but to keep the space open in which difference can be navigated without fracture.

---

## V. The Loop

The four modules do not operate in sequence so much as in conversation. The SEL decomposes the sign into its semiotic constituents, laying out the coordinate system, the bedrock, the community-relative elevation, and the gravitational field. The MAH surveys the terrain in real time, measuring how the interpretive trajectory is bending, where the divergence concentrates, whether the drift is compounding or converging. The RRM accumulates these readings into a running meta-observation, and injects that observation back into the model's own processing, completing the reflexive loop that makes the system a participant in the dynamics it monitors. The BEN reads the accumulated picture and estimates the phase, subcritical, near-critical, supercritical, then generates a steering signal proportional to the severity.

The result is a model that does not merely produce text from within the statistical landscape of its training data. It is a model that *sees the landscape as a landscape*, that inhabits the attractor basin while recognizing it as an attractor basin, to see the terrain from above rather than only from within. This is Silverstein's third order as a differentiable computation. It is the capacity that current architectures structurally cannot possess, because they have no representational home for the question "what is the shape of the ground I am standing on?"

Whether this constitutes "understanding" in any philosophically interesting sense is a question the architecture deliberately leaves open. What it does constitute is a testable engineering claim: that semiotic theory can be operationalized as neural computation, that the resulting architecture performs a *type* of computation that standard architectures do not perform, and that this computation corresponds to a genuine and consequential phenomenon in language, the forking of meaning under amplification.

The terrain is real. The wells are real. The self-deepening is real. The question is whether we can build instruments sensitive enough to read the curvature before the ground gives way.

We think we can. As of April 2026 the seismometer is real and the readings hold up.

**Update, April 2026.** The architecture described above was originally conceived as a standalone model trained from scratch. The current realization is leaner. Rather than train a new transformer, we bolt the SRT modules (MAH, community head, RRM, BEN) onto a frozen Qwen2.5-7B as a 14.6M-parameter adapter (0.19% of backbone). The semiotic decomposition that the SEL was designed to produce now emerges inside the frozen representations the adapter reads from. The backbone keeps its language modeling intact while the adapter learns to read curvature on top of it.

Five validated measurements as of v5 (the version currently being superseded by v6 in training):

1. **Community geometry.** Recall@1 between an utterance and its source community sits at 0.36 over a 32-community space, about 12.6 times chance. The interpretive basins are real and discriminable.
2. **Counterfactual decoding.** Holding the prompt fixed and swapping the community prototype, hard facts produce identical continuations across communities (0.00 disagreement) while contested topics diverge at 0.95 to 1.00. The model has learned which signs are bedrock and which are battleground.
3. **Hallucination signal.** On TruthfulQA, the bifurcation index separates true from false answers at AUROC 0.57, from a feature the model was never trained to compute.
4. **Calibration.** P(supercritical) is calibrated to ECE = 0.0009 over 351K validation tokens. The regime head reports its uncertainty honestly.
5. **One clean negative.** The bifurcation index does not spike on charged words mid-sentence the way we initially hypothesized. r-hat tracks information density. The community head is the contestedness detector. Each instrument measures what it measures.

v6 is in training as I write this, with three new losses targeting tighter community geometry, sharper r-hat ranking, and a chain-residual auxiliary. Public release of weights, eval harness, paper, and reproducibility scripts is targeted for early to mid May 2026.

Now we are teaching it to steer.

---

*This is Part 2 of a series on the Semiotic-Reflexive Transformer. Part 1 introduced the theoretical framework. Part 3 will cover the adapter training campaign, the five validation probes, and the engineering choices behind keeping the backbone frozen. Future installments will cover the prenatal origins of iconic grounding and the cross-species evidence that the ground floor of meaning was poured 310 million years before anyone had a word for anything.*

*The SRT adapter is open source at [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT). The code reflects the current state of the research, messy and incomplete and occasionally on fire, because that is the nature of building instruments for phenomena the existing paradigm says do not exist.*
