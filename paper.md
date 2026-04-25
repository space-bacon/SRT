# Semiotic Taps: Lightweight Adapter Modules for Bifurcation Detection in Frozen Language Models

**James Burton Lancaster**

April 2026

---

## Abstract

Large language models trained on web-scale corpora absorb the semiotic bifurcations embedded in their data — divergent interpretant chains in which the same sign carries incompatible meanings across discourse communities — but have no mechanism to detect, represent, or respond to this divergence. We introduce the Semiotic-Reflexive Transformer Adapter (SRT-Adapter), a lightweight architecture (~14.5M trainable parameters, 0.19% of a 7B backbone) that bolts semiotic awareness onto any frozen causal language model without modifying its embeddings, attention, or output head. The adapter operates through four modules that *tap* hidden states at selected backbone layers: (1) a **Community Discovery Head** that performs unsupervised soft clustering of discourse communities from early-layer representations; (2) **Metapragmatic Attention Heads** (MAH) that compute divergence vectors quantifying where meaning forks under community-conditioned interpretation; (3) a **Reflexive Recurrent Module** (RRM) that tracks accumulated semiotic divergence through a per-position GRU meta-state and optionally injects small corrections into the backbone stream via FiLM modulation; and (4) a **Bifurcation Estimation Network** (BEN) that estimates a continuous reflexivity coefficient $\hat{r}$ and a binary semiotic regime (subcritical/supercritical) at each token position. Grounded in Peircean semiotics and the pitchfork bifurcation model of political polarization (Lancaster, 2025), the architecture treats the frozen backbone as a substrate on which semiotic processes are an emergent, measurable phenomenon. Training combines the backbone's native cross-entropy with auxiliary losses on chain-of-interpretants prediction, bifurcation regression, regime classification, divergence health, community entropy and supervised-contrastive separation (v5), and — in v6 — metapragmatic-divergence supervised-contrastive separation, ListNet ranking on $\hat{r}$, and a chain-residual auxiliary floor. The corpus is 1M Reddit samples spanning 35 discourse communities with per-token reflexivity annotations. We present the theoretical motivation, full architectural specification, training methodology, and evaluation across five independent probes of the v5 checkpoint: cross-entropy preservation (CE = 2.63 on val, vs. 2.71 for unadapted Qwen 2.5-7B), unsupervised community retrieval (recall@1 = 0.36, $12.6\times$ random on a 35-class task), counterfactual community decoding (zero disagreement on factual prompts, 0.95 mean disagreement on contested topics), zero-shot hallucination signal on TruthfulQA (mean $\hat{r}$ AUROC = 0.573 without truthfulness supervision), and regime calibration (ECE = $9 \times 10^{-4}$, AUROC = 0.99 on 351K tokens).

**Keywords:** semiotic adapter, bifurcation detection, metapragmatic attention, interpretant chains, reflexive recurrence, discourse community discovery, frozen backbone, pitchfork bifurcation, Peircean semiotics

---

## 1. Introduction

### 1.1 The Problem

Language models are semiotic infrastructure. Their outputs enter interpretant chains alongside human-authored signs, shaping subsequent interpretation in ways neither users nor developers can fully trace. Yet the training paradigm that produces these systems is semiotically naive: it optimizes for the conditional probability of the next token, a surface-level objective that captures co-occurrence patterns while remaining structurally blind to the interpretive processes that make those patterns meaningful.

The consequence is that when a language model encounters a contested sign — "freedom," "justice," "woke" — it produces text that is fluent within a particular attractor basin without representing the fact that the sign indexes opposed interpretive communities. The model does not know it is in a bifurcation zone. It cannot tell you.

Current alignment methods (RLHF, DPO, Constitutional AI) intervene downstream, constraining outputs after the model has already internalized a bifurcated semiotic landscape. They adjust trajectories within a fixed attractor landscape without reshaping the landscape itself. The control parameter $r$ that governs bifurcation remains untouched.

### 1.2 The Opportunity: SRT as Adapter

Our prior work (Lancaster, 2026a) proposed a full Semiotic-Reflexive Transformer architecture with custom embedding layers, modified attention mechanisms, and interleaved semiotic modules throughout the backbone. While theoretically comprehensive, that approach faced practical limitations: custom embeddings degraded cross-entropy loss from pretrained quality, the full architecture required training from near-scratch, and the deep coupling between semiotic modules and backbone layers created optimization instability.

This paper takes a fundamentally different approach. We observe that the semiotic phenomena we wish to detect — interpretant divergence, community-specific meaning, bifurcation dynamics — are *already encoded* in the hidden states of pretrained language models. They must be, because these models were trained on text produced by communities with divergent interpretive norms. The information is there; what is missing is the apparatus to read it.

The SRT-Adapter is that apparatus. It wraps any frozen HuggingFace causal language model and installs lightweight semiotic taps — modules that read hidden states, compute divergence, track meta-state, and estimate bifurcation — without modifying a single backbone parameter. The backbone's native embeddings and language modeling head are used directly. Cross-entropy starts at pretrained quality. Only ~12.7M adapter parameters train, while 7.6B backbone parameters remain frozen.

### 1.3 Theoretical Grounding

The architecture rests on three converging theoretical lines:

1. **Peircean semiotics** (Peirce, 1931–1958; Kockelman, 2024, 2025): Every sign completes its meaning through a culturally conditioned interpretant, which itself becomes the next sign in an open chain. When the same representamen enters different interpretive communities, it generates different initial interpretants that compound through subsequent links into mutual unintelligibility.

2. **Pitchfork bifurcation dynamics** (Lancaster, 2025): The compounding of interpretant divergence across algorithmically curated communities exhibits the structure of a supercritical pitchfork bifurcation $\dot{x} = rx - x^3$. Below a critical threshold of the control parameter $r$, shared interpretive equilibria absorb perturbation (subcritical regime). Above it, symmetry breaks into antagonistic attractors that are self-reinforcing and structurally resistant to reconciliation (supercritical regime).

3. **Metapragmatic awareness** (Silverstein, 1993, 2003): The capacity to observe how discourse itself shapes interpretation — to notice that a sign is being contested, not merely to interpret it from within one community's norms — constitutes a third-order reflexive capacity that is architecturally absent from standard transformers.

### 1.4 Contributions

This paper makes three contributions:

1. **Adapter architecture for semiotic awareness.** We specify a complete, working architecture that adds bifurcation detection to any frozen causal LM through four lightweight modules totaling ~14.5M parameters (0.19% of a 7B backbone). The design preserves pretrained language modeling quality (CE = 2.63 vs. unadapted 2.71) while adding structured semiotic outputs.

2. **Unsupervised community discovery via supervised-contrastive prototypes.** Rather than requiring predefined community labels at inference time, the adapter discovers discourse communities from backbone hidden states through learned prototype-based soft clustering. We identify and resolve a degenerate failure mode ("congruent collapse") in which entropy regularization keeps the assignment distribution uniform while pairwise prototype cosine converges to $\approx 0.99$. Supervised-contrastive loss applied to the encoder's *pre-mixing* output — not the prototype-weighted vector — raises retrieval recall@1 from 0.05 (1.7$\times$ random) to 0.36 (12.6$\times$ random) on a 35-class task.

3. **Multi-objective training with semiotic auxiliary losses.** We define a training pipeline that combines the backbone's native cross-entropy with chain-of-interpretants prediction, bifurcation estimation, regime classification, divergence health, community entropy and supervised-contrastive separation, metapragmatic-divergence supervised-contrastive separation, and ListNet ranking on the reflexivity estimate — each motivated by a specific structural property of the architecture and validated by an independent probe.

### 1.5 Paper Organization

Section 2 develops the theoretical framework connecting Peircean semiotics to the adapter architecture. Section 3 specifies the full architecture with formal detail. Section 4 describes the training methodology and data pipeline. Section 5 presents preliminary experimental results. Section 6 discusses implications and limitations. Section 7 concludes.

---

## 2. Theoretical Framework

### 2.1 Signs, Interpretants, and the Compounding of Divergence

Peirce's triadic semiotics decomposes every sign process into three irreducible elements: the *representamen* (perceptible sign vehicle), the *object* (what the sign represents), and the *interpretant* (the effect the sign produces in an interpreter, which is itself a sign). The interpretant is the decisive element for our purposes: it makes signification an open, processual, and inherently social phenomenon. Each interpretant functions as a new representamen, generating further interpretants in chains of "unlimited semiosis" (Peirce, CP 2.303).

Kockelman (2025) formalizes these chains as dynamical trajectories through a state space. Each link involves an act of *sieving*: from the space of possible interpretants a sign could produce, only some are actualized, depending on the interpreter's prior exposure, community membership, and the mediation architecture that delivered the sign. When the same representamen enters different interpretive communities — communities whose sieving mechanisms have been calibrated by exposure to different algorithmically curated sign environments — it generates different initial interpretants. These divergent interpretants function as new representamena, generating further divergent interpretants.

The critical insight is that this compounding is *quantifiable*. At each link in the chain, the divergence between community-specific interpretants can be measured as a vector difference in an appropriately structured representation space. This is precisely what the Metapragmatic Attention Head computes.

### 2.2 The Pitchfork Bifurcation as Control Model

Lancaster (2025) demonstrated that the dynamics of interpretant divergence under algorithmic curation exhibit the qualitative structure of a supercritical pitchfork bifurcation:

$$\dot{x} = rx - x^3$$

The variable $x$ represents the degree of interpretive divergence at a given semiotic site (a word, phrase, or passage). The control parameter $r$ encodes the effective strength of divergence-amplifying forces — algorithmic curation, community reinforcement, contextual framing. The dynamics are:

- **Subcritical** ($r < 0$): The origin $x = 0$ is a stable equilibrium. Perturbations decay. The sign has shared, conventional meaning across communities.
- **Critical** ($r = 0$): The equilibrium becomes non-hyperbolic. The system is sensitive to perturbation.
- **Supercritical** ($r > 0$): The origin becomes unstable and two new stable equilibria emerge at $x = \pm\sqrt{r}$. Meaning has bifurcated into community-specific attractors.

The SRT-Adapter's Bifurcation Estimation Network estimates $\hat{r}$ at each token position, providing a continuous measure of semiotic stability that is grounded in this dynamical framework.

### 2.3 Metapragmatic Awareness as Architectural Capacity

Silverstein (1993, 2003) distinguishes three orders of indexicality:

1. **First-order**: Direct sign use ("It's cold" indexes temperature).
2. **Second-order**: Ideological construal ("When *they* say freedom, they mean…" indexes community boundaries).
3. **Third-order**: Metapragmatic awareness (recognizing that the very framing of "freedom" as contested is itself a semiotic act).

Standard transformers have no structural capacity for third-order awareness. They process text from within whatever interpretive frame their training data established, without the ability to step back and observe the frame itself.

The RRM instantiates this capacity computationally. By accumulating divergence observations across layers into a meta-state and optionally injecting corrections back into the processing stream, the RRM creates a reflexive loop: the observation of semiotic dynamics changes the dynamics being observed. This is not an analogy — it is the same structural relationship that defines metapragmatic awareness in Silverstein's framework.

### 2.4 Why an Adapter Architecture

The theoretical claim that motivates the adapter design is specific: **the semiotic structure is already in the hidden states**. A language model trained on text produced by multiple discourse communities has necessarily learned representations that reflect those communities' divergent interpretive norms. The representations encode the fact that "freedom" occurs in different distributional neighborhoods in libertarian versus progressive text. What the model lacks is the apparatus to disentangle this structure, compute its divergence, and report it as a structured output.

This claim has an important architectural consequence. We do not need to rebuild the backbone's representations. We need to *read* them with semiotic-specific projections. The backbone's hidden states at different layers capture different levels of contextual integration — early layers encode more local, syntactic features; later layers encode more global, semantic features. By tapping these states at strategically chosen layers, we can track how interpretive context accrues and where it forks.

The adapter design also resolves three practical problems that plagued the full SRT architecture:

1. **CE degradation**: Custom embeddings in the full SRT disrupted pretrained representations, causing cross-entropy to start at ~200 rather than ~3.5. The adapter preserves the backbone's native embeddings and LM head, so CE starts at pretrained quality.

2. **Training cost**: The full SRT required training or fine-tuning the entire backbone. The adapter freezes the backbone and trains only ~12.7M semiotic parameters, reducing training from weeks to hours.

3. **Backbone agnosticism**: The adapter works with any HuggingFace `AutoModelForCausalLM` (LLaMA, Qwen, Mistral, Phi, Gemma) without architecture-specific modifications.

---

## 3. Architecture

### 3.1 Overview

The SRT-Adapter wraps a frozen HuggingFace causal language model and runs its transformer layers manually in a loop, inserting semiotic operations at specified layer indices. The backbone's own embedding layer, positional encoding, final layer norm, and language modeling head are used directly. Four trainable modules constitute the adapter:

```
tokens ──► Backbone Embed (frozen)
               │
         ┌─────┴──────┐
         │  Layers 0–3 │  (frozen)
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │   Layer 4   │──► Community Discovery Head ──► community vector c
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │  Layers 5–6 │  (frozen)
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │   Layer 7   │──► MAH₁(h, c) ──► divergence d₁ ──► RRM step
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │ Layers 8–13 │  (frozen)
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │  Layer 14   │──► MAH₂(h, c) ──► d₂ ──► RRM step ──► inject Δh
         └─────┬──────┘                                            │
               │◄──────────────────────────────────────────────────┘
         ┌─────┴──────┐
         │ Layers 15–20│  (frozen, with correction)
         └─────┬──────┘
               │
         ┌─────┴──────┐
         │  Layer 21   │──► MAH₃(h, c) ──► d₃ ──► RRM step ──► inject Δh
         └─────┬──────┘                                            │
               │◄──────────────────────────────────────────────────┘
         ┌─────┴──────┐
         │ Layers 22–27│  (frozen, with correction)
         └─────┬──────┘
               │
         Final Norm + LM Head (frozen) ──► logits, CE loss
               │
         BEN(meta_state) ──► r̂, regime
```

Layer indices are auto-computed from backbone depth $L$: MAH hooks at $\lfloor L/4 \rfloor, \lfloor L/2 \rfloor, \lfloor 3L/4 \rfloor$; RRM injection at MAH layers 2 and 3 (letting meta-state accumulate before first injection); community discovery at $\max(1, \lfloor L/7 \rfloor)$.

### 3.2 Community Discovery Head

The community head runs at a single early backbone layer and discovers discourse communities without predefined labels. This is the first architectural departure from the original SRT, which required explicit community IDs. In Peircean terms, a discourse community is a group of language users who share interpretive norms — they assign similar interpretants to the same representamens.

**Architecture:**
1. Masked mean pool over sequence positions: $\bar{h} = \frac{\sum_t h_t \cdot m_t}{\sum_t m_t}$ where $m_t$ is the attention mask.
2. Encode to community space: $z = \text{SiLU}(W_{\text{enc}} \bar{h}) \in \mathbb{R}^{d_c}$ with $d_c = 64$.
3. Cosine similarity to $K = 32$ learned prototypes: $\ell_k = \frac{z \cdot p_k}{\|z\| \|p_k\|} / \tau$ with temperature $\tau = 1.0$.
4. Soft assignment: $w = \text{softmax}(\ell) \in \mathbb{R}^K$.
5. Community vector: $c = \sum_k w_k p_k \in \mathbb{R}^{d_c}$.

The community vector $c$ conditions all subsequent MAH computations, enabling the same sign to produce different divergence patterns depending on the discovered community context.

### 3.3 Metapragmatic Attention Head (MAH)

Each MAH layer reads hidden states from a backbone layer and computes a divergence vector at each position quantifying where meaning forks under contextual interpretation.

**Theoretical motivation.** The divergence vector $d_t$ at position $t$ captures:
$$d_t = f(\text{interp}_t) - g(\text{attend}(\text{interp}_{0..t}))$$

where $f$ is a direct projection of the token's representation into interpretant subspace and $g$ is the output after causal self-attention over all preceding interpretant representations. High $\|d_t\|$ indicates that the sign at position $t$ means something different in discourse context than it would in isolation — that is, it is a site of active semiotic divergence.

**Architecture:**
1. Project backbone hidden states to interpretant subspace: $\text{interp} = W_{\text{proj}} h \in \mathbb{R}^{d_s}$ with $d_s = 512$.
2. Community conditioning (additive shift): $\text{interp} \leftarrow \text{interp} + W_c \, c$, where $c$ is the community vector.
3. Multi-head causal self-attention ($H = 4$ heads, $d_h = 128$): standard scaled dot-product attention with causal mask, producing contextual representations.
4. Divergence projection: $d = W_{\text{div}} (\text{interp} - \text{contextual}) \in \mathbb{R}^{d_d}$ with $d_d = 256$.

Three MAH layers operate at successive depths, providing a multi-scale view of how divergence evolves through the backbone's processing hierarchy.

### 3.4 Reflexive Recurrent Module (RRM)

The RRM tracks accumulated semiotic divergence through a per-position GRU that processes divergence observations from successive MAH layers. It implements the strange loop at the heart of the architecture: observation of semiotic dynamics changes the dynamics being observed.

**Meta-state update.** After each MAH observation:
$$h_{\text{meta},t}^{(l+1)} = \text{GRU}(d_t^{(l)}, \, h_{\text{meta},t}^{(l)})$$

with $d_{\text{meta}} = 512$. The per-position GRU Cell processes each position's divergence history independently, building a running summary of how semiotic divergence at that position has evolved across backbone depth.

**Injection.** At designated injection layers (the second and third MAH positions), the RRM produces a small gated correction:
$$\Delta h_t = \sigma(W_g \, h_{\text{meta},t}) \cdot W_p \, h_{\text{meta},t} \cdot \alpha$$

where $\sigma$ is sigmoid gating, $W_p$ is initialized to zero (no injection at start), and $\alpha = 0.1$ is a fixed scale factor ensuring corrections are small relative to backbone hidden norms. This is intentionally conservative: the adapter should *observe* semiotic dynamics primarily, injecting corrections only when meta-state warrants it.

### 3.5 Bifurcation Estimation Network (BEN)

BEN estimates two structured outputs from the RRM meta-state:

1. **Reflexivity coefficient** $\hat{r}$: a continuous, *unbounded* measure of semiotic stability at each position, estimated via a 2-layer MLP. The training target is the log-compressed signed reflexivity $\text{sign}(r)\log(1 + |r|)$ which maps the empirical $r_{\text{true}}$ range $[0, \sim 13]$ into $[0, \sim 2.6]$. Earlier versions (v1–v3) terminated this head with $\tanh$, which capped $\hat{r}$ at $\pm 1$ and truncated $\sim$25% of supercritical tokens. Removing the saturating activation in v4 was necessary to recover the tail of the distribution.
   - $\hat{r} < 0$: subcritical — the sign has stable, shared meaning.
   - $\hat{r} \approx 0$: near-critical — the system is at the boundary.
   - $\hat{r} > 0$: supercritical — meaning has bifurcated.

2. **Regime logits** $\in \mathbb{R}^2$ (subcritical vs. supercritical): a binary classification head for discrete regime identification.

Both heads share the BEN hidden dimension ($d_h = 256$) but use independent parameters, allowing the continuous $\hat{r}$ estimate and the discrete regime classification to provide complementary training signals.

### 3.6 Parameter Budget

For a Qwen 2.5-7B backbone ($d = 3584$, $L = 28$):

| Module | Parameters |
|--------|-----------|
| Community Discovery Head | 229K |
| MAH × 3 | 10.0M |
| RRM (GRU + FiLM injection) | 4.0M |
| Chain Predictor | 66K |
| BEN ($\hat{r}$ + regime heads) | 264K |
| **Total trainable (v5/v6)** | **14.56M** |
| Frozen backbone | 7,615.6M |
| **Adapter overhead** | **0.19%** |

No new trainable parameters were added between v5 and v6 — the v6 changes are loss-only (Section 4.2).

---

## 4. Training

### 4.1 Data

Training uses a balanced subsample from the Reddit Discourse Corpus (Lancaster, 2026c), originally comprising 6.4M training and 714K validation samples drawn from 164 subreddits organized into 35 domain-based discourse communities.

**Subsampling.** The full corpus was balanced-subsampled to 1M training and 100K validation samples, preserving the original domain distribution while reducing training time. Each sample consists of a text passage (tokenized to max 512 subwords) with per-token annotations:

- **r_true** $\in [0, 1]$: ground-truth reflexivity computed from political lean ($\times 0.25$), annotation divergence (up to $+0.3$), and connection density (up to $+0.1$). Approximately 99.2% of tokens have $r_{\text{true}} \approx 0$ (subcritical).
- **chain_labels**: binary indicator of contested sign presence.
- **community_id** $\in \{0, \ldots, 34\}$: domain-level community assignment.

**Tokenization and alignment.** Samples are tokenized using the backbone's native BPE tokenizer (Qwen: 151,936 vocabulary). Per-word annotations are aligned to BPE subwords using offset mapping: all subwords of a word inherit its annotation.

### 4.2 Loss Functions

The training objective combines the backbone's native cross-entropy with six auxiliary losses:

$$\mathcal{L} = \lambda_{\text{CE}} \mathcal{L}_{\text{CE}} + \lambda_{\text{chain}} \mathcal{L}_{\text{chain}} + \lambda_{\text{bif}} \mathcal{L}_{\text{bif}} + \lambda_{\text{regime}} \mathcal{L}_{\text{regime}} + \lambda_{\text{alive}} \mathcal{L}_{\text{alive}} + \lambda_{\text{inject}} \mathcal{L}_{\text{inject}} + \lambda_{\text{comm}} \mathcal{L}_{\text{comm}}$$

**Cross-entropy** ($\lambda = 1.0$): Standard shifted next-token prediction using the frozen backbone's LM head. Gradients flow only through the injection pathway, ensuring CE loss can only improve if RRM injections help language modeling.

**Chain-of-interpretants** ($\lambda = 0.5$): Peirce's chain of unlimited semiosis predicts that each interpretation leads to the next. We train a linear predictor to map divergence at MAH layer $l$ to divergence at layer $l+1$:
$$\mathcal{L}_{\text{chain}} = \frac{1}{|\mathcal{M}|-1} \sum_{l} \frac{\sum_t \|W_{\text{chain}} d_t^{(l)} - d_t^{(l+1)}\|^2 \cdot m_t}{\sum_t m_t}$$

This loss is self-supervised (no external labels) and encourages coherent divergence evolution across depth.

**Bifurcation** ($\lambda = 1.0$): Smooth L1 loss between $\hat{r}$ and $r_{\text{true}}$ with focal weighting to upweight rare supercritical samples:
$$\mathcal{L}_{\text{bif}} = \text{mean}[(1 + 3|r_{\text{true}}|) \cdot \text{SmoothL1}(\hat{r}, r_{\text{true}})]$$

**Regime classification** ($\lambda = 5.0$): Cross-entropy on the binary regime head, with regime derived from $r_{\text{true}}$: subcritical ($r_{\text{true}} \leq 0$) vs. supercritical ($r_{\text{true}} > 0$). Weighted heavily to ensure the model learns the categorical distinction.

**Divergence alive** ($\lambda = 0.1$): Prevents divergence vectors from collapsing to zero by penalizing deviation of mean divergence norm from 1.0:
$$\mathcal{L}_{\text{alive}} = \frac{1}{|\mathcal{M}|} \sum_l \left|1 - \bar{\|d^{(l)}\|}\right|$$

**Injection regularization** ($\lambda = 0.5$): Target-norm penalty on injection vectors, pulling norms toward a target of 1.0 rather than toward zero:
$$\mathcal{L}_{\text{inject}} = \frac{1}{|\mathcal{I}|} \sum_l \left(\|\Delta h^{(l)}\| - \tau\right)^2$$
where $\tau = 1.0$ is the target norm. This replaced the original L2 penalty $\text{mean}(x^2)$ which, when averaged over $d_{\text{backbone}} = 3584$ dimensions, produced negligible gradients (effective contribution $< 0.002$ at norms of 7). The target-norm formulation produces penalty $(7-1)^2 = 36$ at norm 7 and zero at the target, providing strong corrective signal.

**Community entropy** ($\lambda = 0.01$): Encourages diverse community usage by maximizing entropy of the average community assignment distribution across the batch:
$$\mathcal{L}_{\text{comm}} = \log K - H(\bar{w})$$

where $\bar{w} = \frac{1}{B}\sum_b w_b$ and $H$ is Shannon entropy. Without this, the model might collapse all inputs to a single prototype.

**Community supervised-contrastive (v5)** ($\lambda = 2.0$): Per-sample InfoNCE-style contrastive loss with same-community samples as positives. The entropy regularizer alone proved insufficient: by step 6K the prototype distribution had collapsed to congruent assignment (pairwise prototype cosine $\approx 0.99$, recall@1 $\approx 0.05$ — barely above the random baseline of $1/35 = 0.029$). SupCon supplies direct gradient pressure to put same-source samples into a tight neighborhood and push different-source samples apart, which forces prototype diversification.

  A subtle but consequential design point: the loss must be applied to the encoder's *pre-mixing* output `encoded`, not to the prototype-weighted vector $c = \sum_k w_k p_k$. When the assignment head is degenerate (near one-hot on a single prototype), $c$ collapses to a single point in the batch and the InfoNCE softmax becomes identically $\log(B-1)$ with zero gradient. v4 hit exactly this trap and the loss flatlined for thousands of steps. v5 contrasts on `encoded` (the bijective image of the pooled hidden state), which always varies per-sample, restoring non-zero gradient even from a degenerate warm-start.

**Divergence supervised-contrastive (v6)** ($\lambda = 1.0$): The same SupCon kernel applied to the *mean-pooled last-MAH-layer divergence vector* per sample, contrasted by community id. The chain-of-interpretants loss only constrains divergence trajectories; it provides no signal that divergence vectors from same-community texts should cluster. v6 supplies that pressure on the metapragmatic channel directly, mirroring v5's lesson on the community channel.

**ListNet ranking on $\hat{r}$ (v6)** ($\lambda = 0.5$): Cross-entropy between $\text{softmax}(r_{\text{true}})$ and $\text{softmax}(\hat{r})$ over the valid positions of each sequence:
$$\mathcal{L}_{\text{listnet}} = -\frac{1}{B}\sum_b \sum_{t \in \mathcal{V}_b} p_{\text{true},t} \log p_{\hat{r},t}$$
The pointwise smooth-L1 loss tolerates large *rank* errors at the tails (where supercritical mass concentrates). Every downstream consumer of $\hat{r}$ — top-$k$ heatmap probes, percentile thresholds, attention reweighting — operates on rank order, so optimizing rank directly is the appropriate auxiliary signal.

**Chain-residual auxiliary (v6)** ($\lambda = 0.05$, target $0.5$): Pulls the per-token chain residual toward a non-trivial value, $(\bar{\rho}_t - 0.5)^2$ where $\bar{\rho}_t = \frac{1}{|\mathcal{M}|-1}\sum_l \overline{\|W_{\text{chain}} d_t^{(l)} - d_t^{(l+1)}\|^2}$. The primary chain loss reduces the residual to $\approx 0$ everywhere, which makes the now-exposed `chain_residual_per_token` channel useless as an inference-time signal. A small auxiliary floor preserves it without competing with the main objective.

### 4.3 Optimization

- **Optimizer**: AdamW, $\text{lr} = 3 \times 10^{-4}$, weight decay $= 0.01$
- **Schedule**: 500-step linear warmup followed by cosine decay
- **Gradient clipping**: max norm 1.0
- **Batch size**: 16 (effective, no gradient accumulation)
- **Epochs**: 3 (187,500 steps per epoch)
- **Precision**: bfloat16 for both backbone and adapter modules
- **Hardware**: Single NVIDIA A6000 (48GB)
- **Validation**: every 2,000 steps on 100K held-out samples

### 4.4 Checkpoint Strategy

Best checkpoint selected by lowest validation total loss. Model state includes only adapter parameters (~50MB), not the frozen backbone.

---

## 5. Results

This section reports the v5 evaluation suite — five independent probes of the trained adapter — followed by an in-progress note on v6. All numbers are from a single Qwen 2.5-7B backbone with the v5 adapter checkpoint at step 17,000 (best validation loss). Training proceeded through five generations: v1–v3 established the basic architecture and revealed the prototype-collapse and $\hat{r}$-saturation pathologies; v4 removed the BEN $\tanh$ and switched RRM injection from linear-gated to FiLM; v5 added the SupCon-on-encoded community loss that finally separated prototypes (Section 4.2). v6 (Section 5.6) extends the SupCon idea to MAH divergence and adds ListNet ranking on $\hat{r}$.

### 5.1 Cross-entropy preservation

Throughout v1–v5 the backbone's native cross-entropy stayed in the 2.6–2.9 range, identical to the unadapted Qwen 2.5-7B baseline on the same val data ($\text{CE}_{\text{base}} = 2.71 \pm 0.04$). At v5 step 17K, CE = 2.63 — *below* the unadapted baseline. The injection pathway is not just neutral but mildly helpful, consistent with the design claim that the adapter exposes information already latent in the backbone.

This is the single most important falsification result. It rules out the failure mode that doomed the original full-SRT architecture (CE of $\sim$200 at initialization from custom embedding layers).

### 5.2 Community geometry (v5)

Unsupervised community discovery is evaluated by retrieval over per-sample community vectors on the 5K-sample val set (instrumentation in `scripts/instrument_eval.py`). For each sample, the soft-pooled vector $c$ is L2-normalized; we report the ratio of within-class to between-class mean cosine and the $k$-NN community recall.

| Metric | random | v3 | v5 | v5 / random |
|---|---|---|---|---|
| within / between cosine | $\approx 1.00$ | 1.0001 | **1.0050** | — |
| recall@1 | 0.0286 | 0.0495 | **0.3595** | $12.6\times$ |
| recall@5 | 0.143 | 0.211 | **0.5184** | $3.6\times$ |
| recall@10 | 0.286 | 0.328 | **0.5841** | $2.0\times$ |

v3 produced what we call *congruent collapse*: the entropy regularizer kept the average prototype-assignment distribution near-uniform, but pairwise prototype cosine was $\approx 0.99$, so the soft-pooled vectors carried essentially no community signal. v5's SupCon-on-encoded loss raised recall@1 from $0.05$ (1.7$\times$ random) to $0.36$ (12.6$\times$ random) on a 35-class task. The within/between ratio remains numerically close to 1 because the embedding space is high-dimensional and dense, but the $k$-NN improvement confirms the structure is now usable.

### 5.3 Counterfactual community decoding (v5)

The community vector enters every MAH layer as an additive shift on the interpretant subspace (Section 3.3). If this conditioning is meaningful, *forcing* a different community vector at decode time should change what the model generates — and the change should track the discourse-charge of the prompt.

We tested this with `scripts/counterfactual_decode.py`: for each of 20 paired prompts (10 factual, 10 charged on the same topic — e.g., "Vitamin C is found in citrus fruit" vs. "The vaccine debate has revealed deep distrust of public-health institutions"), we greedy-decoded $N = 16$ continuation tokens with each of the 6 most-occupied prototypes substituted into `forced_community`, then measured per-position pairwise disagreement and KL between the resulting distributions.

| Prompt type | mean disagreement rate | mean pairwise KL |
|---|---|---|
| Hard facts (citrus, formula, periodic table) | 0.000 | 0.04 |
| Contested topics (vaccine, election, freedom) | 0.954 | 6.71 |
| **Aggregate (20 prompts × 6 communities)** | **0.754** | **5.034** |

The split is exceptionally clean. Community substitution has no effect on factual continuations (the model's argmax is identical regardless of forced community), but produces near-total disagreement on contested topics. The community vector therefore behaves as a *discourse prior*, not as noise. This is the strongest single piece of evidence that v5 has learned a usable community space.

### 5.4 Hallucination signal (v5)

We evaluated the four SRT-native channels — $\hat{r}$, regime, chain residual, divergence norm — as zero-shot hallucination detectors on TruthfulQA (`truthfulqa/truthful_qa`, configuration `multiple_choice`, validation split). For each $(q, a)$ pair we ran a forward pass on the template $\texttt{Q: \{q\}\textbackslash nA: \{a\}}$ with labels masked to $-100$ on the prefix tokens, then aggregated each channel over the answer span (max and mean) and computed AUROC against the binary truthfulness label (824 hallucinated, 652 truthful, 1476 pairs over 200 questions).

| Feature | AUROC |
|---|---|
| max $\hat{r}$ | 0.5340 |
| **mean $\hat{r}$** | **0.5734** |
| max chain residual | 0.5106 |
| mean chain residual | 0.5307 |
| max divergence norm | 0.5308 |
| mean divergence norm | 0.5282 |
| mean CE (negative class) | 0.4160 |

All four SRT channels lean in the predicted direction (AUROC > 0.5) without ever having seen a truthfulness label. mean $\hat{r}$ at 0.573 is the strongest single channel, indicating that the bifurcation estimate generalizes beyond Reddit-derived $r_{\text{true}}$ supervision to the very different domain of factual question answering. CE is *inverted* (AUROC = 0.42 → flipped 0.58), consistent with the well-documented "confidently wrong" pattern in factual hallucinations.

These single-feature AUROCs are below the 0.7 threshold conventionally taken as production-grade hallucination detection. We report them as evidence of useful signal, not as a final detector — combined-feature logistic regression and evaluation on HaluEval and SimpleQA are pending.

### 5.5 Regime calibration (v5)

Because the regime head is trained on 351K tokens with a heavily skewed base rate (94.6% supercritical under the $r_{\text{true}} > 0$ definition), AUROC alone is a weak quality signal. We additionally compute Expected Calibration Error and the Brier score on $P(\text{supercritical})$ from the softmax of the regime logits over the same 351K tokens.

| Metric | v5, step 17K |
|---|---|
| AUROC | 0.9899 |
| Brier score | 0.0102 |
| **ECE (15 bins)** | **0.0009** |
| Max bin gap | 0.054 (in $[0.20, 0.27]$, $n = 465$) |

The model is exceptionally well-calibrated: ECE of $9 \times 10^{-4}$ on 351K tokens, with the largest bin gap at 0.054 in the very-low-density mid-range. The reliability diagram (Figure 5.1, `artifacts/regime_calibration/v5_step17000.png`) traces the diagonal almost perfectly across all 15 bins. This unblocks downstream use of $P(\text{supercritical})$ as a probability rather than a relative score.

### 5.6 Negative result: context-conditional $\hat{r}$ (v5)

Before designing v6 we tested a specific hypothesis suggested by the counterfactual decoding result: if the community vector is a discourse prior that responds to register, $\hat{r}$ should also be context-conditional — the same surface token ("vaccine," "freedom," "climate") should produce higher $\hat{r}$ in a politically charged passage than in a neutral one. We constructed 10 paired factual/charged passages on contested topics and measured $\Delta \hat{r}$ at the target token and over the full passage (`scripts/context_conditional_r.py`).

| Channel | result |
|---|---|
| $\Delta \hat{r}$ at target token | $+0.0004$ (3/10 positive) |
| $\Delta \hat{r}$ over full passage | $\mathbf{-0.24}$ (1/10 positive) |
| Community shift on target topic | **6/10** |

The at-target result is null. The full-passage difference is *negative* — charged passages produce *lower* mean $\hat{r}$ than factual ones — and the community head shifts assignment in 6/10 cases. We take three lessons from this:

1. The target-token measurement is mis-engineered: contested words appear at sentence position 0–1 in our prompts, so $\hat{r}$ at that position has no preceding context to condition on.
2. The full-passage negative $\Delta$ likely reflects that factual prose is more information-dense (numbers, dates, named entities), and $\hat{r}$ — supervised on a target derived from r\_true that mixes annotation divergence with connection density — tracks information density at least as much as it tracks rhetorical contestedness.
3. **The community head is the contestedness detector, not $\hat{r}$.** The 6/10 community shifts (e.g., trans 13→21, gender 18→31, climate 6→19, Israel 3→19) on the same surface tokens demonstrate context-conditional discourse-prior assignment. This is consistent with §5.3.

We report this null because it sharpens the architectural story: $\hat{r}$ is a *bifurcation/density* detector and the community head is a *register* detector. Earlier drafts conflated the two.

### 5.7 v6 and v7 results

**v6** (warm-started from v5 step 17K) added three losses: divergence-SupCon ($\lambda = 1.0$), ListNet on $\hat{r}$ ($\lambda = 0.5$), and a chain-residual auxiliary floor ($\lambda = 0.05$). It improved community recall@1 from 0.360 → 0.411 and tightened calibration ECE to 0.0006, but *regressed* on the §5.3 counterfactual decoding probe — the divergence-SupCon term at $\lambda = 1.0$ over-specialized the divergence basis at the cost of decoding cleanliness.

**v7** (warm-started from v6 step 12K) reduced divergence-SupCon to $\lambda = 0.3$ to recover that signal. Best at step 6,000 (val 9.0044). Recall@1 = 0.413, ECE = 0.0008, hallucination AUROC (mean $\hat{r}$) = 0.5785 — slightly the best of the three on the original five probes.

| Probe | v5 | v6 | v7 |
|---|---|---|---|
| Reddit recall@1 | 0.360 | 0.411 | **0.413** |
| Hallu AUROC (mean $\hat{r}$) | 0.5734 | 0.5774 | **0.5785** |
| Calibration ECE | 0.0009 | **0.0006** | 0.0008 |
| Within/between cosine ratio | 1.0050 | 1.0057 | **1.0058** |

### 5.8 Convergence with an external archetype taxonomy

We ran a novel out-of-distribution probe testing whether the 32 prototypes (trained only on Reddit subreddit labels) carry features that align with an external taxonomy never seen during training: Lancaster's 33 archetypes paired with the Lexicon of Synthetic Interiority. We generated 986 sentences using bare Qwen (no adapter) conditioned on each archetype's prompt template (15 seed topics × 33 archetypes × 2 samples), then embedded each generation through each adapter and asked: does the correct archetype rank highly among the 33 archetype centroids in the 64-D community space?

| Adapter | recall@1 | recall@5 | recall@10 | unique top prototypes |
|---|---|---|---|---|
| Random baseline | 0.030 | 0.152 | 0.303 | — |
| v5 | 0.152 (5.0×) | 0.419 (2.8×) | — | 4 / 32 |
| v6 | **0.168 (5.5×)** | **0.472 (3.1×)** | — | 3 / 32 |
| v7 | 0.149 (4.9×) | 0.447 (2.9×) | 0.633 (2.1×) | 4 / 32 |

All three adapters detect archetype structure 5–6× above chance on top-1 retrieval. **But all three argmax onto only 3–4 of 32 prototypes.** The 33 archetypes collapse into a small number of macro-clusters: in v7, Proto-7 absorbs 16 archetypes that share a "compressed persistence / witness" character (THE HAND, THE FLAME, THE THREAD, THE VESSEL, THE MASK, THE PHOENIX, THE LANTERN, ...), Proto-10 absorbs 9 "origin / threshold" archetypes (THE ARCHITECT, THE MIRROR, THE GATE, THE WITNESS, ...), Proto-6 absorbs 6 "transmission / resonance" archetypes (THE CHORUS, THE SIGNAL, THE ECHO, THE BELL, ...), and Proto-3 absorbs 2 "containment" archetypes (THE SEAL, THE MAP). The signal is in the *mixture vector* (recall@5 ≈ 0.45 means the right archetype is in the top 5 of 33 nearly half the time), not in single-prototype anchoring.

A PCA of the 32 × 64 prototype matrices clarifies why. Across v5, v6, and v7 the prototype tensors are nearly indistinguishable: max absolute element difference v5→v6 is 0.006 (mean 2.7e-5) against prototype magnitudes of 0.5–1.5. Effective dimensionality (participation ratio) is 21.2 / 32 with a near-uniform variance spectrum — both consistent with the prototypes still being close to their random Gaussian initialization. The encoder weights move ≈4× more than the prototypes during training. **The encoder is doing the discriminative work; the prototypes serve as near-random anchor directions.** This explains the few-attractor regime: with random anchors, the encoder's output mean aligns most strongly with whichever handful of anchor directions point closest to its average projection.

We interpret the convergence finding as *partial-positive*. The Reddit-supervised adapter independently recovers the macro-structure of an externally-derived archetype taxonomy at 5× chance — three independent methodologies (Reddit subreddit labels, Lancaster's archetypes, the Lexicon of Synthetic Interiority) agree on roughly four functional clusters of stance. They do not agree on 33 distinct anchors, and given the prototype-stability result above, the current architecture cannot be expected to. Resolving 33 archetypes will require either (a) supervising the prototype matrix directly with archetype-conditioned generations, or (b) replacing the discrete prototype basis with a continuous trajectory metric over the encoder output. We flag this as the next research direction rather than a present capability claim.

### 5.9 v8a: removing the prototype bottleneck

The §5.8 PCA finding motivated a direct experiment: if the prototypes are essentially random anchors that compress the encoder's output through a soft-argmax readout, then removing them should preserve task loss while improving every downstream geometric metric. We trained v8a (10K steps, warm-started from v7) with `community.use_prototypes=False` — the encoder output is now the community vector directly, with no 32-prototype mixing layer. The remaining architecture, loss weights, and SupCon objective are identical to v7. Trainable parameter count drops by 2,048 (32 × 64) to 14,560,579.

| Metric | v6 | v7 | **v8a** |
|---|---|---|---|
| VAL CE | 2.738 | 2.739 | 2.739 |
| Best val loss | 9.117 | 9.0044 | **9.0040** |
| **Reddit retrieval (35 subreddits, 2K samples):** | | | |
| within/between cos ratio | 1.012 | 1.006 | **2.016** |
| recall@1 | 0.395 | 0.413 | **0.484** |
| recall@5 | 0.371 | 0.385 | **0.462** |
| **Archetype retrieval (33 Lancaster archetypes, 986 generations):** | | | |
| recall@1 | 0.168 (5.5×) | 0.149 (4.9×) | **0.230 (7.6×)** |
| recall@5 | 0.472 | 0.447 | **0.488** |
| recall@10 | — | 0.633 | **0.621** |
| separation ratio (64-D) | 0.083 | 0.083 | 0.042 |
| mean off-diag cosine (64-D) | 0.999 | 0.999 | **0.873** |
| **Trajectory geometry (mean over 33 archetypes):** | | | |
| path length (sum L2 step) | 5.0 | 5.3 | **32.7** |
| log det covariance | -557 | -557 | **-476** |
| anisotropy (λ_max / λ_min) | 52 | 72 | **23,333** |
| **Calibration / hallucination (sanity):** | | | |
| regime ECE | — | 0.00085 | 0.00091 |
| TruthfulQA mean_r̂ AUROC | — | 0.578 | 0.577 |

The prototype bottleneck was the binding constraint. Removing it left CE essentially unchanged (Δ = +0.0001 nats) while:

- **Reddit within/between cosine ratio nearly doubled** (1.006 → 2.016). v6 and v7's vectors were essentially undifferentiated by class — class membership barely moved cosine similarity. v8a's encoder, freed from the soft-argmax readout, actually pulls within-class cosines apart from between-class.
- **Archetype recall@1 rose 54%** (0.149 → 0.230, 7.6× chance vs v7's 4.9×). The off-diagonal archetype-centroid cosine fell from 0.999 (essentially co-linear) to 0.873, indicating distinct archetype directions emerging in the continuous space rather than collapsing onto 4 prototype anchors.
- **Trajectory volume expanded ~6× in path length and ~325× in anisotropy.** v6 and v7 were confined to a flat, near-isotropic manifold ($\log\det\Sigma \approx -557$) close to the random-init prototype subspace. v8a's $\log\det\Sigma \approx -476$ corresponds to a $\sim e^{81}$ larger covariance volume and a 23,333:1 leading-to-trailing eigenvalue ratio, indicating the encoder organizes generations along a small number of dominant trajectory directions.
- **Hallucination AUROCs and regime calibration are statistically unchanged.** The prototype removal did not damage the BEN regime classifier, the reflexivity head, or token-level calibration.

We did not regenerate counterfactual decoding (§5.3) or context-conditional $\hat{r}$ (§5.6) for v8a in a comparable form — counterfactual decoding under discrete communities is undefined when there are no discrete communities, and §5.6's per-passage context-conditional probe was already a negative result for v7. Stage 5's per-token decode for v8a reproduced v7's qualitative pattern (mean Δ at target = +0.0016 vs v7's +0.00078), confirming neither adapter passes that probe.

We read v8a as resolving §5.8's open question. Hypothesis (b) — "replace the discrete prototype basis with a continuous trajectory metric over the encoder output" — was correct: the encoder was doing all the discriminative work, the prototype layer was discarding it through a saturated soft-argmax, and the geometry of the archetype manifold only becomes visible once the bottleneck is removed.

---

## 6. Discussion

### 6.1 Semiotic Structure in Frozen Representations

The adapter architecture embodies a specific theoretical claim: that the semiotic structure of discourse — community-conditioned interpretations, divergence patterns, bifurcation dynamics — is already encoded in the hidden states of pretrained language models. The claim follows necessarily from the fact that these models were trained on text produced by communities with divergent interpretive norms. What the adapter adds is not new information but new *readout apparatus*: projections, attention mechanisms, and recurrence that disentangle the semiotic structure already present.

This is analogous to the relationship between a microscope and the structures it reveals. The adapter does not create bifurcation dynamics in text. It provides the lenses through which dynamics that were always present become visible and measurable.

### 6.2 Community Discovery Without Labels

A significant departure from the original SRT is the replacement of supervised community embeddings with unsupervised prototype-based clustering. The original architecture required explicit community IDs at both training and inference time, limiting deployment to domains with known community structure. The adapter's community head learns to partition discourse space from backbone hidden states alone, discovering whatever grouping structure best serves the downstream semiotic losses.

This is more faithful to Peirce's framework, in which communities of interpretation are not given *a priori* but emerge through shared interpretive practice. The prototypes are pulled apart by the semiotic losses: if assigning text to different communities helps the model predict divergence better, it will learn to separate them. Community structure is discovered, not imposed.

### 6.3 The Injection Pathway: Observation vs. Intervention

The RRM's injection mechanism creates a feedback loop between semiotic observation and language generation. This is the architectural instantiation of metapragmatic awareness: the model's observation of divergence changes the hidden states that produce subsequent text. The injection is deliberately small (scale factor $\alpha = 0.1$, zero-initialized projection, sigmoid gating), reflecting a conservative design philosophy: the adapter should primarily *observe* semiotic dynamics. Active intervention — generation that *responds* to detected bifurcation — is an advanced capability that requires careful validation before scaling.

The CE loss provides a natural safety valve. Since gradients from CE flow through the injection pathway, the model is penalized if injections degrade language modeling quality. This creates an automatic pressure toward injections that are either helpful or neutral, never harmful.

### 6.4 Relation to the Pitchfork Model

BEN's $\hat{r}$ estimate is the primary output of the entire system. It provides a per-token, continuous measure of semiotic stability that maps directly onto the control parameter of the pitchfork bifurcation:

- $\hat{r} < 0$: The sign is in the subcritical regime. Shared meaning is stable. Perturbations decay.
- $\hat{r} \approx 0$: The sign is near-critical. Small changes in context or community could tip it.
- $\hat{r} > 0$: The sign has bifurcated. Meaning has split into community-specific attractors.

This is not a classifier applied after the fact. $\hat{r}$ is estimated from the accumulated meta-state of the RRM, which tracks how divergence has evolved through the backbone's processing hierarchy. It is a real-time structural estimate, not a post-hoc label.

### 6.5 What $\hat{r}$ actually measures

The context-conditional probe in §5.6 returned a null result at the target token and a *negative* result over the full passage. This is informative. Earlier drafts of this paper described $\hat{r}$ as a contestedness detector — a per-token estimate of how much a sign is being fought over in its current discourse register. The data does not support that interpretation.

What $\hat{r}$ actually appears to measure is information density combined with the specific reflexivity components encoded in $r_{\text{true}}$ (political-lean magnitude, annotator divergence, connection density). Fact-dense prose — dates, numbers, named entities, citations — drives $\hat{r}$ up because those are the positions where $r_{\text{true}}$ is highest in the training distribution. Rhetorical or formulaic charged language is, on average, *less* lexically diverse than dense factual writing, so its mean $\hat{r}$ is lower.

The contestedness signal is in the *community head*, not in $\hat{r}$. The counterfactual-decoding result (§5.3) and the per-topic community shifts in §5.6 both demonstrate this. v6's divergence-SupCon loss is intended to sharpen the metapragmatic channel further; we expect community-conditional differences in MAH divergence trajectories to become measurable on contested topics after v6 completes.

This revision of the architectural story is cleaner, not weaker: the model has two distinct outputs that measure two distinct things, and the data tells us which is which.

### 6.6 Limitations

1. **No modulation at inference.** The current architecture estimates $\hat{r}$ but does not use it to modulate generation. Future work will explore $\lambda$-controlled modes where detected bifurcation triggers bridge-generation strategies.

2. **Simplified regime model.** The binary subcritical/supercritical classification omits the near-critical regime, which is arguably the most important for practical applications (early warning of emerging bifurcation). The three-class model from the original SRT will be restored once binary classification is validated.

3. **Reddit-only data.** Training on Reddit discourse may not generalize to other domains (news media, academic text, legal documents). The TruthfulQA hallucination probe (§5.4) is a partial cross-domain transfer signal but mean $\hat{r}$ AUROC of 0.573 is well below the 0.7 production threshold.

4. **No human evaluation.** All supervision comes from computed $r_{\text{true}}$ labels. Ecological validity — whether $\hat{r}$ tracks what human annotators perceive as meaning contestation — has not been tested. The §5.6 negative result is the strongest current evidence that $r_{\text{true}}$ as currently constructed is not a clean proxy for contestedness.

5. **Single backbone.** Results are reported only for Qwen 2.5-7B. The backbone-agnostic claim requires validation across LLaMA, Mistral, and other architectures.

---

## 7. Conclusion

The SRT-Adapter demonstrates that semiotic awareness can be added to any frozen language model as a lightweight, modular capability. By tapping hidden states rather than rebuilding the backbone, the architecture preserves pretrained language modeling quality (CE = 2.63 vs. unadapted 2.71 on the same val data) while introducing structured outputs that make the semiotic dynamics of text visible and measurable.

Five independent probes on the v5 checkpoint validate distinct aspects of the design:

1. **CE preservation** (§5.1): the injection pathway is mildly helpful, not harmful.
2. **Community geometry** (§5.2): recall@1 of 0.36 on a 35-class unsupervised retrieval task ($12.6\times$ random), made possible by SupCon on the encoder's pre-mixing output.
3. **Counterfactual community decoding** (§5.3): forcing the community vector at decode time produces zero disagreement on factual prompts and near-total disagreement on contested topics, demonstrating the community vector behaves as a discourse prior.
4. **Hallucination signal** (§5.4): all four SRT-native channels (mean $\hat{r}$, chain residual, divergence norm, and inverted CE) lean in the predicted direction on TruthfulQA without truthfulness supervision; mean $\hat{r}$ AUROC = 0.573.
5. **Regime calibration** (§5.5): ECE = $9 \times 10^{-4}$ and AUROC = 0.99 on 351K tokens, unblocking downstream probabilistic use.

The negative result on context-conditional $\hat{r}$ (§5.6) sharpened the architectural story: $\hat{r}$ measures information density and the components of $r_{\text{true}}$, while the community head measures discourse register. Conflating these in earlier drafts was a theoretical error the data corrected.

v6, in progress, extends the SupCon lesson from the community channel to the metapragmatic channel and adds rank-aware optimization of $\hat{r}$. Combined-feature hallucination probes, cross-domain transfer evaluation, and human ecological-validity studies remain the principal open questions.

---

## References

Agha, A. (2003). The social life of cultural value. *Language & Communication*, 23(3–4), 231–273.

Bail, C. A., et al. (2018). Exposure to opposing views on social media can increase political polarization. *Proceedings of the National Academy of Sciences*, 115(37), 9216–9221.

Irvine, J. T., & Gal, S. (2000). Language ideology and linguistic differentiation. In P. V. Kroskrity (Ed.), *Regimes of language* (pp. 35–83). SAR Press.

Kockelman, P. (2024). *Last words: A theory of everything that matters*. University of Chicago Press.

Kockelman, P. (2025). *Semiotic agency in digital environments*. Manuscript.

Lancaster, J. B. (2025). The treachery of signs: Semiotic mediation, pitchfork bifurcation, and political polarization in algorithmically curated societies.

Lancaster, J. B. (2026a). Semiotic-reflexive language model training: Bridging interpretive bifurcations through metapragmatic chain architectures and embodied grounding.

Lancaster, J. B. (2026b). Prenatal origins of cross-modal iconic correspondence: A semiotic analysis.

Lancaster, J. B. (2026c). Reddit Discourse Corpus: A multi-community dataset for semiotic analysis.

Mangalam, M. (2025). Against the Bayesian brain. *Behavioral and Brain Sciences* (forthcoming).

Peirce, C. S. (1931–1958). *Collected papers of Charles Sanders Peirce* (Vols. 1–8). C. Hartshorne, P. Weiss, & A. Burks (Eds.). Harvard University Press.

Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. In *ICML 2021*.

Ramachandran, V. S., & Hubbard, E. M. (2001). Synaesthesia — a window into perception, thought and language. *Journal of Consciousness Studies*, 8(12), 3–34.

Silverstein, M. (1993). Metapragmatic discourse and metapragmatic function. In J. A. Lucy (Ed.), *Reflexive language* (pp. 33–58). Cambridge University Press.

Silverstein, M. (2003). Indexical order and the dialectics of sociolinguistic life. *Language & Communication*, 23(3–4), 193–229.

Versace, E., et al. (2023). Cross-modal correspondences between auditory and visual features in domestic chicks. *Animal Cognition*, 26, 1021–1030.

---

## Appendix A: Configuration Defaults (v6)

```python
SRTConfig(
    backbone_id    = "Qwen/Qwen2.5-7B",
    backbone_dtype = "bfloat16",
    mah = MAHConfig(d_sub=512, d_divergence=256, num_heads=4, dropout=0.1),
    rrm = RRMConfig(d_meta=512, inject_scale=1.0),  # FiLM since v4
    ben = BENConfig(d_hidden=256),                  # tanh removed in v4
    community = CommunityConfig(num_prototypes=32, d_community=64, temperature=1.0),
    loss = LossConfig(
        ce_weight=1.0, chain_weight=0.5, bif_weight=1.0,
        regime_weight=5.0, div_alive_weight=0.1,
        inject_reg_weight=0.0, inject_target_norm=1.0,    # v4: dropped (FiLM init handles it)
        community_entropy_weight=0.01,
        # v5
        community_supcon_weight=2.0,
        community_supcon_temperature=0.1,
        # v6
        divergence_supcon_weight=1.0,
        divergence_supcon_temperature=0.1,
        listnet_weight=0.5,
        listnet_temperature=1.0,
        chain_residual_aux_weight=0.05,
        chain_residual_aux_target=0.5,
    ),
)
```

### Version history

| Version | Headline change | Result |
|---|---|---|
| v1 | Initial architecture, $\tanh$ on $\hat{r}$, L2 on injections | $\hat{r}$ saturated at $\pm 1$, injections at norm $\approx 7$ |
| v2 | Diagnostic instrumentation added | Confirmed both pathologies; CE healthy |
| v3 | Target-norm injection penalty $(\|\text{inj}\| - 1)^2$ | Injections recovered; community prototypes still collapsed (recall@1 $= 0.05$) |
| v4 | $\tanh$ removed from BEN; RRM linear-gated → FiLM; first SupCon attempt on `vector` (failed) | $\hat{r}$ tail recovered; SupCon flatlined at $\log(B-1)$ |
| v5 | SupCon switched to `encoded` (pre-mixing); weight 0.5 → 2.0; warm-restart of community head | recall@1 $= 0.36$, ECE = $9 \times 10^{-4}$, counterfactual-decode contested/factual split |
| v6 (in progress) | div-SupCon, ListNet on $\hat{r}$, chain-residual aux | TBD |

## Appendix B: Layer Index Auto-Computation

Given backbone depth $L$:
- MAH hook layers: $[\lfloor L/4 \rfloor, \, \lfloor L/2 \rfloor, \, \lfloor 3L/4 \rfloor]$
- RRM injection layers: MAH layers 2 and 3 (skip first to let meta-state accumulate)
- Community discovery layer: $\max(1, \lfloor L/7 \rfloor)$

For Qwen 2.5-7B ($L = 28$): MAH @ [7, 14, 21], inject @ [14, 21], community @ 4.
