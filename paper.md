# Semiotic Taps: Lightweight Adapter Modules for Bifurcation Detection in Frozen Language Models

**James Burton Lancaster**

April 2026

---

## Abstract

Large language models trained on web-scale corpora absorb the semiotic bifurcations embedded in their data — divergent interpretant chains in which the same sign carries incompatible meanings across discourse communities — but have no mechanism to detect, represent, or respond to this divergence. We introduce the Semiotic-Reflexive Transformer Adapter (SRT-Adapter), a lightweight architecture (~12.7M parameters) that bolts semiotic awareness onto any frozen causal language model without modifying its embeddings, attention, or output head. The adapter operates through four modules that *tap* hidden states at selected backbone layers: (1) a **Community Discovery Head** that performs unsupervised soft clustering of discourse communities from early-layer representations; (2) **Metapragmatic Attention Heads** (MAH) that compute divergence vectors quantifying where meaning forks under community-conditioned interpretation; (3) a **Reflexive Recurrent Module** (RRM) that tracks accumulated semiotic divergence through a per-position GRU meta-state and optionally injects small corrections into the backbone stream; and (4) a **Bifurcation Estimation Network** (BEN) that estimates a continuous reflexivity coefficient $\hat{r} \in [-1, 1]$ and a binary semiotic regime (subcritical/supercritical) at each token position. Grounded in Peircean semiotics and the pitchfork bifurcation model of political polarization (Lancaster, 2025), the architecture treats the frozen backbone as a substrate on which semiotic processes are an emergent, measurable phenomenon. Training uses six auxiliary losses alongside the backbone's native cross-entropy, supervised on a corpus of 1M Reddit samples spanning 35 discourse communities with per-token reflexivity annotations. We present the theoretical motivation, full architectural specification, training methodology, and preliminary results from the first training run on a Qwen 2.5-7B backbone.

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

1. **Adapter architecture for semiotic awareness.** We specify a complete, working architecture that adds bifurcation detection to any frozen causal LM through four lightweight modules totaling ~12.7M parameters. The design preserves pretrained language modeling quality while adding structured semiotic outputs.

2. **Unsupervised community discovery.** Rather than requiring predefined community labels at inference time, the adapter discovers discourse communities from backbone hidden states through learned prototype-based soft clustering, enabling community-conditioned divergence detection on arbitrary text.

3. **Training methodology with six auxiliary losses.** We define a multi-objective training pipeline that supervises semiotic modules on chain prediction, bifurcation estimation, regime classification, divergence health, injection regularization, and community diversity — all alongside the backbone's native cross-entropy.

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

1. **Reflexivity coefficient** $\hat{r} \in [-1, 1]$: a continuous measure of semiotic stability at each position, estimated via a 2-layer MLP with Tanh output.
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
| RRM (GRU + injection) | 2.1M |
| Chain Predictor | 66K |
| BEN (r̂ + regime heads) | 264K |
| **Total trainable** | **12.7M** |
| Frozen backbone | 7,615.6M |
| **Adapter overhead** | **0.17%** |

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

**Injection regularization** ($\lambda = 0.01$): L2 penalty on injection vectors, ensuring corrections remain small relative to backbone hidden states.

**Community entropy** ($\lambda = 0.01$): Encourages diverse community usage by maximizing entropy of the average community assignment distribution across the batch:
$$\mathcal{L}_{\text{comm}} = \log K - H(\bar{w})$$

where $\bar{w} = \frac{1}{B}\sum_b w_b$ and $H$ is Shannon entropy. Without this, the model might collapse all inputs to a single prototype.

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

## 5. Preliminary Results

*This section reports early training metrics from the first run. Full evaluation will follow in a subsequent revision.*

### 5.1 Early Training Dynamics (Steps 100–300)

Training was conducted on a single NVIDIA A6000 (48 GB) with the Qwen 2.5-7B backbone frozen in bfloat16. The adapter's 12.7M trainable parameters were optimized with AdamW (lr = $3 \times 10^{-4}$, linear warmup over 500 steps, cosine decay). Diagnostic instrumentation was added at step 0 of the current run to log divergence norms, injection magnitudes, and $\hat{r}$ distribution statistics at every logging interval.

| Step | Total | CE | Chain | Bif | div\_norms (L7/L14/L21) | inj\_norms (L14/L21) | $\hat{r}$ mean ± std [min, max] | LR |
|------|-------|----|-------|-----|------------------------|---------------------|--------------------------------|-----|
| 100 | 20.47 | 2.78 | 7.78 | 11.01 | 8.9 / 8.3 / 18.5 | 0.03 / 0.14 | 0.89 ± 0.07 [−0.21, 0.99] | 6e-5 |
| 200 | 13.97 | 2.67 | 1.10 | 9.71 | 4.1 / 6.0 / 11.1 | 0.79 / 2.23 | 0.76 ± 0.28 [−0.45, 1.00] | 1.2e-4 |
| 300 | 10.99 | 2.64 | 1.39 | 6.47 | 4.4 / 5.6 / 8.3 | 3.44 / 5.78 | 0.73 ± 0.28 [−0.65, 1.00] | 1.8e-4 |

**Observations:**

1. **CE stability (2.64–2.78).** Cross-entropy remains near pretrained quality throughout, confirming the core design claim: the frozen backbone's native LM head is not degraded by adapter injection. This stands in sharp contrast to the original SRT's CE of ~200 at initialization.

2. **Chain convergence.** Chain loss drops from 7.78 → 1.10 within 100 steps after warmup begins, indicating the linear chain predictor rapidly learns to map divergence at layer $l$ to layer $l+1$. This is the fastest-converging loss, consistent with its simple regression structure.

3. **Divergence vectors are alive.** Mean L2 norms of 4.4–18.5 across the three MAH hook layers confirm the divergence subspaces are not collapsing. Layer 21 consistently produces the largest divergence, suggesting that deeper representations carry more semiotic information — consistent with the Peircean expectation that later interpretants incorporate more community-specific processing.

4. **Injection magnitudes are small but growing.** Injection norms rose from 0.03/0.14 at step 100 to 3.44/5.78 at step 300, relative to a backbone hidden norm of ~60 ($\sqrt{3584}$). At step 300, injections represent ~5–10% of the hidden state norm — large enough to carry signal, small enough to not corrupt the backbone. The zero-initialized projection and sigmoid gating are functioning as designed.

5. **$\hat{r}$ desaturation.** At step 100, BEN produced a near-constant $\hat{r} \approx 0.89$ (std = 0.07), indicating Tanh saturation. By step 300, the distribution has spread to mean 0.73 ± 0.28 with min reaching −0.65. The bifurcation loss is successfully driving BEN away from the trivial constant-prediction solution. If this trend continues, $\hat{r}$ should cover the full [−1, 1] range by step 1000.

### 5.2 Expected Training Trajectory

Based on the loss structure and learning rate schedule:
- Chain loss should stabilize below 0.1 once lr reaches peak at step 500 and the predictor fully converges.
- Bifurcation loss will continue to decrease as $\hat{r}$ predictions spread to match the r\_true distribution (99.2% subcritical, with focal weighting upweighting the rare supercritical cases).
- CE should remain near 2.6–2.9 throughout. Any sustained climb above 3.5 would indicate injection harm.
- Divergence norms should stabilize, with the divergence\_alive loss keeping them from collapsing to zero.
- First validation checkpoint at step 2000 will provide the first generalization signal.

*Full training curves, validation metrics, and evaluation against the falsification criteria will be added upon completion of the three-epoch run.*

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

### 6.5 Limitations

1. **No modulation at inference.** The current architecture estimates $\hat{r}$ but does not use it to modulate generation. Future work will explore $\lambda$-controlled modes where detected bifurcation triggers bridge-generation strategies.

2. **Simplified regime model.** The binary subcritical/supercritical classification omits the near-critical regime, which is arguably the most important for practical applications (early warning of emerging bifurcation). The three-class model from the original SRT will be restored once binary classification is validated.

3. **Reddit-only data.** Training on Reddit discourse may not generalize to other domains (news media, academic text, legal documents). Cross-domain evaluation is needed.

4. **No human evaluation.** All supervision comes from computed $r_{\text{true}}$ labels. Ecological validity — whether $\hat{r}$ tracks what human annotators perceive as meaning contestation — has not been tested.

5. **Single backbone.** Results are reported only for Qwen 2.5-7B. The backbone-agnostic claim requires validation across LLaMA, Mistral, and other architectures.

---

## 7. Conclusion

The SRT-Adapter demonstrates that semiotic awareness can be added to any frozen language model as a lightweight, modular capability. By tapping hidden states rather than rebuilding the backbone, the architecture preserves pretrained language modeling quality while introducing structured outputs — divergence vectors, community assignments, reflexivity estimates, and regime classifications — that make the semiotic dynamics of text visible and measurable.

The theoretical framework connects these architectural choices to a rich tradition in Peircean semiotics, linguistic anthropology, and nonlinear dynamics. The adapter does not merely detect "bias" or "toxicity" — it estimates the control parameter of a pitchfork bifurcation that governs whether shared meaning is stable or actively forking. This is a fundamentally different kind of output from anything current alignment methods produce.

Preliminary training confirms the core design claim: cross-entropy starts at pretrained quality (2.73), the semiotic modules receive meaningful gradient signal, and the system fits within the memory of a single A6000. Full training curves, validation metrics, and evaluation against the falsification criteria will be reported in subsequent work.

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

## Appendix A: Configuration Defaults

```python
SRTConfig(
    backbone_id    = "Qwen/Qwen2.5-7B",
    backbone_dtype = "bfloat16",
    mah = MAHConfig(d_sub=512, d_divergence=256, num_heads=4, dropout=0.1),
    rrm = RRMConfig(d_meta=512, inject_scale=0.1),
    ben = BENConfig(d_hidden=256),
    community = CommunityConfig(num_prototypes=32, d_community=64, temperature=1.0),
    loss = LossConfig(
        ce_weight=1.0, chain_weight=0.5, bif_weight=1.0,
        regime_weight=5.0, div_alive_weight=0.1,
        inject_reg_weight=0.01, community_entropy_weight=0.01,
    ),
)
```

## Appendix B: Layer Index Auto-Computation

Given backbone depth $L$:
- MAH hook layers: $[\lfloor L/4 \rfloor, \, \lfloor L/2 \rfloor, \, \lfloor 3L/4 \rfloor]$
- RRM injection layers: MAH layers 2 and 3 (skip first to let meta-state accumulate)
- Community discovery layer: $\max(1, \lfloor L/7 \rfloor)$

For Qwen 2.5-7B ($L = 28$): MAH @ [7, 14, 21], inject @ [14, 21], community @ 4.
