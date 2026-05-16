# SRT-NLA: corpus-free activation verbalization on the full backbone

Plan for the next core SRT capability: a Natural-Language-Autoencoder–style readout of the residual stream that is **trained without a labelled corpus** and uses the **full backbone as its own reverse model** — i.e. no separate truncated AR, no Anthropic-API explanations, no WildChat / FineWeb pretraining stage. Just the frozen Qwen2.5-7B + our existing SRT adapter, closed-loop.

---

## 1. What Anthropic's NLA does (and the parts we should keep)

Source: [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) ([Transformer Circuits 2026](https://transformer-circuits.pub/2026/nla/index.html)).

NLA is a pair of full-7B fine-tunes:

| Half | Direction | Mechanism |
|---|---|---|
| **AV** (activation verbalizer) | vector → text | Inject vector as a single token embedding into a fixed prompt; autoregress description. |
| **AR** (activation reconstructor) | text → vector | Truncated K+1-layer LM + `Linear(d, d)` head; extract at final token. |

Training pipeline:
1. **Data generation** — sample text from WildChat-1M + Ultra-FineWeb, extract layer-L hidden states, call **Claude API** to label each vector with an explanation.
2. **AR SFT** — MSE on raw activations.
3. **AV SFT** — next-token CE on the API-generated explanations with vector injection.
4. **RL** — GRPO on AV (simultaneous supervised AR), reward = `-mse_nrm` where `mse_nrm = ‖AR(AV(v))/‖·‖ − v/‖v‖‖² = 2(1 − cos)`.

**What we should keep:**
- The L2-normalised MSE / cosine round-trip metric — it's the right scalar reward and it's well-behaved.
- Single-token embedding injection at a fixed prompt slot — clean, isolates the verbalization channel from prompt tokens.
- The 2/3-depth extraction layer convention (Qwen7B = L20). Matches our MAH hook placement closely.
- The `nla_meta.yaml` sidecar pattern — prompt template, injection token IDs, scale factors. Good engineering hygiene we should copy verbatim.

**What we should not replicate:**
- **Two full 7B fine-tunes per checkpoint.** We have a 12.7M-parameter adapter. We should not be shipping 14B of new weights per backbone.
- **Claude API for label generation.** Cost, leakage, dependence on a frontier model to bootstrap interpretability of a smaller one — bad incentive structure.
- **External corpus (WildChat + FineWeb).** Forces a distribution-shift gap between training and any downstream domain (semiosphere news, BFI archetypes, etc.).
- **Direction-only loss.** NLA throws away magnitude. Magnitude is informative — MAH norm trends and BEN bifurcation scores both correlate with it.
- **Single extraction layer.** NLA picks L20 and ignores the rest. We already extract from multiple MAH hook points; we should verbalize all of them.
- **No clustering prior.** NLA's AV has to discover "this is about X" from scratch every time. We have a 961-community softmax (zooL4nD3r) and a 64-D continuous community space (v8a) — free interpretability handles.

---

## 2. The reverse-model trick: the backbone *is* the AR

NLA fine-tunes a truncated copy of the LM to play the AR role. But the backbone we already load *is itself* a text → vector function: feed text `T`, read `hidden_states[L]`, done.

> **AR := `lambda T: backbone.forward(T, output_hidden_states=True).hidden_states[L]`**

This collapses the NLA pipeline from "two fine-tuned 7Bs" to "one frozen 7B + one tiny adapter":

| | NLA | SRT-NLA |
|---|---|---|
| AV weights | full 7B fine-tune | SRT adapter (12.7M) + 1 injection token embedding |
| AR weights | truncated 7B + `Linear(d,d)` SFT | **0 new params** — frozen backbone forward pass |
| AR training | MSE SFT stage required | none |
| AV training | SFT (on API labels) → GRPO | GRPO only — no SFT, no labels |
| Corpus | WildChat + FineWeb + Claude annotations | backbone's own samples (self-distilled) |
| Cost per checkpoint | 2×8×H100 RL + SFT + datagen | 1×A6000 RL (existing rig) |

**Why this is sound.** Anthropic trained AR separately because (a) they wanted to avoid optimizer interference between AV and AR during RL — solvable by keeping AR frozen; (b) the truncated LM is faster than full forward — true, but cheap given we already pay full backbone forward for the SRT adapter; (c) the `Linear(d,d)` head lets AR recover the *specific* extraction layer's representation from the final-token hidden — but we *control* extraction, so we just read layer L directly and skip the linear head.

The only thing the frozen backbone can't do that NLA's AR can is "land precisely at extraction layer's geometry from the *final token's* hidden". We sidestep that by reading from the same layer L on both sides of the round trip — the metric is `‖backbone(AV(v)).hidden[L][-1] − v‖` rather than `‖AR(AV(v)) − v‖`. The geometry agrees by construction.

---

## 3. Corpus-free training: bootstrap from the model itself

NLA needs WildChat + FineWeb + Claude. We need none of them.

**Self-distilled target vectors.** Sample N unconditional generations from the frozen backbone at temperature 0.9–1.2 (sequence length 128–512). For each sample, extract `hidden_states[L]` at every position (or pooled). Those vectors are by definition in-distribution for the backbone — the only distribution the AV ever has to verbalize.

```
for step in range(N):
    text = backbone.generate(prompt=<bos>, max_new_tokens=256, temperature=1.0)
    v_targets = backbone(text, output_hidden_states=True).hidden_states[L]  # [seq, d]
    enqueue(v_targets)
```

This replaces NLA's entire datagen stage (`scripts/datagen/`, ~40% of their repo) with a single generator loop. No external API. No external corpus. Reproducible from `backbone_id + seed` alone.

**Optional: domain mix-in.** For task-specific deployments (e.g. verbalizing semiosphere community vectors) seed the generator with domain prompts instead of `<bos>`. Still no labels — the targets are still backbone-emitted activations on backbone-emitted text. Pure self-supervision.

**No SFT stage at all.** NLA uses SFT on Claude-generated explanations to keep the AV legible before RL takes over. We replace SFT with two regularizers added to the RL reward:
1. **Legibility KL** — `KL(AV-policy || frozen-backbone-base)` on the same prompt without injection. Keeps the AV from drifting into gibberish during early RL exploration.
2. **Token-entropy floor** — penalize per-token entropy below a threshold; prevents the degenerate "always output 'the the the'" attractor.

Together these substitute for the SFT stage and remove the Anthropic API dependency entirely.

---

## 4. Use the entirety of the model, not a truncated copy

NLA's AR uses L+1 layers of the backbone; everything above L is discarded. We use **all 28 layers** plus the SRT adapter's MAH/RRM hooks. Concretely:

- **AR side**: full 28-layer forward, read at layer L (same as NLA), but the surrounding layers are still active — gives the round-trip a stronger geometric attractor because the residual stream above L is consistent with the trained backbone.
- **AV side**: full 28-layer forward with the **SRT adapter active**. The adapter's existing RRM injection points become the verbalization channel — we already have the wiring for vector → residual-stream injection. The adapter's MAH hooks let us *condition* AV generation on the community / divergence channel.

This is the key architectural divergence from NLA. Their AV is a vanilla LM with a single injected embedding. Ours is a *semiotically-conditioned* generation: the adapter sees the target vector, computes community / divergence / bifurcation features for it, and shapes the residual stream throughout the entire 28-layer pass. The verbalization is therefore not just "describe this vector" but "describe this vector, given that the model has already classified it as community-K and predicted divergence-D".

---

## 5. Architecture: SRT-NLA v0

```
target vector v  ───────────────────────────────────────────┐
                                                            │
                ┌─────────── inference path ────────────┐  │
                │                                         │  │
[<bos> Verbalize: <INJ> →]  Qwen layers 0..L              │  │
                                  │                       │  │
                            inject v at slot <INJ>        │  │
                                  ▼                       │  │
                          MAH hook @ L  ─→ CommunityHead  │  │
                                              │           │  │
                                              ▼           │  │
                                       community vec,     │  │
                                       divergence vec     │  │
                                  │                       │  │
                  RRM injection at L+k  ←─── community-conditioned correction
                                  │                       │  │
                          Qwen layers L+k..28              │  │
                                  │                       │  │
                                  ▼                       │  │
                          token T1, T2, ... TM   (the verbalization)
                                  │                       │  │
                                  ▼                       │  │
                    ┌─ same frozen backbone ──────────────┘  │
                    │  hidden_states[L][-1]   =   v̂           │
                    └───────────────────────────────────────────
                                                                │
            reward = − ‖v̂/‖·‖ − v/‖v‖‖²  −  λ_norm·(‖v̂‖ − ‖v‖)²  − β·KL(AV || base) ───┘
```

**Trainable parameters:** SRT adapter (12.7M) + one new injection-token embedding (3584 d). Everything else frozen.

**Loss:**
```
R = − mse_nrm                       # NLA's reward, kept
    − λ_mag · (‖v̂‖ − ‖v‖)²          # NEW: magnitude penalty (NLA discards this)
    − β    · KL(π_AV ‖ π_base)       # NEW: legibility regularizer (replaces SFT)
    − γ    · max(0, H_min − H(π_AV))  # NEW: entropy floor
    + δ    · community_consistency    # NEW: top-k community of v̂ should match top-k of v
```

`community_consistency` uses the existing CommunityDiscoveryHead — if the verbalization round-trips through the same community cluster, the description captured the categorical structure.

---

## 6. Phasing

| Phase | Scope | Validation |
|---|---|---|
| **N0** — scaffold | `srt/nla/` package; `nla_meta.yaml` sidecar; injection-token registration; round-trip eval harness on a held-out 1K sample of random unit vectors. | Round-trip cos > 0.0 on random vecs (sanity: AV is producing *something* the backbone reads consistently). |
| **N1** — corpus-free RL | Implement self-distilled target sampler; GRPO loop with the 5-term loss above; train on 2× A6000 (existing rig). 100K steps. | Reach `fve_nrm ≥ 0.5` on a held-out backbone-sampled set, without any external data. NLA Qwen7B reference is 0.752 trained on full WildChat+FineWeb+API labels. |
| **N2** — adapter-only AV | Freeze backbone entirely, train only SRT adapter + injection embed. Validates the "12.7M parameter AV" claim. | Match N1 `fve_nrm` ± 0.03 with <0.2% the trainable params of NLA. |
| **N3** — community-conditioned AV | Add CommunityDiscoveryHead conditioning + community_consistency term. Verbalize 961 zooL4nD3r communities; spot-check 50 by hand. | Top-1 community of v̂ matches top-1 of v on ≥ 70% of held-out vectors. |
| **N4** — multi-layer verbalization | Train per-layer AV heads on every MAH hook layer (3 hooks today). Shared adapter trunk + per-layer injection embeddings. | Per-layer `fve_nrm` reported; lowest layer should be most concrete ("the word 'bank'"), highest most abstract ("financial-institution discussion register"). |
| **N5** — trajectory verbalization | Verbalize *sequences* of activations (a path through the v8a continuous 64-D community space). Treat (v₁, v₂, …, vₖ) as the input — narrative summary as output. | Sentence-transformer cosine between predicted narrative and gold news-article summary ≥ 0.5 on a 200-trajectory hold-out drawn from semiosphere. |
| **N6** — semiosphere integration | Register `nla_verbalizer` head in `engine_zoolander`; expose `POST /api/v1/signs/{id}/explain` and `POST /api/v1/communities/{id}/explain`. | Live explanations for every sign and every community in the production graph. |

---

## 7. Why this is strictly better than NLA for our use case

1. **No corpus dependency** → can be retrained for any domain by sampling from the backbone in that domain. Drop-in for semiosphere, BFI archetypes, financial-news, anything.
2. **No API dependency** → no Anthropic billing, no rate limits, no leakage of training labels back to a competitor's model.
3. **~500× fewer trainable parameters** (12.7M vs 7B + 7B). Trains on the rig we already own.
4. **Magnitude-preserving** → unlocks BEN bifurcation magnitude and MAH norm as readout signals NLA throws away.
5. **Community-conditioned** → interpretability handles ("this vector is community-42, here's what 42 is about, here's how this vector is unusual within 42") that NLA structurally cannot produce.
6. **Multi-layer** → readouts at every MAH hook give layer-resolved interpretability, not a single 2/3-depth slice.
7. **Trajectory-aware** → verbalize *change over time* in the community channel; NLA is single-vector only and has no notion of trajectory.
8. **Inference re-uses the same backbone process** → no second SGLang server, no second model in GPU memory; the AR is just a normal forward pass through the model that's already loaded for SRT inference.

---

## 8. Known risks & open questions

| Risk | Mitigation |
|---|---|
| Mode collapse from corpus-free training (AV emits the same description for all vectors) | Diversity bonus on per-prompt token distribution; reject batches with > X% description overlap. |
| Self-distilled targets are too narrow (only covers in-distribution backbone manifold) | Phase N5 adds semiosphere news-derived targets; we can also seed generation with domain prompts to widen support. |
| RL exploration without SFT warm-start might be slow | Legibility-KL + entropy-floor are designed to handle this; falls back to ~5K-sample SFT on backbone-sampled (vector, generated-caption) pairs where the caption is the *same backbone's* greedy continuation — still no external data. |
| Community-consistency reward could overfit the CommunityDiscoveryHead's biases | Hold out 10% of communities from the consistency reward; report per-community fve_nrm separately. |
| "Frozen backbone as AR" assumes our backbone == the backbone whose activations we're verbalizing. Cross-model AR (e.g. verbalize Llama activations with Qwen) is out of scope. | Document the constraint; not a problem for the SRT use case where adapter and target are bonded to one backbone by design. |

**Open question** — should the injection-token embedding be learned per-domain (one vector for `<INJ_news>`, another for `<INJ_chat>`) or shared? N1 uses shared; N3 may justify per-domain.

**Open question** — does the legibility-KL regularizer cause the AV to refuse to verbalize OOD vectors (it would rather emit base-distribution text than describe something unusual)? Test in N1 by injecting random gaussian noise and checking that AV at least *tries* to describe it.

---

## 9. Concrete next actions

N0 status (branch `nla`):

- [x] `srt/nla/` package — [config.py](../srt/nla/config.py) · [verbalizer.py](../srt/nla/verbalizer.py) · [reconstructor.py](../srt/nla/reconstructor.py) · [loss.py](../srt/nla/loss.py) · [sidecar.py](../srt/nla/sidecar.py)
- [x] [scripts/sample_targets.py](../scripts/sample_targets.py) — backbone self-sampler (pt/parquet/jsonl output)
- [x] [scripts/train_nla.py](../scripts/train_nla.py) — N1 REINFORCE loop (GRPO upgrade deferred to N1.5)
- [x] [scripts/bench_nla_n0.py](../scripts/bench_nla_n0.py) — round-trip on N random unit vectors with JSON report
- [x] [tests/test_nla_smoke.py](../tests/test_nla_smoke.py) — passes on `hf-internal-testing/tiny-random-LlamaForCausalLM` in ~5s
- [x] `HEAD_FACTORIES = {"nla": ...}` registered in [srt/__init__.py](../srt/__init__.py)

Implementation notes worth remembering:

- Adapter params (`proj`, `prefix_embeds`) are kept in **float32** even when the backbone is bf16. AdamW on low-precision params is numerically unstable; the boundary cast happens inside `ActivationVerbalizer._inject_prefix`.
- REINFORCE log-softmax runs in float32 for the same reason.
- AV and AR share a single backbone instance to halve GPU memory.
- HF returns only *new* tokens when `inputs_embeds` is supplied to `generate`, so no prefix-stripping needed.

Next (N1):

1. On the A6000, run `scripts/sample_targets.py --backbone Qwen/Qwen2.5-7B --num-sequences 1000 --seq-len 256 --out artifacts/nla/targets_q7b_L20.pt`.
2. `scripts/train_nla.py --steps 1000 --batch-size 4` smoke — target `fve_nrm` trending up from ~0.
3. Scale to 100K steps; target `fve_nrm ≥ 0.5` (NLA reference is 0.752 with two 7B fine-tunes; we expect lower with 12.7M params).

---

## 10. Re-discovering communities from the model itself (not the corpus)

The same self-distilled sampling pipeline that powers SRT-NLA training (§3) is also the right substrate for **re-running the CommunityDiscoveryHead from scratch against the backbone's own manifold**, rather than against whatever corpus the v0.1 head saw.

### The point

zooL4nD3r v0.1 contains 961 communities. That number is an artifact of one training run on one corpus — it tells us "961 prototypes were enough to separate **that** dataset under **that** loss", not "the Qwen2.5-7B residual stream has 961 attractors". The model itself almost certainly has a different, richer community structure. We can read it out directly:

> **Treat the backbone's self-sampled hidden-state distribution as the dataset.** Run CommunityDiscoveryHead on it from scratch. The communities that emerge are properties of the *model*, not of any external corpus.

### Why this is now feasible (and wasn't before)

We previously needed a labelled corpus (Reddit source-ids, BFI archetypes) to give `community_supcon` a positive-pair signal — same source → positive, different source → negative. With backbone-self-sampled data there are no labels. Two new positive-pair signals replace source-ids without any external labels:

1. **Within-trajectory positives.** Adjacent positions (or positions within a sliding window) in a single backbone-generated sequence are positive pairs. The intuition is the same as SimCSE / contiguous-trajectory contrastive learning: a generation stays mostly in the same community for a while before drifting. Window size W is a hyperparameter; W = 16–32 tokens at temperature 1.0 is the starting point.
2. **AV-verbalization positives.** Once even a weak SRT-NLA AV exists (N1), two vectors whose verbalizations share top-k content tokens are positive pairs. This is a bootstrapped signal: AV improves community discovery, better communities improve AV's `community_consistency` reward, etc. — joint EM-style schedule.

Together these give a fully unsupervised SupCon signal that does not depend on Reddit, Lancaster, or any external taxonomy.

### Architecture

Re-use the existing [CommunityDiscoveryHead](../srt/modules/community.py) with three changes:

| Change | What | Why |
|---|---|---|
| **Non-parametric K** | Initialise `num_prototypes` large (e.g. K=4096); add a "prototype-pruning" pass at every val step that drops any prototype whose softmax mass < ε for > N consecutive batches. | Lets the data choose K. 961 was preset. We don't know the true number. |
| **Within-trajectory SupCon** | Positives = same-sequence positions within window W; negatives = positions from other sequences in the batch. Drops the source-id requirement. | Removes external-label dependency. |
| **v8a continuous mode by default** | `use_prototypes=False` once K has stabilised. Encoded vector becomes the community coordinate. | The PCA finding (v7 notes in `CommunityConfig` docstring: "prototype tensors barely move from random init") says the encoder was already doing all the work. Once we're discovery-mode (not classification-mode), keep the continuous coord and use prototypes only as named anchors for the post-hoc top-k. |

### Training pipeline

Phase **N3.5** (slots between N3 and N4 in §6):

```
generate_self_samples(N=2M, seq_len=256, temp=1.0)         # 1 GPU-day on A6000
  ↓
extract_pooled_hidden(layer=L, window_pool='mean')         # streamed; ~600 GB parquet
  ↓
train CommunityDiscoveryHead with:                         # 2 GPU-days
  - within-trajectory SupCon (no labels)
  - entropy regularizer (existing)
  - prototype-pruning every val step
  ↓
auto-discover K (read off pruning trace)
  ↓
freeze, ship as `community-self-v1`
```

Training cost: same order as the original zooL4nD3r run, on data we generate for free.

### How communities get named

961 v0.1 communities were named by inspecting top-k Reddit subreddits per cluster. With no corpus there are no subreddits. Naming uses the SRT-NLA verbalizer instead:

```
for cluster_id in discovered_clusters:
    members = top_k_vectors_by_assignment(cluster_id, k=64)
    descriptions = [AV(v) for v in members]
    cluster_name = mode_or_centroid_description(descriptions)
```

Names are therefore *also* model-intrinsic — they describe what the model itself thinks the cluster is about, not what a Reddit moderator thinks.

### What this changes downstream

- **semiosphere Phase 6.2.3** (the doc one level up) currently plans to swap graph-side Louvain for the **v0.1 961-community softmax**. After N3.5 it swaps for the **self-discovered K-community head**, which is strictly better: covers the full backbone manifold, not just the regions visited by the original training corpus.
- **News articles that fall outside the v0.1 corpus distribution** (e.g. niche scientific topics, foreign-language transliterations, post-2024 events that weren't in Reddit) will now have community assignments instead of falling through to the "weakly classified" bucket.
- **The 961 number becomes an artifact**, and we can report the actual community count of the model — useful as a published research result in its own right ("Qwen2.5-7B exhibits K ≈ X stable activation attractors at layer L").

### Bootstrapping order with SRT-NLA

```
N0 — scaffold
N1 — SRT-NLA RL (uses v0.1 community head as community_consistency anchor)
N2 — adapter-only AV
N3 — community-conditioned AV (still v0.1)
N3.5 — re-discover communities from backbone (uses AV from N3 as the
        bootstrapped naming + AV-verbalization positive-pair signal)
N4 — multi-layer AV (now community-conditioned on community-self-v1)
N5 — trajectory verbalization (trajectories in the self-discovered space
      have model-intrinsic meaning; v0.1 trajectories were corpus-artefactual)
N6 — semiosphere integration: communities, explanations, and the
      semiotic regime tags are all now properties of the model itself
```

### One-line summary

The model knows its own communities; the v0.1 corpus was just a noisy way of asking it. Self-sampling lets us ask it directly.
