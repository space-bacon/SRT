# SRT-NLA: Status, Capabilities, Implications, Use Cases, Premortem

*Snapshot: 2026-05-16, after the target-generation bug fix (commit `902b746`, branch `nla`).*

This document is the single source of truth for **what the SRT-NLA system currently does, what it could do, what we just learned, and what is most likely to go wrong**. It supersedes the "Honest expectations" section in [nla_mission.md](nla_mission.md) and the phase plan in [SRT_NLA_PLAN.md](SRT_NLA_PLAN.md) §6 until the next snapshot is written.

---

## 0. TL;DR

1. **The entire prior NLA training history (N1a–N2-v3) is invalid.** All runs trained against a 10K target file that was secretly **one constant vector repeated 10000 times**, because Qwen2.5 sets `bos_token_id == eos_token_id == 151643` and the sampler's EOS detector flagged the BOS prompt at position 0 as the first EOS.
2. **The bug is fixed** (`scripts/sample_targets.py`, commit `902b746`). Regeneration of `targets_q7b_L20_10k.pt` is running on the remote A6000 / RTX PRO 6000 box.
3. **The "0.6181 REINFORCE plateau" was not a ceiling.** It was AV memorising the single best constant text for the single repeated target. There is currently **no measured ceiling**. We will know the true baseline only after the regenerated targets land.
4. **A second, milder issue remains**: real L20 hidden norms are ≈107, token-embedding norms are ≈0.79. With `proj=eye` init the injection prefix is ~135× larger than a normal embedding token. Tractable (the linear `proj` is trainable and can absorb scale), but worth watching.
5. **Path to 0.85** is unchanged in shape (PPO-lite → soft bridge → scale), but every probability estimate from the previous plan should be treated as untrustworthy until re-measured on real data.

---

## 1. What the system actually does (capability inventory)

### 1.1 Today, verified on the smoke run with the bug fix in place

| Capability | Status | Evidence |
|---|---|---|
| Sample diverse target activations from a frozen backbone | ✅ working | 16-sample smoke: per-element std across samples = 1.24; pairwise off-diag cosine 0.275 mean (0.534 max); norms 107 ± 15.6 |
| `ActivationVerbalizer.generate(v)` → text | ✅ working | Used at every training + eval step in `scripts/train_nla.py` |
| `ActivationReconstructor.reconstruct(ids, attn)` → vector at layer L | ✅ working | Layer-20 last-token pool, no trainable params |
| Round-trip metric `fve_nrm = 1 − mse_nrm / 2` (direction-only after L2-norm) | ✅ working | Computed every val step in `_val_fve` |
| REINFORCE on `fve_nrm` reward with KL + entropy regularisation | ✅ working | `scripts/train_nla.py`; soft-bridge variant in N2 code path |
| Soft-bridge variant (forward AR on `softmax(logits) · E` instead of sampled ids) | ⚠️ implemented but harmful in the only run on real(ish) targets — see §3.1 | N2-v1/v2/v3 logs; expected to be re-evaluated on real targets |
| BoN curation: K rollouts per target → keep argmin-MSE → JSONL | ✅ implemented, smoke-tested (degenerate data) | `scripts/curate_bon.py` |
| SFT distillation on curated `(target, gold tokens)` pairs with prefix-injection masked CE | ✅ implemented, smoke-tested (degenerate data) | `scripts/train_nla_sft.py` |
| End-to-end Phase C launcher (curate → SFT, iterable) | ✅ scripted | `scripts/run_phase_c.sh` |

### 1.2 What we now *suspect* the system will be able to do, once trained on real targets

These are the previously-claimed capabilities from `nla_mission.md` §"What success unlocks". The bug discovery does not invalidate the *aspirations*, but every quantitative claim ("80× compression with bounded loss", "sub-10ms safety monitor") becomes provisional until measured on real diverse data.

- **Activation → English translator** at a chosen extraction layer.
- **Text-controlled steering** by encoding a desired-behavior sentence and injecting at the same layer.
- **Realtime safety monitor** (classify/regex the verbalization for concept presence).
- **Activation compression**: ~7 KB float32 vector → ~90 B text.
- **Cross-model bridge** (lossy translator between embedding spaces).
- **Synthetic paired (text, activation) corpus** for downstream probes / distillation, with no human or API labels.

### 1.3 Future-state capabilities tied to other SRT modules

These are unlocked by combining a working AV with the rest of the SRT stack (`MAH`, `RRM`, `BEN`, `CommunityHead`):

- **Per-layer verbalization.** Train one AV head per MAH hook layer. Read lower-layer descriptions as concrete ("the token 'bank'") and higher-layer descriptions as abstract ("financial-institution discourse register").
- **Community-conditioned verbalization.** Inject the v8a 64-D continuous community coordinate alongside the target — describe a vector *as a member of its community* rather than in isolation.
- **Trajectory verbalization.** Given a sequence of activations along a generation, emit a narrative summary of the model's internal state evolution. The same loop generates training pairs for downstream regime/community classifiers.
- **Self-rediscovery of communities.** Use the AV's descriptions as a positive-pair signal for an unsupervised CommunityDiscoveryHead retrain — [SRT_NLA_PLAN.md §10](SRT_NLA_PLAN.md) covers this.
- **Bifurcation-point captions.** BEN's r̂ marks supercritical positions; AV captions of those positions become a direct readout of *what is being contested* at the meaning-fork moment.
- **Audit handle for chain-of-thought.** A frozen-backbone audit stream that doesn't depend on the model emitting its reasoning as text.

---

## 2. Implications of the May-16 findings

### 2.1 What is definitely true

1. **No prior result on this branch was real.** Every fve_nrm number reported in the chat log, `artifacts/nla/n1*`, `artifacts/nla/n2*`, the BoN smoke (`fve_best=0.6199`), and the SFT smoke (`val_fve=0.6175`) was a measurement of "how well AV reconstructs **one** constant vector". The numerical floor (~0.617) is the cosine between the single mystery target and the L20 hidden state of AV's best-constant-output text. It carries zero generalisation signal.
2. **"4-decimal val agreement across independent runs" is now explained.** It wasn't reproducibility — it was the val pass scoring the same constant target every time.
3. **The mode-collapse / "AV ignores v_target" pathology** that triggered the dataset deep-dive was *exactly correct as a diagnosis*, but the cause was upstream: there was no `v_target` variation to ignore.
4. **The soft-bridge failure of N2 is uninterpretable.** With a constant target, the soft path and the sample path collapse to the same trivial objective. Whether soft-bridge actually helps or hurts on real data is now an open question.
5. **The norm 12232 we kept seeing** was the BOS-position residual stream, not a typical mid-layer activation. The real L20 norms cluster at 107 ± 15.6. **Every previous worry about "inject scale 18000× normal" was overstated by ~140×.** The true mismatch is ≈135×, not 18000×.

### 2.2 What is now in doubt

1. **The "REINFORCE plateau" hypothesis.** We do not currently know whether REINFORCE on real diverse targets plateaus at 0.6, 0.4, 0.7, or something else. The N1g/N1h chatter about needing PPO-lite to escape the plateau is unsupported by any real data.
2. **The need for the soft-bridge.** If REINFORCE on real data exceeds 0.65 on its own, the soft-bridge becomes optional / experimental rather than load-bearing.
3. **Phase C (BoN + SFT distillation).** This was designed as a Hail-Mary because REINFORCE was thought to be stuck. Worth re-evaluating only after we see how vanilla REINFORCE behaves on real targets.
4. **Headline probability estimates** in [nla_mission.md](nla_mission.md) ("P(≥0.65) ≈ 70%", etc.) are now stale; do not quote them.

### 2.3 What this changes operationally

| Area | Before May 16 | After May 16 |
|---|---|---|
| Confidence in any fve_nrm number from before today | High (incorrectly) | Zero |
| Next-step priority | Phase C (BoN+SFT) | (a) Sanity-check that regenerated targets pass `pool.std(dim=0).mean() > 0.1`, (b) re-measure baseline by running `_val_fve` of the existing warm-start AV on the real targets, (c) only then choose between vanilla N1 continuation / Phase 2 / Phase C |
| Pre-training data assertion | Implicit | **Required**: `scripts/sample_targets.py` should refuse to write a file where `pool.std(dim=0).mean() < 0.05`, and `train_nla.py` / `train_nla_sft.py` should refuse to start training on such a file |
| GPU-hour debt from invalidated runs | n/a | ~2 weeks of A6000 / RTX-PRO-6000 wall clock plus all human triage time. Treat as the cost of not having the sanity check. |

### 2.4 What this *does not* change

- The architectural plan in [SRT_NLA_PLAN.md](SRT_NLA_PLAN.md) §§1–5 still stands. Frozen-backbone-as-AR, single trainable `proj` + prefix, self-distilled targets, no external corpus — all of this is sound.
- The mission framing in [nla_mission.md](nla_mission.md) §"Why this matters" is unchanged.
- The published v1.0 / v8a / v18 / v21a / v22c_a050 adapter checkpoints are **completely independent** of the NLA work and are unaffected.

---

## 3. Use cases (now and future)

### 3.1 Already-validated use cases (SRT adapter, pre-NLA)

These ship today via the published checkpoints and are *not* affected by anything in this document:
- Discourse-community classification (961-class softmax, v0.1 / v8a continuous 64-D).
- Sentence embedding / STS (v18, v21a, v22c_a050 souped).
- Regime / r̂ readouts (BEN).
- Live demos on HF Spaces.

### 3.2 Use cases that become available with a working NLA at fve_nrm ≥ 0.65

| Use case | Customer / audience | Why it needs NLA specifically |
|---|---|---|
| **Internal monitor**: tap any transformer at layer L, get a stream of natural-language summaries of what the model is thinking at each token. | Interp / safety teams, alignment researchers. | Bypasses the "model didn't say it out loud" problem of chain-of-thought audits. |
| **Concept-presence classifier without probe training.** | Red teamers, content-moderation researchers. | Verbalize → regex / classify the verbalization. No per-concept probe-training step. |
| **Activation compression / replay buffer.** | Anyone training distillation models. | ~80× compression, text-shaped storage (cache, version-control, diff). |
| **Steering by natural-language description.** | Product builders. | Write the intent in English, encode, inject. No prompt engineering for the *base* model — the steering signal is mid-stream. |
| **Cross-model translator** (lossy). | Open-source ecosystem. | Same verbalization round-trips through other LMs at degraded but functional fidelity. |
| **Synthetic paired (vector, text) datasets** for downstream probes. | Interp researchers. | No Claude API, no human labels, no corpus. |

### 3.3 Use cases that become available at fve_nrm ≥ 0.75

| Use case | Why the higher fidelity matters |
|---|---|
| **Reliable text-driven interventions.** Verbalize → edit text → re-encode → inject and *measurably steer* model behavior. | Below 0.75 the encode-edit-decode loop accumulates too much loss. |
| **Activation diff narratives.** "Here is what changed between layer-L hidden at token t and token t+1." | Requires AV to capture per-token variation, not just topic. |
| **Per-layer concreteness curve.** Lower layer "the word 'bank'" → upper layer "register-shift toward financial discourse". | Needs AV to be faithful enough that the layer-to-layer abstraction is in the description, not in the noise. |

### 3.4 Use cases that become available at fve_nrm ≥ 0.85

| Use case | Why it requires 0.85+ |
|---|---|
| **Hard quantitative bound on mid-layer redundancy** ("activations are *strings*"). | Becomes a publishable interp claim only when the residual MSE is small enough that text is provably a sufficient statistic. |
| **Activation-shaped DSL** for steering / debugging. | Engineering tools assume the description is reversible-enough to round-trip in production. |
| **Cross-model alignment via shared text channel.** | Cross-model fidelity drops further than same-model; 0.85 same-model is roughly 0.7 cross-model in expectation. |

### 3.5 Use cases tied to SRT-specific structure (not generic NLA)

These require *both* an NLA-quality AV and the existing SRT modules (`MAH`, `RRM`, `BEN`, `CommunityHead`). They are the actual differentiator vs Anthropic's NLA work:

- **Community-conditioned descriptions** ("this vector, in the context of community 42").
- **Bifurcation-point captions** (BEN says r̂ > θ at token t → AV captions exactly that position).
- **Trajectory-as-narrative** in the v8a continuous space.
- **Self-rediscovered communities, named by AV** (no Reddit, no external taxonomy).
- **Semiosphere graph integration**: every sign and every community gets a model-intrinsic explanation, exposed as `POST /v1/signs/{id}/explain` and `POST /v1/communities/{id}/explain`. See SRT_NLA_PLAN.md §6 (N6).

---

## 4. Premortem

Format: each row is a way this can fail, with the strongest available early-warning signal and the mitigation if we see it.

### 4.1 Data-pipeline failure modes

| Failure | Early warning | Mitigation |
|---|---|---|
| Sampler degenerates again (silent — different mechanism, e.g. all-EOS-on-first-token sampling at a temperature setting). | `pool.std(dim=0).mean() < 0.05` or `pool.norm(dim=-1).std() < 1.0` post-generation. | **Hard assertion** in `sample_targets.py` before write; hard assertion in both training scripts before step 0. Both added as a follow-up commit. |
| Self-sampled targets are too narrow (entire training distribution is "Qwen unconditional pretrain mode"). | Per-cluster fve_nrm dispersion is low; verbalizations look generic. | Add domain-prompt seeding (semiosphere news, BFI archetypes) as additional target shards. |
| Targets contain so much PII / personal text that we cannot publish them. | Manual spot-check of `sequences`. | Run a quick PII filter before publishing the artifact; keep the (vector, text) pairs internal if needed. |

### 4.2 Training-loop failure modes

| Failure | Early warning | Mitigation |
|---|---|---|
| **Mode collapse** (AV emits the same string for every target). | Pairwise edit-distance of decoded BoN candidates → 0; per-token entropy collapses below H_min within first 1K steps. | Existing entropy-floor + KL-to-base already implemented. If they fail, add diversity-bonus term as in NLA. |
| **Reward hacking via length.** AV learns that longer rollouts get higher fve. | Mean rollout length saturates at `max_new_tokens` with no MSE improvement. | Cap `max_new_tokens` low (~32–48) early; only raise after fve clears 0.55. |
| **`inj_norm` ≫ `embed_norm` blows up attention.** With proj=eye, inject is ~135× a normal token embed. Backbone attention may degenerate even on real targets. | `attn_weight[:, :, 0, 1+P:]` (attention from generated tokens *to* the inject prefix) saturates above 0.9 across heads. | (a) Re-init `proj` as `eye * (E_rms / target_rms)` ≈ `eye * 0.0074`; (b) or add an RMSNorm-style rescale inside `_inject_prefix`. Either patch is small. |
| **REINFORCE high variance.** A handful of outlier sequences dominate every update; `clip_grad_norm` drowns the signal. | `pg_loss` per-batch percentiles diverge; `clip_frac` consistently > 0.5. | PPO-lite (ratio-clip + adv-clip + 2 inner epochs) was the N1i plan and is still the right next step. |
| **Soft-bridge bias re-emerges.** Forwarding `softmax(logits) · E` lives off the one-hot manifold, AR's hidden for it doesn't match real-token hidden. | `soft_loss` rises monotonically while `pg_loss` falls. | Use soft-bridge only as warmup (α anneal 1→0 in first ~5K steps). If still harmful, drop it; vanilla REINFORCE with PPO-lite may be enough. |
| **AV ignores v_target entirely** (different cause than the dataset bug). Could happen if `proj` learns to zero out. | `‖proj.weight‖_F` decays toward 0; AV's outputs are independent of v. | Add weight-decay floor; monitor `‖proj‖`; consider a "v injection magnitude reward" — gradient through *how much* AV's output depends on v. |
| **AR (frozen backbone last-token pool) is the actual bottleneck.** No matter how good AV gets, AR cannot resolve the target. | Even oracle text (decode v's nearest-neighbor sequence from a held-out backbone corpus) gets fve < 0.7. | Add a tiny trainable `Linear(d,d)` AR head as in NLA; we explicitly rejected this for purity reasons, but it's an escape valve worth keeping in mind. |

### 4.3 Architectural / fundamental failure modes

| Failure | Early warning | Mitigation |
|---|---|---|
| **Bit-budget cap.** 30 tokens × log₂|V| ≈ 500 bits; target carries ≈57K bits (3584 × 16). The information-theoretic ceiling is around 1% of the target's bits → fve cap may be far below 0.85 for legible text. | Greedy text-only fve saturates well below the metric ceiling even at very long sequences. | Allow longer rollouts (up to 96–128 tokens); accept neuralese (let KL-to-base relax) past a fidelity threshold; or train a "compressed-token" variant where AV outputs from a learned vocabulary that's denser per-token. |
| **Mid-layer redundancy is just not that high.** Real upper bound on legible-text fve at L20 may be ~0.7. | Multiple independent recipes (REINFORCE, soft-bridge, BoN+SFT, PPO-lite) all converge to the same ceiling. | Re-frame the paper: "we measured the legibility-fidelity Pareto frontier at layer L". Still publishable, less headline. |
| **Frozen-Qwen specifics.** Layer 20 of *this* model happens to be ill-suited; another layer or model would be better. | Same ceiling across recipes; different layer (L12, L27) shows different ceiling. | Multi-layer ablation in N4. Plan was to do this; it now becomes a higher-priority probe of the true ceiling. |
| **AV capacity ceiling.** 12.7M params for `proj + 1 prefix embed` are not enough. | Loss curve at convergence is the same for 12.7M and a 4× scaled-up variant. | Scale AV (add trainable LoRA on the first few layers; expand prefix; add small MLP on `proj`). Stops being "12.7M parameters" claim, becomes "≤100M". |

### 4.4 Process failure modes (us, not the model)

| Failure | Early warning | Mitigation |
|---|---|---|
| **Another silent-data bug.** Today's bug went undetected for weeks because every metric "agreed". | New runs cross-validating to the 4th decimal across recipes. That's now a red flag, not a green one. | Add a "data-fingerprint" line at the top of every train log: `sha256` of the targets file + `pool.std(dim=0).mean()` + `norm.mean(),std()`. Cross-check between launches. |
| **Tunnel vision on REINFORCE because it was the original plan.** | Three weeks of "the next clip / KL / entropy tweak will fix it". | Time-box. If real-targets baseline doesn't clear 0.55 within one focused training cycle, escalate to soft-bridge or SFT-bootstrap by date, not by hope. |
| **Premature publication.** Tempting to write up the moment we clear 0.65 on real targets to recover from the lost time. | First public claim that fails to reproduce. | Require: (a) seed-pinned reproducibility across two independent runs, (b) L20 *and* one other layer in the report, (c) at least the L1 cross-model bridge sanity check. |
| **Compute budget exhausted on debugging vs training.** | We did ~2 weeks of training on bad data. | Reset budget conversation with the funder / decision-maker; current cost is sunk. |

### 4.5 What I am *not* worried about

- Reproducibility of the bug fix. The fix is one line, deterministic, and the smoke shows the expected post-fix data shape.
- Re-running the regeneration. ETA ~45 min on the existing rig. Cheap.
- Published v1.0 / v8a / etc. checkpoints. Entirely independent codepath, untouched.
- The conceptual story. The architectural argument in [SRT_NLA_PLAN.md](SRT_NLA_PLAN.md) §2 (frozen backbone *is* the AR) survives this completely.

---

## 5. Action queue (post-regen)

1. ✅ Fix `sample_targets.py` — done (`902b746`).
2. ⏳ Regenerate `targets_q7b_L20_10k.pt` — running.
3. ⬜ Sanity-assert script: `python -c "assert torch.stack([a[-1] for a in torch.load('artifacts/nla/targets_q7b_L20_10k.pt')['activations']]).std(dim=0).mean() > 0.1"`.
4. ⬜ Add the same assertion as a hard precondition at the top of `train_nla.py` and `train_nla_sft.py` (load-time, before model init).
5. ⬜ Add data-fingerprint line to train log header: file sha256, N, `pool.std`, `norm.mean ± std`.
6. ⬜ Run `_val_fve` on the existing warm-start AV (`artifacts/nla/n1i_v2_best/av_step002500.pt`) against the **new** targets to establish the real baseline. This is the first honest number we will have.
7. ⬜ Based on (6), choose the next training recipe (vanilla N1 continuation, PPO-lite, soft-bridge, or Phase C). Estimates will be re-derived from observed loss curves, not from the old plan's stale probabilities.
8. ⬜ Decide whether `proj` init needs rescaling (`eye * 0.0074`) or whether learned `proj` absorbs the 135× scale gap on its own — measured, not guessed.

---

*Last updated 2026-05-16. Next snapshot will be written after step (6) — the first honest baseline number on real targets.*
