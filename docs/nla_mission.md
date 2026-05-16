# SRT-NLA: Mission & Stakes

Quick-reference doc for "why are we doing this and what does success look like." Written mid-N1h run, May 2026.

## The system in one paragraph

A frozen Qwen2.5-7B has a hidden state at layer 20 (3584-dim fp16). We train a small (~12.8M-param) **Activation Verbalizer (AV)** that, given such a target vector, produces a short text sequence. That text is re-encoded by the same frozen Qwen; we read its L20 hidden state back out; we measure how close that reconstruction is to the original target. The round-trip metric is `fve_nrm = 1 - ||v_tgt - v_hat||² / ||v_tgt||²`. AV is trained by REINFORCE on this MSE reward — **no paired (activation, text) corpus, ever**. That's the "corpus-free" part, and it's the novel claim.

## Why this matters

NLA = Natural Language Autoencoder. Prior work required paired data (activations labeled with descriptions). We're showing the bridge can be learned from MSE alone, which means it generalizes to any layer, any model, any domain where you can sample activations — i.e. all of them.

## Headline thresholds

| fve_nrm | What it means | What we ship |
|---|---|---|
| ~0.50 | Verbalizer is doing *something* but noisy. | Demo only. |
| **0.60** | Lossy summarization. Usable for qualitative interp. | Internal tool. (Current SOTA: N1g/N1h ≈ 0.618.) |
| **0.65** | Crosses the credibility line for corpus-free NLA. Downstream probes/steering become tractable. | Paper headline + activation→English translator. |
| **0.75** | Verbalization captures most behaviorally-relevant variance. Text can drive reliable interventions. | Steering API; cross-model bridges become plausible. |
| **0.85** | Text effectively *substitutes for* the activation. ~80× compression with bounded loss. Hard quantitative bound on mid-layer redundancy. | "Hidden state has a string type." Top-tier interp venue. |

## What success unlocks (practical)

1. **Activation → English translator.** Read any L20 state on any prompt as a sentence. Audit chain-of-thought *internally*, not just from emitted tokens.
2. **Text-controlled steering.** Write a sentence describing desired behavior → encode → inject at L20. Reverse direction works too: diff two activations, verbalize the delta.
3. **Realtime safety monitor.** AV forward pass at every token; classify/regex the output for concept presence ("did the model just internally consider X?"). Sub-10ms overhead per token at the target layer.
4. **Activation compression.** ~7 KB vector → ~90 B text. Cache, retrieve, version-control, diff using ordinary text infrastructure.
5. **Cross-model bridges.** Same English string can round-trip through other LMs at degraded but functional fidelity — lossy translator between embedding spaces, no shared training.
6. **Synthetic (text, activation) data.** Free training corpus for downstream probes, distillation, cross-model alignment work — without ever collecting paired data.

## What it does NOT give us

- **Source attribution.** NLA describes *what* the model is representing, not *where it learned it*. Citation/provenance requires influence functions, RAG-with-sources, or training-data watermarks — separate, harder problems.
- **Human-readable text by default.** The bit budget (≈500 bits from 30 tokens vs ≈57K bits in the target vector) forces shortcut "neuralese" sequences at high fidelity. Above ~0.75 we expect readability to degrade; mitigating this (KL-to-base-LM regularizer, etc.) likely costs 0.10–0.15 fve. Pareto frontier of fidelity vs readability is the follow-up paper.
- **Guaranteed cross-layer / cross-model transfer.** Current evidence is L20 of Qwen2.5-7B only. Generality is a separate ablation suite.

## Open problems (in order)

1. **Magnitude pinned at 0.98.** Predictions are ~2× over-norm vs targets; the scale-invariant `mag` penalty is bounded and not biting. Likely need an additive norm-matching loss term once base fve clears 0.65.
2. **High-H exploration regime is the actual learning signal.** β=0.3 KL is the sweet spot; β=0.5 over-anchors and collapses (see N1f). The "bimodal H" oscillation between explore (H~2) and exploit (H~0.3) is a feature, not a bug. Two-sided hinge `h_max=999` disables the high-H cap intentionally.
3. **Run-to-run variance.** No `--seed` pinning yet; reproducibility currently anecdotal across runs.
4. **Single layer, single model.** Need L{12,20,27} × {Qwen, Llama} ablation.
5. **Neuralese audit unmeasured.** Need a readability metric + samples.

## The framing for the paper

> A frozen 7B language model can learn to describe its own internal states in natural language, trained from scratch with no description supervision — only round-trip reconstruction error. The resulting verbalizer reaches X% reconstruction at layer 20, enabling [translator / steering / monitor] applications, and provides a hard upper bound on how compressible mid-layer activations are.

If X ≥ 65 corpus-free, that paper is publishable. If X ≥ 85 corpus-free, it's a headline result.

## Honest expectations

- N1g/N1h plateau near 0.62 is the current ceiling at this recipe. Clearing 0.65 likely requires the magnitude fix.
- Clearing 0.85 likely requires *all* of: magnitude fix, longer rollouts (more tokens = more bits), larger AV, possibly architectural changes. Not a continuation of the current curve.
- "Corpus-free" is the defended-to-the-death property. A supervised warmup would still produce a useful tool but would weaken the claim from "we discovered REINFORCE-only NLA works" to "we improved NLA with a novel REINFORCE refinement stage." Both are wins, only the first is historic.
