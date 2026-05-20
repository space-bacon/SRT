---
title: SRT-NLA v1 demo
emoji: 🪞
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10.13"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
short_description: Latent autoencoder over Qwen2.5-7B L20
models:
  - RiverRider/srt-nla-av-v1
  - Qwen/Qwen2.5-7B
---

# SRT-NLA v1 — latent autoencoder demo

Two interactive views of the `RiverRider/srt-nla-av-v1` verbalizer.

1. **Round-trip autoencoder.** Encode a passage with frozen Qwen2.5-7B at L20,
   decode the resulting 3584-dim vector with the 12.7M-parameter NLA verbalizer,
   and re-encode the verbalization to measure round-trip fidelity
   (`centered fve_nrm`, `rho_norm`). Best-of-N rerank is included — this is the
   technique that takes greedy `ρ ≈ 0.26` up to `ρ ≈ 0.92` in the paper.
2. **Latent arithmetic.** Encode two passages, slide an interpolation `α`, and
   verbalize the mixture vector.

## Plain-English summary

The big language model (frozen 7B Qwen2.5) has a "thought" sitting in the
middle of its forward pass — a 3584-dim hidden vector at layer 20 that
nobody can read directly. The NLA verbalizer is a tiny (12.7M-param) module
trained to write English text that, when fed back through the same frozen
model, re-creates that hidden vector.

We measure success on a 0–1 scale calibrated against:
- **0** = a random unrelated sentence
- **1** = a human-written paraphrase of the original

If you let the verbalizer write its single best guess (greedy decoding), it
scores around **0.29** — better than random, but well short of paraphrase
quality. If you let it write 64 candidates and automatically pick the one
whose hidden state lands closest to the target ("best-of-N rerank" in the
demo), it scores around **1.0** — i.e. matches a human paraphrase. That's
the headline result: paraphrase-quality activation reconstruction with no
extra training, just sampling + a cheap reranker.

We also tried to close the greedy gap by *training* the verbalizer to
imitate the best-of-K rollout (paper §6, "Lever B"). It didn't help: the
objective either collapses sampling diversity or plateaus at the
warm-start. So best-of-N rerank at inference time isn't a stopgap, it's
the answer.

## Anchors (paper §3)

| Anchor | centered fve_nrm |
|---|---|
| Random floor | 0.510 |
| NN-retrieval over pool=2000 | 0.71 |
| Qwen paraphrase ceiling | 0.799 |

`rho_norm = (cen − 0.510) / 0.289` linearly remaps to `[0, 1]` against these
anchors.

## Source

Code: https://github.com/space-bacon/SRT (branch `nla`, `docs/hf/nla_v1_demo/`).
