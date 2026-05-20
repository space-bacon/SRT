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
