---
title: SRT-NLA gpt-oss-20b Trace
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
python_version: "3.10"
pinned: true
hardware: zerogpu
license: apache-2.0
short_description: See what gpt-oss-20b is thinking at every layer & token
models:
- openai/gpt-oss-20b
- RiverRider/srt-adapter-gptoss20b
- RiverRider/srt-nla-av-gptoss20b
datasets:
- RiverRider/srt-nla-gptoss20b-artifacts
tags:
- interpretability
- gpt-oss
- activation-verbalization
- hidden-states
- introspection
- mechanistic-interpretability
---

# SRT-NLA · gpt-oss-20b — full input→output trace

A frozen **`openai/gpt-oss-20b`** narrating its own internals, end to end.
Sibling of [srt-showcase](https://huggingface.co/spaces/RiverRider/srt-showcase)
(Qwen-2.5-7B), extended with three new capabilities:

- **Full trace grid** — the model generates a continuation, and every hidden
  state across layers 6/12/18/24 at every (strided) token position — input
  *and* output — is assigned an integer **magic number** via a 4096-code VQ
  state codebook. Recurring states are highlighted; hovering a cell shows the
  codebook's retrieval verbalization.
- **Retrieval decoding** — O(1) nearest-of-4096-codes lookup, the recommended
  decoder on this backbone.
- **AV best-of-K comparison** — the trained Activation Verbalizer, oracle-
  reranked, scored with the anisotropy-**centered** metric against published
  anchors (floor 0.500 / NN 0.744 / replay 0.999) — shown honestly, including
  that it stays below the retrieval baseline on this backbone.

Models & data: [srt-adapter-gptoss20b](https://huggingface.co/RiverRider/srt-adapter-gptoss20b) ·
[srt-nla-av-gptoss20b](https://huggingface.co/RiverRider/srt-nla-av-gptoss20b) ·
[srt-nla-gptoss20b-artifacts](https://huggingface.co/datasets/RiverRider/srt-nla-gptoss20b-artifacts)
