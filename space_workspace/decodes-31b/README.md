---
title: A 0.6B model decodes a 31B model's hidden state
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# A 0.6B model decodes a 31B model's hidden state

`gemma-4-31B` was given a photograph. At layer 47 of 60, during the prefill and
**before a single token had been generated**, one 5376-dimensional vector was
taken, pooled over the image token positions.

`Qwen3-0.6B`, a text-only model that has never seen a photograph, receives that
vector and nothing else, and describes what is in the frame.

Nothing is fine-tuned. gemma-4 is frozen and was never trained to be read.

## What changed

gemma's half is now **precomputed at full precision**, so this runs on a CPU
with no GPU queue. The previous version loaded a 4-bit gemma-4 on ZeroGPU to
produce one vector and then did the interesting part in milliseconds, which
meant a wait on every visit and a quantisation that changed the reader's
wording.

The freed budget buys two controls you can turn:

**Corrupt** scrambles a chosen fraction of the 5376 dimensions. Every value and
the norm are preserved exactly at each setting, so the reader cannot be losing
magnitude: only the arrangement is destroyed. This was a single yes/no aside
before. Continuous, it is the most interesting thing here.

**Walk** moves the state from one photograph towards another and reads the
midpoints, which are states no camera produced.

## Layers

Extra depths are included, with a caveat the interface repeats whenever you
select one: the verbalizer was **trained at layer 47**. Reading another depth
measures how far it sits from layer 47 in the reader's frame. It is not a
measurement of where meaning lives.

## Artifacts

Reader `RiverRider/srt-verbalizer-v1`, retrieval head
`RiverRider/srt-browser-head-118k`. The trade-off taken deliberately: CPU means
no live upload, and the gallery is fixed at the 1000 indexed photographs plus 8
held outside the index.
