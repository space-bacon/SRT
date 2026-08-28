---
title: Read the state before the model speaks
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Read a frozen 31B model's state before it speaks
models:
  - RiverRider/srt-verbalizer-v1
  - RiverRider/srt-browser-head-118k
  - RiverRider/gemma-4-31B-it-nf4
---

# Read the state before the model speaks

Give a photograph to **gemma-4-31B**. At layer 47 of 60, before it has produced
a single token, take one 5376-dimensional vector out of the forward pass, pooled
over the image token positions.

That vector is the only thing **Qwen3-0.6B** receives. It is a text-only model
with no vision tower and it never sees your photograph, and it tells you what is
in the picture. The two sentences usually agree.

## The control is the point

The third line shuffles that vector's 5376 dimensions and feeds it to the same
reader. Identical values, identical norm, wrong arrangement. The reader falls
apart, typically into something like `NYC NYC NYC NYC`. If the description had
survived that, the vector would not have been carrying it.

## What is and is not trained

gemma-4-31B is frozen and was never trained to be read. The reader is a 0.6B
decoder trained on COCO captions to turn that vector into English. The gallery
search is a single linear map applied to the same vector, so the sentence and
the retrieved photographs come from one forward pass.

## Honest scoping

The host here is 4-bit NF4 so that it fits free hardware. We measured what that
costs on 100 COCO val2017 images. Anisotropy on this backbone is 0.9874, so raw
cosine is not usable: matched bf16/NF4 pairs read 0.9986 while *mismatched*
pairs read 0.9867. Centered on the pool mean, matched pairs are 0.9631 against a
mismatched floor of 0.0237.

- Retrieval survives. An NF4 query lands on its own image inside the bf16 index
  at rank 1 in 100 of 100 cases.
- Deeper ranking moves. Top-10 neighbour agreement with bf16 is 0.704, so the
  two indices are not interchangeable.
- Wording is content-stable but not token-stable. 8 of 8 sampled pairs named the
  same subject, 2 of 8 were identical strings.

The gallery is 1000 COCO val2017 photographs indexed from bf16 states. Greedy
decoding at 40 tokens loops, so repeated sentences are dropped for display. A
gallery image retrieving itself at cosine 1.000 is arithmetic rather than
evidence, so exact self-matches are excluded.

Use of the host is governed by the [Gemma Terms of
Use](https://ai.google.dev/gemma/terms).
