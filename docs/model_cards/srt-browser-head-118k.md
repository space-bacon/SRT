---
license: apache-2.0
language:
- en
tags:
- retrieval
- image-retrieval
- text-to-image-retrieval
- cross-modal
- frozen-backbone
- readout-head
- linear-probe
- on-device
- browser
- wasm
- webassembly
- rust
- candle
- quantized
- coco
base_model:
- Qwen/Qwen3-0.6B
- unsloth/Qwen3-0.6B-GGUF
base_model_relation: adapter
pipeline_tag: feature-extraction
library_name: safetensors
datasets:
- RiverRider/srt-qwen38-coco-states
- RiverRider/srt-coco-thumbs
---

# SRT Browser Read-Out Head (118K)

A **2.1 MB linear head** that lets a 0.6B-parameter quantized language model,
running entirely in a web browser, search 123,287 photographs that a
27B-parameter model encoded offline.

The large model is never downloaded and never runs. It encoded the gallery
once and left behind a file. The small model meets it in that space.

The head reads a hidden state the model **already computed while replying to
you**. Chat and retrieval come from one forward pass, so search costs one
matrix multiply on top of generation — not a second model, not a second
encoder, not a server round-trip.

## What it does

| | |
|---|---|
| Text-to-image retrieval, 123,287-image gallery | R@1 **0.1108**, R@5 0.2510, R@10 0.3373, **median rank 33** |
| Image-to-text, 1,000-image val pool | R@1 **0.5348**, R@5 0.8124, R@10 0.8952 |
| Text-to-image, 1,000-image val pool | R@1 **0.3985**, R@5 0.6856, R@10 0.7887 |
| Shuffled-pair control | 0.0002 |

Median rank 33 out of 123,287 puts the correct image in the top **0.027%**
of the gallery, which is the number we would ask you to look at. R@1 on a
123K pool is a harsh summary: at that scale many photographs are equally
good answers to a caption, and the metric scores a better match than the
gold image as a miss.

**Always read a retrieval number with its pool size.** The same head and
captions score R@1 0.4959 against 1,000 images and 0.0628 against 123,287.
Numbers quoted without a pool size are uninterpretable, including our own
earlier ones.

## Files

| file | size | what it is |
|---|---|---|
| `head_v3.safetensors` | 2.1 MB | **the head that ships.** Text side only, fp16 |
| `browser_head_v2_best40.pt` | 25.2 MB | full v3 training checkpoint, both sides, fp32 |
| `gallery_123k_v3.srtidx` | 130 MB | 123,287 COCO images projected by the v3 head, int8 |
| `anchor_candle_q4_text.bin` | 4 KB | runtime recalibration for candle/Q4\_0 — **read §Anchor** |
| `browser_head_v3_report.json` | — | v3 evaluation, val pool and full gallery |
| `browser_head_v2_arms.json` | — | six-arm ablation with predictions registered before running |
| `browser_head_118k.pt` | 25.2 MB | previous head (v2), superseded |
| `gallery_123k_v2.srtidx` | 130 MB | previous gallery, superseded |

Only the first, third and fourth files reach a visitor. The image head never
ships: it runs offline, and the browser receives its output, not its weights.

## The anchor is not optional

A head fitted against PyTorch/fp16 states does **not** transfer unchanged to
candle/Q4\_0, and the failure is silent — you get confidently ranked results
with plausible scores that are simply the wrong images.

| runtime | t2i R@1 | R@5 | R@10 |
|---|---|---|---|
| PyTorch fp16 (reference) | 0.2300 | 0.4907 | 0.6215 |
| candle Q4\_0, head as-is | **0.0154** | 0.0568 | 0.0896 |
| candle Q4\_0, + 4 KB anchor | **0.1952** | 0.4321 | 0.5457 |

A fifteen-fold collapse, recovered by one 4,096-byte mean vector measured on
200 held-out sentences. No retraining.

This failure is invisible to agreement metrics, which is why we flag it
loudly. Agreement applies the **same transform to both sides** of its
comparison, so a consistent displacement reads as healthy. Only a task with
external ground truth catches it. If you port this head to a new runtime,
measure recall, not agreement.

## Architecture

```
image (offline, datacenter)          text (in the visitor's browser)
  Qwen/Qwen3.8-27B, layer 52           Qwen3-0.6B Q4_0, layer 28 (final block)
  mean-pool over image tokens          mean-pool over positions
  center by image anchor               center by text anchor  <- 4 KB, per runtime
  linear -> 1024, L2-norm              linear -> 1024, L2-norm
        |                                        |
        +----------- dot product -----------------+
                  int8 index, 123,287 rows
```

Both sides are **linear**. This is deliberate and it is measured: a non-linear
image head, which costs zero shipped bytes because the image side never
ships, was the single worst thing we tried (−0.108 R@1). Free in bytes is not
free in statistics.

Tap layer 28 is the **last block** of Qwen3-0.6B. Our layer scan was still
climbing when it ran out of model (L22 0.293, L25 0.351, L28 0.401), so depth
is an open boundary here, not a tuned choice.

## How it was trained

- 118,287 COCO train2017 images, all 5 captions each, one sampled per epoch so
  that five captions of one image never sit in a batch as false negatives
- symmetric InfoNCE, 40 epochs
- both backbones frozen throughout; only the two projections train

The v3 recipe came from a six-arm ablation in which **four of six registered
predictions were wrong**:

| arm | i2t R@1 | Δ |
|---|---|---|
| baseline (1 caption, last-token) | 0.4236 | — |
| **mean pooling** | 0.4668 | **+0.043** |
| 5 captions | 0.4552 | +0.032 |
| 3-layer concat | 0.4192 | −0.004 (null, costs 4 MB) |
| 40 epochs | 0.4112 | −0.012 |
| non-linear image head | 0.3154 | **−0.108** |
| everything at once | 0.4636 | +0.040 |
| **mean pooling + 5 captions, 40 ep** | **0.5348** | **+0.111** |

The two winners are **super-additive**: +0.043 and +0.032 alone, +0.111
together. Note that the `all` arm scored *below mean pooling alone*, because
it bundled the winners with three levers that hurt. An "everything" arm is
not a substitute for testing the winners together.

More epochs hurt with one caption per image (−0.012) and helped with five
(+0.010). Reporting "more epochs hurt" without that conditional would have
been wrong.

## Gallery index format

`SRTIDX02`: `magic | dim u32 | count u32 | scales f32×count | int8 data |
length-prefixed keys`. Rows are unit-norm, so one symmetric scale per row is
sufficient.

int8 storage is **free** on this task:

| store | t2i R@1 | 123K resident |
|---|---|---|
| f32 | 0.2300 | 505 MB |
| f16 | 0.2300 | 252 MB |
| **int8, per-row scale** | **0.2306** | **127 MB** |

The reader is [`srt-geometry`](https://github.com/space-bacon/SRT), a Rust
crate with no model dependency that builds for both native and
`wasm32-unknown-unknown`. Python writes the format, Rust reads it, and top-1
agreement between int8 and f16 is 1.0000 over the parity fixture.

## Steering

A direction in head space shifts what the gallery returns:
`q' = normalize(q + α·axis)`. Axes are built from **captions** and applied to
**image** queries, so a positive result is a claim about a shared space rather
than about memorized neighbours.

Calibrate on **retention** — the share of the query's own unsteered top-k that
survives — not on how strong the effect looks:

| α | class purity | random control | retention |
|---|---|---|---|
| 0 | 0.024 / 0.004 / 0.009 | same | 1.00 |
| **0.5** | 0.252 / 0.117 / 0.128 | 0.009–0.024 | **0.61–0.77** |
| 1.0 | 0.744 / 0.640 / 0.587 | 0.010–0.025 | 0.13–0.29 |
| 2.0 | 0.908 / 0.851 / 0.895 | 0.013–0.029 | **0.01–0.03** |

Purity climbing to 0.9 is not a success. At α = 2 retention is 0.01: the axis
has **replaced** the query, and every input returns the same images.
`DEFAULT_ALPHA = 0.5` is set on retention. 32 matched-norm random axes stay
flat at baseline throughout (z = 108–302 for the real axis at the operating
point), so the direction carries meaning rather than degrading the query into
a class prior.

This is query-side steering only. It does not touch generation.

## Limitations

- **One backbone pair, one domain**: Qwen3-0.6B text against Qwen3.8-27B image, COCO only.
- **The anchor is validated on one runtime pair.** The silence argument is structural; we have measured it once.
- **The text tower is the bottleneck** and we did not test a larger one, because the deployment budget fixes the payload.
- **Steering is validated on keyword-defined classes**, chosen to be objective rather than subtle.
- **Generation quality is not evaluated.** A 0.6B model answers; we make no claim about how well.
- COCO licensing applies to the images; the gallery here contains projected vectors and keys, not photographs.

## Running it on a phone

The browser tier runs on iOS Safari, but only because the model is parsed
directly out of a Blob rather than staged in WebAssembly memory first.

Staging is the obvious implementation and it is fatal here. The 382 MB file
exists twice while candle converts it into tensors, and wasm linear memory
grows but never shrinks, so the doubled peak is the permanent footprint rather
than a spike. iOS terminates the tab. Reading through a Blob with
`FileReaderSync` costs the model only its tensors.

Smaller payloads do not substitute for this. Shrinking the gallery from 130 MB
to 2 MB changed nothing, and candle cannot read llama.cpp's i-quants, so the
smallest usable quantization is Q2\_K at 296 MB, 22% under Q4\_0. The copies
were the problem, not the size.

Smaller gallery shards are published anyway for memory-constrained tiers:
`gallery_20k_v3.srtidx` (21 MB) and `gallery_2k_v3.srtidx` (2.1 MB). Recall
improves as the pool shrinks, so quote the pool size with any number from them.

## Reproducing

Every number above is backed by a committed JSON artifact in
[space-bacon/SRT](https://github.com/space-bacon/SRT) under
`artifacts/nla/q4/`, with a claim-to-artifact mapping in `arxiv_deploy/README.md`.
The raw pre-projection hidden states are published as
[`RiverRider/srt-qwen38-coco-states`](https://huggingface.co/datasets/RiverRider/srt-qwen38-coco-states),
so a new head can be fitted without re-encoding 118K images on a 27B model.

The gallery encoder refuses to run until it reproduces vectors already in the
shipped gallery to cosine > 0.99. A pooling or layer mismatch produces vectors
that look entirely reasonable on their own and are quietly incomparable with
what they are meant to extend.

## Citation

```bibtex
@techreport{lancaster2026readeverywhere,
  title  = {Read Everywhere, Verify There: What It Takes to Put a
            Frozen-Model Read-Out on the Visitor's Hardware},
  author = {Lancaster, James Burton},
  year   = {2026}
}

@techreport{lancaster2026trainonce,
  title  = {Train Once, Read Everywhere: Substrate Invariance of the Linearly
            Readable Structure in Frozen Language Models},
  author = {Lancaster, James Burton},
  year   = {2026},
  type   = {SSRN Working Paper},
  number = {7264778},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7264778}
}
```
