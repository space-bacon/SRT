---
title: Gemma-4-31B-it SRT-Sunstone
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - semiotic
  - cross-modal
  - interpretability
  - read-out
  - gemma-4
  - representation-learning
  - vision-language
short_description: A read-out that reads images — trained only on words
---

# 🜲 Gemma-4-31B-it SRT-Sunstone

### A read-out that reads images — trained only on words

**SRT Sunstone** is a live demonstration that a read-out head trained on **text
alone** can **interpret images** — with **zero image training**. It is a 12.3 M
side-channel head on a **frozen `google/gemma-4-31B-it`**, taught one thing from
prose: to tell apart *communities of language use*. Point it at a picture and it
names **what the image means** and the **discourse it evokes**.

This is the semiotic claim of the **SRT** (Semiotic-Reflexive Transformer)
program made concrete: the read-out interprets a sign by its place in a shared
system of meaning, independent of whether the sign arrived as a **word** or an
**image**.

---

## What you can do here

Pick any category in the gallery. For each one you see:

- **The words it means** — the image, placed in a shared word space, retrieves
  its own words: `bicycle → bicycle`, `rose → flower`, `dog → pet`, `car → car`.
  This is the precise, load-bearing result.
- **The discourse it evokes** — the nearest of 35 discourse communities the head
  learned from text: cars into the automotive community, deer and mushrooms into
  gardening, cats and dogs into the cozy-domestic communities. A *flavour*, not a
  label.

---

## Why this is a semiotic read-out, not an image classifier

An image classifier is trained on labelled images and learns a fixed pixel→label
map; it only knows the labels it was shown. This read-out is different in kind:
**it never saw an image in training.** It was trained only on text — to separate
discourse communities in the residual stream of a frozen gemma-4-31B. It can read
a picture because gemma-4 already fuses image and word into **one representational
stream**, and the read-out taps the shared **interpretant** — the meaning a sign
carries, whatever form it arrived in. It does not classify the image; it tells you
what the image *means* to a system that learned meaning from words.

---

## The evidence (held-out)

| measurement | result | chance |
|---|---|---|
| image → correct semantic class (kNN) | **0.64** | 0.10 |
| image ↔ its class-word retrieval | **0.27** | 0.10 |
| held-out text community assignment | **0.535** | 0.029 |

The head was selected by held-out community accuracy (not training loss), which
peaks at step 2250 of 3000 (0.535, an 18.7× lift over chance).

---

## How the head was trained

- **Backbone:** `google/gemma-4-31B-it`, frozen (never modified).
- **Head:** 12.3 M parameters (a discourse-community encoder plus MAH divergence
  heads), reading the residual stream at an early text-tower layer.
- **Data:** 150 K discourse passages across 35 communities.
- **Objective:** supervised-contrastive on community id (grouped sampling) plus
  self-supervised divergence and chain terms. No images, no labels on images, no
  fine-tuning of the base model.

---

## About this Space (precomputed, CPU)

gemma-4-31B is **62.5 GB**, far too large to run inside a Space. Every read-out
and thumbnail here was **precomputed once on a GPU box**
(`scripts/build_crossmodal_gallery.py`) with per-modality centering to remove
backbone anisotropy. This Space serves the cache, so it runs on **free CPU** with
no model download.

---

## Scope & limitations

- A **semiotic read-out, not a classifier**: community labels reflect discourse
  association, not object taxonomy, and the head knows only 35 communities — far
  too few to label the visual world precisely.
- Image↔word retrieval (0.27) is well above chance but lossy: the 64-dim
  discourse code discards most fine visual detail.
- Assignments are aggregated per category; single 32×32 images are individually
  noisier.

## Links

- **Model (read-out head):** [`RiverRider/Gemma-4-31B-it-SRT-Sunstone`](https://huggingface.co/RiverRider/Gemma-4-31B-it-SRT-Sunstone)
- **Code & paper:** https://github.com/space-bacon/SRT
- **Base model:** [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)

License: Apache-2.0. Base model governed by Google's gemma-4 license.
