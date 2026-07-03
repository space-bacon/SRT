---
title: SRT Cross-Modal Read-Out
emoji: 🜲
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: A text-trained SRT read-out that reads images into discourse communities
---

# 🜲 A read-out that reads images — trained only on words

The **SRT read-out** is a 12.3 M side-channel head on a **frozen
`google/gemma-4-31B-it`**. It was taught one thing, from **text alone**: to tell
apart communities of language use. It **never saw a single image** in training.
Yet applied to a picture it maps the image to **the words it means** and to **the
discourse it evokes**, with **zero image training**.

This is a **semiotic read-out, not an image classifier**. A classifier is trained
on labelled images and learns a fixed pixel→label map. This head was trained only
on text and interprets a picture because gemma-4 fuses image and word into one
representational stream — the read-out taps the shared *interpretant*, the meaning
a sign carries whatever form it arrived in.

Because gemma-4-31B is 62.5 GB, the Space does **not** run the model live: the
read-outs and thumbnails were precomputed once on a GPU box
(`scripts/build_crossmodal_gallery.py`). The Space serves the cache on free CPU.

Held-out evidence: image→referent kNN **0.64** (chance 0.10), image↔word retrieval
**0.27** (chance 0.10), text community assignment **0.535** (chance 0.029).

Source: https://github.com/space-bacon/SRT
