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

# 🜲 SRT Cross-Modal Read-Out

A demonstration that the SRT read-out is **semiotic, not linguistic**.

The community head was trained on **text only** — discourse-community contrastive
learning on a **frozen `google/gemma-4-31B-it`** (a 12.3 M side-channel head; the
backbone never moves). This Space applies that same head to **image tokens**, with
**zero image training**, and shows the resulting discourse-community assignments
and image↔word matches for a curated gallery.

Because gemma-4-31B is 62.5 GB, the Space does **not** run the model live: the
assignments and thumbnails were precomputed once on a GPU box
(`scripts/build_crossmodal_gallery.py`). The Space serves the cache on free CPU.

Key held-out results: image→semantic-class kNN **0.64** (chance 0.10), image↔word
retrieval **0.27** (chance 0.10), text community assignment **0.535** (chance
0.029).

Source: https://github.com/space-bacon/SRT
