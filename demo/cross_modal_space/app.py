"""Cross-modal SRT read-out demo — a semiotic adapter that reads images.

The SRT community head was trained on TEXT only (discourse-community contrastive
learning on a frozen gemma-4-31B). This Space shows what happens when that same
text-trained read-out is applied to IMAGE tokens: with zero image training, it
organizes pictures by semantic referent and assigns each to a sensible discourse
community, and it matches images to words. That is the semiotic claim of the SRT
program: the read-out interprets a sign by its place in a system of interpretants,
independent of whether the sign arrived as a word or an image.

gemma-4-31B is 62.5 GB, so this Space does NOT run the model live. Everything was
precomputed once on a GPU box (scripts/build_crossmodal_gallery.py); the Space
just serves the cached assignments + thumbnails, so it runs on a free CPU Space.
"""
from __future__ import annotations

import json
import os

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cached")


def load_gallery() -> dict:
    with open(os.path.join(CACHE, "gallery.json")) as f:
        return json.load(f)


G = load_gallery()
CATS = G["categories"]
TILES = [(os.path.join(CACHE, c["thumbs"][0]), c["category"]) for c in CATS]

INTRO = f"""
# 🜲 The read-out is semiotic, not linguistic

The **SRT community read-out** was trained on **text only** — discourse-community
contrastive learning on a **frozen `google/gemma-4-31B-it`**, updating a 12.3 M
side-channel head while the backbone never moves. This demo applies that *same
text-trained head* to **image tokens**, with **zero image training**.

It still works. A community structure learned from Reddit prose projects onto
pictures and lands in the semantically right place:

| held-out measurement | result | chance |
|---|---|---|
| image → correct semantic class (kNN) | **0.64** | 0.10 |
| image ↔ its class-word retrieval | **0.27** | 0.10 |
| text community assignment | **0.535** | 0.029 |

Pick a category below to see which **discourse community** its images are read
into and the **words** they land nearest. All {G['n_categories']} categories
({G['n_images']} images) were scored offline through the frozen gemma-4-31B — the
model is 62.5 GB, so nothing runs live here. The head saw only text in training.
"""

HOWTO = """
### How it works
1. Each image is fed to a **frozen gemma-4-31B**; the picture enters the shared
   residual stream as ~266 soft-tokens.
2. At an early text-tower layer, the **SRT community head** pools over the image
   tokens into a 64-dim discourse code — the *interpretant*.
3. That code is compared (in a centered frame) to **community centroids** and
   **word anchors** built from text. The head was **never trained on images**.

### Honest scope
This is a **semiotic read-out**, not an image classifier. Community labels are
Reddit discourse communities from the training corpus, so mappings reflect
*discourse* association, not object taxonomy: cats and dogs land in the
cozy-domestic communities (`Mommit`, `knitting`), deer and mushrooms in
`gardening`, cars and buses in `cars`/`cycling`. Assignments are aggregated over
each category's images; single 32×32 images are individually noisier.
"""


def describe(evt: gr.SelectData):
    c = CATS[evt.index]
    comms = {lbl: float(p) for lbl, p in c["top_communities"]}
    words = "  ".join(f"`{w}`" for w, _ in c["top_words"])
    tc = c["top_communities"]
    cap = (f"### {c['category']}\n\n"
           f"Reads into **{tc[0][0]}** ({tc[0][1]:.0%}), then "
           f"{tc[1][0]} ({tc[1][1]:.0%}) and {tc[2][0]} ({tc[2][1]:.0%}).\n\n"
           f"**Nearest words:** {words}")
    examples = [os.path.join(CACHE, f) for f in c["thumbs"]]
    return comms, cap, examples


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SRT cross-modal read-out", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(value=TILES, columns=6, height=430,
                                     label="Pick a category", allow_preview=False)
            with gr.Column(scale=2):
                caption = gr.Markdown("Select a category to read its interpretant.")
                comm = gr.Label(label="Discourse community it reads into",
                                num_top_classes=3)
                examples = gr.Gallery(label="Images in this category", columns=5,
                                      height=110, allow_preview=False)
        gallery.select(describe, None, [comm, caption, examples])
        with gr.Accordion("How it works / scope", open=False):
            gr.Markdown(HOWTO)
        gr.Markdown(
            "Source: [github.com/space-bacon/SRT](https://github.com/space-bacon/SRT) · "
            "Read-out trained on frozen `google/gemma-4-31B-it` · text-only training.")
    return demo


if __name__ == "__main__":
    build_app().launch()
