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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');
:root { --didot: 'Didot','GFS Didot','Bodoni MT','Playfair Display',Georgia,serif; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { font-family: var(--didot) !important; }
h1 { font-size: 2.7rem !important; font-weight: 400 !important; letter-spacing: .4px;
     line-height: 1.12; margin: .2em 0 !important; }
h2 { font-size: 1.55rem !important; font-weight: 400 !important; }
h3 { font-weight: 400 !important; letter-spacing: .3px; }
.hero { text-align: center; }
.hero em { color: #8b5cf6; font-style: italic; }
.lead { font-size: 1.12rem; line-height: 1.5; }
.readout { border-left: 3px solid #8b5cf6; padding: 2px 0 2px 16px; }
.word-chip { display:inline-block; padding: 3px 11px; margin: 3px 4px 0 0;
     border: 1px solid #d8cfe8; border-radius: 999px; font-size: 1.05rem; }
.word-chip.top { background: #efe8fb; border-color: #b79df0; font-weight: 600; }
a { color: #8b5cf6 !important; }
table td, table th { font-size: 1.02rem !important; }
.footnote { color: #8a8a8a; font-size: .92rem; }
"""

INTRO = """
<div class="hero">

# A read-out that *reads images* — trained only on words

</div>

<div class="lead">

The **SRT read-out** is a 12.3 M side-channel head on a **frozen
`google/gemma-4-31B-it`**. It was taught **one thing, from text alone**: to tell
apart communities of language use. **It never saw a single image.** Yet applied
to a picture it maps the image to **the words it means** and to **the discourse
it evokes** — with **zero image training**.

</div>
"""

CAPABILITIES = """
|  |  |
|---|---|
| 🜲 **Text-only training** | The head learned meaning from prose. No images, no labels, the base model never fine-tuned. |
| 🖼️ **Cross-modal transfer** | Hand it a photo and it lands next to the *right words* — `bicycle → bicycle`, `rose → flower`, `dog → pet`. |
| 🧭 **Meaning, not pixels** | It groups images by what they *mean*, not how they look: image→referent kNN **0.64** vs chance **0.10**. |
"""

WHY = """
### Why this is a *semiotic read-out*, not an image classifier

An **image classifier** is trained on labelled images. It learns a fixed map from
pixels to a closed label set, and it only knows the labels it was shown.

This read-out is different in kind. **It never saw an image in training.** It was
trained only on *text* — to separate discourse communities in the residual stream
of a frozen gemma-4-31B. What you are watching is that *text-only meaning space*
being asked to interpret a picture, and succeeding, because gemma-4 already fuses
image and word into **one representational stream** and the read-out taps the
shared **interpretant**: the meaning a sign carries, whatever form it arrived in.

So it does not *classify* the image. It tells you **what the image means to a
system that learned meaning from words** — a transfer across modality, from a head
that was never cross-trained. That is the result.

### How to read the two panels
- **The words it means** — the load-bearing evidence. The image is placed in a
  shared word space and the nearest words are shown. This is precise, and it is
  the point.
- **The discourse it evokes** — a *flavour*, not a category. The head knows only
  **35** discourse communities from its training corpus, far too few to label the
  visual world precisely. Read it as the *nearest cultural neighbourhood* a
  picture leans toward (cars → the automotive community; deer and mushrooms →
  gardening; cats and dogs → the cozy-domestic communities), never as a class
  label.
"""


def describe(evt: gr.SelectData):
    c = CATS[evt.index]
    chips = "".join(
        f'<span class="word-chip {"top" if i == 0 else ""}">{w}</span>'
        for i, (w, _) in enumerate(c["top_words"]))
    tc = c["top_communities"]
    cap = (
        "<div class='readout'>\n\n"
        f"### {c['category']}\n\n"
        f"**The words it means**\n\n{chips}\n\n"
        "**The discourse it evokes** &nbsp;·&nbsp; nearest of 35 communities\n\n"
        f"{tc[0][0]} ({tc[0][1]:.0%}) · {tc[1][0]} ({tc[1][1]:.0%}) · "
        f"{tc[2][0]} ({tc[2][1]:.0%})\n\n</div>")
    examples = [os.path.join(CACHE, f) for f in c["thumbs"]]
    return cap, examples


def build_app() -> gr.Blocks:
    # Inject CSS via a <style> tag rather than the Blocks/launch `css=` param,
    # which moved between gradio 5.x (Space) and 6.x (local) — this works in both.
    with gr.Blocks(title="SRT · a read-out that reads images") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown(INTRO)
        gr.Markdown(CAPABILITIES)
        gr.Markdown("### Pick a category — see what a text-trained head makes of a picture")
        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(value=TILES, columns=6, height=440,
                                     show_label=False, allow_preview=False)
            with gr.Column(scale=2):
                caption = gr.Markdown(
                    "<div class='readout'><em>Select a category to read its "
                    "interpretant.</em></div>")
                examples = gr.Gallery(label="Images in this category", columns=5,
                                      height=110, allow_preview=False)
        gallery.select(describe, None, [caption, examples])
        with gr.Accordion("Why this is a semiotic read-out, not a classifier", open=True):
            gr.Markdown(WHY)
        gr.Markdown(
            "<div class='footnote'>Scored offline through a frozen "
            "<code>google/gemma-4-31B-it</code> (62.5 GB, too large to run live). "
            "Read-out trained on text only. "
            "Source: <a href='https://github.com/space-bacon/SRT'>github.com/space-bacon/SRT</a>."
            "</div>")
    return demo


# Module-level object so Hugging Face Spaces auto-detects and launches it.
demo = build_app()

if __name__ == "__main__":
    demo.launch()
