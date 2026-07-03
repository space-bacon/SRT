"""SRT Sunstone — a text-trained read-out that interprets images.

The SRT community head was trained on TEXT only (discourse-community contrastive
learning on a frozen gemma-4-31B). It never saw an image in training. This Space
shows that same head applied to IMAGE tokens: with zero image training it maps a
picture to the words it means and to the discourse it evokes. That is the
semiotic claim of the SRT program — the read-out interprets a sign by its place
in a shared system of meaning, independent of whether the sign arrived as a word
or an image.

gemma-4-31B is 62.5 GB, so this Space does NOT run the model live. Everything was
precomputed once on a GPU box (scripts/build_crossmodal_gallery.py); the Space
serves the cached read-outs + thumbnails and runs on a free CPU Space.
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
:root { --didot: 'Didot','GFS Didot','Bodoni MT','Playfair Display',Georgia,serif;
        --violet:#7c5cff; --ink:#2b2536; --muted:#6b6577; --line:#ece7f5; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important;
        padding: 8px 22px 48px !important; background: #ffffff; }
.gradio-container, .gradio-container * { font-family: var(--didot) !important;
        color: var(--ink); }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { font-weight: 400 !important; }

/* hero */
.hero { text-align:center; padding: 54px 16px 10px; }
.hero .mark { font-size: 2rem; color: var(--violet); letter-spacing: 6px; }
.hero h1 { font-size: 4rem !important; letter-spacing: 1.5px; margin: 10px 0 2px;
        line-height: 1.02; }
.hero .subtitle { font-size: 1.7rem; font-style: italic; color:#4a4458; margin: 6px 0 0; }
.hero .subtitle em { color: var(--violet); font-style: italic; }
.hero .valueprop { max-width: 680px; margin: 22px auto 0; font-size: 1.16rem;
        line-height: 1.6; color: var(--muted); }

/* capability cards */
.cap-grid { display:flex; gap: 22px; margin: 46px 0 8px; flex-wrap: wrap; }
.cap-card { flex:1; min-width: 230px; background:#fdfcff; border:1px solid var(--line);
        border-radius: 18px; padding: 26px 24px; box-shadow: 0 4px 22px rgba(90,60,160,.06); }
.cap-card .ico { font-size: 1.7rem; }
.cap-card h3 { margin: .5em 0 .35em; font-size: 1.28rem; letter-spacing:.3px; }
.cap-card p { color: var(--muted); font-size: 1.04rem; line-height: 1.55; margin: 0; }
.cap-card code { font-family: var(--didot) !important; background:#f2edfb;
        padding: 1px 7px; border-radius: 6px; }

/* section + panels */
.section-title { text-align:center; font-size: 1.9rem; margin: 56px 0 4px; }
.section-sub { text-align:center; color: var(--muted); font-size: 1.08rem; margin: 0 0 22px; }
.readout { border:1px solid var(--line); border-radius: 16px; padding: 22px 24px;
        background:#fdfcff; box-shadow: 0 4px 22px rgba(90,60,160,.05); min-height: 150px; }
.readout h3 { font-size: 1.5rem; margin: 0 0 12px; }
.word-chip { display:inline-block; padding: 5px 14px; margin: 4px 6px 0 0;
        border: 1px solid #ddd3f2; border-radius: 999px; font-size: 1.08rem; color:#4a4458; }
.word-chip.top { background: var(--violet); border-color: var(--violet); color:#fff;
        font-weight: 600; }
.evoke { color: var(--muted); font-size: 1.02rem; margin-top: 16px; }
a { color: var(--violet) !important; }
.footnote { color:#9a94a6; font-size:.94rem; text-align:center; margin-top: 34px;
        line-height: 1.6; }
"""

HERO = """
<div class="hero">
  <div class="mark">&#10022;</div>
  <h1>SRT&nbsp;Sunstone</h1>
  <div class="subtitle">A read-out that <em>reads images</em> &mdash; trained only on words</div>
  <div class="valueprop">
    A 12.3&nbsp;M side-channel head on a <b>frozen <code>gemma-4-31B</code></b>,
    taught meaning from <b>text alone</b>. Hand it a picture and it names
    <b>what the image means</b> &mdash; with <b>zero image training</b>.
  </div>
</div>
"""

CAPABILITIES = """
<div class="cap-grid">
  <div class="cap-card">
    <div class="ico">&#9633;</div>
    <h3>Text-only training</h3>
    <p>The head learned meaning from prose. No images, no labels, the base model
       never fine-tuned &mdash; it never saw a single picture.</p>
  </div>
  <div class="cap-card">
    <div class="ico">&#9635;</div>
    <h3>Cross-modal transfer</h3>
    <p>Hand it a photo and it lands next to the <i>right words</i>:
       <code>bicycle &rarr; bicycle</code>, <code>rose &rarr; flower</code>,
       <code>dog &rarr; pet</code>.</p>
  </div>
  <div class="cap-card">
    <div class="ico">&#9678;</div>
    <h3>Meaning, not pixels</h3>
    <p>It groups images by what they <i>mean</i>, not how they look:
       image&rarr;referent kNN <b>0.64</b> against chance <b>0.10</b>.</p>
  </div>
</div>
"""

WHY = """
### Why this is a *semiotic read-out*, not an image classifier

An **image classifier** is trained on labelled images. It learns a fixed map from
pixels to a closed label set, and it only knows the labels it was shown.

This read-out is different in kind. **It never saw an image in training.** It was
trained only on *text* &mdash; to separate discourse communities in the residual
stream of a frozen gemma-4-31B. What you are watching is that *text-only meaning
space* being asked to interpret a picture, and succeeding, because gemma-4 already
fuses image and word into **one representational stream** and the read-out taps
the shared **interpretant**: the meaning a sign carries, whatever form it arrived
in.

So it does not *classify* the image. It tells you **what the image means to a
system that learned meaning from words** &mdash; a transfer across modality, from
a head that was never cross-trained. That is the result.

### How to read the two panels
- **The words it means** &mdash; the load-bearing evidence. The image is placed in
  a shared word space and the nearest words are shown. This is precise, and it is
  the point.
- **The discourse it evokes** &mdash; a *flavour*, not a category. The head knows
  only **35** discourse communities from its training corpus, far too few to label
  the visual world precisely. Read it as the *nearest cultural neighbourhood* a
  picture leans toward (cars into the automotive community; deer and mushrooms
  into gardening; cats and dogs into the cozy-domestic communities), never as a
  class label.
"""


def describe(evt: gr.SelectData):
    c = CATS[evt.index]
    chips = "".join(
        f'<span class="word-chip {"top" if i == 0 else ""}">{w}</span>'
        for i, (w, _) in enumerate(c["top_words"]))
    tc = c["top_communities"]
    cap = (
        "<div class='readout'>"
        f"<h3>{c['category']}</h3>"
        "<div><b>The words it means</b></div>"
        f"<div style='margin:8px 0 4px'>{chips}</div>"
        "<div class='evoke'><b>The discourse it evokes</b> &nbsp;&middot;&nbsp; "
        "nearest of 35 communities<br>"
        f"{tc[0][0]} ({tc[0][1]:.0%}) &middot; {tc[1][0]} ({tc[1][1]:.0%}) &middot; "
        f"{tc[2][0]} ({tc[2][1]:.0%})</div>"
        "</div>")
    examples = [os.path.join(CACHE, f) for f in c["thumbs"]]
    return cap, examples


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SRT Sunstone") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(HERO)
        gr.HTML(CAPABILITIES)
        gr.HTML("<div class='section-title'>See it read a picture</div>"
                "<div class='section-sub'>Pick a category &mdash; the read-out "
                "places each image in a meaning space built from words.</div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gallery = gr.Gallery(value=TILES, columns=6, height=460,
                                     show_label=False, allow_preview=False)
            with gr.Column(scale=2):
                caption = gr.HTML(
                    "<div class='readout'><em>Select a category to read its "
                    "interpretant.</em></div>")
                examples = gr.Gallery(label="Images in this category", columns=5,
                                      height=112, allow_preview=False)
        gallery.select(describe, None, [caption, examples])
        with gr.Accordion("Why this is a semiotic read-out, not a classifier", open=True):
            gr.Markdown(WHY)
        gr.HTML(
            "<div class='footnote'>Scored offline through a frozen "
            "<code>google/gemma-4-31B-it</code> (62.5 GB, too large to run live). "
            "Read-out trained on text only.<br>"
            "Source: <a href='https://github.com/space-bacon/SRT'>github.com/space-bacon/SRT</a>"
            "</div>")
    return demo


# Module-level object so Hugging Face Spaces auto-detects and launches it.
demo = build_app()

if __name__ == "__main__":
    demo.launch()
