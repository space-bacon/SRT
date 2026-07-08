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
        --bg:#131233; --panel:#1c1a47; --violet:#a48dff; --ink:#eae7fb;
        --muted:#a7a1cc; --line:#332e66; }
html, body, gradio-app, .gradio-container { background: var(--bg) !important;
        max-width: 100% !important; overflow-x: hidden !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important;
        padding: 8px 22px 48px !important; }
.gradio-container, .gradio-container * { font-family: var(--didot) !important;
        color: var(--ink); }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { font-weight: 400 !important; }
/* make gradio's own wrappers blend into the dark canvas */
.gradio-container .block, .gradio-container .form, .gradio-container .gr-box,
.gradio-container .panel, .gradio-container .wrap { background: transparent !important;
        border: none !important; box-shadow: none !important; }
.gradio-container .gallery, .gradio-container .grid-wrap,
.gradio-container .thumbnail-item { background: transparent !important; }
/* gallery: size to exactly fit all tiles — no inner scroll, no edge shadows */
.gradio-container .gallery, .gradio-container .grid-wrap,
.gradio-container .grid-container, .gradio-container .thumbnails {
        height: auto !important; max-height: none !important; min-height: 0 !important;
        overflow: visible !important; box-shadow: none !important; }
.gradio-container .gallery::before, .gradio-container .gallery::after,
.gradio-container .grid-wrap::before, .gradio-container .grid-wrap::after {
        display: none !important; content: none !important; box-shadow: none !important; }
.gradio-container .thumbnail-item, .gradio-container .thumbnail-item img {
        box-shadow: none !important; }
.gradio-container details, .gradio-container .accordion { background: var(--panel) !important;
        border: 1px solid var(--line) !important; border-radius: 14px !important; }

/* hero */
.hero { text-align:center; padding: 54px 16px 10px; }
.hero .mark { font-size: 2rem; color: var(--violet); letter-spacing: 6px; }
.hero h1 { font-size: clamp(1.85rem, 6.5vw, 4rem) !important; letter-spacing: 1px;
        margin: 10px 0 2px; line-height: 1.05; overflow-wrap: break-word; word-break: break-word; }
.hero .subtitle { font-size: clamp(1.15rem, 3.4vw, 1.7rem); font-style: italic; color:#c9c3ea; margin: 6px 0 0; }
.hero .subtitle em { color: var(--violet); font-style: italic; }
.hero .valueprop { max-width: 680px; margin: 22px auto 0; font-size: clamp(1rem, 2.6vw, 1.16rem);
        line-height: 1.6; color: var(--muted); }

/* capability cards */
.cap-grid { display:flex; gap: 22px; margin: 46px 0 8px; flex-wrap: wrap; }
.cap-card { flex:1; min-width: 230px; background: var(--panel); border:1px solid var(--line);
        border-radius: 18px; padding: 26px 24px; box-shadow: 0 8px 30px rgba(0,0,0,.35); }
.cap-card .ico { font-size: 1.7rem; color: var(--violet); }
.cap-card h3 { margin: .5em 0 .35em; font-size: 1.28rem; letter-spacing:.3px; }
.cap-card p { color: var(--muted); font-size: 1.04rem; line-height: 1.55; margin: 0; }
.cap-card code { font-family: var(--didot) !important; background:#2a2662;
        color:#d8d0fb; padding: 1px 7px; border-radius: 6px; }

/* section + panels */
.section-title { text-align:center; font-size: clamp(1.4rem, 4.2vw, 1.9rem); margin: 56px 0 4px; }
.section-sub { text-align:center; color: var(--muted); font-size: 1.08rem; margin: 0 0 22px; }
.readout { border:1px solid var(--line); border-radius: 16px; padding: 22px 24px;
        background: var(--panel); box-shadow: 0 8px 30px rgba(0,0,0,.3); min-height: 150px; }
.readout h3 { font-size: 1.5rem; margin: 0 0 12px; }
.word-chip { display:inline-block; padding: 5px 14px; margin: 4px 6px 0 0;
        border: 1px solid #494285; border-radius: 999px; font-size: 1.08rem;
        color:#d7d0f2; background: rgba(255,255,255,.03); }
.word-chip.top { background: var(--violet); border-color: var(--violet); color:#1a1533;
        font-weight: 600; }
.evoke { color: var(--muted); font-size: 1.02rem; margin-top: 16px; }
a { color: var(--violet) !important; }
.footnote { color:#8f89b6; font-size:.94rem; text-align:center; margin-top: 34px;
        line-height: 1.6; }

/* --- single-column, fully fluid layout ---
   Everything stacks in one vertical column, so no row can ever be forced wider
   than the screen. Galleries reflow their tiles to whatever width is available. */
.cap-card { min-width: 0 !important; flex: 1 1 230px !important; }
/* Gradio 6.x renders the real gallery grid as `.grid-container` with FIXED px
   columns (e.g. `100px 100px ...`). That is what overflowed narrow screens and
   got clipped. Force every gallery grid to reflow to the available width. */
.gradio-container .grid-container, .gradio-container .grid-wrap,
.gradio-container .gallery .grid-wrap, .gradio-container .thumbnails {
        grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)) !important;
        width: 100% !important; }
.gradio-container .thumbnail-item { min-width: 0 !important; }
.gradio-container .thumbnail-item img { max-width: 100% !important; height: auto !important; }
.main-gallery { margin: 0 0 22px !important; }
.readout { margin-top: 4px; }

/* small-frame typographic tightening (on top of the clamp() fluid sizes) */
@media (max-width: 860px) {
  .gradio-container { padding: 6px 12px 32px !important; }
  .hero { padding: 26px 6px 6px; }
  .cap-grid { gap: 14px; margin: 30px 0 6px; }
  .cap-card { padding: 20px 18px; }
  .section-title { margin: 36px 0 4px; }
  .readout { padding: 18px 18px; }
}
"""

HERO = """
<div class="hero">
  <div class="mark">&#10022;</div>
  <h1>Gemma-4-31B-it SRT-Sunstone</h1>
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
    <div class="ico">&#10086;</div>
    <h3>Open-vocabulary captions</h3>
    <p>The same image state retrieves a full <i>sentence</i> from
       <b>10,000</b> COCO captions &mdash; zero training:
       <code>horse &rarr; &ldquo;a person riding a horse&rdquo;</code>.</p>
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

### How to read the three panels
- **The words it means** &mdash; the load-bearing evidence. The image is placed in
  a shared word space and the nearest words are shown. This is precise, and it is
  the point.
- **The caption it retrieves** &mdash; the same image state, asked to pick a full
  sentence out of **10,000 COCO captions** it has never been trained against.
  Open vocabulary, zero training: the image's place in meaning space is precise
  enough to select a whole description, not just a word.
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
    cap_block = ""
    if c.get("top_captions"):
        best = c["top_captions"][0]
        rest = " &middot; ".join(
            f'<i>&ldquo;{t}&rdquo;</i>' for t, _ in c["top_captions"][1:])
        cap_block = (
            "<div style='margin-top:16px'><b>The caption it retrieves</b> "
            "&nbsp;&middot;&nbsp; nearest of 10,000 COCO captions</div>"
            f"<div style='margin:8px 0 0; font-size:1.12rem'>&ldquo;{best[0]}&rdquo;</div>"
            + (f"<div class='evoke' style='margin-top:6px'>{rest}</div>" if rest else ""))
    cap = (
        "<div class='readout'>"
        f"<h3>{c['category']}</h3>"
        "<div><b>The words it means</b></div>"
        f"<div style='margin:8px 0 4px'>{chips}</div>"
        f"{cap_block}"
        "<div class='evoke'><b>The discourse it evokes</b> &nbsp;&middot;&nbsp; "
        "nearest of 35 communities<br>"
        f"{tc[0][0]} ({tc[0][1]:.0%}) &middot; {tc[1][0]} ({tc[1][1]:.0%}) &middot; "
        f"{tc[2][0]} ({tc[2][1]:.0%})</div>"
        "</div>")
    examples = [os.path.join(CACHE, f) for f in c["thumbs"]]
    return cap, examples


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Gemma-4-31B-it SRT-Sunstone") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(HERO)
        gr.HTML(CAPABILITIES)
        gr.HTML("<div class='section-title'>See it read a picture</div>"
                "<div class='section-sub'>Pick a category &mdash; the read-out "
                "places each image in a meaning space built from words.</div>")
        # Single vertical column: gallery, then read-out, then example strip.
        # No side-by-side row, so the layout cannot overflow horizontally on any
        # screen width (this is what makes it robust on mobile).
        gallery = gr.Gallery(value=TILES, columns=6, show_label=False,
                             allow_preview=False, elem_classes="main-gallery")
        caption = gr.HTML(
            "<div class='readout'><em>Select a category above to read its "
            "interpretant.</em></div>")
        examples = gr.Gallery(label="Images in this category", columns=5,
                              height=112, allow_preview=False,
                              elem_classes="ex-gallery")
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
