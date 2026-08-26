#!/usr/bin/env python3
"""Build the self-contained Substack post for the Lab map instrument.

Substack has no asset uploader worth using from a script, but it does paste
rich HTML: base64-embedded images survive a select-all and copy. So the
deliverable is one HTML file you open in a browser and copy out of.

The prose lives here rather than in the generated file so the post can be
rebuilt whenever a figure is regenerated, instead of being hand-patched
around three megabytes of base64.

    python scripts/make_lab_post.py
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from PIL import Image

ASSETS = Path("artifacts/marketing/lab_map_post")
MAX_W = 1200  # plenty for a 700px column on a retina screen


def embed(name: str, alt: str, width: int = 700, caption: str | None = None) -> str:
    """Inline one asset as a data URI, downscaling oversized PNGs.

    GIFs are passed through untouched: re-encoding an animation through PIL
    costs more quality than the bytes it saves.
    """
    path = ASSETS / name
    raw = path.read_bytes()
    mime = "gif" if path.suffix == ".gif" else "png"

    if mime == "png":
        im = Image.open(io.BytesIO(raw))
        if im.width > MAX_W:
            h = round(im.height * MAX_W / im.width)
            im = im.resize((MAX_W, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG", optimize=True)
        if buf.tell() < len(raw) or im.width == MAX_W:
            raw = buf.getvalue()

    b64 = base64.b64encode(raw).decode()
    print(f"  {name:24s} {len(raw) / 1024:7.0f} KB", flush=True)
    html = (f'<img src="data:image/{mime};base64,{b64}" width="{width}" '
            f'alt="{alt}" />')
    if caption:
        html += f"\n\n<p><em>{caption}</em></p>"
    return html


def body() -> str:
    return f"""
<h3>The result we nearly published</h3>

<p><em>We built a map of how one frozen 31-billion-parameter model saw 40,000
photographs. Then we found something striking in it. Then we found out why it
was wrong.</em></p>

<p>Start with what the instrument is, because the rest of this only makes sense
once that is clear.</p>

<p>We showed 40,000 photographs to a very large model and kept the private
working notes it made while looking. Those notes are nothing but long lists of
numbers. We arranged the lists on a flat sheet so that photographs the model had
seen as similar ended up near one another. That sheet is the map, and it
organised itself: animals gathered in the west, transport in the southwest,
sport along the north, kitchens and bathrooms down in the southeast. Nobody
sorted them.</p>

<p>Then a second, much smaller model reads any point you touch and writes a
sentence about what is there. It has no eyes. There is no vision component
anywhere inside it. It has never been shown a photograph in its life. It is
reading somebody else&rsquo;s handwriting and telling you what it says.</p>

<p><a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
&middot; pane 04</p>

{embed("fig_poster.png",
       "The map: 40,000 photographs of a 123,287-image gallery in two "
       "dimensions, region names written by the reader",
       caption="Every dot is a photograph. Nothing arranged that. It is where "
               "the photographs fell.")}

<h3>How this works, without the jargon</h3>

<p>When a large model looks at a photograph it does not keep the picture. It
builds a working impression: a list of several thousand numbers that it uses to
answer whatever is asked next. The technical name is a hidden state. Think of it
as the model&rsquo;s private notes on the image, written in its own handwriting,
and normally thrown away the instant the answer is produced.</p>

<p>We kept the notes, and then did two things with them.</p>

<p><strong>First, we shortened them.</strong> A small trained matrix squeezes
each list from 5,376 numbers down to 1,024, and it is trained so that a
photograph and a sentence describing that photograph come out in nearly the same
place. Picture, meet caption, in one shared filing system.</p>

<p>That shortening step is a single linear layer, which matters more than it
sounds. A linear map can rotate a space, stretch it and pick out directions
within it. What it cannot do is invent structure that was not already there. So
the islands on this map are not something we built into the data. They were
already sitting in the model&rsquo;s notes, and a linear view was enough to find
them. Flattening the result onto a screen is our doing. The neighbourhoods are
not.</p>

<p><strong>Second, we taught a small model to read the notes back.</strong> A
frozen 382&nbsp;MB Qwen3-0.6B, with a small adapter in front of it, takes one of
those 1,024-number lists and writes an English sentence about it.</p>

{embed("lab_01_map.png",
       "The map instrument running in the Lab, pane 04",
       caption="The same map inside the Lab, live.")}

<h3>We did not name the regions</h3>

<p>Every label on that map was written by the same small reader. We handed it
the centre of each of the 24 regions and printed what came back. The names you
are reading are the machine&rsquo;s account of its own filing system, arrived at
without us, months after the big model made the notes and on a different
computer.</p>

<h3>The thing we thought we had</h3>

<p>Most of the map looks empty. The islands are where photographs cluster and
the space between them looks like open water. Click there anyway and the reader
still answers.</p>

<p>Between the skiing photographs and the skateboarding photographs there is a
gap. There is no snowboarding island and no snowboarding label anywhere on the
map. We clicked the gap and the reader wrote:</p>

<blockquote><p>A man is doing a trick on a snowboard.</p></blockquote>

<p>We had the headline written. Skiing on one side, skateboarding on the other,
and the space between them composing a third thing that nobody had put there.
A map that does arithmetic.</p>

<h3>Then we measured it</h3>

<p>Before publishing we went to check the obvious objection, which is whether
that spot is actually empty. It is not, and the way it is not is instructive.</p>

<p>Clicking does not read empty space. It cannot. The instrument takes the 48
nearest photographs to wherever you touched and averages their number-lists, so
every click is an average of real photographs. The question is only whether that
average lands somewhere a photograph already sits.</p>

<p>So we asked what the nearest photographs to that average actually are. All
eight of them are snowboarding photographs. The closest one matches at 0.92, and
its own human caption reads &ldquo;A person a snowboard in the air doing a
trick.&rdquo; The reader had written &ldquo;A man is doing a trick on a
snowboard.&rdquo;</p>

<p>It was describing photographs that are there. COCO, the photograph set, has a
snowboarding category, and those pictures sit between skiing and skateboarding
exactly where anyone would file them. Our clustering only names 24 regions, so
snowboarding never got a label of its own. The flattening onto two dimensions
happened to leave a small hole at that spot, so it looked like open water on
screen.</p>

<p>Three things lined up: no label, a visual gap, and a correct sentence. We
checked the first two and assumed the third.</p>

<h3>What is actually true</h3>

<p>We built the measurement we should have had first. For any point it asks two
questions that can fail. Does the averaged vector match some real photograph as
well as a real photograph matches its own nearest neighbour, in which case it
simply is that photograph? And where do the 48 contributing photographs come
from, since half from each neighbour is a blend and forty-seven from one is not?</p>

<p>Run against the eight midpoints we had planned to publish, only one is both
genuinely mixed and genuinely novel. It is the midpoint between the kitchen
region and the plate-of-food region, its contributors split 52 to 48 between
them, its best match in the whole gallery weaker than 60% of photographs manage
with their own neighbour, and it returns:</p>

<blockquote><p>A bowl of fruit and a knife sitting on a table.</p></blockquote>

<p>That one is real, and it is a good deal less exciting than snowboarding. The
snowboard result scored in the 94th percentile for matching an existing
photograph, and its neighbours were 98% skiing. It was retrieval wearing the
costume of arithmetic.</p>

{embed("fig_openwater.png",
       "Eight midpoints between named regions, each with the sentence the map "
       "returns there",
       width=740,
       caption="Eight real clicks. Most of them are describing photographs that "
               "are genuinely there, which is the instrument working correctly "
               "rather than the instrument being clever.")}

<p>The map is not doing arithmetic. It is a browsable index of one model&rsquo;s
reading of 40,000 photographs, and you can stand anywhere in it and get a
description. That is a smaller claim and it is the one that survives.</p>

<h3>Type something and watch where it lands</h3>

<p>The map runs in both directions. Type a sentence and it is placed on the map
by meaning, then read back from wherever it landed:</p>

{embed("lab_04_sentence.png",
       "Typing a sentence places it on the map; the eight nearest photographs "
       "are all skiing",
       caption="&ldquo;a man riding skis down a snowy slope&rdquo; lands in the "
               "snow country, and the eight photographs nearest that spot are "
               "all skiing. Nothing matched on words.")}

<h3>The half only the Lab can do</h3>

<p>A hosted demo can show you a map. It cannot read <em>your</em> photograph into
it, because that needs the big model running, right now, on the other end.</p>

<p>The Lab has one. A gemma-4-31B is served from a single desktop computer, so
when you drop in a picture it is encoded here, live, into the same
1,024-dimensional space the gallery was built in. The marker lands among its
actual neighbours rather than near words about it.</p>

{embed("lab_upload.gif",
       "Uploading a photograph: encoded by the live 31B, lands on the map, described in a sentence",
       caption="2.0 seconds to encode on the 31B, 0.7 seconds for the 0.6B to "
               "read it: &ldquo;Two cats are laying on a couch.&rdquo;")}

<h3>The control is in the panel, not the footnote</h3>

<p>A sentence that sounds right proves nothing, which is the whole lesson of the
snowboard. So the instrument shows the eight photographs nearest wherever the
marker landed, every time, without being asked. If we had been looking at that
panel instead of at the empty-looking gap, we would have seen eight snowboarders
and caught this a week earlier.</p>

{embed("lab_03_neighbours.png",
       "The eight photographs nearest the uploaded cat picture: all eight are cats",
       caption="All eight are cats. If the marker landed anywhere at all, these "
               "would be unrelated, and you would be able to see it without "
               "trusting us.")}

<h3>The numbers</h3>

<p>Measured on 500 held-out photographs against a pool of 118,287, so chance is
a median rank of 59,143:</p>

{embed("fig_ladder.png",
       "Every arm of the reader evaluation on one log axis, with the chance "
       "wall marked", width=740)}

<p>In plain terms, the test runs like this. Take a photograph the reader has
never seen. Hand it only the numbers, never the picture, and let it write a
sentence. Now take that sentence and ask which of 118,287 photographs it best
describes. Half the time the correct photograph comes back in the top 64. If the
sentence carried nothing at all you would expect it to land around 59,143, which
is the middle of the pile.</p>

<p>The two controls are the reason to believe the first number. Hand the reader
some <em>other</em> photograph&rsquo;s numbers and its sentence lands at 55,866,
which is nowhere. Hand it the average of every photograph at once and it still
writes fluent, plausible English, and that lands nowhere too. So the sentences
are not a caption habit it learned and repeats. They follow the particular
numbers they were given.</p>

<h3>A second thing we got wrong, in the other direction</h3>

<p>The same week we found the snowboard problem, we found one that had been
making us look worse than we are.</p>

<p>We had been quoting a human caption at median rank 8 against our reader&rsquo;s
64, and calling that the gap we fall short of. That figure is contaminated. The
shortening step was trained on pairs made from each photograph&rsquo;s
<em>first</em> caption, and the pool we score against contains those same
photographs, so scoring that exact caption measures a pair the system was
explicitly taught to align.</p>

<p>Score a different human&rsquo;s description of the same photograph, one the
system never saw, and the honest reference is 45. Rebuild the whole thing with
the evaluation photographs held out of every training step and the two come out
level: the reader at 46, a human at 48, and a second human at 57. Counted per
photograph, the reader places the image above the human on 49.6% of them, where
a second human manages 48.0%.</p>

<p>So one error had us claiming arithmetic we could not do, and the other had us
conceding a gap that was not there. We are publishing both.</p>

<h3>What the map carries, and what it does not</h3>

<p>The head this map is built on was measured against COCO&rsquo;s own
annotations rather than against a competing caption, so no text prior can reach
it. It recovers a scene&rsquo;s inventory nearly in full and its arrangement far
less well: per-category detection AUC <strong>0.883</strong> across the 80 COCO
categories, and per-image recovery of the full annotated object list at
<strong>0.543</strong> against a 0.038 chance floor, about fourteen times.</p>

<p>Three numbers are in play on the map itself and they are not interchangeable:
40,000 photographs are laid out on it, a click reads the average of the roughly
forty-eight nearest of those, and the eight neighbours shown beside the sentence
are retrieved from the full 123,287-image gallery.</p>

<p>None of this required retraining the 31B or touching one of its weights. It
was frozen throughout. The instruments are small trained artifacts that read a
hidden state the model had already computed.</p>

<h3>Press the buttons</h3>

<p><a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
is the Lab, pane 04. Click anywhere. Upload something of your own. The eight
nearest photographs are always on screen, so you can check the sentence against
them the way we should have.</p>

<p><a href="https://huggingface.co/spaces/RiverRider/walk-the-space">walk-the-space</a>
runs the same idea over a different model&rsquo;s gallery, hosted, with no upload
needed.</p>

<p><a href="https://github.com/space-bacon/SRT">github.com/space-bacon/SRT</a> has
the code, checkpoints, artifacts, and the negative results, including the
measurement that killed the headline of this post.</p>

<p>The reader, the map and its evaluation are on
<a href="https://huggingface.co/RiverRider/srt-sunstone-linear-head">Hugging Face</a>,
so the numbers above can be checked rather than taken.</p>
"""

HEAD = """<!doctype html><meta charset="utf-8">
<title>Lab map post - select all and copy into Substack</title>
<style>
 body { max-width: 760px; margin: 40px auto; padding: 0 20px;
        font: 18px/1.6 Georgia, "Times New Roman", serif; color: #1a1a1a; }
 img { max-width: 100%; height: auto; display: block; margin: 24px 0; }
 h3 { font-size: 24px; margin-top: 42px; }
 blockquote { border-left: 3px solid #c46a42; margin-left: 0; padding-left: 18px; color: #333; }
 table { border-collapse: collapse; width: 100%; font-size: 16px; margin: 20px 0; }
 th, td { border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }
 .howto { background:#fffbe6; border:1px solid #e8d9a0; padding:14px 18px;
           font:14px/1.5 -apple-system, sans-serif; margin-bottom:30px; }
</style>
<div class="howto"><strong>To publish:</strong> click into the article, select all (Cmd&nbsp;A),
copy (Cmd&nbsp;C), paste into Substack. Images travel with the copy. This box sits outside the
article and will not come along if you begin the selection at the headline.</div>
<article>
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ASSETS / "post.html"))
    a = p.parse_args()
    html = HEAD + body() + "\n</article>\n"
    Path(a.out).write_text(html)
    print(f"wrote {a.out}  ({len(html) / 1024 / 1024:.1f} MB)", flush=True)
