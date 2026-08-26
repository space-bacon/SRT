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
<h3>A 382 MB model reads the notes a 31-billion-parameter model leaves behind</h3>

<p><em>The notes are what the big model computes and would otherwise discard.
Handed them and nothing else, the small one writes a sentence. Search 118,287
photographs with that sentence and the right one comes back in the top
0.05%.</em></p>

<p>When a large model looks at a photograph it does not keep the picture. It
builds a working impression: a list of several thousand numbers it uses to
answer whatever is asked next. The technical name is a hidden state. Think of it
as the model&rsquo;s working notes on the image, written in numbers rather than
words. It is a record of what the model made of the picture, and it is normally
thrown away the instant the answer is produced.</p>

<p>Every time one of these models looks at
anything it builds one of these records, in enough detail to answer
questions about it, and then discards it. You paid for that. It is gone
before you could use it for anything else.</p>

<p>We kept the notes for a gallery of 123,287 photographs, the standard COCO
image set, and then taught something very
small to read them. A 382&nbsp;MB language model, Qwen3-0.6B, with a small
trained piece in front of it and its own weights never altered, takes one of
those number-lists and writes an English sentence
about a photograph it was never given. We call it the reader.</p>

<p>The reader&rsquo;s blindness is the point rather than a curiosity. If it could
see, you could never tell whether its sentence came from the record or from the
picture. Everything it says came out of the numbers, because the numbers are all
it was given.</p>

<p>The notes turn out to be durable, too. The 31B did its looking once, months
ago, on a different machine, and has not run since. The record it left behind
still answers questions today. Compute once, read for as long as you like.</p>

<p><a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
&middot; pane 04. The whole thing is served from one desktop computer.</p>

<h3>The notes sort themselves</h3>

<p>What follows is a map, and the map is a demonstration rather than the
product. What it demonstrates is that the discarded record is rich enough to
search, to describe and to navigate, using components small enough to run
alongside whatever you already have.</p>

<p>A second trained piece shortens each list from 5,376 numbers to 1,024, in a
way that puts a photograph and a sentence describing it in nearly the same
place. Lay 40,000 of those shortened lists on a flat sheet, positioned so that
photographs the model saw as similar land near one another, and the sheet
organises itself. Animals gather in the west, transport in the southwest, sport
along the north, kitchens and bathrooms down in the southeast.</p>

<p>That shortening step is a single linear layer.
A linear map can rotate a space, stretch it and pick out directions
within it. What it cannot do is invent structure the notes did not contain. So
the regions came with the notes, and a linear view was enough to find them. The
flattening onto a screen is ours. The arrangement is the model&rsquo;s.</p>

{embed("fig_poster.png",
       "The map: 40,000 photographs of a 123,287-image gallery in two "
       "dimensions, region names written by the reader",
       caption="Every dot is a photograph, sitting where the model&rsquo;s "
               "reading of it put it.")}

<h3>The reader named the regions</h3>

<p>Every label on that map was written by the same small reader. We handed it
the centre of each of the 24 regions and printed what came back. The names you
are reading are the machine&rsquo;s account of its own filing system, written
months after the big model made the notes and on a different computer.</p>

<h3>Every point has a sentence, not just the photographs</h3>

<p>Touch any point and the instrument
averages the number-lists of the 48 nearest photographs and hands that average
to the reader, so a point between two regions returns a sentence for a place
rather than for a picture. Here are eight of them, each one the midpoint
between two named regions:</p>

{embed("fig_openwater.png",
       "Eight midpoints between named regions, each with the sentence the map "
       "returns there",
       width=740,
       caption="Most of these are describing photographs that genuinely sit "
               "nearby, which is the instrument working rather than the "
               "instrument being clever. The midpoint between the kitchen and "
               "the plate of food is the one that is not: its neighbours split "
               "52 to 48 between the two regions and its best match anywhere in "
               "the gallery is weaker than 60% of photographs manage with their "
               "own nearest neighbour.")}

<p>The map is mostly ivory because a dot is a photograph, and 40,000 photographs
do not fill a space this size. The pale areas are not dead, though. The
instrument takes the 48 nearest photographs however far away they are, so every
click returns a sentence. What varies is how much of a real place that sentence
describes, and it is measurable: across these eight midpoints the best matching
photograph in the gallery ranges from the 40th to the 95th percentile of what an
ordinary photograph manages with its own nearest neighbour. Dense ground reads
as a scene. Thin ground reads as a distant crowd.</p>

<h3>Every stop on the way is its own scene</h3>

<p>If any point can be read, so can every point on a line drawn between two of
them. This line runs from the bathroom region in the southeast to the giraffe
region in the west, read at six evenly spaced stops along the way:</p>

{embed("fig_walk.gif",
       "Six stops along a line across the map, from the bathroom region to the "
       "giraffe region, each with the sentence it returns",
       caption="Domestic interior, to rail, to road, to water, to wildlife. Six "
               "coherent scenes, four of them nowhere near either end.")}

<p>The line does not fade from one end to the other. It crosses whatever else
lies between them and reads each thing correctly on the way. Every stop is
somewhere, so you can search this map by moving as well as by typing.</p>

<h3>The map runs in both directions</h3>

<p>Type a sentence and it is placed by meaning,
then read back from wherever it landed:</p>

{embed("lab_04_sentence.png",
       "Typing a sentence places it on the map; the eight nearest photographs "
       "are all skiing",
       caption="&ldquo;a man riding skis down a snowy slope&rdquo; lands in the "
               "snow country, and the eight photographs nearest that spot are "
               "all skiing. The match is on meaning; the words never met.")}

<h3>Your own photograph needs the big model live</h3>

<p>Everything above comes from notes computed months ago, which a static page
could serve. Your own photograph is different. No notes exist for it yet, and
making them needs the 31-billion-parameter model running at that moment.</p>

<p>The Lab runs one, though not the way that sentence usually implies. The
gemma-4-31B here is quantized to four bits and served from that same desktop, an
Apple M2 Ultra with 64&nbsp;GB of memory. When you drop in a picture it is
encoded here, live, into the same 1,024-dimensional space the gallery was built
in. The marker lands among its actual neighbours rather than near words about
it.</p>

{embed("lab_upload.gif",
       "Uploading a photograph: encoded by the live 31B, lands on the map, described in a sentence",
       caption="2.0 seconds to encode on the 31B, 0.7 seconds for the 0.6B to "
               "read it: &ldquo;Two cats are laying on a couch.&rdquo;")}

<h3>The control is on the screen, not in a footnote</h3>

<p>A sentence that sounds right proves nothing. A model that had simply learned
the house style of photograph captions would also produce fluent, plausible
English about nothing at all. So the instrument shows you the eight photographs
nearest wherever the marker landed, every time, without being asked. You can
check the sentence against them yourself.</p>

{embed("lab_03_neighbours.png",
       "The eight photographs nearest the uploaded cat picture: all eight are cats",
       caption="All eight are cats. If the marker had landed anywhere at all, "
               "these would be unrelated, and you would be able to see it "
               "without trusting us.")}

<h3>How good is it, in numbers</h3>

<p>Measured on 500 held-out photographs against a pool of 118,287, so chance is
a median rank of 59,143:</p>

{embed("fig_ladder.png",
       "Every arm of the reader evaluation on one log axis, with the chance "
       "wall marked", width=740)}

<p>In plain terms, the test runs like this. Take a photograph the reader has
never seen. Hand it only the numbers and let it write a
sentence. Now take that sentence and ask which of 118,287 photographs it best
describes. Half the time the correct photograph comes back in the top 64, which
is the top 0.05% of the pile. If the sentence carried nothing you would expect
around 59,143.</p>

<p>The two controls are the reason to believe that. Hand the reader some
<em>other</em> photograph&rsquo;s numbers and its sentence lands at 55,866, which
is nowhere. Hand it the average of every photograph at once and it still writes
fluent, plausible English, and that lands nowhere too. The sentences follow the
particular numbers they were given.</p>

<h3>Top of the human range, once you measure it properly</h3>

<p>The obvious comparison is a person. Ours was wrong for a while, and it was
wrong against us.</p>

<p>We had been quoting a human caption at median rank 8 against the
reader&rsquo;s 64 and calling that the gap. That figure is contaminated. The
shortening step was trained on pairs built from each photograph&rsquo;s
<em>first</em> caption, and the pool we score against contains those same
photographs, so scoring that exact caption measures a pair the system was
explicitly taught to align. It is a wiring check, not a ceiling.</p>

<p>Score a different person&rsquo;s description of the same photograph, one the
system never saw, and the honest reference for the instrument above is 45
against its 64. So on the version you can go and click, a human still wins.</p>

<p>Then we rebuilt the whole thing with the evaluation photographs held out of
<em>every</em> training step, not just the reader&rsquo;s, which is what should
have been true the first time. On that rebuild the reader comes out top of the
three:</p>

<table>
<tr><th align="left">arm</th><th align="right">median rank of 118,287</th></tr>
<tr><td><strong>the reader</strong></td><td align="right"><strong>46</strong></td></tr>
<tr><td>a second person&rsquo;s caption</td><td align="right">48</td></tr>
<tr><td>the first person&rsquo;s caption</td><td align="right">57</td></tr>
</table>

<p>Counted per photograph rather than per median, the reader places the image
above the first person&rsquo;s caption on <strong>49.6%</strong> of them. A
second person manages <strong>48.0%</strong>. The reader beats a human caption
slightly more often than another human does.</p>

<p>A 382&nbsp;MB model describing a
photograph from numbers alone, as well as a person who is
looking straight at it. Not because it is a good captioner, but because the
record it was handed already contained the scene.</p>
<p>The margins are small and we are not going to inflate them. Two ranks out of
118,287 and 1.6 points of win rate put the reader at the top of the human band
rather than clear of it. The claim is parity, and parity is the result. The
ordering inside that band runs our way on both measures.</p>
<p>That rebuild is a measurement, not a release. The Lab serves the instrument
in the table before it, and the two should not be quoted as one number.</p>

<h3>What it carries, and what it does not</h3>

<p>Measured against COCO&rsquo;s own object labels rather than against a competing
caption, so nothing can be won by guessing what usually goes with what, the
shortened record recovers a
scene&rsquo;s inventory nearly in full and its arrangement far less well:
per-category detection AUC <strong>0.883</strong> across the 80 categories, and
per-image recovery of the full annotated object list at <strong>0.543</strong>
against a 0.038 chance floor, about fourteen times.</p>

<p>It recovers
what is in a scene. It is much weaker on how the scene is arranged. Ask it what
is there and it is reliable; ask it who is doing what to whom and it is not.</p>

<p>Three numbers are in play on the map and they are not interchangeable: 40,000
photographs are laid out on it, a click reads the average of the
forty-eight nearest of those, and the eight neighbours shown beside the sentence
are retrieved from the full 123,287-image gallery.</p>

<p>None of this required retraining the 31B or touching one of its weights. It
was frozen throughout. The instruments are small trained artifacts that read a
hidden state the model computed anyway, which is why the whole thing fits
on one desktop, four-bit host and all.</p>

<p>If you
already run a multimodal model, every forward pass already produces this record
and you are already paying for it. A linear read-out turns it into search,
tagging and description without a second model in the stack, without a
fine-tune, and without the original model changing by one weight.</p>

<h3>Press the buttons</h3>

<p><a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
is the Lab, pane 04. Click anywhere on the map, type a sentence into it, or drop
in a photograph of your own. The eight nearest photographs are always on screen,
so nothing here needs taking on trust.</p>

<p><a href="https://huggingface.co/spaces/RiverRider/walk-the-space">walk-the-space</a>
runs the same idea over a different model&rsquo;s gallery, hosted, with no upload
needed.</p>

<p><a href="https://github.com/space-bacon/SRT">github.com/space-bacon/SRT</a> has
the code, the checkpoints, the artifacts and the negative results, including the
measurements that took claims out of this piece.</p>

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
