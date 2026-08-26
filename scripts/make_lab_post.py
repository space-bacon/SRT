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
<h3>Nobody put snowboarding there</h3>

<p><em>We drew a map of how one frozen 31-billion-parameter model saw 40,000
photographs. Then we clicked the empty spaces between the islands, and a
382&nbsp;MB model with no eyes told us what belongs in them.</em></p>

<p>There is a new instrument on the Sunstone North Lab, and the short version is
this. We took 40,000 photographs, showed each one to a very large model, and
kept the private working notes it made while looking. Those notes are nothing
but long lists of numbers. We then arranged the 40,000 lists on a flat sheet so
that photographs the model had seen as similar ended up near one another. That
sheet is the map, and it organised itself: animals gathered in the west,
transport in the southwest, sport along the north, kitchens and bathrooms down
in the southeast. Nobody sorted them.</p>

<p>Then comes the part that surprised us. Most of that sheet is empty. You can
put a finger down in a spot where there is no photograph at all, and a second,
much smaller model will write you a sentence describing the scene that belongs
there.</p>

<p>Here is the one that made us stop. The skiing photographs form an island. So
do the skateboarding photographs. Between them is a gap with nothing in it. We
clicked the gap, and the small model wrote:</p>

<blockquote><p>A man is doing a trick on a snowboard.</p></blockquote>

<p>Nobody put snowboarding on this map. There is no snowboarding island, no
snowboarding label, and no photograph at the spot we clicked. It is simply where
the arithmetic of the other two lands.</p>

{embed("fig_blend.png",
       "Zoomed: the skiing island, the skateboarding island, and the empty "
       "midpoint between them, which reads as snowboarding")}

{embed("lab_05_border.png",
       "The same click in the Lab: the reader writes snowboarding",
       caption="The same click in the Lab, live under pane 04 at "
               "lab.sunstonenorth.com.")}

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
those 1,024-number lists and writes an English sentence about it. It has no
eyes. There is no vision component anywhere inside it. It has never been shown a
photograph in its life. It is reading somebody else&rsquo;s handwriting and
telling you what it says.</p>

<p>The map is what you get when all 40,000 lists are laid on a sheet by
similarity. And because the reader wants a list of numbers rather than a
picture, you can point at a bare patch of sheet, average the lists surrounding
it, and hand the reader that average. No photograph ever had those exact
numbers. It writes a sentence anyway.</p>

<p>The map holds 40,000 photographs drawn from a gallery of 123,287, and you can
stand anywhere on it.
<a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
&middot; pane 04</p>

{embed("fig_poster.png",
       "The map: 40,000 photographs of a 123,287-image gallery in two "
       "dimensions, region names written by the reader",
       caption="Every dot is a photograph. Nothing arranged that. It is where "
               "the photographs fell.")}

<h3>We did not name the regions</h3>

<p>Every label on that map was written by the same small reader. We handed it
the centre of each of the 24 regions and printed what came back. The names you
are reading are the machine&rsquo;s account of its own filing system, arrived at
without us, months after the big model made the notes and on a different
computer.</p>

{embed("lab_01_map.png",
       "The map instrument running in the Lab, pane 04",
       caption="The same map inside the Lab, live.")}

<h3>It keeps doing this</h3>

<p>The snowboard was not a lucky click. Most of the map is empty: the islands
are where photographs cluster and the space between them holds none at all.
Click there anyway and the reader still answers, because a point in that space
is a scene rather than a picture.</p>

<p>Every panel below is a real click on the midpoint between two named regions.
Each coordinate is derived from the two region centres rather than picked by
hand, so the caption under each panel is true by construction:</p>

{embed("fig_openwater.png",
       "Eight midpoints between named regions, each with the sentence it returns",
       width=740)}

<p>Four of those are genuine blends. Three lean to one parent instead of
combining both, and the motorcycle-and-bus midpoint moves sideways to a parking
meter. We are showing all eight rather than the four that worked, because not
every midpoint is meaningful and a demo that only shows its wins is selling
something.</p>

<h3>Then walk across it</h3>

<p>If the space between regions is readable, you can travel through it. This is
a straight line from the bathroom island to the giraffe island, sampled six
times. Neither endpoint appears in the middle:</p>

{embed("fig_walk.gif",
       "Six stops along a straight line from the bathroom island to the giraffe "
       "island, each with the sentence it returns",
       caption="Domestic interior, to rail, to road, to water, to wildlife. Six "
               "coherent scenes, four of them nowhere near either end.")}

<p>That is the difference between a map and an index. An index can only return
things it holds.</p>

<p>Another, from snow to surf:</p>

<blockquote><p>
0% &middot; A man riding skis down a snowy slope.<br />
20% &middot; A young boy holding a frisbee in a park.<br />
40% &middot; A man flying a kite on a beach.<br />
60% &middot; A man carrying a surfboard in the ocean.<br />
80% &middot; A person standing on a beach with birds flying above.<br />
100% &middot; A group of people paddling in a body of water.
</p></blockquote>

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

<p>A sentence that sounds right proves nothing. A caption prior would also
produce plausible sentences about photographs. So the instrument shows the eight
photographs nearest wherever the marker landed, every time, without being
asked.</p>

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

<ul>
<li>The reader&rsquo;s sentence retrieves the picture it was written from at
<strong>median rank 64</strong>, the top 0.05% of the pool.</li>
<li>A human caption of the same photograph, one the head has never seen, gets
<strong>median 45</strong>.</li>
<li>Handed another photograph&rsquo;s point, the same reader scores at
<strong>chance</strong>: median 55,866.</li>
<li>Handed the average of all points, also chance: 59,135.</li>
</ul>

<p>Counted per photograph instead of per median, the reader places the image
above the human who was looking at it <strong>17% of the time</strong>. A second
human describing the same photograph beats the first <strong>20% of the
time</strong>. The reader sits inside the spread between two people rather than
below a line.</p>

<h3>The correction we made while writing this</h3>

<p>An earlier draft of this post quoted the human caption at median rank 8 and
called it the ceiling we fall short of. That figure is contaminated, and the
reason is worth stating plainly. The shortening step was trained on pairs made
from each photograph&rsquo;s <em>first</em> caption, and the pool we score
against contains those same photographs. So scoring that exact caption is not
measuring a human. It is measuring a pair the system was explicitly trained to
put in the same place, which it is very good at, because we taught it to be.</p>

<p>Use a different caption of the same photograph, one the system never saw, and
the honest human reference is 45. We had been quoting a number that made our own
result look considerably worse than it is.</p>

<p>The arm stays in the evaluation, because it has a job. It runs first and
aborts the whole run if human captions cannot retrieve their own images. A
reader pointed at the wrong space still produces perfectly fluent sentences over
a completely dead test, and that failure looks like a result rather than a bug.
It is a smoke alarm, not a scoreboard, and we had been reading it as a
scoreboard.</p>

<p>A different reader in the same family, one that reads the big model&rsquo;s
raw notes rather than this shortened 1,024-number version, does beat the human
caption outright: median rank 20 of 123,287 against 39. Part of that margin is
length. It keeps talking, the extra detail about the scene is often right, and a
retrieval metric rewards naming more true things. We quote it with that caveat
attached every time.</p>

<h3>What the map carries, and what it does not</h3>

<p>The head this map is built on was measured against COCO&rsquo;s own
annotations rather than against a competing caption, so no text prior can reach
it. It recovers a scene&rsquo;s inventory nearly in full and its arrangement far
less well: per-category detection AUC <strong>0.883</strong> across the 80 COCO
categories, and per-image recovery of the full annotated object list at
<strong>0.543</strong> against a 0.038 chance floor, about fourteen times.</p>

<p>That asymmetry is the honest description of the whole instrument. The map
knows what is in a scene and is much weaker on how the scene is arranged. When
you click open water and a coherent sentence comes back, what you are watching
is inventory arithmetic rather than a model picturing a photograph.</p>

<h3>What you are actually looking at</h3>

<p>The room has 1,024 dimensions. The map is a two-dimensional projection, so
what you see is neighbourhood structure and the coordinates mean nothing on
their own. Three numbers are in play and they are not interchangeable: 40,000
photographs are laid out on the map, a click reads the average of the roughly
forty-eight nearest of those, and the eight neighbours shown beside the sentence
are retrieved from the full 123,287-image gallery.</p>

<p>None of this required retraining the 31B or touching one of its weights. It
was frozen throughout. The instruments are small trained artifacts that read a
hidden state the model had already computed.</p>

<h3>What we tested while writing this</h3>

<p>Writing this turned up something we did not expect. The head was fitted on
one caption per image, the first of COCO&rsquo;s five. That is what contaminated
the number above, and it raises a second question. A head only ever asked to
match one description of a scene is never required to keep the rest of it. If
that were costing the reader, showing the head all five captions should improve
what the map can put into words without changing the reader at all.</p>

<p>We ran it while writing, and said we would publish it either way. It is a
negative result, so here it is. Showing the head all five captions improves the
space a great deal. It barely improves the reader:</p>

<table>
<tr><th align="left">median rank</th><th align="right">one caption</th><th align="right">all five</th></tr>
<tr><td>a human caption</td><td align="right">57</td><td align="right">39</td></tr>
<tr><td>a second human caption</td><td align="right">48</td><td align="right">28</td></tr>
<tr><td>the reader</td><td align="right">46</td><td align="right">40</td></tr>
</table>

<p>Everyone gets better and the reader gets better least, so it actually loses
ground: at one caption it edges a human 49.6% of the time, and at five captions
only 44.8%. Whatever is limiting the reader, it is not how many descriptions the
space was built from. The space and the sentence-writing turn out to be
separable problems, which is worth knowing and is not what we expected to
find.</p>

<h3>Press the buttons</h3>

<p><a href="https://lab.sunstonenorth.com"><strong>lab.sunstonenorth.com</strong></a>
is the Lab, pane 04. Click the empty water. Upload something of your own.</p>

<p><a href="https://huggingface.co/spaces/RiverRider/walk-the-space">walk-the-space</a>
runs the same idea over a different model&rsquo;s gallery, hosted, with no upload
needed.</p>

<p><a href="https://github.com/space-bacon/SRT">github.com/space-bacon/SRT</a> has
the code, checkpoints, artifacts, and the negative results.</p>

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
