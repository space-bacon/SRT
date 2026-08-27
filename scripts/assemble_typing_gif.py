#!/usr/bin/env python
"""Assemble the pane-04 typing GIF from Playwright screenshots.

The frames are real captures of the live Lab, taken while filling the sentence
box two characters at a time and then clicking Place it. Playwright cannot
render a mouse pointer, and the point of the figure is that a person types and
presses a button, so the pointer and the click pulse are drawn here.

Capture (browser at 2100x1250, which is 1680x1000 CSS at dpr 1.25):

    await input.click(); await input.fill('');
    for (let i = 2; i <= s.length; i += 2) { await input.fill(s.slice(0, i));
        await page.screenshot({path: `${dir}/f${...}.png`}); }
    await page.evaluate(() => [...document.querySelectorAll('button')]
        .find(b => b.textContent.trim() === 'Place it').click());
    // then screenshot on a widening delay while the reader answers

Click through page.evaluate rather than locator.click(): the input narrows as
text is entered and the button slides left, so Playwright's actionability
check keeps racing the reflow and times out against the paragraph above.

    python scripts/assemble_typing_gif.py
"""
from __future__ import annotations

import argparse
import glob

from PIL import Image, ImageDraw

CROP = (420, 300, 1760, 1090)
OUT_W = 980
INK = (38, 30, 26)
IVORY = (250, 247, 242)
TERRA = (196, 106, 66)
# element centres in CROP coordinates, before scaling
INPUT_C = (901, 193)
BUTTON_C = (1242, 193)
TYPING_END = 22


def pointer(im, x, y, press=0.0):
    """Arrow pointer, with a click pulse when press > 0."""
    d = ImageDraw.Draw(im, "RGBA")
    if press:
        r = int(10 + 26 * press)
        a = int(150 * (1 - press))
        d.ellipse((x - r, y - r, x + r, y + r), outline=TERRA + (a,), width=3)
    arrow = [(x, y), (x, y + 17), (x + 4, y + 13), (x + 7, y + 20),
             (x + 10, y + 19), (x + 7, y + 12), (x + 12, y + 12)]
    d.polygon(arrow, fill=IVORY, outline=IVORY)
    d.line(arrow + [(x, y)], fill=INK, width=3, joint="curve")
    d.polygon(arrow, fill=INK)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="/tmp/typeframes")
    p.add_argument("--out",
                   default="artifacts/marketing/lab_map_post/lab_type.gif")
    a = p.parse_args()

    src = sorted(glob.glob(f"{a.frames}/*.png"))
    if not src:
        raise SystemExit(f"no frames in {a.frames}")
    sc = OUT_W / (CROP[2] - CROP[0])
    ic = (INPUT_C[0] * sc, INPUT_C[1] * sc)
    bc = (BUTTON_C[0] * sc, BUTTON_C[1] * sc)

    def load(i):
        im = Image.open(src[i]).convert("RGB").crop(CROP)
        return im.resize((OUT_W, int((CROP[3] - CROP[1]) * sc)), Image.LANCZOS)

    frames, durs = [], []

    def add(im, ms):
        frames.append(im.quantize(colors=96, method=Image.MEDIANCUT))
        durs.append(ms)

    for i in range(min(TYPING_END, len(src))):
        im = load(i)
        pointer(im, *ic)
        add(im, 700 if i < 2 else 90)

    last = load(TYPING_END - 1)
    for k in range(1, 7):                      # pointer travels to the button
        im = last.copy()
        pointer(im, *lerp(ic, bc, k / 6))
        add(im, 90)
    for press in (0.35, 0.75):                 # click
        im = last.copy()
        pointer(im, *bc, press=press)
        add(im, 170)

    tail = src[TYPING_END:]
    for i in range(len(tail)):
        im = load(TYPING_END + i)
        pointer(im, *bc)
        add(im, 260 if i < len(tail) - 1 else 3200)

    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=durs, loop=0, optimize=True)
    print(f"wrote {a.out}  {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
