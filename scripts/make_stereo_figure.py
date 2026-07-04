"""Compose the before/after autostereogram figure for sharing.

Left: the raw autostereogram (what gemma-4 reads as noise).
Right: the figure recovered by simulated binocular fusion (what it then names).
Each panel is annotated with gemma-4's own greedy caption and the SRT read-out's
top words, so the flip from texture to shape is visible at a glance.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

BG = (19, 18, 51)
PANEL = (28, 26, 71)
VIOLET = (164, 141, 255)
INK = (234, 231, 251)
MUTED = (167, 161, 204)
LINE = (51, 46, 102)


def load_font(size: int, bold: bool = False, serif: bool = True):
    candidates = [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold and serif else None,
        "/System/Library/Fonts/Supplemental/Georgia.ttf" if serif else None,
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else None,
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def panel(draw, x, y, w, img_path, thumb, title, caption, readout, fonts):
    f_h, f_lbl, f_cap, f_read = fonts
    # image
    im = Image.open(img_path).convert("RGB").resize((thumb, thumb))
    ix = x + (w - thumb) // 2
    draw._img.paste(im, (ix, y))
    draw.rectangle([ix, y, ix + thumb - 1, y + thumb - 1], outline=LINE, width=1)
    cy = y + thumb + 18
    draw.text((x, cy), title, font=f_lbl, fill=VIOLET)
    cy += 30
    # gemma caption (quoted)
    for ln in wrap(draw, f'gemma-4: \u201c{caption}\u201d', f_cap, w):
        draw.text((x, cy), ln, font=f_cap, fill=INK); cy += 26
    cy += 6
    for ln in wrap(draw, "read-out: " + readout, f_read, w):
        draw.text((x, cy), ln, font=f_read, fill=MUTED); cy += 23
    return cy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stereo", default="artifacts/nla/gemma4/stereo/stereogram.png")
    p.add_argument("--decoded", default="artifacts/nla/gemma4/stereo/disparity.png")
    p.add_argument("--out", default="artifacts/nla/gemma4/stereo/stereo_figure.png")
    return p.parse_args()


def main():
    args = parse_args()
    W, H, thumb = 1120, 770, 380
    margin, gap = 56, 60
    pw = (W - 2 * margin - gap) // 2
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d._img = canvas

    f_title = load_font(38, bold=True)
    f_sub = load_font(21)
    f_lbl = load_font(20, bold=True)
    f_cap = load_font(19)
    f_read = load_font(17)
    fonts = (None, f_lbl, f_cap, f_read)

    d.text((margin, 34), "What a model sees vs. what it means", font=f_title, fill=INK)
    sy = 86
    for ln in wrap(d, "gemma-4 reads the flat pixels. Simulating binocular fusion "
                      "(eye vergence + correspondence) recovers the hidden figure.",
                   f_sub, W - 2 * margin):
        d.text((margin, sy), ln, font=f_sub, fill=MUTED); sy += 28

    top = 168
    py = top
    panel(d, margin, py, pw,
          args.stereo, thumb, "the autostereogram",
          "multicolored static or random noise",
          "speckles \u00b7 mosaic \u00b7 abstract", fonts)
    rx = margin + pw + gap
    panel(d, rx, py, pw,
          args.decoded, thumb, "after simulated fusion",
          "a white heart on a black background",
          "circle \u00b7 square \u00b7 shape (= the true silhouette)", fonts)

    # arrow between panels
    ay = top + thumb // 2
    ax0, ax1 = margin + pw + 12, rx - 12
    d.line([ax0, ay, ax1, ay], fill=VIOLET, width=3)
    d.polygon([(ax1, ay), (ax1 - 14, ay - 8), (ax1 - 14, ay + 8)], fill=VIOLET)

    foot = ("The base model cannot solve stereograms: the figure lives in disparity, "
            "which a flat-raster encoder never computes. Add the missing stereo stage "
            "and it names the heart, verbatim identical to the real silhouette.")
    fy = H - 66
    for ln in wrap(d, foot, f_read, W - 2 * margin):
        d.text((margin, fy), ln, font=f_read, fill=MUTED); fy += 22

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
