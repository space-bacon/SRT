#!/usr/bin/env python
"""Composed figures of the Lab's map, for writing about it.

The Lab renders the map into a 480px panel inside a scrolling app shell, which
is right for the instrument and wrong for a figure: the labels shrink to 8px
and the panel clips. These are drawn from the same artifacts the Lab serves,
at a size where the region names are legible.

Nothing here is illustrative. The layout is `space_map.npz`, the region names
are the ones the reader wrote, and every caption is a live response from the
running model server.

    python scripts/make_map_figures.py --walk --poster
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import urllib.request

import certifi
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HELV = "/System/Library/Fonts/HelveticaNeue.ttc"
DIDOT = "/System/Library/Fonts/Supplemental/Didot.ttc"
IVORY, INK, MUTED = (250, 247, 242), (38, 30, 26), (140, 126, 118)
TERRA, TERRA_SOFT = (196, 106, 66), (196, 106, 66, 120)
PANEL = (243, 238, 231)

SERVER = "http://127.0.0.1:8765"
# same source the Lab frontend uses (sunstone_panel.rs COCO_BASE)
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
_THUMBS: dict[tuple, Image.Image] = {}


def thumb(split, fname, size):
    """Centre-cropped square of a gallery photograph, fetched once."""
    key = (split, fname, size)
    if key not in _THUMBS:
        url = f"{COCO_BASE}/{split}/{fname}"
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2,
                      (w - s) // 2 + s, (h - s) // 2 + s))
        _THUMBS[key] = im.resize((size, size), Image.LANCZOS)
    return _THUMBS[key]


def font(size, idx=0, didot=False):
    return ImageFont.truetype(DIDOT if didot else HELV, size, index=idx)


def read_at_full(x, y):
    req = urllib.request.Request(
        f"{SERVER}/map/read", data=json.dumps({"x": x, "y": y}).encode(),
        headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def read_at(x, y):
    return read_at_full(x, y)["text"]


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def base_map(xy, size, pad, dot=2.0, alpha=110):
    """The dot field, drawn 2x and downsampled so the dots stay soft."""
    ss = 2
    im = Image.new("RGBA", (size * ss, size * ss), IVORY + (255,))
    d = ImageDraw.Draw(im)
    span = size * ss - 2 * pad * ss
    for px, py in xy:
        cx = pad * ss + (px + 1) * 0.5 * span
        cy = pad * ss + (1 - (py + 1) * 0.5) * span
        r = dot * ss
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=TERRA + (alpha,))
    return im.resize((size, size), Image.LANCZOS).convert("RGB")


def to_px(x, y, size, pad):
    span = size - 2 * pad
    return pad + (x + 1) * 0.5 * span, pad + (1 - (y + 1) * 0.5) * span


def marker(d, cx, cy, r=13):
    # Ink rather than terracotta: the dots are terracotta, so a terracotta
    # marker disappears into the very field it is meant to stand out from.
    d.ellipse((cx - r - 9, cy - r - 9, cx + r + 9, cy + r + 9),
              fill=(250, 247, 242, 190))
    d.ellipse((cx - r - 9, cy - r - 9, cx + r + 9, cy + r + 9),
              outline=INK + (140,), width=2)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=INK,
              outline=(250, 247, 242), width=4)


def load():
    z = np.load("artifacts/local/space_map.npz")
    labels = json.load(open("artifacts/local/space_map_labels.json"))["regions"]
    return z["xy"], labels


def make_walk(out, size=980, pad=34):
    xy, labels = load()
    stops = [(0.01, -0.62), (0.18, -0.36), (0.36, -0.10),
             (0.53, 0.15), (0.71, 0.40), (0.88, 0.66)]
    reads = [read_at_full(*s) for s in stops]
    says = [r["text"] for r in reads]
    nears = [r["near"][0] for r in reads]
    for s, t, n in zip(stops, says, nears):
        print(f"  {s} -> {t}\n      nearest {n['file']} cos {n['score']}",
              flush=True)

    base = base_map(xy, size, pad, dot=1.9, alpha=78)
    CAP = 150
    f_cap = font(30, 0)
    f_small = font(17, 10)
    f_pct = font(15, 1)

    frames = []
    TH = 150
    for i, ((mx, my), text) in enumerate(zip(stops, says)):
        im = Image.new("RGB", (size, size + CAP), IVORY)
        im.paste(base, (0, 0))
        d = ImageDraw.Draw(im, "RGBA")
        pts = [to_px(*s, size, pad) for s in stops[: i + 1]]
        if len(pts) > 1:
            d.line(pts, fill=INK + (90,), width=3, joint="curve")
        for px, py in pts[:-1]:
            d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(250, 247, 242, 220))
            d.ellipse((px - 6, py - 6, px + 6, py + 6), outline=INK + (150,), width=2)
        mxp, myp = pts[-1]
        tx = int(min(max(mxp + 18, 16), size - TH - 16))
        ty = int(min(max(myp - 18 - TH, 16), size - TH - 16))
        d.line([mxp, myp, tx + TH // 2, ty + TH // 2], fill=TERRA_SOFT, width=3)
        d.rectangle([tx - 5, ty - 5, tx + TH + 4, ty + TH + 4], fill=IVORY)
        im.paste(thumb(nears[i]["split"], nears[i]["file"], TH), (tx, ty))
        d.rectangle([tx - 5, ty - 5, tx + TH + 4, ty + TH + 4],
                    outline=TERRA, width=3)
        marker(d, mxp, myp)
        d.rectangle((0, size, size, size + CAP), fill=PANEL)
        d.line((0, size, size, size), fill=(226, 216, 205), width=2)
        d.text((34, size + 22), f"STOP {i + 1} OF {len(stops)}", font=f_pct, fill=TERRA)
        for k, line in enumerate(wrap(d, text, f_cap, size - 68)[:2]):
            d.text((34, size + 48 + k * 38), line, font=f_cap, fill=INK)
        d.text((size - 34, size + 22), "lab.sunstonenorth.com", font=f_small,
               fill=MUTED, anchor="ra")
        frames.append(im)

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=[2300] * (len(frames) - 1) + [3600], loop=0, optimize=True)
    print(f"wrote {out}", flush=True)


def make_poster(out, size=1500, pad=54):
    """The whole map at a size where the reader's own names are readable."""
    xy, labels = load()
    im = Image.new("RGB", (size, size + 96), IVORY)
    im.paste(base_map(xy, size, pad, dot=1.7, alpha=95), (0, 96))
    d = ImageDraw.Draw(im, "RGBA")

    d.text((pad, 30), "A map of what a 31B has read",
           font=font(40, didot=True), fill=INK)
    d.text((size - pad, 44), f"{len(xy):,} of 123,287 photographs",
           font=font(19, 10), fill=MUTED, anchor="ra")

    f_lab = font(19, 10)
    # Lay every label out first, nudging overlaps apart, then paint all the
    # chips, then all the text. Painting each label complete in turn lets a
    # later chip cover an earlier label's words.
    placed = []
    for r in sorted(labels, key=lambda r: -r["n"])[:16]:
        cx, cy = to_px(r["x"], r["y"], size, pad)
        cy += 96
        lines = wrap(d, r["text"].rstrip("."), f_lab, 260)[:3]
        h = len(lines) * 25 + 14
        w = max(d.textlength(t, font=f_lab) for t in lines) + 22
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        for _ in range(60):
            hit = next((p for p in placed
                        if box[0] < p["box"][2] and box[2] > p["box"][0]
                        and box[1] < p["box"][3] and box[3] > p["box"][1]), None)
            if not hit:
                break
            shift = (hit["box"][3] - box[1]) + 6
            box[1] += shift
            box[3] += shift
        placed.append({"box": box, "lines": lines})

    for p in placed:
        d.rounded_rectangle(tuple(p["box"]), 8, fill=(250, 247, 242, 234))
    for p in placed:
        mid = (p["box"][0] + p["box"][2]) / 2
        for k, t in enumerate(p["lines"]):
            d.text((mid, p["box"][1] + 8 + k * 25), t, font=f_lab, fill=INK, anchor="ma")
    d.text((pad, size + 60), "Every name written by a frozen 0.6B from that "
           "region's own centre, not chosen by us.", font=font(20, 2), fill=MUTED)
    im.save(out, quality=94)
    print(f"wrote {out}", flush=True)


def make_blend(out, size=1180):
    """Skiing, the skiing/skateboard midpoint, and skateboarding.

    RETIRED, and do not restore it without re-measuring. This was published as
    a blend of two regions reading as a third thing with no photograph in it.
    That was false. All eight gallery photographs nearest the midpoint are
    snowboarding shots, the nearest at cos 0.9242 captioned "A person a
    snowboard in the air doing a trick". The midpoint reads as snowboarding
    because snowboarding photographs are sitting there. It was retrieval.

    Three things made it look otherwise: the 24 region labels include no
    snowboarding label, the UMAP projection has a hole at that coordinate so
    the nearest plotted dot is 0.094 away, and the returned sentence was
    correct. Emptiness is a measurement in the 1024-d space, never something
    you can see in a 2-d projection. See probe_open_water.py.
    """
    xy, labels = load()
    a = next(r for r in labels if "skiing" in r["text"])
    b = next(r for r in labels if "skateboard" in r["text"])
    mid = ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
    trio = [(a, read_at(a["x"], a["y"])), (None, read_at(*mid)), (b, read_at(b["x"], b["y"]))]
    for r, t in trio:
        print(f"  {(r['text'] if r else 'THE GAP'):42s} -> {t}", flush=True)

    c = np.array(mid)
    lo, hi = c - 0.29, c + 0.29
    m = ((xy >= lo) & (xy <= hi)).all(1)
    sub = (xy[m] - lo) / (hi - lo)
    print(f"  {m.sum():,} photographs in view", flush=True)

    CAP = 196
    im = Image.new("RGB", (size, size + CAP), IVORY)
    d = ImageDraw.Draw(im, "RGBA")
    pad = 40
    span = size - 2 * pad
    for px, py in sub:
        cx, cy = pad + px * span, pad + (1 - py) * span
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=TERRA + (105,))

    def place(x, y):
        u = (np.array([x, y]) - lo) / (hi - lo)
        return pad + u[0] * span, pad + (1 - u[1]) * span

    f_side = font(23, 10)
    f_gap = font(30, 1)
    pa, pb = place(a["x"], a["y"]), place(b["x"], b["y"])
    d.line((pa, pb), fill=INK + (85,), width=3)
    for r in (a, b):
        cx, cy = place(r["x"], r["y"])
        marker(d, cx, cy, r=9)
        lines = wrap(d, r["text"].rstrip("."), f_side, 330)
        w = max(d.textlength(t, font=f_side) for t in lines) + 24
        h = len(lines) * 30 + 16
        d.rounded_rectangle((cx - w / 2, cy + 26, cx + w / 2, cy + 26 + h), 9,
                            fill=(250, 247, 242, 236))
        for k, t in enumerate(lines):
            d.text((cx, cy + 34 + k * 30), t, font=f_side, fill=MUTED, anchor="ma")

    gx, gy = place(*mid)
    d.ellipse((gx - 46, gy - 46, gx + 46, gy + 46), outline=INK + (110,), width=3)
    marker(d, gx, gy, r=15)

    d.rectangle((0, size, size, size + CAP), fill=PANEL)
    d.line((0, size, size, size), fill=(226, 216, 205), width=2)
    d.text((44, size + 26), "THE EMPTY GAP BETWEEN THEM READS AS",
           font=font(17, 1), fill=TERRA)
    for k, t in enumerate(wrap(d, trio[1][1], f_gap, size - 88)[:2]):
        d.text((44, size + 58 + k * 40), t, font=f_gap, fill=INK)
    d.text((44, size + 148), "No photograph lives there. Nobody put snowboarding "
           "on the map; it is where the arithmetic of the two puts it.",
           font=font(19, 2), fill=MUTED)
    im.save(out, quality=94)
    print(f"wrote {out}", flush=True)


def make_ladder(out, size=1420):
    """Every arm of the reader evaluation on one log axis.

    A table hides how far apart these are. On a log axis the two control
    arms sit against the chance wall and the three real arms are crushed
    into the left margin, which is the actual result.
    """
    rows = [
        ("The image's own first caption", 8, "the head was trained on this pair", True),
        ("A second human caption of it", 45, "never seen by the head", False),
        ("The reader, from the point alone", 64, "no photograph, no caption", False),
        ("Another photograph's point", 55866, "control", False),
        ("The average of every point", 59135, "control", False),
    ]
    POOL, CHANCE = 118287, 59143
    CAP = 176
    im = Image.new("RGB", (size, CAP + len(rows) * 96 + 108), IVORY)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((54, 46), "WHAT THE READER RECOVERS", font=font(21, 1), fill=TERRA)
    d.text((52, 76), "Median rank of the right photograph",
           font=font(44, 2, didot=True), fill=INK)
    d.text((54, 138), f"in a pool of {POOL:,}. Further left is better.",
           font=font(21, 2), fill=MUTED)

    x0, x1 = 560, size - 130
    lg = lambda r: (np.log10(max(r, 1)) / np.log10(POOL))

    # The chance wall: anything landing in here has recovered nothing.
    cx = x0 + lg(CHANCE) * (x1 - x0)
    d.rectangle((cx, CAP - 26, x1 + 46, CAP + len(rows) * 96 - 30),
                fill=(196, 106, 66, 26))
    d.line((cx, CAP - 26, cx, CAP + len(rows) * 96 - 30), fill=TERRA + (140,), width=2)
    d.text((cx + 10, CAP - 50), f"CHANCE  {CHANCE:,}", font=font(16, 1), fill=TERRA)

    for i, (label, rank, note, trained) in enumerate(rows):
        y = CAP + i * 96
        d.text((54, y - 8), label, font=font(25, 10), fill=INK)
        d.text((54, y + 26), note, font=font(18, 2),
               fill=TERRA if trained else MUTED)
        px = x0 + lg(rank) * (x1 - x0)
        d.line((x0, y + 8, px, y + 8), fill=INK + (58,), width=3)
        r = 12
        d.ellipse((px - r, y + 8 - r, px + r, y + 8 + r),
                  fill=TERRA if trained else INK,
                  outline=IVORY, width=3)
        d.text((px + 24, y - 6), f"{rank:,}", font=font(27, 1),
               fill=TERRA if trained else INK)

    y = CAP + len(rows) * 96 + 4
    d.line((54, y, size - 54, y), fill=(226, 216, 205), width=2)
    d.text((54, y + 20),
           "The top row is not a human ceiling. The head was fitted on each "
           "image's first caption, so that arm scores a pair it was trained to "
           "align.",
           font=font(19, 2), fill=MUTED)
    im.save(out, quality=95)
    print(f"wrote {out}", flush=True)


def make_openwater(out, cols=4, cell=400):
    """Midpoints between named regions, each with the sentence the map returns.

    Every coordinate here is derived from the two region centres it sits
    between, so the caption is true by construction. Writing the pairs down
    from memory produced captions that named the wrong neighbours.

    Each panel also carries the single nearest gallery photograph, so a reader
    can see what the sentence is describing. This figure deliberately does NOT
    claim these points are empty. Whether a point is genuinely somewhere no
    photograph sits is a measurement, not a look, and it lives in
    `scripts/probe_open_water.py`. Most of these are not.
    """
    xy, labels = load()

    def find(kw):
        hits = [r for r in labels if kw.lower() in r["text"].lower()]
        if not hits:
            raise SystemExit(f"no region matching {kw!r}")
        return max(hits, key=lambda r: r["n"])

    pairs = [("skiing", "skiing", "skateboard", "skateboarding"),
             ("kite", "a kite on a beach", "frisbee", "a frisbee"),
             ("living room", "a living room", "cat sitting on a bed", "a cat on a bed"),
             ("kitchen", "a kitchen", "plate of food", "a plate of food"),
             ("giraffes", "giraffes", "elephants", "elephants"),
             ("tennis", "tennis", "baseball", "baseball"),
             ("motorcycle", "a motorcycle", "bus is parked", "a parked bus"),
             ("jetliner", "a jetliner", "train traveling", "a train")]

    spots = []
    for ka, na, kb, nb in pairs:
        a, b = find(ka), find(kb)
        mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
        got = read_at_full(mx, my)
        said, near0 = got["text"], got["near"][0]
        spots.append((mx, my, f"between {na} and {nb}", said, near0))
        print(f"  ({mx:+.2f},{my:+.2f}) {na} + {nb}\n      {a['text']}\n"
              f"      {b['text']}\n      -> {said}\n"
              f"      nearest {near0['file']} cos {near0['score']}", flush=True)

    rows_n = (len(spots) + cols - 1) // cols
    CAPH = 128
    W, H = cols * cell, rows_n * (cell + CAPH) + 116
    im = Image.new("RGB", (W, H), IVORY)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((38, 32), "EIGHT MIDPOINTS BETWEEN NAMED REGIONS",
           font=font(20, 1), fill=TERRA)
    d.text((36, 60), "the sentence the map returns, and the nearest photograph",
           font=font(36, 2, didot=True), fill=INK)

    base = base_map(xy, cell - 36, 14, dot=1.1, alpha=62)
    TH = 84
    for i, (x, y, where, said, near0) in enumerate(spots):
        cxp, cyp = (i % cols) * cell, 116 + (i // cols) * (cell + CAPH)
        im.paste(base, (cxp + 18, cyp))
        px, py = to_px(x, y, cell - 36, 14)
        mxp, myp = cxp + 18 + px, cyp + py
        # nearest photograph, pinned up-right of the marker and kept in-cell
        tx = int(min(max(mxp + 12, cxp + 22), cxp + cell - TH - 22))
        ty = int(min(max(myp - 12 - TH, cyp + 10), cyp + cell - 52 - TH))
        d.line([mxp, myp, tx + TH // 2, ty + TH // 2], fill=TERRA_SOFT, width=2)
        d.rectangle([tx - 4, ty - 4, tx + TH + 3, ty + TH + 3], fill=IVORY)
        im.paste(thumb(near0["split"], near0["file"], TH), (tx, ty))
        d.rectangle([tx - 4, ty - 4, tx + TH + 3, ty + TH + 3],
                    outline=TERRA, width=2)
        marker(d, mxp, myp, r=8)
        for k, t in enumerate(wrap(d, where.upper(), font(13, 1), cell - 50)[:2]):
            d.text((cxp + 20, cyp + cell - 42 + k * 17), t,
                   font=font(13, 1), fill=TERRA)
        for k, t in enumerate(wrap(d, said, font(19, 10), cell - 52)[:3]):
            d.text((cxp + 20, cyp + cell + 2 + k * 25), t,
                   font=font(19, 10), fill=INK)
    im.save(out, quality=95)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--walk", action="store_true")
    p.add_argument("--poster", action="store_true")
    p.add_argument("--blend", action="store_true")
    p.add_argument("--ladder", action="store_true")
    p.add_argument("--openwater", action="store_true")
    p.add_argument("--outdir", default="artifacts/marketing/lab_map_post")
    a = p.parse_args()
    if a.walk:
        make_walk(f"{a.outdir}/fig_walk.gif")
    if a.poster:
        make_poster(f"{a.outdir}/fig_poster.png")
    if a.blend:
        make_blend(f"{a.outdir}/fig_blend.png")
    if a.ladder:
        make_ladder(f"{a.outdir}/fig_ladder.png")
    if a.openwater:
        make_openwater(f"{a.outdir}/fig_openwater.png")
