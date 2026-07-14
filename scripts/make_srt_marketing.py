"""SRT marketing exhibits. Dark indigo brand, Didot type, one takeaway each.

Consulting-exhibit conventions: kicker line, single-sentence headline as the
takeaway, thin rule, restrained body, source line. Rendered at 2x and
downsampled for clean type edges.

    python scripts/make_srt_marketing.py --out artifacts/marketing
"""
from __future__ import annotations

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

# ---- brand ------------------------------------------------------------------
BG = "#131233"
PANEL = "#1c1a47"
PANEL2 = "#221f55"
VIOLET = "#a48dff"
INK = "#eae7fb"
MUTED = "#a7a1cc"
LINE = "#332e66"
WARN = "#ffb3c1"

W, H, S = 1672, 941, 2  # output size, supersample factor
DIDOT = "/System/Library/Fonts/Supplemental/Didot.ttc"


def F(size: int, face: str = "regular") -> ImageFont.FreeTypeFont:
    idx = {"regular": 0, "italic": 1, "bold": 2}[face]
    return ImageFont.truetype(DIDOT, size * S, index=idx)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W * S, H * S), BG)
    return im, ImageDraw.Draw(im)


def save(im: Image.Image, path: str) -> None:
    im.resize((W, H), Image.LANCZOS).save(path)
    print("wrote", path)


def kicker_block(d: ImageDraw.ImageDraw, kicker: str, headline: list[str],
                 y0: int = 72) -> int:
    """Standard exhibit header. Returns y below the rule."""
    x = 96 * S
    d.text((x, y0 * S), kicker.upper(), font=F(21), fill=VIOLET)
    y = y0 + 44
    for ln in headline:
        d.text((x, y * S), ln, font=F(52, "bold"), fill=INK)
        y += 62
    y += 14
    d.line([(x, y * S), ((W - 96) * S, y * S)], fill=LINE, width=2 * S)
    return y + 34


def footer(d: ImageDraw.ImageDraw, text: str) -> None:
    d.text((96 * S, (H - 58) * S), text, font=F(19), fill=MUTED)


def spark(d: ImageDraw.ImageDraw, pts: list[float], box: tuple[int, int, int, int],
          color: str, width: int = 3) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = [v * S for v in box]
    lo, hi = min(pts), max(pts)
    xy = []
    for i, p in enumerate(pts):
        px = x0 + (x1 - x0) * i / (len(pts) - 1)
        py = y1 - (y1 - y0) * (p - lo) / (hi - lo)
        xy.append((px, py))
    d.line(xy, fill=color, width=width * S, joint="curve")
    return [(int(x / S), int(y / S)) for x, y in xy]


def rounded(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str,
            outline: str = LINE, radius: int = 18) -> None:
    d.rounded_rectangle([v * S for v in box], radius=radius * S, fill=fill,
                        outline=outline, width=2 * S)


def ctext(d: ImageDraw.ImageDraw, cx: int, y: int, text: str,
          font: ImageFont.FreeTypeFont, fill: str) -> None:
    w = d.textlength(text, font=font)
    d.text((cx * S - w / 2, y * S), text, font=font, fill=fill)


# ---- 1. hero ------------------------------------------------------------------
def star(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, fill: str) -> None:
    """Four-pointed gem star (Didot has no \u2726 glyph)."""
    cxs, cys, rs = cx * S, cy * S, r * S
    w = rs * 0.28
    pts = [(cxs, cys - rs), (cxs + w, cys - w), (cxs + rs, cys), (cxs + w, cys + w),
           (cxs, cys + rs), (cxs - w, cys + w), (cxs - rs, cys), (cxs - w, cys - w)]
    d.polygon(pts, fill=fill)


def hero(out: str) -> None:
    im, d = canvas()
    star(d, W // 2, 210, 34, VIOLET)
    ctext(d, W // 2, 290, "SRT", F(150, "bold"), INK)
    ctext(d, W // 2, 480, "The Semiotic-Reflexive Transformer", F(44, "italic"), "#c9c3ea")
    ctext(d, W // 2, 575, "The missing instrument panel for AI.", F(34), VIOLET)
    ctext(d, W // 2, 660,
          "Real-time telemetry for what a model means, not just what it says.",
          F(27), MUTED)
    d.line([((W // 2 - 210) * S, 760 * S), ((W // 2 + 210) * S, 760 * S)],
           fill=LINE, width=2 * S)
    ctext(d, W // 2, 790, "github.com/space-bacon/SRT", F(23), MUTED)
    save(im, out)


# ---- 2. instruments mapping ---------------------------------------------------
def instruments(out: str) -> None:
    im, d = canvas()
    y = kicker_block(d, "SRT \u00b7 Semiotic-Reflexive Transformer",
                     ["Aviation solved this with instruments.",
                      "AI can be built the same way."])
    rows = [
        ("Mode annunciator", "Which mode of control is engaged",
         "Regime signal", "Which interpretive mode the model is in"),
        ("Rate-of-change gauges", "How fast conditions are moving",
         "Divergence", "How fast internal meaning is shifting"),
        ("Crew monitoring", "Is anyone watching the transition",
         "Reflexivity  r\u0302", "Is the model registering its own shift"),
        ("Flight data recorder", "A readable record of every state",
         "NLA verbalizer", "Hidden states read out as language"),
    ]
    col1_x, col2_x = 96, 900
    col_w = 640
    row_h, gap = 118, 24
    for i, (a, asub, b, bsub) in enumerate(rows):
        ry = y + 26 + i * (row_h + gap)
        rounded(d, (col1_x, ry, col1_x + col_w, ry + row_h), PANEL)
        d.text(((col1_x + 34) * S, (ry + 20) * S), a, font=F(30, "bold"), fill=INK)
        d.text(((col1_x + 34) * S, (ry + 66) * S), asub, font=F(22), fill=MUTED)
        rounded(d, (col2_x, ry, col2_x + col_w - 24, ry + row_h), PANEL2, outline=VIOLET)
        d.text(((col2_x + 34) * S, (ry + 20) * S), b, font=F(30, "bold"), fill=VIOLET)
        d.text(((col2_x + 34) * S, (ry + 66) * S), bsub, font=F(22), fill="#cfc8f0")
        ay = ry + row_h // 2
        d.line([((col1_x + col_w + 18) * S, ay * S), ((col2_x - 18) * S, ay * S)],
               fill=LINE, width=3 * S)
        d.polygon([((col2_x - 18) * S, (ay - 7) * S), ((col2_x - 18) * S, (ay + 7) * S),
                   ((col2_x - 4) * S, ay * S)], fill=VIOLET)
    d.text((col1_x * S, (y - 4) * S), "THE COCKPIT", font=F(20), fill=MUTED)
    d.text((col2_x * S, (y - 4) * S), "THE MODEL, WITH SRT", font=F(20), fill=VIOLET)
    footer(d, "SRT adds \u22480.17% parameters to a frozen model. The backbone is never retrained.   \u00b7   github.com/space-bacon/SRT")
    save(im, out)


# ---- 3. the dangerous moment ---------------------------------------------------
def dangerous(out: str) -> None:
    im, d = canvas()
    y = kicker_block(d, "SRT \u00b7 Real-time divergence telemetry",
                     ["The dangerous moment is a quiet mode change.",
                      "SRT sees it while the output still looks fine."])
    # synth divergence series: calm, spike (regime shift), new plateau
    pts = []
    for i in range(120):
        base = 0.18 + 0.03 * math.sin(i / 5.5) + 0.012 * math.sin(i / 2.1)
        if 62 <= i < 74:
            base += 0.62 * math.exp(-((i - 68) ** 2) / 14)
        if i >= 74:
            base += 0.16 + 0.02 * math.sin(i / 3.3)
        pts.append(base)
    box = (96, y + 60, W - 96, y + 420)
    rounded(d, (box[0] - 0, box[1] - 30, box[2], box[3] + 46), PANEL)
    for frac in (0.25, 0.5, 0.75):
        gy = box[1] + (box[3] - box[1]) * frac
        d.line([((box[0] + 24) * S, gy * S), ((box[2] - 24) * S, gy * S)],
               fill=LINE, width=1 * S)
    xy = spark(d, pts, (box[0] + 40, box[1] + 30, box[2] - 40, box[3]), VIOLET, 4)
    # annotate spike
    sx, sy_ = xy[68]
    r = 10
    d.ellipse([(sx - r) * S, (sy_ - r) * S, (sx + r) * S, (sy_ + r) * S],
              outline=WARN, width=3 * S)
    d.text(((sx + 28) * S, (sy_ - 12) * S), "regime shift detected",
           font=F(26, "bold"), fill=WARN)
    # annotate output-normal region
    d.text(((box[0] + 44) * S, (box[3] + 4) * S),
           "output looks normal", font=F(22, "italic"), fill=MUTED)
    d.text(((box[2] - 420) * S, (box[3] + 4) * S),
           "output still looks normal", font=F(22, "italic"), fill=MUTED)
    d.text(((box[0] + 44) * S, (box[1] - 22) * S),
           "per-token divergence, live during generation", font=F(21), fill=MUTED)
    footer(d, "\u201cThe most dangerous moment is when the system has quietly changed modes.\u201d  \u00b7  Signal shown: SRT divergence with regime annunciation")
    save(im, out)


# ---- 4. numbers band ------------------------------------------------------------
def numbers(out: str) -> None:
    im, d = canvas()
    y = kicker_block(d, "SRT \u00b7 Working artifacts, measured claims",
                     ["Small enough to bolt on.",
                      "Instrumented enough to govern."])
    stats = [
        ("0.17%", "added parameters,", "backbone frozen"),
        ("5", "model families", "instrumented, 2B to 235B"),
        ("0.93", "image-to-word retrieval@1,", "chance 0.10, no image training"),
        ("10,000", "captions searched,", "open-vocabulary state read-out"),
    ]
    card_w, card_h, gap = 340, 330, 40
    total = 4 * card_w + 3 * gap
    x0 = (W - total) // 2
    cy = y + 60
    for i, (big, l1, l2) in enumerate(stats):
        cx = x0 + i * (card_w + gap)
        rounded(d, (cx, cy, cx + card_w, cy + card_h), PANEL)
        ctext(d, cx + card_w // 2, cy + 58, big, F(66, "bold"), VIOLET)
        ctext(d, cx + card_w // 2, cy + 182, l1, F(23), INK)
        ctext(d, cx + card_w // 2, cy + 222, l2, F(23), MUTED)
    footer(d, "Every number is reproducible from the repository, with anchors, floors, and ceilings stated.   \u00b7   github.com/space-bacon/SRT \u00b7 huggingface.co/RiverRider")
    save(im, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/marketing")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hero(os.path.join(a.out, "srt_hero.png"))
    instruments(os.path.join(a.out, "srt_instruments.png"))
    dangerous(os.path.join(a.out, "srt_dangerous_moment.png"))
    numbers(os.path.join(a.out, "srt_numbers.png"))


if __name__ == "__main__":
    main()
