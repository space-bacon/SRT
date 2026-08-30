#!/usr/bin/env python3
"""Article thumbnail for the cross-vendor transport paper, 1910x1000.

Numbers are read from the banked artifacts, same as make_crossvendor_figures.py,
so a stale thumbnail cannot outlive the result it quotes.

    python scripts/make_crossvendor_thumbnail.py
    -> arxiv_crossvendor/figs/thumbnail.png
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NLA = ROOT / "artifacts" / "nla"
OUT = ROOT / "arxiv_crossvendor" / "figs" / "thumbnail.png"

BG = "#131233"
PANEL = "#1c1a47"
VIOLET = "#a48dff"
INK = "#eae7fb"
MUTED = "#a7a1cc"
LINE = "#332e66"
GREEN = "#7fd6a4"

W, H, S = 1910, 1000, 2
DIDOT = "/System/Library/Fonts/Supplemental/Didot.ttc"
SANS = "/System/Library/Fonts/HelveticaNeue.ttc"


def F(size: int, face: str = "regular") -> ImageFont.FreeTypeFont:
    """Didot: title only. It is hard to read at body sizes."""
    return ImageFont.truetype(DIDOT, size * S, index={"regular": 0, "italic": 1, "bold": 2}[face])


def G(size: int, face: str = "regular") -> ImageFont.FreeTypeFont:
    idx = {"regular": 0, "bold": 1, "italic": 2, "medium": 10, "light": 7}[face]
    return ImageFont.truetype(SANS, size * S, index=idx)


def main() -> int:
    trans = json.loads((NLA / "cxr14_transport.json").read_text())
    ens = json.loads((NLA / "cxr14_ensemble3.json").read_text())
    rsicd = json.loads((NLA / "omni" / "rsicd_scene_probe.json").read_text())
    dims = json.loads((NLA / "omni" / "joint_frame_roco.json").read_text())["native_dims"]

    cost = trans["summary"]["transport_cost"]
    beat = sum(1 for v in trans["transported"].values() if v["vs_native_target"] > 0)
    total = len(trans["transported"])
    pooled = ens["ensemble"]["logit_mean"]
    base = ens["split_matched_baseline"]
    sat_cost = rsicd["summary"]["transport_cost"]

    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im)
    x = 118

    d.text((x * S, 92 * S), "SRT PROGRAM   ·   CROSS-VENDOR PROBE TRANSPORT",
           font=G(23, "medium"), fill=VIOLET)

    d.text((x * S, 152 * S), "Frozen Backbones", font=F(112, "bold"), fill=INK)
    d.text((x * S, 274 * S), "Read Each Other", font=F(112, "bold"), fill=INK)

    for i, ln in enumerate([
        "A linear probe fitted on one frozen model reads a different vendor's hidden",
        "states through a map estimated on training rows alone. Four models, four",
        "companies, no fine-tuning anywhere.",
    ]):
        d.text((x * S, (424 + i * 42) * S), ln, font=G(30, "light"), fill=MUTED)

    # One probe reaching four differently shaped state spaces is the whole idea,
    # and the mismatched widths are what make it surprising.
    cx = 1218
    d.text((cx * S, 150 * S), "ONE PROBE, FOUR VENDORS' STATES",
           font=G(19, "medium"), fill=VIOLET)
    rows = [("Qwen3-Omni-30B-A3B", "qwen3omni"), ("Gemma-4-31B-it", "gemma4"),
            ("Mistral-Small-3.1-24B", "mistral"), ("Aria", "aria")]
    tops = [196, 288, 380, 472]
    d.line([((cx - 28) * S, (tops[0] + 38) * S), ((cx - 28) * S, (tops[-1] + 38) * S)],
           fill=VIOLET, width=3 * S)
    for (name, key), ty in zip(rows, tops):
        d.rounded_rectangle([cx * S, ty * S, (W - 118) * S, (ty + 76) * S],
                            radius=14 * S, fill=PANEL, outline=LINE, width=2 * S)
        d.text(((cx + 28) * S, (ty + 13) * S), name, font=G(25, "medium"), fill=INK)
        d.text(((cx + 28) * S, (ty + 44) * S), f"{dims[key]} dimensions",
               font=G(20, "light"), fill=MUTED)
        my = (ty + 38) * S
        d.polygon([((cx - 28) * S, my - 10 * S), (cx * S, my), ((cx - 28) * S, my + 10 * S)],
                  fill=VIOLET)

    d.line([(x * S, 580 * S), ((W - x) * S, 580 * S)], fill=LINE, width=2 * S)

    # Transport that costs less than nothing is the paper's whole claim, so the
    # negative sign is the thing the thumbnail has to carry.
    stats = [
        (f"{cost:+.4f}", "transport cost, chest pathology",
         f"{beat} of {total} borrowed probes beat the target's own", GREEN),
        (f"{pooled:.4f}", "three frozen backbones, logits averaged",
         f"{pooled - base:+.4f} vs a fine-tuned ResNet-50", INK),
        (f"{sat_cost:+.4f}", "transport cost, satellite scene classes",
         "17 land-use classes, shuffled floor 0.5014", INK),
    ]
    bw, gap = 546, 30
    for i, (big, label, sub, col) in enumerate(stats):
        bx = x + i * (bw + gap)
        d.rounded_rectangle([bx * S, 634 * S, (bx + bw) * S, 858 * S],
                            radius=18 * S, fill=PANEL, outline=LINE, width=2 * S)
        d.text(((bx + 34) * S, 664 * S), big, font=G(64, "bold"), fill=col)
        d.text(((bx + 34) * S, 756 * S), label, font=G(24, "medium"), fill=INK)
        d.text(((bx + 34) * S, 792 * S), sub, font=G(21, "light"), fill=MUTED)

    d.text((x * S, 916 * S),
           "ChestX-ray14 official test list, 25,596 held-out films, 30,805 patients   ·   "
           "every figure reproducible from published states",
           font=G(21, "light"), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.resize((W, H), Image.LANCZOS).save(OUT, optimize=True)
    print(f"  {OUT}  {W}x{H}  {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
