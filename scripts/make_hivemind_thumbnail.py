#!/usr/bin/env python3
"""Article thumbnail for the hivemind paper, 1910x1000.

Numbers are read from the banked artifacts, same rule as the figure generator, so
a stale thumbnail cannot outlive the result it quotes.

The card has to carry one claim, and it is the ratio: the deployment format does
several times the work that instruction tuning does. The persona null sits beside
it because it is the explanation everyone reaches for first, and it is worth less
than nothing.

    python scripts/make_hivemind_thumbnail.py
    -> arxiv_hivemind/figs/thumbnail.png
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NLA = ROOT / "artifacts" / "nla"
OUT = ROOT / "arxiv_hivemind" / "figs" / "thumbnail.png"

BG = "#131233"
PANEL = "#1c1a47"
VIOLET = "#a48dff"
INK = "#eae7fb"
MUTED = "#a7a1cc"
LINE = "#332e66"
GREEN = "#7fd6a4"
RED = "#e08585"

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
    iso = json.loads((NLA / "atlas" / "hivemind_posttraining_isolation.json").read_text())
    dec = json.loads((NLA / "atlas" / "hivemind_template_decomp.json").read_text())
    atlas = json.loads((NLA / "atlas" / "openweight_transport_atlas.json").read_text())

    tuning = iso["delta"]["instruct_raw"]["intra"]
    fmt = iso["delta"]["instruct_chat"]["intra"]
    chat_intra = iso["instruct_chat"]["intra_mean"]
    ratio = fmt / tuning

    gains = dec["intra_gain_over_raw"]
    arms = dec["arms"]
    share = dec["decomposition"]["share_reachable_without_native_tokens"]

    cross = atlas["summary"]["mean_cross_lab"]
    floor = atlas["summary"]["mean_shuffled_floor"]
    n_pairs = len(iso["pairs"])
    n_pair_labs = len({p["lab"] for p in iso["pairs"]})
    n_models = len(atlas["models"])
    # One entry is tagged "Alibaba/fp32", a precision control rather than a ninth
    # vendor, so the vendor count takes the part before the slash.
    n_labs = len({m["lab"].split("/")[0] for m in atlas["models"].values()})

    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im)
    x = 118

    d.text((x * S, 92 * S),
           "SRT PROGRAM   \u00b7   GEOMETRY, TUNING AND FORMAT, SEPARATED ON OPEN WEIGHTS",
           font=G(23, "medium"), fill=VIOLET)

    for i, ln in enumerate(["The Hivemind Is Not", "in the Weights.", "It Is the Turn."]):
        d.text((x * S, (136 + i * 88) * S), ln, font=F(80, "bold"), fill=INK)

    for i, ln in enumerate([
        "Language models write alike. With open weights the cause separates: shared",
        "geometry does not do it, instruction tuning barely does it, and the chat",
        f"template does most of it. {n_pairs} matched base/instruct pairs, {n_pair_labs} labs.",
    ]):
        d.text((x * S, (444 + i * 40) * S), ln, font=G(29, "light"), fill=MUTED)

    # The decomposition ladder is the argument: each rung adds one thing.
    cx = 1218
    d.text((cx * S, 150 * S), "WHAT EACH FRAMING BUYS",
           font=G(19, "medium"), fill=VIOLET)
    rows = [
        ("the bare stem", arms["raw"]["intra_mean"], None, MUTED),
        ("+ \u201chelpful assistant\u201d", arms["persona"]["intra_mean"], gains["persona"], RED),
        ("+ role markers", arms["shared"]["intra_mean"], gains["shared"], VIOLET),
        ("+ its own template", arms["chat"]["intra_mean"], gains["chat"], GREEN),
    ]
    tops = [196, 288, 380, 472]
    d.line([((cx - 28) * S, (tops[0] + 38) * S), ((cx - 28) * S, (tops[-1] + 38) * S)],
           fill=VIOLET, width=3 * S)
    for (label, intra, gain, col), ty in zip(rows, tops):
        d.rounded_rectangle([cx * S, ty * S, (W - 118) * S, (ty + 76) * S],
                            radius=14 * S, fill=PANEL, outline=LINE, width=2 * S)
        d.text(((cx + 28) * S, (ty + 13) * S), label, font=G(25, "medium"), fill=INK)
        sub = f"intra {intra:.4f}" + (f"      {gain:+.4f}" if gain is not None else "")
        d.text(((cx + 28) * S, (ty + 44) * S), sub, font=G(20, "light"), fill=col)
        my = (ty + 38) * S
        d.polygon([((cx - 28) * S, my - 10 * S), (cx * S, my), ((cx - 28) * S, my + 10 * S)],
                  fill=VIOLET)

    d.line([(x * S, 580 * S), ((W - x) * S, 580 * S)], fill=LINE, width=2 * S)

    # The three cards are the three headline claims, in the order the paper makes
    # them: we recover the reported level, the format is what recovers it, and most
    # of that is a convention rather than anything in the weights.
    # Share of the climb from our base arm to their stated 0.8 line. Both endpoints
    # are ours and the target is their threshold, so no floor assumption enters.
    climb = fmt / (0.80 - iso["base"]["intra_mean"])

    # Floor-corrected recovery of their level. They report the floor as a band and
    # inter-model as a band, so this is a span rather than a point, and shipping the
    # span is the only honest form of it.
    their_floors = (0.10, 0.20)
    their_levels = (("intra", 0.80, chat_intra),
                    ("inter", 0.71, iso["instruct_chat"]["inter_mean"]),
                    ("inter", 0.82, iso["instruct_chat"]["inter_mean"]))
    ours_floor = iso["instruct_chat"]["floor"]
    rec = [(ours - ours_floor) / (theirs - tf)
           for tf in their_floors for _, theirs, ours in their_levels]

    stats = [
        (f"{climb * 100:.0f}%", "of the way from base models to their 0.8 line",
         f"{min(rec) * 100:.0f}% to {max(rec) * 100:.0f}% of their level, "
         "floor-corrected", GREEN),
        (f"{ratio:.1f}\u00d7", "the format outweighs the tuning",
         f"+{fmt:.4f} through the template against +{tuning:.4f} tuned", INK),
        (f"{share * 100:.0f}%", "recovered by markers nobody trained on",
         f"the persona sentence itself is worth {gains['persona']:+.4f}", VIOLET),
    ]
    bw, gap = 546, 30
    for i, (big, label, sub, col) in enumerate(stats):
        bx = x + i * (bw + gap)
        d.rounded_rectangle([bx * S, 634 * S, (bx + bw) * S, 858 * S],
                            radius=18 * S, fill=PANEL, outline=LINE, width=2 * S)
        d.text(((bx + 34) * S, 664 * S), big, font=G(64, "bold"), fill=col)
        d.text(((bx + 34) * S, 756 * S), label, font=G(23, "medium"), fill=INK)
        d.text(((bx + 34) * S, 792 * S), sub, font=G(20, "light"), fill=MUTED)

    d.text((x * S, 916 * S),
           f"{n_models} open-weight models, {n_labs} labs   \u00b7   states transport across "
           f"labs at {cross:.4f} against a shuffled floor of {floor:.5f}   \u00b7   "
           "every figure reproducible from published artifacts",
           font=G(21, "light"), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    im.resize((W, H), Image.LANCZOS).save(OUT, optimize=True)
    print(f"  {OUT}  {W}x{H}  {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
