#!/usr/bin/env python3
"""Get a markdown paper into a WYSIWYG editor with its figures intact.

Relative image links cannot survive a paste: the editor has no repo to resolve
them against. Two editors want two different answers.

Substack takes rich HTML, so inline every image as a base64 data URI and paste
the rendered page. That is the default.

    python scripts/md_to_paste_html.py paper_crossvendor.md
    -> docs/paper_crossvendor_paste.html

Hugging Face articles strip data URIs on paste but do accept markdown, so there
the figures must be hosted and linked absolutely. --image-base rewrites every
local image to <base>/<filename> and emits markdown instead.

    python scripts/md_to_paste_html.py paper_crossvendor.md \\
      --image-base https://huggingface.co/datasets/<repo>/resolve/main/figs \\
      --md-out docs/paper_crossvendor_hf.md
"""
from __future__ import annotations

import argparse
import base64
import html as htmlmod
import io
import re
import sys
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from PIL import Image, ImageDraw, ImageFont

LANCZOS = Image.Resampling.LANCZOS

SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
TS = 2  # table supersample
T_PAD, T_PADV, T_LH, T_MAXCOL, T_FS = 14, 9, 21, 430, 15
T_HEAD, T_BORDER, T_INK = "#f3f1fb", "#d9d5ea", "#1c1a47"

ROOT = Path(__file__).resolve().parents[1]
MAX_W = 1400  # a 700px column at 2x; beyond this is bytes nobody sees

CSS = """
body{margin:0;background:#fff;color:#1c1a47;
 font:17px/1.62 "Helvetica Neue",Helvetica,Arial,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:48px 22px 96px}
h1{font-size:34px;line-height:1.16;letter-spacing:-.01em;margin:.2em 0 .1em}
h2{font-size:24px;margin:1.9em 0 .5em;letter-spacing:-.01em}
h3{font-size:19px;margin:1.5em 0 .35em}
p{margin:0 0 1.05em}
a{color:#7a5fd0}
code{font:0.88em ui-monospace,SFMono-Regular,Menlo,monospace;
 background:#f3f1fb;padding:1px 5px;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:22px 0;font-size:15.5px}
th,td{border:1px solid #d9d5ea;padding:7px 10px;text-align:left}
th{background:#f3f1fb;font-weight:600}
td[align=right],th[align=right]{text-align:right}
figure{margin:30px 0}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:14px;color:#6a6688;margin-top:9px;line-height:1.5}
blockquote{border-left:3px solid #a48dff;margin:22px 0;padding:2px 0 2px 18px;color:#3a3660}
hr{border:0;border-top:1px solid #e2dff0;margin:34px 0}
"""


def data_uri(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    mime = path.suffix.lstrip(".").lower().replace("jpg", "jpeg")
    if mime == "png":
        im = Image.open(io.BytesIO(raw))
        if im.width > MAX_W:
            small = im.resize((MAX_W, round(im.height * MAX_W / im.width)), LANCZOS)
            buf = io.BytesIO()
            small.save(buf, "PNG", optimize=True)
            # Resampling a flat line plot adds antialiasing gradients that cost
            # more to compress than the pixels saved, so only keep a real win.
            if buf.tell() < len(raw):
                raw = buf.getvalue()
    return f"data:image/{mime};base64," + base64.b64encode(raw).decode(), len(raw)


IMG = re.compile(r'<img\s+src="([^"]+)"\s+alt="([^"]*)"\s*/?>')


def inline(html: str, base: Path) -> str:
    """Swap every local <img> for a data-URI <figure>, alt text becoming caption."""
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        src, alt = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        # Paper paths are written relative to the repo root, not to the paper.
        for cand in (base / src, ROOT / src):
            if cand.is_file():
                uri, nbytes = data_uri(cand)
                print(f"  {src:52s} {nbytes / 1024:6.0f} KB", flush=True)
                cap = f"<figcaption>{alt}</figcaption>" if alt else ""
                return f'<figure><img alt="{alt}" src="{uri}">{cap}</figure>'
        missing.append(src)
        return m.group(0)

    out = IMG.sub(sub, html)
    if missing:
        print("\n!! could not resolve, left as-is:", file=sys.stderr)
        for s in missing:
            print(f"   {s}", file=sys.stderr)
    return out


MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def rewrite_md(text: str, base: str) -> str:
    """Point every local image at a hosted copy, matched on filename alone.

    The image is also collapsed onto one line. The paper hard wraps, so alt text
    routinely spans three lines, and a line-oriented parser (which is what a
    WYSIWYG paste handler usually is) leaves that as a broken link.
    """
    base = base.rstrip("/")
    seen: list[tuple[str, str]] = []

    def sub(m: re.Match[str]) -> str:
        alt, src = " ".join(m.group(1).split()), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return f"![{alt}]({src})"
        url = f"{base}/{Path(src).name}"
        seen.append((src, url))
        return f"![{alt}]({url})"

    out = MD_IMG.sub(sub, text)
    for s, u in seen:
        print(f"  {s}\n    -> {u}")
    if not seen:
        print("  no local images found", file=sys.stderr)
    return out


TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL_RE = re.compile(r"<(th|td)([^>]*)>(.*?)</\1>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def _tfont(bold: bool) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(SANS, T_FS * TS, index=1 if bold else 0)


def table_to_image(block: str) -> Image.Image:
    """Draw one markdown-rendered table as a picture.

    Substack drops <table> on paste, taking the column structure with it. A
    picture is the only form that survives, so alignment and emphasis have to be
    reproduced here rather than left to CSS. Emphasis is tracked per word: cells
    in the negative-results table open with a bold verdict and continue in
    regular weight, and a per-cell flag renders that whole column bold.
    """
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def tw(s: str, bold: bool) -> float:
        return probe.textlength(s, font=_tfont(bold)) / TS

    def tokens(inner: str, force: bool) -> list[tuple[str, bool]]:
        out = []
        for part in re.split(r"(<strong>.*?</strong>)", inner, flags=re.S):
            if part:
                bold = force or part.startswith("<strong>")
                out += [(w, bold) for w in htmlmod.unescape(TAG_RE.sub("", part)).split()]
        return out

    def line_w(line: list[tuple[str, bool]]) -> float:
        return sum(tw(w, b) + (tw(" ", b) if i else 0) for i, (w, b) in enumerate(line))

    def wrap(toks: list[tuple[str, bool]]) -> list[list[tuple[str, bool]]]:
        lines: list[list[tuple[str, bool]]] = []
        cur: list[tuple[str, bool]] = []
        for word, bold in toks:
            if cur and line_w(cur + [(word, bold)]) > T_MAXCOL:
                lines.append(cur)
                cur = [(word, bold)]
            else:
                cur.append((word, bold))
        lines.append(cur)
        return lines

    rows: list[list[dict[str, Any]]] = []
    for r in ROW_RE.findall(block):
        cells = [{"lines": wrap(tokens(inner, tag == "th")),
                  "right": "text-align:right" in attrs.replace(" ", ""),
                  "head": tag == "th"}
                 for tag, attrs, inner in CELL_RE.findall(r)]
        if cells:
            rows.append(cells)

    ncol = max(len(r) for r in rows)
    blank: dict[str, Any] = {"lines": [[]], "right": False, "head": False}
    grid = [[(r[j] if j < len(r) else blank) for j in range(ncol)] for r in rows]

    colw = [round(max(max(line_w(ln) for ln in row[j]["lines"]) for row in grid)) + 2 * T_PAD
            for j in range(ncol)]
    rowh = [max(len(c["lines"]) for c in row) * T_LH + 2 * T_PADV for row in grid]

    w, h = sum(colw) + 2, sum(rowh) + 2
    im = Image.new("RGB", (w * TS, h * TS), "#ffffff")
    d = ImageDraw.Draw(im)
    y = 1
    for row, rh in zip(grid, rowh):
        x = 1
        for c, cw in zip(row, colw):
            box = [x * TS, y * TS, (x + cw) * TS, (y + rh) * TS]
            if c["head"]:
                d.rectangle(box, fill=T_HEAD)
            d.rectangle(box, outline=T_BORDER, width=TS)
            ty = y + T_PADV
            for line in c["lines"]:
                tx = x + cw - T_PAD - line_w(line) if c["right"] else x + T_PAD
                for i, (word, bold) in enumerate(line):
                    if i:
                        tx += tw(" ", bold)
                    d.text((tx * TS, ty * TS), word, font=_tfont(bold), fill=T_INK)
                    tx += tw(word, bold)
                ty += T_LH
            x += cw
        y += rh
    return im.resize((w, h), LANCZOS)


def image_uri(im: Image.Image) -> tuple[str, int]:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), buf.tell()


def insert_hero(html: str, path: Path, after: int) -> str:
    """Drop the OG image in after the Nth closing </p>."""
    idx = -1
    for _ in range(after):
        idx = html.find("</p>", idx + 1)
        if idx < 0:
            raise SystemExit(f"!! only {after - 1} paragraphs, cannot place hero after {after}")
    uri, nbytes = data_uri(path)
    print(f"  hero {path.name:47s} {nbytes / 1024:6.0f} KB  after paragraph {after}")
    return f'{html[:idx + 4]}<figure><img alt="" src="{uri}"></figure>{html[idx + 4:]}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("markdown")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image-base", help="host prefix for figures; switches output to markdown")
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--hero", default=None, help="image to inline near the top")
    ap.add_argument("--hero-after", type=int, default=1, help="paragraph to place it after")
    ap.add_argument("--tables-as-images", action="store_true",
                    help="render each table to a picture; Substack strips <table>")
    args = ap.parse_args()

    src = Path(args.markdown).resolve()

    if args.image_base:
        md_out = Path(args.md_out).resolve() if args.md_out else ROOT / "docs" / f"{src.stem}_hosted.md"
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(rewrite_md(src.read_text(), args.image_base))
        print(f"\n  {md_out}")
        return 0

    out = Path(args.out).resolve() if args.out else ROOT / "docs" / f"{src.stem}_paste.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    body = inline(md.render(src.read_text()), src.parent)
    if args.tables_as_images:
        n = [0]

        def as_img(m: re.Match[str]) -> str:
            uri, nbytes = image_uri(table_to_image(m.group(0)))
            n[0] += 1
            print(f"  table {n[0]:>2d}{'':44s} {nbytes / 1024:6.0f} KB")
            return f'<figure><img alt="table" src="{uri}"></figure>'

        body = TABLE_RE.sub(as_img, body)
    if args.hero:
        body = insert_hero(body, Path(args.hero).resolve(), args.hero_after)

    title = re.search(r"^#\s+(.+)$", src.read_text(), re.M)
    out.write_text(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title.group(1) if title else src.stem}</title>"
        f"<style>{CSS}</style></head><body><div class=wrap>{body}</div></body></html>"
    )
    kb = out.stat().st_size / 1024
    print(f"\n  {out}  {kb:.0f} KB")
    if kb > 4000:
        print("  large: some editors truncate a very big clipboard payload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
