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
import io
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image

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
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=True)
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
