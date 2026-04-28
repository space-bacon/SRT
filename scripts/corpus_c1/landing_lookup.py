"""Landing-page PDF discovery.

Fetches an HTML page (an author's publications page, lab page, etc.) and
prints all candidate PDF links with their anchor text. Use to discover
self-archived PDFs for manifest entries.

Usage:
    python scripts/corpus_c1/landing_lookup.py http://www.bruno-latour.fr/node/484.html
    python scripts/corpus_c1/landing_lookup.py --pdf-only <url>
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = (
    "srt-adapter-c1-corpus/1.0 "
    "(+https://github.com/space-bacon/SRT; research)"
)


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []  # (href, text)
        self._cur_href: str | None = None
        self._cur_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            d = dict(attrs)
            self._cur_href = d.get("href")
            self._cur_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._cur_href:
            text = " ".join("".join(self._cur_text).split())
            self.links.append((self._cur_href, text))
            self._cur_href = None
            self._cur_text = []

    def handle_data(self, data: str) -> None:
        if self._cur_href is not None:
            self._cur_text.append(data)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    # Detect charset crudely
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--pdf-only", action="store_true", help="show only .pdf links")
    ap.add_argument("--filter", default=None, help="regex anchor-text filter")
    args = ap.parse_args()

    try:
        html = fetch(args.url)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR fetching: {e}", file=sys.stderr)
        return 2

    p = LinkExtractor()
    p.feed(html)

    rx = re.compile(args.filter, re.I) if args.filter else None
    seen: set[str] = set()
    n = 0
    for href, text in p.links:
        absurl = urllib.parse.urljoin(args.url, href)
        if absurl in seen:
            continue
        seen.add(absurl)
        is_pdf = absurl.lower().endswith(".pdf")
        if args.pdf_only and not is_pdf:
            continue
        if rx and not rx.search(text or ""):
            continue
        marker = "PDF" if is_pdf else "   "
        print(f"  [{marker}] {text[:80]:80s}  {absurl}")
        n += 1
    print(f"\n{n} link(s) shown ({len(seen)} total on page)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
