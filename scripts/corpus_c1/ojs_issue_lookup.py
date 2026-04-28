"""OJS issue → manifest YAML helper.

Given an OJS (PKP) issue URL like:
  https://ojs.utlib.ee/index.php/sss/issue/view/48.1
enumerate every article in the issue and emit YAML entries (one per article)
ready to paste into a corpus_c1 manifest.

Each article in OJS has:
  /article/view/<articleId>           -- abstract page (title in anchor text)
  /article/view/<articleId>/<galleyId> -- PDF galley download

The article anchor appears immediately followed by a "PDF" anchor for its
galley. We pair them by order on the page.

Usage:
    python scripts/corpus_c1/ojs_issue_lookup.py <issue-url> --id-prefix sss48_1
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "srt-adapter-c1-corpus/1.0 (+https://github.com/space-bacon/SRT; research)"
ARTICLE_VIEW_RE = re.compile(r"/article/view/([^/]+)(?:/(\d+))?/?$")
DOWNLOAD_RE = re.compile(r"/article/download/[^\"' ]+")


def resolve_pdf(viewer_url: str) -> str | None:
    """OJS galley viewer pages are HTML; the actual PDF lives at
    /article/download/<id>/<galley>/<file>. Fetch the viewer and pull
    the first /article/download/ URL out of the markup.
    """
    try:
        req = urllib.request.Request(viewer_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    m = DOWNLOAD_RE.search(body)
    if not m:
        return None
    # The viewer page often emits a host-absolute /article/download/... path
    # that drops the OJS sub-path (e.g. /index.php/sss). urljoin against the
    # viewer URL would lose that, so splice the viewer's path-prefix back in.
    rel = m.group(0)
    parsed = urllib.parse.urlparse(viewer_url)
    prefix = parsed.path.split("/article/")[0]  # e.g. /index.php/sss
    if prefix and not rel.startswith(prefix):
        rel = prefix + rel
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, rel, "", "", ""))


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._h: str | None = None
        self._t: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._h = dict(attrs).get("href")
            self._t = []

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._h:
            self.links.append((self._h, " ".join("".join(self._t).split())))
            self._h, self._t = None, []

    def handle_data(self, data):
        if self._h is not None:
            self._t.append(data)


def slugify(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:maxlen].rstrip("_") or "untitled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--id-prefix", required=True, help="e.g. sss48_1")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--venue", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    req = urllib.request.Request(args.url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    p = LinkExtractor()
    p.feed(html)

    # Walk links pairing (article-page, next PDF galley with same article id)
    items: list[dict] = []
    pending_title: str | None = None
    pending_id: str | None = None
    for href, text in p.links:
        absurl = urllib.parse.urljoin(args.url, href)
        m = ARTICLE_VIEW_RE.search(urllib.parse.urlparse(absurl).path)
        if not m:
            continue
        art_id, galley_id = m.group(1), m.group(2)
        if galley_id is None:
            # abstract page; capture title
            if text and len(text) > 3:
                pending_title = text
                pending_id = art_id
        else:
            # galley viewer link; resolve to actual /article/download/ PDF URL
            if pending_id == art_id and pending_title:
                pdf = resolve_pdf(absurl)
                if pdf is None:
                    print(f"# WARN no PDF download link for {art_id}", file=sys.stderr)
                else:
                    items.append({
                        "title": pending_title,
                        "pdf_url": pdf,
                        "art_id": art_id,
                        "galley_id": galley_id,
                    })
                pending_title, pending_id = None, None

    if args.limit:
        items = items[: args.limit]

    print(f"# {len(items)} articles from {args.url}", file=sys.stderr)
    print(f"# id-prefix: {args.id_prefix}", file=sys.stderr)
    print()
    for i, it in enumerate(items, 1):
        slug = slugify(it["title"])
        sid = f"{args.id_prefix}_{i:02d}_{slug}"
        print(f"  - id: {sid}")
        print(f"    type: url_pdf")
        print(f"    url: \"{it['pdf_url']}\"")
        if args.year:
            print(f"    year: {args.year}")
        if args.venue:
            print(f"    venue: \"{args.venue}\"")
        print(f"    title: \"{it['title'].replace(chr(34), chr(39))}\"")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
