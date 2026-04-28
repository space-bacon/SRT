"""Resolve arXiv IDs from title + author queries via the arXiv API.

Usage:
    python scripts/corpus_c1/arxiv_lookup.py "title words" --author "Surname"
    python scripts/corpus_c1/arxiv_lookup.py --batch queries.txt

Prints top 5 matches with id, year, authors[:3], title. Hand-verify before
adding to a manifest.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def search(query: str, author: str | None = None, max_results: int = 5) -> list[dict]:
    parts = [f"all:{query}"]
    if author:
        parts.append(f'au:"{author}"')
    q = "+AND+".join(urllib.parse.quote(p) for p in parts)
    url = f"{API}?search_query={q}&start=0&max_results={max_results}&sortBy=relevance"
    req = urllib.request.Request(url, headers={"User-Agent": "srt-adapter-c1/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    out = []
    for entry in root.findall("a:entry", NS):
        eid = entry.findtext("a:id", default="", namespaces=NS)
        # id is like http://arxiv.org/abs/1302.6906v1
        arxiv_id = eid.rsplit("/", 1)[-1]
        title = " ".join(
            (entry.findtext("a:title", default="", namespaces=NS) or "").split()
        )
        published = entry.findtext("a:published", default="", namespaces=NS)
        year = published[:4] if published else ""
        authors = [
            a.findtext("a:name", default="", namespaces=NS)
            for a in entry.findall("a:author", NS)
        ]
        out.append(
            {"id": arxiv_id, "year": year, "title": title, "authors": authors}
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="title keywords")
    ap.add_argument("--author", default=None)
    ap.add_argument("--batch", help="file with 'title|author' per line")
    ap.add_argument("--max", type=int, default=5)
    args = ap.parse_args()
    queries: list[tuple[str, str | None]] = []
    if args.batch:
        for line in open(args.batch):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                t, a = line.split("|", 1)
                queries.append((t.strip(), a.strip() or None))
            else:
                queries.append((line, None))
    elif args.query:
        queries.append((args.query, args.author))
    else:
        ap.error("provide a query or --batch file")
    for i, (q, a) in enumerate(queries):
        if i > 0:
            time.sleep(3.0)  # arXiv API politeness
        hdr = f"=== {q!r} author={a!r} ==="
        print(hdr)
        try:
            for r in search(q, a, args.max):
                au = ", ".join(r["authors"][:3])
                if len(r["authors"]) > 3:
                    au += ", et al."
                print(f"  {r['id']:18s} {r['year']}  {au}")
                print(f"                     {r['title']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
