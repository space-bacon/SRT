"""arXiv bulk harvester for cs.CL/cs.AI/cs.LG, filtered by semiotic vocabulary.

Queries the arXiv API for each (category × vocab-term) pair, dedups by
arxiv_id, and emits a manifest YAML to stdout that the build pipeline can
fetch as PDFs via the existing 'arxiv' extractor.

Usage:
    python scripts/corpus_c1/arxiv_bulk_lookup.py > data/corpus_c1/manifests/arxiv_semiotic_vocab.yaml

Tunables: --max-per-query (default 200), --categories, --terms.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "srt-adapter-c1-corpus/1.0 (research; contact: github.com/space-bacon/SRT)"
DELAY_S = 3.5  # arXiv asks for >=3s between API calls

DEFAULT_CATEGORIES = ("cs.CL", "cs.AI", "cs.LG")

# Vocabulary terms targeting the v9 computational-semiotics curriculum.
# Quoted = exact phrase match in arXiv's abstract index.
DEFAULT_TERMS = (
    '"semiotic"',
    '"semiotics"',
    '"metapragmatic"',
    '"interpretant"',
    '"discourse community"',
    '"metalinguistic"',
    '"sign system"',
    '"speech act"',
    '"pragmatic competence"',
    '"hermeneutic"',
    '"second-order cybernetic"',
    '"reflexivity"',
    '"polysemy"',
    '"indexicality"',
    '"performativity"',
    '"language game"',
    '"meaning negotiation"',
    '"intersubjectiv"',
    '"frame semantics"',
    '"presupposition"',
    '"deixis"',
)


def _slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen]


def _http_get(url: str, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt + DELAY_S)
    raise RuntimeError(f"failed: {url}: {last!r}")


def search(category: str, term: str, max_per_query: int) -> list[dict]:
    """One paged query against arXiv API. Returns at most `max_per_query` hits."""
    out: list[dict] = []
    page = 100  # arXiv recommends <=200 per request
    start = 0
    while start < max_per_query:
        size = min(page, max_per_query - start)
        # Search abstract+title for term, restricted to category.
        # cat:cs.CL AND (abs:"semiotic" OR ti:"semiotic")
        q = f'cat:{category} AND (abs:{term} OR ti:{term})'
        url = (
            f"{API}?search_query={urllib.parse.quote(q)}"
            f"&start={start}&max_results={size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        xml = _http_get(url)
        root = ET.fromstring(xml)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for entry in entries:
            eid = entry.findtext("a:id", default="", namespaces=NS) or ""
            arxiv_id = eid.rsplit("/", 1)[-1]
            # strip version suffix to dedupe
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            title = " ".join(
                (entry.findtext("a:title", default="", namespaces=NS) or "").split()
            )
            published = entry.findtext("a:published", default="", namespaces=NS) or ""
            year = published[:4] if published else ""
            primary_cat_el = entry.find("arxiv:primary_category", NS)
            primary_cat = (
                primary_cat_el.attrib.get("term", "") if primary_cat_el is not None else ""
            )
            authors = [
                (a.findtext("a:name", default="", namespaces=NS) or "")
                for a in entry.findall("a:author", NS)
            ]
            out.append(
                {
                    "arxiv_id": base_id,
                    "version_id": arxiv_id,
                    "year": year,
                    "title": title,
                    "primary_cat": primary_cat,
                    "authors": authors,
                    "matched_term": term,
                    "matched_cat": category,
                }
            )
        start += len(entries)
        if len(entries) < size:
            break
        time.sleep(DELAY_S)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="arXiv categories to search",
    )
    ap.add_argument(
        "--terms",
        nargs="+",
        default=list(DEFAULT_TERMS),
        help="vocabulary terms (quoted phrases recommended)",
    )
    ap.add_argument(
        "--max-per-query",
        type=int,
        default=200,
        help="max results per (category × term) pair",
    )
    ap.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="drop entries published before this year",
    )
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    n_queries = len(args.categories) * len(args.terms)
    done = 0
    for cat in args.categories:
        for term in args.terms:
            done += 1
            try:
                hits = search(cat, term, args.max_per_query)
            except Exception as e:  # noqa: BLE001
                print(
                    f"# WARN [{done}/{n_queries}] cat={cat} term={term}: {e}",
                    file=sys.stderr,
                )
                time.sleep(DELAY_S)
                continue
            new = 0
            for h in hits:
                if args.min_year and h["year"] and int(h["year"]) < args.min_year:
                    continue
                if h["arxiv_id"] in seen:
                    continue
                seen[h["arxiv_id"]] = h
                new += 1
            print(
                f"# [{done}/{n_queries}] cat={cat} term={term} hits={len(hits)} new={new} total={len(seen)}",
                file=sys.stderr,
            )
            time.sleep(DELAY_S)

    entries = sorted(seen.values(), key=lambda h: (h.get("year", ""), h["arxiv_id"]))

    print("# arXiv cs.CL/cs.AI/cs.LG, filtered by semiotic-vocab regex.")
    print(f"# Harvested via scripts/corpus_c1/arxiv_bulk_lookup.py")
    print(f"# Categories: {', '.join(args.categories)}")
    print(f"# Terms: {len(args.terms)} vocab phrases")
    print(f"# Total deduped entries: {len(entries)} (min-year={args.min_year})")
    print()
    print("manifest_version: 1")
    print('school_default: "computational-semiotics-arxiv"')
    print("tier_default: 1")
    print("weight_default: 1.3")
    print('license_default: "arXiv-non-exclusive"')
    print()
    print("sources:")
    used_ids: set[str] = set()
    for h in entries:
        slug = _slugify(h["title"]) or h["arxiv_id"].replace(".", "_")
        sid = f"arxiv_{h['arxiv_id'].replace('.', '_')}_{slug}"[:90]
        if sid in used_ids:
            sid = f"{sid}_{h['arxiv_id'].replace('.', '_')}"
        used_ids.add(sid)
        title_safe = h["title"].replace('"', "'").replace("\\", "")
        print()
        print(f"  - id: {sid}")
        print(f"    type: arxiv")
        print(f'    arxiv_id: "{h["arxiv_id"]}"')
        print(f'    venue: "arXiv {h["primary_cat"]}"')
        print(f"    year: {h['year'] or 'null'}")
        print(f'    title: "{title_safe}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
