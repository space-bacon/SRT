"""SEP + IEP encyclopedia harvester.

Scrapes the Stanford Encyclopedia of Philosophy (https://plato.stanford.edu/)
and the Internet Encyclopedia of Philosophy (https://iep.utm.edu/) indexes,
filters entries by a semiotics-adjacent keyword regex, and emits a manifest
YAML to stdout that can be fed to scripts/corpus_c1/build.py.

Usage:
    python scripts/corpus_c1/sep_iep_lookup.py --site sep > data/corpus_c1/manifests/sep_semiotics.yaml
    python scripts/corpus_c1/sep_iep_lookup.py --site iep > data/corpus_c1/manifests/iep_semiotics.yaml

License notes:
- SEP: research/scholarly use is allowed; we cite the entry URL per row.
- IEP: open-access encyclopedia, scholarly reuse with attribution.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from string import ascii_lowercase

from bs4 import BeautifulSoup


USER_AGENT = (
    "srt-adapter-c1-corpus/1.0 "
    "(research; contact: github.com/space-bacon/SRT)"
)
FETCH_DELAY_S = 1.0

KEYWORDS = re.compile(
    r"(sign|semiot|semant|pragmat|meaning|interpret|hermeneut|languag|"
    r"peirce|saussure|wittgenstein|frege|davidson|quine|kripke|grice|"
    r"austin|searle|chomsky|cogniti|metaphor|reference|truth|logic|"
    r"inferenti|symbol|representation|consciousness|content|concept|"
    r"expression|relativ|deflation|holism|compositional|propositi|"
    r"speech ?act|discourse|rhetoric|narrat|hermeneut|phenomenolog|"
    r"intent|communicat|category|metaphysics of mean)",
    re.I,
)


def http_get_text(url: str, retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            time.sleep(FETCH_DELAY_S)
            return data.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}: {last!r}")


def harvest_sep() -> list[dict]:
    html = http_get_text("https://plato.stanford.edu/contents.html")
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    seen: set[str] = set()
    for a in soup.select('a[href^="entries/"]'):
        title = a.get_text(strip=True)
        href = a.get("href", "").strip()
        if not title or not href:
            continue
        if not KEYWORDS.search(title):
            continue
        slug = href.rstrip("/").split("/")[-1]
        if slug in seen:
            continue
        seen.add(slug)
        out.append(
            {
                "id": f"sep_{slug.replace('-', '_')}",
                "title": title,
                "slug": slug,
                "url": f"https://plato.stanford.edu/{href.lstrip('/')}",
            }
        )
    return out


def harvest_iep() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for letter in ascii_lowercase:
        try:
            html = http_get_text(f"https://iep.utm.edu/{letter}/")
        except RuntimeError as e:
            print(f"# WARN: /{letter}/ failed: {e}", file=sys.stderr)
            continue
        soup = BeautifulSoup(html, "html.parser")
        ec = soup.select_one(".entry-content") or soup
        for a in ec.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "").strip()
            if not title or not href:
                continue
            m = re.match(r"https?://iep\.utm\.edu/([a-z0-9-]+)/?$", href)
            if not m:
                continue
            slug = m.group(1)
            # skip known non-entry slugs
            if slug in {
                "about",
                "category",
                "contact",
                "faq",
                "search",
                "feed",
                "submissions",
                "table-of-contents",
            } or len(slug) <= 1:
                continue
            if slug in seen:
                continue
            if not KEYWORDS.search(title):
                continue
            seen.add(slug)
            out.append(
                {
                    "id": f"iep_{slug.replace('-', '_')}",
                    "title": title,
                    "slug": slug,
                    "url": f"https://iep.utm.edu/{slug}/",
                }
            )
    return out


def emit_yaml(entries: list[dict], site: str) -> None:
    if site == "sep":
        venue = "Stanford Encyclopedia of Philosophy"
        school = "encyclopedic-philosophy"
        license_str = "SEP-research-use"
        type_str = "sep"
    else:
        venue = "Internet Encyclopedia of Philosophy"
        school = "encyclopedic-philosophy"
        license_str = "IEP-research-use"
        type_str = "iep"

    print(f"# {venue}, semiotics-adjacent entries.")
    print(f"# Harvested via scripts/corpus_c1/sep_iep_lookup.py --site {site}")
    print(f"# Total entries: {len(entries)}")
    print()
    print("manifest_version: 1")
    print(f'school_default: "{school}"')
    print("tier_default: 1")
    print("weight_default: 1.2")
    print(f'license_default: "{license_str}"')
    print()
    print("sources:")
    for e in entries:
        title_safe = e["title"].replace('"', "'")
        print()
        print(f"  - id: {e['id']}")
        print(f"    type: {type_str}")
        print(f'    url: "{e["url"]}"')
        print(f'    venue: "{venue}"')
        print(f'    title: "{title_safe}"')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["sep", "iep"], required=True)
    args = ap.parse_args()

    if args.site == "sep":
        entries = harvest_sep()
    else:
        entries = harvest_iep()

    print(f"# fetched {len(entries)} entries", file=sys.stderr)
    emit_yaml(entries, args.site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
