"""Wikipedia harvester for C1.

Walks a set of seed categories (semiotics-adjacent), pulls all article
members (recursing one level into subcategories), fetches the plain text
extract via the MediaWiki REST API, then runs the standard chunker.

Usage:
    python scripts/corpus_c1/wikipedia_lookup.py
    python scripts/corpus_c1/wikipedia_lookup.py --max-articles 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import (  # noqa: E402
    CHUNK_MIN,
    JSONL_DIR,
    chunk_text,
    normalise,
    sha1_hex,
    token_count,
)

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia UA policy: identify a contact. Bot-style UA + project URL.
USER_AGENT = (
    "srt-adapter-c1-corpus/1.0 (https://github.com/space-bacon/SRT; "
    "research corpus build for an adapter on Qwen2.5-7B) Python-urllib/3.12"
)
DELAY_S = 1.2  # be very polite to avoid 429

SEED_CATEGORIES = [
    "Semiotics",
    "Semioticians",
    "Philosophy of language",
    "Pragmatics",
    "Discourse analysis",
    "Sociolinguistics",
    "Speech act theory",
    "Hermeneutics",
    "Philosophical logic",
    "Cognitive science",
    "Cybernetics",
    "Systems theory",
    "Linguistic anthropology",
    "Communication theory",
    "Philosophy of mind",
]


def _http_json(params: dict, retries: int = 5) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    url = f"{API}?{qs}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                # exponential backoff for rate limit
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}: {last!r}")


def members(category: str, depth: int = 1) -> list[str]:
    """Return article titles in `category` and (one level of) its subcategories."""
    titles: list[str] = []
    seen: set[str] = set()

    def walk(cat: str, d: int):
        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat}",
                "cmlimit": "500",
                "cmtype": "page|subcat" if d > 0 else "page",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            data = _http_json(params)
            for m in data.get("query", {}).get("categorymembers", []):
                ns = m.get("ns")
                title = m.get("title", "")
                if ns == 14 and d > 0:  # subcategory
                    sub = title.split(":", 1)[1]
                    walk(sub, d - 1)
                elif ns == 0:
                    if title not in seen:
                        seen.add(title)
                        titles.append(title)
            cont = data.get("continue")
            if not cont or "cmcontinue" not in cont:
                break
            cmcontinue = cont["cmcontinue"]
            time.sleep(DELAY_S)

    walk(category, depth)
    return titles


def extract(title: str) -> tuple[str, str] | None:
    """Returns (plain_text, page_url) or None on miss."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "redirects": "1",
    }
    data = _http_json(params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None
    text = page.get("extract", "") or ""
    if not text.strip():
        return None
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    return text, url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=SEED_CATEGORIES)
    ap.add_argument("--depth", type=int, default=1, help="subcategory recursion depth")
    ap.add_argument("--max-articles", type=int, default=15000)
    ap.add_argument("--output-name", default="wikipedia_semiotics")
    args = ap.parse_args()

    out_path = JSONL_DIR / f"{args.output_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: enumerate articles
    print(f"Enumerating titles across {len(args.categories)} categories...", file=sys.stderr)
    all_titles: list[str] = []
    seen: set[str] = set()
    for cat in args.categories:
        try:
            ts = members(cat, depth=args.depth)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN cat={cat}: {e}", file=sys.stderr)
            continue
        new = 0
        for t in ts:
            if t not in seen:
                seen.add(t)
                all_titles.append(t)
                new += 1
        print(f"  cat={cat}: total_in_cat={len(ts)} new={new} cumulative={len(all_titles)}", file=sys.stderr)
        time.sleep(DELAY_S)
        if len(all_titles) >= args.max_articles:
            break
    all_titles = all_titles[: args.max_articles]
    print(f"Fetching extracts for {len(all_titles)} articles...", file=sys.stderr)

    # Phase 2: fetch + chunk
    seen_chunks: set[str] = set()
    n_articles = 0
    n_tokens = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, title in enumerate(all_titles, 1):
            try:
                got = extract(title)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP fetch-fail {title}: {e}", file=sys.stderr)
                time.sleep(DELAY_S)
                continue
            if not got:
                continue
            raw, url = got
            text = normalise(raw)
            if token_count(text) < CHUNK_MIN:
                continue
            doc_sha = sha1_hex(text)
            chunks = chunk_text(text)
            for ci, chunk in enumerate(chunks):
                csha = sha1_hex(chunk)
                if csha in seen_chunks:
                    continue
                seen_chunks.add(csha)
                ntok = token_count(chunk)
                row = {
                    "text": chunk,
                    "source_id": f"wiki_{title.replace(' ', '_').replace('/', '_')[:80]}",
                    "source_url": url,
                    "title": title,
                    "venue": "Wikipedia",
                    "author": [],
                    "school": "encyclopedic-wikipedia",
                    "tier": 2,
                    "year": None,
                    "license": "CC-BY-SA-4.0",
                    "weight": 1.0,
                    "doc_sha1": doc_sha,
                    "chunk_idx": ci,
                    "chunk_sha1": csha,
                    "n_tokens_ws": ntok,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_tokens += ntok
            n_articles += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(all_titles)}] articles_kept={n_articles} chunks={len(seen_chunks)} ~tokens={n_tokens:,}", file=sys.stderr)
            time.sleep(DELAY_S)

    print(
        f"DONE: articles={n_articles} chunks={len(seen_chunks)} approx_tokens={n_tokens:,} → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
