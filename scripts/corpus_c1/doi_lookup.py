"""DOI -> OA PDF resolver (Crossref + Unpaywall).

Usage:
    python scripts/corpus_c1/doi_lookup.py 10.1073/pnas.1804840115
    python scripts/corpus_c1/doi_lookup.py --batch dois.txt

For each DOI, prints metadata (title, year, authors, license) and the best OA
PDF URL Unpaywall knows about. Use this to populate a manifest with
`type: doi` entries; the build pipeline then fetches+verifies each one.

Set env CONTACT_EMAIL to a real address for Unpaywall (default below).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UNPAYWALL_EMAIL = os.environ.get("CONTACT_EMAIL", "research@srt-adapter.example")
UA = "srt-adapter-c1-corpus/1.0 (mailto:%s)" % UNPAYWALL_EMAIL


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def crossref(doi: str) -> dict:
    return _get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}")


def unpaywall(doi: str) -> dict:
    return _get_json(
        f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='')}?email={UNPAYWALL_EMAIL}"
    )


def resolve(doi: str) -> dict:
    out: dict = {"doi": doi}
    try:
        cr = crossref(doi)["message"]
        out["title"] = " ".join(cr.get("title", [""])[0].split())
        out["year"] = (
            (cr.get("issued", {}).get("date-parts") or [[None]])[0][0]
        )
        out["authors"] = [
            f"{a.get('family','')}, {a.get('given','')}".strip(", ")
            for a in cr.get("author", [])[:6]
        ]
        out["container"] = (cr.get("container-title") or [""])[0]
        # Crossref license block (may be empty)
        lic = cr.get("license") or []
        out["crossref_license"] = [l.get("URL") for l in lic if l.get("URL")]
    except Exception as e:  # noqa: BLE001
        out["crossref_error"] = repr(e)
    try:
        up = unpaywall(doi)
        out["is_oa"] = up.get("is_oa")
        out["oa_status"] = up.get("oa_status")
        loc = up.get("best_oa_location") or {}
        out["pdf_url"] = loc.get("url_for_pdf") or loc.get("url")
        out["host_type"] = loc.get("host_type")
        out["unpaywall_license"] = loc.get("license")
    except Exception as e:  # noqa: BLE001
        out["unpaywall_error"] = repr(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", nargs="?")
    ap.add_argument("--batch", help="file with one DOI per line (# = comment)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    dois: list[str] = []
    if args.batch:
        for line in open(args.batch):
            line = line.strip()
            if line and not line.startswith("#"):
                dois.append(line.split()[0])
    elif args.doi:
        dois.append(args.doi)
    else:
        ap.error("provide a DOI or --batch")

    results = []
    for i, d in enumerate(dois):
        if i > 0:
            time.sleep(1.0)
        r = resolve(d)
        results.append(r)
        if args.json:
            continue
        print(f"\n=== {d} ===")
        print(f"  title:   {r.get('title')}")
        print(f"  year:    {r.get('year')}  venue: {r.get('container')}")
        print(f"  authors: {', '.join(r.get('authors', [])[:4])}")
        print(f"  is_oa:   {r.get('is_oa')}  status={r.get('oa_status')}  license={r.get('unpaywall_license')}")
        print(f"  pdf_url: {r.get('pdf_url')}")
        if r.get("crossref_error"):
            print(f"  crossref_error: {r['crossref_error']}", file=sys.stderr)
        if r.get("unpaywall_error"):
            print(f"  unpaywall_error: {r['unpaywall_error']}", file=sys.stderr)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
