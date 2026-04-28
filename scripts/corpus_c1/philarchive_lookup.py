"""PhilArchive OA harvester for C1.

PhilArchive (https://philarchive.org) hosts open-access philosophy
preprints. We use its OAI-PMH endpoint to enumerate records, filter to
items with a publicly downloadable PDF, and run them through the standard
build pipeline by emitting a manifest YAML the existing url_pdf extractor
can consume.

Usage:
    python scripts/corpus_c1/philarchive_lookup.py --max-records 8000 \
        > data/corpus_c1/manifests/philarchive_oa.yaml
    python scripts/corpus_c1/build.py --manifest data/corpus_c1/manifests/philarchive_oa.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

OAI = "https://philarchive.org/oai.pl"
USER_AGENT = (
    "srt-adapter-c1-corpus/1.0 (https://github.com/space-bacon/SRT; "
    "research corpus build) Python-urllib/3.12"
)
DELAY_S = 2.0
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

# Topical filter: only keep records whose dc:subject or dc:title contains
# at least one of these terms (case-insensitive).
KEYWORD_RE = re.compile(
    r"\b(semiot|sign|pragmat|meaning|interpret|hermeneut|"
    r"discours|languag|peirce|saussure|wittgenstein|frege|"
    r"speech ?act|reference|inferenti|metaphor|narrat|"
    r"phenomenolog|consciousness|content|representation|"
    r"intent|communicat|relativ|holism|composition|"
    r"propositi|rhetoric|cogniti|symbol|epistem)\b",
    re.I,
)


def _http(url: str, retries: int = 5) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503):
                time.sleep(8 * (attempt + 1))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up: {url}: {last!r}")


def _slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:maxlen] or "philarchive"


def parse_records(xml_bytes: bytes) -> tuple[list[dict], str | None]:
    root = ET.fromstring(xml_bytes)
    out: list[dict] = []
    for rec in root.findall(".//oai:record", NS):
        header = rec.find("oai:header", NS)
        if header is None:
            continue
        if header.attrib.get("status") == "deleted":
            continue
        meta = rec.find(".//oai_dc:dc", NS)
        if meta is None:
            continue
        identifier = (header.findtext("oai:identifier", default="", namespaces=NS) or "").strip()
        title = " ".join((meta.findtext("dc:title", default="", namespaces=NS) or "").split())
        creators = [c.text or "" for c in meta.findall("dc:creator", NS) if (c.text or "").strip()]
        subjects = [s.text or "" for s in meta.findall("dc:subject", NS) if (s.text or "").strip()]
        date = meta.findtext("dc:date", default="", namespaces=NS) or ""
        m_year = re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b", date)
        year = m_year.group(1) if m_year else ""
        # dc:identifier may be repeated; PhilArchive includes a PDF URL.
        identifiers = [i.text or "" for i in meta.findall("dc:identifier", NS) if (i.text or "").strip()]
        pdf_url = ""
        page_url = ""
        for ident in identifiers:
            if ident.endswith(".pdf"):
                pdf_url = ident
            elif "philarchive.org" in ident and not page_url:
                page_url = ident
        if not pdf_url:
            # try to construct one from page url like https://philarchive.org/rec/SLUG
            for ident in identifiers:
                m = re.match(r"https?://philarchive\.org/rec/([A-Z0-9]+)", ident)
                if m:
                    pdf_url = f"https://philarchive.org/archive/{m.group(1)}.pdf"
                    page_url = ident
                    break
        if not pdf_url:
            continue
        rights = (meta.findtext("dc:rights", default="", namespaces=NS) or "").lower()
        # OAI feed exposes only OA records but be defensive
        if rights and "no-derivatives" in rights and "noncommercial" in rights:
            # Still permitted for research text-reuse generally; keep
            pass
        out.append(
            {
                "oai_id": identifier,
                "title": title,
                "creators": creators,
                "subjects": subjects,
                "year": year,
                "pdf_url": pdf_url,
                "page_url": page_url or pdf_url,
            }
        )
    token_el = root.find(".//oai:resumptionToken", NS)
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return out, token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=8000)
    ap.add_argument(
        "--metadata-prefix",
        default="oai_dc",
        help="OAI-PMH metadataPrefix",
    )
    args = ap.parse_args()

    seen_pdf: set[str] = set()
    kept: list[dict] = []
    n_seen = 0

    url = (
        f"{OAI}?verb=ListRecords&metadataPrefix={urllib.parse.quote(args.metadata_prefix)}"
    )
    page = 0
    while True:
        page += 1
        try:
            xml = _http(url)
        except Exception as e:  # noqa: BLE001
            print(f"# WARN page={page}: {e}", file=sys.stderr)
            break
        recs, token = parse_records(xml)
        n_seen += len(recs)
        new = 0
        for r in recs:
            if r["pdf_url"] in seen_pdf:
                continue
            blob = " ".join([r["title"], " ".join(r["subjects"])])
            if not KEYWORD_RE.search(blob):
                continue
            seen_pdf.add(r["pdf_url"])
            kept.append(r)
            new += 1
            if len(kept) >= args.max_records:
                break
        print(
            f"# page={page} batch={len(recs)} new_kept={new} total_kept={len(kept)} total_seen={n_seen}",
            file=sys.stderr,
        )
        if len(kept) >= args.max_records or not token:
            break
        url = f"{OAI}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
        time.sleep(DELAY_S)

    # Emit manifest YAML
    print("# PhilArchive open-access records, filtered by semiotics-adjacent keyword regex.")
    print(f"# Harvested via scripts/corpus_c1/philarchive_lookup.py")
    print(f"# Total kept: {len(kept)} (scanned ~{n_seen} OAI records)")
    print()
    print("manifest_version: 1")
    print('school_default: "philarchive-oa"')
    print("tier_default: 1")
    print("weight_default: 1.2")
    print('license_default: "PhilArchive-OA"')
    print()
    print("sources:")
    used_ids: set[str] = set()
    for r in kept:
        slug = _slugify(r["title"]) or _slugify(r["oai_id"])
        sid = f"phil_{slug}"[:90]
        i = 1
        base_sid = sid
        while sid in used_ids:
            i += 1
            sid = f"{base_sid}_{i}"[:90]
        used_ids.add(sid)
        title_safe = (r["title"] or "").replace("\\", "\\\\").replace('"', "'")
        print()
        print(f"  - id: {sid}")
        print(f"    type: url_pdf")
        print(f'    url: "{r["pdf_url"]}"')
        print(f'    venue: "PhilArchive"')
        print(f"    year: {r['year'] or 'null'}")
        print(f'    title: "{title_safe}"')
        if r["creators"]:
            print(f"    authors:")
            for c in r["creators"][:6]:
                c_safe = c.replace("\\", "\\\\").replace('"', "'")
                print(f'      - "{c_safe}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
