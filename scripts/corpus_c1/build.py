"""C1 corpus builder.

Reads YAML manifests under data/corpus_c1/manifests/ and produces normalized
JSONL training samples under data/corpus_c1/jsonl/<manifest>.jsonl.

Usage:
    python scripts/corpus_c1/build.py --manifest data/corpus_c1/manifests/lancaster.yaml
    python scripts/corpus_c1/build.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "data" / "corpus_c1"
MANIFEST_DIR = CORPUS_ROOT / "manifests"
EXTRACTED_DIR = CORPUS_ROOT / "extracted"
JSONL_DIR = CORPUS_ROOT / "jsonl"
CACHE_DIR = CORPUS_ROOT / "cache"
REJECTED_PATH = CORPUS_ROOT / "rejected.jsonl"

USER_AGENT = (
    "srt-adapter-c1-corpus/1.0 "
    "(research; contact: github.com/space-bacon/SRT)"
)
FETCH_DELAY_S = 1.5  # polite default between network requests

# Chunk parameters: target ~600-1200 whitespace tokens per chunk, hard min 80,
# hard max 2000. Chunks split on paragraph breaks; oversized paragraphs split
# on sentence boundaries.
CHUNK_TARGET = 900
CHUNK_MIN = 80
CHUNK_MAX = 2000


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_local_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_local_pdf(path: Path) -> str:
    # Use pdftotext (poppler) for layout-aware extraction.
    result = subprocess.run(
        ["pdftotext", "-layout", "-nopgbrk", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_local_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Minimal tag stripping; for serious HTML use a manifest-level note to
    # convert via pandoc beforehand. C1 is mostly PDF + MD so this is rare.
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _cache_path(url: str, suffix: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{h}{suffix}"


def _http_get(url: str, suffix: str, *, retries: int = 3, expect_pdf: bool = False) -> Path:
    """Fetch URL into cache (sha1-named). Returns local path. Idempotent.

    If `expect_pdf` is True, the response must start with %PDF- or the file is
    discarded and a RuntimeError is raised (so HTML interstitials don't poison
    the cache).
    """
    cached = _cache_path(url, suffix)
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/pdf,*/*;q=0.8" if expect_pdf else "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if expect_pdf and not data[:5].startswith(b"%PDF-"):
                raise RuntimeError(
                    f"response is not a PDF (first bytes: {data[:16]!r})"
                )
            cached.write_bytes(data)
            time.sleep(FETCH_DELAY_S)
            return cached
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_err!r}")


def extract_url_pdf(path_or_url: str) -> str:
    cached = _http_get(path_or_url, ".pdf", expect_pdf=True)
    return extract_local_pdf(cached)


def extract_url_html(path_or_url: str) -> str:
    cached = _http_get(path_or_url, ".html")
    return extract_local_html(cached)


def extract_arxiv(arxiv_id: str) -> str:
    """`arxiv_id` like '2603.03248' or '2603.03248v2'."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    return extract_url_pdf(url)


def extract_europepmc(pmc_id: str) -> str:
    """`pmc_id` like 'PMC6140520'. Uses Europe PMC OA renderer (no interstitial)."""
    pmc_id = pmc_id if pmc_id.upper().startswith("PMC") else f"PMC{pmc_id}"
    url = f"https://europepmc.org/articles/{pmc_id}?pdf=render"
    return extract_url_pdf(url)


EXTRACTORS = {
    "local_md": extract_local_md,
    "local_pdf": extract_local_pdf,
    "local_html": extract_local_html,
    "url_pdf": extract_url_pdf,
    "url_html": extract_url_html,
    "arxiv": extract_arxiv,
    "europepmc": extract_europepmc,
}


# ---------------------------------------------------------------------------
# Normalisation + chunking
# ---------------------------------------------------------------------------


_CITE_RE = re.compile(r"^\s*\[\d+\]")  # bracket-cite list lines
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")  # bare page numbers


def normalise(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    # Collapse common PDF artifacts: hard hyphens at line breaks
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # Drop bare-page-number lines + bracket-cite lines (heuristic; safe-ish)
    lines = []
    for ln in text.splitlines():
        if _PAGE_RE.match(ln):
            continue
        lines.append(ln.rstrip())
    text = "\n".join(lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def split_sentences(para: str) -> list[str]:
    # Conservative sentence splitter; preserves abbreviations as best-effort.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])", para)
    return [p for p in parts if p.strip()]


def token_count(s: str) -> int:
    return len(s.split())


def chunk_text(text: str) -> list[str]:
    """Pack paragraphs into chunks within [CHUNK_MIN, CHUNK_MAX] tokens.

    Oversized paragraphs are split on sentence boundaries. Tiny trailing
    fragments are merged into the previous chunk.
    """
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for para in split_paragraphs(text):
        ptok = token_count(para)
        if ptok > CHUNK_MAX:
            # flush current
            if buf:
                out.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            # split paragraph by sentence
            sub: list[str] = []
            sub_tok = 0
            for sent in split_sentences(para):
                stok = token_count(sent)
                if sub_tok + stok > CHUNK_TARGET and sub_tok >= CHUNK_MIN:
                    out.append(" ".join(sub))
                    sub, sub_tok = [], 0
                sub.append(sent)
                sub_tok += stok
            if sub:
                out.append(" ".join(sub))
            continue
        if buf_tokens + ptok > CHUNK_TARGET and buf_tokens >= CHUNK_MIN:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += ptok
    if buf:
        chunk = "\n\n".join(buf)
        # Merge tiny tail into previous
        if out and token_count(chunk) < CHUNK_MIN:
            out[-1] = out[-1] + "\n\n" + chunk
        else:
            out.append(chunk)
    # Final pass: drop chunks below CHUNK_MIN if no neighbour to merge with
    out = [c for c in out if token_count(c) >= CHUNK_MIN]
    return out


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest processing
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def process_source(src: dict, defaults: dict, seen_chunks: set[str]) -> Iterable[dict]:
    src_type = src["type"]
    if src_type not in EXTRACTORS:
        raise NotImplementedError(f"extractor for type={src_type!r} not implemented")
    if src_type.startswith("local_"):
        src_path = Path(src["path"])
        if not src_path.exists():
            print(f"  SKIP missing: {src_path}", file=sys.stderr)
            return
        raw = EXTRACTORS[src_type](src_path)
    elif src_type == "arxiv":
        try:
            raw = EXTRACTORS[src_type](src["arxiv_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP fetch-fail {src['id']}: {e}", file=sys.stderr)
            return
    elif src_type == "europepmc":
        try:
            raw = EXTRACTORS[src_type](src["pmc_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP fetch-fail {src['id']}: {e}", file=sys.stderr)
            return
    else:  # url_pdf, url_html
        try:
            raw = EXTRACTORS[src_type](src["url"])
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP fetch-fail {src['id']}: {e}", file=sys.stderr)
            return
    text = normalise(raw)
    if not text:
        print(f"  SKIP empty after normalise: {src['id']}", file=sys.stderr)
        return
    doc_sha = sha1_hex(text)
    # Stash extracted text for inspection
    out_extracted = EXTRACTED_DIR / f"{src['id']}.txt"
    out_extracted.write_text(text, encoding="utf-8")
    chunks = chunk_text(text)
    n_emit = 0
    for i, chunk in enumerate(chunks):
        chunk_sha = sha1_hex(chunk)
        if chunk_sha in seen_chunks:
            continue
        seen_chunks.add(chunk_sha)
        row = {
            "text": chunk,
            "source_id": src["id"],
            "source_url": src.get("url") or src.get("path"),
            "title": src.get("title"),
            "venue": src.get("venue"),
            "author": src.get("author") or defaults["author"],
            "school": src.get("school") or defaults["school"],
            "tier": src.get("tier") or defaults["tier"],
            "year": src.get("year"),
            "license": src.get("license") or defaults["license"],
            "weight": src.get("weight") or defaults["weight"],
            "doc_sha1": doc_sha,
            "chunk_idx": i,
            "chunk_sha1": chunk_sha,
            "n_tokens_ws": token_count(chunk),
        }
        n_emit += 1
        yield row
    print(
        f"  {src['id']:40s} chunks={n_emit:4d} doc_sha={doc_sha[:8]} "
        f"src_tokens~={token_count(text)}",
        file=sys.stderr,
    )


def process_manifest(manifest_path: Path) -> tuple[int, int]:
    manifest = load_manifest(manifest_path)
    defaults = {
        "author": manifest.get("author_default", []),
        "school": manifest.get("school_default", "unknown"),
        "tier": manifest.get("tier_default", 2),
        "license": manifest.get("license_default", "unknown"),
        "weight": manifest.get("weight_default", 1.0),
    }
    sources = manifest.get("sources", [])
    name = manifest_path.stem
    out_path = JSONL_DIR / f"{name}.jsonl"
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_rows = 0
    n_tokens = 0
    print(f"=== {name} ({len(sources)} sources) ===", file=sys.stderr)
    with out_path.open("w") as f:
        for src in sources:
            for row in process_source(src, defaults, seen):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_tokens += row["n_tokens_ws"]
    print(
        f"  -> {out_path}: rows={n_rows} approx_tokens={n_tokens:,}",
        file=sys.stderr,
    )
    return n_rows, n_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, help="single manifest YAML")
    ap.add_argument("--all", action="store_true", help="process every manifest")
    args = ap.parse_args()

    if not args.manifest and not args.all:
        ap.error("specify --manifest <path> or --all")

    manifests = (
        sorted(MANIFEST_DIR.glob("*.yaml")) if args.all else [args.manifest]
    )
    total_rows = 0
    total_tokens = 0
    for m in manifests:
        r, t = process_manifest(m)
        total_rows += r
        total_tokens += t
    print(
        f"\nTOTAL: {len(manifests)} manifests, rows={total_rows}, "
        f"approx_tokens={total_tokens:,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
