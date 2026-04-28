"""Project Gutenberg PD harvester for C1.

Fetches a curated whitelist of public-domain works relevant to the C1
spec (philosophy of language, semiotics, logic, hermeneutics) and runs
the standard chunker. Uses the Project Gutenberg "Plain Text UTF-8" or
"Plain Text" mirror URLs (https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt).

Usage:
    python scripts/corpus_c1/gutenberg_lookup.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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

USER_AGENT = "srt-adapter-c1-corpus/1.0 (research; contact: github.com/space-bacon/SRT)"
DELAY_S = 2.0  # PG asks scrapers to be slow

# Curated PD whitelist. Each entry: gutenberg_id, slug, title, author, year.
# Focus: philosophy of language / signs / logic / mind / rhetoric / hermeneutics.
WORKS = [
    # Wittgenstein (PD in US)
    (5740,  "wittgenstein_tractatus", "Tractatus Logico-Philosophicus", "Ludwig Wittgenstein", 1922),
    # Bertrand Russell
    (2529,  "russell_problems_of_philosophy", "The Problems of Philosophy", "Bertrand Russell", 1912),
    (5827,  "russell_analysis_of_mind", "The Analysis of Mind", "Bertrand Russell", 1921),
    (25447, "russell_mysticism_logic", "Mysticism and Logic and Other Essays", "Bertrand Russell", 1918),
    (44932, "russell_intro_math_phil", "Introduction to Mathematical Philosophy", "Bertrand Russell", 1919),
    (52821, "russell_our_knowledge_external_world", "Our Knowledge of the External World", "Bertrand Russell", 1914),
    # William James (semiotics-adjacent: pragmatism, meaning, mind)
    (5116,  "james_pragmatism", "Pragmatism: A New Name for Some Old Ways of Thinking", "William James", 1907),
    (32547, "james_meaning_of_truth", "The Meaning of Truth", "William James", 1909),
    (11984, "james_principles_psychology_v1", "The Principles of Psychology, Volume 1", "William James", 1890),
    # John Dewey (pragmatist semiotics)
    (852,   "dewey_democracy_education", "Democracy and Education", "John Dewey", 1916),
    (40794, "dewey_how_we_think", "How We Think", "John Dewey", 1910),
    # Charles Sanders Peirce — chance, logic, scientific method (essays in PD)
    (27580, "peirce_chance_love_logic", "Chance, Love, and Logic: Philosophical Essays", "Charles S. Peirce", 1923),
    # Aristotle: foundational logic + rhetoric + poetics
    (6762,  "aristotle_organon",          "The Categories", "Aristotle", -350),
    (1974,  "aristotle_rhetoric",         "Rhetoric", "Aristotle", -350),
    (1974,  "aristotle_poetics",          "Poetics", "Aristotle", -350),
    # Plato (sign-bearing dialogues on meaning)
    (1643,  "plato_cratylus",             "Cratylus", "Plato", -360),
    (1726,  "plato_sophist",              "Sophist", "Plato", -360),
    (1735,  "plato_theaetetus",           "Theaetetus", "Plato", -369),
    # Locke / Berkeley / Hume (signs, ideas, language)
    (10615, "locke_essay",                "An Essay Concerning Human Understanding, Vol 1", "John Locke", 1689),
    (4723,  "berkeley_treatise",          "A Treatise Concerning the Principles of Human Knowledge", "George Berkeley", 1710),
    (4705,  "hume_enquiry",               "An Enquiry Concerning Human Understanding", "David Hume", 1748),
    (9662,  "hume_treatise",              "A Treatise of Human Nature", "David Hume", 1739),
    # Mill — System of Logic (signs, naming, denotation)
    (27942, "mill_system_of_logic",       "A System of Logic", "John Stuart Mill", 1843),
    # Frege (Foundations of Arithmetic — German PG; English ed not free, but sense/reference essays appear in collections; skip if missing)
    # F.H. Bradley
    (43479, "bradley_appearance_reality", "Appearance and Reality", "F.H. Bradley", 1893),
    # Henri Bergson (semiotic of consciousness)
    (28278, "bergson_creative_evolution", "Creative Evolution", "Henri Bergson", 1907),
    (54076, "bergson_intro_metaphysics",  "An Introduction to Metaphysics", "Henri Bergson", 1903),
    # G.E. Moore
    (53430, "moore_principia_ethica",     "Principia Ethica", "G.E. Moore", 1903),
    # Ernst Mach
    (54550, "mach_analysis_sensations",   "Contributions to the Analysis of the Sensations", "Ernst Mach", 1897),
    # Boole (laws of thought)
    (15114, "boole_laws_of_thought",      "An Investigation of the Laws of Thought", "George Boole", 1854),
    # De Morgan
    (39088, "demorgan_formal_logic",      "Formal Logic", "Augustus De Morgan", 1847),
    # John Venn
    (45911, "venn_principles_logic",      "The Principles of Empirical or Inductive Logic", "John Venn", 1889),
    # Royce, James Mark Baldwin (semiotic-adjacent psychology of signs)
    (38381, "royce_world_individual",     "The World and the Individual, First Series", "Josiah Royce", 1899),
    # Susanne Langer not PD. Cassirer not PD.
    # Rhetoric
    (16654, "campbell_philosophy_rhetoric", "The Philosophy of Rhetoric", "George Campbell", 1776),
    # Linguistic foundations (Müller, Whitney)
    (33836, "muller_lectures_language_v1", "Lectures on the Science of Language", "Max Müller", 1864),
    (32484, "whitney_life_growth_language", "The Life and Growth of Language", "William Dwight Whitney", 1875),
    # James Sully — outlines of psychology (semiotic-adjacent)
    (15466, "sully_outlines_psychology", "Outlines of Psychology", "James Sully", 1884),
    # Dewey — Logic essays
    (40794, "dewey_essays_in_experimental_logic", "Essays in Experimental Logic", "John Dewey", 1916),
]

# Strip Project Gutenberg headers / footers
PG_START_RE = re.compile(r"\*\*\* ?START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", re.I)
PG_END_RE   = re.compile(r"\*\*\* ?END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", re.I)


def fetch_pg(gid: int) -> str:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]
    last: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise RuntimeError(f"all PG mirror URLs failed for {gid}: {last!r}")


def strip_pg(text: str) -> str:
    m1 = PG_START_RE.search(text)
    if m1:
        text = text[m1.end():]
    m2 = PG_END_RE.search(text)
    if m2:
        text = text[: m2.start()]
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-name", default="gutenberg_pd")
    args = ap.parse_args()

    out_path = JSONL_DIR / f"{args.output_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_doc_sha: set[str] = set()
    seen_chunks: set[str] = set()
    n_works = 0
    n_tokens = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for gid, slug, title, author, year in WORKS:
            try:
                raw = fetch_pg(gid)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP fetch-fail {slug} (PG#{gid}): {e}", file=sys.stderr)
                time.sleep(DELAY_S)
                continue
            text = normalise(strip_pg(raw))
            if token_count(text) < CHUNK_MIN:
                print(f"  SKIP empty after strip: {slug}", file=sys.stderr)
                continue
            doc_sha = sha1_hex(text)
            if doc_sha in seen_doc_sha:
                print(f"  SKIP duplicate doc: {slug}", file=sys.stderr)
                time.sleep(DELAY_S)
                continue
            seen_doc_sha.add(doc_sha)
            chunks = chunk_text(text)
            emitted = 0
            for ci, chunk in enumerate(chunks):
                csha = sha1_hex(chunk)
                if csha in seen_chunks:
                    continue
                seen_chunks.add(csha)
                ntok = token_count(chunk)
                row = {
                    "text": chunk,
                    "source_id": f"pg_{slug}",
                    "source_url": f"https://www.gutenberg.org/ebooks/{gid}",
                    "title": title,
                    "venue": "Project Gutenberg",
                    "author": [author],
                    "school": "philosophy-pd-canon",
                    "tier": 1,
                    "year": year if year and year > 0 else None,
                    "license": "Public Domain (US)",
                    "weight": 1.3,
                    "doc_sha1": doc_sha,
                    "chunk_idx": ci,
                    "chunk_sha1": csha,
                    "n_tokens_ws": ntok,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_tokens += ntok
                emitted += 1
            print(f"  {slug:<40s} chunks={emitted:5d} doc_sha={doc_sha[:8]} src_tokens~={token_count(text):,}", file=sys.stderr)
            n_works += 1
            time.sleep(DELAY_S)

    print(
        f"DONE: works={n_works} chunks={len(seen_chunks)} approx_tokens={n_tokens:,} → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
