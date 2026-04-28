# C1 — Computational-semiotics author-targeted corpus

Naturally-labeled scholarly corpus for SRT-Adapter v9 training.

## Structure

```
data/corpus_c1/
  manifests/    one YAML per author or source group; defines what to pull
  raw/          downloaded originals (PDFs, HTML, MD), sha1-named, license-tagged sidecar
  extracted/    plain-text after extraction (one .txt per raw, sha1-aligned)
  jsonl/        normalized samples ready for training (one .jsonl per manifest)
  cache/        per-source HTTP cache + ETags; safe to delete and re-run
```

## Schema (per JSONL row)

```json
{
  "text": "<paragraph or section, 200-2000 tokens>",
  "source_id": "lancaster_treachery_2025",
  "source_url": "https://papers.ssrn.com/abstract=5987495",
  "author": ["Lancaster, James Burton"],
  "school": "srt",
  "tier": 1,
  "year": 2025,
  "license": "self-archived-preprint",
  "weight": 5.0,
  "doc_sha1": "<sha1 of full source>",
  "chunk_idx": 7,
  "chunk_sha1": "<sha1 of this chunk's normalized text>"
}
```

`weight` is the curriculum-sampling multiplier in v9 training (Lancaster=5.0,
Tier 1 OA cited authors=2.0, Tier 2 broader semiotics=1.0).

## Build

```bash
python scripts/corpus_c1/build.py --manifest data/corpus_c1/manifests/<name>.yaml
python scripts/corpus_c1/build.py --all   # iterate every manifest
```

Reproducible: re-running uses the cache and skips already-fetched URLs by sha1.

## License policy

Every source carries a `license` tag. Hard rules:

- **Reject** any source without a verifiable OA / PD / permissive license.
- **Reject** personal correspondence by default (override per correspondent only).
- **Reject** paywalled monographs unless flagged `excerpt_fair_use: true` with
  a per-source justification and ≤10% of total work.

Sources that fail license check are written to `data/corpus_c1/rejected.jsonl`
with reason, never to the trainable JSONL.
