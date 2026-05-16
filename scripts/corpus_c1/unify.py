"""Unify C1 jsonl shards into a single shuffled train/val split.

Reads every *.jsonl under data/corpus_c1/jsonl/ (skipping empty files),
normalizes each row to the SRTAdapterDataset schema (`text` + a populated
`community_label` so the dataset's hash-fallback yields a stable id),
deduplicates by `chunk_sha1`, shuffles deterministically (seed=137), and
writes:

    data/corpus_c1/train.jsonl
    data/corpus_c1/val.jsonl

Usage:
    python scripts/corpus_c1/unify.py --val-size 5000 --min-tokens 64
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL_DIR = REPO_ROOT / "data" / "corpus_c1" / "jsonl"
OUT_DIR = REPO_ROOT / "data" / "corpus_c1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--min-tokens", type=int, default=64,
                    help="drop rows with fewer than this many whitespace tokens")
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--out-train", type=Path, default=OUT_DIR / "train.jsonl")
    ap.add_argument("--out-val", type=Path, default=OUT_DIR / "val.jsonl")
    args = ap.parse_args()

    rows: list[dict] = []
    seen_chunks: set[str] = set()
    per_school: Counter[str] = Counter()
    per_file: Counter[str] = Counter()
    dropped_empty = 0
    dropped_short = 0
    dropped_dup = 0

    for path in sorted(JSONL_DIR.glob("*.jsonl")):
        if path.stat().st_size == 0:
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    dropped_empty += 1
                    continue
                if len(text.split()) < args.min_tokens:
                    dropped_short += 1
                    continue
                chunk_sha1 = row.get("chunk_sha1")
                if chunk_sha1 and chunk_sha1 in seen_chunks:
                    dropped_dup += 1
                    continue
                if chunk_sha1:
                    seen_chunks.add(chunk_sha1)

                school = row.get("school") or row.get("source_id") or path.stem
                # SRTAdapterDataset hashes `community_label` (first non-empty
                # of community_label/source/community); set it explicitly so
                # every C1 row maps to a stable per-school community id.
                row["community_label"] = school
                per_school[school] += 1
                per_file[path.name] += 1
                rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val = rows[: args.val_size]
    train = rows[args.val_size :]

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.out_val.open("w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== C1 unified ===")
    print(f"train: {len(train):>8}  ->  {args.out_train}")
    print(f"val:   {len(val):>8}  ->  {args.out_val}")
    print(f"dropped empty:  {dropped_empty}")
    print(f"dropped short:  {dropped_short} (< {args.min_tokens} ws-tokens)")
    print(f"dropped dup:    {dropped_dup} (by chunk_sha1)")
    print(f"\nPer-source-file row counts:")
    for name, n in per_file.most_common():
        print(f"  {name:40s} {n:>8}")
    print(f"\nPer-school (community_label) counts (top 30):")
    for s, n in per_school.most_common(30):
        print(f"  {s:40s} {n:>8}")
    print(f"\nDistinct schools (=> distinct community ids): {len(per_school)}")


if __name__ == "__main__":
    main()
