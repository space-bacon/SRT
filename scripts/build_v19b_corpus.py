#!/usr/bin/env python3
"""Build v19b supervised STS corpus: STSB + NLI-as-pairs.

v18 saturated the 5K-pair STSB corpus. The dev-set (1500 pairs) couldn't
discriminate further training -- but MTEB showed real gains at v18's
hyperparameters. Hypothesis: more graded pair data, varied scores per
batch, will unlock further gains under the same recipe.

Schema (matches train_supervised_sts_ddp.py):
    {"sentence1": str, "sentence2": str, "score": float in [0,1], "source": str}

Sources (NO MTEB-eval contamination):
  - sentence-transformers/stsb (train+dev)            ~7,249 graded pairs
    Provides: graded supervision in the in-distribution domain.
  - sentence-transformers/all-nli (triplet config)    cap N triplets ->
                                                       2N pairs (1.0 + 0.0)
    Provides: large-scale binary rank signal across diverse domains.
    {anchor, positive} -> score 1.0
    {anchor, negative} -> score 0.0
    Within a CoSENT batch this gives strong rank signal: any pos-neg pair
    in the same batch contributes a margin gradient.

Excluded (would contaminate MTEB STS eval):
  - stsb_multi_mt        -> STSBenchmarkMultilingualSTS
  - sickr-sts            -> SICK-R
  - any sts12-22 variant -> STS12-22 tasks

Dev set: STSB validation only (matches v17/v18 for direct comparability).

Usage:
    python scripts/build_v19b_corpus.py \\
        --output-dir data/supervised_sts_v19b \\
        --nli-triplets 100000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def _stream_stsb(split: str):
    ds = load_dataset("sentence-transformers/stsb", split=split)
    for row in ds:
        s1 = (row.get("sentence1") or "").strip()
        s2 = (row.get("sentence2") or "").strip()
        sc = row.get("score")
        if not s1 or not s2 or sc is None:
            continue
        yield {
            "sentence1": s1,
            "sentence2": s2,
            "score": max(0.0, min(1.0, float(sc))),
            "source": f"stsb-{split}",
        }


def _stream_nli_triplets(limit: int):
    """all-nli triplet -> 2 pairs per row (positive=1.0, negative=0.0)."""
    ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train",
                      streaming=True)
    n = 0
    for row in ds:
        a = (row.get("anchor") or "").strip()
        p = (row.get("positive") or "").strip()
        ng = (row.get("negative") or "").strip()
        if not a or not p or not ng:
            continue
        yield {"sentence1": a, "sentence2": p, "score": 1.0,
               "source": "allnli-pos"}
        yield {"sentence1": a, "sentence2": ng, "score": 0.0,
               "source": "allnli-neg"}
        n += 1
        if n >= limit:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--nli-triplets", type=int, default=100000,
                    help="Cap on all-nli triplet rows (each yields 2 pairs).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []

    n0 = len(train_rows)
    train_rows.extend(_stream_stsb("train"))
    print(f"  + stsb train: +{len(train_rows)-n0} rows")

    stsb_dev = list(_stream_stsb("validation"))
    dev_rows.extend(stsb_dev)
    print(f"  dev pool stsb-dev: {len(stsb_dev)} rows")
    n0 = len(train_rows)
    train_rows.extend(stsb_dev)
    print(f"  + stsb dev (also in train): +{len(train_rows)-n0} rows")

    n0 = len(train_rows)
    train_rows.extend(_stream_nli_triplets(args.nli_triplets))
    print(f"  + all-nli triplets ({args.nli_triplets} rows -> "
          f"{len(train_rows)-n0} pairs)")

    rng.shuffle(train_rows)

    train_path = out / "train.jsonl"
    dev_path = out / "dev.jsonl"
    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with dev_path.open("w") as f:
        for r in dev_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {train_path}: {len(train_rows)} rows")
    print(f"wrote {dev_path}: {len(dev_rows)} rows")
    if train_rows:
        scores = [r["score"] for r in train_rows]
        print(f"  train score range: [{min(scores):.3f}, {max(scores):.3f}], "
              f"mean={sum(scores)/len(scores):.3f}")
        from collections import Counter
        src = Counter(r["source"] for r in train_rows)
        for k, v in src.most_common():
            print(f"  source {k}: {v} rows")


if __name__ == "__main__":
    main()
