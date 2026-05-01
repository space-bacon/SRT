#!/usr/bin/env python3
"""Build supervised STS training data from labeled pair datasets.

Produces JSONL with rows: {"sentence1": str, "sentence2": str, "score": float in [0, 1]}.

Source: sentence-transformers/stsb -- the canonical STS Benchmark splits with
scores already normalized to [0, 1]:
  - train (5749 pairs): training pool
  - validation (1500 pairs): training pool + dev pool (selection on dev Spearman)
  - test (1379 pairs): NOT used (held out -- this is what MTEB STSBenchmark evaluates)

Why dev is in both pools: dev is small (1500) and adds useful supervision.
Best-checkpoint criterion is Spearman *rank* correlation, which generalizes
from rank-preserving fits -- this is the standard sentence-transformers recipe.
The MTEB target is `test`, which we never touch in training.

Datasets explicitly excluded to avoid contaminating MTEB STS eval:
  - stsb_multi_mt: contaminates STSBenchmarkMultilingualSTS (eval uses dev+test)
  - sickr-sts: only `test` split available on the Hub (legacy script not loadable)
  - biosses, sts12-22, sts17, sem_rel24, hume*, indic-crosslingual: eval uses test

Usage:
    python scripts/build_supervised_sts_data.py \
        --output-dir data/supervised_sts_v17

Outputs:
    {output-dir}/train.jsonl  (stsb train + stsb dev, shuffled)
    {output-dir}/dev.jsonl    (stsb dev only -- best-checkpoint criterion)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DS_ID = "sentence-transformers/stsb"


def _stream(split: str):
    ds = load_dataset(DS_ID, split=split)
    for row in ds:
        s1 = (row.get("sentence1") or "").strip()
        s2 = (row.get("sentence2") or "").strip()
        sc = row.get("score")
        if not s1 or not s2 or sc is None:
            continue
        # sentence-transformers/stsb scores are already in [0, 1]; clamp defensively.
        yield {
            "sentence1": s1,
            "sentence2": s2,
            "score": max(0.0, min(1.0, float(sc))),
            "source": f"stsb-{split}",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dev-in-train", action="store_true",
                    help="Exclude STSB dev from train.jsonl (strict no-leak).")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []

    n0 = len(train_rows)
    train_rows.extend(_stream("train"))
    print(f"  + stsb train: +{len(train_rows)-n0} rows")

    stsb_dev = list(_stream("validation"))
    dev_rows.extend(stsb_dev)
    print(f"  dev pool stsb-dev: {len(stsb_dev)} rows")
    if not args.no_dev_in_train:
        n0 = len(train_rows)
        train_rows.extend(stsb_dev)
        print(f"  + stsb dev (also in train): +{len(train_rows)-n0} rows")

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


if __name__ == "__main__":
    main()
