#!/usr/bin/env python3
"""v11: stratified train/heldout split of archetype-conditioned generations.

Reads artifacts/archetype_probe/generations.jsonl (986 rows × 33 classes,
~30 rows/class) and writes two disjoint JSONLs partitioned per archetype_id
so neither side has a class missing.

Outputs:
    --train-out: rows used for training mix (--archetype-data on train.py)
    --val-out:   rows used for val mix (--val-archetype-data on train.py,
                 NEW in v11) and for separatrix/eval probes that should be
                 disjoint from training data.

Default split fraction: 0.25 to val (≈7 rows/class), 0.75 to train.
Deterministic given --seed.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="artifacts/archetype_probe/generations.jsonl",
        help="Source generations JSONL.",
    )
    p.add_argument(
        "--train-out",
        default="data/archetype_probe/generations_train_v1.jsonl",
        help="Destination for training partition.",
    )
    p.add_argument(
        "--val-out",
        default="data/archetype_probe/generations_heldout_v1.jsonl",
        help="Destination for held-out partition (used for val mix).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of each class routed to val.",
    )
    p.add_argument("--seed", type=int, default=137)
    args = p.parse_args()

    src = Path(args.input)
    rows = [json.loads(l) for l in src.open()]
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        aid = int(r.get("archetype_id", -1) or -1)
        if aid < 0:
            raise ValueError(f"Row missing archetype_id: {r}")
        by_class[aid].append(r)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for aid, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * args.val_frac)))
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])

    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with val_path.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")

    print(f"source:    {src} ({len(rows)} rows, {len(by_class)} classes)")
    print(f"train_out: {train_path} ({len(train_rows)} rows)")
    print(f"val_out:   {val_path} ({len(val_rows)} rows)")
    print(f"per-class val rows: min={min(len(g[: max(1, int(round(len(g) * args.val_frac)))]) for g in by_class.values())}"
          f" max={max(len(g[: max(1, int(round(len(g) * args.val_frac)))]) for g in by_class.values())}")


if __name__ == "__main__":
    main()
