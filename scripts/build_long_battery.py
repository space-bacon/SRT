#!/usr/bin/env python3
"""Build a length-stratified battery from probe_battery_v1.jsonl by
concatenating N=1,3,5,8 v1 prompts within each regime.

Yields {n_chunks=1: 11*5, n_chunks=3: 11*5, ...} prompts so we can ask:
how does the BOS-sink amplitude scale with prompt length?
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/probes/probe_battery_v1.jsonl")
    ap.add_argument("--out", default="data/probes/probe_battery_long.jsonl")
    ap.add_argument("--per-bin", type=int, default=5,
                    help="number of synthetic prompts per (regime, n_chunks) cell")
    ap.add_argument("--n-chunks", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ln in Path(args.src).read_text().splitlines():
        if not ln.strip():
            continue
        item = json.loads(ln)
        by_label[item["label"]].append(item)

    out_rows = []
    for n in args.n_chunks:
        for lbl, items in sorted(by_label.items()):
            for k in range(args.per_bin):
                # sample n prompts from this regime (with replacement only if needed)
                if n <= len(items):
                    chunks = rng.sample(items, n)
                else:
                    chunks = [rng.choice(items) for _ in range(n)]
                text = " ".join(c["text"].strip() for c in chunks)
                out_rows.append({
                    "id": f"{lbl}__n{n}__{k:02d}",
                    "label": lbl,
                    "n_chunks": n,
                    "text": text,
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} prompts -> {args.out}")
    print(f"  bins: {args.n_chunks} x {args.per_bin} prompts/regime x {len(by_label)} regimes")


if __name__ == "__main__":
    main()
