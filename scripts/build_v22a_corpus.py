#!/usr/bin/env python3
"""Build v22a corpus = v21a corpus with STSB rows upsampled 4x.

v21a lesson: teacher cosines beat NLI categorical labels, but English
graded benchmarks (HUMESTSBenchmark, STSBenchmark, STS13/14) still
regressed because the 200K NLI premise/hypothesis pairs (re-scored by
mxbai) outweigh the 7K STSB pairs ~28:1 in batch frequency.

v22a fix: keep teacher cosines on the full pool, but replicate every
STSB row N times (default 4) before shuffling. This shifts the effective
mix from ~3.5% STSB -> ~12% STSB without losing the diverse-pair-pool
cross-lingual benefit.

Usage:
    python scripts/build_v22a_corpus.py \\
        --in-train data/supervised_sts_v21a/train.jsonl \\
        --in-dev   data/supervised_sts_v21a/dev.jsonl \\
        --output-dir data/supervised_sts_v22a \\
        --stsb-upsample 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", required=True)
    p.add_argument("--in-dev", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stsb-upsample", type=int, default=4,
                   help="Replicate STSB rows this many times in train")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # train: upsample STSB
    rows = _read_jsonl(Path(args.in_train))
    stsb = [r for r in rows if "stsb" in r.get("source", "").lower()]
    other = [r for r in rows if "stsb" not in r.get("source", "").lower()]
    print(f"[v22a] train input: {len(rows)} rows ({len(stsb)} stsb, {len(other)} other)")
    upsampled = stsb * args.stsb_upsample
    combined = other + upsampled
    rng.shuffle(combined)
    _write_jsonl(out_dir / "train.jsonl", combined)
    pct_stsb = 100.0 * len(upsampled) / max(1, len(combined))
    print(f"[v22a] train output: {len(combined)} rows ({len(upsampled)} stsb x{args.stsb_upsample} = {pct_stsb:.1f}%)")

    # dev: passthrough
    dev = _read_jsonl(Path(args.in_dev))
    _write_jsonl(out_dir / "dev.jsonl", dev)
    print(f"[v22a] dev: {len(dev)} rows (passthrough)")


if __name__ == "__main__":
    main()
