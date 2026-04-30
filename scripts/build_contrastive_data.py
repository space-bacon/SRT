#!/usr/bin/env python3
"""Build a contrastive training corpus in the format expected by
``scripts/train_contrastive.py``.

Pulls the standard MTEB pre-training mix from HuggingFace and emits a
single JSONL where each row is

    {"query": str, "positive": str, "negatives": [str, ...]}

Sources (each can be toggled with --include flags):

* ``sentence-transformers/all-nli`` (triplet config) — anchor/pos/neg
  triplets from SNLI+MultiNLI. Yields ~280K rows. NLI hard negatives
  are entailment-vs-contradiction, which is the canonical recipe for
  embedding training.
* ``sentence-transformers/msmarco-hard-negatives`` — query/positive
  with mined hard negatives from a strong dense retriever. The
  retrieval workhorse on MTEB. Large (~500K queries); use
  --msmarco-limit to cap.
* ``sentence-transformers/quora-duplicates`` (triplets config) —
  paraphrase pairs with mined negatives.

Usage:
    python scripts/build_contrastive_data.py \
        --output-dir data/contrastive \
        --include nli,msmarco,quora \
        --msmarco-limit 200000 \
        --negatives-per-row 7

Writes ``train.jsonl`` and ``val.jsonl`` (1% holdout) under
``--output-dir``. Set ``HF_HUB_OFFLINE=0`` so ``datasets`` can
download.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("srt.contrastive_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--include",
        default="nli,msmarco,quora",
        help="Comma-separated subset of {nli, msmarco, quora}.",
    )
    p.add_argument(
        "--negatives-per-row",
        type=int,
        default=7,
        help="Cap on negatives per row (msmarco can have many).",
    )
    p.add_argument(
        "--msmarco-limit",
        type=int,
        default=200_000,
        help="Max number of MS MARCO rows to include.",
    )
    p.add_argument(
        "--nli-limit",
        type=int,
        default=300_000,
        help="Max number of NLI triplets to include.",
    )
    p.add_argument(
        "--quora-limit",
        type=int,
        default=100_000,
        help="Max number of Quora triplets to include.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.01,
        help="Fraction held out for validation.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _stream_nli(limit: int) -> Iterator[dict]:
    from datasets import load_dataset

    log.info("Streaming sentence-transformers/all-nli (triplet, train) limit=%d", limit)
    ds = load_dataset(
        "sentence-transformers/all-nli", "triplet", split="train", streaming=True
    )
    n = 0
    for row in ds:
        anchor = row.get("anchor")
        pos = row.get("positive")
        neg = row.get("negative")
        if not (anchor and pos and neg):
            continue
        yield {"query": anchor, "positive": pos, "negatives": [neg]}
        n += 1
        if n >= limit:
            break
    log.info("  NLI rows emitted: %d", n)


def _stream_quora(limit: int) -> Iterator[dict]:
    from datasets import load_dataset

    log.info(
        "Streaming sentence-transformers/quora-duplicates (triplet, train) limit=%d",
        limit,
    )
    ds = load_dataset(
        "sentence-transformers/quora-duplicates",
        "triplet",
        split="train",
        streaming=True,
    )
    n = 0
    for row in ds:
        anchor = row.get("anchor")
        pos = row.get("positive")
        neg = row.get("negative")
        if not (anchor and pos and neg):
            continue
        yield {"query": anchor, "positive": pos, "negatives": [neg]}
        n += 1
        if n >= limit:
            break
    log.info("  Quora rows emitted: %d", n)


def _stream_msmarco(limit: int, max_negs: int) -> Iterator[dict]:
    from datasets import load_dataset

    log.info(
        "Streaming sentence-transformers/msmarco-hard-negatives "
        "(triplet, train) limit=%d max_negs=%d",
        limit,
        max_negs,
    )
    # The "triplet-hard" config gives one hard negative per row; for
    # the full mined set we pull the "default" config and pluck the
    # "negative" column (single hard negative per row already curated).
    ds = load_dataset(
        "sentence-transformers/msmarco-hard-negatives",
        split="train",
        streaming=True,
    )
    n = 0
    for row in ds:
        q = row.get("query") or row.get("anchor")
        pos = row.get("positive")
        # Some configs name negatives differently; collect any list-typed neg field
        negs: list[str] = []
        for k, v in row.items():
            if k in {"query", "anchor", "positive"}:
                continue
            if isinstance(v, list):
                negs.extend([x for x in v if isinstance(x, str)])
            elif isinstance(v, str) and "negative" in k:
                negs.append(v)
        if not (q and pos):
            continue
        yield {"query": q, "positive": pos, "negatives": negs[:max_negs]}
        n += 1
        if n >= limit:
            break
    log.info("  MS MARCO rows emitted: %d", n)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    sources = {s.strip() for s in args.include.split(",") if s.strip()}
    log.info("Sources: %s", sources)

    streams: list[Iterator[dict]] = []
    if "nli" in sources:
        streams.append(_stream_nli(args.nli_limit))
    if "quora" in sources:
        streams.append(_stream_quora(args.quora_limit))
    if "msmarco" in sources:
        streams.append(_stream_msmarco(args.msmarco_limit, args.negatives_per_row))

    n_train = 0
    n_val = 0
    with train_path.open("w") as ftr, val_path.open("w") as fval:
        for stream in streams:
            for row in stream:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                if rng.random() < args.val_fraction:
                    fval.write(line)
                    n_val += 1
                else:
                    ftr.write(line)
                    n_train += 1

    log.info(
        "Done. train=%d (%s)  val=%d (%s)",
        n_train,
        train_path,
        n_val,
        val_path,
    )


if __name__ == "__main__":
    main()
