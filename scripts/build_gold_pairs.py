"""Build a {target_idx, gold_ids} JSONL from a sample_targets.py .pt file.

Re-tokenizes the saved ``sequences`` field with the backbone tokenizer and
emits one record per sequence that fits inside ``--max-len`` tokens.

Example:
    python scripts/build_gold_pairs.py \\
        --targets artifacts/nla/targets_q7b_L20_seq64_10k.pt \\
        --backbone Qwen/Qwen2.5-7B \\
        --max-len 64 \\
        --out artifacts/nla/curated/gold_pairs_seq64.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--backbone", required=True, type=str)
    p.add_argument("--max-len", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    seqs = obj["sequences"]
    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with args.out.open("w") as f:
        for idx, s in enumerate(seqs):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) > args.max_len:
                skipped += 1
                continue
            f.write(json.dumps({"target_idx": idx, "gold_ids": ids, "n_tokens": len(ids)}) + "\n")
            kept += 1
    print(f"wrote {kept} pairs (skipped {skipped}) -> {args.out}")


if __name__ == "__main__":
    main()
