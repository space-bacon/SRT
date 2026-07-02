"""Build multi-position, multi-layer NLA training pairs from a sample file.

Consumes a ``scripts/sample_targets.py`` ``.pt`` file (saved with the updated
script, so it carries ``token_ids`` and ``activations_by_layer``) and emits:

  * ``<out>``            — JSONL of ``{target_idx, gold_ids, seq_idx, pos,
                          layer, n_tokens}`` records (one per traced state).
  * ``<out>.targets.pt`` — a flat ``{"targets": (N, d), "meta": {...}}`` tensor
                          aligned so ``targets[target_idx]`` is that record's
                          hidden vector.

Feed both to ``scripts/train_nla_act.py`` (``--pairs <out> --targets
<out>.targets.pt``). Because attention is causal, each ``gold_ids`` prefix
exactly reproduces its target vector at every layer, so CE training on these
pairs teaches the AV to verbalize the whole input->output loop across layers.

Example (full-stack, strided positions, capped):
    python scripts/build_trace_pairs.py \\
        --sample artifacts/nla/targets_q7b_allL_seq64_10k.pt \\
        --layers all --position-stride 4 --min-prefix-len 2 \\
        --max-pairs 200000 \\
        --out artifacts/nla/curated/trace_pairs_allL.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from srt.nla.trace_data import build_trace_pairs, resolve_saved_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", required=True, type=Path,
                   help="sample_targets.py .pt file (needs token_ids + activations_by_layer)")
    p.add_argument("--layers", type=str, default="all",
                   help="'all' or a comma-separated subset of saved layers")
    p.add_argument("--position-stride", type=int, default=1)
    p.add_argument("--min-prefix-len", type=int, default=1)
    p.add_argument("--max-prefix-len", type=int, default=None)
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    obj = torch.load(args.sample, map_location="cpu", weights_only=False)
    saved = resolve_saved_layers(obj)
    layers = None if args.layers.strip() == "all" else [
        int(x) for x in args.layers.split(",") if x.strip()
    ]

    targets, records = build_trace_pairs(
        obj,
        layers=layers,
        position_stride=args.position_stride,
        min_prefix_len=args.min_prefix_len,
        max_prefix_len=args.max_prefix_len,
        max_sequences=args.max_sequences,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if targets.numel() == 0:
        raise SystemExit("no pairs produced — check --layers / --min-prefix-len bounds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    targets_path = args.out.with_suffix(args.out.suffix + ".targets.pt")
    meta = dict(obj.get("meta", {}))
    meta.update({"saved_layers": saved, "used_layers": layers or saved, "n_pairs": len(records)})
    torch.save({"targets": targets, "meta": meta}, targets_path)

    by_layer: dict[int, int] = {}
    for r in records:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + 1
    print(f"wrote {len(records)} pairs -> {args.out}")
    print(f"wrote targets {tuple(targets.shape)} -> {targets_path}")
    print(f"pairs per layer: {dict(sorted(by_layer.items()))}")


if __name__ == "__main__":
    main()
