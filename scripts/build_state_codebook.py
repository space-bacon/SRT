"""Build a persistent magic-number state codebook from trace pairs.

Consumes the flat targets tensor + pairs JSONL produced by
``scripts/build_trace_pairs.py`` and builds a ``srt.nla.StateIndex``:

  * each hidden state -> a magic-number code (SimHash or VQ),
  * the code's canonical verbalization = the decoded *gold prefix* (which, by
    construction, exactly reproduces that state), so no AV is needed.

The result is a reusable dictionary of the model's internal states: given any
new hidden state, ``StateIndex.decode(v)`` is an O(1)/NN lookup returning a
verbalization, instead of re-running the AV or scanning the whole pool.

Example:
    python scripts/build_state_codebook.py \\
        --targets artifacts/nla/curated/trace_pairs_allL.jsonl.targets.pt \\
        --pairs   artifacts/nla/curated/trace_pairs_allL.jsonl \\
        --backbone Qwen/Qwen2.5-7B \\
        --mode simhash --n-bits 32 \\
        --out artifacts/nla/state_codebook_allL.pt

    # or a semantic VQ codebook with 4096 centroids:
    python scripts/build_state_codebook.py ... --mode vq --k 4096
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from srt.nla import StateIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path,
                   help="flat {'targets': (N,d)} .pt from build_trace_pairs.py")
    p.add_argument("--pairs", required=True, type=Path,
                   help="pairs JSONL with target_idx + gold_ids")
    p.add_argument("--backbone", required=True, type=str, help="tokenizer id")
    p.add_argument("--mode", choices=("simhash", "vq"), default="simhash")
    p.add_argument("--n-bits", type=int, default=32, help="SimHash bits (mode=simhash)")
    p.add_argument("--k", type=int, default=4096, help="num centroids (mode=vq)")
    p.add_argument("--vq-iters", type=int, default=25)
    p.add_argument("--max-states", type=int, default=None,
                   help="random-subsample the pool to at most this many states")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    targets = (obj["targets"] if isinstance(obj, dict) and "targets" in obj else obj).float()
    n, d = targets.shape

    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    texts: list[str | None] = [None] * n
    with args.pairs.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ti = int(r["target_idx"])
            if 0 <= ti < n:
                texts[ti] = tok.decode(r["gold_ids"], skip_special_tokens=True)

    idx_all = torch.arange(n)
    if args.max_states is not None and n > args.max_states:
        g = torch.Generator().manual_seed(args.seed)
        idx_all = torch.randperm(n, generator=g)[: args.max_states].sort().values
    sel_targets = targets[idx_all]
    sel_texts = [texts[i] for i in idx_all.tolist()]

    mu = sel_targets.mean(0)
    if args.mode == "vq":
        si = StateIndex.fit_vq(sel_targets, args.k, mu=mu, iters=args.vq_iters, seed=args.seed)
    else:
        si = StateIndex(d, mode="simhash", n_bits=args.n_bits, mu=mu, seed=args.seed)

    si.add_many(sel_targets, texts=sel_texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    si.save(args.out)

    s = si.stats()
    print(f"built {args.mode} codebook: {s}")
    print(f"  {s['n_codes']} distinct states out of {s['n_states']} "
          f"(collision/compression {1 - s['n_codes'] / max(s['n_states'], 1):.1%})")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
