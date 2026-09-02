"""Execute the frontier ensemble pools once, so the demo can show them.

Six models from five labs answered the same 164 HumanEval problems eight times
each. `pooled_select.py` executed those candidates and threw the results away,
so nothing downstream can render them. This runs the 7,872 executions once and
persists a pass matrix per arm, in the same format as
`artifacts/nla/verifier/passmat_he_k32`, which is what `bank_consensus_demo.py`
reads.

Run it on a disposable machine. It executes model-written code.

    python scripts/bank_frontier_passmat.py --workers 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from code_select import execute_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/ensemble")
    ap.add_argument("--out", default="artifacts/nla/verifier/passmat_ensemble")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    print(f"{len(files)} arms, {len(probs)} problems", flush=True)

    for f in files:
        arm = os.path.basename(f)[:-5]
        p = os.path.join(out, f"{arm}.npz")
        if os.path.isfile(p):
            print(f"  {arm:24s} cached", flush=True)
            continue
        rows = json.load(open(f))
        if len(rows) != len(probs):
            print(f"  {arm:24s} SKIP {len(rows)} rows", flush=True)
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = execute_file(rows, probs, a.timeout, a.workers)
        np.savez_compressed(p, ok=ok)
        print(f"  {arm:24s} K={kmin}  pass@1 {ok.mean():.4f}  "
              f"oracle@{kmin} {ok.any(1).mean():.4f}", flush=True)

    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
