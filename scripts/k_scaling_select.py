#!/usr/bin/env python3
"""How far does a small model climb by sampling more, and what does it reach?

Section 5.4 measures selection at K = 8 and finds it worth about one size class.
That was one point on a curve. The product question is the other axis: given a
model small enough to serve on hardware you own, how much of a larger model's
unaided score can more samples buy back.

Everything here is CPU work over generations that already exist, so the curve
costs no GPU. For each arm the pass matrix is computed once over the full pool
and then sliced, which makes K = 8 and K = 32 exactly nested rather than two
independent draws, so the curve is within-pool and not confounded by sampling.

The cache is namespaced by generation directory. `coder7B_inst__chat` names a
different pool in `coder_ladder` than in `he_k32`, and a shared cache key would
silently score one pool with the other's matrix.

    python scripts/k_scaling_select.py --gen-dir artifacts/nla/he_k32 \
        --validate humaneval --ks 1 2 4 8 16 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-dir", default="artifacts/nla/he_k32")
    p.add_argument("--validate", choices=["humaneval", "mbpp"], default="humaneval")
    p.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--cases", type=int, default=2)
    p.add_argument("--timeout", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    from concurrent.futures import ThreadPoolExecutor
    from chat_consensus import choose
    from verifier_select import pass_matrix

    probs = json.load(open(os.path.join(
        HERE, "data", "humaneval.json" if a.validate == "humaneval" else "mbpp.json")))
    tag = os.path.basename(a.gen_dir.rstrip("/"))
    cache = os.path.join(HERE, "artifacts/nla/verifier/passmat_" + tag)
    out = a.out or f"artifacts/nla/{tag}_k_scaling.json"

    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    if a.arms:
        files = [f for f in files if os.path.basename(f)[:-5] in a.arms]

    res = {"question": "what does sampling more buy a small model",
           "bench": a.validate, "gen_dir": a.gen_dir, "n_problems": len(probs),
           "cases": a.cases, "arms": {},
           "note": "pass matrix computed once per arm over the full pool, then sliced, "
                   "so the K points are nested subsets of one pool"}

    for f in files:
        arm = os.path.basename(f)[:-5]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            print(f"  {arm:30s} SKIP {len(rows)} rows vs {len(probs)} problems", flush=True)
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        t0 = time.time()
        ok = pass_matrix(arm, rows, probs, cache, a.timeout, a.workers)
        res["arms"][arm] = {"pool_k": kmin, "k": {}}
        for k in [k for k in a.ks if k <= kmin]:
            sub = [c[:k] for c in rows]
            okk = ok[:, :k]
            if k == 1:
                # One candidate is the floor by definition; there is nothing to choose.
                sel_full, resolved = okk[:, 0].astype(bool).tolist(), len(rows)
            else:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    picks = list(ex.map(
                        lambda i: choose(probs[i]["prompt"], sub[i], a.cases, a.timeout)[0],
                        range(len(sub))))
                resolved = sum(1 for p in picks if p is not None)
                sel_full = [bool(okk[i, picks[i] if picks[i] is not None else 0])
                            for i in range(len(sub))]
            res["arms"][arm]["k"][str(k)] = {
                "selected": round(float(np.mean(sel_full)), 4),
                "floor": round(float(okk.mean()), 4),
                "oracle": round(float(okk.any(1).mean()), 4),
                "resolved": resolved,
                "gain_over_floor": round(float(np.mean(sel_full) - okk.mean()), 4)}
            v = res["arms"][arm]["k"][str(k)]
            print(f"  {arm:30s} K={k:<3d} sel {v['selected']:.4f}  floor {v['floor']:.4f}"
                  f"  oracle {v['oracle']:.4f}  gain {v['gain_over_floor']:+.4f}", flush=True)
        print(f"  {arm:30s} done in {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(os.path.join(HERE, out)), exist_ok=True)
        json.dump(res, open(os.path.join(HERE, out), "w"), indent=2)

    json.dump(res, open(os.path.join(HERE, out), "w"), indent=2)
    print(f"\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
