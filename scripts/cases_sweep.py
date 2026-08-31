#!/usr/bin/env python3
"""How many probe inputs does agreement actually need.

`minority_clusters.py` moved the bottleneck. The majority cluster is correct on
160 of the 162 solvable problems, so ranking clusters is very nearly solved and
only 2 problems are genuinely minority-held. But the selector scores 154. The
missing problems are lost INSIDE the correct cluster: purity is 0.7730 and 59
problems carry a cluster holding both passing and failing candidates, so we pick
a failing member of the right cluster.

A cluster is impure when two candidates agree on every synthesised input and
differ on the real tests. That is a resolution failure, and `n_cases` is the
resolution knob. It has sat at its default of 6 since the first version and was
never swept.

The economics are lopsided. Another probe input is one more call to a function
that is already loaded, costing microseconds. Another candidate is a full
generation, costing seconds. If accuracy is still climbing at 6, it is close to
free to buy.

Reports purity and selection accuracy against n_cases, on the same pools, so the
knob can be set from evidence rather than from a default nobody chose.

    python scripts/cases_sweep.py --workers 48
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import execute_file, extract  # noqa: E402
from consensus_select import (inputs_from_example, probe_program,  # noqa: E402
                              run_capture, synth_inputs)


def build_cases(probs, n, seed=0):
    out = []
    for p in probs:
        c = synth_inputs(p["prompt"], p["entry_point"], n, seed)
        if not c:
            c = inputs_from_example(p.get("visible_test"), p["entry_point"], n, seed)
        out.append(c)
    return out


def evaluate(probs, pool, ok, cases, timeout, workers):
    """Selection accuracy and cluster purity at one n_cases setting."""
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        if not cases[i]:
            return None, None
        entry = probs[i]["entry_point"]
        sigs = [run_capture(probe_program(extract(probs[i]["prompt"], c, entry), entry,
                                          cases[i]), timeout) for c in pool[i]]
        groups = {}
        for j, s in enumerate(sigs):
            if s is not None:
                groups.setdefault(s, []).append(j)
        if not groups:
            return None, None
        best = max(groups.values(), key=len)
        pure = sum(1 for g in groups.values() if len({bool(ok[i, j]) for j in g}) == 1)
        return bool(ok[i, best[0]]), pure / len(groups)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, range(len(probs))))

    picked = [r[0] for r in res]
    purity = [r[1] for r in res if r[1] is not None]
    # Unresolved problems fall back to an arbitrary candidate, as a deployment would.
    full = [p if p is not None else bool(ok[i, 0]) for i, p in enumerate(picked)]
    return {
        "selected": round(float(np.mean(full)), 4),
        "resolved": sum(1 for p in picked if p is not None),
        "purity": round(float(np.mean(purity)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/ensemble")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/ensemble/cases_sweep.json")
    ap.add_argument("--cases", nargs="*", type=int, default=[2, 4, 6, 12, 24, 48])
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    pool = [[] for _ in probs]
    for f in files:
        rows = json.load(open(f))
        for i, cands in enumerate(rows):
            pool[i].extend(cands)
    print(f"{len(files)} arms, {len(pool[0])} candidates per problem", flush=True)

    ok = execute_file(pool, probs, a.timeout, a.workers)
    floor = float(ok.mean())
    oracle = float(ok.any(1).mean())
    print(f"floor {floor:.4f}  oracle {oracle:.4f}\n", flush=True)

    res = {"question": "does agreement need more probe inputs than the default 6",
           "n_problems": len(probs), "pool_size": len(pool[0]),
           "floor": round(floor, 4), "oracle": round(oracle, 4), "sweep": {}}
    for n in a.cases:
        r = evaluate(probs, pool, ok, build_cases(probs, n), a.timeout, a.workers)
        res["sweep"][str(n)] = r
        print(f"  n_cases {n:3d}   selected {r['selected']:.4f}   "
              f"purity {r['purity']:.4f}   resolved {r['resolved']}/{len(probs)}",
              flush=True)

    best = max(res["sweep"], key=lambda k: res["sweep"][k]["selected"])
    res["best_n_cases"] = int(best)
    res["gain_over_default_6"] = round(
        res["sweep"][best]["selected"] - res["sweep"].get("6", {}).get("selected", 0), 4)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\n  best n_cases {best}, {res['gain_over_default_6']:+.4f} over the default")
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
