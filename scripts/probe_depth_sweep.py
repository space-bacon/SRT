#!/usr/bin/env python3
"""How many probe inputs, and how many samples. The deployment tradeoff surface.

`minority_clusters.py` moved the target. Agreement's ranking is nearly right
already: the majority cluster holds a correct answer on 160 of the 162 solvable
problems, and only 2 are held by a minority. Yet the selector scores 154. The
six in between are lost INSIDE the correct cluster, because a cluster is only as
sharp as the inputs used to build it. Two candidates that differ on the real
tests land together when six synthesised inputs cannot tell them apart. Measured
purity is 0.7730.

So the lever is `n_cases`, which has been 6 since the first run and was never
swept. It is also the cheapest knob in the system by orders of magnitude:

  K          costs a full generation each. Seconds. This is the latency budget.
  n_cases    costs one function call each. Microseconds against a decode.

Both matter for deployment and they are not interchangeable, so this sweeps the
grid rather than one line through it. What we want out of it:

  does purity rise with n_cases, and does accuracy follow
  where does n_cases saturate, since past some depth the inputs stop separating
  how much K can be traded away once the probe is sharp, because K is the cost

    python scripts/probe_depth_sweep.py --workers 48
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
from code_select import extract  # noqa: E402
from consensus_select import (inputs_from_example, probe_program,  # noqa: E402
                              run_capture, synth_inputs)
from verifier_select import pass_matrix  # noqa: E402


def signatures(problem, candidates, cases, timeout, pool):
    entry = problem["entry_point"]
    jobs = [probe_program(extract(problem["prompt"], c, entry), entry, cases)
            for c in candidates]
    return list(pool.map(lambda j: run_capture(j, timeout), jobs))


def score(sigs, ok_row, k):
    """Agreement over the first k candidates: did the pick pass, and was its cluster pure."""
    sub = sigs[:k]
    ran = [i for i, s in enumerate(sub) if s is not None]
    if not ran:
        return None, None
    top, _ = Counter(sub[i] for i in ran).most_common(1)[0]
    members = [i for i in ran if sub[i] == top]
    pick = members[0]
    fates = {bool(ok_row[i]) for i in members}
    return bool(ok_row[pick]), len(fates) == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/mbpp_ladder")
    ap.add_argument("--probs", default="data/mbpp.json")
    ap.add_argument("--cache", default="artifacts/nla/verifier/passmat")
    ap.add_argument("--out", default="artifacts/nla/verifier/probe_depth.json")
    ap.add_argument("--cases", type=int, nargs="+", default=[2, 4, 6, 12, 24, 48])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    if a.arms:
        files = [f for f in files if os.path.basename(f)[:-5] in a.arms]
    print(f"{len(files)} arms, {len(probs)} problems", flush=True)

    res = {"question": "does a sharper probe buy accuracy, and can it buy back K",
           "bench": os.path.basename(a.probs)[:-5],
           "n_problems": len(probs), "cases_swept": a.cases, "ks_swept": a.ks,
           "arms": {}}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in files:
            arm = os.path.basename(f)[:-5]
            rows = json.load(open(f))
            if len(rows) != len(probs):
                continue
            kmin = min(len(c) for c in rows)
            rows = [c[:kmin] for c in rows]
            ok = pass_matrix(arm, rows, probs, os.path.join(HERE, a.cache),
                             a.timeout, a.workers)
            entry = {"floor": round(float(ok.mean()), 4),
                     "oracle": round(float(ok.any(1).mean()), 4),
                     "grid": {}}

            for n_cases in a.cases:
                cases = []
                for p in probs:
                    c = synth_inputs(p["prompt"], p["entry_point"], n_cases)
                    if not c:
                        c = inputs_from_example(p.get("visible_test"),
                                                p["entry_point"], n_cases)
                    cases.append(c)
                sigs = [signatures(probs[i], rows[i], cases[i], a.timeout, pool)
                        if cases[i] else [None] * len(rows[i])
                        for i in range(len(probs))]
                for k in a.ks:
                    if k > kmin:
                        continue
                    passed, pure = [], []
                    for i in range(len(probs)):
                        p_, u_ = score(sigs[i], ok[i], k)
                        # Unresolved falls back to an arbitrary candidate, as deployment would.
                        passed.append(bool(ok[i, 0]) if p_ is None else p_)
                        if u_ is not None:
                            pure.append(u_)
                    entry["grid"][f"cases{n_cases}_k{k}"] = {
                        "selected": round(float(np.mean(passed)), 4),
                        "purity": round(float(np.mean(pure)), 4) if pure else None,
                    }
                    print(f"  {arm:26s} cases {n_cases:3d} k {k}  "
                          f"sel {np.mean(passed):.4f}  pure {np.mean(pure):.4f}",
                          flush=True)
            res["arms"][arm] = entry

    # Aggregate each grid cell across arms so the shape is readable at a glance.
    cells = {}
    for e in res["arms"].values():
        for cell, v in e["grid"].items():
            cells.setdefault(cell, []).append(v["selected"])
    res["summary"] = {c: round(float(np.mean(v)), 4) for c, v in sorted(cells.items())}
    res["summary_floor"] = round(
        float(np.mean([e["floor"] for e in res["arms"].values()])), 4)
    res["summary_oracle"] = round(
        float(np.mean([e["oracle"] for e in res["arms"].values()])), 4)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\n  floor {res['summary_floor']:.4f}  oracle {res['summary_oracle']:.4f}")
    for c, v in res["summary"].items():
        print(f"  {c:16s} {v:.4f}")
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
