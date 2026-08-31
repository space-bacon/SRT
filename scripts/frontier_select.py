#!/usr/bin/env python3
"""Does the selector do anything for reasoning-first MoE models.

The decay law says selection is worth +0.0097 at 32B and falls -0.0704 per
decade, so at frontier scale it should be worth nothing. That law was fit on
Qwen2.5-Coder: dense, non-reasoning, one family, one lab. It has no standing over
DeepSeek-V4, Qwen3.8-Next and GLM-5.3, which are mixture-of-experts and
reasoning-first, and which we run with the thinking block suppressed.

Two ways it could break, in opposite directions:

  collapse   heavy RL post-training drives every sample toward one answer, the
             pool has nothing to choose between, and selection is worth less
             than the law predicts rather than more.
  branching  suppressed CoT still leaves the model exploring distinct solution
             routes, candidate diversity is HIGHER than a dense model's, and
             selection is worth more even though pass@1 is already high.

Distinguishing them needs the diversity measured, not just the score, so this
reports distinct output clusters per problem alongside the usual floor/selected/
oracle. A high score with one cluster means collapse; a high score with many
means the pool has something a better selector could still win.

    python scripts/frontier_select.py --gen-dir artifacts/nla/frontier
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import execute_file, extract  # noqa: E402
from consensus_select import (inputs_from_example, probe_program,  # noqa: E402
                              run_capture, synth_inputs)

# Qwen2.5-Coder, dense non-reasoning, for the comparison the whole point rests on.
DENSE_REFERENCE = {"3B": 0.1538, "7B": 0.1188, "14B": 0.0765, "32B": 0.0097}


def clusters_for(problem, candidates, cases, timeout):
    entry = problem["entry_point"]
    sigs = [run_capture(probe_program(extract(problem["prompt"], c, entry), entry, cases),
                        timeout) for c in candidates]
    groups = {}
    for i, s in enumerate(sigs):
        if s is not None:
            groups.setdefault(s, []).append(i)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/frontier")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/frontier/frontier_select.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(glob.glob(os.path.join(HERE, a.gen_dir, "*__chat.json")))
    if not files:
        print(f"no generations in {a.gen_dir}", flush=True)
        return 1

    cases = []
    for p in probs:
        c = synth_inputs(p["prompt"], p["entry_point"], a.cases)
        if not c:
            c = inputs_from_example(p.get("visible_test"), p["entry_point"], a.cases)
        cases.append(c)

    from concurrent.futures import ThreadPoolExecutor
    res = {"question": "is selection worth anything on reasoning-first MoE models",
           "n_problems": len(probs), "dense_reference": DENSE_REFERENCE, "members": {}}

    for f in files:
        tag = os.path.basename(f)[:-len("__chat.json")]
        rows = json.load(open(f))
        if len(rows) != len(probs):
            print(f"  {tag:20s} SKIP {len(rows)} rows != {len(probs)}", flush=True)
            continue
        kmin = min(len(c) for c in rows)
        rows = [c[:kmin] for c in rows]
        ok = execute_file(rows, probs, a.timeout, a.workers)

        with ThreadPoolExecutor(max_workers=8) as ex:
            groups = list(ex.map(
                lambda i: clusters_for(probs[i], rows[i], cases[i], a.timeout)
                if cases[i] else {}, range(len(probs))))

        picks, ncl = [], []
        for i, g in enumerate(groups):
            if not g:
                picks.append(None)
                continue
            ncl.append(len(g))
            picks.append(max(g.values(), key=len)[0])
        sel = [bool(ok[i, p if p is not None else 0]) for i, p in enumerate(picks)]

        floor, oracle = float(ok.mean()), float(ok.any(1).mean())
        selected = float(np.mean(sel))
        res["members"][tag] = {
            "k": int(ok.shape[1]),
            "floor": round(floor, 4),
            "selected": round(selected, 4),
            "oracle": round(oracle, 4),
            "gain": round(selected - floor, 4),
            "headroom_captured": round((selected - floor) / (oracle - floor), 4)
                                 if oracle > floor else None,
            "mean_clusters": round(statistics.mean(ncl), 2) if ncl else None,
            "single_cluster_problems": sum(1 for c in ncl if c == 1),
            "resolved": sum(1 for p in picks if p is not None),
        }
        v = res["members"][tag]
        print(f"  {tag:20s} k={v['k']:3d}  floor {floor:.4f}  sel {selected:.4f}  "
              f"oracle {oracle:.4f}  gain {v['gain']:+.4f}  "
              f"clusters {v['mean_clusters']}", flush=True)

    if res["members"]:
        gains = [m["gain"] for m in res["members"].values()]
        cl = [m["mean_clusters"] for m in res["members"].values() if m["mean_clusters"]]
        res["summary"] = {
            "mean_gain": round(float(np.mean(gains)), 4),
            "dense_32b_gain": DENSE_REFERENCE["32B"],
            "mean_clusters": round(float(np.mean(cl)), 2) if cl else None,
            "reading": "gain at or below the 32B dense value means the decay law "
                       "extends to reasoning MoE. A low cluster count with a high "
                       "score means collapse, not a good selector",
        }
        s = res["summary"]
        print(f"\n  mean gain {s['mean_gain']:+.4f} vs dense-32B {s['dense_32b_gain']:+.4f}"
              f"   mean clusters {s['mean_clusters']}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
