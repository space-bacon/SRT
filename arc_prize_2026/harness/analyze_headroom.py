#!/usr/bin/env python
"""Selection headroom of the program arm: top-1, top-2 and the oracle (the gold grid is among the distinct grids any candidate produced), task-level means over the 120 public
evaluation tasks, runs g4 and g5 pooled by averaging the two runs' per-task means. Candidates are the own program of a trace that ended before the cap or the eight forced programs
at the cap, ranked by demo-weighted votes (analyze_direct.candidates). Writes through a temp file and a rename."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_direct as D

OUT = os.path.join(D.R, "selection_headroom.json")
res = {}
for cap in (32768, 49152):
    cand = D.candidates(cap)
    s1, s2, so = {}, {}, {}
    for k, (rk, t) in cand.items():
        g = D.gold(k[1])
        s1[k], s2[k], so[k] = float(bool(rk) and rk[0] == g), float(g in rk[:2]), float(g in rk)
    tm = lambda d: round(100 * float(np.mean(list(D.task_mean(d).values()))), 2)
    res[str(cap)] = {"top1": tm(s1), "top2": tm(s2), "oracle": tm(so), "oracle_minus_top1": round(tm(so) - tm(s1), 2), "n_outputs": len(cand)}
tmp = OUT + ".tmp"
json.dump(res, open(tmp, "w"), indent=1)
os.replace(tmp, OUT)
print(json.dumps(res, indent=1))
