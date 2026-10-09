#!/usr/bin/env python
"""Accuracy and token cost of code-mode runs under reasoning caps with forced code finalization.

Inputs: the run JSONL, executor results for the run, forced completions with their executor results. For a cap C a sample at or below C keeps its
own program; a longer sample contributes the programs sampled from its truncated reasoning. Programs that pass every demo vote with weight 2,
partial passes with 0.25 x their pass fraction; the top two distinct grids are the attempts. Task-level means over the public evaluation set.
"""
import argparse, json, math, os
from collections import defaultdict
import numpy as np


def tkey(g):
    return tuple(map(tuple, g)) if g else None


def task_avg(d):
    by = defaultdict(list)
    for k, v in d.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exec", required=True)
    ap.add_argument("--force", required=True)
    ap.add_argument("--force-exec", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--caps", default="16384,32768,49152,64000")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
    toks = {}
    for l in open(a.run):
        r = json.loads(l)
        if r.get("status") == "ok":
            toks[(r["key"], r["sample"])] = r["completion_tokens"]
    own = {}
    for l in open(a.exec):
        r = json.loads(l)
        own[(r["key"], r["sample"])] = r
    fexec = defaultdict(list)
    for l in open(a.force_exec):
        r = json.loads(l)
        s, cap, j = str(r["sample"]).split("_")
        fexec[(r["key"], int(s), int(cap[1:]))].append(r)
    keys = sorted({k for k, _ in toks})
    res = {"n_outputs": len(keys), "n_samples": len(toks), "caps": {}}
    for cap in [int(c) for c in a.caps.split(",")]:
        top2, top1, tokens_used, n_forced_ok = {}, {}, [], 0
        for k in keys:
            score = defaultdict(float)
            samples = [s for (kk, s) in toks if kk == k]
            for s in samples:
                t = toks[(k, s)]
                tokens_used.append(min(t, cap) + (a.n * 1500 if t > cap and cap < 64000 else 0))
                cands = [own[(k, s)]] if (t <= cap and (k, s) in own) else (fexec.get((k, s, cap), []) if t > cap else [])
                for c in cands:
                    if c.get("pred") is None:
                        continue
                    full = c["n_pass"] == c["n_train"]
                    score[tkey(c["pred"])] += 2.0 if full else 0.25 * c["n_pass"] / max(c["n_train"], 1)
            rank = [g for g, _ in sorted(score.items(), key=lambda kv: -kv[1])]
            top2[k] = float(gold(k) in rank[:2])
            top1[k] = float(bool(rank) and rank[0] == gold(k))
        res["caps"][cap] = {"verified_top2_tasks": round(100 * float(np.mean(list(task_avg(top2).values()))), 2),
                            "verified_top1_tasks": round(100 * float(np.mean(list(task_avg(top1).values()))), 2),
                            "mean_tokens_per_sample": round(float(np.mean(tokens_used)), 0)}
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
