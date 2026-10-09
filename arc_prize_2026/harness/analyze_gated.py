#!/usr/bin/env python
"""Verification-gated anytime reasoning, evaluated on saved traces with forced code finalizations.

At each checkpoint the trace is closed and several programs are sampled from the partial reasoning. A program that passes every demo ends the
episode; otherwise the trace continues to the next checkpoint. Token cost counts the reasoning generated so far plus the decode tokens of every
forced batch evaluated. A fixed-cap policy that only looks at its last checkpoint is reported beside each gated policy.
Task-level means; attempts are the first two distinct grids ranked by demo-verified votes.
"""
import argparse, json, os
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
    ap.add_argument("--force", nargs="+", required=True)
    ap.add_argument("--force-exec", nargs="+", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--checkpoints", default="8192,16384,24576,32768,40960,49152,56000,63000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-k", type=int, default=1)
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
    toks = {}
    for l in open(a.run):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] < a.first_k:
            toks[(r["key"], r["sample"])] = r
    own = {}
    for l in open(a.exec):
        r = json.loads(l)
        own[(r["key"], r["sample"])] = r
    fcost = {}
    for p in a.force:
        for l in open(p):
            r = json.loads(l)
            if "completions" in r:
                fcost[(r["key"], r["sample"], r["cap"])] = r["completion_tokens"]
    fex = defaultdict(list)
    for p in a.force_exec:
        for l in open(p):
            r = json.loads(l)
            s, cap, j = str(r["sample"]).split("_")
            fex[(r["key"], int(s), int(cap[1:]))].append(r)
    cps = [int(c) for c in a.checkpoints.split(",")]

    def cands_at(k, s, c):
        t = toks[(k, s)]["completion_tokens"]
        if t <= c:
            return [own[(k, s)]] if (k, s) in own else []
        return fex.get((k, s, c), [])

    def evaluate(cp_list, dump=None):
        acc1, acc2, tok = {}, {}, []
        have_all = True
        for (k, s) in sorted(toks):
            t = toks[(k, s)]["completion_tokens"]
            used, answer = 0, None
            for c in cp_list:
                if t <= c and (k, s) in own:
                    cs, reasoning = [own[(k, s)]], t
                else:
                    cs, reasoning = cands_at(k, s, c), min(t, c)
                    if t > c and (k, s, c) not in fcost:
                        have_all = False
                batch = fcost.get((k, s, c), 0) if t > c else 0
                used = reasoning + sum(fcost.get((k, s, cc), 0) for cc in cp_list if cc <= c and t > cc)
                full = [x for x in cs if x.get("pred") is not None and x["n_pass"] == x["n_train"]]
                answer = cs
                if full or t <= c:
                    answer = full or cs
                    break
            votes = defaultdict(float)
            for x in answer or []:
                if x.get("pred") is None:
                    continue
                votes[tkey(x["pred"])] += 2.0 if x["n_pass"] == x["n_train"] else 0.25 * x["n_pass"] / max(x["n_train"], 1)
            rank = [g for g, _ in sorted(votes.items(), key=lambda kv: -kv[1])]
            acc1[k] = float(bool(rank) and rank[0] == gold(k))
            acc2[k] = float(gold(k) in rank[:2])
            tok.append(used)
            if dump is not None:
                dump[k] = {"top1": acc1[k], "top2": acc2[k], "verified": bool([x for x in (answer or []) if x.get('pred') is not None and x['n_pass'] == x['n_train']]), "tokens": used}
        return {"top1_tasks": round(100 * float(np.mean(list(task_avg(acc1).values()))), 2), "top2_tasks": round(100 * float(np.mean(list(task_avg(acc2).values()))), 2),
                "mean_tokens": round(float(np.mean(tok)), 0), "complete_data": have_all}

    res = {"n_samples": len(toks), "policies": {}}
    dumps = {}
    for name, cp_list in [("fixed_16K", [16384]), ("fixed_32K", [32768]), ("fixed_49K", [49152]), ("fixed_63K", [63000]),
                          ("gated_16K_32K_49K_63K", [16384, 32768, 49152, 63000]), ("gated_8K_to_63K", cps),
                          ("gated_8K_16K_24K_32K", [8192, 16384, 24576, 32768]), ("gated_24K_40K_56K", [24576, 40960, 56000])]:
        dumps[name] = {}
        res["policies"][name] = evaluate(cp_list, dumps[name])
    json.dump(dumps, open(a.out + ".per_output.json", "w"))
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
