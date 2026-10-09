#!/usr/bin/env python
"""Does pooling the programs sampled at two checkpoints of the same trace beat using only the later checkpoint? (g4 data: caps 8K..63K, eight programs each.)

Per-output design, task-level means. Candidate pools: final(C2) versus pool(C1, C2) = programs at C1 plus programs at C2 (a trace that ended before a cap contributes its own program).
Votes: demo-passing programs weigh 2, partial passes 0.25 x fraction; an optional recency weight multiplies the later checkpoint. Paired signed differences, sem, smallest effect at 80% power.
"""
import json, math, sys, os
from collections import defaultdict
import numpy as np

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "g4full", "g4_code.jsonl")
SOL = "/tmp/arcdata/arc-agi_evaluation_solutions.json"
W = "/tmp/arcllm/pertask"


def tkey(g):
    return tuple(map(tuple, g)) if g else None


toks, own, ex = {}, {}, defaultdict(list)
for l in open(RUN):
    r = json.loads(l)
    if r.get("status") == "ok" and r["sample"] == 0:
        toks[r["key"]] = r["completion_tokens"]
for l in open(f"{W}/flat_exec.jsonl"):
    r = json.loads(l)
    s = r["sample"]
    if s == "own":
        own[r["key"]] = r
    else:
        ex[(r["key"], int(s.split("_")[0][1:]))].append(r)
sol = json.load(open(SOL))
gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])


def cands(k, cap):
    if toks[k] <= cap:
        return [own[k]] if k in own else []
    return ex.get((k, cap), [])


def rank(cs_w, j):
    v = defaultdict(float)
    for c, wt in cs_w:
        ps = c.get("preds")
        p = ps[j] if ps and j < len(ps) else None
        if p is None:
            continue
        v[tkey(p)] += wt * (2.0 if c["n_pass"] == c["n_train"] else 0.25 * c["n_pass"] / max(c["n_train"], 1))
    return [g for g, _ in sorted(v.items(), key=lambda kv: -kv[1])]


def task_avg(d):
    by = defaultdict(list)
    for k, v in d.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def score(spec):
    t1 = {}
    for k in toks:
        cs = []
        for cap, wt in spec:
            cs += [(c, wt) for c in cands(k, cap)]
        r_ = rank(cs, int(k.rsplit("_", 1)[1]))
        t1[k] = float(bool(r_) and r_[0] == gold(k))
    return task_avg(t1)


def paired(a, b):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return round(100 * d.mean(), 2), round(100 * sem, 2), round(float(d.mean() / sem), 2) if sem else None


have = sorted({cap for (_, cap) in ex})
print("caps with programs:", have)
for c2 in (32768, 40960, 49152, 63000):
    base = score([(c2, 1.0)])
    print(f"final({c2}): top1 {100 * np.mean(list(base.values())):.1f}")
    for c1 in (8192, 16384, 24576, 32768, 40960, 49152):
        if c1 >= c2 or c1 not in have:
            continue
        for w2 in (1.0, 2.0):
            p = score([(c1, 1.0), (c2, w2)])
            m, s, z = paired(p, base)
            print(f"   pool({c1},{c2}) later-weight {w2}: top1 {100 * np.mean(list(p.values())):.1f}  minus final: {m:+.2f} (sem {s}, mean/sem {z})")
