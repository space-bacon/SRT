#!/usr/bin/env python
"""Accuracy of the forced-finalization program arm when only the first n of the 8 sampled completions are used.

Reads the flattened, executed g4 programs from per_task_eval's work dir (flat.jsonl, flat_exec.jsonl). Own-trace programs (finished before the cap) are kept as is.
Votes: demo-passing programs weigh 2, partial passes 0.25 x pass fraction. Per-output design, task-level means (mean over a task's outputs, then over tasks).
Contrast: paired signed difference over the 120 tasks versus n = 8, with sem and the smallest effect detectable at 80% power (2.8 x sem).
"""
import json, math, sys, os
from collections import defaultdict
import numpy as np

W = "/tmp/arcllm/pertask"
sol = json.load(open("/tmp/arcdata/solutions.json")) if len(sys.argv) < 2 else json.load(open(sys.argv[1]))
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "g4full", "g4_code.jsonl")


def tkey(g):
    return tuple(map(tuple, g)) if g else None


toks = {}
for l in open(RUN):
    r = json.loads(l)
    if r.get("status") == "ok" and r["sample"] == 0:
        toks[r["key"]] = r["completion_tokens"]
ex = defaultdict(list)
own = {}
for l in open(f"{W}/flat_exec.jsonl"):
    r = json.loads(l)
    sid = r["sample"]
    if sid == "own":
        own[r["key"]] = r
    else:
        cap, fi, j = sid.split("_")
        ex[(r["key"], int(cap[1:]))].append((int(j[1:]), r))


def gold(k):
    t, j = k.rsplit("_", 1)
    return tkey(sol[t][int(j)])


def cands(k, cap, n):
    if toks[k] <= cap:
        return [own[k]] if k in own else []
    return [r for j, r in ex.get((k, cap), []) if j < n]


def rank(cs, j):
    v = defaultdict(float)
    for c in cs:
        ps = c.get("preds")
        p = ps[j] if ps and j < len(ps) else None
        if p is None:
            continue
        v[tkey(p)] += 2.0 if c["n_pass"] == c["n_train"] else 0.25 * c["n_pass"] / max(c["n_train"], 1)
    return [g for g, _ in sorted(v.items(), key=lambda kv: -kv[1])]


def score(cap, n):
    out = defaultdict(list)
    for k in toks:
        tid, j = k.rsplit("_", 1)
        r_ = rank(cands(k, cap, n), int(j))
        out[tid].append(float(bool(r_) and r_[0] == gold(k)))
    return {t: float(np.mean(v)) for t, v in out.items()}


res = {}
for cap in (16384, 32768, 49152, 63000):
    base = score(cap, 8)
    row = {"n8_top1": round(100 * np.mean(list(base.values())), 2)}
    for n in (1, 2, 4):
        s = score(cap, n)
        ks = sorted(base)
        d = np.array([s[k] - base[k] for k in ks])
        sem = d.std(ddof=1) / math.sqrt(len(d))
        row[f"n{n}_top1"] = round(100 * np.mean(list(s.values())), 2)
        row[f"n{n}_minus_n8"] = {"mean": round(100 * d.mean(), 2), "sem": round(100 * sem, 2), "mean_over_sem": round(d.mean() / sem, 2) if sem else None, "mde80": round(100 * 2.8 * sem, 2)}
    res[str(cap)] = row
print(json.dumps(res, indent=1))
