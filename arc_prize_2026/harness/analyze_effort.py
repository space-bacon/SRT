#!/usr/bin/env python
"""Reasoning effort medium against the default xhigh on the 120 public evaluation tasks, task-level top-1, paired over tasks.

Medium traces (run e1, 167 outputs, max 41,000 tokens) are scored at caps 12,288, 24,576 and 36,864 with the programs sampled where the trace was cut, or where it stopped
without an answer (force_end.py), or its own program. Xhigh traces (runs g4 and g5) are scored at their 32,768 cap from the saved candidates. The contrast is medium minus
xhigh per run, paired over tasks (signed mean first, sd, sem, mean/sem, smallest effect at 80% power 2.8 x sem). Medium costs more decoded tokens at every cap it is scored at
than xhigh at 32,768 (see tokens), so a negative contrast is not explained by a smaller budget. Output goes through a temp file and a rename.
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results") if os.path.isdir(os.path.join(HERE, "results")) else os.path.join(HERE, "..", "results")
SOL = "/tmp/arcdata/arc-agi_evaluation_solutions.json"
WORK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/e1work"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "effort_analysis.json")
sol = json.load(open(SOL))


def tkey(g):
    return tuple(map(tuple, g)) if g else None


gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])


def rank(cs, j):
    v = defaultdict(float)
    for c in cs:
        ps = c.get("preds")
        p = ps[j] if ps and j < len(ps) else None
        if p is None:
            continue
        v[tkey(p)] += 2.0 if c["n_pass"] == c["n_train"] else 0.25 * c["n_pass"] / max(c["n_train"], 1)
    return [g for g, _ in sorted(v.items(), key=lambda kv: -kv[1])]


def task_mean(d):
    by = defaultdict(list)
    for k, v in d.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * float(d.mean()), 2), "sd_points": round(100 * float(d.std(ddof=1)), 2), "sem_points": round(100 * float(sem), 2),
            "mean_over_sem": round(float(d.mean() / sem), 2), "min_detectable_points_80pct": round(100 * 2.8 * float(sem), 2)}


# medium
toks, own, ex = {}, {}, defaultdict(lambda: defaultdict(list))
for l in open(os.path.join(R, "raw", "e1", "e1_code.jsonl")):
    r = json.loads(l)
    if r.get("status") == "ok" and r["sample"] == 0:
        toks[r["key"]] = r["completion_tokens"]
for l in open(os.path.join(WORK, "flat_exec.jsonl")):
    r = json.loads(l)
    if r["sample"] == "own":
        own[r["key"]] = r
    else:
        ex[r["key"]][int(r["sample"].split("_")[0][1:])].append(r)


def medium_scores(cap):
    s = {}
    for k, t in toks.items():
        cs = ([own[k]] if k in own else []) + ex[k].get(t, []) if t <= cap else ex[k].get(cap, [])
        rk = rank(cs, int(k.rsplit("_", 1)[1]))
        s[k] = float(bool(rk) and rk[0] == gold(k))
    return s


def xhigh_scores(run, cap=32768):
    s = {}
    if run == "g4":
        t4, own4, f4 = {}, {}, defaultdict(list)
        for l in open(os.path.join(R, "raw", "g4full", "g4_code.jsonl")):
            r = json.loads(l)
            if r.get("status") == "ok" and r["sample"] == 0:
                t4[r["key"]] = r["completion_tokens"]
        for l in open("/tmp/arcllm/pertask/flat_exec.jsonl"):
            r = json.loads(l)
            if r["sample"] == "own":
                own4[r["key"]] = r
            elif r["sample"].startswith(f"c{cap}_"):
                f4[r["key"]].append(r)
        for k, t in t4.items():
            cs = ([own4[k]] if k in own4 else []) if t <= cap else f4.get(k, [])
            rk = rank(cs, int(k.rsplit("_", 1)[1]))
            s[k] = float(bool(rk) and rk[0] == gold(k))
    else:
        A = os.path.join(R, "raw", "g5full", "artifacts")
        t5, own5, f5 = {}, {}, defaultdict(list)
        for l in open(os.path.join(A, "g5_code.jsonl")):
            r = json.loads(l)
            if r.get("status") == "ok" and r["sample"] == 0:
                t5[r["key"]] = r["completion_tokens"]
        for l in open(os.path.join(A, "g5_exec.jsonl")):
            r = json.loads(l)
            own5[r["key"]] = r
        for l in open(os.path.join(A, "g5_force_exec.jsonl")):
            r = json.loads(l)
            _, c, _j = str(r["sample"]).split("_")
            if int(c[1:]) == cap:
                f5[r["key"]].append(r)
        for k, t in t5.items():
            cs = ([own5[k]] if k in own5 else []) if t <= cap else f5.get(k, [])
            rk = rank(cs, int(k.rsplit("_", 1)[1]))
            s[k] = float(bool(rk) and rk[0] == gold(k))
    return s


res = {"medium_outputs": len(toks), "caps": {}}
x = {run: task_mean(xhigh_scores(run)) for run in ("g4", "g5")}
res["xhigh_32768_top1"] = {run: round(100 * float(np.mean(list(v.values()))), 2) for run, v in x.items()}
for cap in (12288, 24576, 36864):
    m = task_mean(medium_scores(cap))
    res["caps"][str(cap)] = {"medium_top1": round(100 * float(np.mean(list(m.values()))), 2),
                             "contrast_vs_xhigh_32768": {run: paired(m, x[run], f"medium cap {cap} minus xhigh {run} cap 32768, top-1") for run in x}}
tmp = OUT + ".tmp"
json.dump(res, open(tmp, "w"), indent=1)
os.replace(tmp, OUT)
print(json.dumps(res, indent=1))
