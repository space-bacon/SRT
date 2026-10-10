#!/usr/bin/env python
"""Run-to-run variation and pooled K=2 for the program arm, from two independent runs (g4 and g5) on the same 120 public evaluation tasks.

Per-output design (each test output has its own trace and its own eight forced programs per cap), task-level means. A run contributes, at a given cap, its own
program when its trace ended before the cap and its forced programs otherwise. Pooled K=2 takes the union of both runs' candidates and votes: demo-passing
programs weigh 2, partial passes 0.25 x fraction. Contrasts are paired over tasks (signed mean, sd, sem, mean/sem, smallest effect detectable at 80% power = 2.8 x sem).
Writes through a temp file and rename.
"""
import json, math, os
from collections import defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
G4_RUN = f"{R}/raw/g4full/g4_code.jsonl"
G4_FLAT = "/tmp/arcllm/pertask/flat_exec.jsonl"
G5 = f"{R}/raw/g5full/artifacts"
SOL = "/tmp/arcdata/arc-agi_evaluation_solutions.json"
CAPS = (16384, 32768, 49152, 63000)


def tkey(g):
    return tuple(map(tuple, g)) if g else None


def load_g4():
    toks, own, forced = {}, {}, defaultdict(list)
    for l in open(G4_RUN):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] == 0:
            toks[r["key"]] = r["completion_tokens"]
    for l in open(G4_FLAT):
        r = json.loads(l)
        sid = r["sample"]
        if sid == "own":
            own[r["key"]] = r
        else:
            forced[(r["key"], int(sid.split("_")[0][1:]))].append(r)
    return toks, own, forced


def load_g5():
    toks, own, forced = {}, {}, defaultdict(list)
    for l in open(f"{G5}/g5_code.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] == 0:
            toks[r["key"]] = r["completion_tokens"]
    for l in open(f"{G5}/g5_exec.jsonl"):
        r = json.loads(l)
        own[r["key"]] = r
    for l in open(f"{G5}/g5_force_exec.jsonl"):
        r = json.loads(l)
        _, cap, _j = str(r["sample"]).split("_")
        forced[(r["key"], int(cap[1:]))].append(r)
    return toks, own, forced


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * float(d.mean()), 2), "sd_points": round(100 * float(d.std(ddof=1)), 2),
            "sem_points": round(100 * float(sem), 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None, "min_detectable_points_80pct": round(100 * 2.8 * float(sem), 2)}


def main():
    sol = json.load(open(SOL))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
    runs = {"g4": load_g4(), "g5": load_g5()}
    keys = sorted(set(runs["g4"][0]) & set(runs["g5"][0]))

    def cands(run, k, cap):
        toks, own, forced = runs[run]
        if toks[k] <= cap:
            return [own[k]] if k in own else []
        return forced.get((k, cap), [])

    def rank(cs, j):
        v = defaultdict(float)
        for c in cs:
            ps = c.get("preds")
            p = ps[j] if ps and j < len(ps) else None
            if p is None:
                continue
            v[tkey(p)] += 2.0 if c["n_pass"] == c["n_train"] else 0.25 * c["n_pass"] / max(c["n_train"], 1)
        return [g for g, _ in sorted(v.items(), key=lambda kv: -kv[1])]

    def task_avg(d):
        by = defaultdict(list)
        for k, v in d.items():
            by[k.rsplit("_", 1)[0]].append(v)
        return {t: float(np.mean(v)) for t, v in by.items()}

    def score(spec):
        """spec: list of (run, cap); returns task-level top1 and top2 dicts."""
        t1, t2 = {}, {}
        for k in keys:
            cs = []
            for run, cap in spec:
                cs += cands(run, k, cap)
            r_ = rank(cs, int(k.rsplit("_", 1)[1]))
            g = gold(k)
            t1[k] = float(bool(r_) and r_[0] == g)
            t2[k] = float(g in r_[:2])
        return task_avg(t1), task_avg(t2)

    mean = lambda d: round(100 * float(np.mean(list(d.values()))), 2)
    res = {"n_outputs": len(keys), "n_tasks": len({k.rsplit('_', 1)[0] for k in keys}), "single": {}, "pooled_k2": {}, "contrasts": []}
    single = {}
    for run in ("g4", "g5"):
        for cap in CAPS:
            t1, t2 = score([(run, cap)])
            single[(run, cap)] = (t1, t2)
            res["single"][f"{run}_cap{cap}"] = {"top1": mean(t1), "top2": mean(t2)}
    pooled = {}
    for cap in CAPS:
        t1, t2 = score([("g4", cap), ("g5", cap)])
        pooled[cap] = (t1, t2)
        res["pooled_k2"][f"cap{cap}"] = {"top1": mean(t1), "top2": mean(t2)}
    for cap in CAPS:
        res["contrasts"].append(paired(single[("g5", cap)][0], single[("g4", cap)][0], f"cap {cap}: g5 minus g4, top-1 (run-to-run)"))
    for cap in CAPS:
        a = {k: (single[("g4", cap)][0][k] + single[("g5", cap)][0][k]) / 2 for k in single[("g4", cap)][0]}
        res["contrasts"].append(paired(pooled[cap][0], a, f"cap {cap}: pooled K=2 minus mean of the two single runs, top-1"))
    # equal-budget style comparisons (mean decode tokens per trace per cap from analyze_gated: about 24.5K, 41.9K, 58.2K, 70.1K)
    for lo, hi in ((16384, 32768), (32768, 49152), (32768, 63000)):
        a = {k: (single[("g4", hi)][0][k] + single[("g5", hi)][0][k]) / 2 for k in single[("g4", hi)][0]}
        res["contrasts"].append(paired(pooled[lo][0], a, f"pooled K=2 at cap {lo} minus mean single run at cap {hi}, top-1"))
    per_task = {f"{run}_cap{cap}": {"top1": single[(run, cap)][0], "top2": single[(run, cap)][1]} for run in ("g4", "g5") for cap in CAPS}
    tmp_pt = os.path.join(R, "two_runs_per_task.json.tmp")
    json.dump(per_task, open(tmp_pt, "w"))
    os.replace(tmp_pt, os.path.join(R, "two_runs_per_task.json"))
    res["mean_tokens_per_trace_approx"] = {"16384": 24.5, "32768": 41.9, "49152": 58.2, "63000": 70.1, "unit": "thousand tokens, g4 and g5 agree to within 0.2K"}
    tmp = os.path.join(R, "two_runs.json.tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(R, "two_runs.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
