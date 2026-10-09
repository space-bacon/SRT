#!/usr/bin/env python
"""Pool analysis of run_llm_arc.py output against the public ARC-AGI-2 evaluation solutions.

Reports pass@k, vote-based top-2 selection, token-cap curves, and the union with stored NVARC-lineage runs.
Numbers are task-level means (mean over a task's test outputs, then over tasks), the competition's convention.
Every contrast is a paired signed mean over tasks with its standard error and the smallest effect detectable at 80% power.
"""
import argparse, json, math, os
from collections import Counter, defaultdict
import numpy as np
from score_llm import parse_grid, pass_at


def load(paths):
    S = defaultdict(dict)
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                S[r["key"]][r["sample"]] = r
    return S


def tkey(grid):
    return tuple(map(tuple, grid)) if grid else None


def task_avg(per_output):
    by = defaultdict(list)
    for k, v in per_output.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * d.mean(), 2), "sd_points": round(100 * d.std(ddof=1), 2),
            "sem_points": round(100 * sem, 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None,
            "min_detectable_points_80pct": round(100 * 2.8 * sem, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--nvarc", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="count a sample as failed when it used more completion tokens than this")
    ap.add_argument("--first-k", type=int, default=0, help="use only samples with index below this")
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    S = load(a.runs)
    per = {}
    for key, ss in S.items():
        tid, ti = key.rsplit("_", 1)
        gold = sol[tid][int(ti)]
        rec = []
        for s, r in sorted(ss.items()):
            if a.first_k and s >= a.first_k:
                continue
            g = parse_grid(r["content"])
            over = a.cap and r["completion_tokens"] > a.cap
            rec.append({"s": s, "grid": tkey(g), "ok": (g == gold) and not over, "tok": r["completion_tokens"], "parsed": g is not None and not over})
        if rec:
            per[key] = rec
    res = {"n_outputs": len(per), "n_samples": sum(len(v) for v in per.values()), "cap": a.cap, "first_k": a.first_k}
    ns = Counter(len(v) for v in per.values())
    res["samples_per_output"] = dict(sorted(ns.items()))
    for k in (1, 2, 4):
        po = {key: pass_at(len(r), sum(x["ok"] for x in r), k) for key, r in per.items() if len(r) >= 1}
        res[f"pass@{k}"] = round(100 * float(np.mean(list(task_avg(po).values()))), 2)
    res["any_correct_oracle"] = round(100 * float(np.mean(list(task_avg({k: float(any(x["ok"] for x in r)) for k, r in per.items()}).values()))), 2)

    # vote-based top-2: rank distinct parsed grids by count, then by shorter mean tokens
    def vote_top2(rec):
        groups = defaultdict(list)
        for x in rec:
            if x["grid"] is not None:
                groups[x["grid"]].append(x)
        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), np.mean([y["tok"] for y in kv[1]])))
        return [g for g, _ in ranked[:2]]
    vote_ok, vote1_ok = {}, {}
    for key, rec in per.items():
        gold = tkey(sol[key.rsplit("_", 1)[0]][int(key.rsplit("_", 1)[1])])
        top = vote_top2(rec)
        vote_ok[key] = float(gold in top)
        vote1_ok[key] = float(bool(top) and top[0] == gold)
    res["vote_top2"] = round(100 * float(np.mean(list(task_avg(vote_ok).values()))), 2)
    res["vote_top1"] = round(100 * float(np.mean(list(task_avg(vote1_ok).values()))), 2)
    p2 = task_avg({k: pass_at(len(r), sum(x["ok"] for x in r), 2) for k, r in per.items()})
    res["contrast_vote_top2_vs_random_pair"] = paired(task_avg(vote_ok), p2, "vote top-2 minus unbiased pass@2 of random pairs")

    # union with stored NVARC runs
    for p in a.nvarc:
        nv = json.load(open(p))["per_output"]
        tag = os.path.basename(p).replace("score_", "").replace(".json", "")
        nv2 = {k: float(v["kgmon"]) for k, v in nv.items() if k in per}
        nv1 = {k: float(v["kgmon_top1"]) for k, v in nv.items() if k in per}
        res[f"nvarc_{tag}_top2"] = round(100 * float(np.mean(list(task_avg(nv2).values()))), 2)
        union = {k: float(nv2[k] or vote_ok[k]) for k in nv2}
        hyb = {}
        for k in nv2:
            gold = tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
            top = vote_top2(per[k])
            first = top[0] if top else None
            hyb[k] = float(first == gold or bool(nv1[k]))
        res[f"hybrid_{tag}_llmtop1_plus_nvarctop1"] = round(100 * float(np.mean(list(task_avg(hyb).values()))), 2)
        res[f"union_oracle_{tag}"] = round(100 * float(np.mean(list(task_avg(union).values()))), 2)
        res[f"contrast_hybrid_vs_llm_vote_{tag}"] = paired(task_avg(hyb), task_avg(vote_ok), "hybrid minus LLM vote top-2")
    res["tokens_mean"] = round(float(np.mean([x["tok"] for r in per.values() for x in r])), 0)
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
