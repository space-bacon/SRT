#!/usr/bin/env python
"""Combine a verified LLM policy with stored NVARC-lineage runs (s0, s1) on the same public evaluation outputs.

Reads the per-output dump of analyze_gated.py. Reports task-level means for: LLM alone (top-1 and top-2), NVARC alone (top-1 and top-2),
LLM top-1 plus NVARC top-1, and the verified gate (use LLM attempts when its program passed every demo, otherwise NVARC top-2).
Contrasts against the better single method are paired signed means over tasks with sem and the 80%-power minimum detectable effect.
"""
import argparse, json, math
from collections import defaultdict
import numpy as np


def task_avg(d):
    by = defaultdict(list)
    for k, v in d.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * d.mean(), 2), "sd_points": round(100 * d.std(ddof=1), 2),
            "sem_points": round(100 * sem, 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None,
            "min_detectable_points_80pct": round(100 * 2.8 * sem, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-output", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--nvarc", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = json.load(open(a.per_output))[a.policy]
    res = {"policy": a.policy, "n_outputs": len(llm)}
    m = lambda d: round(100 * float(np.mean(list(task_avg(d).values()))), 2)
    res["llm_top1"] = m({k: v["top1"] for k, v in llm.items()})
    res["llm_top2"] = m({k: v["top2"] for k, v in llm.items()})
    res["llm_verified_fraction"] = round(float(np.mean([v["verified"] for v in llm.values()])), 3)
    res["llm_mean_tokens"] = round(float(np.mean([v["tokens"] for v in llm.values()])), 0)
    for p in a.nvarc:
        nv = json.load(open(p))["per_output"]
        tag = p.split("score_")[-1].replace(".json", "")
        ks = [k for k in llm if k in nv]
        n1 = {k: float(nv[k]["kgmon_top1"]) for k in ks}
        n2 = {k: float(nv[k]["kgmon"]) for k in ks}
        l1 = {k: llm[k]["top1"] for k in ks}
        l2 = {k: llm[k]["top2"] for k in ks}
        hyb = {k: float(bool(l1[k]) or bool(n1[k])) for k in ks}
        gate = {k: (l2[k] if llm[k]["verified"] else n2[k]) for k in ks}
        gate2 = {k: (float(bool(l1[k]) or bool(n1[k])) if llm[k]["verified"] else float(bool(n2[k]) or bool(l1[k]))) for k in ks}
        union = {k: float(bool(l2[k]) or bool(n2[k])) for k in ks}
        res[tag] = {"nvarc_top1": m(n1), "nvarc_top2": m(n2), "llm_top1_plus_nvarc_top1": m(hyb), "verified_gate": m(gate),
                    "gate_llm_verified_else_nvarc_with_llm_top1": m(gate2), "oracle_union_top2s": m(union),
                    "contrasts": [paired(task_avg(hyb), task_avg(l2), "LLM top1 + NVARC top1 minus LLM top2"),
                                  paired(task_avg(hyb), task_avg(n2), "LLM top1 + NVARC top1 minus NVARC top2"),
                                  paired(task_avg(gate2), task_avg(l2), "verified gate (LLM top1 + NVARC top1 if verified, else NVARC top2 + LLM top1) minus LLM top2")]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
