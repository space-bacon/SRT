#!/usr/bin/env python
"""Summarize a Kaggle replica or rerun log of solver2.py (solver_log.jsonl): score, token use, caps and the controller's throughput estimate.

Score = task-level top-1 and top-2 (mean over a task's outputs, then over tasks) when the log carries correct_top1 / correct_top2 (runs with solutions). A 95% interval comes from a task
bootstrap (2,000 resamples, seed 0). The expected score at the observed mean decoded tokens per task is read from the xhigh frontier measured on the same 120 tasks (mean of runs g4 and
g5, per-output design: 24.7K tokens 13.75, 42.0K 27.75, 58.0K 37.5, 70.2K 40.2, linear in between); the difference between the replica and that expectation is reported with the
replica's bootstrap sem, which does not include the frontier's own run-to-run sd of about 3 points. Writes through a temp file and a rename.
"""
import json, os, sys
from collections import Counter
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else "solver_log.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "replica_analysis.json"
FRONTIER = [(24.7e3, 13.75), (42.0e3, 27.75), (58.0e3, 37.5), (70.2e3, 40.2)]


def frontier(tokens):
    pts = FRONTIER
    if tokens <= pts[0][0]:
        return pts[0][1] * tokens / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return pts[-1][1]


def main():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    by = {}
    for r in recs:
        by.setdefault(r["event"], []).append(r)
    done = by.get("done", [])
    res = {"events": {k: len(v) for k, v in by.items()}, "n_done": len(done), "n_skipped": len(by.get("skipped", []))}
    if not done:
        print(json.dumps(res, indent=1))
        return
    caps = np.array([r["cap"] for r in done])
    toks = np.array([r["ntok"] + r.get("forced_tokens", 0) for r in done], dtype=float)
    res["cap"] = {"mean": round(float(caps.mean())), "q10": int(np.quantile(caps, .1)), "median": int(np.median(caps)), "q90": int(np.quantile(caps, .9)), "min": int(caps.min()), "max": int(caps.max())}
    res["decoded_tokens_per_task"] = {"mean": round(float(toks.mean())), "reasoning_mean": round(float(np.mean([r["ntok"] for r in done]))), "forced_mean": round(float(np.mean([r.get("forced_tokens", 0) for r in done])))}
    res["finish"] = dict(Counter(r["finish"] for r in done))
    res["n_verified_ge1_share"] = round(float(np.mean([r["n_verified"] > 0 for r in done])), 3)
    starts = by.get("start", [])
    if starts:
        chunks = [starts[i * len(starts) // 10:(i + 1) * len(starts) // 10] for i in range(10)]
        chunks = [c for c in chunks if c]
        res["controller"] = {"first_thr": starts[0]["thr"], "last_thr": starts[-1]["thr"], "median_thr": float(np.median([s["thr"] for s in starts])),
                             "thr_by_decile_of_starts": [round(float(np.mean([s["thr"] for s in c])), 1) for c in chunks],
                             "cap_by_decile_of_starts": [round(float(np.mean([s["cap"] for s in c]))) for c in chunks]}
    res["time_s"] = {"first_done": min(r["t"] for r in done), "last_event": max(r["t"] for r in recs)}
    if "correct_top1" in done[0]:
        t1 = np.array([np.mean(r["correct_top1"]) for r in done])
        t2 = np.array([np.mean(r["correct_top2"]) for r in done])
        rng = np.random.default_rng(0)
        boot = lambda x: [round(100 * float(np.quantile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], q)), 2) for q in (0.025, 0.975)]
        sem = float(np.std([t1[rng.integers(0, len(t1), len(t1))].mean() for _ in range(2000)]))
        exp = frontier(float(toks.mean()))
        res["score"] = {"top1": round(100 * float(t1.mean()), 2), "top1_ci95": boot(t1), "top2": round(100 * float(t2.mean()), 2), "top2_ci95": boot(t2), "n_tasks": len(done),
                        "frontier_expectation_top1_at_mean_tokens": round(exp, 2), "top1_minus_frontier": round(100 * float(t1.mean()) - exp, 2), "bootstrap_sem_points": round(100 * sem, 2)}
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
