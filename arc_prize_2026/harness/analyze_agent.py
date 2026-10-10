#!/usr/bin/env python
"""Score a run of arc_agent.py (tool-integrated reasoning) on the 120 public evaluation tasks and place it on the xhigh program-arm frontier.

Reads the `done` records of the agent log (one per task and rollout; correct_top1 / correct_top2 per test output). Task score = mean over the task's outputs, then over tasks. Decoded tokens per
task = gen_tokens (reasoning, forced tool calls and tool-call turns) + forced_tokens (the final eight programs when the budget ended first). The program-arm frontier (mean of runs g4 and g5,
per-output design, same tasks) is 13.75 at 24.6K tokens, 27.7 at 41.8K, 37.5 at 58.1K and 40.2 at 70.2K, linear in between; the difference is reported with a task-bootstrap sem of the agent score,
which leaves out the frontier's own run-to-run sd of about 3 points. When NVARC runs s0 and s1 are given, also the 2-attempt hybrid (agent top-1 plus NVARC top-1) per output. Atomic write.
"""
import argparse, json, math, os
from collections import Counter, defaultdict
import numpy as np

FRONTIER = [(24.6e3, 13.75), (41.8e3, 27.7), (58.1e3, 37.5), (70.2e3, 40.2)]


def frontier(tokens):
    pts = FRONTIER
    if tokens <= pts[0][0]:
        return pts[0][1] * tokens / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="")
    ap.add_argument("--nvarc", nargs="*", default=[], help="score_s0.json score_s1.json (per_output with kgmon_top1)")
    a = ap.parse_args()
    done = {}
    for l in open(a.log):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("event") == "done" and r.get("rollout", 0) == 0:
            # solver3.py (Kaggle) logs ntok and finish where arc_agent.py logs gen_tokens and status
            r.setdefault("gen_tokens", r.get("ntok", 0))
            r.setdefault("status", {"turn_cut": "turn_cut"}.get(r.get("finish"), r.get("finish", "budget")))
            done[r["task"]] = r
    res = {"n_tasks_done": len(done)}
    if not done:
        print(json.dumps(res))
        return
    t1 = np.array([np.mean(r["correct_top1"]) for r in done.values()])
    t2 = np.array([np.mean(r["correct_top2"]) for r in done.values()])
    toks = np.array([r["gen_tokens"] + r["forced_tokens"] for r in done.values()], dtype=float)
    rng = np.random.default_rng(0)
    boots = np.array([t1[rng.integers(0, len(t1), len(t1))].mean() for _ in range(2000)])
    res.update({"top1": round(100 * float(t1.mean()), 2), "top2": round(100 * float(t2.mean()), 2), "top1_ci95": [round(100 * float(np.quantile(boots, q)), 2) for q in (0.025, 0.975)],
                "bootstrap_sem_points": round(100 * float(boots.std()), 2)})
    res["tokens_per_task"] = {"mean": round(float(toks.mean())), "median": round(float(np.median(toks))), "gen_mean": round(float(np.mean([r["gen_tokens"] for r in done.values()]))),
                              "forced_mean": round(float(np.mean([r["forced_tokens"] for r in done.values()])))}
    res["status"] = dict(Counter(r["status"] for r in done.values()))
    res["turns_mean"] = round(float(np.mean([r["turns"] for r in done.values()])), 1)
    res["calls_mean"] = round(float(np.mean([r["calls"] for r in done.values()])), 1)
    res["share_with_verified_program"] = round(float(np.mean([r["n_verified"] > 0 for r in done.values()])), 3)
    ans = [r for r in done.values() if r["status"] == "answered"]
    if ans:
        res["answered"] = {"n": len(ans), "mean_tokens": round(float(np.mean([r["gen_tokens"] for r in ans]))), "top1": round(100 * float(np.mean([np.mean(r["correct_top1"]) for r in ans])), 2)}
    rest = [r for r in done.values() if r["status"] != "answered"]
    if rest:
        res["budget_ended"] = {"n": len(rest), "top1": round(100 * float(np.mean([np.mean(r["correct_top1"]) for r in rest])), 2)}
    exp = frontier(float(toks.mean()))
    res["program_arm_frontier_top1_at_mean_tokens"] = round(exp, 2)
    res["top1_minus_frontier"] = round(100 * float(t1.mean()) - exp, 2)
    # token percentiles of answered tasks (how early the tool loop finishes)
    res["gen_tokens_quartiles"] = [int(np.quantile([r["gen_tokens"] for r in done.values()], q)) for q in (0.25, 0.5, 0.75, 0.9)]
    for p in a.nvarc:
        nv = json.load(open(p))["per_output"]
        tag = os.path.basename(p).replace("score_", "").replace(".json", "")
        hyb, nvs, ag = {}, {}, {}
        for tid, r in done.items():
            for j, c in enumerate(r["correct_top1"]):
                k = f"{tid}_{j}"
                if k in nv:
                    ag[k] = float(c)
                    nvs[k] = float(nv[k]["kgmon_top1"])
                    hyb[k] = float(bool(c) or bool(nv[k]["kgmon_top1"]))
        by = lambda d: round(100 * float(np.mean([np.mean([v for kk, v in d.items() if kk.rsplit("_", 1)[0] == t]) for t in {kk.rsplit("_", 1)[0] for kk in d}])), 2)
        res[f"hybrid_{tag}"] = {"agent_top1": by(ag), "nvarc_top1": by(nvs), "agent_top1_plus_nvarc_top1": by(hyb), "n_outputs": len(hyb)}
    if a.out:
        tmp = a.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
