#!/usr/bin/env python
"""Paired task-level contrasts for the grid-facts prompt experiments on the 120 public evaluation tasks (program arm, cap 32,768, eight forced programs, seed 3).

Population: the 120 public-eval tasks; a task scores the mean of correct_top1 over its test outputs. No-summary controls are g4 and g5 (two independent runs at a 63K cap scored
at a fixed 32K cap, results/two_runs_per_task.json written by analyze_two_runs.py) and p0 (a direct run at cap 32,768 with solver2_v6 and no prompt option). Summary arms are ps (summary 1), psc (summary 1 through
the chat API), ps2 (summary 2, stopped at 48 tasks), psi (summary 1 plus a picture of the examples, stopped at 55) and psw (stopped after one task).
Every contrast is the signed per-task difference averaged over the common tasks; the report gives mean, sd, sem, mean/sem and the smallest effect 80% power could have seen (2.8 x sem).
The pre-registered contrast is ps minus mean(g4, g5); every contrast that includes p0 or psc was added after seeing the data and is labelled post hoc.
Writes results/prompt_exps_analysis.json atomically.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results") if os.path.isdir(os.path.join(HERE, "results")) else os.path.join(HERE, "..", "results")
RAW = os.path.join(RES, "raw", "box_2026-10-10")
FRONTIER = [(24.6e3, 13.75), (41.8e3, 27.7), (58.1e3, 37.5), (70.2e3, 40.2)]


def frontier(tokens):
    if tokens <= FRONTIER[0][0]:
        return FRONTIER[0][1] * tokens / FRONTIER[0][0]
    for (x0, y0), (x1, y1) in zip(FRONTIER, FRONTIER[1:]):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return FRONTIER[-1][1]


def load_done(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "done":
            d[r["task"]] = r
    return d


def contrast(a, b):
    ts = sorted(set(a) & set(b))
    d = 100 * np.array([a[t] - b[t] for t in ts])
    if len(d) < 3:
        return {"n": len(d)}
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return {"n": len(d), "mean": round(d.mean(), 2), "sd": round(d.std(ddof=1), 2), "sem": round(sem, 2),
            "mean_over_sem": round(d.mean() / sem, 2), "mde80": round(2.8 * sem, 2)}


def mean_of(*runs):
    ts = sorted(set.intersection(*[set(r) for r in runs]))
    return {t: float(np.mean([r[t] for r in runs])) for t in ts}


def main():
    res = RES
    two = json.load(open(os.path.join(res, "two_runs_per_task.json")))
    runs, info = {"g4": two["g4_cap32768"]["top1"], "g5": two["g5_cap32768"]["top1"]}, {}
    for n in ("p0", "ps", "psc", "ps2", "psi", "psw", "p2", "r1ps", "r1p0"):
        full = load_done(os.path.join(RAW, n + ".jsonl"))
        runs[n] = {t: float(np.mean(r["correct_top1"])) for t, r in full.items()}
        tok = float(np.mean([r["ntok"] + r["forced_tokens"] for r in full.values()]))
        info[n] = {"n_tasks": len(full), "top1": round(100 * np.mean(list(runs[n].values())), 2), "tokens_per_task": round(tok),
                   "share_verified_program": round(float(np.mean([r["n_verified"] > 0 for r in full.values()])), 3),
                   "finish_stop": int(sum(r["finish"] == "stop" for r in full.values())),
                   "frontier_top1_at_tokens": round(frontier(tok), 2)}
    for n in ("g4", "g5"):
        info[n] = {"n_tasks": len(runs[n]), "top1": round(100 * np.mean(list(runs[n].values())), 2)}
    ctrl_all = ("g4", "g5", "p0", "r1p0")
    ctrl_ids = ("g4", "g5")
    arms = {}
    pre = mean_of(runs["g4"], runs["g5"])
    pool = mean_of(runs["g4"], runs["g5"], runs["p0"])
    arms["ps minus mean(g4,g5) [pre-registered]"] = contrast(runs["ps"], pre)
    for c in ctrl_all:
        arms[f"ps minus {c}"] = contrast(runs["ps"], runs[c])
    arms["ps minus mean(g4,g5,p0) [post hoc]"] = contrast(runs["ps"], pool)
    arms["psc minus mean(g4,g5,p0) [post hoc]"] = contrast(runs["psc"], pool)
    arms["mean(ps,psc) minus mean(g4,g5,p0) [post hoc]"] = contrast(mean_of(runs["ps"], runs["psc"]), pool)
    arms["psc minus ps [post hoc]"] = contrast(runs["psc"], runs["ps"])
    arms["r1ps minus r1p0 [pre-registered replicate, seed 4]"] = contrast(runs["r1ps"], runs["r1p0"])
    two_ps, two_p0 = mean_of(runs["ps"], runs["r1ps"]), mean_of(runs["p0"], runs["r1p0"])
    arms["mean(ps,r1ps) minus mean(p0,r1p0) [pre-registered, pooled over seeds]"] = contrast(two_ps, two_p0)
    arms["mean(ps,r1ps) minus mean(g4,g5,p0,r1p0) [pre-registered, pooled over seeds]"] = contrast(two_ps, mean_of(runs["g4"], runs["g5"], runs["p0"], runs["r1p0"]))
    arms["r1p0 minus p0 [same configuration, two seeds]"] = contrast(runs["r1p0"], runs["p0"])
    arms["r1ps minus ps [same configuration, two seeds]"] = contrast(runs["r1ps"], runs["ps"])
    arms["p0 minus mean(g4,g5) [control drift, post hoc]"] = contrast(runs["p0"], pre)
    arms["p0 minus g4 [control drift, post hoc]"] = contrast(runs["p0"], runs["g4"])
    arms["p0 minus g5 [control drift, post hoc]"] = contrast(runs["p0"], runs["g5"])
    arms["g5 minus g4 [run to run]"] = contrast(runs["g5"], runs["g4"])
    arms["ps2 minus ps (first 48 tasks, stopped)"] = contrast(runs["ps2"], runs["ps"])
    arms["psi minus ps (first 55 tasks, stopped)"] = contrast(runs["psi"], runs["ps"])
    for n, sub in (("a24", "kaggle_final_a24"), ("rep1", "kaggle_final_replica")):
        full = load_done(os.path.join(res, sub, "solver_log.jsonl"))
        runs[n] = {t: float(np.mean(r["correct_top1"])) for t, r in full.items()}
        tok = float(np.mean([r["ntok"] + r["forced_tokens"] for r in full.values()]))
        info[n] = {"n_tasks": len(full), "top1": round(100 * np.mean(list(runs[n].values())), 2), "tokens_per_task": round(tok),
                   "share_verified_program": round(float(np.mean([r["n_verified"] > 0 for r in full.values()])), 3),
                   "finish_stop": int(sum(r["finish"] == "stop" for r in full.values())), "frontier_top1_at_tokens": round(frontier(tok), 2),
                   "where": "Kaggle 4xL4, commit replica at 163.5 s per task"}
    arms["a24 (Kaggle, staggered controller, summary 1) minus ps (box) [post hoc, different hardware, cap policy and context]"] = contrast(runs["a24"], runs["ps"])
    arms["a24 minus mean(g4,g5,p0) [post hoc, different hardware and cap policy]"] = contrast(runs["a24"], pool)
    arms["a24 minus rep1 (Kaggle, old controller, no summary) [post hoc, controller and prompt change together]"] = contrast(runs["a24"], runs["rep1"])
    ctrl_tops = [info[c]["top1"] for c in ctrl_all]
    out = {"population": "120 public-eval tasks, program arm, cap 32,768, 8 forced programs, task-level top-1 (mean over test outputs)",
           "runs": info, "contrasts": arms,
           "no_summary_run_to_run": {"runs": list(ctrl_all), "top1": ctrl_tops, "mean": round(float(np.mean(ctrl_tops)), 2),
                                     "sd": round(float(np.std(ctrl_tops, ddof=1)), 2)},
           "decision_rule": "adopt summary 1 if ps minus mean(g4,g5) >= +5 (PREREG 2026-10-10 02:40 UTC); summary 2 replaces it only if >= +4 over summary 1 with at most 5% more tokens"}
    path = os.path.join(res, "prompt_exps_analysis.json")
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
