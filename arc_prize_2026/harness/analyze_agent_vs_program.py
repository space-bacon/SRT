#!/usr/bin/env python
"""Tool-integrated agent arm against the program arm on the same tasks, from the run logs on disk (no new compute).

Population: the 120 public ARC-AGI-2 evaluation tasks, one rollout per task, task-level top-1 (mean of correct_top1 over a task's test outputs). Agent runs: agent_b16k and agent_b32k (arc_agent.py
default flags), agent_v5_b16k (--auto-summary --auto-check --all-tests --n-forced 4) and agent_v6_b32k (the same bundle at a 32,768 budget, stopped at 60 tasks). Program-arm runs: g4 and g5
(two independent 63K-cap runs scored at the 16K, 32K, 49K and 63K caps, results/two_runs_per_task.json), p0 and r1p0 (cap 32,768, no facts in the prompt), ps, r1ps and psc (cap 32,768, grid facts).

Contrast 1 (matched frontier). The program-arm frontier is the mean of g4 and g5 at the four caps (24.6K, 41.8K, 58.1K and 70.2K decoded tokens per task), restricted to the tasks the agent run
finished, interpolated linearly at the agent run's mean decoded tokens per task (generated + forced). Per task the difference agent minus frontier is taken, so the sem is paired over tasks; the
frontier's own run-to-run sd (about 3 points) is not in it. Contrast 2 compares the agent run with each program-arm run at a 32K cap on the same tasks and with the mean of the two groups (no facts:
g4, g5 at the 32K cap, p0, r1p0; facts: ps, r1ps, psc). Every contrast reports mean, sd, sem, mean/sem and the smallest effect 80 percent power could see (2.8 x sem); the sign is kept.

Unique solves. A task is solved by a run when its task score is above 0. For each run the number of tasks it solves that no other run in the set solves is counted. If the agent solved
tasks by a different route its count would exceed what a program-arm run of the same strength has against the others. Writes results/agent_vs_program.json atomically.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results") if os.path.isdir(os.path.join(HERE, "results")) else os.path.join(HERE, "..", "results")
RAW = os.path.join(RES, "raw", "box_2026-10-10")
CAPS = ("cap16384", "cap32768", "cap49152", "cap63000")
CAP_TOKENS = (24.6e3, 41.8e3, 58.1e3, 70.2e3)
# Kaggle agent smoke: 2.343 prefill tokens per decoded token against 1.179 for the program arm (a24); prefill took 20.4 percent of the program arm's active time
PREFILL_SHARE_PROGRAM, PREFILL_P_PROGRAM, PREFILL_P_AGENT = 0.204, 1.179, 2.343


def load_done(path, agent=False):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "done" and r.get("rollout", 0) == 0:
            out[r["task"]] = r
    return out


def contrast(d):
    d = 100 * np.asarray(d, dtype=float)
    if len(d) < 3:
        return {"n": int(len(d))}
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return {"n": int(len(d)), "mean": round(float(d.mean()), 2), "sd": round(float(d.std(ddof=1)), 2), "sem": round(float(sem), 2),
            "mean_over_sem": round(float(d.mean() / sem), 2), "mde80": round(float(2.8 * sem), 2)}


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0] * x / xs[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def main():
    two = json.load(open(os.path.join(RES, "two_runs_per_task.json")))
    frontier_tasks = {c: {t: (two[f"g4_{c}"]["top1"][t] + two[f"g5_{c}"]["top1"][t]) / 2 for t in two[f"g4_{c}"]["top1"]} for c in CAPS}
    prog = {"g4": two["g4_cap32768"]["top1"], "g5": two["g5_cap32768"]["top1"]}
    for n in ("p0", "r1p0", "ps", "r1ps", "psc"):
        prog[n] = {t: float(np.mean(r["correct_top1"])) for t, r in load_done(os.path.join(RAW, n + ".jsonl")).items()}
    nofacts, facts = ("g4", "g5", "p0", "r1p0"), ("ps", "r1ps", "psc")
    out = {"population": "120 public ARC-AGI-2 evaluation tasks; agent runs have one rollout per task; task-level top-1", "runs": {}}
    # prefill-time factor of the agent relative to the program arm (Kaggle smoke counters); an upper bound for the v5 bundle, whose harness makes the first call itself
    c_over_d = PREFILL_SHARE_PROGRAM / (1 - PREFILL_SHARE_PROGRAM) / PREFILL_P_PROGRAM
    time_factor = (1 + PREFILL_P_AGENT * c_over_d) / (1 + PREFILL_P_PROGRAM * c_over_d)
    out["time_factor_agent_per_decoded_token_on_kaggle"] = round(time_factor, 3)
    solved_sets = {}
    for name in ("agent_b16k", "agent_b32k", "agent_v5_b16k", "agent_v6_b32k"):
        done = load_done(os.path.join(RAW, name + ".jsonl"))
        tasks = sorted(done)
        a = {t: float(np.mean(done[t]["correct_top1"])) for t in tasks}
        toks = float(np.mean([done[t]["gen_tokens"] + done[t]["forced_tokens"] for t in tasks]))
        fr_by_cap = [float(np.mean([frontier_tasks[c][t] for t in tasks])) for c in CAPS]
        x = toks
        per_task_fr = [interp(x, CAP_TOKENS, [frontier_tasks[c][t] for c in CAPS]) for t in tasks]
        x_time = toks * time_factor
        per_task_fr_time = [interp(x_time, CAP_TOKENS, [frontier_tasks[c][t] for c in CAPS]) for t in tasks]
        r = {"n_tasks": len(tasks), "top1": round(100 * float(np.mean(list(a.values()))), 2), "decoded_tokens_per_task": round(toks),
             "frontier_on_these_tasks_by_cap": [round(100 * v, 2) for v in fr_by_cap],
             "agent_minus_matched_frontier_equal_tokens": contrast([a[t] - f for t, f in zip(tasks, per_task_fr)]),
             "agent_minus_matched_frontier_equal_kaggle_time": contrast([a[t] - f for t, f in zip(tasks, per_task_fr_time)]),
             "vs_program_runs_at_32K_cap_same_tasks": {}}
        for p in nofacts + facts:
            common = [t for t in tasks if t in prog[p]]
            r["vs_program_runs_at_32K_cap_same_tasks"][p] = {"program_top1": round(100 * float(np.mean([prog[p][t] for t in common])), 2),
                                                             "agent_minus_program": contrast([a[t] - prog[p][t] for t in common])}
        for lab, grp in (("no_facts_mean", nofacts), ("facts_mean", facts)):
            common = [t for t in tasks if all(t in prog[p] for p in grp)]
            r[f"vs_{lab}_32K_cap"] = contrast([a[t] - np.mean([prog[p][t] for p in grp]) for t in common])
        out["runs"][name] = r
        solved_sets[name] = {t for t in tasks if a[t] > 0}
    # unique solves on the tasks of each agent run
    out["unique_solves"] = {}
    for name in ("agent_v6_b32k", "agent_b32k", "agent_v5_b16k", "agent_b16k"):
        tasks = sorted(load_done(os.path.join(RAW, name + ".jsonl")))
        agent = {t: float(np.mean(load_done(os.path.join(RAW, name + ".jsonl"))[t]["correct_top1"])) for t in tasks}
        sets = {name: {t for t in tasks if agent[t] > 0}}
        for p in nofacts + facts:
            sets[p] = {t for t in tasks if prog[p].get(t, 0) > 0}
        row = {}
        for k, s in sets.items():
            others = set().union(*[v for kk, v in sets.items() if kk != k])
            row[k] = {"solved": len(s), "unique": len(s - others)}
        prog_union = set().union(*[sets[p] for p in nofacts + facts])
        out["unique_solves"][name] = {"n_tasks": len(tasks), "per_run": row, "union_of_program_runs": len(prog_union),
                                      "agent_solved_not_in_any_program_run": sorted(sets[name] - prog_union),
                                      "union_with_agent": len(prog_union | sets[name])}
    path = os.path.join(RES, "agent_vs_program.json")
    json.dump(out, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
