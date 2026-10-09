#!/usr/bin/env python
"""One reasoning trace per task versus one per test output, on saved code-arm results.

Every saved program (own trace program and forced finalizations) is executed on ALL test inputs of its task. The per-output design scores
output (task, j) with the programs of its own trace; the per-task design scores it with the programs of the task's first trace (j = 0).
Fixed-cap policies only. Votes: demo-passing programs weigh 2, partial passes 0.25 x pass fraction. Task-level means; contrasts are paired signed
means over tasks with sem and the smallest effect detectable at 80% power (2.8 x sem). Writes the summary through a temp file and rename.
"""
import argparse, json, math, os, subprocess, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def tkey(g):
    return tuple(map(tuple, g)) if g else None


def task_avg(d):
    by = defaultdict(list)
    for k, v in d.items():
        by[k.rsplit("_", 1)[0]].append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * float(d.mean()), 2), "sd_points": round(100 * float(d.std(ddof=1)), 2),
            "sem_points": round(100 * float(sem), 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None,
            "min_detectable_points_80pct": round(100 * 2.8 * float(sem), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--force", nargs="+", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--caps", default="16384,32768,49152,63000")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    sol = json.load(open(a.solutions))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])

    toks = {}
    flat = os.path.join(a.work, "flat.jsonl")
    ex_path = os.path.join(a.work, "flat_exec.jsonl")
    fcost, fcount = {}, defaultdict(int)
    with open(flat, "w") as f:
        for l in open(a.run):
            r = json.loads(l)
            if r.get("status") == "ok" and r["sample"] == 0:
                toks[r["key"]] = r["completion_tokens"]
                f.write(json.dumps({"status": "ok", "key": r["key"], "sample": "own", "content": r["content"]}) + "\n")
        for fi, p in enumerate(a.force):
            for l in open(p):
                r = json.loads(l)
                if "completions" not in r or r["sample"] != 0:
                    continue
                fcost[(r["key"], r["cap"])] = r["completion_tokens"]
                for j, c in enumerate(r["completions"]):
                    f.write(json.dumps({"status": "ok", "key": r["key"], "sample": f"c{r['cap']}_f{fi}_j{j}", "content": c}) + "\n")
                    fcount[(r["key"], r["cap"])] += 1
    if not os.path.exists(ex_path):
        subprocess.run([sys.executable, os.path.join(HERE, "exec_code.py"), "--challenges", a.challenges, "--runs", flat, "--out", ex_path, "--workers", str(a.workers)], check=True)
    ex = defaultdict(list)
    own = {}
    for l in open(ex_path):
        r = json.loads(l)
        sid = r["sample"]
        if sid == "own":
            own[r["key"]] = r
        else:
            cap = int(sid.split("_")[0][1:])
            ex[(r["key"], cap)].append(r)

    def cands(trace, cap):
        t = toks[trace]
        if t <= cap:
            # a trace that stopped by itself keeps its own program; one that stopped without an answer (force_end.py output passed as an extra --force file, recorded at the
            # trace length) is closed where it stopped and uses the programs sampled there
            return ([own[trace]] if trace in own else []) + ex.get((trace, t), [])
        return ex.get((trace, cap), [])

    def forced_cost(k, cap):
        return fcost.get((k, cap if toks[k] > cap else toks[k]), 0)

    def rank(cs, j):
        v = defaultdict(float)
        for c in cs:
            ps = c.get("preds")
            p = ps[j] if ps and j < len(ps) else None
            if p is None:
                continue
            v[tkey(p)] += 2.0 if c["n_pass"] == c["n_train"] else 0.25 * c["n_pass"] / max(c["n_train"], 1)
        return [g for g, _ in sorted(v.items(), key=lambda kv: -kv[1])]

    outputs = sorted(toks)
    res = {"n_outputs": len(outputs), "n_tasks": len({k.rsplit('_', 1)[0] for k in outputs}), "caps": {}}
    for cap in [int(c) for c in a.caps.split(",")]:
        rec = {}
        for design in ("per_output", "per_task"):
            t1, t2 = {}, {}
            for k in outputs:
                tid, j = k.rsplit("_", 1)
                trace = k if design == "per_output" else f"{tid}_0"
                if trace not in toks:
                    t1[k] = t2[k] = 0.0
                    continue
                r_ = rank(cands(trace, cap), int(j))
                g = gold(k)
                t1[k] = float(bool(r_) and r_[0] == g)
                t2[k] = float(g in r_[:2])
            if design == "per_output":
                used = [min(toks[k], cap) + forced_cost(k, cap) for k in outputs]
                rec["per_output_tokens_per_trace"] = round(float(np.mean(used)))
                rec["per_output_traces"] = len(outputs)
            else:
                firsts = sorted({f"{k.rsplit('_', 1)[0]}_0" for k in outputs if f"{k.rsplit('_', 1)[0]}_0" in toks})
                used = [min(toks[k], cap) + forced_cost(k, cap) for k in firsts]
                rec["per_task_tokens_per_trace"] = round(float(np.mean(used)))
                rec["per_task_traces"] = len(firsts)
            rec[design] = {"top1": round(100 * float(np.mean(list(task_avg(t1).values()))), 2), "top2": round(100 * float(np.mean(list(task_avg(t2).values()))), 2)}
            rec[f"_{design}_t1"], rec[f"_{design}_t2"] = task_avg(t1), task_avg(t2)
        rec["contrast_top1"] = paired(rec["_per_task_t1"], rec["_per_output_t1"], f"cap {cap}: per-task minus per-output, top-1")
        rec["contrast_top2"] = paired(rec["_per_task_t2"], rec["_per_output_t2"], f"cap {cap}: per-task minus per-output, top-2")
        for k in [k for k in rec if k.startswith("_")]:
            del rec[k]
        tp = rec["per_task_tokens_per_trace"]
        rec["tokens_per_second_for_240_tasks_in_42600s"] = {"one_trace_per_task": round(240 * tp / 42600), "one_trace_per_output_at_1.383_outputs_per_task": round(240 * 1.383 * rec["per_output_tokens_per_trace"] / 42600)}
        res["caps"][str(cap)] = rec
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
