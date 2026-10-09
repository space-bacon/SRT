#!/usr/bin/env python
"""Analysis of the execution-feedback experiment (fb_experiment.py output) against plain continuation to the same total reasoning budget.

Arms, per saved trace that was still reasoning at the 16K checkpoint and had no program passing every demo:
  control  the same trace continued without feedback to 32K (programs sampled at 32K, from the saved g4 / g5 runs)
  A        feedback inside the thinking block, 16K more tokens          B  feedback as a fresh chat turn, 16K more tokens
  S        as A, with the think-end token suppressed (logit bias -100) for the whole continuation
Reported: task-level top-1 / top-2 on the subset, paired signed differences over tasks (mean, sd, sem, mean/sem, smallest effect at 80% power = 2.8 x sem),
the whole-benchmark score of each policy (outputs verified at 16K stop there, the rest use the arm), and token use. Writes through a temp file and rename.
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FB = (sys.argv[1] if len(sys.argv) > 1 else f"{R}/raw/fb1/fb_main.jsonl,{R}/raw/fb1/fb_S.jsonl").split(",")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{R}/fb_analysis.json"
SOL = "/tmp/arcdata/arc-agi_evaluation_solutions.json"
CAP = 32768


def tkey(g):
    return tuple(map(tuple, g)) if g else None


sol = json.load(open(SOL))
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


def control_scores():
    """key -> (top1, top2) at cap 32768 for g4 (per_task_eval work files) and g5 (artifacts)."""
    out = {}
    g4 = defaultdict(list)
    own4, toks4 = {}, {}
    for l in open(f"{R}/raw/g4full/g4_code.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] == 0:
            toks4[r["key"]] = r["completion_tokens"]
    for l in open("/tmp/arcllm/pertask/flat_exec.jsonl"):
        r = json.loads(l)
        if r["sample"] == "own":
            own4[r["key"]] = r
        elif r["sample"].startswith(f"c{CAP}_"):
            g4[r["key"]].append(r)
    for k, t in toks4.items():
        cs = [own4[k]] if t <= CAP and k in own4 else ([] if t <= CAP else g4.get(k, []))
        rk = rank(cs, int(k.rsplit("_", 1)[1]))
        out[("g4", k)] = (float(bool(rk) and rk[0] == gold(k)), float(gold(k) in rk[:2]))
    A = f"{R}/raw/g5full/artifacts"
    toks5, own5, g5 = {}, {}, defaultdict(list)
    for l in open(f"{A}/g5_code.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] == 0:
            toks5[r["key"]] = r["completion_tokens"]
    for l in open(f"{A}/g5_exec.jsonl"):
        r = json.loads(l)
        own5[r["key"]] = r
    for l in open(f"{A}/g5_force_exec.jsonl"):
        r = json.loads(l)
        _, cap, _j = str(r["sample"]).split("_")
        if int(cap[1:]) == CAP:
            g5[r["key"]].append(r)
    for k, t in toks5.items():
        cs = [own5[k]] if t <= CAP and k in own5 else ([] if t <= CAP else g5.get(k, []))
        rk = rank(cs, int(k.rsplit("_", 1)[1]))
        out[("g5", k)] = (float(bool(rk) and rk[0] == gold(k)), float(gold(k) in rk[:2]))
    return out


def task_mean(vals):
    """vals: {(run, key): x} -> {task: mean over runs of the within-run mean over that task's outputs}"""
    by = defaultdict(lambda: defaultdict(list))
    for (run, k), x in vals.items():
        by[k.rsplit("_", 1)[0]][run].append(x)
    return {t: float(np.mean([np.mean(v) for v in runs.values()])) for t, runs in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    if len(d) < 2:
        return {"contrast": label, "n_tasks": len(d)}
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * float(d.mean()), 2), "sd_points": round(100 * float(d.std(ddof=1)), 2),
            "sem_points": round(100 * float(sem), 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None, "min_detectable_points_80pct": round(100 * 2.8 * float(sem), 2)}


def main():
    prep, units = {}, defaultdict(dict)
    for path in FB:
        if not os.path.exists(path):
            continue
        for l in open(path):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("kind") == "prep":
                prep[(r["run"], r["key"])] = r
            elif r.get("kind") == "unit":
                units[r["variant"]][(r["run"], r["key"])] = r
    ctl = control_scores()
    res = {"n_prep": len(prep), "n_verified16": sum(1 for p in prep.values() if p["n_verified16"] > 0), "variants": {}}
    variants = [v for v in sorted(units) if len(units[v]) >= 10]
    # every variant is scored on the outputs it finished, against the control on the same outputs; a run that was stopped early is a random subset because units were queued in shuffled order
    for v in variants:
        have = set(units[v])
        u = units[v]
        ctl1 = {k: ctl[k][0] for k in have}
        ctl2 = {k: ctl[k][1] for k in have}
        s1 = {k: float(u[k]["top1"]) for k in have}
        s2 = {k: float(u[k]["top2"]) for k in have}
        res["variants"][v] = {
            "n_outputs": len(have), "n_tasks": len(task_mean(s1)),
            "top1": round(100 * float(np.mean(list(task_mean(s1).values()))), 2), "top2": round(100 * float(np.mean(list(task_mean(s2).values()))), 2),
            "control_top1": round(100 * float(np.mean(list(task_mean(ctl1).values()))), 2), "control_top2": round(100 * float(np.mean(list(task_mean(ctl2).values()))), 2),
            "mean_cont_tokens": round(float(np.mean([u[k]["cont_tokens"] for k in have]))), "mean_forced_tokens": round(float(np.mean([u[k]["forced_tokens"] for k in have]))),
            "own_program_fraction": round(float(np.mean([u[k]["own"] for k in have])), 3), "any_verified_fraction": round(float(np.mean([u[k]["n_verified"] > 0 for k in have])), 3),
            "contrast_top1_vs_control": paired(task_mean(s1), task_mean(ctl1), f"{v} minus control, top-1, outputs the arm finished"),
            "contrast_top2_vs_control": paired(task_mean(s2), task_mean(ctl2), f"{v} minus control, top-2, outputs the arm finished"),
        }
        by_pass = defaultdict(lambda: [[], []])
        for k in have:
            b = by_pass["p_star_0" if prep[k].get("p_star_n_pass", 0) == 0 else "p_star_1plus"]
            b[0].append(float(u[k]["top1"]))
            b[1].append(ctl[k][0])
        res["variants"][v]["by_p_star"] = {name: {"n": len(x[0]), "arm_top1": round(100 * float(np.mean(x[0])), 1), "control_top1": round(100 * float(np.mean(x[1])), 1)} for name, x in by_pass.items()}
        # sub-benchmark: outputs verified at the 16K checkpoint stop there; the others use the arm. Control scored on the same outputs.
        pop = [k for k, p in prep.items() if p["n_verified16"] > 0 or k in u]
        arm1, arm2, c1, c2 = {}, {}, {}, {}
        for k in pop:
            p = prep[k]
            if p["n_verified16"] > 0:
                arm1[k] = c1[k] = float(p["top1_16"])
                arm2[k] = c2[k] = float(p["top2_16"])
            else:
                arm1[k], arm2[k] = float(u[k]["top1"]), float(u[k]["top2"])
                c1[k], c2[k] = ctl[k]
        res["variants"][v]["gated_sub_benchmark"] = {
            "n_outputs": len(pop), "arm_top1": round(100 * float(np.mean(list(task_mean(arm1).values()))), 2), "control_top1": round(100 * float(np.mean(list(task_mean(c1).values()))), 2),
            "contrast_top1": paired(task_mean(arm1), task_mean(c1), f"{v} minus control, top-1, gated, outputs verified at 16K or finished by the arm")}
    if "A" in units and "B" in units:
        common = sorted(set(units["A"]) & set(units["B"]))
        sa = {k: float(units["A"][k]["top1"]) for k in common}
        sb = {k: float(units["B"][k]["top1"]) for k in common}
        res["contrast_a_vs_b_top1"] = paired(task_mean(sa), task_mean(sb), f"A minus B, top-1, {len(common)} outputs both finished")
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
