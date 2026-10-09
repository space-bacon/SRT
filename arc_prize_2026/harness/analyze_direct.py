#!/usr/bin/env python
"""Score 'program top-1 plus a direct answer from the same truncated reasoning' against the program arm's own top-2, on the same outputs (direct_from_trace.py output).

For every saved trace longer than a cap, the program-arm candidates at that cap (own program or eight forced programs, demo-verified voting) are ranked from the artifacts
already on disk. Policies per cap, per output:
  base          the program arm's two heaviest distinct grids
  direct_n      attempt 1 is the program top-1 (or the top direct grid if there is no program), attempt 2 the most frequent grid among the first n direct samples that
                differs from attempt 1, then the program's second grid; n = 1, 2, 4
Outputs that finished before the cap keep the base policy. The score of an output is 1 if the gold grid is among its two attempts; a task scores the mean over its outputs,
averaged over the two runs where both have the task. Contrasts are paired over tasks: signed mean difference, sd, sem, mean/sem, smallest effect at 80% power (2.8 x sem).
Writes through a temp file and rename.
"""
import json, math, os, sys
from collections import Counter, defaultdict
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DIRECT = sys.argv[1] if len(sys.argv) > 1 else f"{R}/raw/fb1/direct_main.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{R}/direct_analysis.json"
SOL = "/tmp/arcdata/arc-agi_evaluation_solutions.json"
TABLES = "/tmp/arcllm/fbdata"
CAPS = [32768, 49152]


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


def candidates(cap):
    """(run, key) -> ranked grids of the program arm at this cap (own program if the trace ended before the cap)."""
    out = {}
    own4, g4 = {}, defaultdict(list)
    for l in open("/tmp/arcllm/pertask/flat_exec.jsonl"):
        r = json.loads(l)
        if r["sample"] == "own":
            own4[r["key"]] = r
        elif r["sample"].startswith(f"c{cap}_"):
            g4[r["key"]].append(r)
    A = f"{R}/raw/g5full/artifacts"
    own5, g5 = {}, defaultdict(list)
    for l in open(f"{A}/g5_exec.jsonl"):
        r = json.loads(l)
        own5[r["key"]] = r
    for l in open(f"{A}/g5_force_exec.jsonl"):
        r = json.loads(l)
        _, c, _j = str(r["sample"]).split("_")
        if int(c[1:]) == cap:
            g5[r["key"]].append(r)
    for run, own, forced in (("g4", own4, g4), ("g5", own5, g5)):
        for l in open(f"{TABLES}/{run}/table.jsonl"):
            row = json.loads(l)
            k, t = row["key"], row["tokens"]
            cs = [own[k]] if t <= cap and k in own else ([] if t <= cap else forced.get(k, []))
            out[(run, k)] = (rank(cs, int(k.rsplit("_", 1)[1])), t > cap)
    return out


def direct_rank(grids):
    seen = Counter()
    first = {}
    for i, g in enumerate(grids):
        if g is None:
            continue
        seen[tkey(g)] += 1
        first.setdefault(tkey(g), i)
    return [g for g, _ in sorted(seen.items(), key=lambda kv: (-kv[1], first[kv[0]]))]


def attempts(prog, direct):
    first = prog[0] if prog else (direct[0] if direct else None)
    rest = [g for g in direct if g != first] + [g for g in prog[1:] if g != first]
    seen, picks = set(), []
    for g in ([first] if first else []) + rest:
        if g not in seen:
            seen.add(g)
            picks.append(g)
    return picks[:2]


def task_mean(vals):
    by = defaultdict(lambda: defaultdict(list))
    for (run, k), x in vals.items():
        by[k.rsplit("_", 1)[0]][run].append(x)
    return {t: float(np.mean([np.mean(v) for v in runs.values()])) for t, runs in by.items()}


def paired(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * float(d.mean()), 2), "sd_points": round(100 * float(d.std(ddof=1)), 2), "sem_points": round(100 * float(sem), 2),
            "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None, "min_detectable_points_80pct": round(100 * 2.8 * float(sem), 2)}


def main():
    units, tokens = {}, defaultdict(list)
    for l in open(DIRECT):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("kind") == "unit":
            units[(r["run"], r["key"], r["cap"])] = r["grids"]
            tokens[r["cap"]].append(r["tokens"])
    res = {"direct_units": {str(c): len([k for k in units if k[2] == c]) for c in CAPS}, "caps": {}}
    for cap in CAPS:
        cand = candidates(cap)
        trunc = [k for k, (rk, t) in cand.items() if t]
        have = [k for k in trunc if (k[0], k[1], cap) in units]
        res["caps"][str(cap)] = {"outputs": len(cand), "longer_than_cap": len(trunc), "with_direct_samples": len(have), "mean_direct_tokens_n4": round(float(np.mean(tokens[cap])), 0) if tokens[cap] else None}
        if len(have) < 10:
            continue
        # evaluate on the outputs that have direct samples, so every policy is scored on the same population (outputs that ended before the cap are excluded: they keep the base policy)
        sub = {k: cand[k][0] for k in have}
        score = {}
        base2 = {k: float(gold(k[1]) in rk[:2]) for k, rk in sub.items()}
        base1 = {k: float(bool(rk) and rk[0] == gold(k[1])) for k, rk in sub.items()}
        score["base_top1"], score["base_top2"] = base1, base2
        for n in (1, 2, 4):
            score[f"direct_{n}"] = {k: float(gold(k[1]) in attempts(rk, direct_rank(units[(k[0], k[1], cap)][:n]))) for k, rk in sub.items()}
        score["direct_alone_top1_n4"] = {k: float(bool(direct_rank(units[(k[0], k[1], cap)])) and direct_rank(units[(k[0], k[1], cap)])[0] == gold(k[1])) for k in sub}
        score["direct_alone_top1_n1"] = {k: float(units[(k[0], k[1], cap)][0] is not None and tkey(units[(k[0], k[1], cap)][0]) == gold(k[1])) for k in sub}
        tm = {name: task_mean({(k[0], k[1]): v for k, v in s.items()}) for name, s in score.items()}
        c = res["caps"][str(cap)]
        c["subset_scores"] = {name: round(100 * float(np.mean(list(t.values()))), 2) for name, t in tm.items()}
        c["program_wrong_but_direct_right_n4"] = int(sum(1 for k in sub if base2[k] == 0 and score["direct_alone_top1_n4"][k] == 1))
        c["program_right_and_direct_right_n4"] = int(sum(1 for k in sub if base2[k] == 1 and score["direct_alone_top1_n4"][k] == 1))
        c["contrasts_vs_base_top2"] = {f"direct_{n}": paired(tm[f"direct_{n}"], tm["base_top2"], f"cap {cap}: direct_{n} minus base top-2, outputs longer than the cap") for n in (1, 2, 4)}
        # whole benchmark: outputs that ended before the cap keep the base policy
        whole_base, whole_pol = {}, {}
        for k, (rk, t) in cand.items():
            b = float(gold(k[1]) in rk[:2])
            whole_base[k] = b
            if t and (k[0], k[1], cap) in units:
                whole_pol[k] = float(gold(k[1]) in attempts(rk, direct_rank(units[(k[0], k[1], cap)])))
            else:
                whole_pol[k] = b
        wb, wp = task_mean(whole_base), task_mean(whole_pol)
        c["whole_benchmark"] = {"base_top2": round(100 * float(np.mean(list(wb.values()))), 2), "direct_4": round(100 * float(np.mean(list(wp.values()))), 2),
                                "contrast_direct_4_minus_base": paired(wp, wb, f"cap {cap}: direct_4 minus base top-2, whole benchmark")}
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
