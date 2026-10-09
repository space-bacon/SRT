#!/usr/bin/env python
"""Program-synthesis arm: demo-verified selection among executed candidates, optionally pooled with direct answers and NVARC.

Task-level means (mean over a task's test outputs, then over tasks). Contrasts are paired signed means over tasks with sem and
the smallest effect detectable at 80% power (2.8 x sem).
"""
import argparse, json, math, os
from collections import defaultdict
import numpy as np
from score_llm import parse_grid, pass_at


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
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return {"contrast": label, "n_tasks": len(ks), "mean_diff_points": round(100 * d.mean(), 2), "sd_points": round(100 * d.std(ddof=1), 2),
            "sem_points": round(100 * sem, 2), "mean_over_sem": round(float(d.mean() / sem), 2) if sem else None,
            "min_detectable_points_80pct": round(100 * 2.8 * sem, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-run", required=True)
    ap.add_argument("--exec", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--direct", default="")
    ap.add_argument("--nvarc", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-k", type=int, default=0)
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    gold = lambda k: tkey(sol[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
    tok = {}
    for line in open(a.code_run):
        r = json.loads(line)
        if r.get("status") == "ok":
            tok[(r["key"], r["sample"])] = r["completion_tokens"]
    cand = defaultdict(list)
    for line in open(a.exec):
        r = json.loads(line)
        if a.first_k and r["sample"] >= a.first_k:
            continue
        cand[r["key"]].append({"s": r["sample"], "full": r["n_pass"] == r["n_train"], "frac": r["n_pass"] / max(r["n_train"], 1),
                               "pred": tkey(r["pred"]), "tok": tok.get((r["key"], r["sample"]), 0)})
    direct = defaultdict(list)
    if a.direct:
        for line in open(a.direct):
            r = json.loads(line)
            if r.get("status") == "ok" and (not a.first_k or r["sample"] < a.first_k):
                direct[r["key"]].append(tkey(parse_grid(r["content"])))

    res = {"n_outputs": len(cand), "n_samples": sum(len(v) for v in cand.values())}
    flat = [c for v in cand.values() for c in v]
    res["frac_all_demos_pass"] = round(float(np.mean([c["full"] for c in flat])), 3)
    res["frac_has_prediction"] = round(float(np.mean([c["pred"] is not None for c in flat])), 3)
    res["mean_tokens"] = round(float(np.mean([c["tok"] for c in flat])), 0)
    correct = {k: [c["pred"] is not None and c["pred"] == gold(k) for c in v] for k, v in cand.items()}
    for k_ in (1, 2, 4):
        res[f"pass@{k_}_raw"] = round(100 * float(np.mean(list(task_avg({k: pass_at(len(v), sum(v), k_) for k, v in correct.items()}).values()))), 2)
    res["oracle_any_correct"] = round(100 * float(np.mean(list(task_avg({k: float(any(v)) for k, v in correct.items()}).values()))), 2)
    full_flags = [(c["full"], ok) for k, v in cand.items() for c, ok in zip(v, correct[k])]
    nfull = sum(f for f, _ in full_flags)
    res["verification"] = {"samples_all_pass": nfull, "precision_correct_given_all_pass": round(sum(1 for f, o in full_flags if f and o) / max(nfull, 1), 3),
                           "recall_of_correct": round(sum(1 for f, o in full_flags if f and o) / max(sum(o for _, o in full_flags), 1), 3)}

    def rank(k, use_direct, w_direct=1.0, w_full=2.0, w_part=0.25):
        score = defaultdict(float)
        for c in cand[k]:
            if c["pred"] is None:
                continue
            score[c["pred"]] += w_full if c["full"] else w_part * c["frac"]
        if use_direct:
            for g in direct.get(k, []):
                if g is not None:
                    score[g] += w_direct
        return [g for g, _ in sorted(score.items(), key=lambda kv: -kv[1])][:2]

    sel = {k: float(gold(k) in rank(k, False)) for k in cand}
    sel1 = {k: float(bool(rank(k, False)) and rank(k, False)[0] == gold(k)) for k in cand}
    res["verified_top2"] = round(100 * float(np.mean(list(task_avg(sel).values()))), 2)
    res["verified_top1"] = round(100 * float(np.mean(list(task_avg(sel1).values()))), 2)
    contrasts = [paired(task_avg(sel), task_avg({k: pass_at(len(v), sum(v), 2) for k, v in correct.items()}), "verified top-2 minus unbiased pass@2 of random pairs")]
    if direct:
        dvote = {}
        for k in cand:
            cnt = defaultdict(int)
            for g in direct.get(k, []):
                if g is not None:
                    cnt[g] += 1
            dvote[k] = float(gold(k) in [g for g, _ in sorted(cnt.items(), key=lambda kv: -kv[1])][:2])
        comb = {k: float(gold(k) in rank(k, True)) for k in cand}
        res["direct_vote_top2"] = round(100 * float(np.mean(list(task_avg(dvote).values()))), 2)
        res["combined_top2"] = round(100 * float(np.mean(list(task_avg(comb).values()))), 2)
        contrasts += [paired(task_avg(sel), task_avg(dvote), "code verified top-2 minus direct vote top-2"),
                      paired(task_avg(comb), task_avg(dvote), "combined minus direct vote top-2"),
                      paired(task_avg(comb), task_avg(sel), "combined minus code verified top-2")]
    for p in a.nvarc:
        nv = json.load(open(p))["per_output"]
        tag = os.path.basename(p).replace("score_", "").replace(".json", "")
        top1 = {k: float(bool(rank(k, bool(direct))) and rank(k, bool(direct))[0] == gold(k)) for k in cand}
        hyb = {k: float(bool(top1[k]) or bool(nv[k]["kgmon_top1"])) for k in cand if k in nv}
        res[f"hybrid_{tag}_llmtop1_plus_nvarctop1"] = round(100 * float(np.mean(list(task_avg(hyb).values()))), 2)
        union = {k: float(any(correct[k]) or bool(nv[k]["kgmon"])) for k in cand if k in nv}
        res[f"union_oracle_{tag}"] = round(100 * float(np.mean(list(task_avg(union).values()))), 2)
    res["contrasts"] = contrasts
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
