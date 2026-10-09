#!/usr/bin/env python
"""Score the forced-step variants of force_variants.py against the control (closing sentence A, temperature 0.7) on the same truncated traces.

For every trace longer than the 32,768 cap the control programs are the eight saved forced programs (g4: flat_exec.jsonl; g5: g5_force_exec.jsonl); each variant has eight new
programs sampled from the same truncated reasoning. Programs run on all test inputs, demo-passing programs weigh 2 and partial passes 0.25 x fraction. Task score = mean over a
task's outputs, averaged over the two runs where both have the task. Contrasts are paired over tasks: signed mean first, sd, sem, mean/sem, smallest effect at 80% power
(2.8 x sem). Also reported: the share of programs that pass every demo, and decoded tokens per sample. Outputs that ended before the cap keep their own program in every arm and
are left out of the contrast. Writes through a temp file and a rename.
"""
import json, math, os, subprocess, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_direct as D

R = D.R
VARIANT_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "raw", "fb1", "force_variants.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "forced_variants_analysis.json")
WORK = "/tmp/fvwork"
CHAL = "/tmp/arcdata/arc-agi_evaluation_challenges.json"
CAP = 32768


def main():
    os.makedirs(WORK, exist_ok=True)
    flat, exe = f"{WORK}/flat.jsonl", f"{WORK}/exec.jsonl"
    recs = []
    for l in open(VARIANT_FILE):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "completions" in r:
            recs.append(r)
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(VARIANT_FILE):
        with open(flat, "w") as f:
            for r in recs:
                for j, c in enumerate(r["completions"]):
                    f.write(json.dumps({"status": "ok", "key": r["key"], "sample": f"{r['variant']}_{r['run']}_{j}", "content": c}) + "\n")
        subprocess.run([sys.executable, os.path.join(HERE, "exec_code.py"), "--challenges", CHAL, "--runs", flat, "--out", exe, "--workers", "16"], check=True)
    ex = defaultdict(list)
    for l in open(exe):
        r = json.loads(l)
        v, run, _j = r["sample"].split("_")
        ex[(v, run, r["key"])].append(r)
    toks = {(r["run"], r["key"]): r["completion_tokens"] / max(len(r["completions"]), 1) for r in recs}
    cand = D.candidates(CAP)
    variants = sorted({r["variant"] for r in recs})
    have = {v: {(r["run"], r["key"]) for r in recs if r["variant"] == v} for v in variants}
    trunc = {k for k, (rk, t) in cand.items() if t}
    res = {"variants": {}}

    def score(rank_list, k):
        g = D.gold(k[1])
        return float(bool(rank_list) and rank_list[0] == g), float(g in rank_list[:2])

    # each variant is scored on the truncated outputs it finished (queue order was by trace, so a stopped run is a random subset), against the control on the same outputs
    for v in variants:
        pop = sorted(have[v] & trunc)
        if len(pop) < 10:
            continue
        ctl = {k: score(cand[k][0], k) for k in pop}
        ctl1 = D.task_mean({k: x[0] for k, x in ctl.items()})
        ctl2 = D.task_mean({k: x[1] for k, x in ctl.items()})
        s1, s2, allpass, nprog = {}, {}, 0, 0
        for k in pop:
            cs = ex[(v, k[0], k[1])]
            rk = D.rank(cs, int(k[1].rsplit("_", 1)[1]))
            s1[k], s2[k] = score(rk, k)
            allpass += sum(1 for c in cs if not c.get("error") and c["n_pass"] == c["n_train"])
            nprog += len(cs)
        t1, t2 = D.task_mean(s1), D.task_mean(s2)
        res["variants"][v] = {"n_outputs": len(pop), "top1": round(100 * float(np.mean(list(t1.values()))), 2), "top2": round(100 * float(np.mean(list(t2.values()))), 2),
                              "control_top1": round(100 * float(np.mean(list(ctl1.values()))), 2), "control_top2": round(100 * float(np.mean(list(ctl2.values()))), 2),
                              "share_programs_passing_every_demo": round(allpass / max(nprog, 1), 4), "mean_tokens_per_sample": round(float(np.mean([toks[k] for k in pop])), 1),
                              "contrast_top1_vs_control": D.paired(t1, ctl1, f"{v} minus control, top-1, outputs cut at {CAP}"),
                              "contrast_top2_vs_control": D.paired(t2, ctl2, f"{v} minus control, top-2, outputs cut at {CAP}")}
    res["n_truncated_outputs"] = len(trunc)
    # control share passing every demo (g4 flat_exec and g5 force_exec carry n_pass) and decoded tokens per control sample (g4 and g5 forced files)
    cp, cn = 0, 0
    tr_g4 = {k[1] for k in trunc if k[0] == "g4"}
    tr_g5 = {k[1] for k in trunc if k[0] == "g5"}
    for l in open("/tmp/arcllm/pertask/flat_exec.jsonl"):
        r = json.loads(l)
        if r["sample"].startswith(f"c{CAP}_") and r["key"] in tr_g4:
            cn += 1
            cp += int(not r.get("error") and r["n_pass"] == r["n_train"])
    for l in open(f"{R}/raw/g5full/artifacts/g5_force_exec.jsonl"):
        r = json.loads(l)
        _, c, _j = str(r["sample"]).split("_")
        if int(c[1:]) == CAP and r["key"] in tr_g5:
            cn += 1
            cp += int(not r.get("error") and r["n_pass"] == r["n_train"])
    res["control"] = {"share_programs_passing_every_demo": round(cp / max(cn, 1), 4)}
    import glob
    seen, ctok = set(), []
    for path in sorted(glob.glob(f"{R}/raw/g4full/g4_force*.jsonl")) + [f"{R}/raw/g5full/artifacts/g5_force.jsonl"]:
        for l in open(path):
            r = json.loads(l)
            if r.get("cap") == CAP and "completions" in r and (path.split("/")[-1][:2], r["key"]) not in seen:
                seen.add((path.split("/")[-1][:2], r["key"]))
                ctok.append(r["completion_tokens"] / max(len(r["completions"]), 1))
    res["control"]["mean_tokens_per_sample"] = round(float(np.mean(ctok)), 1) if ctok else None
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
