#!/usr/bin/env python
"""Collect, for each saved run (g4, g5), the traces that were still reasoning at the 16K checkpoint and the eight programs sampled from their first 16K tokens.

Output (one directory per run): table.jsonl with {run, key, tokens, progs16: [completion strings]} and the trace files for those keys, ready to copy to a GPU box.
g5 reads g5_force.jsonl; g4 reads the flattened candidates written by per_task_eval.py (flat.jsonl, samples named c<cap>_f<file>_j<index>); when several forced files
contain the same key at 16K the lowest file index is used.
"""
import gzip, json, os, shutil, sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arcllm/fbdata"
CAP = 16384

runs = {
    "g4": {"code": f"{R}/g4full/g4_code.jsonl", "traces": f"{R}/g4full/traces_g4"},
    "g5": {"code": f"{R}/g5full/artifacts/g5_code.jsonl", "traces": f"{R}/g5full/artifacts/traces_g5"},
}
for run, p in runs.items():
    toks = {}
    for l in open(p["code"]):
        r = json.loads(l)
        if r.get("status") == "ok" and r["sample"] == 0:
            toks[r["key"]] = r["completion_tokens"]
    progs = defaultdict(dict)
    if run == "g5":
        for l in open(f"{R}/g5full/artifacts/g5_force.jsonl"):
            r = json.loads(l)
            if "completions" in r and r["cap"] == CAP and r["sample"] == 0:
                progs[r["key"]][0] = r["completions"]
    else:
        by = defaultdict(lambda: defaultdict(dict))
        for l in open("/tmp/arcllm/pertask/flat.jsonl"):
            r = json.loads(l)
            s = r["sample"]
            if s == "own" or not s.startswith(f"c{CAP}_"):
                continue
            _, fi, j = s.split("_")
            by[r["key"]][int(fi[1:])][int(j[1:])] = r["content"]
        for k, files in by.items():
            fi = min(files)
            progs[k][fi] = [files[fi][j] for j in sorted(files[fi])]
    out = os.path.join(OUT, run)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    n = 0
    tmp = os.path.join(out, "table.jsonl.tmp")
    with open(tmp, "w") as f:
        for k in sorted(toks):
            if toks[k] <= CAP or k not in progs:
                continue
            fi = min(progs[k])
            src = os.path.join(p["traces"], f"{k}__0.txt.gz")
            shutil.copy(src, os.path.join(out, "traces", f"{k}__0.txt.gz"))
            f.write(json.dumps({"run": run, "key": k, "tokens": toks[k], "progs16": progs[k][fi]}) + "\n")
            n += 1
    os.replace(tmp, os.path.join(out, "table.jsonl"))
    print(run, "outputs", len(toks), "with trace beyond 16K and programs", n, "missing programs", sum(1 for k in toks if toks[k] > CAP and k not in progs))
