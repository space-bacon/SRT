#!/usr/bin/env python
"""Score run_llm_arc.py output: pass@k (unbiased), accuracy under token caps, token statistics. Writes JSON via temp file and rename."""
import argparse, json, math, os, re
from collections import defaultdict


def parse_grid(text):
    if not text:
        return None
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    for src in ([blocks[-1]] if blocks else []) + [text]:
        rows, last = [], []
        for line in src.splitlines():
            s = re.sub(r"[\s,\[\]]", "", line)
            if s and s.isdigit():
                rows.append([int(c) for c in s])
            else:
                if rows:
                    last = rows
                rows = []
        if rows:
            last = rows
        if last and len({len(r) for r in last}) == 1:
            return last
    return None


def pass_at(n, c, k):
    if n == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    keep = set(json.load(open(a.tasks))) if a.tasks else None
    samples = defaultdict(dict)
    for p in a.runs:
        for line in open(p):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            samples[r["key"]][r["sample"]] = r
    per = {}
    for key, ss in samples.items():
        tid, ti = key.rsplit("_", 1)
        if keep is not None and tid not in keep:
            continue
        gold = sol[tid][int(ti)]
        rec = []
        for s, r in sorted(ss.items()):
            g = parse_grid(r["content"])
            rec.append({"s": s, "ok": g == gold, "parsed": g is not None, "tok": r["completion_tokens"], "finish": r["finish"]})
        per[key] = rec
    caps = [4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
    res = {"runs": a.runs, "n_outputs": len(per), "n_samples": sum(len(v) for v in per.values())}

    def task_mean(fn):
        by = defaultdict(list)
        for key, rec in per.items():
            by[key.rsplit("_", 1)[0]].append(fn(rec))
        return sum(sum(v) / len(v) for v in by.values()) / max(len(by), 1), len(by)

    for k in (1, 2, 4, 8):
        if any(len(v) >= k for v in per.values()):
            res[f"pass@{k}_tasks"], res["n_tasks"] = task_mean(lambda rec, k=k: pass_at(len(rec), sum(x["ok"] for x in rec), k))
            res[f"pass@{k}_outputs"] = sum(pass_at(len(r), sum(x["ok"] for x in r), k) for r in per.values()) / max(len(per), 1)
    flat = [x for rec in per.values() for x in rec]
    res["mean_tokens"] = sum(x["tok"] for x in flat) / max(len(flat), 1)
    toks = sorted(x["tok"] for x in flat)
    res["tokens_quantiles"] = {q: toks[min(int(q * len(toks)), len(toks) - 1)] for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if toks else {}
    res["frac_truncated"] = sum(x["finish"] == "length" for x in flat) / max(len(flat), 1)
    res["frac_unparsed"] = sum(not x["parsed"] for x in flat) / max(len(flat), 1)
    res["sample_acc"] = sum(x["ok"] for x in flat) / max(len(flat), 1)
    res["acc_under_cap"] = {}
    for cap in caps:
        capped = {key: [{**x, "ok": x["ok"] and x["tok"] <= cap} for x in rec] for key, rec in per.items()}
        by1, by2 = defaultdict(list), defaultdict(list)
        for key, rec in capped.items():
            t = key.rsplit("_", 1)[0]
            by1[t].append(pass_at(len(rec), sum(x["ok"] for x in rec), 1))
            by2[t].append(pass_at(len(rec), sum(x["ok"] for x in rec), 2))
        res["acc_under_cap"][cap] = {
            "pass@1_tasks": sum(sum(v) / len(v) for v in by1.values()) / max(len(by1), 1),
            "pass@2_tasks": sum(sum(v) / len(v) for v in by2.values()) / max(len(by2), 1),
            "mean_tokens_used": sum(min(x["tok"], cap) for x in flat) / max(len(flat), 1)}
    res["per_output"] = per
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_output", "runs")}, indent=1))


if __name__ == "__main__":
    main()
