#!/usr/bin/env python
"""Task-level top-1 and top-2 of the unchanged program arm (summary 1, cap 32,768, eight forced programs, one trace per task) on 120 ARC-AGI-1 public evaluation tasks.

The tasks are `random.Random(20261010).sample` of the 400 evaluation tasks of fchollet/ARC-AGI (commit 399030444e0a); a task scores the mean of correct_top1 (correct_top2) over its test outputs.
The interval is a task bootstrap (10,000 resamples, seed 0). Writes results/arc1_sample_analysis.json atomically.
"""
import collections, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results") if os.path.isdir(os.path.join(HERE, "results")) else os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "raw", "box_2026-10-10", "a1ps.jsonl")


def boot(x, rng):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
    return [round(float(v), 1) for v in np.percentile(100 * np.array(b), [2.5, 97.5])]


def main():
    done = {}
    for line in open(LOG):
        r = json.loads(line)
        if r.get("event") == "done":
            done[r["task"]] = r
    rng = np.random.default_rng(0)
    t1 = np.array([np.mean(r["correct_top1"]) for r in done.values()])
    t2 = np.array([np.mean(r["correct_top2"]) for r in done.values()])
    tok = np.array([r["ntok"] + r["forced_tokens"] for r in done.values()])
    stop = [r for r in done.values() if r["finish"] == "stop"]
    cap = [r for r in done.values() if r["finish"] != "stop"]
    out = {"population": "120 ARC-AGI-1 public evaluation tasks drawn with random.Random(20261010), program arm with summary 1, cap 32,768, 8 forced programs, 4 x A100",
           "n_tasks": len(done), "n_outputs": int(sum(len(r["correct_top1"]) for r in done.values())),
           "top1": round(100 * float(t1.mean()), 2), "top1_ci95": boot(t1, rng), "top2": round(100 * float(t2.mean()), 2), "top2_ci95": boot(t2, rng),
           "decoded_tokens_per_task_mean": round(float(tok.mean())), "decoded_tokens_per_task_median": round(float(np.median(tok))),
           "finish": dict(collections.Counter(r["finish"] for r in done.values())),
           "top1_when_trace_stopped_by_itself": round(100 * float(np.mean([np.mean(r["correct_top1"]) for r in stop])), 1), "n_stopped_by_itself": len(stop),
           "top1_when_trace_hit_cap": round(100 * float(np.mean([np.mean(r["correct_top1"]) for r in cap])), 1), "n_hit_cap": len(cap),
           "share_with_verified_program": round(float(np.mean([r["n_verified"] > 0 for r in done.values()])), 3)}
    path = os.path.join(RES, "arc1_sample_analysis.json")
    json.dump(out, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
