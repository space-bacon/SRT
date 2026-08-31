#!/usr/bin/env python3
"""Why agreement misses the problems only the oracle reaches.

Pooling is settled: 48 candidates across five labs solve the same 154 of 164
problems as one model's own eight, +0.0000. The union oracle reaches 162. So the
pool contains eight more correct answers than any selector we have finds, and
since agreement returns the largest output cluster, those answers must sit in
clusters that are not the largest.

That reframes the remaining headroom. It is not coverage, every problem here is
already covered. It is ranking, and the question is whether the correct cluster
is distinguishable from the majority cluster by anything other than its size.

Three things this measures, in increasing order of usefulness:

  purity      do all members of a cluster share a fate. If clusters are pure,
              picking the right cluster IS solving the problem, and cluster-level
              scoring is a well-posed and much smaller problem than candidate
              scoring: a handful of clusters instead of 48 candidates.
  incidence   on how many problems does a correct cluster exist while the
              majority cluster is wrong. This is the reachable prize.
  separation  on exactly those problems, does any cheap feature rank the correct
              cluster first. Features are structural, not lexical, because the
              candidate-level verifier already failed at the lexical version.

A null here is worth as much as a hit. It would say the eight problems are not
recoverable by reweighting clusters, and that the oracle gap needs information
the pool does not contain.

    python scripts/minority_clusters.py --workers 48
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from code_select import execute_file, extract  # noqa: E402
from consensus_select import (inputs_from_example, probe_program,  # noqa: E402
                              run_capture, synth_inputs)


def cluster(problem, candidates, cases, timeout):
    """Group candidate indices by the outputs they produce. None means it did not run."""
    if not cases:
        return None
    entry = problem["entry_point"]
    sigs = [run_capture(probe_program(extract(problem["prompt"], c, entry), entry, cases),
                        timeout) for c in candidates]
    groups = {}
    for i, s in enumerate(sigs):
        if s is not None:
            groups.setdefault(s, []).append(i)
    return list(groups.values()) or None


def visible_pass(problem, candidate, timeout):
    """Does this candidate satisfy the examples the prompt itself states."""
    vt = problem.get("visible_test")
    if not vt:
        return None
    body = extract(problem["prompt"], candidate, problem["entry_point"])
    from srt_select.sandbox import run
    return run(body + "\n\n" + vt + "\n", timeout) is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="artifacts/nla/ensemble")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/ensemble/minority_clusters.json")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    probs = json.load(open(os.path.join(HERE, a.probs)))
    files = sorted(f for f in glob.glob(os.path.join(HERE, a.gen_dir, "*.json"))
                   if "__" in os.path.basename(f))
    members = [os.path.basename(f)[:-5].split("__")[0] for f in files]
    print(f"{len(members)} members: {', '.join(members)}", flush=True)

    # Pool the candidates, remembering which member produced each one.
    pooled, owner = [[] for _ in probs], []
    for f, m in zip(files, members):
        rows = json.load(open(f))
        for i, cands in enumerate(rows):
            pooled[i].extend(cands)
        owner.extend([m] * len(rows[0]))
    k = len(pooled[0])
    print(f"pooled to {k} candidates per problem", flush=True)

    ok = execute_file(pooled, probs, a.timeout, a.workers)

    cases = []
    for p in probs:
        c = synth_inputs(p["prompt"], p["entry_point"], a.cases)
        if not c:
            c = inputs_from_example(p.get("visible_test"), p["entry_point"], a.cases)
        cases.append(c)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        clustered = list(ex.map(
            lambda i: cluster(probs[i], pooled[i], cases[i], a.timeout), range(len(probs))))

    impure, minority_held, majority_correct, no_cluster = [], [], 0, 0
    purity_rates = []
    for i, groups in enumerate(clustered):
        if not groups:
            no_cluster += 1
            continue
        fates = [{bool(ok[i, j]) for j in g} for g in groups]
        purity_rates.append(sum(len(f) == 1 for f in fates) / len(fates))
        if any(len(f) > 1 for f in fates):
            impure.append(i)
        biggest = max(range(len(groups)), key=lambda g: len(groups[g]))
        correct = [g for g, f in enumerate(fates) if True in f]
        if not correct:
            continue
        if biggest in correct:
            majority_correct += 1
        else:
            minority_held.append({
                "task": probs[i].get("task_id", i),
                "majority_size": len(groups[biggest]),
                "n_clusters": len(groups),
                "correct_sizes": [len(groups[g]) for g in correct],
                "correct_rank_by_size": sorted(
                    (len(groups[g]) for g in range(len(groups))), reverse=True
                ).index(max(len(groups[g]) for g in correct)) + 1,
                "correct_labs": sorted({owner[j] for g in correct for j in groups[g]}),
                "majority_labs": sorted({owner[j] for j in groups[biggest]}),
                "correct_visible": [visible_pass(probs[i], pooled[i][groups[g][0]], a.timeout)
                                    for g in correct],
                "majority_visible": visible_pass(
                    probs[i], pooled[i][groups[biggest][0]], a.timeout),
            })

    n = len(probs)
    res = {
        "question": "are the oracle-only problems held by clusters a selector could rank",
        "n_problems": n, "pool_size": k, "members": members,
        "purity": {
            "mean_pure_cluster_fraction": round(float(np.mean(purity_rates)), 4),
            "problems_with_any_impure_cluster": len(impure),
            "note": "a pure cluster is one whose members all pass or all fail. If purity "
                    "is high, choosing the right cluster is equivalent to solving the "
                    "problem, and cluster scoring is well posed",
        },
        "incidence": {
            "oracle_solvable": int(ok.any(1).sum()),
            "majority_cluster_correct": majority_correct,
            "minority_held": len(minority_held),
            "unclusterable": no_cluster,
            "note": "minority_held is the reachable prize: a correct cluster exists and "
                    "agreement does not pick it",
        },
        "separation": {},
        "minority_cases": minority_held,
    }

    if minority_held:
        vis = [c for c in minority_held if c["majority_visible"] is not None]
        beats = [c for c in vis if any(c["correct_visible"]) and not c["majority_visible"]]
        res["separation"] = {
            "stated_examples_would_flip": len(beats),
            "of_minority_cases_with_examples": len(vis),
            "median_correct_rank_by_size": statistics.median(
                c["correct_rank_by_size"] for c in minority_held),
            "correct_cluster_is_a_singleton": sum(
                1 for c in minority_held if max(c["correct_sizes"]) == 1),
            "lab_counts_in_correct": dict(Counter(
                lab for c in minority_held for lab in c["correct_labs"])),
        }

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)

    p, inc, sep = res["purity"], res["incidence"], res["separation"]
    print(f"\n  cluster purity            {p['mean_pure_cluster_fraction']:.4f}"
          f"  ({p['problems_with_any_impure_cluster']} problems have an impure cluster)")
    print(f"  oracle solvable           {inc['oracle_solvable']} / {n}")
    print(f"  majority cluster correct  {inc['majority_cluster_correct']}")
    print(f"  MINORITY-HELD (the prize) {inc['minority_held']}")
    if sep:
        print(f"\n  stated examples would flip {sep['stated_examples_would_flip']}"
              f" of {sep['of_minority_cases_with_examples']} that state any")
        print(f"  correct cluster is a singleton in {sep['correct_cluster_is_a_singleton']}")
        print(f"  median size-rank of correct cluster {sep['median_correct_rank_by_size']}")
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
