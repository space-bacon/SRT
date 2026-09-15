#!/usr/bin/env python3
"""No selector column scored over all problems may exceed its own arm's oracle.

A selector picks one of K candidates per problem. The oracle is "at least one of the K passes",
scored over the same problems, so no column computed over all problems can exceed it. When one
does, a subset has been averaged against a full denominator, which is how `consensus_select.py`
once let 12 of 36 arms beat their own oracle.

Coverage is the default. The first version of this file matched one key, `d["arms"]`, and held an
allowlist of eleven column names. Dipankar Sarkar counted what that could see: 266 of 1,373
bounded cells, five of the eleven names never appearing as a column anywhere, and `None` returning
silently so a file that was never read looked exactly like a file that passed. All three are the
same failure as the bug the check was written for, one level up.

So: walk to any dict holding a numeric `oracle`, wherever it sits; judge every numeric sibling
unless it is exempt for a declared reason; and print what was scanned, judged and skipped, so the
output says which files it looked at rather than only what it found.

    python scripts/check_oracle_bound.py               # artifacts/nla, exit 1 on a violation
    python scripts/check_oracle_bound.py path.json ... # specific files
    python scripts/check_oracle_bound.py --verbose     # list every skipped column and why
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9

# Reported beside the oracle, never judged: rates over a subset of problems, which exceed an
# all-rows oracle by design. Adding a key here is a claim that the column has its own denominator,
# and it belongs in the script that writes the column.
SUBSET = {
    "consensus_on_covered": "rate over covered problems only",
    "on_resolved_only": "rate over resolved problems only",
}

# Not pass rates, so the bound does not apply. Counts, sizes, and scores on other scales.
NOT_A_RATE = {
    "n": "count", "n_problems": "count", "pool_size": "count",
    "covered_problems": "count", "resolved": "count", "best_n_cases": "count",
    "params_b": "model size", "auroc": "ranking score, not a pass rate",
    "headroom_captured": "fraction of headroom, can exceed 1 legitimately",
}


def exempt(key: str, val: float) -> str | None:
    """Why this numeric sibling is not held to the bound, or None to judge it."""
    if key in SUBSET:
        return f"subset rate, {SUBSET[key]}"
    if key in NOT_A_RATE:
        return NOT_A_RATE[key]
    if "_minus_" in key or key.startswith("gain"):
        return "difference between two columns"
    if not 0.0 <= val <= 1.0:
        return "outside [0, 1], so not a pass rate"
    return None


def walk(o, path: str = ""):
    """Every dict carrying a numeric `oracle`, at whatever depth and under whatever key."""
    if isinstance(o, dict):
        if isinstance(o.get("oracle"), (int, float)):
            yield path or ".", o
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")


def check(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"path": path, "unreadable": str(e)[:60]}
    nodes = list(walk(d))
    if not nodes:
        return None
    superseded = "SUPERSEDED" in d if isinstance(d, dict) else False
    judged, bad, subset_over, skipped = 0, [], [], collections.Counter()
    for where, node in nodes:
        oracle = float(node["oracle"])
        for k, v in node.items():
            if k == "oracle" or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            why = exempt(k, float(v))
            if why:
                skipped[f"{k}: {why}"] += 1
                if k in SUBSET and float(v) > oracle + EPS:
                    subset_over.append((where, k, float(v), oracle))
                continue
            judged += 1
            if float(v) > oracle + EPS:
                bad.append((where, k, float(v), oracle))
    return {"path": path, "superseded": superseded, "nodes": len(nodes),
            "judged": judged, "bad": bad, "subset_over": subset_over, "skipped": skipped}


def main(argv: list[str]) -> int:
    verbose = "--verbose" in argv
    argv = [a for a in argv if a != "--verbose"]
    files = [Path(a) for a in argv] if argv else sorted(
        Path(p) for p in glob.glob(str(ROOT / "artifacts/nla/**/*.json"), recursive=True))

    scanned = with_oracle = fails = total_judged = 0
    all_skipped: collections.Counter = collections.Counter()
    for f in files:
        scanned += 1
        r = check(f)
        if r is None:
            continue
        rel = f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f
        if "unreadable" in r:
            print(f"{'UNREADABLE':11s} {rel}  {r['unreadable']}")
            continue
        with_oracle += 1
        total_judged += r["judged"]
        all_skipped.update(r["skipped"])
        tag = "SUPERSEDED" if r["superseded"] else ("FAIL" if r["bad"] else "ok")
        extra = (f", {len(r['subset_over'])} subset-rate cells above oracle, by design"
                 if r["subset_over"] else "")
        print(f"{tag:11s} {rel}  ({r['nodes']} oracle nodes, {r['judged']} judged{extra})")
        for where, col, x, o in r["bad"]:
            print(f"            {where:44s} {col} {x:.4f} > oracle {o:.4f}")
        if r["bad"] and not r["superseded"]:
            fails += 1

    skipped_n = sum(all_skipped.values())
    print(f"\nscanned {scanned}, carried an oracle {with_oracle}, "
          f"judged {total_judged} cells, skipped {skipped_n}")
    if verbose:
        for reason, n in all_skipped.most_common():
            print(f"    {n:6d}  {reason}")
    elif all_skipped:
        print("    (--verbose lists every skipped column and why)")

    if fails:
        print(f"\n{fails} file(s) violate the bound: "
              f"a column scored over all problems exceeds its oracle.")
        return 1
    print("no column scored over all problems exceeds its oracle in any file scanned above")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
