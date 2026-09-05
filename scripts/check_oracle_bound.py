#!/usr/bin/env python3
"""No arm may exceed its own oracle. One pass over every results file in the repo.

A selector picks one of K candidates per problem. The oracle is "at least one of the
K passes", scored over the same problems, so no selector column computed over all
problems can exceed it. When one does, a subset has been averaged against a full
denominator, which is how `consensus_select.py` once let 12 of 36 arms beat their
own oracle. This check runs after any selector run and in CI.

Only columns scored over every problem are held to the bound. Columns that are
declared subset rates (`consensus_on_covered`, `on_resolved_only`) are reported,
not judged: they sit beside an all-rows oracle by design and exceed it legitimately.
Files carrying a `SUPERSEDED` key are reported as such and never fail the run.

    python scripts/check_oracle_bound.py               # artifacts/nla, exit 1 on a violation
    python scripts/check_oracle_bound.py path.json ... # specific files
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Selector columns scored over all problems, and therefore bounded by the oracle.
ALL_ROWS = ("consensus", "consensus_strict", "chat_consensus", "verifier", "exec_guided",
            "example_filter", "example_filter_plus_verifier", "medoid", "medoid_centered", "pooled", "selected")
# Subset rates: shown, never judged.
SUBSET = ("consensus_on_covered", "on_resolved_only")
EPS = 1e-9


def check(path: Path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    arms = d.get("arms") if isinstance(d, dict) else None
    if not isinstance(arms, dict) or not arms:
        return None
    superseded = "SUPERSEDED" in d
    bad, subset_over, checked = [], [], 0
    for arm, v in arms.items():
        if not isinstance(v, dict) or "oracle" not in v or v["oracle"] is None:
            continue
        oracle = float(v["oracle"])
        for col in ALL_ROWS:
            x = v.get(col)
            if isinstance(x, (int, float)):
                checked += 1
                if x > oracle + EPS:
                    bad.append((arm, col, float(x), oracle))
        for col in SUBSET:
            x = v.get(col)
            if isinstance(x, (int, float)) and x > oracle + EPS:
                subset_over.append((arm, col, float(x), oracle))
    return {"path": path, "superseded": superseded, "checked": checked, "bad": bad, "subset_over": subset_over}


def main(argv):
    files = [Path(a) for a in argv] if argv else sorted(Path(p) for p in glob.glob(str(ROOT / "artifacts/nla/**/*.json"), recursive=True))
    fails = 0
    for f in files:
        r = check(f)
        if not r or (not r["checked"] and not r["subset_over"]):
            continue
        rel = f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f
        tag = "SUPERSEDED" if r["superseded"] else ("FAIL" if r["bad"] else "ok")
        print(f"{tag:10s} {rel}  ({r['checked']} bounded cells"
              f"{f', {len(r['subset_over'])} subset-rate cells above oracle, by design' if r['subset_over'] else ''})")
        for arm, col, x, o in r["bad"]:
            print(f"           {arm:34s} {col} {x:.4f} > oracle {o:.4f}")
        if r["bad"] and not r["superseded"]:
            fails += 1
    if fails:
        print(f"\n{fails} file(s) violate the bound: a selector column scored over all problems exceeds its oracle.")
        return 1
    print("\nno all-rows selector exceeds its oracle in any current file")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
