#!/usr/bin/env python3
"""Source the two numbers the card was asserting on its own authority.

`verify_cards.py` gates every figure in a card against an artifact. Two figures
in the hivemind card had no artifact to gate against: the corpus census
("110,704 generations across 97 arms") and the selection-decay slope
(-0.0704 per decade). Both were real, both were computed once in a shell and
transcribed, and neither could be rechecked by a reader or by the gate.

A number that only exists in prose is a number nobody can audit, including us.

    python scripts/hivemind_census.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re

import numpy as np

ROOT = pathlib.Path("artifacts/nla")
OUT = ROOT / "hivemind_census.json"

# Arms the corpus claim covers. Excludes atlas prose samples, which are scored
# for similarity rather than executed, and the three superseded directories.
CENSUS_DIRS = [
    "coder_matrix1024", "ministral_ladder", "mbpp_ladder",
    "ensemble", "crosslab_suppression", "ministral_suppression",
]
EXCLUDED_DIRS = ["coder_ladder", "ensemble/excluded", "ministral_ladder/corrupt_chat_pre_fix"]

PARAMS_B = {"1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0}


def count(dirs):
    arms, gens = {}, 0
    for d in dirs:
        for f in sorted(glob.glob(str(ROOT / d / "*.json"))):
            name = os.path.basename(f)[:-5]
            if "__" not in name:
                continue
            rows = json.load(open(f))
            if not isinstance(rows, list):
                continue
            n = sum(len(r) for r in rows if isinstance(r, list))
            if n:
                arms[f"{d}/{name}"] = n
                gens += n
    return arms, gens


# Each read scores the same arms, so the slope is only defined once you say which
# one. The paper quotes the chat-native read, because that is the deployable one.
READS = [
    ("chat_consensus", "chat_consensus_mbpp.json", "arms", "chat_consensus"),
    ("consensus", "consensus_mbpp.json", "arms", "consensus"),
    ("exec_guided", "exec_guided_mbpp.json", "per_arm", "exec"),
]
PUBLISHED_READ = "chat_consensus"


def fit_read(filename, container, key):
    """Selection gain against log10 parameters, on the MBPP rungs."""
    src = json.load(open(ROOT / "verifier" / filename))[container]
    pts = []
    for arm, v in src.items():
        m = re.search(r"mbpp([\d.]+B)_", arm)
        if not m or m.group(1) not in PARAMS_B:
            continue
        pts.append((PARAMS_B[m.group(1)], v[key] - v["floor"], arm, v))
    pts.sort()
    x = np.log10([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "metric": f"{key} minus random-single floor, MBPP",
        "slope_per_decade": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "n_rungs": len(pts),
        "rungs": [{"params_b": p[0], "arm": p[2], "gain": round(p[1], 4),
                   "floor": p[3]["floor"], "selected": round(p[3][key], 4),
                   "oracle": p[3]["oracle"],
                   "headroom_captured": round(
                       (p[3][key] - p[3]["floor"]) / (p[3]["oracle"] - p[3]["floor"]), 4)
                   if p[3]["oracle"] > p[3]["floor"] else None} for p in pts],
    }


def decay_slopes():
    by_read = {name: fit_read(f, c, k) for name, f, c, k in READS}
    slopes = [v["slope_per_decade"] for v in by_read.values()]
    return {
        "published_read": PUBLISHED_READ,
        "slope_per_decade": by_read[PUBLISHED_READ]["slope_per_decade"],
        "by_read": by_read,
        "robustness": {
            "all_negative": all(s < 0 for s in slopes),
            "range": [round(min(slopes), 4), round(max(slopes), 4)],
            "note": "the direction does not depend on the read, the magnitude does. "
                    "Execution-guided filtering decays fastest and fits tightest; "
                    "quote the read alongside the slope",
        },
    }


def main():
    arms, gens = count(CENSUS_DIRS)
    excl_arms, excl_gens = count(EXCLUDED_DIRS)
    res = {
        "question": "how much was generated, and does selection value survive scale",
        "census": {
            "generations": gens,
            "arms": len(arms),
            "directories": CENSUS_DIRS,
            "note": "executed candidate pools only; atlas prose samples are scored "
                    "for similarity, not executed, and are not counted here",
            "per_arm": arms,
        },
        "excluded": {
            "generations": excl_gens, "arms": len(excl_arms),
            "directories": EXCLUDED_DIRS,
            "note": "superseded runs, kept for audit and not counted in the census",
        },
        "selection_decay": decay_slopes(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    c, d = res["census"], res["selection_decay"]
    print(f"{c['generations']:,} generations across {c['arms']} arms "
          f"({excl_gens:,} more in {len(excl_arms)} superseded arms)")
    for name, v in d["by_read"].items():
        mark = "<- published" if name == d["published_read"] else ""
        print(f"  {name:16s} {v['slope_per_decade']:+.4f} per decade  "
              f"r={v['r']:+.3f}  n={v['n_rungs']}  {mark}")
    for r in d["by_read"][d["published_read"]]["rungs"]:
        print(f"    {r['params_b']:>5.1f}B  gain {r['gain']:+.4f}  floor {r['floor']:.4f}"
              f"  selected {r['selected']:.4f}  headroom {r['headroom_captured']:.1%}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
