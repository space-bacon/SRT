"""Precompute consensus clustering for the banked pools, so the demo runs no code.

The live selector has to execute candidates to cluster them. A public demo must
not, so the clustering is done once here, against generations we already own,
and the result is a lookup table: which candidates agreed with which, what the
selector picked, and (from the banked pass matrix) which candidates actually
passed the hidden tests.

The demo that reads this file never executes anything.

    python scripts/bank_consensus_demo.py --k 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from chat_consensus import code_of, infer_kinds, pick_entry  # noqa: E402
from consensus_select import _value, probe_program, run_capture  # noqa: E402

RUNGS = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]


def discover(gen_dir: str, passmat: str) -> list[tuple[str, str]]:
    """(label, arm) pairs. The size ladder keeps its short labels; any other
    directory, such as the frontier ensemble, is labelled by arm name."""
    ladder = [(r, f"coder{r}_inst__chat") for r in RUNGS]
    if any(os.path.exists(os.path.join(gen_dir, f"{a}.json")) for _, a in ladder):
        return ladder
    out = []
    for f in sorted(glob.glob(os.path.join(gen_dir, "*.json"))):
        arm = os.path.basename(f)[:-5]
        if "__" not in arm:
            continue
        if os.path.exists(os.path.join(passmat, f"{arm}.npz")):
            out.append((arm.split("__")[0], arm))
    return out


def cluster(prompt: str, replies: list[str], n_cases: int, timeout: float, seed: int = 0):
    """choose(), but keeping the per-candidate signature the UI needs."""
    codes = [code_of(r) for r in replies]
    entry = pick_entry(codes, prompt)
    if entry is None:
        return {"ok": False, "reason": "no function defined by any candidate"}
    kinds = infer_kinds(codes, entry, prompt)
    if not kinds:
        return {"ok": False, "reason": "could not infer arguments", "entry": entry}

    rng = random.Random(seed)
    cases = [tuple(_value(k, rng) for k in kinds) for _ in range(n_cases)]
    sigs = [run_capture(probe_program(c, entry, cases), timeout) for c in codes]

    valid = [i for i, s in enumerate(sigs) if s is not None]
    if not valid:
        return {"ok": False, "reason": "no candidate ran", "entry": entry}

    top, votes = Counter(sigs[i] for i in valid).most_common(1)[0]
    pick = next(i for i in valid if sigs[i] == top)

    # Stable small integers for the UI; -1 means the candidate never ran.
    order = [s for s, _ in Counter(sigs[i] for i in valid).most_common()]
    group = {s: g for g, s in enumerate(order)}
    groups = [group[s] if s is not None else -1 for s in sigs]

    return {
        "ok": True,
        "entry": entry,
        "kinds": kinds,
        "pick": pick,
        "groups": groups,
        "cluster_size": votes,
        "n_clusters": len(order),
        "ran": len(valid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-dir", default="artifacts/nla/he_k32")
    ap.add_argument("--passmat", default="artifacts/nla/verifier/passmat_he_k32")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/consensus_demo.json")
    a = ap.parse_args()

    probs = json.load(open(a.probs))
    out = {"k": a.k, "n_cases": a.cases, "bench": "humaneval", "rungs": {}}
    out["problems"] = [
        {"task_id": p["task_id"], "prompt": p["prompt"], "entry_point": p["entry_point"]}
        for p in probs
    ]

    for rung, arm in discover(a.gen_dir, a.passmat):
        gen_p = os.path.join(a.gen_dir, f"{arm}.json")
        mat_p = os.path.join(a.passmat, f"{arm}.npz")
        if not (os.path.exists(gen_p) and os.path.exists(mat_p)):
            print(f"skip {arm}: missing generations or pass matrix", flush=True)
            continue

        rows = json.load(open(gen_p))
        ok = np.load(mat_p)["ok"][:, : a.k]

        def one(i: int):
            return cluster(probs[i]["prompt"], rows[i][: a.k], a.cases, a.timeout)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            sel = list(ex.map(one, range(len(rows))))

        entries = []
        for i, s in enumerate(sel):
            passed = [bool(x) for x in ok[i]]
            rec = {
                "candidates": [code_of(c) for c in rows[i][: a.k]],
                "passed": passed,
                "any_passed": any(passed),
                **s,
            }
            rec["pick_passed"] = passed[s["pick"]] if s.get("ok") else None
            entries.append(rec)

        picked = [e for e in entries if e.get("ok")]
        # Two conventions, because they differ on strong models. `selected_pass`
        # falls back to an arbitrary candidate when the pool does not resolve,
        # as a deployment would and as chat_consensus.py already does.
        # `selected_pass_strict` charges every unresolved pool as a failure,
        # which penalises declining a problem the model may have answered.
        deploy = [
            (1.0 if e["pick_passed"] else 0.0) if e.get("ok")
            else (1.0 if e["passed"][0] else 0.0)
            for e in entries
        ]
        out["rungs"][rung] = {
            "arm": arm,
            "single_sample_pass": float(ok.mean()),
            "oracle_pass": float(ok.any(1).mean()),
            "selected_pass": (sum(deploy) / len(deploy) if deploy else 0.0),
            "selected_pass_strict": (
                sum(1 for e in picked if e["pick_passed"]) / len(entries) if entries else 0.0
            ),
            "resolved": len(picked),
            "entries": entries,
        }
        print(
            f"{arm}: single {ok.mean():.4f} selected "
            f"{out['rungs'][rung]['selected_pass']:.4f} oracle {ok.any(1).mean():.4f} "
            f"({len(picked)}/{len(entries)} resolved)",
            flush=True,
        )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
