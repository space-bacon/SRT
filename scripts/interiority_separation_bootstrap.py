#!/usr/bin/env python3
"""Bootstrap CIs on the community separation_ratio and per-pair centroid distances.

Reuses the v2 readouts: each row has `label` + `community_vector` (64-D).
For B bootstrap iterations, resample prompts within each regime with
replacement, recompute centroids and separation_ratio, and report
percentile CIs.

Also bootstraps:
  - top-distant centroid pairs (does code <-> literal stay #1?)
  - top-close centroid pairs (does lyric <-> metaphor stay closest?)
  - 1-NN classification accuracy via leave-one-prompt-out + bootstrap.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--readouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_by_label(path: str) -> dict[str, np.ndarray]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by: dict[str, list[list[float]]] = defaultdict(list)
    for r in rows:
        v = r.get("community_vector")
        if v is None:
            continue
        by[r["label"]].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in by.items()}


def separation(X_by_lbl: dict[str, np.ndarray]) -> tuple[float, dict[tuple[str, str], float], dict[str, float]]:
    """Return (separation_ratio, pair_distances, intra_distances)."""
    regimes = sorted(X_by_lbl.keys())
    centroids = {r: X_by_lbl[r].mean(axis=0) for r in regimes}
    intra = {r: float(np.linalg.norm(X_by_lbl[r] - centroids[r], axis=1).mean())
             for r in regimes}
    pairs: dict[tuple[str, str], float] = {}
    for i, a in enumerate(regimes):
        for b in regimes[i + 1:]:
            pairs[(a, b)] = float(np.linalg.norm(centroids[a] - centroids[b]))
    sep = float(np.mean(list(pairs.values())) / max(np.mean(list(intra.values())), 1e-9))
    return sep, pairs, intra


def nn_accuracy(X_by_lbl: dict[str, np.ndarray]) -> float:
    """Leave-one-out 1-NN accuracy: each prompt vs. centroids of all classes
    (its own class centroid is computed without it)."""
    regimes = sorted(X_by_lbl.keys())
    correct = 0
    total = 0
    for true_r in regimes:
        Xr = X_by_lbl[true_r]
        n = Xr.shape[0]
        for i in range(n):
            # centroid of true class without sample i
            mask = np.ones(n, dtype=bool); mask[i] = False
            this_centroid = Xr[mask].mean(axis=0)
            best_r, best_d = None, float("inf")
            for r in regimes:
                c = this_centroid if r == true_r else X_by_lbl[r].mean(axis=0)
                d = float(np.linalg.norm(Xr[i] - c))
                if d < best_d:
                    best_d, best_r = d, r
            if best_r == true_r:
                correct += 1
            total += 1
    return correct / total


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    X_by_lbl = load_by_label(args.readouts)
    regimes = sorted(X_by_lbl.keys())
    n_per = {r: X_by_lbl[r].shape[0] for r in regimes}
    print(f"loaded {sum(n_per.values())} prompts across {len(regimes)} regimes "
          f"(per-regime n: {min(n_per.values())}..{max(n_per.values())})")

    # point estimates
    sep_pt, pairs_pt, intra_pt = separation(X_by_lbl)
    nn_pt = nn_accuracy(X_by_lbl)
    print(f"point separation_ratio = {sep_pt:.4f}")
    print(f"point 1-NN accuracy    = {nn_pt:.4f}  (chance = {1.0/len(regimes):.4f})")

    # bootstrap
    seps = np.empty(args.n_boot, dtype=np.float64)
    nns = np.empty(args.n_boot, dtype=np.float64)
    pair_dists = {p: np.empty(args.n_boot, dtype=np.float64) for p in pairs_pt}
    for b in range(args.n_boot):
        boot = {r: X_by_lbl[r][rng.integers(0, n_per[r], size=n_per[r])] for r in regimes}
        s, pp, _ = separation(boot)
        seps[b] = s
        for p, d in pp.items():
            pair_dists[p][b] = d
        # nn_accuracy on bootstrap is expensive; subsample
        if b < 200:
            nns[b] = nn_accuracy(boot)

    nn_lo, nn_hi = float(np.percentile(nns[:200], 2.5)), float(np.percentile(nns[:200], 97.5))
    sep_lo, sep_hi = float(np.percentile(seps, 2.5)), float(np.percentile(seps, 97.5))

    # rank stability
    pair_pt_sorted = sorted(pairs_pt.items(), key=lambda kv: kv[1], reverse=True)
    top1 = pair_pt_sorted[0]
    bottom1 = pair_pt_sorted[-1]
    # how often does top1 stay #1 in bootstrap?
    top1_stays = 0
    bottom1_stays = 0
    for b in range(args.n_boot):
        boot_dists = {p: pair_dists[p][b] for p in pairs_pt}
        ranked = sorted(boot_dists.items(), key=lambda kv: kv[1], reverse=True)
        if ranked[0][0] == top1[0]:
            top1_stays += 1
        if ranked[-1][0] == bottom1[0]:
            bottom1_stays += 1

    summary = {
        "n_boot": args.n_boot,
        "n_regimes": len(regimes),
        "n_per_regime": n_per,
        "separation_ratio": {
            "point": sep_pt,
            "mean": float(seps.mean()),
            "std": float(seps.std()),
            "ci95": [sep_lo, sep_hi],
        },
        "nn_accuracy": {
            "point": nn_pt,
            "mean": float(nns[:200].mean()),
            "std": float(nns[:200].std()),
            "ci95": [nn_lo, nn_hi],
            "chance": 1.0 / len(regimes),
            "n_boot_used": 200,
        },
        "top_pair": {
            "pair": list(top1[0]),
            "point_dist": top1[1],
            "stays_top_in_bootstrap": top1_stays / args.n_boot,
        },
        "bottom_pair": {
            "pair": list(bottom1[0]),
            "point_dist": bottom1[1],
            "stays_bottom_in_bootstrap": bottom1_stays / args.n_boot,
        },
        "pair_dist_ci95": {
            f"{a}__{b}": [
                float(np.percentile(pair_dists[(a, b)], 2.5)),
                float(np.percentile(pair_dists[(a, b)], 97.5)),
                float(pairs_pt[(a, b)]),
            ]
            for (a, b) in pairs_pt
        },
    }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print()
    print(f"separation_ratio: {sep_pt:.3f}  CI95 [{sep_lo:.3f}, {sep_hi:.3f}]")
    print(f"1-NN accuracy   : {nn_pt:.3f}  CI95 [{nn_lo:.3f}, {nn_hi:.3f}]   "
          f"(chance = {1.0/len(regimes):.3f})")
    print(f"top pair {top1[0]}  d = {top1[1]:.3f}  "
          f"stays #1 in {top1_stays/args.n_boot*100:.1f}% of bootstraps")
    print(f"bot pair {bottom1[0]}  d = {bottom1[1]:.3f}  "
          f"stays last in {bottom1_stays/args.n_boot*100:.1f}% of bootstraps")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
