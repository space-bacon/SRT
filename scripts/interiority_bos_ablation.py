#!/usr/bin/env python3
"""BOS-sink ablation: re-pool the trajectory readouts with the first token
excluded and compare the regime ranking against v1's BOS-included pooling.

If the regime ranking survives, v1's findings stand. If it collapses, v1
was largely a sink-amplitude artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--readouts", required=True,
                   help="trajectory readouts.jsonl")
    p.add_argument("--out", required=True,
                   help="path for ablation summary JSON")
    p.add_argument("--drop-first", type=int, default=1,
                   help="number of leading tokens to drop before pooling")
    return p.parse_args()


def pool(arr: list[float], drop: int) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size <= drop:
        return float("nan")
    return float(np.mean(a[drop:]))


def pool_with_bos(arr: list[float]) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def zscore(d: dict[str, float]) -> dict[str, float]:
    vals = np.asarray([v for v in d.values()], dtype=np.float64)
    mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
    if sd < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - mu) / sd for k, v in d.items()}


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in Path(args.readouts).read_text().splitlines() if l.strip()]
    drop = args.drop_first

    # collect per-(regime, channel) lists of pooled values
    chans_inc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    chans_exc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def push(name: str, lbl: str, arr: list[float]) -> None:
        chans_inc[name][lbl].append(pool_with_bos(arr))
        chans_exc[name][lbl].append(pool(arr, drop))

    for r in rows:
        lbl = r["label"]
        if "r_hat" in r: push("r_hat", lbl, r["r_hat"])
        if "p_supercritical" in r: push("P(super)", lbl, r["p_supercritical"])
        if "regime_entropy" in r: push("regime H", lbl, r["regime_entropy"])
        if "chain_residual" in r: push("chain res", lbl, r["chain_residual"])
        if "inj_norm_per_layer" in r:
            for li, vals in enumerate(r["inj_norm_per_layer"]):
                name = f"inj L{r['inject_layers'][li]}"
                push(name, lbl, vals)
        if "div_norm_per_layer" in r:
            for li, vals in enumerate(r["div_norm_per_layer"]):
                name = f"div L{r['mah_layers'][li]}"
                push(name, lbl, vals)

    summary: dict = {
        "n_rows": len(rows),
        "drop_first": drop,
        "channels": {},
    }
    print(f"loaded {len(rows)} rows; dropping first {drop} token(s) for ablation")
    print()
    print(f"{'channel':12s}  {'top_inc':22s}  {'top_exc':22s}  {'spearman':>8s}  {'kept':>5s}")
    print("-" * 80)

    for name in chans_inc:
        regime_inc = {r: float(np.nanmean(v)) for r, v in chans_inc[name].items()}
        regime_exc = {r: float(np.nanmean(v)) for r, v in chans_exc[name].items()}
        z_inc = zscore(regime_inc)
        z_exc = zscore(regime_exc)
        rank_inc = sorted(regime_inc.keys(), key=lambda k: regime_inc[k], reverse=True)
        rank_exc = sorted(regime_exc.keys(), key=lambda k: regime_exc[k], reverse=True)

        # spearman rank correlation
        all_regimes = sorted(regime_inc.keys())
        ri = np.asarray([rank_inc.index(r) for r in all_regimes], dtype=float)
        re = np.asarray([rank_exc.index(r) for r in all_regimes], dtype=float)
        ri -= ri.mean(); re -= re.mean()
        spearman = float((ri * re).sum() / (np.sqrt((ri * ri).sum()) * np.sqrt((re * re).sum()) + 1e-12))

        # Δ for top-ranked regime in v1
        top_v1 = rank_inc[0]
        kept = (rank_exc.index(top_v1) == 0)

        summary["channels"][name] = {
            "regime_means_with_bos": regime_inc,
            "regime_means_no_bos": regime_exc,
            "z_with_bos": z_inc,
            "z_no_bos": z_exc,
            "rank_with_bos": rank_inc,
            "rank_no_bos": rank_exc,
            "spearman_rank_corr": spearman,
            "top_regime_with_bos": top_v1,
            "top_regime_no_bos": rank_exc[0],
            "top_kept": kept,
        }

        top_inc_str = f"{top_v1} ({z_inc[top_v1]:+.2f})"
        top_exc_str = f"{rank_exc[0]} ({z_exc[rank_exc[0]]:+.2f})"
        kept_str = "yes" if kept else "NO"
        print(f"{name:12s}  {top_inc_str:22s}  {top_exc_str:22s}  {spearman:+.3f}    {kept_str:>5s}")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print()
    print(f"-> {args.out}")

    # aggregate
    spearmans = [c["spearman_rank_corr"] for c in summary["channels"].values()]
    kept = sum(1 for c in summary["channels"].values() if c["top_kept"])
    print(f"\naggregate: mean spearman = {np.mean(spearmans):+.3f}  "
          f"min = {min(spearmans):+.3f}  "
          f"top-regime kept in {kept}/{len(spearmans)} channels")


if __name__ == "__main__":
    main()
