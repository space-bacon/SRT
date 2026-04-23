#!/usr/bin/env python3
"""Render reliability diagram (PNG) from regime_calibration JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text())
    bins = [b for b in data["bins"] if b["n"] > 0]
    mids = [(b["lo"] + b["hi"]) / 2 for b in bins]
    pred = [b["mean_pred"] for b in bins]
    emp = [b["empirical_pos_rate"] for b in bins]
    counts = [b["n"] for b in bins]
    total = sum(counts)
    weights = [c / total for c in counts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
    ax1.plot(pred, emp, "o-", color="#1f6feb", lw=2, ms=7,
             label=f"adapter (ECE={data['ece']:.4f})")
    ax1.set_ylabel("empirical P(supercritical)")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_title(
        f"Regime-head calibration  (N={data['n_tokens']:,}  "
        f"AUROC={data['auroc']:.4f}  Brier={data['brier']:.4f}  "
        f"base={data['base_rate']:.3f})"
    )

    ax2.bar(mids, weights, width=1.0 / data["n_bins"] * 0.9,
            color="#1f6feb", alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("predicted P(supercritical)")
    ax2.set_ylabel("token frac (log)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
