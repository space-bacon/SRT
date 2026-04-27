#!/usr/bin/env python3
"""Length-scaling analysis of the BOS sink.

For each (channel, n_chunks) cell, compute:
  - mean prompt length T (real tokens)
  - mean BOS amplitude (value at token 0)
  - mean tail amplitude (mean over tokens [1:])
  - sink_share = BOS_value / sum_of_all_tokens   (what fraction of the
    total per-prompt magnitude lives at token 0)

Plot two figures per channel:
  (a) sink_share vs prompt length      \u2014 does the sink saturate?
  (b) BOS value vs prompt length       \u2014 does the spike grow?

Also dumps a JSON table with the same numbers per (channel, n_chunks) bin
and per regime.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SITE_BG = "#0f0f1a"
SITE_BG2 = "#181828"
SITE_FG = "#e8e8f0"
SITE_MUTED = "#9a9ab0"
SITE_BORDER = "#2a2a40"
SITE_PINK = "#ff5e8a"
SITE_BLUE = "#6e8cff"
SITE_VIOLET = "#9b5de5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--readouts", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def channel_arrays(row: dict) -> dict[str, np.ndarray]:
    """Return {channel_name: per-token array} for one row."""
    out: dict[str, np.ndarray] = {}
    if "r_hat" in row: out["r_hat"] = np.asarray(row["r_hat"], dtype=np.float64)
    if "p_supercritical" in row: out["P(super)"] = np.asarray(row["p_supercritical"], dtype=np.float64)
    if "regime_entropy" in row: out["regime H"] = np.asarray(row["regime_entropy"], dtype=np.float64)
    if "chain_residual" in row: out["chain res"] = np.asarray(row["chain_residual"], dtype=np.float64)
    if "inj_norm_per_layer" in row:
        for li, vals in enumerate(row["inj_norm_per_layer"]):
            name = f"inj L{row['inject_layers'][li]}"
            out[name] = np.asarray(vals, dtype=np.float64)
    if "div_norm_per_layer" in row:
        for li, vals in enumerate(row["div_norm_per_layer"]):
            name = f"div L{row['mah_layers'][li]}"
            out[name] = np.asarray(vals, dtype=np.float64)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.readouts).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} rows")

    # group by n_chunks (recovered from id "{label}__nN__kk" if missing)
    def n_of(r: dict) -> int:
        if "n_chunks" in r:
            return int(r["n_chunks"])
        rid = r.get("id", "")
        for part in rid.split("__"):
            if part.startswith("n") and part[1:].isdigit():
                return int(part[1:])
        return 1

    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_n[n_of(r)].append(r)
    n_bins = sorted(by_n.keys())
    print(f"length bins: {n_bins}")
    print(f"  n per bin: {[len(by_n[n]) for n in n_bins]}")
    print(f"  mean T per bin: {[float(np.mean([r['n_tokens'] for r in by_n[n]])) for n in n_bins]}")

    # collect channels by sniffing first row
    chans = list(channel_arrays(rows[0]).keys())

    # per (channel, n_chunks): list of (T, bos_value, tail_mean, sink_share)
    table: dict[str, dict[int, list[tuple[float, float, float, float]]]] = \
        {c: defaultdict(list) for c in chans}

    for r in rows:
        n = n_of(r)
        T = r["n_tokens"]
        if T < 2:
            continue
        ca = channel_arrays(r)
        for c, arr in ca.items():
            if arr.size < 2:
                continue
            bos = float(arr[0])
            tail = float(arr[1:].mean())
            total = float(np.abs(arr).sum())
            share = abs(bos) / max(total, 1e-12)
            table[c][n].append((float(T), bos, tail, share))

    # build summary numbers
    summary = {"n_bins": n_bins, "channels": {}}
    print()
    print(f"{'channel':12s}  {'n_chunks':>9s}  {'mean T':>8s}  "
          f"{'BOS val':>10s}  {'tail mean':>10s}  {'sink share':>11s}")
    print("-" * 80)
    for c in chans:
        per_bin: dict[int, dict] = {}
        for n in n_bins:
            data = np.asarray(table[c][n], dtype=np.float64)
            if data.size == 0:
                continue
            T_mu = float(data[:, 0].mean())
            bos_mu = float(data[:, 1].mean())
            bos_sd = float(data[:, 1].std())
            tail_mu = float(data[:, 2].mean())
            share_mu = float(data[:, 3].mean())
            share_sd = float(data[:, 3].std())
            per_bin[int(n)] = {
                "n_prompts": int(data.shape[0]),
                "mean_tokens": T_mu,
                "bos_value_mean": bos_mu,
                "bos_value_std": bos_sd,
                "tail_mean": tail_mu,
                "sink_share_mean": share_mu,
                "sink_share_std": share_sd,
            }
            print(f"{c:12s}  {n:>9d}  {T_mu:>8.1f}  {bos_mu:>10.3f}  "
                  f"{tail_mu:>10.3f}  {share_mu:>11.3f}")
        summary["channels"][c] = per_bin
        print()

    Path(out_dir / "length_scaling.json").write_text(json.dumps(summary, indent=2))

    # plot: per channel, two-panel (sink_share vs T,  BOS value vs T)
    n = len(chans)
    cols = 3
    rows_g = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_g, cols, figsize=(3.2 * cols, 2.4 * rows_g),
                             facecolor=SITE_BG)
    axes = np.atleast_2d(axes)
    if rows_g == 1: axes = axes.reshape(1, -1)
    if cols == 1: axes = axes.reshape(-1, 1)

    for i, c in enumerate(chans):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor(SITE_BG2)
        Ts, shares, share_sds = [], [], []
        for n_chunks in n_bins:
            d = summary["channels"][c].get(n_chunks)
            if d is None: continue
            Ts.append(d["mean_tokens"])
            shares.append(d["sink_share_mean"])
            share_sds.append(d["sink_share_std"])
        Ts = np.asarray(Ts); shares = np.asarray(shares); share_sds = np.asarray(share_sds)
        ax.errorbar(Ts, shares, yerr=share_sds, color=SITE_PINK,
                    marker="o", linewidth=1.5, capsize=3,
                    ecolor=SITE_VIOLET, elinewidth=1.0)
        # naive 1/T reference (normalised so it passes through first point)
        if len(Ts) > 0 and shares[0] > 0:
            ref = shares[0] * Ts[0] / Ts
            ax.plot(Ts, ref, color=SITE_BLUE, linewidth=1.0, linestyle="--",
                    alpha=0.7, label="1/T reference")
        ax.set_title(c, color=SITE_FG, fontsize=9, pad=3)
        ax.tick_params(colors=SITE_MUTED, labelsize=7)
        ax.set_xlabel("mean prompt length (tokens)", color=SITE_MUTED, fontsize=7)
        ax.set_ylabel("BOS sink share", color=SITE_MUTED, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(SITE_BORDER)
        if i == 0:
            ax.legend(facecolor=SITE_BG2, edgecolor=SITE_BORDER,
                      labelcolor=SITE_FG, fontsize=7, loc="upper right")

    for j in range(n, rows_g * cols):
        axes[j // cols][j % cols].set_facecolor(SITE_BG); axes[j // cols][j % cols].axis("off")

    fig.suptitle("BOS sink share vs prompt length \u00b7 v8a",
                 color=SITE_FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "length_scaling_share.png", dpi=140, facecolor=SITE_BG)
    fig.savefig(out_dir / "length_scaling_share.svg", facecolor=SITE_BG)
    plt.close(fig)

    # second figure: BOS value (raw) vs T
    fig, axes = plt.subplots(rows_g, cols, figsize=(3.2 * cols, 2.4 * rows_g),
                             facecolor=SITE_BG)
    axes = np.atleast_2d(axes)
    if rows_g == 1: axes = axes.reshape(1, -1)
    if cols == 1: axes = axes.reshape(-1, 1)
    for i, c in enumerate(chans):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor(SITE_BG2)
        Ts, vals, sds = [], [], []
        for n_chunks in n_bins:
            d = summary["channels"][c].get(n_chunks)
            if d is None: continue
            Ts.append(d["mean_tokens"])
            vals.append(d["bos_value_mean"])
            sds.append(d["bos_value_std"])
        Ts = np.asarray(Ts); vals = np.asarray(vals); sds = np.asarray(sds)
        ax.errorbar(Ts, vals, yerr=sds, color=SITE_PINK,
                    marker="o", linewidth=1.5, capsize=3,
                    ecolor=SITE_VIOLET, elinewidth=1.0)
        ax.set_title(c, color=SITE_FG, fontsize=9, pad=3)
        ax.tick_params(colors=SITE_MUTED, labelsize=7)
        ax.set_xlabel("mean prompt length (tokens)", color=SITE_MUTED, fontsize=7)
        ax.set_ylabel("BOS value (raw)", color=SITE_MUTED, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(SITE_BORDER)
    for j in range(n, rows_g * cols):
        axes[j // cols][j % cols].set_facecolor(SITE_BG); axes[j // cols][j % cols].axis("off")
    fig.suptitle("BOS-token value (raw) vs prompt length \u00b7 v8a",
                 color=SITE_FG, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "length_scaling_value.png", dpi=140, facecolor=SITE_BG)
    fig.savefig(out_dir / "length_scaling_value.svg", facecolor=SITE_BG)
    plt.close(fig)

    print(f"-> {out_dir}/length_scaling.json")
    print(f"-> {out_dir}/length_scaling_share.png")
    print(f"-> {out_dir}/length_scaling_value.png")


if __name__ == "__main__":
    main()
