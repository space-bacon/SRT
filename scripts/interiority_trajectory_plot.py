#!/usr/bin/env python3
"""Render per-position trajectory plots from interiority_trajectory.py output.

For each channel (r_hat, P(super), regime entropy, inj L14/L21, div L7/L14/L21,
chain residual), draw small-multiples: one subplot per regime, showing
mean \u00b1 1\u03c3 across the regime's 25 prompts as a function of relative token
position (0 \u2192 1 along the prompt).

Also writes summary.json with per-regime peak position, peak value, and
"local-vs-global" score (peak / mean).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# site palette
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
    p.add_argument("--n-bins", type=int, default=20,
                   help="Number of relative-position bins (0..1).")
    return p.parse_args()


def relbin(arr: list[float], n_bins: int) -> np.ndarray:
    """Resample variable-length per-token array onto n_bins evenly-spaced positions."""
    a = np.asarray(arr, dtype=np.float64)
    if len(a) == 0:
        return np.full(n_bins, np.nan)
    if len(a) == 1:
        return np.full(n_bins, a[0])
    src_x = np.linspace(0.0, 1.0, len(a))
    dst_x = np.linspace(0.0, 1.0, n_bins)
    return np.interp(dst_x, src_x, a)


def collect_channels(rows: list[dict]) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Return {channel_name: [(label, per_token_array), ...]}."""
    chans: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for r in rows:
        lbl = r["label"]
        if "r_hat" in r:
            chans["r_hat"].append((lbl, np.asarray(r["r_hat"], dtype=np.float64)))
        if "p_supercritical" in r:
            chans["P(super)"].append((lbl, np.asarray(r["p_supercritical"], dtype=np.float64)))
        if "regime_entropy" in r:
            chans["regime H"].append((lbl, np.asarray(r["regime_entropy"], dtype=np.float64)))
        if "chain_residual" in r:
            chans["chain res"].append((lbl, np.asarray(r["chain_residual"], dtype=np.float64)))
        if "inj_norm_per_layer" in r:
            inj_layers = r.get("inject_layers", [])
            for li, vals in enumerate(r["inj_norm_per_layer"]):
                name = f"inj L{inj_layers[li]}" if li < len(inj_layers) else f"inj#{li}"
                chans[name].append((lbl, np.asarray(vals, dtype=np.float64)))
        if "div_norm_per_layer" in r:
            mah_layers = r.get("mah_layers", [])
            for li, vals in enumerate(r["div_norm_per_layer"]):
                name = f"div L{mah_layers[li]}" if li < len(mah_layers) else f"div#{li}"
                chans[name].append((lbl, np.asarray(vals, dtype=np.float64)))
    return chans


def regime_curves(channel_data: list[tuple[str, np.ndarray]],
                  n_bins: int) -> dict[str, dict]:
    """Per regime: mean curve, std curve, n, peak position, peak value."""
    by_regime: dict[str, list[np.ndarray]] = defaultdict(list)
    for lbl, arr in channel_data:
        if arr.size == 0:
            continue
        by_regime[lbl].append(relbin(arr, n_bins))
    result: dict[str, dict] = {}
    for r, curves in by_regime.items():
        M = np.stack(curves, axis=0)               # (n_prompts, n_bins)
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
        peak_idx = int(np.nanargmax(mu))
        result[r] = {
            "n": len(curves),
            "mean": mu.tolist(),
            "std": sd.tolist(),
            "peak_pos": peak_idx / max(n_bins - 1, 1),
            "peak_value": float(mu[peak_idx]),
            "mean_value": float(np.nanmean(mu)),
            "peak_over_mean": float(mu[peak_idx] / max(np.nanmean(mu), 1e-9)),
        }
    return result


def render_channel(channel_name: str,
                   per_regime: dict[str, dict],
                   n_bins: int,
                   out_png: str, out_svg: str) -> None:
    regimes = sorted(per_regime.keys())
    n = len(regimes)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.0 * cols, 2.1 * rows),
                             facecolor=SITE_BG, sharex=True)
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    x = np.linspace(0.0, 1.0, n_bins)

    # global y-range for comparability
    ymins, ymaxs = [], []
    for r in regimes:
        mu = np.asarray(per_regime[r]["mean"])
        sd = np.asarray(per_regime[r]["std"])
        ymins.append(float(np.nanmin(mu - sd)))
        ymaxs.append(float(np.nanmax(mu + sd)))
    y_lo, y_hi = min(ymins), max(ymaxs)
    pad = 0.05 * (y_hi - y_lo + 1e-9)
    y_lo -= pad
    y_hi += pad

    for i, r in enumerate(regimes):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor(SITE_BG2)
        mu = np.asarray(per_regime[r]["mean"])
        sd = np.asarray(per_regime[r]["std"])
        ax.fill_between(x, mu - sd, mu + sd, color=SITE_BLUE, alpha=0.18, linewidth=0)
        ax.plot(x, mu, color=SITE_PINK, linewidth=1.6)
        ax.axvline(per_regime[r]["peak_pos"], color=SITE_VIOLET,
                   linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{r}  n={per_regime[r]['n']}  P/M={per_regime[r]['peak_over_mean']:.2f}",
                     color=SITE_FG, fontsize=8.5, pad=4)
        ax.tick_params(colors=SITE_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(SITE_BORDER)

    # blank unused
    for j in range(n, rows * cols):
        ax = axes[j // cols][j % cols]
        ax.set_facecolor(SITE_BG)
        ax.axis("off")

    fig.suptitle(f"trajectory \u00b7 {channel_name} \u00b7 mean \u00b1 1\u03c3 across prompts \u00b7 v8a",
                 color=SITE_FG, fontsize=11)
    fig.text(0.5, 0.02, "relative token position (0 \u2192 1)",
             color=SITE_MUTED, fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_png, dpi=140, facecolor=SITE_BG)
    fig.savefig(out_svg, facecolor=SITE_BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in Path(args.readouts).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} rows")

    chans = collect_channels(rows)
    print(f"channels: {list(chans.keys())}")

    summary: dict[str, dict] = {
        "n_rows": len(rows),
        "n_bins": args.n_bins,
        "channels": {},
    }
    for name, data in chans.items():
        per_regime = regime_curves(data, args.n_bins)
        summary["channels"][name] = per_regime
        slug = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        render_channel(name, per_regime, args.n_bins,
                       str(out_dir / f"trajectory_{slug}.png"),
                       str(out_dir / f"trajectory_{slug}.svg"))
        print(f"  rendered {name}: {len(per_regime)} regimes")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # compact ranking: which regimes have the most local (peak/mean) signal in each channel?
    print()
    print("local-vs-global (peak / mean) per channel \u2014 top 3 most-local regimes:")
    for name, per_regime in summary["channels"].items():
        ranked = sorted(per_regime.items(),
                        key=lambda kv: kv[1]["peak_over_mean"], reverse=True)
        top = ", ".join(f"{r}={d['peak_over_mean']:.2f}@{d['peak_pos']:.2f}"
                        for r, d in ranked[:3])
        print(f"  {name:12s}  {top}")


if __name__ == "__main__":
    main()
