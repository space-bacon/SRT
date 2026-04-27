#!/usr/bin/env python3
"""Layer-as-organ heatmap from interiority readouts.

Inputs:
  artifacts/interiority/v1/readouts.jsonl   (one row per probe prompt)

Outputs:
  artifacts/interiority/v1/summary.json     (per-class means + class-vs-class MI proxy)
  artifacts/interiority/v1/heatmap.png      (regime × channel heatmap)
  artifacts/interiority/v1/heatmap.svg      (vector version)

Channels (one column per):
  - div_L<i>    mean ‖divergence‖ at MAH layer i
  - inj_L<i>    mean ‖injection‖ at inject layer i
  - r_hat       mean BEN scalar reflexivity
  - P(super)    mean supercritical prob
  - regime_H    mean regime classifier entropy
  - chain_res   mean chain residual
  - comm_top1   mean community top-1 weight
  - comm_H      mean community entropy

Each cell is z-scored *within column* across regimes so colors are comparable
(otherwise div norms dominate). Cell value displayed is the raw mean.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SITE_BG = "#0f0f1a"
SITE_BG2 = "#181828"
SITE_FG = "#e8e8f0"
SITE_MUTED = "#9a9ab0"
SITE_PINK = "#ff5e8a"
SITE_VIOLET = "#c376c6"
SITE_BLUE = "#6e8cff"
SITE_BORDER = "#2a2a40"

# diverging colormap matching the demo palette
SITE_CMAP = LinearSegmentedColormap.from_list(
    "srt_diverging",
    [SITE_BLUE, "#3a3a55", SITE_BG2, "#5a3a55", SITE_PINK],
    N=256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--readouts", default="artifacts/interiority/v1/readouts.jsonl")
    p.add_argument("--out-dir", default="artifacts/interiority/v1")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def channel_matrix(rows: list[dict]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Return (regime_labels, channel_labels, mean_matrix, std_matrix).

    mean_matrix[r, c] = mean of channel c across all prompts in regime r
    std_matrix[r, c]  = std  of channel c across all prompts in regime r
    """
    n_div = len(rows[0]["div_norm_mean"])
    n_inj = len(rows[0]["inj_norm_mean"])

    channel_labels: list[str] = []
    channel_labels += [f"div L{rows[0]['mah_layers'][i]}" for i in range(n_div)]
    channel_labels += [f"inj L{rows[0]['inject_layers'][i]}" for i in range(n_inj)]
    extras = [
        ("r_hat", "r_hat_mean"),
        ("P(super)", "p_supercritical_mean"),
        ("regime H", "regime_entropy_mean"),
        ("chain res", "chain_residual_mean"),
        ("comm norm", "community_norm"),
    ]
    channel_labels += [name for name, _ in extras]

    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        vec: list[float] = []
        vec += list(row["div_norm_mean"])
        vec += list(row["inj_norm_mean"])
        for _name, key in extras:
            vec.append(float(row.get(key, float("nan"))))
        by_label[row["label"]].append(vec)

    regime_labels = sorted(by_label.keys())
    M = np.array([np.nanmean(np.array(by_label[k]), axis=0) for k in regime_labels])
    S = np.array([np.nanstd(np.array(by_label[k]), axis=0) for k in regime_labels])
    return regime_labels, channel_labels, M, S


def col_zscore(M: np.ndarray) -> np.ndarray:
    mu = np.nanmean(M, axis=0, keepdims=True)
    sd = np.nanstd(M, axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (M - mu) / sd


def community_separation(rows: list[dict]) -> dict:
    """Quantify how cleanly the v8a community vector separates regimes.

    Returns: regimes, per-class centroid norms, mean intra/inter distance,
    separation ratio, top/bottom centroid-pair distances.
    """
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        v = row.get("community_vector")
        if v is None:
            continue
        by_label[row["label"]].append(v)
    if not by_label:
        return {"available": False, "reason": "no community_vector in readouts"}

    regimes = sorted(by_label.keys())
    X = {r: np.asarray(by_label[r], dtype=np.float64) for r in regimes}
    centroids = {r: X[r].mean(axis=0) for r in regimes}
    grand = np.concatenate([X[r] for r in regimes], axis=0).mean(axis=0)

    intra = {r: float(np.linalg.norm(X[r] - centroids[r], axis=1).mean()) for r in regimes}
    pairs = []
    for i, a in enumerate(regimes):
        for b in regimes[i + 1:]:
            pairs.append({
                "a": a, "b": b,
                "dist": float(np.linalg.norm(centroids[a] - centroids[b])),
            })
    pairs.sort(key=lambda p: p["dist"], reverse=True)
    return {
        "available": True,
        "regimes": regimes,
        "centroid_norms": {r: float(np.linalg.norm(centroids[r])) for r in regimes},
        "centroid_dist_from_grand": {
            r: float(np.linalg.norm(centroids[r] - grand)) for r in regimes
        },
        "mean_intra_dist": intra,
        "mean_intra_dist_overall": float(np.mean(list(intra.values()))),
        "mean_inter_centroid_dist": float(np.mean([p["dist"] for p in pairs])),
        "separation_ratio":
            float(np.mean([p["dist"] for p in pairs]))
            / max(float(np.mean(list(intra.values()))), 1e-9),
        "top_pairs": pairs[:10],
        "bottom_pairs": pairs[-10:],
    }


def render_heatmap(regimes, channels, M, Z, out_png, out_svg):
    n_r, n_c = M.shape
    fig_w = max(8.0, 0.55 * n_c + 3.5)
    fig_h = max(5.0, 0.45 * n_r + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=SITE_BG)
    ax.set_facecolor(SITE_BG)

    vmax = float(np.nanmax(np.abs(Z)))
    vmax = max(vmax, 1e-3)
    im = ax.imshow(Z, aspect="auto", cmap=SITE_CMAP, vmin=-vmax, vmax=vmax)

    # cell text: raw mean
    for r in range(n_r):
        for c in range(n_c):
            v = M[r, c]
            if not np.isfinite(v):
                txt = ""
            elif abs(v) >= 100:
                txt = f"{v:.0f}"
            elif abs(v) >= 1:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.3f}"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8, color=SITE_FG)

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(channels, rotation=40, ha="right", color=SITE_FG, fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(regimes, color=SITE_FG, fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(SITE_BORDER)

    ax.tick_params(colors=SITE_MUTED)
    ax.set_title("layer-as-organ map · v8a · per-class column z-score, cell shows raw mean",
                 color=SITE_FG, fontsize=11, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("column z-score", color=SITE_FG, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=SITE_MUTED, labelcolor=SITE_FG)
    cbar.outline.set_edgecolor(SITE_BORDER)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SITE_BG)
    fig.savefig(out_svg, facecolor=SITE_BG)
    plt.close(fig)


def render_community_pca(rows: list[dict], out_png: str, out_svg: str) -> None:
    """2-D PCA of per-prompt community vectors, colored by regime."""
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        v = row.get("community_vector")
        if v is None:
            continue
        by_label[row["label"]].append(v)
    if not by_label:
        return
    regimes = sorted(by_label.keys())
    X_list, y_list = [], []
    for r in regimes:
        X_list.append(np.asarray(by_label[r], dtype=np.float64))
        y_list.extend([r] * len(by_label[r]))
    X = np.concatenate(X_list, axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Xc @ Vt[:2].T  # (N, 2)
    var_explained = (S[:2] ** 2 / (S ** 2).sum()).tolist()

    # palette: cycle through site colors + matplotlib tab10 for the rest
    base_palette = [SITE_PINK, SITE_BLUE, SITE_VIOLET, "#ffd166", "#06d6a0",
                    "#9b5de5", "#f15bb5", "#fee440", "#00bbf9", "#00f5d4",
                    "#ef476f"]
    color_for = {r: base_palette[i % len(base_palette)] for i, r in enumerate(regimes)}

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=SITE_BG)
    ax.set_facecolor(SITE_BG)
    for r in regimes:
        sel = [i for i, lab in enumerate(y_list) if lab == r]
        ax.scatter(PC[sel, 0], PC[sel, 1], s=42, alpha=0.78,
                   color=color_for[r], edgecolor=SITE_BG, linewidth=0.6,
                   label=r)
        # centroid marker
        cx, cy = PC[sel, 0].mean(), PC[sel, 1].mean()
        ax.scatter([cx], [cy], s=180, color=color_for[r],
                   edgecolor=SITE_FG, linewidth=1.2, marker="X", zorder=5)

    ax.set_xlabel(f"PC1 ({100*var_explained[0]:.1f}% var)", color=SITE_FG)
    ax.set_ylabel(f"PC2 ({100*var_explained[1]:.1f}% var)", color=SITE_FG)
    ax.set_title("community channel · PCA of per-prompt vectors · v8a",
                 color=SITE_FG, fontsize=11, pad=12)
    ax.tick_params(colors=SITE_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(SITE_BORDER)
    leg = ax.legend(loc="best", fontsize=8, facecolor=SITE_BG2,
                    edgecolor=SITE_BORDER, labelcolor=SITE_FG, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SITE_BG)
    fig.savefig(out_svg, facecolor=SITE_BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.readouts))
    regimes, channels, M, S = channel_matrix(rows)
    Z = col_zscore(M)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_rows": len(rows),
        "regimes": regimes,
        "channels": channels,
        "mean": M.tolist(),
        "std": S.tolist(),
        "z": Z.tolist(),
        "rank_per_channel": {},
        "community_separation": community_separation(rows),
    }
    # for each channel, which regime ranks highest / lowest in z?
    for c, name in enumerate(channels):
        order = np.argsort(-Z[:, c])
        summary["rank_per_channel"][name] = [
            {"regime": regimes[i], "z": float(Z[i, c]), "mean": float(M[i, c])}
            for i in order
        ]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_heatmap(regimes, channels, M, Z,
                   str(out_dir / "heatmap.png"),
                   str(out_dir / "heatmap.svg"))
    render_community_pca(rows,
                         str(out_dir / "community_pca.png"),
                         str(out_dir / "community_pca.svg"))

    # console highlights
    print(f"n_rows={len(rows)}  regimes={len(regimes)}  channels={len(channels)}")
    for c, name in enumerate(channels):
        top = summary["rank_per_channel"][name][0]
        bot = summary["rank_per_channel"][name][-1]
        print(f"  {name:12s}  high={top['regime']:18s} z={top['z']:+.2f}   "
              f"low={bot['regime']:18s} z={bot['z']:+.2f}")

    sep = summary["community_separation"]
    if sep.get("available"):
        print()
        print("community separation (continuous 64-D vector):")
        print(f"  mean intra-class radius     = {sep['mean_intra_dist_overall']:.3f}")
        print(f"  mean inter-centroid dist    = {sep['mean_inter_centroid_dist']:.3f}")
        print(f"  separation ratio (inter/intra) = {sep['separation_ratio']:.2f}")
        print("  top centroid pairs (most distant):")
        for p in sep["top_pairs"][:5]:
            print(f"    {p['a']:18s} <-> {p['b']:18s}  d={p['dist']:.3f}")
        print("  bottom centroid pairs (closest):")
        for p in sep["bottom_pairs"][:5]:
            print(f"    {p['a']:18s} <-> {p['b']:18s}  d={p['dist']:.3f}")


if __name__ == "__main__":
    main()
