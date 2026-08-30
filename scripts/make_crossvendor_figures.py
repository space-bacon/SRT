#!/usr/bin/env python3
"""Figures for paper_crossvendor.md.

Every value is READ from the banked artifact, not transcribed into this file.
The sibling generator for the program paper states that its numbers are "copied
from the banked artifacts", which is exactly the arrangement that put three
stale figures on live model cards in one day. A figure that reads its own source
cannot drift from the paper, and if an artifact is regenerated the plot changes
with it.

    python scripts/make_crossvendor_figures.py --out arxiv_crossvendor/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIOLET = "#7a5fd0"       # darkened brand violet, legible on white
VIOLET_LT = "#a48dff"
INK = "#1c1a47"
MUTED = "#8a86a8"
GREY = "#c9c6dd"
RED = "#c0504d"
GREEN = "#3f8f5f"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

A = Path("artifacts/nla")
SHORT = {"qwen3omni": "Qwen3-Omni", "gemma4": "Gemma-4", "aria": "Aria",
         "mistral": "Mistral"}


def load(p):
    return json.load(open(p))


def fig1_transport(out: Path) -> None:
    """Every cross direction against the probe the target trained for itself.

    The point of the figure is the sign of each bar, so the zero line is the
    target's own native score rather than an arbitrary axis floor.
    """
    d = load(A / "cxr14_transport.json")
    items = sorted(d["transported"].items(), key=lambda kv: kv[1]["vs_native_target"])
    labels, deltas, scores = [], [], []
    for k, v in items:
        src, dst = k.split(" probe on ")
        # "A -> B" alone reads as data flow. Spell out which side is the probe.
        labels.append(f"{SHORT.get(src, src)} probe  \u2192  {SHORT.get(dst, dst)} states")
        deltas.append(v["vs_native_target"])
        scores.append(v["mean_auroc"])

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [GREEN if x > 0 else RED for x in deltas]
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colors, height=0.62)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for i, (dv, sc) in enumerate(zip(deltas, scores)):
        ax.text(dv + (0.0012 if dv > 0 else -0.0012), i, f"{sc:.4f}",
                va="center", ha="left" if dv > 0 else "right",
                fontsize=8.5, color=INK)
    ax.set_xlabel("mean AUROC relative to the target backbone's own probe")
    n_win = sum(1 for x in deltas if x > 0)
    ax.set_title(f"A probe read on a backbone it was never fitted on\n"
                 f"{n_win} of {len(deltas)} directions beat the target's own probe",
                 color=INK, fontsize=11, loc="left")
    lim = max(abs(min(deltas)), max(deltas)) * 1.45
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out / "fig1_transport.png", bbox_inches="tight")
    plt.close(fig)


def fig2_pooling(out: Path) -> None:
    """Pooling against each backbone and against the split-matched baseline."""
    d = load(A / "cxr14_ensemble3.json")
    base = d["split_matched_baseline"]
    ctrl_key = next(k for k in d["control"] if k.startswith("concat_"))
    singles = sorted(d["single"].items(), key=lambda kv: -kv[1])

    names = [SHORT.get(k, k) for k, _ in singles] + \
            ["mean of the\nthree logits", "concatenated\nfeatures",
             "CONTROL\nbest + itself"]
    vals = [v for _, v in singles] + [d["ensemble"]["logit_mean"],
                                      d["ensemble"]["concat"],
                                      d["control"][ctrl_key]]
    cols = [GREY] * len(singles) + [VIOLET, MUTED, MUTED]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.62)
    ax.axhline(base, color=RED, lw=1.2, ls="--")
    ax.text(len(vals) - 0.4, base + 0.0018,
            f"split-matched baseline {base}", color=RED, fontsize=8.5, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0018, f"{v:.4f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("mean AUROC, 14 findings, official split")
    ax.set_ylim(0.68, max(vals) + 0.014)

    # The concat bar means nothing without its duplicate-vendor control beside
    # it, so the gap between them is annotated rather than left to the eye.
    gap = abs(d["ensemble"]["concat"] - d["control"][ctrl_key])
    i, j = len(vals) - 2, len(vals) - 1
    top = max(vals[i], vals[j]) + 0.006
    ax.plot([i, i, j, j], [top, top + 0.002, top + 0.002, top], color=MUTED, lw=1)
    ax.text((i + j) / 2, top + 0.003, f"differ by {gap:.4f}",
            ha="center", fontsize=8, color=MUTED)

    ax.set_title("Averaging three probes' logits adds no parameters and wins\n"
                 "Aria is behind the baseline alone and still improves the mean",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_pooling.png", bbox_inches="tight")
    plt.close(fig)


def fig3_cost(out: Path) -> None:
    """Native, transported and floor side by side in both labelled domains."""
    sat = load(A / "omni/rsicd_scene_probe.json")["summary"]
    med = load(A / "cxr14_transport.json")["summary"]
    panels = [("Satellite, 17 land-use classes", sat, "transport_cost"),
              ("Chest radiographs, 14 findings", med, "transport_cost")]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, (title, s, ck) in zip(axes, panels):
        keys = ["mean_native", "mean_self_map", "mean_transported", "mean_shuffled"]
        labs = ["native", "self-map\ncontrol", "transported", "shuffled\nfloor"]
        vals = [s[k] for k in keys]
        cols = [GREY, GREY, VIOLET, MUTED]
        ax.bar(np.arange(len(vals)), vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labs, fontsize=8.5)
        ax.set_ylim(0.4, 1.02)
        cost = s[ck]
        sign = "costs" if cost > 0 else "gains"
        ax.set_title(f"{title}\ntransport {sign} {abs(cost):.4f}",
                     color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("mean AUROC")
    fig.tight_layout()
    fig.savefig(out / "fig3_transport_cost.png", bbox_inches="tight")
    plt.close(fig)


def fig4_ladder(out: Path) -> None:
    """Degradation against how many distinct vendor boundaries a route crosses.

    Area is held at zero and every edge is retraced in all three routes, so the
    only variable left is the number of distinct boundaries.
    """
    doms = [("Radiology", "omni/holonomy_palindrome_roco.json"),
            ("Satellite", "omni/holonomy_palindrome_rsicd.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (name, path) in zip(axes, doms):
        lad = load(A / path)["edge_count_ladder"]
        for i, (edge, series) in enumerate(sorted(lad.items())):
            hops = sorted(series, key=int)
            ys = [series[h]["r@1"] for h in hops]
            ax.plot([int(h) for h in hops], ys, marker="o", ms=4.5,
                    color=[VIOLET, VIOLET_LT, MUTED][i % 3],
                    label=edge.replace(" distinct edge", " edge"))
        ax.set_xticks([int(h) for h in hops])
        ax.set_xlabel("hops (matched across routes)")
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("r@1 after the round trip")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("More distinct vendor boundaries, more degradation, in both domains",
                 color=INK, fontsize=11, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig4_ladder.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arxiv_crossvendor/figs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_transport, fig2_pooling, fig3_cost, fig4_ladder):
        fn(out)
        print(f"  {fn.__name__}")
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
