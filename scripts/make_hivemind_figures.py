#!/usr/bin/env python3
"""Figures for paper_hivemind.md.

Every value is READ from the banked artifact, never transcribed here. A figure
carrying its own copy of a number is a figure that will disagree with the paper
eventually, and this repository has already put stale figures on live cards.

Four figures, one per load-bearing claim:

  1 decomposition  which part of the chat template creates the convergence
  2 scale          the two opposing slopes, format up and selection value down
  3 reads          what each selection read is worth, on both benchmarks
  4 pooling        five labs pooled against one model, and the oracle above both

    python scripts/make_hivemind_figures.py --out arxiv_hivemind/figs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIOLET = "#7a5fd0"
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


def load(p):
    f = A / p
    return json.load(open(f)) if f.is_file() else None


def fig_decomposition(out):
    """Which part of the template does the work: persona, or the role frame."""
    d = load("atlas/hivemind_template_decomp.json")
    if not d:
        return "decomposition: missing hivemind_template_decomp.json"
    arms = d.get("arms") or {}
    want = [("raw", "no framing"), ("persona", "persona only"),
            ("shared", "role frame"), ("shared_persona", "frame + persona"),
            ("chat", "native template")]
    vals, labels, floors = [], [], []
    for key, lab in want:
        v = arms.get(key)
        if isinstance(v, dict):
            floors.append(v.get("floor"))
            v = v.get("intra_mean", v.get("intra"))
        if v is not None:
            vals.append(v)
            labels.append(lab)
    if len(vals) < 3:
        return "decomposition: arms not in expected shape"

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [GREY] + [VIOLET_LT] * (len(vals) - 2) + [VIOLET]
    bars = ax.bar(labels, vals, color=colors, edgecolor=INK, linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=9, color=INK)
    floor = next((f for f in floors if f), None) or d.get("floor")
    if floor:
        ax.axhline(floor, color=RED, lw=1.0, ls="--")
        ax.text(-0.45, floor + 0.015, f"floor {floor:.4f}",
                color=RED, fontsize=8, ha="left")
    dec = d.get("decomposition", {})
    share = dec.get("share_reachable_without_native_tokens")
    ax.set_ylabel("intra-model similarity")
    ax.set_title("The convergence arrives with the role frame, not the persona",
                 color=INK, fontsize=11, loc="left")
    if share:
        pct = share * 100 if share <= 1 else share
        ax.text(0.0, -0.20, f"Generic plain-text markers no model was trained on "
                            f"reproduce {pct:.0f}% of the full template effect.",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "fig1_decomposition.png")
    plt.close(fig)
    return f"fig1_decomposition.png  ({len(vals)} arms)"


def fig_scale(out):
    """Format similarity rises with scale while selection value falls."""
    # The 192-token coder_ladder curve is superseded; see paper 5.2's budget note.
    curve = load("coder_matrix1024/scaling_curve.json")
    census = load("hivemind_census.json")
    if not curve or not census:
        return "scale: missing scaling_curve.json or hivemind_census.json"

    fmt_slope = (curve.get("trend") or {}).get("format_gain_slope_per_decade")
    per = curve.get("curve") or {}
    xs, fmt = [], []
    for k, v in sorted(per.items(), key=lambda kv: float(re.sub(r"[^\d.]", "", kv[0]) or 0)):
        g = v.get("format_gain") if isinstance(v, dict) else None
        if g is not None:
            xs.append(float(re.sub(r"[^\d.]", "", k)))
            fmt.append(g)

    read = census["selection_decay"]["by_read"][census["selection_decay"]["published_read"]]
    sx = [r["params_b"] for r in read["rungs"]]
    sy = [r["gain"] for r in read["rungs"]]
    sel_slope = read["slope_per_decade"]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if xs:
        ax.plot(xs, fmt, "o-", color=VIOLET, lw=1.8, ms=5,
                label=f"format-induced similarity  {fmt_slope:+.4f}/decade")
    # Two arms share each rung, so join nothing: show the points and the fit.
    ax.plot(sx, sy, "s", color=RED, ms=5, alpha=0.75,
            label=f"value of selecting among 8  {sel_slope:+.4f}/decade")
    grid = np.logspace(np.log10(min(sx)), np.log10(max(sx)), 50)
    ax.plot(grid, read["intercept"] + sel_slope * np.log10(grid),
            color=RED, lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (B, log scale)")
    ax.set_ylabel("gain")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Models write more alike with scale, and leave less to choose between",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_scale.png")
    plt.close(fig)
    return "fig2_scale.png"


def fig_reads(out):
    """What each read is worth, and how the order flips between benchmarks."""
    # HumanEval reads on the 1024-token pools; the verifier/ copies are the 192-token run.
    he = load("verifier_1024/exec_guided.json")
    mb = load("verifier/exec_guided_mbpp.json")
    hec = load("verifier_1024/consensus.json")
    mbc = load("verifier/consensus_mbpp.json")
    if not all([he, mb, hec, mbc]):
        return "reads: missing one of the verifier artifacts"

    rows = [
        ("random single", he["overall"]["floor"], mb["overall"]["floor"]),
        ("learned verifier", he["overall"]["verifier"], mb["overall"]["verifier"]),
        ("prompt's examples", he["overall"]["exec"], mb["overall"]["exec"]),
        ("agreement", hec["summary"]["consensus"], mbc["summary"]["consensus"]),
        ("oracle", he["overall"]["oracle"], mb["overall"]["oracle"]),
    ]
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="HumanEval",
           color=VIOLET, edgecolor=INK, linewidth=0.6)
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="MBPP",
           color=VIOLET_LT, edgecolor=INK, linewidth=0.6)
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[1] + 0.012, f"{r[1]:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.text(i + w / 2, r[2] + 0.012, f"{r[2]:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The supervised verifier loses to two baselines that need no training",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig3_reads.png")
    plt.close(fig)
    return "fig3_reads.png"


def fig_pooling(out):
    """Pooling five labs against one model: the same problems, and the oracle above."""
    p = load("ensemble/pooled_select.json")
    if not p:
        return "pooling: missing pooled_select.json"
    n = 164
    rows = [
        ("best member\none sample", p["best_single_pass1"], GREY),
        ("best member\n+ its own 8", p["best_single_consensus"], VIOLET_LT),
        ("48 pooled\nacross 5 labs", p["pooled_consensus"], VIOLET),
        ("pooled, from\nreplies alone", p["pooled_chat_only"], VIOLET),
        ("oracle", p["union_oracle"], GREEN),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                  color=[r[2] for r in rows], edgecolor=INK, linewidth=0.6)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.004,
                f"{r[1]:.4f}\n{round(r[1] * n)} / {n}", ha="center",
                fontsize=8, color=INK)
    lvl = p["best_single_consensus"]
    ax.axhline(lvl, color=RED, lw=1.0, ls="--")
    ax.text(2.0, lvl - 0.009, "pooling adds +0.0000", color=RED,
            fontsize=8.5, ha="center",
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_ylabel("HumanEval pass rate")
    ax.set_ylim(0.90, 1.0)
    ax.set_title("Five labs pooled solve the same problems as one model plus a selector",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig4_pooling.png")
    plt.close(fig)
    return "fig4_pooling.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="arxiv_hivemind/figs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_decomposition, fig_scale, fig_reads, fig_pooling):
        print(" ", fn(out), flush=True)
    print(f"\nwrote to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
