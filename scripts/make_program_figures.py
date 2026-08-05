"""Generate the figures for paper_srt_program.md / arxiv_program/.

All numbers are copied from the banked artifacts (see paper §14):
  procrustes_xmodal.json, mlp_align_scaled.json, mlp_align_full.json,
  karpathy_eval.json, procrustes_qwen3b.json, mlp_align_qwen3b.json,
  q4_drift.json, and the local Mac validation run.

Style: clean white-background scientific figures with the SRT violet
(#a48dff) as the accent; Helvetica Neue for labels.

    python scripts/make_program_figures.py --out arxiv_program/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIOLET = "#7a5fd0"      # darkened brand violet for white background
VIOLET_LT = "#a48dff"
INK = "#1c1a47"
MUTED = "#8a86a8"
GREY = "#c9c6dd"
RED = "#c0504d"

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


def fig1_ladder(out: Path) -> None:
    """The method ladder on gemma-4-31B-it (val2017 protocol)."""
    methods = ["shuffled\ncontrol", "Procrustes\n(rotation)", "centered\ncosine",
               "MLP\n(117k pairs)", "linear\n(117k pairs)"]
    r1 = [0.001, 0.247, 0.288, 0.567, 0.661]
    r5 = [0.008, 0.506, 0.523, 0.887, 0.911]
    r10 = [0.013, 0.656, 0.648, 0.943, 0.967]
    x = np.arange(len(methods))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w, r1, w, label="R@1", color=VIOLET)
    ax.bar(x, r5, w, label="R@5", color=VIOLET_LT)
    ax.bar(x + w, r10, w, label="R@10", color=GREY)
    for xi, v in zip(x - w, r1):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.288, color=MUTED, lw=0.8, ls=":")
    ax.text(0.02, 0.305, "zero-training baseline (R@1)", fontsize=7,
            color=MUTED, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.set_xticks(x, methods)
    ax.set_ylabel("COCO i2t recall (1,000 imgs vs 5,000 captions)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("A rotation cannot express the modality gap; a linear map captures it;\n"
                 "depth finds nothing beyond it", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig2_scaling(out: Path) -> None:
    """Train-size scaling: 31B and 3B, linear and MLP. The 3B curve
    sitting on the 31B curve *is* the scale-invariance result."""
    n31 = [5000, 10000, 20000, 39000, 60000, 90000, 117000]
    lin31 = [0.409, 0.450, 0.528, 0.553, 0.611, 0.638, 0.661]
    lin31_39k_alt = 0.590   # first-sweep 39k (different early-stop fold)
    mlp31 = [0.298, 0.355, 0.428, 0.506, 0.546, 0.565, 0.567]
    n3 = [5000, 10000, 20000, 39000]
    lin3 = [0.414, 0.497, 0.537, 0.577]
    mlp3 = [0.296, 0.388, 0.473, 0.537]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(n31, lin31, "o-", color=VIOLET, lw=1.8, ms=4,
            label="linear, 31B host")
    ax.plot([39000, 39000], [0.553, lin31_39k_alt], color=VIOLET, lw=1.0)
    ax.plot([39000], [lin31_39k_alt], "o", color=VIOLET, ms=4, mfc="white")
    ax.plot(n31, mlp31, "s-", color=MUTED, lw=1.4, ms=3.5,
            label="MLP, 31B host")
    ax.plot(n3, lin3, "o--", color=RED, lw=1.8, ms=4,
            label="linear, 3B host (10× smaller)")
    ax.plot(n3, mlp3, "s--", color="#d9a5a3", lw=1.2, ms=3.5,
            label="MLP, 3B host")
    ax.axhline(0.288, color=INK, lw=0.8, ls=":")
    ax.text(5200, 0.294, "zero-training baseline, 31B", fontsize=7, color=INK)
    ax.axhline(0.180, color=RED, lw=0.8, ls=":")
    ax.text(5200, 0.186, "zero-training baseline, 3B", fontsize=7, color=RED)
    ax.annotate("fold variance,\nsame train pairs", xy=(39000, 0.572),
                xytext=(52000, 0.47), fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xscale("log")
    ax.set_xticks(n31, ["5k", "10k", "20k", "39k", "60k", "90k", "117k"])
    ax.minorticks_off()
    ax.set_xlabel("training pairs (COCO train2017)")
    ax.set_ylabel("i2t R@1")
    ax.set_ylim(0.1, 0.72)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("The linear map keeps absorbing data; the MLP never catches it;\n"
                 "the 3B host tracks the 31B host", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig3_invariance(out: Path) -> None:
    """Precision and hardware invariance, two panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))

    # Panel A: weight precision (3B host, head trained on bf16, unchanged)
    labels = ["bf16 states\n(reference)", "4-bit states,\nhead unchanged",
              "4-bit + 42KB\nmean recal"]
    r1 = [0.577, 0.566, 0.569]
    bars = ax1.bar(range(3), r1, 0.55,
                   color=[VIOLET, VIOLET_LT, VIOLET_LT])
    for b, v in zip(bars, r1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                 ha="center", fontsize=8, color=INK)
    ax1.set_xticks(range(3), labels, fontsize=7.5)
    ax1.set_ylim(0.5, 0.62)
    ax1.set_ylabel("i2t R@1 (3B host, 39k-pair head)")
    ax1.set_title("Quantization: −0.011 R@1,\nno retraining", fontsize=9, color=INK)

    # Panel B: hardware/runtime (31B head, datacenter -> Mac)
    # Full 5,000-caption pool, cross-runtime self-retrieval R@1
    # (artifacts/nla/q4/head_space_validation_20260805.json)
    labels2 = ["raw states", "mean-\ncentered", "through head,\nshared mu",
               "through head,\n+ 42KB recal"]
    v2 = [0.930, 0.931, 0.964, 0.967]
    bars2 = ax2.bar(range(4), v2, 0.55,
                    color=[GREY, GREY, VIOLET_LT, VIOLET])
    for b, v in zip(bars2, v2):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                 ha="center", fontsize=7.5, color=INK)
    ax2.axhline(0.996, color=INK, lw=0.8, ls="--", alpha=0.6)
    ax2.text(3.35, 0.998, "same-runtime ceiling 0.996", ha="right",
             fontsize=7, color=INK, alpha=0.8)
    ax2.set_xticks(range(4), labels2, fontsize=7)
    ax2.set_ylim(0.85, 1.03)
    ax2.set_ylabel("cross-runtime R@1 (n=5,000)")
    ax2.set_title("Hardware: CUDA/bf16 → Apple/MLX-Q4;\nhead +3.7, centering inert",
                  fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_invariance.png", bbox_inches="tight")
    plt.close(fig)


def fig4_karpathy(out: Path) -> None:
    """Karpathy 5k placement."""
    systems = ["centered cosine\n(zero training)", "SRT linear head\n(this work)",
               "VSE++ (2018,\ntrained dual encoder)", "CLIP zero-shot\n(400M pairs)"]
    r1 = [0.143, 0.416, 0.413, 0.584]
    r5 = [0.306, 0.710, 0.711, 0.815]
    r10 = [0.386, 0.818, 0.812, 0.881]
    colors_r1 = [GREY, VIOLET, MUTED, MUTED]
    x = np.arange(len(systems))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(x - w, r1, w, color=colors_r1, label="R@1")
    ax.bar(x, r5, w, color=[c + "99" if c.startswith("#") else c for c in colors_r1],
           alpha=0.65, label="R@5")
    ax.bar(x + w, r10, w, color=colors_r1, alpha=0.35, label="R@10")
    for xi, v in zip(x - w, r1):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, systems, fontsize=8)
    ax.set_ylabel("COCO Karpathy 5k test, i2t recall")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("A linear map over a frozen chat model matches a fully-trained\n"
                 "2018 dual encoder (leakage-controlled)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig4_karpathy.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("arxiv_program/figs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig1_ladder(args.out)
    fig2_scaling(args.out)
    fig3_invariance(args.out)
    fig4_karpathy(args.out)
    for f in sorted(args.out.glob("*.png")):
        print(f, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
