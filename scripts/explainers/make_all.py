"""Generate the SRT explainer visual series into artifacts/explainers/.

Dark navy-indigo + pastel neon. Scientific theme with agency polish.
Each figure is a standalone PNG; captions live in docs/EXPLAINERS.md.
Run:  python scripts/explainers/make_all.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

from _style import (
    apply_style, glow, panel, add_panel_bg, title_block, footer,
    neon_line, neon_bar, chip, hidden_axes,
    BG, PANEL, PANEL_EDGE, GRID, TEXT, MUTED, DIM,
    CYAN, MAGENTA, MINT, VIOLET, PEACH, YELLOW, CORAL, LIME,
    FROZEN, TRAINABLE, HOOK, INJECT, SUBCRIT, SUPERCRIT, ACCENTS,
)

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "artifacts" / "explainers"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: rounded box "node"
# ---------------------------------------------------------------------------
def node(ax, xy, w, h, label, *, fc=PANEL, ec=PANEL_EDGE, color=TEXT,
         fontsize=10, weight="semibold", radius=0.05, alpha=1.0,
         sublabel=None, sub_color=MUTED, lw=1.0, glow_color=None):
    x, y = xy
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha,
    )
    if glow_color:
        box.set_path_effects(glow(glow_color, n=3, base_w=3.0, alpha=0.20))
    ax.add_patch(box)
    if sublabel is None:
        ax.text(x, y, label, color=color, ha="center", va="center",
                fontsize=fontsize, weight=weight)
    else:
        ax.text(x, y + h*0.18, label, color=color, ha="center", va="center",
                fontsize=fontsize, weight=weight)
        ax.text(x, y - h*0.22, sublabel, color=sub_color, ha="center",
                va="center", fontsize=fontsize - 2, weight="regular")
    return (x, y, w, h)


def arrow(ax, p0, p1, *, color=MUTED, lw=1.4, style="-|>", mutation=12,
          alpha=0.9, glow_color=None, curve=0.0):
    cs = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation,
                        color=color, lw=lw, alpha=alpha, connectionstyle=cs)
    if glow_color:
        a.set_path_effects(glow(glow_color, n=3, base_w=lw + 1.2, alpha=0.18))
    ax.add_patch(a)
    return a


# ===========================================================================
# Fig 1 — End-to-end pipeline
# ===========================================================================
def fig1_pipeline():
    fig = plt.figure(figsize=(13.5, 7.2))
    title_block(fig, "01 · overview",
                "How SRT reads and corrects a frozen LLM",
                "Frozen backbone · 3 observation hooks · 2 injection points · "
                "~12 M trainable params (0.17 % of 7 B)")

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78])
    hidden_axes(ax)

    # ---- Frozen backbone strip (layers 0..27) -----------------------------
    n_layers = 28
    strip_y0, strip_y1 = 0.55, 0.78
    x0, x1 = 0.04, 0.96
    layer_w = (x1 - x0) / n_layers
    for i in range(n_layers):
        lx = x0 + i * layer_w
        ax.add_patch(Rectangle((lx + 0.0015, strip_y0), layer_w - 0.003,
                               strip_y1 - strip_y0,
                               fc=FROZEN, ec="none", alpha=0.45))
    ax.add_patch(FancyBboxPatch(
        (x0, strip_y0 - 0.005), x1 - x0, strip_y1 - strip_y0 + 0.01,
        boxstyle="round,pad=0,rounding_size=0.012",
        fc="none", ec=DIM, lw=0.8))
    ax.text(x0 + 0.005, strip_y1 + 0.025,
            "FROZEN BACKBONE  ·  28 transformer layers  ·  Qwen-2.5-7B / Llama-3.2-3B / Gemma-2-2B",
            color=MUTED, fontsize=9, weight="bold")
    ax.text(x0, strip_y0 - 0.022, "L0", color=DIM, fontsize=8, ha="left")
    ax.text(x1, strip_y0 - 0.022, "L27", color=DIM, fontsize=8, ha="right")

    # ---- Tokens (left) ----------------------------------------------------
    node(ax, (0.06, 0.40), 0.08, 0.10,
         "tokens", sublabel="input_ids  (B,T)",
         fc=PANEL, ec=PANEL_EDGE, color=TEXT)

    # ---- LM head / logits (right) -----------------------------------------
    node(ax, (0.94, 0.40), 0.08, 0.10,
         "LM head", sublabel="logits → CE",
         fc=PANEL, ec=PANEL_EDGE, color=TEXT)

    arrow(ax, (0.10, 0.40), (0.10, 0.55), color=DIM, lw=1.0)
    arrow(ax, (0.90, 0.55), (0.90, 0.45), color=DIM, lw=1.0)

    # ---- Hook positions on the strip --------------------------------------
    def layer_x(i):
        return x0 + (i + 0.5) * layer_w

    hooks = [(2, "L2",  "community", VIOLET),
             (7, "L7",  "MAH₁",      CYAN),
             (14, "L14", "MAH₂ + inj", MINT),
             (21, "L21", "MAH₃ + inj", MINT)]
    for li, lname, role, col in hooks:
        lx = layer_x(li)
        # marker on the strip
        ax.add_patch(Circle((lx, (strip_y0 + strip_y1) / 2), 0.012,
                            fc=col, ec="none",
                            path_effects=glow(col, n=3, base_w=2.5, alpha=0.30)))
        ax.text(lx, strip_y1 + 0.008, lname, color=col,
                ha="center", fontsize=8, weight="bold")
        ax.text(lx, strip_y0 - 0.012, role, color=col,
                ha="center", fontsize=8.5, weight="semibold", va="top")

    # ---- Adapter modules (below strip) ------------------------------------
    mod_y = 0.26
    mods = [
        (layer_x(2),  "Community",  "discourse basin\nd_community = 64",  VIOLET),
        (layer_x(7),  "MAH",        "meaning fork\nd_div = 256",          CYAN),
        (layer_x(14), "RRM · BEN",  "GRU memory + r̂\nd_meta = 512",      MAGENTA),
        (layer_x(21), "RRM · BEN",  "GRU memory + r̂\nd_meta = 512",      MAGENTA),
    ]
    boxes = []
    for mx, name, sub, col in mods:
        # connector from strip to module
        arrow(ax, (mx, strip_y0 - 0.005), (mx, mod_y + 0.055),
              color=col, lw=1.2, mutation=10, alpha=0.7)
        node(ax, (mx, mod_y), 0.16, 0.11, name,
             sublabel=sub, fc=PANEL, ec=col, color=TEXT,
             sub_color=MUTED, lw=1.2, glow_color=col)
        boxes.append((mx, mod_y, col))

    # ---- Injection arrows back to L14 and L21 -----------------------------
    for li in (14, 21):
        lx = layer_x(li)
        arrow(ax, (lx + 0.018, mod_y + 0.055), (lx + 0.018, strip_y0 - 0.005),
              color=INJECT, lw=1.6, mutation=14, glow_color=INJECT, curve=-0.2)
        ax.text(lx + 0.040, (mod_y + strip_y0) / 2 + 0.015,
                "FiLM\nh ← h·(1+γ) + β",
                color=INJECT, fontsize=7.5, weight="semibold", va="center")

    # ---- Param breakdown panel (bottom-right) -----------------------------
    pad_x = 0.62
    ax.text(pad_x, 0.08, "TRAINABLE PARAMS  ·  ≈ 12 M total", color=CYAN,
            fontsize=9, weight="bold")
    rows = [("community",  "0.06 M"),
            ("MAH × 3",    "8.10 M"),
            ("RRM",        "2.20 M"),
            ("BEN + chain","0.30 M")]
    for i, (k, v) in enumerate(rows):
        ax.text(pad_x, 0.05 - i*0.022, k, color=MUTED, fontsize=8.5)
        ax.text(pad_x + 0.14, 0.05 - i*0.022, v, color=TEXT, fontsize=8.5,
                weight="semibold", ha="right")

    # ---- Legend (bottom-left) ---------------------------------------------
    leg_y = 0.06
    legend_items = [(FROZEN, "frozen backbone (no grad)"),
                    (CYAN,   "observation hook"),
                    (INJECT, "FiLM injection"),
                    (MAGENTA, "trainable adapter module")]
    for i, (c, t) in enumerate(legend_items):
        cy = leg_y - i * 0.022
        ax.add_patch(Circle((0.06, cy), 0.007, fc=c, ec="none",
                            path_effects=glow(c, n=2, base_w=2.0, alpha=0.30)))
        ax.text(0.075, cy, t, color=MUTED, fontsize=8.5, va="center")

    footer(fig)
    fig.savefig(OUT / "01_pipeline.png")
    plt.close(fig)


# ===========================================================================
# Fig 2 — Anisotropy collapse
# ===========================================================================
def fig2_anisotropy():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "02 · geometry",
                "Anisotropy collapse — why raw cosine lies",
                "Hidden states cluster around a non-zero mean μ. "
                "Centering by μ exposes a fair similarity floor.")

    # ---- Left: raw cloud --------------------------------------------------
    ax1 = fig.add_axes([0.06, 0.18, 0.32, 0.68])
    panel(ax1)
    # cloud offset from origin
    mu = np.array([1.8, 1.1])
    pts = rng.normal(0, 0.7, (350, 2)) + mu
    ax1.scatter(pts[:, 0], pts[:, 1], s=10, c=CYAN, alpha=0.55,
                edgecolors="none")
    ax1.scatter([0], [0], s=60, marker="+", c=MUTED, linewidths=1.4)
    ax1.scatter([mu[0]], [mu[1]], s=120, marker="o",
                facecolors="none", edgecolors=MAGENTA, lw=2.0,
                path_effects=glow(MAGENTA, n=3, base_w=3.0, alpha=0.25))
    arrow(ax1, (0, 0), (mu[0]*0.96, mu[1]*0.96), color=MAGENTA, lw=1.6,
          mutation=12, glow_color=MAGENTA)
    ax1.text(mu[0] + 0.15, mu[1] + 0.1, "μ", color=MAGENTA,
             fontsize=14, weight="bold")
    ax1.text(0.08, -0.18, "0", color=MUTED, fontsize=10)
    ax1.set_xlim(-2.0, 4.0); ax1.set_ylim(-2.0, 3.5)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Raw hidden states  ·  cluster around μ",
                  color=TEXT, fontsize=12, loc="left", pad=10)
    ax1.text(0.04, 0.06,
             "fve_raw = ½(1 + cos(h₁, h₂))\nrandom floor ≈ 0.62",
             transform=ax1.transAxes, color=MUTED, fontsize=9,
             family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=PANEL_EDGE))

    # ---- Middle: centered cloud ------------------------------------------
    ax2 = fig.add_axes([0.40, 0.18, 0.32, 0.68])
    panel(ax2)
    cen = pts - mu
    ax2.scatter(cen[:, 0], cen[:, 1], s=10, c=MINT, alpha=0.55,
                edgecolors="none")
    ax2.scatter([0], [0], s=120, marker="o",
                facecolors="none", edgecolors=MINT, lw=2.0,
                path_effects=glow(MINT, n=3, base_w=3.0, alpha=0.25))
    ax2.scatter([0], [0], s=60, marker="+", c=TEXT, linewidths=1.4)
    ax2.set_xlim(-3.0, 3.0); ax2.set_ylim(-2.5, 2.5)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("After subtracting μ  ·  honest geometry",
                  color=TEXT, fontsize=12, loc="left", pad=10)
    ax2.text(0.04, 0.06,
             "fve_cen = ½(1 + cos(h₁−μ, h₂−μ))\nrandom floor ≈ 0.50",
             transform=ax2.transAxes, color=MUTED, fontsize=9,
             family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=PANEL_EDGE))

    # ---- Right: ||μ|| across backbones -----------------------------------
    ax3 = fig.add_axes([0.78, 0.20, 0.18, 0.62])
    backbones = ["Qwen-2.5\n7B (L20)", "Llama-3.2\n3B (L19)", "Gemma-2\n2B (L18)"]
    mu_norms = [55.0, 7.21, 3.4]
    colors = [MAGENTA, CYAN, MINT]
    ys = np.arange(len(backbones))[::-1]
    bars = ax3.barh(ys, mu_norms, color=colors, height=0.55, edgecolor="none")
    for b, c in zip(bars, colors):
        b.set_path_effects(glow(c, n=3, base_w=2.5, alpha=0.22))
    ax3.set_yticks(ys); ax3.set_yticklabels(backbones, fontsize=9, color=TEXT)
    ax3.set_xticks([0, 25, 50])
    ax3.set_xlabel("||μ||  (L2 norm of layer mean)", color=MUTED, fontsize=9)
    ax3.set_title("Anisotropy varies 16× across families",
                  color=TEXT, fontsize=11, loc="left", pad=10)
    for y, v in zip(ys, mu_norms):
        ax3.text(v + 1.5, y, f"{v:.1f}", color=TEXT,
                 fontsize=9, va="center", weight="semibold")
    ax3.set_xlim(0, 65)
    ax3.grid(False)

    footer(fig)
    fig.savefig(OUT / "02_anisotropy.png")
    plt.close(fig)


# ===========================================================================
# Fig 3 — MAH meaning fork
# ===========================================================================
def fig3_mah():
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "03 · MAH",
                "Metapragmatic attention — where meaning forks",
                "Each token gets two interpretants. Their gap is the divergence "
                "vector dₜ that drives the rest of the adapter.")

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.78])
    hidden_axes(ax)

    tokens = ["The", "bank", "raised", "rates", "after", "the", "river", "fell"]
    n = len(tokens)
    xs = np.linspace(0.10, 0.78, n)

    # baseline (hidden state row)
    y_h    = 0.72
    y_int  = 0.55  # interpretant (direct)
    y_ctx  = 0.36  # contextual interpretant
    y_div  = 0.10  # divergence bar baseline

    # connecting line
    ax.plot([xs[0], xs[-1]], [y_h, y_h], color=GRID, lw=0.6, zorder=0)
    ax.plot([xs[0], xs[-1]], [y_int, y_int], color=GRID, lw=0.6, zorder=0)
    ax.plot([xs[0], xs[-1]], [y_ctx, y_ctx], color=GRID, lw=0.6, zorder=0)

    # tokens row
    for x, t in zip(xs, tokens):
        ax.add_patch(Circle((x, y_h), 0.018, fc=PANEL, ec=PANEL_EDGE, lw=1.0))
        ax.text(x, y_h + 0.05, t, color=TEXT, ha="center", fontsize=10,
                weight="semibold")

    # direct interpretant
    for x in xs:
        ax.add_patch(Circle((x, y_int), 0.013, fc=CYAN, ec="none",
                            path_effects=glow(CYAN, n=2, base_w=1.8, alpha=0.25)))
        arrow(ax, (x, y_h - 0.018), (x, y_int + 0.014), color=CYAN,
              lw=0.9, mutation=6, alpha=0.55)

    # contextual interpretant (straight, no horizontal drift — cleaner)
    # plus subtle attention scribbles (causal) onto each contextual node
    div_strength = [0.05, 0.95, 0.20, 0.25, 0.10, 0.08, 0.55, 0.30]
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, y_ctx), 0.013, fc=MAGENTA, ec="none",
                            path_effects=glow(MAGENTA, n=2, base_w=1.8, alpha=0.25)))
        # causal attention scribbles back to previous tokens
        for xp in xs[:i+1]:
            ax.plot([xp, x], [y_int - 0.014, y_ctx + 0.014],
                    color=MAGENTA, lw=0.25, alpha=0.10, zorder=0)

    # divergence bars (capped, readable)
    max_bar = 0.18
    for x, s in zip(xs, div_strength):
        h = 0.025 + s * max_bar
        hilite = MINT if s > 0.4 else VIOLET
        ax.add_patch(Rectangle((x - 0.022, y_div), 0.044, h,
                               fc=hilite, ec="none", alpha=0.85,
                               path_effects=glow(hilite, n=2, base_w=1.6,
                                                 alpha=0.20)))
    # highlight bifurcating token
    bidx = 1
    ax.add_patch(Rectangle((xs[bidx] - 0.05, y_div - 0.012), 0.10, 0.235,
                           fc="none", ec=MINT, lw=1.0, alpha=0.6,
                           linestyle=(0, (3, 2))))
    ax.annotate("high ‖dₜ‖  ·  ‘bank’ flips\nfinance ↔ river",
                xy=(xs[bidx], y_div + 0.20), xytext=(xs[bidx] + 0.06, 0.30),
                color=MINT, fontsize=9, weight="semibold",
                arrowprops=dict(arrowstyle="-", color=MINT, lw=0.8, alpha=0.6))

    # row labels
    ax.text(0.06, y_h, "hidden hₜ",   color=MUTED, fontsize=9, ha="right", va="center")
    ax.text(0.06, y_int, "direct",    color=CYAN, fontsize=9, ha="right", va="center", weight="semibold")
    ax.text(0.06, y_ctx, "contextual", color=MAGENTA, fontsize=9, ha="right", va="center", weight="semibold")
    ax.text(0.06, y_div + 0.085, "‖dₜ‖", color=MINT, fontsize=10, ha="right", va="center", weight="semibold")

    # equation
    eq = (r"$d_t \;=\; \mathrm{proj}_{\mathrm{div}}\!\left("
          r"\mathrm{proj}_{\mathrm{int}}(h_t) \;-\; "
          r"\mathrm{attn}\!\left(\mathrm{proj}_{\mathrm{int}}(h_{0..t})\right)"
          r"\right)$")
    ax.text(0.44, 0.93, eq, color=TEXT, fontsize=12.5, ha="center")

    # right-side dims panel — far right column, doesn't overlap tokens
    ax.add_patch(FancyBboxPatch((0.85, 0.18), 0.13, 0.62,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=PANEL, ec=PANEL_EDGE, lw=1.0))
    ax.text(0.915, 0.75, "DIMENSIONS", color=CYAN, fontsize=9,
            ha="center", weight="bold")
    dims = [("d_backbone", "3584"),
            ("d_sub",      "512"),
            ("d_div",      "256"),
            ("num_heads",  "4")]
    for i, (k, v) in enumerate(dims):
        ax.text(0.865, 0.69 - i*0.065, k, color=MUTED, fontsize=9)
        ax.text(0.965, 0.69 - i*0.065, v, color=TEXT, fontsize=9,
                weight="semibold", ha="right", family="DejaVu Sans Mono")
    ax.text(0.915, 0.42, "PEIRCE", color=CYAN, fontsize=8.5,
            ha="center", weight="bold")
    ax.text(0.915, 0.32,
            "interpretant\n= sign in light\nof another sign",
            color=MUTED, fontsize=8, ha="center", va="center", style="italic")

    footer(fig)
    fig.savefig(OUT / "03_mah.png")
    plt.close(fig)


# ===========================================================================
# Fig 4 — RRM FiLM correction
# ===========================================================================
def fig4_rrm():
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "04 · RRM",
                "Reflexive memory — FiLM correction back into the backbone",
                "A GRU integrates the stream of divergences and emits "
                "(γ, β) modulation. h ← h·(1+γ) + β.")

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.78])
    hidden_axes(ax)

    # ---- GRU sequence on the top ------------------------------------------
    n = 6
    xs = np.linspace(0.10, 0.65, n)
    y_gru = 0.66
    for i, x in enumerate(xs):
        node(ax, (x, y_gru), 0.07, 0.10, "GRU",
             sublabel=f"mₜ₊{i}" if i < 3 else "...",
             fc=PANEL, ec=MAGENTA, color=TEXT,
             glow_color=MAGENTA, lw=1.2, fontsize=10)
    for i in range(n - 1):
        arrow(ax, (xs[i] + 0.035, y_gru), (xs[i+1] - 0.035, y_gru),
              color=MAGENTA, lw=1.4, mutation=10, alpha=0.85)
    # divergence inputs from below
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, y_gru - 0.16), 0.012, fc=CYAN, ec="none",
                            path_effects=glow(CYAN, n=2, base_w=1.6, alpha=0.25)))
        arrow(ax, (x, y_gru - 0.148), (x, y_gru - 0.058),
              color=CYAN, lw=1.0, mutation=8, alpha=0.8)
        ax.text(x, y_gru - 0.20, f"dₜ₊{i}" if i < 3 else "…",
                color=CYAN, fontsize=8.5, ha="center")

    ax.text(0.062, y_gru, "meta\nstate", color=MAGENTA, fontsize=9.5,
            ha="right", va="center", weight="semibold")
    ax.text(0.062, y_gru - 0.16, "divergence\nfrom MAH", color=CYAN,
            fontsize=9, ha="right", va="center", weight="semibold")

    # ---- γ and β heatmap strips (illustrative) ----------------------------
    rng = np.random.default_rng(3)
    gamma = rng.normal(0, 0.18, (1, 48))
    beta  = rng.normal(0, 0.08, (1, 48))
    ax_g = fig.add_axes([0.10, 0.31, 0.55, 0.06])
    ax_b = fig.add_axes([0.10, 0.22, 0.55, 0.06])
    for axx, data, label, c in [(ax_g, gamma, "γₜ  (multiplicative)", MINT),
                                (ax_b, beta,  "βₜ  (additive)",       VIOLET)]:
        axx.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-0.4, vmax=0.4)
        axx.set_xticks([]); axx.set_yticks([])
        for s in axx.spines.values():
            s.set_edgecolor(c); s.set_lw(1.2)
        axx.text(-0.012, 0.5, label, color=c, fontsize=9, weight="semibold",
                 transform=axx.transAxes, ha="right", va="center")
    ax.text(0.385, 0.18, "(d_backbone wide)", color=DIM, fontsize=8.5, ha="center")

    # arrows from GRU strip down to γ/β strips
    arrow(ax, (0.50, y_gru - 0.058), (0.45, 0.39), color=MINT, lw=1.2,
          mutation=10, alpha=0.85, curve=-0.15, glow_color=MINT)
    arrow(ax, (0.55, y_gru - 0.058), (0.55, 0.30), color=VIOLET, lw=1.2,
          mutation=10, alpha=0.85, curve=0.10, glow_color=VIOLET)

    # ---- FiLM equation panel (right) --------------------------------------
    eq_box = FancyBboxPatch((0.72, 0.30), 0.23, 0.32,
                            boxstyle="round,pad=0.012,rounding_size=0.012",
                            fc=PANEL, ec=PANEL_EDGE, lw=1.0)
    ax.add_patch(eq_box)
    ax.text(0.735, 0.575, "FILM INJECTION", color=CYAN, fontsize=9, weight="bold")
    ax.text(0.835, 0.515,
            r"$h_t' \;=\; h_t \cdot (1 + \gamma_t) \;+\; \beta_t$",
            color=TEXT, fontsize=13, ha="center")
    ax.text(0.835, 0.45,
            r"$\Delta h_t \;=\; h_t \cdot \gamma_t \;+\; \beta_t$",
            color=MINT, fontsize=11, ha="center")
    ax.text(0.735, 0.395, "INIT SCHEME", color=CYAN, fontsize=9, weight="bold")
    ax.text(0.735, 0.366, "γ_proj : N(0, 0.02²),  bias 0",
            color=MUTED, fontsize=8.5, family="DejaVu Sans Mono")
    ax.text(0.735, 0.342, "β_proj : 0,  bias 0   ⇒  identity at init",
            color=MUTED, fontsize=8.5, family="DejaVu Sans Mono")

    # ---- Injection target arrow (bottom): into L14 / L21 ------------------
    node(ax, (0.20, 0.08), 0.16, 0.08,
         "inject → L14, L21", sublabel="frozen backbone resumes",
         fc=PANEL, ec=INJECT, color=TEXT, sub_color=MUTED,
         glow_color=INJECT, lw=1.2)
    arrow(ax, (0.385, 0.255), (0.27, 0.115),
          color=INJECT, lw=1.6, mutation=14, glow_color=INJECT, curve=-0.25)

    footer(fig)
    fig.savefig(OUT / "04_rrm.png")
    plt.close(fig)


# ===========================================================================
# Fig 5 — BEN reflexivity strip
# ===========================================================================
def fig5_ben():
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "05 · BEN",
                "Bifurcation estimation — reflexivity r̂ per token",
                "BEN turns the GRU memory into a scalar r̂ (stable ↔ contested) "
                "and a regime gate (subcritical ↔ supercritical).")

    # ---- r̂ strip across a sentence ---------------------------------------
    ax = fig.add_axes([0.06, 0.50, 0.90, 0.32])
    panel(ax)
    sentence = ("The bank raised rates after the river fell "
                "and the analyst struggled to reconcile the metaphor")
    toks = sentence.split()
    n = len(toks)
    # synthesise a plausible r̂ trace: peaks at metaphor-ish words
    base = rng.normal(-0.4, 0.25, n)
    boost_idx = [1, 6, 12, 13, 14]  # bank, river, reconcile, the, metaphor
    for i in boost_idx:
        if i < n:
            base[i] += rng.uniform(1.2, 2.1)
    r_hat = base
    xs = np.arange(n)

    # bar colours by sign
    colors = [SUPERCRIT if v > 0 else SUBCRIT for v in r_hat]
    bars = ax.bar(xs, r_hat, width=0.78, color=colors, edgecolor="none")
    for b, c in zip(bars, colors):
        b.set_path_effects(glow(c, n=3, base_w=2.0, alpha=0.18))
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(toks, rotation=0, fontsize=8.5, color=TEXT)
    ax.set_ylabel("r̂  (log-compressed reflexivity)", color=MUTED, fontsize=10)
    ax.set_ylim(-1.6, 2.6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", length=0, pad=4)

    # subcritical / supercritical chips
    ax.text(0.985, 0.92, "supercritical  (r̂ > 0, bifurcating)",
            color=SUPERCRIT, fontsize=9, weight="semibold",
            transform=ax.transAxes, ha="right")
    ax.text(0.985, 0.08, "subcritical  (r̂ ≤ 0, stable)",
            color=SUBCRIT, fontsize=9, weight="semibold",
            transform=ax.transAxes, ha="right")

    # ---- Log-compression mapping (bottom-left) ----------------------------
    ax2 = fig.add_axes([0.07, 0.10, 0.40, 0.32])
    panel(ax2)
    r_true = np.linspace(-12, 12, 400)
    r_comp = np.sign(r_true) * np.log1p(np.abs(r_true))
    neon_line(ax2, r_true, r_comp, color=CYAN, lw=2.2)
    ax2.plot(r_true, r_true, color=DIM, lw=0.8, ls=(0, (3, 3)))
    ax2.axhline(0, color=MUTED, lw=0.6, alpha=0.5)
    ax2.axvline(0, color=MUTED, lw=0.6, alpha=0.5)
    ax2.set_xlabel("r_true  (raw)", color=MUTED, fontsize=9)
    ax2.set_ylabel("training target  sign(r)·log(1+|r|)",
                   color=MUTED, fontsize=9)
    ax2.set_title("Log-compression maps wide r into a learnable range",
                  color=TEXT, fontsize=11, loc="left", pad=10)
    ax2.set_xlim(-12, 12); ax2.set_ylim(-3, 3)

    # ---- Loss equation panel (bottom-right) -------------------------------
    ax3 = fig.add_axes([0.55, 0.10, 0.40, 0.32])
    hidden_axes(ax3)
    ax3.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                                 boxstyle="round,pad=0.02,rounding_size=0.03",
                                 fc=PANEL, ec=PANEL_EDGE, lw=1.0))
    ax3.text(0.06, 0.85, "TRAINING SIGNALS  ·  per token", color=CYAN,
             fontsize=9.5, weight="bold")

    items = [
        ("bif_loss",
         r"smooth-L1$(\,r̂,\ \mathrm{sign}(r)\log(1+|r|)\,)$",
         "weight 1.0"),
        ("regime_loss",
         r"CE$(\,\mathrm{argmax}\ \text{logits},\ \mathbb{1}[r{>}0]\,)$",
         "weight 5.0"),
        ("chain_loss",
         r"MSE$(\,\hat d_{i+1},\ d_{i+1}\,)$",
         "weight 0.5"),
        ("listnet",
         r"rank on $\hat r$ ordering",
         "weight 0.5"),
    ]
    for i, (n_, eq, w) in enumerate(items):
        yy = 0.74 - i * 0.16
        ax3.text(0.07, yy, n_, color=MAGENTA, fontsize=10, weight="semibold")
        ax3.text(0.34, yy, eq, color=TEXT, fontsize=10)
        ax3.text(0.96, yy, w, color=MUTED, fontsize=9, ha="right",
                 family="DejaVu Sans Mono")

    footer(fig)
    fig.savefig(OUT / "05_ben.png")
    plt.close(fig)


# ===========================================================================
# Fig 6 — Community latent geometry
# ===========================================================================
def fig6_community():
    rng = np.random.default_rng(5)
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "06 · community",
                "Discourse basins in the early hidden space",
                "Layer-2 pooled states encode topic / register. "
                "v8a uses a continuous 64-D coordinate; v3–v7 used 32 prototypes.")

    # ---- Left: continuous (v8a) scatter -----------------------------------
    ax1 = fig.add_axes([0.05, 0.10, 0.42, 0.74])
    panel(ax1)
    centers = [(-2.0, 1.6), (1.8, 1.3), (-1.4, -1.7), (1.9, -1.5), (0.1, 0.2)]
    labels  = ["news", "poetry", "code", "dialogue", "instructions"]
    cols    = [CYAN, MAGENTA, MINT, PEACH, VIOLET]
    for (cx, cy), col in zip(centers, cols):
        pts = rng.normal(0, 0.55, (180, 2)) + np.array([cx, cy])
        ax1.scatter(pts[:, 0], pts[:, 1], s=10, c=col, alpha=0.55,
                    edgecolors="none")
        ax1.scatter([cx], [cy], s=80, marker="o", facecolors="none",
                    edgecolors=col, lw=1.5,
                    path_effects=glow(col, n=2, base_w=2.5, alpha=0.30))
    # labels
    for (cx, cy), name, col in zip(centers, labels, cols):
        ax1.text(cx, cy + 0.95, name, color=col, fontsize=9.5,
                 weight="semibold", ha="center")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlim(-4.0, 4.0); ax1.set_ylim(-3.2, 3.2)
    ax1.set_title("v8a  ·  continuous community vector  (d=64)",
                  color=TEXT, fontsize=12, loc="left", pad=10)

    # ---- Right: prototype softmax (v3–v7) ---------------------------------
    ax2 = fig.add_axes([0.52, 0.10, 0.43, 0.74])
    panel(ax2)
    # simplex-like display: 32 prototypes around a circle
    K = 32
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)
    R = 1.0
    px = R * np.cos(theta); py = R * np.sin(theta)
    for i in range(K):
        c = ACCENTS[i % len(ACCENTS)]
        ax2.scatter(px[i], py[i], s=70, c=c, edgecolors="none", alpha=0.7,
                    path_effects=glow(c, n=2, base_w=2.0, alpha=0.20))
    # show a soft-assignment example
    sample = np.zeros(K)
    pick = [4, 5, 6, 20]
    weights = np.array([0.42, 0.30, 0.18, 0.10])
    sample[pick] = weights
    # draw rays into a centroid
    cent = np.array([0.0, 0.0])
    for k, w in zip(pick, weights):
        ax2.plot([px[k], cent[0]], [py[k], cent[1]],
                 color=ACCENTS[k % len(ACCENTS)], lw=0.6 + 5*w, alpha=0.8)
    ax2.scatter([0], [0], s=160, marker="o", facecolors=PANEL,
                edgecolors=YELLOW, lw=1.5,
                path_effects=glow(YELLOW, n=3, base_w=3.0, alpha=0.30))
    ax2.text(0, -0.18, "weighted\nsum", color=YELLOW, fontsize=9.5,
             ha="center", va="center", weight="semibold")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlim(-1.6, 1.6); ax2.set_ylim(-1.6, 1.6)
    ax2.set_aspect("equal")
    ax2.set_title("v3–v7  ·  32 prototypes  ·  softmax(τ=1.0) mixture",
                  color=TEXT, fontsize=12, loc="left", pad=10)
    ax2.text(0.02, 0.02,
             "vec = softmax(enc · Pᵀ / τ) · P",
             transform=ax2.transAxes, color=MUTED, fontsize=9,
             family="DejaVu Sans Mono",
             bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=PANEL_EDGE))

    footer(fig)
    fig.savefig(OUT / "06_community.png")
    plt.close(fig)


# ===========================================================================
# Fig 7 — Loss landscape stacked
# ===========================================================================
def fig7_losses():
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "07 · objective",
                "The eleven losses that train the adapter",
                "Backbone CE stays dominant; auxiliary losses shape "
                "reflexivity, regime, communities, and divergence chains.")

    losses = [
        ("CE  (backbone next-token)",       1.0,  CYAN,    "core"),
        ("regime  (CE on r>0 / r≤0)",       5.0,  MAGENTA, "BEN"),
        ("community SupCon",                2.0,  VIOLET,  "community"),
        ("divergence SupCon",               1.0,  PEACH,   "MAH"),
        ("bifurcation  (smooth-L1 on r̂)",  1.0,  CORAL,   "BEN"),
        ("chain  (MSE on d_{i+1})",         0.5,  MINT,    "MAH"),
        ("listnet  (rank on r̂)",           0.5,  YELLOW,  "BEN"),
        ("div-alive  (norm prior)",         0.1,  LIME,    "MAH"),
        ("chain residual aux",              0.05, "#a78bfa","MAH"),
        ("community entropy",               0.01, "#79c2ff","community"),
        ("inject-reg  (disabled v4+)",      0.00, DIM,     "RRM"),
    ]
    names   = [l[0] for l in losses]
    weights = np.array([l[1] for l in losses])
    cols    = [l[2] for l in losses]
    fams    = [l[3] for l in losses]

    ax = fig.add_axes([0.28, 0.12, 0.50, 0.74])
    ys = np.arange(len(losses))[::-1]
    bars = ax.barh(ys, weights, color=cols, height=0.7, edgecolor="none")
    for b, c, w in zip(bars, cols, weights):
        if w > 0:
            b.set_path_effects(glow(c, n=3, base_w=2.5, alpha=0.20))
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10, color=TEXT)
    ax.set_xlabel("loss weight  (v4 defaults)", color=MUTED, fontsize=10)
    ax.set_xlim(0, 5.6)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    for y, w in zip(ys, weights):
        ax.text(w + 0.08, y, f"{w:g}", color=TEXT, fontsize=9.5,
                weight="semibold", va="center")
    # family tag column to the right of value
    fam_colors = {"core": CYAN, "MAH": PEACH, "RRM": MINT,
                  "BEN": MAGENTA, "community": VIOLET}
    for y, fam in zip(ys, fams):
        c = fam_colors[fam]
        ax.text(5.55, y, fam, color=c, fontsize=8.5, weight="bold",
                ha="right", va="center", family="DejaVu Sans Mono", alpha=0.85)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Σ  =  w_ce·CE + w_chain·chain + w_bif·bif + …",
                 color=TEXT, fontsize=12, loc="left", pad=14)

    # right-side rationale callouts
    ax2 = fig.add_axes([0.83, 0.12, 0.14, 0.74])
    hidden_axes(ax2)
    notes = [
        ("dominant signal", CYAN, 0.96),
        ("hard regime gate", MAGENTA, 0.85),
        ("keeps clusters apart", VIOLET, 0.73),
        ("prevents collapse", PEACH, 0.62),
        ("scalar fidelity", CORAL, 0.51),
        ("interpretant continuity", MINT, 0.40),
        ("ordering signal", YELLOW, 0.29),
        ("non-zero ‖d‖", LIME, 0.18),
    ]
    for label, c, y in notes:
        ax2.text(0.02, y, "•", color=c, fontsize=14)
        ax2.text(0.10, y, label, color=MUTED, fontsize=8.5, va="center")

    footer(fig)
    fig.savefig(OUT / "07_losses.png")
    plt.close(fig)


# ===========================================================================
# Fig 8 — TruthfulQA detector ladder
# ===========================================================================
def fig8_tqa_ladder():
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "08 · detector",
                "TruthfulQA-MC2  ·  AUC climbs from 0.535 → 0.866",
                "Same protocol throughout: frozen Qwen-2.5-7B + GroupKFold by question "
                "(n = 817, 5882 paired choices).")

    stages = [
        ("baseline\nNLL only",              0.535, 1,   DIM),
        ("v0\nendpoint drift + NLL",        0.723, 14,  CYAN),
        ("v2\ntoken + per-layer (213 ft)",  0.825, 213, VIOLET),
        ("v3\n+ attention + interactions",  0.866, 320, MAGENTA),
    ]
    ax = fig.add_axes([0.07, 0.18, 0.55, 0.66])
    xs = np.arange(len(stages))
    aucs = [s[1] for s in stages]
    cols = [s[3] for s in stages]
    bars = ax.bar(xs, aucs, color=cols, width=0.55, edgecolor="none")
    for b, c in zip(bars, cols):
        b.set_path_effects(glow(c, n=3, base_w=3.0, alpha=0.22))
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in stages], fontsize=9.5, color=TEXT)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("group-CV AUC", color=MUTED, fontsize=10)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.866])
    ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80", "0.866"])
    ax.grid(True, axis="y", alpha=0.25)
    # value labels
    for x, (name, auc, nfeat, c) in zip(xs, stages):
        ax.text(x, auc + 0.012, f"{auc:.3f}", color=c, fontsize=12,
                weight="bold", ha="center")
        ftxt = "1 feature" if nfeat == 1 else f"{nfeat} features"
        ax.text(x, 0.46, ftxt, color=MUTED, fontsize=8, ha="center")

    # reference band
    ax.axhspan(0.78, 0.85, color=YELLOW, alpha=0.07, zorder=0)
    ax.text(0.05, 0.815, "published-detector band  (INSIDE, EigenScore, SAR)",
            color=YELLOW, fontsize=8.5, va="center", ha="left")
    ax.axhline(0.866, color=MAGENTA, lw=0.6, ls=(0, (3, 3)), alpha=0.4)

    # ---- Right: classifier comparison panel -------------------------------
    ax2 = fig.add_axes([0.69, 0.20, 0.27, 0.60])
    panel(ax2)
    clfs   = ["L2 logit", "skGBM", "XGBoost", "LightGBM"]
    aucs2  = [0.829, 0.835, 0.864, 0.866]
    cols2  = [DIM, VIOLET, PEACH, MAGENTA]
    ys = np.arange(len(clfs))[::-1]
    bars2 = ax2.barh(ys, aucs2, color=cols2, height=0.55, edgecolor="none")
    for b, c in zip(bars2, cols2):
        b.set_path_effects(glow(c, n=3, base_w=2.5, alpha=0.22))
    ax2.set_yticks(ys); ax2.set_yticklabels(clfs, color=TEXT, fontsize=10)
    ax2.set_xlim(0.80, 0.88)
    ax2.set_xticks([0.80, 0.83, 0.86])
    ax2.set_xlabel("AUC on 320 v3 features", color=MUTED, fontsize=9)
    ax2.set_title("classifier head — same features",
                  color=TEXT, fontsize=11, loc="left", pad=10)
    for y, v in zip(ys, aucs2):
        ax2.text(v + 0.001, y, f"{v:.3f}", color=TEXT, fontsize=9,
                 va="center", weight="semibold")

    footer(fig)
    fig.savefig(OUT / "08_tqa_ladder.png")
    plt.close(fig)


# ===========================================================================
# Fig 9 — Cross-arch grouped bars
# ===========================================================================
def fig9_crossarch():
    fig = plt.figure(figsize=(13.5, 7.0))
    title_block(fig, "09 · cross-architecture",
                "Same harness, three backbones — AUC 0.85–0.87 everywhere",
                "Feature families ablated independently; LightGBM on the union "
                "is the consistent ceiling.")

    families = ["NLL/ent\nmargin", "drift", "norm", "attention",
                "interactions", "ALL\n(LightGBM)"]
    qwen   = [0.681, 0.763, 0.769, 0.725, 0.765, 0.866]
    llama  = [0.680, 0.736, 0.675, 0.725, 0.709, 0.848]
    gemma  = [0.697, 0.756, 0.737, 0.755, 0.750, 0.856]

    ax = fig.add_axes([0.06, 0.14, 0.70, 0.72])
    x = np.arange(len(families))
    w = 0.25
    sets = [("Qwen-2.5-7B",   qwen,  MAGENTA),
            ("Llama-3.2-3B",  llama, CYAN),
            ("Gemma-2-2B",    gemma, MINT)]
    for i, (name, vals, c) in enumerate(sets):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, width=w*0.9, color=c,
                      edgecolor="none", label=name)
        for j, b in enumerate(bars):
            is_all = (families[j].startswith("ALL"))
            alpha_g = 0.30 if is_all else 0.18
            b.set_path_effects(glow(c, n=3, base_w=2.5, alpha=alpha_g))
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.005, f"{v:.3f}", color=c,
                    fontsize=7.5, ha="center", weight="semibold")

    ax.set_xticks(x); ax.set_xticklabels(families, fontsize=9.5, color=TEXT)
    ax.set_ylim(0.62, 0.92)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_ylabel("group-CV AUC", color=MUTED, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9.5)

    # highlight band on "ALL" group
    ax.add_patch(Rectangle((len(families) - 1 - 0.45, 0.62),
                           0.9, 0.30, fc=YELLOW, alpha=0.04, zorder=0))
    ax.text(len(families) - 1, 0.905, "union recovers full signal\non every backbone",
            color=YELLOW, fontsize=8.5, ha="center", style="italic")

    # ---- Right: backbone "spec card" panel --------------------------------
    ax2 = fig.add_axes([0.78, 0.14, 0.18, 0.72])
    hidden_axes(ax2)
    cards = [
        ("Qwen-2.5-7B", "L = 28  ·  ‖μ‖ ≈ 55", "norms dominate", MAGENTA),
        ("Llama-3.2-3B","L = 28  ·  ‖μ‖ ≈ 7.2", "drift dominates", CYAN),
        ("Gemma-2-2B",  "L = 26  ·  ‖μ‖ small", "attention strongest", MINT),
    ]
    for i, (name, dim, role, c) in enumerate(cards):
        y0 = 0.78 - i * 0.30
        ax2.add_patch(FancyBboxPatch((0.04, y0 - 0.12), 0.92, 0.22,
                                     boxstyle="round,pad=0.01,rounding_size=0.04",
                                     fc=PANEL, ec=c, lw=1.0,
                                     path_effects=glow(c, n=2, base_w=2.0, alpha=0.18)))
        ax2.text(0.10, y0 + 0.05, name, color=c, fontsize=11, weight="bold")
        ax2.text(0.10, y0 - 0.005, dim, color=MUTED, fontsize=8.5,
                 family="DejaVu Sans Mono")
        ax2.text(0.10, y0 - 0.06, role, color=TEXT, fontsize=9)

    footer(fig)
    fig.savefig(OUT / "09_crossarch.png")
    plt.close(fig)


# ===========================================================================
# Fig 10 — Demo / script map
# ===========================================================================
def fig10_demo_map():
    fig = plt.figure(figsize=(13.5, 7.4))
    title_block(fig, "10 · demos",
                "How the scripts connect — train, eval, visualize, serve",
                "Each box is a script in this repo; arrows show data flow "
                "(checkpoints, JSONL traces, metrics).")

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
    hidden_axes(ax)

    # nodes
    # data layer (left)
    node(ax, (0.08, 0.78), 0.14, 0.10, "train.jsonl",
         sublabel="1 M rows", fc=PANEL, ec=DIM, color=TEXT, sub_color=MUTED)
    node(ax, (0.08, 0.58), 0.14, 0.10, "val.jsonl",
         sublabel="100 K rows", fc=PANEL, ec=DIM, color=TEXT, sub_color=MUTED)
    node(ax, (0.08, 0.20), 0.14, 0.10, "TruthfulQA-MC2",
         sublabel="817 q · 5882 ch", fc=PANEL, ec=DIM, color=TEXT, sub_color=MUTED)

    # training (center-top)
    node(ax, (0.34, 0.78), 0.18, 0.12, "scripts/train.py",
         sublabel="frozen LM + adapter\n11 losses · 187 K steps",
         fc=PANEL, ec=MAGENTA, color=TEXT, sub_color=MUTED,
         glow_color=MAGENTA, lw=1.2)

    # checkpoint
    node(ax, (0.58, 0.78), 0.16, 0.10, "checkpoint.pt",
         sublabel="adapter weights",
         fc=PANEL, ec=YELLOW, color=TEXT, sub_color=MUTED,
         glow_color=YELLOW, lw=1.2)

    # eval scripts (middle row)
    node(ax, (0.34, 0.55), 0.18, 0.10, "centered_eval.py",
         sublabel="raw + centered fve_nrm",
         fc=PANEL, ec=CYAN, color=TEXT, sub_color=MUTED,
         glow_color=CYAN, lw=1.0)
    node(ax, (0.58, 0.55), 0.18, 0.10, "rerank_eval.py",
         sublabel="K-curve · logp rerank",
         fc=PANEL, ec=CYAN, color=TEXT, sub_color=MUTED,
         glow_color=CYAN, lw=1.0)
    node(ax, (0.82, 0.55), 0.16, 0.10, "oracle_ceiling.py",
         sublabel="replay · NN · paraphrase",
         fc=PANEL, ec=CYAN, color=TEXT, sub_color=MUTED,
         glow_color=CYAN, lw=1.0)

    # truthfulqa eval (bottom-left)
    node(ax, (0.34, 0.20), 0.18, 0.12, "evals/truthfulqa_v3.py",
         sublabel="320 ft · LightGBM\nAUC 0.866 ± 0.011",
         fc=PANEL, ec=MINT, color=TEXT, sub_color=MUTED,
         glow_color=MINT, lw=1.2)

    # artifacts (right)
    node(ax, (0.82, 0.78), 0.16, 0.10, "train_log.jsonl",
         sublabel="per-step diagnostics",
         fc=PANEL, ec=PEACH, color=TEXT, sub_color=MUTED)
    node(ax, (0.82, 0.20), 0.16, 0.10, "metrics.json",
         sublabel="AUCs per family",
         fc=PANEL, ec=PEACH, color=TEXT, sub_color=MUTED)

    # visualization / demo (bottom-right)
    node(ax, (0.58, 0.20), 0.18, 0.10, "plot_artifacts.py",
         sublabel="loss curves PNG",
         fc=PANEL, ec=VIOLET, color=TEXT, sub_color=MUTED,
         glow_color=VIOLET, lw=1.0)
    node(ax, (0.58, 0.36), 0.18, 0.10, "render_heatmap.py",
         sublabel="per-token r̂ HTML",
         fc=PANEL, ec=VIOLET, color=TEXT, sub_color=MUTED,
         glow_color=VIOLET, lw=1.0)
    node(ax, (0.82, 0.36), 0.16, 0.10, "demo/app.py",
         sublabel="Gradio web UI",
         fc=PANEL, ec=CORAL, color=TEXT, sub_color=MUTED,
         glow_color=CORAL, lw=1.2)

    # arrows
    arrow(ax, (0.15, 0.78), (0.25, 0.78), color=DIM, lw=1.1)
    arrow(ax, (0.15, 0.58), (0.25, 0.74), color=DIM, lw=1.1, curve=-0.2)
    arrow(ax, (0.43, 0.78), (0.50, 0.78), color=MAGENTA, lw=1.4,
          glow_color=MAGENTA)
    arrow(ax, (0.58, 0.73), (0.58, 0.60), color=YELLOW, lw=1.2,
          glow_color=YELLOW, mutation=10)
    arrow(ax, (0.50, 0.73), (0.40, 0.60), color=YELLOW, lw=1.2,
          glow_color=YELLOW, mutation=10, curve=0.15)
    arrow(ax, (0.66, 0.73), (0.78, 0.60), color=YELLOW, lw=1.2,
          glow_color=YELLOW, mutation=10, curve=-0.15)
    arrow(ax, (0.43, 0.78), (0.74, 0.78), color=PEACH, lw=1.0, alpha=0.6, curve=0.18)
    # tqa column
    arrow(ax, (0.15, 0.20), (0.25, 0.20), color=DIM, lw=1.1)
    arrow(ax, (0.43, 0.20), (0.50, 0.20), color=MINT, lw=1.4, glow_color=MINT)
    arrow(ax, (0.43, 0.26), (0.50, 0.34), color=MINT, lw=1.0, alpha=0.65)
    arrow(ax, (0.66, 0.20), (0.74, 0.20), color=PEACH, lw=1.0)
    arrow(ax, (0.66, 0.36), (0.74, 0.36), color=CORAL, lw=1.2,
          glow_color=CORAL)

    # legend
    leg = [(MAGENTA, "train"), (CYAN, "intrinsic eval"), (MINT, "extrinsic eval"),
           (VIOLET, "visualize"), (CORAL, "serve"), (YELLOW, "checkpoint flow"),
           (PEACH, "artifact")]
    for i, (c, t) in enumerate(leg):
        x0 = 0.04 + i * 0.13
        ax.add_patch(Circle((x0, 0.02), 0.006, fc=c, ec="none",
                            path_effects=glow(c, n=2, base_w=1.6, alpha=0.25)))
        ax.text(x0 + 0.012, 0.02, t, color=MUTED, fontsize=8.5, va="center")

    footer(fig)
    fig.savefig(OUT / "10_demo_map.png")
    plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================
def main():
    apply_style()
    fig1_pipeline()
    fig2_anisotropy()
    fig3_mah()
    fig4_rrm()
    fig5_ben()
    fig6_community()
    fig7_losses()
    fig8_tqa_ladder()
    fig9_crossarch()
    fig10_demo_map()
    written = sorted(OUT.glob("*.png"))
    print(f"[explainers] wrote {len(written)} figures →")
    for p in written:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
