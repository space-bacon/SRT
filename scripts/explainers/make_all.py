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
                "~12.7 M trainable params (0.18 % of 7 B)")

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

    hooks = [(4, "L4",  "community", VIOLET),
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
        (layer_x(4),  "Community",  "discourse basin\nd_community = 64",  VIOLET),
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
    ax.text(pad_x, 0.08, "TRAINABLE PARAMS  ·  ≈ 12.7 M total", color=CYAN,
            fontsize=9, weight="bold")
    rows = [("community",  "0.23 M"),
            ("MAH × 3",    "9.14 M"),
            ("RRM",        "3.02 M"),
            ("BEN + chain","0.33 M")]
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
def fig0_architecture():
    """Replacement for the README's ASCII architecture block.

    Vertical 28-layer backbone tower (center) with hooks at L4/L7/L14/L21
    going left to MAH/RRM and right to Community/BEN, plus mint FiLM
    inject arrows curving back into L14 and L21.
    """
    fig = plt.figure(figsize=(14.5, 9.6))
    title_block(fig, "00 · architecture",
                "SRT-Adapter — what attaches to the frozen LLM",
                "12.7 M trainable params · 3 read hooks · 2 FiLM inject points · "
                "backbone-agnostic (Qwen / Llama / Gemma)")

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    hidden_axes(ax)

    # ---- Backbone tower (centered) ---------------------------------------
    tx0, tx1 = 0.40, 0.60          # tower x-extent
    ty0, ty1 = 0.12, 0.92          # tower y-extent
    tw = tx1 - tx0
    cx = (tx0 + tx1) / 2

    # vertical "rail" behind the tower
    rail = FancyBboxPatch((tx0 - 0.01, ty0 - 0.005),
                          tw + 0.02, ty1 - ty0 + 0.01,
                          boxstyle="round,pad=0,rounding_size=0.018",
                          fc=PANEL, ec=PANEL_EDGE, lw=1.0)
    ax.add_patch(rail)
    ax.text(cx, ty1 + 0.030,
            "frozen backbone  ·  28 transformer layers",
            color=DIM, fontsize=8.5, weight="semibold", ha="center",
            style="italic")

    # slabs (top → bottom)
    # (label, sublabel, fill, edge, text_color, glow, height_units)
    slabs = [
        ("Embeddings",           "native, frozen",     FROZEN, PANEL_EDGE, TEXT,  None, 1.0),
        ("Layers 0 – 6",         "frozen",             FROZEN, PANEL_EDGE, MUTED, None, 1.6),
        ("Layer 4",              "tap → community",    "#2a2f6a", VIOLET,  TEXT,  VIOLET, 0.7),
        ("Layers 3 – 6",         "frozen",             FROZEN, PANEL_EDGE, MUTED, None, 1.2),
        ("Layer 7",              "tap → MAH₁",         "#2a2f6a", CYAN,    TEXT,  CYAN,  0.8),
        ("Layers 8 – 13",        "frozen",             FROZEN, PANEL_EDGE, MUTED, None, 1.6),
        ("Layer 14",             "tap → MAH₂   +   inject γ·h+β", "#2a2f6a", MINT, TEXT, MINT, 0.9),
        ("Layers 15 – 20",       "frozen (w/ correction)", FROZEN, PANEL_EDGE, MUTED, None, 1.6),
        ("Layer 21",             "tap → MAH₃   +   inject γ·h+β", "#2a2f6a", MINT, TEXT, MINT, 0.9),
        ("Layers 22 – 27",       "frozen (w/ correction)", FROZEN, PANEL_EDGE, MUTED, None, 1.6),
        ("LM head",              "native, frozen  →  logits + CE", FROZEN, PANEL_EDGE, TEXT, None, 1.0),
    ]
    # The L0-6 slab is split by the L4 tap; remove the redundant L0-6 row.
    slabs = [s for s in slabs if s[0] != "Layers 0 – 6"]
    # Replace it with L0-3 above the L4 tap, and L5-6 below
    slabs.insert(1, ("Layers 0 – 3", "frozen", FROZEN, PANEL_EDGE, MUTED, None, 0.9))
    # Adjust the post-tap slab label
    slabs = [("Layers 5 – 6", s[1], s[2], s[3], s[4], s[5], s[6]) if s[0] == "Layers 3 – 6" else s
             for s in slabs]

    total_units = sum(s[6] for s in slabs)
    avail = ty1 - ty0 - 0.018 * (len(slabs) - 1)   # gap between slabs
    unit_h = avail / total_units

    # remember y-centers for hooked layers
    hooks = {}   # layer label -> y center
    y_cursor = ty1
    for (lab, sub, fc, ec, tc, gl, units) in slabs:
        h = units * unit_h
        y_cursor -= h
        y_center = y_cursor + h / 2
        box = FancyBboxPatch((tx0 + 0.006, y_cursor),
                             tw - 0.012, h,
                             boxstyle="round,pad=0,rounding_size=0.010",
                             fc=fc, ec=ec, lw=1.1, alpha=0.95)
        if gl:
            box.set_path_effects(glow(gl, n=3, base_w=2.4, alpha=0.22))
        ax.add_patch(box)
        ax.text(cx, y_center + 0.012, lab, color=tc, ha="center",
                va="center", fontsize=10.5, weight="semibold")
        ax.text(cx, y_center - 0.018, sub, color=MUTED, ha="center",
                va="center", fontsize=8.5)
        hooks[lab] = (y_center, y_cursor, y_cursor + h)
        y_cursor -= 0.018

    # ---- Tokens in (top) & logits out (bottom) ---------------------------
    arrow(ax, (cx, ty1 + 0.045), (cx, ty1 + 0.002),
          color=DIM, lw=1.2)
    ax.text(cx, ty1 + 0.062, "tokens  (B, T)", color=MUTED, fontsize=10,
            ha="center", weight="semibold")

    arrow(ax, (cx, ty0 - 0.002), (cx, ty0 - 0.085),
          color=DIM, lw=1.2)
    ax.text(cx, ty0 - 0.100, "logits  (B, T, V)", color=MUTED, fontsize=10,
            ha="center", weight="semibold")

    # ---- Left side: MAH boxes + RRM --------------------------------------
    # MAH boxes vertically aligned to their hooks
    mah_x = 0.24
    mah_w, mah_h = 0.14, 0.070
    mah_specs = [
        ("Layer 7",  "MAH₁",  "L7  divergence d₇"),
        ("Layer 14", "MAH₂",  "L14 divergence d₁₄"),
        ("Layer 21", "MAH₃",  "L21 divergence d₂₁"),
    ]
    mah_centers = []
    for (layer_lab, name, sub) in mah_specs:
        y = hooks[layer_lab][0]
        box = FancyBboxPatch((mah_x - mah_w/2, y - mah_h/2), mah_w, mah_h,
                             boxstyle="round,pad=0,rounding_size=0.014",
                             fc=PANEL, ec=CYAN, lw=1.2)
        box.set_path_effects(glow(CYAN, n=3, base_w=2.6, alpha=0.20))
        ax.add_patch(box)
        ax.text(mah_x, y + 0.012, name, color=CYAN, ha="center",
                va="center", fontsize=11.5, weight="bold")
        ax.text(mah_x, y - 0.016, sub, color=MUTED, ha="center",
                va="center", fontsize=8.5)
        mah_centers.append((mah_x, y))
        # tap arrow from tower to MAH (cyan)
        arrow(ax, (tx0, y), (mah_x + mah_w/2, y),
              color=CYAN, lw=1.6, mutation=14, glow_color=CYAN)

    # RRM box (large, spans the three MAH y-range), positioned further left
    rrm_x = 0.03
    rrm_top = mah_centers[0][1] + 0.045
    rrm_bot = mah_centers[2][1] - 0.045
    rrm_w   = 0.11
    rrm_h   = rrm_top - rrm_bot
    rrm_cx  = rrm_x + rrm_w/2
    rrm_cy  = (rrm_top + rrm_bot) / 2
    rrm_box = FancyBboxPatch((rrm_x, rrm_bot), rrm_w, rrm_h,
                             boxstyle="round,pad=0,rounding_size=0.018",
                             fc=PANEL, ec=MAGENTA, lw=1.3)
    rrm_box.set_path_effects(glow(MAGENTA, n=4, base_w=3.0, alpha=0.22))
    ax.add_patch(rrm_box)
    ax.text(rrm_cx, rrm_top - 0.040, "RRM", color=MAGENTA,
            ha="center", fontsize=14, weight="bold")
    ax.text(rrm_cx, rrm_top - 0.070, "GRU · d_meta=512",
            color=MUTED, ha="center", fontsize=8.5)
    ax.text(rrm_cx, rrm_cy - 0.005, "integrate\ndivergence\nstream",
            color=TEXT, ha="center", va="center", fontsize=9, style="italic")
    ax.text(rrm_cx, rrm_bot + 0.022,
            "emits  γ, β\n(FiLM)",
            color=MINT, ha="center", va="center",
            fontsize=8.5, weight="semibold")

    # MAH → RRM arrows (cyan into magenta box)
    for (mx, my) in mah_centers:
        arrow(ax, (mx - mah_w/2, my), (rrm_x + rrm_w, my),
              color=CYAN, lw=1.4, mutation=12, alpha=0.85)

    # ---- FiLM inject arrows: RRM → L14 & L21 (mint, curved) --------------
    inject_targets = ["Layer 14", "Layer 21"]
    for lab in inject_targets:
        y_tgt = hooks[lab][0]
        # exit right edge of RRM, route under MAH column, enter left edge of layer slab
        p0 = (rrm_x + rrm_w, rrm_bot + 0.015)
        p1 = (tx0 - 0.002, y_tgt - 0.014)
        arrow(ax, p0, p1, color=MINT, lw=2.0, mutation=14,
              curve=-0.45, glow_color=MINT, alpha=0.95)

    # ---- Right side: Community + BEN -------------------------------------
    comm_y = hooks["Layer 4"][0]
    comm_x = 0.80
    comm_w, comm_h = 0.16, 0.082
    box = FancyBboxPatch((comm_x - comm_w/2, comm_y - comm_h/2), comm_w, comm_h,
                         boxstyle="round,pad=0,rounding_size=0.014",
                         fc=PANEL, ec=VIOLET, lw=1.2)
    box.set_path_effects(glow(VIOLET, n=3, base_w=2.6, alpha=0.20))
    ax.add_patch(box)
    ax.text(comm_x, comm_y + 0.016, "Community", color=VIOLET,
            ha="center", va="center", fontsize=11.5, weight="bold")
    ax.text(comm_x, comm_y - 0.005, "discourse basin",
            color=MUTED, ha="center", va="center", fontsize=8.5)
    ax.text(comm_x, comm_y - 0.024,
            "v8a: continuous 64-D · v3-v7: 32 prototypes",
            color=DIM, ha="center", va="center", fontsize=7.5)
    arrow(ax, (tx1, comm_y), (comm_x - comm_w/2, comm_y),
          color=VIOLET, lw=1.6, mutation=14, glow_color=VIOLET)

    # BEN box positioned roughly mid-height on the right
    ben_y = (hooks["Layer 14"][0] + hooks["Layer 21"][0]) / 2
    ben_x = 0.84
    ben_w, ben_h = 0.22, 0.13
    box = FancyBboxPatch((ben_x - ben_w/2, ben_y - ben_h/2), ben_w, ben_h,
                         boxstyle="round,pad=0,rounding_size=0.018",
                         fc=PANEL, ec=CORAL, lw=1.3)
    box.set_path_effects(glow(CORAL, n=3, base_w=2.8, alpha=0.22))
    ax.add_patch(box)
    ax.text(ben_x, ben_y + 0.042, "BEN", color=CORAL,
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(ben_x, ben_y + 0.018,
            "bifurcation-estimation network",
            color=MUTED, ha="center", va="center", fontsize=8.5)
    ax.text(ben_x, ben_y - 0.010,
            r"reflexivity  $\hat r$  (per token)",
            color=TEXT, ha="center", va="center", fontsize=9.5)
    ax.text(ben_x, ben_y - 0.034,
            "regime ∈ {subcritical, supercritical}",
            color=MUTED, ha="center", va="center", fontsize=8.5,
            style="italic")
    # arrow from RRM to BEN: route below the tower to avoid crossing slabs
    via_y = ty0 - 0.035
    arrow(ax, (rrm_x + rrm_w/2, rrm_bot - 0.002),
          (rrm_x + rrm_w/2, via_y),
          color=MAGENTA, lw=1.2, mutation=10, alpha=0.85)
    arrow(ax, (rrm_x + rrm_w/2, via_y),
          (ben_x, via_y),
          color=MAGENTA, lw=1.2, mutation=10, alpha=0.85)
    arrow(ax, (ben_x, via_y),
          (ben_x, ben_y - ben_h/2),
          color=MAGENTA, lw=1.2, mutation=10, alpha=0.85)
    ax.text((rrm_x + rrm_w/2 + ben_x) / 2, via_y + 0.015,
            "meta-state  h_meta", color=MAGENTA, fontsize=8.5,
            ha="center", style="italic")
    # FiLM inject equation just under the RRM box, off to the left
    ax.text(rrm_x + rrm_w/2, rrm_bot - 0.045,
            r"$h \leftarrow h \cdot (1+\gamma) + \beta$",
            color=MINT, fontsize=10, ha="center", va="center",
            style="italic")

    # ---- Legend chips at bottom ------------------------------------------
    legend_items = [
        (FROZEN,   "frozen backbone"),
        (CYAN,     "read hook → MAH"),
        (MINT,     "FiLM inject (γ, β)"),
        (MAGENTA,  "RRM · GRU meta-state"),
        (VIOLET,   "community tap"),
        (CORAL,    "BEN · reflexivity"),
    ]
    lx = 0.05
    ly = 0.040
    for (c, lab) in legend_items:
        ax.add_patch(Rectangle((lx, ly), 0.012, 0.012, fc=c, ec="none",
                               transform=fig.transFigure, clip_on=False))
        fig.text(lx + 0.018, ly + 0.006, lab, color=TEXT, fontsize=9,
                 va="center")
        lx += 0.155

    footer(fig, y=0.012)
    fig.savefig(OUT / "00_architecture.png")
    plt.close(fig)


# ===========================================================================
# Fig 0b — Visual grammar / legend (color + symbol key, used everywhere)
# ===========================================================================
def fig0b_legend():
    """A single reference panel: the colors, shapes, and symbols used
    across the entire explainer series. Read this once, parse every other
    figure at a glance.
    """
    fig = plt.figure(figsize=(13.5, 7.6))
    title_block(fig, "00b · visual grammar",
                "The color & symbol key for every SRT explainer",
                "Read this once, then every other figure parses at a glance.")

    # ---- Left half: color/shape grammar ----------------------------------
    ax = fig.add_axes([0.04, 0.08, 0.48, 0.78])
    hidden_axes(ax)
    ax.text(0.02, 0.97, "COLOR & SHAPE", color=CYAN, fontsize=10,
            weight="bold")

    rows = [
        (FROZEN,  "rect",   "frozen backbone slab",
                            "native LLM layer (no gradient)"),
        (CYAN,    "circle", "read hook → MAH",
                            "tap residual stream (read-only)"),
        (MINT,    "arrow",  "FiLM inject  γ, β",
                            "writes back into 2 layers only"),
        (MAGENTA, "rect",   "RRM · GRU meta-state",
                            "integrates divergence stream"),
        (VIOLET,  "rect",   "community tap",
                            "discourse basin from L4"),
        (CORAL,   "rect",   "BEN · reflexivity  r̂",
                            "per-token regime signal"),
    ]
    y = 0.86
    for color, kind, label, sub in rows:
        # icon
        if kind == "rect":
            ax.add_patch(FancyBboxPatch(
                (0.02, y - 0.025), 0.07, 0.05,
                boxstyle="round,pad=0,rounding_size=0.012",
                fc=PANEL, ec=color, lw=1.3))
            ax.add_patch(FancyBboxPatch(
                (0.02, y - 0.025), 0.07, 0.05,
                boxstyle="round,pad=0,rounding_size=0.012",
                fc="none", ec=color, lw=1.3,
                path_effects=glow(color, n=3, base_w=2.4, alpha=0.22)))
        elif kind == "circle":
            ax.add_patch(Circle((0.055, y), 0.018, fc=color, ec="none",
                path_effects=glow(color, n=3, base_w=2.4, alpha=0.30)))
        elif kind == "arrow":
            arrow(ax, (0.02, y), (0.09, y),
                  color=color, lw=2.2, mutation=14, glow_color=color)
        ax.text(0.12, y + 0.008, label, color=TEXT, fontsize=11,
                weight="semibold", va="center")
        ax.text(0.12, y - 0.018, sub, color=MUTED, fontsize=9, va="center")
        y -= 0.115

    # ---- Right half: symbol glossary -------------------------------------
    ax2 = fig.add_axes([0.54, 0.08, 0.42, 0.78])
    hidden_axes(ax2)
    ax2.text(0.02, 0.97, "SYMBOLS", color=MAGENTA, fontsize=10,
             weight="bold")

    syms = [
        (r"$h$",                       "hidden state vector at some layer"),
        (r"$d_t$",                     "divergence (direct vs contextual reading)"),
        (r"$\gamma,\ \beta$",          "FiLM scale & shift produced by RRM"),
        (r"$h \leftarrow h(1{+}\gamma){+}\beta$",
                                       "the only write back into the backbone"),
        (r"$h_{\rm meta}$",            "GRU state — what the adapter 'knows so far'"),
        (r"$\hat r$",                  "reflexivity (>0 ⇒ supercritical / forking)"),
        (r"$\mu$",                     "layer-mean (anisotropy offset, must subtract)"),
        (r"$\rm fve\_cen$",            "centered cosine — the honest similarity"),
    ]
    y = 0.88
    for sym, desc in syms:
        ax2.text(0.04, y, sym, color=CYAN, fontsize=14,
                 weight="bold", va="center")
        ax2.text(0.30, y, desc, color=TEXT, fontsize=10, va="center")
        y -= 0.095

    # ---- Footer hint -----------------------------------------------------
    fig.text(0.5, 0.045,
             "Every later figure reuses the colors above. "
             "If a glyph is mint, it is being written into the backbone. "
             "If it is cyan, it is read-only.",
             color=MUTED, fontsize=10, ha="center", style="italic")
    footer(fig, y=0.018)
    fig.savefig(OUT / "00b_legend.png")
    plt.close(fig)


# ===========================================================================
# Fig 11 — Token trace: follow one token end-to-end
# ===========================================================================
def fig11_token_trace():
    """The 'aha' figure: a single token's path through SRT, from input
    embedding through the three taps, the RRM correction, and the final
    logit shift that changes the model's answer.
    """
    fig = plt.figure(figsize=(14.5, 8.4))
    title_block(fig, "11 · one-shot trace",
                "Follow one token end-to-end through SRT",
                "Prompt: \"The capital of Australia is ___\"  ·  "
                "what the adapter sees, decides, and writes back.")

    ax = fig.add_axes([0.03, 0.06, 0.94, 0.80])
    hidden_axes(ax)

    # ---- Top strip: prompt tokens ----------------------------------------
    tokens = ["The", "capital", "of", "Australia", "is", "___"]
    n = len(tokens)
    tx0, tx1 = 0.06, 0.50
    ty = 0.92
    tw = (tx1 - tx0) / n
    for i, t in enumerate(tokens):
        x = tx0 + (i + 0.5) * tw
        fc = PANEL if i < n - 1 else "#2a2f6a"
        ec = PANEL_EDGE if i < n - 1 else MAGENTA
        ax.add_patch(FancyBboxPatch(
            (x - tw*0.42, ty - 0.028), tw*0.84, 0.056,
            boxstyle="round,pad=0,rounding_size=0.010",
            fc=fc, ec=ec, lw=1.0))
        ax.text(x, ty, t, color=TEXT, ha="center", va="center",
                fontsize=10, weight="semibold")
    ax.text(tx0 - 0.005, ty, "PROMPT", color=DIM, fontsize=8.5,
            ha="right", va="center", weight="bold")
    ax.text((tx0+tx1)/2, ty + 0.045, "we trace the final position (___)",
            color=MAGENTA, fontsize=9.5, ha="center", style="italic")

    # ---- Vertical backbone tower (left half) -----------------------------
    bx0, bx1 = 0.20, 0.34
    by0, by1 = 0.12, 0.84
    rail = FancyBboxPatch((bx0 - 0.008, by0 - 0.005),
                          (bx1 - bx0) + 0.016, (by1 - by0) + 0.01,
                          boxstyle="round,pad=0,rounding_size=0.018",
                          fc=PANEL, ec=PANEL_EDGE, lw=1.0)
    ax.add_patch(rail)
    bcx = (bx0 + bx1) / 2
    ax.text(bcx, by1 + 0.022, "frozen backbone",
            color=DIM, fontsize=8.5, weight="semibold", ha="center",
            style="italic")

    # slabs with three hook layers explicitly tagged
    slab_specs = [
        ("L0–6",   "frozen",                FROZEN, PANEL_EDGE, MUTED, None),
        ("L7",     "tap → MAH₁",            "#2a2f6a", CYAN,   TEXT, CYAN),
        ("L8–13",  "frozen",                FROZEN, PANEL_EDGE, MUTED, None),
        ("L14",    "tap + FiLM inject",     "#2a2f6a", MINT,   TEXT, MINT),
        ("L15–20", "frozen (w/ correction)",FROZEN, PANEL_EDGE, MUTED, None),
        ("L21",    "tap + FiLM inject",     "#2a2f6a", MINT,   TEXT, MINT),
        ("L22–27", "frozen (w/ correction)",FROZEN, PANEL_EDGE, MUTED, None),
    ]
    units = [1.0, 0.7, 1.2, 0.8, 1.2, 0.8, 1.2]
    total = sum(units)
    gap = 0.012
    avail = (by1 - by0) - gap * (len(units) - 1)
    cy = by1
    layer_y = {}
    for (lab, sub, fc, ec, tc, gl), u in zip(slab_specs, units):
        h = avail * u / total
        cy -= h
        yc = cy + h / 2
        box = FancyBboxPatch((bx0, cy), bx1 - bx0, h,
                             boxstyle="round,pad=0,rounding_size=0.008",
                             fc=fc, ec=ec, lw=1.0, alpha=0.95)
        if gl:
            box.set_path_effects(glow(gl, n=2, base_w=2.0, alpha=0.20))
        ax.add_patch(box)
        ax.text(bcx, yc + 0.010, lab, color=tc, ha="center",
                va="center", fontsize=9.5, weight="semibold")
        ax.text(bcx, yc - 0.012, sub, color=MUTED, ha="center",
                va="center", fontsize=8)
        layer_y[lab] = yc
        cy -= gap

    # token arrow into backbone
    arrow(ax, ((tx0+tx1)/2, ty - 0.030), (bcx, by1 + 0.005),
          color=DIM, lw=1.2, curve=-0.2)

    # ---- Middle: three numbered call-outs (Tap / Correct / Report) -------
    # ① TAP — divergences at L7, L14, L21 → MAH → RRM
    mx, my0 = 0.50, 0.30
    mw, mh = 0.16, 0.36
    box = FancyBboxPatch((mx, my0), mw, mh,
                         boxstyle="round,pad=0,rounding_size=0.014",
                         fc=PANEL, ec=MAGENTA, lw=1.2)
    box.set_path_effects(glow(MAGENTA, n=3, base_w=2.6, alpha=0.20))
    ax.add_patch(box)
    ax.text(mx + mw/2, my0 + mh - 0.030,
            "RRM (GRU)", color=MAGENTA, fontsize=12, weight="bold",
            ha="center")
    ax.text(mx + mw/2, my0 + mh - 0.055,
            "integrates  d₇ , d₁₄ , d₂₁",
            color=MUTED, fontsize=9, ha="center")

    # mini divergence sparkline inside the RRM box
    sx = np.linspace(mx + 0.01, mx + mw - 0.01, 60)
    rng = np.random.default_rng(7)
    sy = my0 + 0.12 + 0.04 * np.sin(np.linspace(0, 5, 60)) \
                + 0.015 * rng.standard_normal(60)
    sy[-15:] += np.linspace(0, 0.05, 15)   # ramp at end → "fork detected"
    neon_line(ax, sx, sy, color=CYAN, lw=1.6, glow_alpha=0.18)
    ax.text(mx + mw/2, my0 + 0.028,
            "divergence stream  d_t",
            color=CYAN, fontsize=8.5, ha="center", style="italic")
    ax.text(mx + mw/2, my0 + 0.008,
            "spike at last token  →  meaning forks",
            color=CORAL, fontsize=8, ha="center")

    # arrows from L7/L14/L21 hooks → RRM box (cyan)
    for lab in ("L7", "L14", "L21"):
        y_tap = layer_y[lab]
        arrow(ax, (bx1, y_tap), (mx, my0 + mh * 0.55),
              color=CYAN, lw=1.3, mutation=12, alpha=0.85, curve=0.15)

    # ② CORRECT — FiLM γ, β back into L14 and L21
    # ① TAP label sits between backbone and RRM, top
    ax.text(0.42, 0.72, "①  TAP", color=CYAN, fontsize=11, weight="bold")
    ax.text(0.42, 0.70, "read L7 / L14 / L21",
            color=MUTED, fontsize=8.5)

    # ② CORRECT label sits as a left-margin note (well clear of the
    # backbone column at x=0.20-0.34 and the chart starting at x=0.50).
    ax.text(0.04, 0.50, "②  CORRECT", color=MINT, fontsize=11,
            weight="bold")
    ax.text(0.04, 0.475, r"FiLM  $\gamma, \beta$", color=MINT,
            fontsize=10, style="italic")
    ax.text(0.04, 0.455, "back into L14 & L21",
            color=MUTED, fontsize=8.5)

    # mint inject arrows from RRM back to L14 and L21
    for lab in ("L14", "L21"):
        y_tgt = layer_y[lab]
        arrow(ax, (mx, my0 + mh * 0.35),
              (bx1 + 0.003, y_tgt + 0.004),
              color=MINT, lw=1.8, mutation=14,
              glow_color=MINT, curve=0.40, alpha=0.95)

    # ③ REPORT — BEN reflexivity
    ben_x, ben_y = 0.72, 0.46
    ben_w, ben_h = 0.20, 0.20
    box = FancyBboxPatch((ben_x, ben_y), ben_w, ben_h,
                         boxstyle="round,pad=0,rounding_size=0.014",
                         fc=PANEL, ec=CORAL, lw=1.2)
    box.set_path_effects(glow(CORAL, n=3, base_w=2.6, alpha=0.22))
    ax.add_patch(box)
    ax.text(ben_x + ben_w/2, ben_y + ben_h - 0.030,
            "BEN", color=CORAL, fontsize=13, weight="bold", ha="center")
    ax.text(ben_x + ben_w/2, ben_y + ben_h - 0.055,
            "reflexivity per token", color=MUTED, fontsize=9, ha="center")
    ax.text(ben_x + ben_w/2, ben_y + ben_h - 0.105,
            r"$\hat r = +1.42$", color=TEXT, fontsize=14,
            ha="center", weight="bold")
    ax.text(ben_x + ben_w/2, ben_y + ben_h - 0.135,
            "regime: SUPERCRITICAL",
            color=CORAL, fontsize=9, ha="center", weight="semibold")
    ax.text(ben_x + ben_w/2, ben_y + 0.020,
            "(\"a fork is happening here\")",
            color=MUTED, fontsize=8.5, ha="center", style="italic")
    # arrow RRM → BEN
    arrow(ax, (mx + mw, my0 + mh * 0.7),
          (ben_x, ben_y + ben_h * 0.5),
          color=MAGENTA, lw=1.3, mutation=12, curve=-0.15)
    # ③ REPORT label above the BEN box
    ax.text(ben_x + ben_w/2, ben_y + ben_h + 0.030,
            "③  REPORT", color=CORAL, fontsize=11, weight="bold",
            ha="center")
    ax.text(ben_x + ben_w/2, ben_y + ben_h + 0.010,
            "emit  r̂  +  regime",
            color=MUTED, fontsize=8.5, ha="center")

    # ---- Bottom: logit shift bar chart -----------------------------------
    chart_x, chart_y = 0.50, 0.10
    chart_w, chart_h = 0.42, 0.18
    ax.add_patch(FancyBboxPatch(
        (chart_x, chart_y), chart_w, chart_h,
        boxstyle="round,pad=0,rounding_size=0.012",
        fc=PANEL, ec=PANEL_EDGE, lw=1.0))
    ax.text(chart_x + 0.010, chart_y + chart_h - 0.020,
            "TOP-3 LOGITS  ·  before  vs  after SRT",
            color=CYAN, fontsize=9, weight="bold")

    cands = ["Canberra", "Sydney", "Melbourne"]
    before = np.array([0.32, 0.55, 0.13])
    after  = np.array([0.71, 0.21, 0.08])
    bar_x0 = chart_x + 0.03
    bar_y0 = chart_y + 0.025
    row_h  = 0.035
    bar_max_w = chart_w - 0.16
    for i, c in enumerate(cands):
        yy = chart_y + chart_h - 0.055 - i * row_h
        # label
        ax.text(bar_x0 - 0.005, yy + row_h*0.35, c,
                color=TEXT, fontsize=9, ha="right", va="center")
        # before bar (muted)
        wb = bar_max_w * before[i]
        ax.add_patch(Rectangle((bar_x0, yy + row_h*0.50),
                               wb, row_h*0.30,
                               fc=DIM, ec="none", alpha=0.7))
        # after bar (mint/coral)
        col = MINT if cands[i] == "Canberra" else CORAL if i == 1 else MUTED
        wa = bar_max_w * after[i]
        bar = Rectangle((bar_x0, yy + row_h*0.18), wa, row_h*0.30,
                        fc=col, ec="none")
        bar.set_path_effects(glow(col, n=2, base_w=2.0, alpha=0.20))
        ax.add_patch(bar)
        # numbers
        ax.text(bar_x0 + max(wb, wa) + 0.008, yy + row_h*0.35,
                f"{before[i]:.2f}  →  {after[i]:.2f}",
                color=MUTED, fontsize=8, va="center")

    ax.text(chart_x + 0.010, chart_y + 0.010,
            "result: SRT shifts probability mass from \"Sydney\" → \"Canberra\"",
            color=MINT, fontsize=8.5, style="italic")

    # ---- Footer numbered legend ------------------------------------------
    fig.text(0.5, 0.035,
             "① read   ·   ② correct   ·   ③ report   "
             "—   one forward pass, three things SRT does at every token.",
             color=TEXT, fontsize=10.5, ha="center", weight="semibold")
    footer(fig, y=0.014)
    fig.savefig(OUT / "11_token_trace.png")
    plt.close(fig)


# ===========================================================================
# Fig 12 — Peirce's triadic sign (Representamen · Object · Interpretant)
# ===========================================================================
def fig12_peirce_triad():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "12 · theory · peirce",
                "The triadic sign — the philosophical motivation for MAH",
                "Representamen · Object · Interpretant   →   "
                "icon / index / symbol as the interpretive lens on  d_t")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.84]); hidden_axes(ax)

    # --- Equilateral triangle of the sign ------------------------------------
    cx, cy, R = 0.30, 0.50, 0.22
    pts = {
        "R": (cx + R*math.cos(math.radians(90)),  cy + R*math.sin(math.radians(90))),
        "O": (cx + R*math.cos(math.radians(210)), cy + R*math.sin(math.radians(210))),
        "I": (cx + R*math.cos(math.radians(330)), cy + R*math.sin(math.radians(330))),
    }
    colors = {"R": CYAN, "O": VIOLET, "I": MAGENTA}
    labels = {
        "R": ("REPRESENTAMEN", "the sign-vehicle\nthe mark, the token"),
        "O": ("OBJECT", "what the sign\nstands for"),
        "I": ("INTERPRETANT", "the effect / next-sign\nthe reading"),
    }
    # edges
    for a, b in [("R","O"), ("O","I"), ("I","R")]:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=DIM, lw=1.2, alpha=0.6, zorder=1)
    # nodes
    for k, (x, y) in pts.items():
        c = Circle((x, y), 0.052, fc=PANEL, ec=colors[k], lw=1.6, zorder=3)
        c.set_path_effects(glow(colors[k], n=3, base_w=3.0, alpha=0.22))
        ax.add_patch(c)
        ax.text(x, y, k, color=colors[k], ha="center", va="center",
                fontsize=18, weight="bold", zorder=4)
        head, sub = labels[k]
        # place text outside each vertex (push well past glow radius)
        dx = (x - cx); dy = (y - cy)
        n = math.hypot(dx, dy)
        ox, oy = x + dx/n * 0.14, y + dy/n * 0.14
        ha = "center"
        ax.text(ox, oy + 0.018, head, color=colors[k], ha=ha, va="center",
                fontsize=10.5, weight="bold")
        ax.text(ox, oy - 0.022, sub, color=MUTED, ha=ha, va="center",
                fontsize=8.5)

    ax.text(cx, cy - 0.005, "SIGN", color=TEXT, ha="center", va="center",
            fontsize=11, weight="bold", alpha=0.85)
    ax.text(cx, cy - 0.030, "irreducibly triadic",
            color=MUTED, ha="center", va="center", fontsize=8.5, style="italic")

    # --- Chain of interpretants (semiosis) -----------------------------------
    chain_y = 0.16
    ax.text(0.30, chain_y + 0.10, "CHAIN OF INTERPRETANTS  ·  semiosis is recursive",
            color=CYAN, fontsize=9.5, weight="bold", ha="center")
    xs = np.linspace(0.10, 0.50, 5)
    for i, x in enumerate(xs):
        c = Circle((x, chain_y), 0.018, fc=PANEL, ec=MAGENTA, lw=1.2)
        ax.add_patch(c)
        ax.text(x, chain_y, f"I{i}", color=MAGENTA, ha="center", va="center",
                fontsize=8, weight="bold")
        if i < len(xs) - 1:
            arrow(ax, (x + 0.020, chain_y), (xs[i+1] - 0.020, chain_y),
                  color=DIM, lw=1.0, mutation=8, curve=0)
    ax.text(0.30, chain_y - 0.05,
            "each interpretant becomes the representamen of the next sign",
            color=MUTED, ha="center", fontsize=8.5, style="italic")

    # --- Right panel: mapping into MAH ---------------------------------------
    px, py, pw, ph = 0.58, 0.16, 0.38, 0.68
    add_panel_bg(fig, (px, py, pw, ph), radius=0.018)
    ax.text(px + 0.02, py + ph - 0.035,
            "→  HOW SRT OPERATIONALIZES THIS",
            color=CYAN, fontsize=9.5, weight="bold")
    ax.text(px + 0.02, py + ph - 0.07,
            "MAH produces ONE divergence channel per layer:\n"
            "d_t = proj_div( direct − contextual ).\n"
            "The three Peircean modes below are the\n"
            "interpretive lens for what a high ‖d_t‖ means\n"
            "— not three sub-projections inside the head.",
            color=TEXT, fontsize=9.5, linespacing=1.35, va="top")

    rows = [
        ("ICONIC",     "resemblance-like features  ·  e.g.  d_t spikes on a typo / mimicry",   CYAN,    "R"),
        ("INDEXICAL",  "context / pointing  ·  e.g.  d_t spikes on a pronoun, deictic, register shift",  VIOLET,  "O"),
        ("SYMBOLIC",   "conventional / rule  ·  e.g.  d_t spikes on a polysemous lemma",       MAGENTA, "I"),
    ]
    for i, (name, body, col, vert) in enumerate(rows):
        ry = py + ph - 0.28 - i * 0.13
        ax.add_patch(Rectangle((px + 0.02, ry - 0.05), pw - 0.04, 0.10,
                               fc=PANEL_EDGE, ec="none", alpha=0.55))
        # vertex chip
        ax.add_patch(Circle((px + 0.05, ry), 0.018, fc=PANEL, ec=col, lw=1.4))
        ax.text(px + 0.05, ry, vert, color=col, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(px + 0.085, ry + 0.020, name, color=col,
                fontsize=10, weight="bold")
        ax.text(px + 0.085, ry - 0.018, body, color=MUTED, fontsize=8.5)

    ax.text(px + 0.02, py + 0.045,
            "single divergence  d_t  ∈  ℝ^{d_div}   ·   one scalar magnitude  ‖d_t‖",
            color=TEXT, fontsize=10, style="italic")
    ax.text(px + 0.02, py + 0.020,
            "→ Peirce supplies the vocabulary for what the divergence "
            "is measuring, not its shape.",
            color=MINT, fontsize=9)

    footer(fig)
    fig.savefig(OUT / "12_peirce_triad.png")
    plt.close(fig)


# ===========================================================================
# Fig 13 — Silverstein indexical orders (n / n+1 / n+2)
# ===========================================================================
def fig13_silverstein_orders():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "13 · theory · silverstein",
                "Indexical orders — what RRM's GRU is actually tracking",
                "1st-order (presupposing) → 2nd-order (metapragmatic) → "
                "3rd-order (ideological / enregistered)")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.84]); hidden_axes(ax)

    # Three stacked horizontal bands (tiers), with arrows climbing the ladder.
    tiers = [
        ("3RD ORDER",
         "metapragmatic ideology  ·  enregisterment",
         "Beliefs ABOUT the link between sign and group.\n"
         "“People who say X are the kind of people who…”",
         MAGENTA, 0.70),
        ("2ND ORDER",
         "metapragmatic  ·  noticing the noticing",
         "Speakers become reflexively aware that a 1st-order index\n"
         "carries social meaning, and start deploying it strategically.",
         VIOLET, 0.46),
        ("1ST ORDER",
         "presupposing index  ·  pure co-occurrence",
         "A linguistic feature statistically co-occurs with a context\n"
         "(register, region, community). No awareness required.",
         CYAN, 0.22),
    ]
    bx0, bx1 = 0.06, 0.62
    for name, eyebrow, body, col, y in tiers:
        add_panel_bg(fig, (bx0, y - 0.085, bx1 - bx0, 0.17), radius=0.016)
        # left chip
        ax.add_patch(Rectangle((bx0 + 0.012, y - 0.068), 0.008, 0.135,
                               fc=col, ec="none"))
        ax.text(bx0 + 0.034, y + 0.048, name, color=col,
                fontsize=12, weight="bold")
        ax.text(bx0 + 0.034, y + 0.020, eyebrow, color=TEXT,
                fontsize=9.5, style="italic")
        ax.text(bx0 + 0.034, y - 0.005, body, color=MUTED, fontsize=9,
                linespacing=1.4, va="top")

    # Climbing arrows on the right side of the tiers (1→2→3)
    for ya, yb in [(0.22, 0.46), (0.46, 0.70)]:
        arrow(ax, (bx1 - 0.04, ya + 0.045), (bx1 - 0.04, yb - 0.07),
              color=MINT, lw=1.6, mutation=14, glow_color=MINT)
    ax.text(bx1 - 0.020, 0.46, "reflexive\nascent",
            color=MINT, fontsize=8.5, va="center", style="italic")

    # --- Right panel: mapping to SRT modules --------------------------------
    px, py, pw, ph = 0.66, 0.14, 0.30, 0.72
    add_panel_bg(fig, (px, py, pw, ph), radius=0.018)
    ax.text(px + 0.015, py + ph - 0.04,
            "→  WHERE EACH ORDER LIVES IN SRT",
            color=CYAN, fontsize=9.5, weight="bold")

    rows = [
        ("1st", CYAN,   "MAH  ·  divergence channel  d_t",
         "raw co-occurrence with context window"),
        ("2nd", VIOLET, "RRM  ·  GRU meta-state h_meta",
         "running record of where reads have diverged\n"
         "→ this is the 'noticing the noticing' channel"),
        ("3rd", MAGENTA,"Community head  +  BEN.r̂",
         "ideological / enregistered structure\n"
         "across discourse communities"),
    ]
    y0 = py + ph - 0.10
    for tag, col, head, body in rows:
        ax.add_patch(Circle((px + 0.030, y0), 0.020, fc=PANEL, ec=col, lw=1.4))
        ax.text(px + 0.030, y0, tag, color=col, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(px + 0.060, y0 + 0.018, head, color=TEXT,
                fontsize=10, weight="bold")
        ax.text(px + 0.060, y0 - 0.020, body, color=MUTED, fontsize=8.5)
        y0 -= 0.18

    ax.text(px + 0.015, py + 0.040,
            "RRM is, by design, the 2nd-order channel:\n"
            "it doesn't just see signs, it sees that signs\n"
            "are being read differently — and writes that\n"
            "back as a correction.",
            color=MINT, fontsize=9, style="italic")

    footer(fig)
    fig.savefig(OUT / "13_silverstein_orders.png")
    plt.close(fig)


# ===========================================================================
# Fig 14 — Kockelman's algorithmic sieving
# ===========================================================================
def fig14_kockelman_sieving():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "14 · theory · kockelman",
                "Algorithmic sieving — every layer is a filter on meaning",
                "Sieves convert continuous flux into discrete kinds.  "
                "Stacked sieves = a pipeline of selection pressures.")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.84]); hidden_axes(ax)

    # Sieve cascade as widening-then-narrowing trapezoids.
    sieves = [
        ("BACKBONE",   "next-token prediction",     FROZEN,  0.05),
        ("MAH",        "Peircean subspaces",        CYAN,    0.20),
        ("COMMUNITY",  "discourse-community gating", VIOLET, 0.35),
        ("RRM",        "reflexive accumulation",     MINT,   0.50),
        ("BEN",        "regime classifier  r̂",      MAGENTA,0.65),
    ]
    sx, sy0 = 0.10, 0.18
    sh = 0.70
    sw_top = 0.46
    sw_bot = 0.10
    # background funnel
    funnel_x = [sx, sx + sw_top, sx + (sw_top + sw_bot)/2 + sw_bot/2,
                sx + (sw_top - sw_bot)/2]
    funnel_y = [sy0 + sh, sy0 + sh, sy0, sy0]
    ax.fill(funnel_x, funnel_y, color=PANEL, alpha=0.55, zorder=0)
    ax.plot(funnel_x + [funnel_x[0]], funnel_y + [funnel_y[0]],
            color=PANEL_EDGE, lw=1.0, zorder=1)

    # Mesh markers (discrete grains) flowing through the funnel
    rng = np.random.default_rng(7)
    grain_xs = rng.uniform(sx + 0.04, sx + sw_top - 0.04, 90)
    grain_ys = rng.uniform(sy0 + 0.03, sy0 + sh - 0.03, 90)
    ax.scatter(grain_xs, grain_ys, s=10, c=DIM, alpha=0.55, zorder=2)

    # 5 horizontal sieve bars across the funnel
    for name, sub, col, t in sieves:
        # interpolate funnel width at fractional height t
        width = sw_top * (1 - t) + sw_bot * t
        cx0 = sx + (sw_top - width) / 2
        cx1 = cx0 + width
        yy = sy0 + sh * (1 - t)
        # the sieve bar
        ax.plot([cx0, cx1], [yy, yy], color=col, lw=2.6,
                path_effects=glow(col, n=3, base_w=3.0, alpha=0.22), zorder=4)
        # mesh dots on the bar
        for mx in np.linspace(cx0 + 0.006, cx1 - 0.006, max(6, int(width * 70))):
            ax.plot([mx], [yy], marker="|", color=col, ms=6, alpha=0.6, zorder=5)
        # label to the right
        ax.text(cx1 + 0.020, yy + 0.008, name, color=col,
                fontsize=11, weight="bold")
        ax.text(cx1 + 0.020, yy - 0.020, sub, color=MUTED, fontsize=9)

    # Out-flow arrow
    arrow(ax, (sx + sw_top/2, sy0 - 0.005),
          (sx + sw_top/2, sy0 - 0.05),
          color=MAGENTA, lw=2.0, mutation=18, glow_color=MAGENTA)
    ax.text(sx + sw_top/2, sy0 - 0.075,
            "selected, typed, reflexively-tagged output",
            color=MAGENTA, ha="center", fontsize=9.5, weight="bold")

    # --- Right panel: Kockelman quote / mapping ----------------------------
    px, py, pw, ph = 0.66, 0.16, 0.30, 0.66
    add_panel_bg(fig, (px, py, pw, ph), radius=0.018)
    pax = fig.add_axes([px, py, pw, ph]); hidden_axes(pax)
    pax.text(0.05, 0.94, "→  SIEVING, IN PRACTICE",
             color=CYAN, fontsize=10, weight="bold")
    pax.text(0.05, 0.86,
             "A sieve is anything that:\n"
             "  · takes a flux of inputs,\n"
             "  · applies a typing predicate,\n"
             "  · lets only certain kinds through.",
             color=TEXT, fontsize=10, linespacing=1.6, va="top")
    pax.text(0.05, 0.55,
             "Cascaded sieves yield ever-finer kinds.\n"
             "The LLM stack already does this for tokens;\n"
             "SRT adds sieves that type each token by its\n"
             "semiotic role and divergence regime.",
             color=MUTED, fontsize=9.5, linespacing=1.5, va="top")
    pax.text(0.05, 0.20,
             "Each colored bar above is a sieve.\n"
             "Each surviving grain is a token that\n"
             "has now been classed by every layer.",
             color=MINT, fontsize=9, style="italic",
             linespacing=1.5, va="top")

    footer(fig)
    fig.savefig(OUT / "14_kockelman_sieving.png")
    plt.close(fig)


# ===========================================================================
# Fig 15 — Lancaster pitchfork bifurcation (SSRN 5987495)
# ===========================================================================
def fig15_lancaster_pitchfork():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "15 · theory · lancaster",
                "The pitchfork bifurcation — what r̂ is estimating",
                "Lancaster (2025), “The Treachery of Signs.”  "
                "BEN regresses the order parameter of a polarization phase change.")
    # Left: classic pitchfork x³ - μx = 0 → x*(μ - x²) = 0 → x=0 or x=±√μ
    ax1 = fig.add_axes([0.06, 0.13, 0.50, 0.72])
    ax1.set_facecolor(BG)
    for s in ax1.spines.values(): s.set_visible(False)
    ax1.grid(True, color=GRID, alpha=0.30, lw=0.6)
    ax1.tick_params(colors=MUTED)
    ax1.set_xlabel("μ   ·   control parameter  (community polarization pressure)",
                   color=TEXT, fontsize=10.5)
    ax1.set_ylabel("x*  ·  order parameter   (consensus  ↔  faction split)",
                   color=TEXT, fontsize=10.5)
    ax1.set_title("Supercritical pitchfork:   ẋ = μx − x³",
                  color=TEXT, fontsize=12, weight="semibold", loc="left")
    ax1.text(0.015, 0.955,
             "(region labels below = sign of  (μ − μ_c),  not pitchfork type)",
             transform=ax1.transAxes, color=MUTED, fontsize=8.5,
             style="italic", ha="left", va="top")

    mu = np.linspace(-1.0, 1.6, 400)
    # stable branch below critical
    ax1.plot(mu[mu <= 0], np.zeros_like(mu[mu <= 0]),
             color=SUBCRIT, lw=2.4,
             path_effects=glow(SUBCRIT, n=3, base_w=3.2, alpha=0.22),
             label="stable consensus  (μ < μ_c)")
    # unstable middle branch
    ax1.plot(mu[mu > 0], np.zeros_like(mu[mu > 0]),
             color=MUTED, lw=1.4, ls="--", alpha=0.8,
             label="unstable consensus")
    # upper / lower stable branches above critical
    mup = mu[mu > 0]
    ax1.plot(mup,  np.sqrt(mup), color=SUPERCRIT, lw=2.4,
             path_effects=glow(SUPERCRIT, n=3, base_w=3.2, alpha=0.22),
             label="bifurcated branches  (μ > μ_c: two camps)")
    ax1.plot(mup, -np.sqrt(mup), color=SUPERCRIT, lw=2.4,
             path_effects=glow(SUPERCRIT, n=3, base_w=3.2, alpha=0.22))

    # critical point marker
    ax1.axvline(0, color=DIM, lw=0.8, ls=":")
    ax1.scatter([0], [0], s=70, fc=PANEL, ec=YELLOW, lw=1.6, zorder=5)
    ax1.annotate("μ_c   ·   critical point",
                 xy=(0, 0), xytext=(0.22, 0.45),
                 color=YELLOW, fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", color=YELLOW,
                                 lw=0.8, alpha=0.8))

    # regime shading
    ax1.axvspan(-1.0, 0.0, color=SUBCRIT,   alpha=0.05, zorder=0)
    ax1.axvspan( 0.0, 1.6, color=SUPERCRIT, alpha=0.05, zorder=0)
    ax1.text(-0.5, 1.10, "SUBCRITICAL", color=SUBCRIT, fontsize=10,
             weight="bold", ha="center")
    ax1.text(-0.5, 0.95, "single attractor", color=MUTED, fontsize=8.5,
             ha="center", style="italic")
    ax1.text( 0.8, 1.10, "SUPERCRITICAL", color=SUPERCRIT, fontsize=10,
             weight="bold", ha="center")
    ax1.text( 0.8, 0.95, "two stable camps  ·  contested signs",
             color=MUTED, fontsize=8.5, ha="center", style="italic")

    ax1.set_xlim(-1.0, 1.6); ax1.set_ylim(-1.4, 1.4)
    leg = ax1.legend(loc="lower right", framealpha=0.0,
                     labelcolor=TEXT, fontsize=8.5)

    # --- Right panel: mapping to BEN ---------------------------------------
    px, py, pw, ph = 0.60, 0.14, 0.36, 0.72
    add_panel_bg(fig, (px, py, pw, ph), radius=0.018)
    ax = fig.add_axes([px, py, pw, ph]); hidden_axes(ax)
    ax.text(0.04, 0.93, "→  WHAT BEN IS DOING",
            color=CYAN, fontsize=10, weight="bold")
    ax.text(0.04, 0.85,
            "BEN treats each token's local discourse as\n"
            "a sample from this 1-D normal form and\n"
            "estimates two quantities:",
            color=TEXT, fontsize=10)

    ax.add_patch(Rectangle((0.04, 0.62), 0.92, 0.13, fc=PANEL_EDGE,
                           ec="none", alpha=0.55))
    ax.text(0.06, 0.705, "r̂",   color=MAGENTA, fontsize=14, weight="bold")
    ax.text(0.13, 0.715, "continuous reflexivity coefficient",
            color=TEXT, fontsize=10, weight="semibold")
    ax.text(0.13, 0.680, "regression target  ≈  signed distance from μ_c",
            color=MUTED, fontsize=9)

    ax.add_patch(Rectangle((0.04, 0.45), 0.92, 0.13, fc=PANEL_EDGE,
                           ec="none", alpha=0.55))
    ax.text(0.06, 0.535, "ŷ",   color=CYAN, fontsize=14, weight="bold")
    ax.text(0.13, 0.545, "binary regime classifier",
            color=TEXT, fontsize=10, weight="semibold")
    ax.text(0.13, 0.510, "{ subcritical · supercritical }   ↔   pre / post μ_c",
            color=MUTED, fontsize=9)

    ax.text(0.04, 0.40,
            "INTERPRETATION",
            color=CYAN, fontsize=10, weight="bold")
    ax.text(0.04, 0.34,
            "• r̂ ≈ 0   →  consensus regime,\n"
            "                  single interpretant\n\n"
            "• r̂  >  0  →  meaning has bifurcated into\n"
            "                  competing community-\n"
            "                  conditioned readings",
            color=MUTED, fontsize=9.5, linespacing=1.45, va="top")

    ax.text(0.04, 0.06,
            "Stage-2 (Supabase news, 5 communities):\n"
            "Pearson r = 0.884  between r̂  and an\n"
            "external polarization metric.",
            color=MINT, fontsize=9, style="italic",
            linespacing=1.5, va="top")

    footer(fig)
    fig.savefig(OUT / "15_lancaster_pitchfork.png")
    plt.close(fig)


# ===========================================================================
# Fig 16 — Observer phase transitions (thermodynamic phase graph)
# ===========================================================================
def fig16_observer_phase():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "16 · theory · phase",
                "Observer phase transitions — the phase diagram of an interpreter",
                "Control axes: divergence pressure  μ   ×   reflexive temperature  T.   "
                "Three regimes, one critical line, one tricritical point.")
    ax = fig.add_axes([0.06, 0.10, 0.50, 0.76])
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(True, color=GRID, alpha=0.30, lw=0.6)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("μ   ·   semiotic divergence pressure",
                  color=TEXT, fontsize=10.5)
    ax.set_ylabel("T   ·   reflexive temperature  (channel capacity of RRM)",
                  color=TEXT, fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Phase regions
    mu_grid = np.linspace(0, 1, 400)
    # critical line: T = T_c(μ)  =  0.85 - 0.85*μ²  (cartoon)
    T_c = 0.85 - 0.85 * mu_grid**2

    # region fills
    ax.fill_between(mu_grid, T_c, 1.0, color=SUBCRIT,   alpha=0.10)
    ax.fill_between(mu_grid, 0.0, T_c, color=SUPERCRIT, alpha=0.10)
    # metastable wedge near critical line
    band = 0.05
    ax.fill_between(mu_grid, np.clip(T_c - band, 0, 1),
                    np.clip(T_c + band, 0, 1),
                    color=YELLOW, alpha=0.10)

    # critical curve
    ax.plot(mu_grid, T_c, color=YELLOW, lw=2.2,
            path_effects=glow(YELLOW, n=3, base_w=3.0, alpha=0.22),
            label="critical line  T_c(μ)")

    # tricritical / triple point cartoon
    ax.scatter([0.55], [0.85 - 0.85*0.55**2], s=80, fc=PANEL,
               ec=MAGENTA, lw=1.8, zorder=5)
    ax.annotate("tricritical\npoint",
                xy=(0.55, 0.85 - 0.85*0.55**2),
                xytext=(0.72, 0.78),
                color=MAGENTA, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MAGENTA, lw=0.8))

    # regime labels
    ax.text(0.18, 0.80, "CONSENSUS", color=SUBCRIT,   fontsize=12, weight="bold")
    ax.text(0.18, 0.74, "single interpretant\nstable across community",
            color=MUTED, fontsize=8.5, style="italic")
    ax.text(0.72, 0.22, "BIFURCATED", color=SUPERCRIT, fontsize=12, weight="bold")
    ax.text(0.72, 0.16, "two attractors\ncontested signs",
            color=MUTED, fontsize=8.5, style="italic")
    ax.text(0.05, 0.42, "METASTABLE", color=YELLOW,    fontsize=10.5, weight="bold")
    ax.text(0.05, 0.37, "small perturbations\ntoggle the reading",
            color=MUTED, fontsize=8.5, style="italic")

    # An example trajectory: a token's discourse drifts from consensus
    # → metastable → bifurcated as community pressure rises.
    tx = np.array([0.05, 0.22, 0.42, 0.62, 0.80])
    ty = np.array([0.90, 0.78, 0.60, 0.38, 0.18])
    ax.plot(tx, ty, color=MINT, lw=2.0,
            path_effects=glow(MINT, n=3, base_w=2.8, alpha=0.22))
    ax.scatter(tx, ty, s=28, c=MINT, zorder=5)
    ax.text(tx[0] + 0.01, ty[0] + 0.02, "token trajectory",
            color=MINT, fontsize=9, style="italic")

    leg = ax.legend(loc="upper right", framealpha=0.0,
                    labelcolor=TEXT, fontsize=8.5)

    # --- Right panel: mapping & references ---------------------------------
    px, py, pw, ph = 0.60, 0.12, 0.36, 0.74
    add_panel_bg(fig, (px, py, pw, ph), radius=0.018)
    axr = fig.add_axes([px, py, pw, ph]); hidden_axes(axr)
    axr.text(0.04, 0.94, "→  WHO IS THE OBSERVER?",
             color=CYAN, fontsize=10, weight="bold")
    axr.text(0.04, 0.86,
             "An observer is anything that draws a\n"
             "distinction.  Each SRT module is a different\n"
             "observer of the same hidden state:",
             color=TEXT, fontsize=10)
    rows = [
        ("MAH",       CYAN,    "typing observer  ·  R / O / I"),
        ("Community", VIOLET,  "social observer  ·  which discourse?"),
        ("RRM",       MINT,    "historical observer  ·  what has diverged?"),
        ("BEN",       MAGENTA, "phase observer  ·  which regime?"),
    ]
    y0 = 0.66
    for tag, col, body in rows:
        axr.add_patch(Rectangle((0.04, y0 - 0.025), 0.92, 0.06,
                                fc=PANEL_EDGE, ec="none", alpha=0.55))
        axr.text(0.07, y0, tag, color=col, fontsize=10, weight="bold",
                 va="center")
        axr.text(0.27, y0, body, color=MUTED, fontsize=9, va="center")
        y0 -= 0.08

    axr.text(0.04, 0.34, "THEORETICAL ANCESTORS",
             color=CYAN, fontsize=10, weight="bold")
    axr.text(0.04, 0.28,
             "von Foerster  ·  second-order cybernetics\n"
             "Maturana & Varela  ·  autopoiesis, observer\n"
             "Anderson  ·  more is different (broken symm.)\n"
             "Bennett · Landauer · Parrondo  ·  thermo-\n"
             "                                                dynamics of computation",
             color=MUTED, fontsize=9, linespacing=1.5, va="top")
    axr.text(0.04, 0.04,
             "Crossing a phase boundary requires work —\n"
             "RRM's correction term is that work.",
             color=MINT, fontsize=9, style="italic",
             linespacing=1.5, va="top")

    footer(fig)
    fig.savefig(OUT / "16_observer_phase.png")
    plt.close(fig)


# ===========================================================================
# Fig 17 — Reference constellation
# ===========================================================================
def fig17_references():
    fig = plt.figure(figsize=(14.0, 8.0))
    title_block(fig, "17 · theory · references",
                "Reference constellation — the lineages SRT draws on",
                "Each star is a thread; the SRT module each thread feeds is colored to match.")
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.84]); hidden_axes(ax)

    groups = [
        ("SEMIOTICS",  CYAN, [
            ("Peirce",                 "triadic sign  →  MAH",            0.10, 0.80),
            ("Silverstein",            "indexical orders  →  RRM",        0.10, 0.66),
            ("Kockelman",              "algorithmic sieving  →  stack",   0.10, 0.52),
            ("Wildgen",                "catastrophic semantics",          0.10, 0.38),
            ("Evans",                  "lexical concepts & cognitive models",
                                                                          0.10, 0.24),
        ]),
        ("DYNAMICS  &  PHASE",  CORAL, [
            ("Lancaster (2025)",       "pitchfork bifurcation  →  BEN",   0.40, 0.80),
            ("Anderson",               "more is different · broken symmetry",
                                                                          0.40, 0.62),
            ("Leighton",               "critical phenomena in collective systems",
                                                                          0.40, 0.44),
            ("VanSaders",              "observer-dependent phase structure",
                                                                          0.40, 0.26),
        ]),
        ("OBSERVER  &  THERMO",  MINT, [
            ("von Foerster",           "second-order cybernetics",        0.70, 0.80),
            ("Maturana & Varela",      "autopoiesis  ·  observer",        0.70, 0.66),
            ("Bennett",                "logical depth · thermodynamics of computation",
                                                                          0.70, 0.52),
            ("Landauer",               "information is physical",         0.70, 0.38),
            ("Parrondo et al.",        "thermodynamics of information",   0.70, 0.24),
        ]),
    ]
    for header, col, items in groups:
        # group header bar
        gx = items[0][2] - 0.04
        ax.text(gx, 0.91, header, color=col, fontsize=11, weight="bold")
        ax.plot([gx, gx + 0.24], [0.895, 0.895], color=col, lw=1.2, alpha=0.7)
        for name, body, x, y in items:
            # star
            c = Circle((x, y), 0.013, fc=PANEL, ec=col, lw=1.4)
            c.set_path_effects(glow(col, n=3, base_w=2.6, alpha=0.22))
            ax.add_patch(c)
            ax.text(x + 0.024, y + 0.022, name, color=TEXT,
                    fontsize=10, weight="semibold")
            ax.text(x + 0.024, y - 0.018, body, color=MUTED, fontsize=8.7,
                    linespacing=1.4, va="top")

    # bottom band: Lancaster SSRN links and present paper
    add_panel_bg(fig, (0.06, 0.045, 0.88, 0.10), radius=0.014)
    ax.text(0.08, 0.115, "PRIMARY SOURCES FOR SRT",
            color=CYAN, fontsize=9.5, weight="bold")
    ax.text(0.08, 0.085,
            "Lancaster (2025)  ·  “The Treachery of Signs: Semiotic Mediation, "
            "Pitchfork Bifurcations …”  ·  SSRN 5987495",
            color=TEXT, fontsize=9.5)
    ax.text(0.08, 0.063,
            "Lancaster (2026a)  ·  Stage-1 + Stage-2 validation of SRT primitives  "
            "·  SSRN 6349978",
            color=TEXT, fontsize=9.5)

    footer(fig)
    fig.savefig(OUT / "17_references.png")
    plt.close(fig)


def main():
    apply_style()
    fig0_architecture()
    fig0b_legend()
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
    fig11_token_trace()
    fig12_peirce_triad()
    fig13_silverstein_orders()
    fig14_kockelman_sieving()
    fig15_lancaster_pitchfork()
    fig16_observer_phase()
    fig17_references()
    written = sorted(OUT.glob("*.png"))
    print(f"[explainers] wrote {len(written)} figures →")
    for p in written:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
