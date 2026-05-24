"""Shared dark-scientific / agency-polish style for SRT explainer visuals.

Palette: deep navy-indigo base, pastel neon accents. Subtle glow on key
elements. Minimal chrome, generous whitespace, clean typography.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.patches import FancyBboxPatch

# ---- Palette ---------------------------------------------------------------
BG          = "#0a0e27"   # page background
PANEL       = "#141b3a"   # panel / card fill
PANEL_EDGE  = "#1f2a55"   # panel border
GRID        = "#243066"   # grid lines (low-contrast)
TEXT        = "#e8ecff"   # primary text
MUTED       = "#8a93c4"   # secondary text
DIM         = "#5b669a"   # tertiary text / axis

# pastel neon accents
CYAN        = "#7fd4ff"
MAGENTA     = "#ff79c6"
MINT        = "#7fffd4"
VIOLET      = "#b4a7ff"
PEACH       = "#ffb380"
YELLOW      = "#fff59d"
CORAL       = "#ff8a80"
LIME        = "#c8ff7f"

ACCENTS = [CYAN, MAGENTA, MINT, VIOLET, PEACH, YELLOW, CORAL, LIME]

# Semantic colors
FROZEN      = "#3a4380"   # frozen backbone (cool muted)
TRAINABLE   = MAGENTA     # trainable adapter (hot)
HOOK        = CYAN
INJECT      = MINT
SUBCRIT     = CYAN        # subcritical (stable)
SUPERCRIT   = CORAL       # supercritical (bifurcating)


def apply_style():
    """Set global matplotlib rcParams for the dark-scientific theme."""
    mpl.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     BG,
        "savefig.facecolor":  BG,
        "savefig.edgecolor":  BG,
        "axes.edgecolor":     PANEL_EDGE,
        "axes.labelcolor":    TEXT,
        "axes.titlecolor":    TEXT,
        "axes.titlesize":     14,
        "axes.titleweight":   "semibold",
        "axes.labelsize":     11,
        "axes.labelweight":   "regular",
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
        "grid.color":         GRID,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.6,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "text.color":         TEXT,
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "legend.frameon":     False,
        "legend.labelcolor":  TEXT,
        "legend.fontsize":    9,
        "lines.linewidth":    2.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth":    0.0,
        "figure.dpi":         140,
        "savefig.dpi":        180,
        "savefig.bbox":       "tight",
    })


def glow(color: str, n: int = 4, base_w: float = 4.0, alpha: float = 0.18):
    """Path-effect list approximating a soft neon glow under a stroke."""
    fx = [withStroke(linewidth=base_w + i * 2.5, foreground=color, alpha=alpha)
          for i in range(n)]
    return fx


def panel(ax, *, pad: float = 0.0):
    """Style an axes as a soft rounded 'card'."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def add_panel_bg(fig, rect, *, fill=PANEL, edge=PANEL_EDGE, lw=1.0,
                 radius=0.018, alpha=1.0, zorder=0):
    """Place a rounded rectangle on the figure (in figure-relative coords)."""
    x, y, w, h = rect
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, alpha=alpha,
        transform=fig.transFigure, zorder=zorder, clip_on=False,
    )
    fig.add_artist(box)
    return box


def title_block(fig, eyebrow: str, title: str, subtitle: str | None = None,
                *, x: float = 0.045, y: float = 0.935):
    """Standard top-left title block: small eyebrow + large title + subtitle."""
    fig.text(x, y + 0.048, eyebrow.upper(),
             color=CYAN, fontsize=9, weight="bold",
             family="DejaVu Sans")
    fig.text(x, y, title, color=TEXT, fontsize=19, weight="bold",
             family="DejaVu Sans")
    if subtitle:
        fig.text(x, y - 0.032, subtitle, color=MUTED, fontsize=10.5,
                 family="DejaVu Sans")


def footer(fig, left: str = "SRT · semiotic-reflexive transformer",
           right: str = "github.com/space-bacon/SRT",
           y: float = 0.022):
    fig.text(0.045, y, left, color=DIM, fontsize=8, family="DejaVu Sans")
    fig.text(0.955, y, right, color=DIM, fontsize=8, family="DejaVu Sans",
             ha="right")


def neon_line(ax, x, y, *, color=CYAN, lw=2.2, glow_alpha=0.22, **kw):
    """Plot a glowing neon line."""
    ax.plot(x, y, color=color, lw=lw,
            path_effects=glow(color, n=4, base_w=lw + 1.5, alpha=glow_alpha),
            **kw)


def neon_bar(ax, xs, ys, *, color=CYAN, width=0.7, glow_alpha=0.18,
             edgecolor=None, **kw):
    """Draw bars with a neon halo."""
    bars = ax.bar(xs, ys, width=width, color=color,
                  edgecolor=edgecolor or color, linewidth=0.8, **kw)
    for b in bars:
        b.set_path_effects(glow(color, n=3, base_w=2.5, alpha=glow_alpha))
    return bars


def chip(ax, x, y, text, *, color=CYAN, fc=None, alpha=0.18,
         fontsize=9, weight="semibold"):
    """A small rounded 'chip' label."""
    fc = fc or color
    ax.text(x, y, f" {text} ", color=color, fontsize=fontsize, weight=weight,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.4",
                      fc=fc, ec=color, alpha=alpha, lw=0.8),
            transform=ax.transAxes if not hasattr(x, '__len__') else ax.transData)


def hidden_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax
