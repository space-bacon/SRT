"""Gradio app for `srt_introspect.Trace` — adaptive-density reasoning trace.

A single tab over a frozen Qwen-2.5-7B backbone:

  prompt → generated continuation, every token tinted by SRT adapter
  divergence, with hover-cards showing the activation verbalizer's
  best-guess narration at scheduler-picked positions.

Run locally on a GPU box:
    pip install -r demo/requirements.txt
    PYTHONPATH=. python demo/srt_introspect_app.py

Or deploy as an HF Space (hardware: a10g-small / zero-a100 work fine —
Qwen-7B needs ~16 GB VRAM in bf16, plus the AV decoder ~2 GB).
"""
from __future__ import annotations

import logging
import os
import pathlib
import time

import gradio as gr
import torch

from srt_introspect import Trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("srt_introspect_app")

# Optional ZeroGPU support — same pattern as demo/app.py.
try:
    import spaces  # type: ignore

    _ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU")) or hasattr(spaces, "GPU")

    def _gpu(duration: int = 180):
        if _ON_ZEROGPU:
            return spaces.GPU(duration=duration)
        return lambda fn: fn
except ImportError:  # pragma: no cover
    _ON_ZEROGPU = False

    def _gpu(duration: int = 180):
        return lambda fn: fn


DEVICE = "cuda" if (torch.cuda.is_available() or _ON_ZEROGPU) else "cpu"
log.info("device=%s zero_gpu=%s", DEVICE, _ON_ZEROGPU)

_TRACE: Trace | None = None


def _get_trace() -> Trace:
    global _TRACE
    if _TRACE is None:
        log.info("Loading SRT adapter + activation verbalizer on %s", DEVICE)
        _TRACE = Trace.load(device=DEVICE)
    return _TRACE


# ---------- palette (matches scripts/demos/render_trace_html.py) ----------
BG = "#0a1429"
PANEL = "#16213d"
PANEL_ALT = "#1e2d4f"
INK = "#e6ecf5"
DIM = "#8a9bb8"
RULE = "#26345a"
CYAN = "#7ee0ff"
MINT = "#7eebc0"
PINK = "#ff7eb9"
LAVENDER = "#b8a4ff"


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _hex(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _div_color(div: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (div - lo) / (hi - lo)))
    c, m, p = _hex(CYAN), _hex(MINT), _hex(PINK)
    r, g, b = _lerp_rgb(c, m, t * 2) if t < 0.5 else _lerp_rgb(m, p, (t - 0.5) * 2)
    return f"rgba({r},{g},{b},0.30)"


# Per-MAH-layer sparkline colours, cycled by layer index in the order
# returned by Trace (== adapter.config.mah_layer_indices order).
_LAYER_COLORS = (CYAN, LAVENDER, PINK, MINT)


def _render_aggregate_curve(steps) -> str:
    """SVG with two stacked traces — aggregate divergence (pink) and next-token
    entropy (cyan) — plus a tick row marking verbalization positions. Lets
    viewers see whether the adapter's signal lines up with raw uncertainty
    or whether it's catching something the entropy can't.
    """
    if not steps or len(steps) < 2:
        return ""
    divs = [s.divergence for s in steps]
    ents = [s.entropy for s in steps]
    verb_idxs = [i for i, s in enumerate(steps) if s.verbalization]

    W, H, PAD_X, PAD_Y = 1000, 130, 8, 14
    n = len(steps)

    def _poly(vals, color):
        vmin, vmax = min(vals), max(vals)
        rng = (vmax - vmin) or 1.0
        pts = []
        for i, v in enumerate(vals):
            x = PAD_X + (W - 2 * PAD_X) * (i / (n - 1))
            y = PAD_Y + (H - 2 * PAD_Y) * (1.0 - (v - vmin) / rng)
            pts.append(f"{x:.1f},{y:.1f}")
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="0.95" '
            f'points="{" ".join(pts)}"/>'
        )

    ticks = "".join(
        f'<line x1="{PAD_X + (W - 2 * PAD_X) * (i / (n - 1)):.1f}" '
        f'x2="{PAD_X + (W - 2 * PAD_X) * (i / (n - 1)):.1f}" '
        f'y1="{H - 4}" y2="{H - 1}" stroke="{LAVENDER}" '
        f'stroke-width="1.2" opacity="0.85"/>'
        for i in vrb_safe(verb_idxs, n)
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" '
        f'class="srt-spk-svg" style="height:130px" '
        f'aria-label="aggregate divergence and next-token entropy">'
        f'{_poly(divs, PINK)}{_poly(ents, CYAN)}{ticks}</svg>'
    )
    legend = (
        f'<span class="srt-spk-key"><span class="dot" style="background:{PINK}"></span>'
        f'aggregate div</span>'
        f'<span class="srt-spk-key"><span class="dot" style="background:{CYAN}"></span>'
        f'next-tok entropy</span>'
        f'<span class="srt-spk-key"><span class="dot" '
        f'style="background:{LAVENDER};width:2px;height:10px;border-radius:1px"></span>'
        f'verbalization</span>'
    )
    return (
        '<div class="srt-spk">'
        '<div class="srt-spk-head">'
        '<span class="srt-spk-title">aggregate signal vs. predictive uncertainty</span>'
        f'<span class="srt-spk-legend">{legend}</span>'
        '</div>'
        f'{svg}'
        '<div class="srt-spk-axis"><span>token 0</span>'
        f'<span>token {n - 1}</span></div>'
        '</div>'
    )


def vrb_safe(idxs, n):
    # tiny guard: drop out-of-range indices in case caller mismatched lengths
    return [i for i in idxs if 0 <= i < n]


def _render_layer_sparkline(steps, layer_indices: list[int]) -> str:
    """Tiny inline SVG showing one polyline per MAH layer over token index.

    Makes the "tap layers 7/14/21" claim visible — viewers can see whether
    early vs. late layers are doing the work at each token.
    """
    if not steps:
        return ""
    series = [s.per_layer_divergence for s in steps]
    n_layers = max((len(p) for p in series), default=0)
    if n_layers == 0:
        return ""
    # If config didn't surface labels, fall back to 0..n-1.
    labels = layer_indices if len(layer_indices) == n_layers else list(range(n_layers))

    W, H, PAD_X, PAD_Y = 1000, 110, 8, 14
    n = len(steps)
    if n < 2:
        return ""

    # Per-layer min/max for independent y-scaling per line (so a weak
    # layer is still readable next to a dominant one).
    polylines = []
    for li in range(n_layers):
        vals = [(p[li] if li < len(p) else 0.0) for p in series]
        vmin, vmax = min(vals), max(vals)
        rng = (vmax - vmin) or 1.0
        pts = []
        for i, v in enumerate(vals):
            x = PAD_X + (W - 2 * PAD_X) * (i / (n - 1))
            # plot from top of band; lower y = larger value
            y = PAD_Y + (H - 2 * PAD_Y) * (1.0 - (v - vmin) / rng)
            pts.append(f"{x:.1f},{y:.1f}")
        color = _LAYER_COLORS[li % len(_LAYER_COLORS)]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.4" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'opacity="0.95" points="{" ".join(pts)}"/>'
        )

    legend_bits = "".join(
        f'<span class="srt-spk-key"><span class="dot" '
        f'style="background:{_LAYER_COLORS[li % len(_LAYER_COLORS)]}"></span>'
        f'L{labels[li]}</span>'
        for li in range(n_layers)
    )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" '
        f'class="srt-spk-svg" aria-label="per-layer divergence over tokens">'
        f'{"".join(polylines)}</svg>'
    )
    return (
        '<div class="srt-spk">'
        f'<div class="srt-spk-head">'
        f'<span class="srt-spk-title">per-layer divergence</span>'
        f'<span class="srt-spk-legend">{legend_bits}</span>'
        '</div>'
        f'{svg}'
        '<div class="srt-spk-axis"><span>token 0</span>'
        f'<span>token {n - 1}</span></div>'
        '</div>'
    )


def _render_trace_html(result, prompt: str, elapsed: float,
                       layer_indices: list[int] | None = None,
                       title: str | None = None) -> str:
    import html as _html

    steps = result.steps
    if not steps:
        return '<div style="color:#8a9bb8;padding:1rem">(no tokens generated)</div>'

    divs = [s.divergence for s in steps]
    ds = sorted(divs)
    lo = ds[max(0, int(0.10 * len(ds)))]
    hi = ds[min(len(ds) - 1, int(0.90 * len(ds)))]
    if hi <= lo:
        lo, hi = min(divs), max(divs)

    # Detect regime flips: any step whose regime differs from the previous.
    # The first step is never a flip. These get an extra outline + glyph so
    # viewers can spot BEN-state transitions without hovering each token.
    flip_set: set[int] = set()
    prev_reg = steps[0].regime
    for s in steps[1:]:
        if s.regime != prev_reg:
            flip_set.add(s.token_idx)
            prev_reg = s.regime

    spans = []
    for s in steps:
        disp = _html.escape(s.token).replace("\n", "<br>")
        bg = _div_color(s.divergence, lo, hi)
        classes = ["tok"]
        if s.verbalization:
            classes.append("selected")
        if s.regime == 0:
            classes.append("reg-bif")
        if s.token_idx in flip_set:
            classes.append("reg-flip")
        klass = " ".join(classes)
        title_txt = (
            f"i={s.token_idx}  ·  d={s.divergence:.2f}  ·  "
            f"H={s.entropy:.2f}  ·  r̂={s.r_hat:.2f}  ·  reg={s.regime}"
        )
        if s.verbalization:
            title_txt += f"  →  {s.verbalization[:240]}"
        # onclick pins the verbalization (or the metric line) to the side panel.
        # JS lives in _TRACE_CSS so it's defined once per page load.
        pin_payload = _html.escape(
            (s.verbalization or title_txt), quote=True
        ).replace("\n", " ")
        onclick = f"srtPin(this,&quot;{pin_payload}&quot;)"
        spans.append(
            f'<span class="{klass}" style="background:{bg}" '
            f'data-title="{_html.escape(title_txt)}" '
            f'onclick="{onclick}">{disp}</span>'
        )

    sel = result.selected()
    sel_rows = "".join(
        f'<tr><td class="num">{s.token_idx}</td>'
        f'<td><code>{_html.escape(s.token)}</code></td>'
        f'<td class="num">{s.divergence:.2f}</td>'
        f'<td class="num">{s.r_hat:.2f}</td>'
        f'<td class="num">{s.regime}</td>'
        f'<td class="verb">{_html.escape(s.verbalization or "")}</td></tr>'
        for s in sel
    )

    n_flip = len(flip_set)
    n_bif = sum(1 for s in steps if s.regime == 0)
    title_html = (
        f'<div class="srt-trace-title">{_html.escape(title)}</div>' if title else ""
    )

    return f"""
<div class="srt-trace">
  {title_html}
  <div class="srt-meta">
    <span class="chip"><span class="lbl">tokens</span>{len(steps)}</span>
    <span class="chip"><span class="lbl">verbalizations</span>{len(sel)}</span>
    <span class="chip"><span class="lbl">regime flips</span>{n_flip}</span>
    <span class="chip"><span class="lbl">bif (r=0)</span>{n_bif}</span>
    <span class="chip"><span class="lbl">div range</span>{lo:.2f} → {hi:.2f}</span>
    <span class="chip"><span class="lbl">elapsed</span>{elapsed:.1f}s</span>
  </div>
  <div class="srt-legend">
    <span>divergence</span>
    <span class="grad"></span>
    <span style="color:{CYAN}">low</span><span>→</span><span style="color:{PINK}">high</span>
    <span style="opacity:.5">·</span>
    <span class="box">box</span><span>= verbalization (hover)</span>
    <span style="opacity:.5">·</span>
    <span class="bif-key">▲</span><span>= bifurcating regime (r=0)</span>
    <span style="opacity:.5">·</span>
    <span class="flip-key">⇋</span><span>= regime flip</span>
    <span style="opacity:.5">·</span>
    <span style="color:{LAVENDER}">click any token to pin</span>
  </div>
  <div class="srt-prompt">{_html.escape(prompt)}</div>
  <div class="srt-response">{''.join(spans)}</div>
  {_render_aggregate_curve(steps)}
  {_render_layer_sparkline(steps, layer_indices or [])}
  <div class="srt-pinboard" id="srt-pinboard">
    <div class="srt-pin-head">
      <span class="srt-spk-title">pinned</span>
      <button class="srt-pin-clear" onclick="srtPinClear()">clear</button>
    </div>
    <ol class="srt-pin-list"></ol>
    <div class="srt-pin-empty">click tokens above to pin their verbalization or metrics here</div>
  </div>
  <div class="srt-label">Selected verbalizations</div>
  <table class="srt-table">
    <thead><tr><th>idx</th><th>token</th><th>div</th><th>r̂</th><th>reg</th><th>verbalization (AV)</th></tr></thead>
    <tbody>{sel_rows}</tbody>
  </table>
</div>
"""


# Gradio HTML components are sandboxed, so all styles must be inline-injected
# alongside the markup. We ship the CSS once at app boot via gr.HTML.
_TRACE_CSS = f"""
<style>
.srt-trace {{
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  color: {INK};
}}
.srt-meta {{
  display: flex; flex-wrap: wrap; gap: 0.5rem 0.75rem;
  margin-bottom: 0.8rem;
}}
.srt-meta .chip {{
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: {PANEL}; border: 1px solid {RULE};
  padding: 0.22rem 0.6rem; border-radius: 999px;
  font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.78rem;
  color: {INK};
}}
.srt-meta .chip .lbl {{ color: {DIM}; }}
.srt-legend {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.75rem;
  font-size: 0.78rem; color: {DIM}; margin-bottom: 0.8rem;
}}
.srt-legend .grad {{
  display: inline-block; width: 180px; height: 0.5rem; border-radius: 999px;
  background: linear-gradient(90deg, {CYAN}, {MINT}, {PINK});
  box-shadow: 0 0 10px rgba(126,235,192,0.25);
}}
.srt-legend .box {{
  display: inline-block; padding: 0.05rem 0.45rem; border-radius: 4px;
  outline: 1.5px solid {LAVENDER}; outline-offset: 1px; color: {INK};
}}
.srt-prompt {{
  background: {PANEL}; border-left: 3px solid {CYAN};
  padding: 0.7rem 0.9rem; border-radius: 6px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  white-space: pre-wrap; font-size: 0.82rem; color: {INK};
  margin-bottom: 0.8rem;
}}
.srt-response {{
  background: {PANEL}; border: 1px solid {RULE}; border-radius: 10px;
  padding: 1.2rem 1.4rem; line-height: 2.0; font-size: 1.0rem;
  color: {INK};
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
}}
.srt-response .tok {{
  display: inline; padding: 1px 1px; border-radius: 3px; position: relative;
}}
.srt-response .tok.selected {{
  outline: 1.5px solid {LAVENDER}; outline-offset: 1px;
  cursor: help; box-shadow: 0 0 10px rgba(184,164,255,0.35);
}}
.srt-response .tok:hover {{ background: rgba(184,164,255,0.35) !important; }}
.srt-response .tok:hover::after {{
  content: attr(data-title);
  position: absolute; left: 50%; transform: translateX(-50%);
  top: 1.7em; z-index: 20;
  background: {PANEL_ALT}; color: {INK};
  padding: 0.55rem 0.75rem; border-radius: 6px;
  border: 1px solid {LAVENDER};
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.74rem; line-height: 1.45;
  white-space: pre-wrap; max-width: 460px; min-width: 220px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.45);
  pointer-events: none;
}}
.srt-label {{
  font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: {DIM}; margin: 1.4rem 0 0.5rem; font-weight: 600;
}}
.srt-spk {{
  background: {PANEL}; border: 1px solid {RULE}; border-radius: 8px;
  padding: 0.6rem 0.8rem 0.5rem; margin: 0.6rem 0 0;
}}
.srt-spk-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.8rem; margin-bottom: 0.3rem;
}}
.srt-spk-title {{
  font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: {DIM}; font-weight: 600;
}}
.srt-spk-legend {{
  display: flex; gap: 0.8rem; font-size: 0.72rem; color: {INK};
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}}
.srt-spk-key {{ display: inline-flex; align-items: center; gap: 0.3rem; }}
.srt-spk-key .dot {{
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}}
.srt-spk-svg {{
  display: block; width: 100%; height: 110px;
  background: {PANEL_ALT}; border-radius: 6px;
}}
.srt-spk-axis {{
  display: flex; justify-content: space-between;
  font-size: 0.62rem; color: {DIM}; margin-top: 0.25rem;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}}
.srt-table {{
  border-collapse: collapse; width: 100%; font-size: 0.83rem;
  background: {PANEL}; border: 1px solid {RULE}; border-radius: 10px;
  overflow: hidden;
}}
.srt-table th, .srt-table td {{
  text-align: left; padding: 0.45rem 0.7rem;
  border-bottom: 1px solid {RULE}; vertical-align: top;
}}
.srt-table tr:last-child td {{ border-bottom: none; }}
.srt-table th {{
  background: {PANEL_ALT}; font-weight: 600; color: {DIM};
  font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
}}
.srt-table td.num {{
  font-family: 'JetBrains Mono', ui-monospace, monospace; color: {MINT};
}}
.srt-table td code {{
  background: {BG}; padding: 1px 6px; border-radius: 3px;
  font-family: 'JetBrains Mono', ui-monospace, monospace; color: {CYAN};
  font-size: 0.8rem;
}}
.srt-table td.verb {{ color: {INK}; }}
.srt-trace-title {{
  font-size: 0.74rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: {LAVENDER}; font-weight: 600; margin-bottom: 0.5rem;
}}
.srt-legend .bif-key {{
  display: inline-block; color: {MINT}; font-weight: 700;
}}
.srt-legend .flip-key {{
  display: inline-block; color: {PINK}; font-weight: 700;
}}
/* Component A: bifurcating regime (r=0) and regime-flip markers.
   Both render as small unicode glyphs above the token without consuming
   line height, so the prose still reads naturally. */
.srt-response .tok.reg-bif {{
  box-shadow: inset 0 -1.5px 0 0 {MINT};
}}
.srt-response .tok.reg-bif::before {{
  content: "▲"; position: absolute; top: -0.85em; left: 50%;
  transform: translateX(-50%); font-size: 0.55em; color: {MINT};
  pointer-events: none; opacity: 0.9;
}}
.srt-response .tok.reg-flip {{
  outline: 1px dashed {PINK}; outline-offset: 1px;
}}
.srt-response .tok.reg-flip::before {{
  content: "⇋"; position: absolute; top: -0.85em; left: 50%;
  transform: translateX(-50%); font-size: 0.6em; color: {PINK};
  pointer-events: none; opacity: 0.95;
}}
/* When a token is both bif and flip, the flip glyph wins; use a combined
   visual cue via a top-bar. */
.srt-response .tok.reg-bif.reg-flip {{
  box-shadow: inset 0 -1.5px 0 0 {MINT};
  outline: 1px dashed {PINK}; outline-offset: 1px;
}}
.srt-response .tok.srt-pinned {{
  outline: 2px solid {CYAN} !important; outline-offset: 1px;
  box-shadow: 0 0 12px rgba(126,224,255,0.55);
}}
/* Component D: pinboard side panel. */
.srt-pinboard {{
  margin: 0.8rem 0 0; padding: 0.55rem 0.8rem 0.6rem;
  background: {PANEL}; border: 1px solid {RULE}; border-radius: 8px;
}}
.srt-pin-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.6rem; margin-bottom: 0.35rem;
}}
.srt-pin-clear {{
  background: transparent; color: {DIM}; border: 1px solid {RULE};
  border-radius: 4px; padding: 0.1rem 0.55rem; font-size: 0.7rem;
  cursor: pointer; font-family: 'JetBrains Mono', ui-monospace, monospace;
}}
.srt-pin-clear:hover {{ color: {CYAN}; border-color: {CYAN}; }}
.srt-pin-list {{
  list-style: decimal inside; margin: 0; padding: 0;
  font-size: 0.78rem; color: {INK};
  font-family: 'JetBrains Mono', ui-monospace, monospace; line-height: 1.55;
}}
.srt-pin-list li {{ padding: 0.15rem 0; }}
.srt-pin-empty {{ font-size: 0.74rem; color: {DIM}; }}
.srt-pinboard.has-pins .srt-pin-empty {{ display: none; }}
</style>
"""


# JS for click-to-pin. Lives outside _TRACE_CSS because Gradio strips
# <script> tags from gr.HTML *updates* (sanitization on innerHTML
# replacement) and even when a <script> survives in an initial render,
# scripts inserted via innerHTML do not execute per the HTML5 spec.
# We wire this once at page load via gr.Blocks(js=...) so the symbols
# are defined on `window` before any onclick fires.
_PIN_JS = r"""
() => {
  if (window.__srtPinInstalled) return;
  window.__srtPinInstalled = true;
  window.srtPin = function(el, text) {
    var trace = el.closest('.srt-trace');
    if (!trace) return;
    var board = trace.querySelector('.srt-pinboard');
    if (!board) return;
    var list = board.querySelector('.srt-pin-list');
    if (el.classList.contains('srt-pinned')) {
      el.classList.remove('srt-pinned');
      var key = el.getAttribute('data-pin-key');
      if (key) {
        var existing = list.querySelector('li[data-pin-key="' + key + '"]');
        if (existing) existing.remove();
        el.removeAttribute('data-pin-key');
      }
    } else {
      el.classList.add('srt-pinned');
      var key = 'p' + Math.random().toString(36).slice(2, 9);
      el.setAttribute('data-pin-key', key);
      var li = document.createElement('li');
      li.setAttribute('data-pin-key', key);
      li.textContent = text;
      list.appendChild(li);
    }
    if (list.children.length > 0) board.classList.add('has-pins');
    else board.classList.remove('has-pins');
  };
  window.srtPinClear = function() {
    document.querySelectorAll('.srt-trace .tok.srt-pinned').forEach(function(el) {
      el.classList.remove('srt-pinned');
      el.removeAttribute('data-pin-key');
    });
    document.querySelectorAll('.srt-pinboard').forEach(function(b) {
      var list = b.querySelector('.srt-pin-list');
      if (list) list.innerHTML = '';
      b.classList.remove('has-pins');
    });
  };
}
"""


# ---------- callbacks ----------

MAX_PROMPT_CHARS = 1500

# Qwen-2.5-7B is a *base* completion model, not Instruct. To give visitors
# a chat-style UX without retraining or swapping the backbone (which would
# invalidate the adapter's calibration on base activations), we wrap the
# user's message in a minimal User/Assistant scaffold that the base model
# completes in-context. The trace shows the user's original message; the
# scaffold is implementation detail.
_CHAT_PREFIX = "User: "
_CHAT_SUFFIX = "\nAssistant:"


def _wrap_chat(user_text: str) -> str:
    return f"{_CHAT_PREFIX}{user_text.strip()}{_CHAT_SUFFIX}"

# Pre-rendered HTML for the "before-first-generate" placeholder. We ship a
# cached trace so visitors land on a populated demo instead of an empty
# panel + 60-90s wait on a cold ZeroGPU slice. The cache is produced by
# `scripts/cache_demo_traces.py` and committed to the repo / Space.
_CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cached_traces"


def _initial_trace_html() -> str:
    candidates = [_CACHE_DIR / "default.html"]
    for p in candidates:
        try:
            if p.exists():
                body = p.read_text(encoding="utf-8")
                return (
                    '<div style="color:#8a9bb8;font-size:0.78rem;padding:0 0 .6rem">'
                    'Cached trace from a previous run \u2014 click '
                    '<b>Generate trace</b> for a fresh one. '
                    'First request on a cold ZeroGPU slice loads ~17&nbsp;GB '
                    'of weights and takes 60\u201390 s; subsequent requests '
                    'are ~7\u201310 s. Prompts and outputs are not logged.'
                    '</div>'
                ) + body
        except Exception as e:  # pragma: no cover
            log.warning("cached trace load failed for %s: %s", p, e)
    return (
        '<div style="color:#8a9bb8;padding:1rem">'
        'Click <b>Generate trace</b> to start. First request on a fresh '
        'ZeroGPU slice loads ~17&nbsp;GB of weights and may take '
        '60&ndash;90 s; subsequent requests are ~7&ndash;10 s. '
        'Prompts and outputs are not logged.'
        '</div>'
    )


@_gpu(duration=300)
def cb_generate(prompt: str, mode: str, max_new: int, budget: int, k: int,
                temperature: float, top_p: float, repetition_penalty: float):
    if not prompt.strip():
        return '<div style="color:#8a9bb8;padding:1rem">(enter a prompt above)</div>'
    # Server-side bounds: prompts are O(N²) since the adapter has no KV cache,
    # and we don't want a single user to pin the GPU for minutes.
    user_prompt = prompt[:MAX_PROMPT_CHARS]
    chat_mode = (mode or "").lower().startswith("chat")
    model_prompt = _wrap_chat(user_prompt) if chat_mode else user_prompt
    display_prompt = user_prompt
    max_new = max(8, min(int(max_new), 512))
    budget = max(1, min(int(budget), 20))
    k = max(1, min(int(k), 8))
    t = _get_trace()
    layer_indices = list(getattr(t.adapter.config, "mah_layer_indices", []) or [])

    chat_note = (
        " · chat shim: User:/Assistant: wrapper applied" if chat_mode else ""
    )
    t0 = time.perf_counter()
    result = t.generate(
        model_prompt,
        max_new_tokens=int(max_new),
        budget=int(budget),
        k=int(k),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )
    elapsed = time.perf_counter() - t0
    return _render_trace_html(
        result, display_prompt, elapsed, layer_indices=layer_indices,
        title=(None if not chat_mode else f"chat mode{chat_note}"),
    )


# ---------- UI ----------

INTRO_MD = """\
# SRT&nbsp;·&nbsp;introspect — adaptive-density reasoning trace

**SRT** ([paper](https://github.com/space-bacon/SRT/blob/main/paper.md)) is a ~12 M-parameter adapter that bolts onto a frozen LLM and watches its own residual stream. The backbone here is **Qwen-2.5-7B**, fully frozen, with two sidecars attached:

- **SRT-Adapter** (Stage 3) — three metapragmatic-attention heads tap layers 7/14/21 and feed a GRU. For every token it emits a **divergence** scalar (how hard the heads are working), a **reflexivity** estimate r̂, and a **regime** label.
- **Activation Verbalizer** (Stage 4) — a small decoder trained to translate a single layer-20 hidden state into one English sentence describing what the model is processing.

What this app does:
1. Generates a continuation token-by-token through the frozen backbone, with the SRT adapter scoring each step.
2. A **density scheduler** picks `budget` positions whose divergence values carry equal mass — so verbalizations cluster where the meta-state is *actually* moving instead of being evenly spaced.
3. At each pick, the AV is run K times for a consensus narration.

In the trace below, token background colour encodes divergence (<span style="color:#7ee0ff">**cyan**</span> = coasting, <span style="color:#7eebc0">**mint**</span> = mid, <span style="color:#ff7eb9">**pink**</span> = forking). Boxed tokens have a verbalization — **hover them** to read it.
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="pink",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    ).set(
        body_background_fill=BG,
        body_text_color=INK,
        background_fill_primary=PANEL,
        background_fill_secondary=PANEL_ALT,
        border_color_primary=RULE,
        block_background_fill=PANEL,
        block_border_color=RULE,
        block_label_text_color=DIM,
        block_title_text_color=INK,
        input_background_fill=PANEL_ALT,
        input_border_color=RULE,
        button_primary_background_fill=PINK,
        button_primary_background_fill_hover=LAVENDER,
        button_primary_text_color=BG,
    )

    with gr.Blocks(theme=theme, title="SRT · introspect", js=_PIN_JS, css=f"""
        body, .gradio-container {{ background: {BG} !important; }}
        .gradio-container {{ max-width: 1080px !important; margin: 0 auto; }}
        h1, h2, h3 {{ color: {INK}; }}
        a {{ color: {CYAN}; }}
    """) as app:
        gr.HTML(_TRACE_CSS)
        gr.Markdown(INTRO_MD)

        with gr.Row():
            with gr.Column(scale=3):
                mode = gr.Radio(
                    choices=["Completion", "Chat"],
                    value="Completion",
                    label="Input mode",
                    info=(
                        "Completion: feed the prompt raw (good for code, "
                        "narrative, or mid-sentence continuations). "
                        "Chat: type a question or instruction naturally; "
                        "we wrap it as User:/Assistant: for the base model."
                    ),
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    value=(
                        "def quicksort(arr):\n"
                        "    if len(arr) <= 1:\n"
                        "        return arr\n"
                        "    pivot = arr[len(arr) // 2]\n"
                    ),
                    lines=6,
                )
            with gr.Column(scale=2):
                max_new = gr.Slider(32, 512, value=160, step=8, label="max_new_tokens (capped at 512 on the public demo)")
                budget = gr.Slider(2, 20, value=10, step=1, label="verbalization budget (adaptive slots)")
                k = gr.Slider(1, 8, value=6, step=1, label="AV samples per slot (K)")
                with gr.Accordion("Sampling", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                    top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="top_p")
                    rep_pen = gr.Slider(1.0, 1.5, value=1.15, step=0.01, label="repetition_penalty")
                with gr.Row():
                    go = gr.Button("Generate trace", variant="primary")
                    stop = gr.Button("Stop", variant="secondary")

        gr.Markdown("### Trace")
        out = gr.HTML(_initial_trace_html())

        gr.Examples(
            examples=[
                # ---- Chat mode (natural-language) ----
                ["Explain why warm water sometimes freezes faster than cold water.", "Chat"],
                ["What is the capital of Australia, and why isn't it Sydney?", "Chat"],
                ["A patient has fever, joint pain, and a rash. What should I consider?", "Chat"],
                ["Write the first paragraph of a short story about a lighthouse keeper.", "Chat"],
                ["Is consciousness computable? Argue both sides briefly.", "Chat"],
                # ---- Completion mode (raw continuation) ----
                ["def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n", "Completion"],
                ['<title>The Bell Tower</title>\n<chapter id="1">', "Completion"],
                ["The capital of Australia is", "Completion"],
                ["For the first half of the essay she defended free trade, "
                 "but in the second half she", "Completion"],
            ],
            inputs=[prompt, mode],
            label="Try one",
        )

        gen_event = go.click(
            cb_generate,
            inputs=[prompt, mode, max_new, budget, k, temperature, top_p, rep_pen],
            outputs=[out],
        )
        stop.click(fn=None, cancels=[gen_event])

        # Default code prompt only makes sense in Completion mode. When the
        # user switches to Chat, blank the textbox so they don't try to chat
        # with `def quicksort(...)`; when they switch back to Completion,
        # restore the code seed so the textbox isn't stranded empty.
        _COMPLETION_DEFAULT = (
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
        )

        def _on_mode_change(new_mode: str, current: str):
            if (new_mode or "").lower().startswith("chat"):
                # Only clear if the user hasn't edited the default code seed.
                if current.strip() == _COMPLETION_DEFAULT.strip():
                    return gr.update(value="", placeholder="Ask anything…")
                return gr.update(placeholder="Ask anything…")
            # Switched to Completion: restore code seed only if textbox is empty.
            if not current.strip():
                return gr.update(value=_COMPLETION_DEFAULT, placeholder=None)
            return gr.update(placeholder=None)

        mode.change(_on_mode_change, inputs=[mode, prompt], outputs=[prompt])

        gr.Markdown(f"""
---

### How to read the trace

| signal | what it means | where it comes from |
|---|---|---|
| <span style="color:{PINK}">**divergence**</span> (token tint) | how far the metapragmatic-attention heads are pulling the meta-state at this step — peaks correspond to entity transitions, hedging, or topic shifts | sum of L2 norms of three MAH delta vectors |
| <span style="color:{LAVENDER}">**boxed tokens**</span> | scheduler-picked positions where the AV ran — placed at equal-mass quantiles of the divergence curve | `srt_introspect.scheduler.quantile_by_density` |
| **r̂** (in hover-card) | bifurcation network's reflexivity estimate (0=automatic, 1=actively reflecting) | BEN head on the GRU meta-state |
| **reg** (in hover-card) | discrete regime label, 0 or 1 — often stuck at 1, take with salt | BEN regime classifier |
| **verbalization** | AV decoder's best-guess English summary of the layer-20 hidden state at that position; the same hidden state the model would have continued from | `RiverRider/srt-nla-av-v1` decoder |

The AV is a paraphraser, not a mind-reader. Given one of the model's hidden states, its single best guess matches a known-good description about a quarter of the way from "random text" to "perfect paraphrase." Let it propose 64 candidates and keep the closest one, and it gets ~90% of the way there. This demo samples a handful and picks by consensus, so quality sits in between. Read each verbalization as **roughly what neighborhood of meaning the model is in at that step** — not a transcript of its thoughts.
""")

    return app


if __name__ == "__main__":
    app = build_app()
    # Pre-warm weights when we have a persistent GPU. On ZeroGPU we can't
    # touch CUDA outside an @spaces.GPU function, so skip the warmup there
    # — the first user request takes the load hit instead.
    if not _ON_ZEROGPU and DEVICE == "cuda":
        try:
            _get_trace()
        except Exception as e:  # pragma: no cover
            log.warning("warmup skipped: %s", e)
    app.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
