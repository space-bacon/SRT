"""Render a trace JSON (from `trace_demo.py --out-json ...`) as a
self-contained HTML page in the SRT pastel-neon-on-navy palette.

    python scripts/demos/render_trace_html.py artifacts/trace.json \\
        --out artifacts/trace.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


# ---- SRT palette (matches artifacts/explainers/*.png) ----------------------
BG = "#0a1429"            # deep navy page background
PANEL = "#16213d"         # card / panel
PANEL_ALT = "#1e2d4f"     # lifted panel
INK = "#e6ecf5"           # primary text
DIM = "#8a9bb8"           # secondary text
RULE = "#26345a"          # hairlines
CYAN = "#7ee0ff"          # MAH / read hook
MINT = "#7eebc0"          # FiLM / correct
PINK = "#ff7eb9"          # RRM / divergence high
ORANGE = "#ffb380"        # BEN / regime
LAVENDER = "#b8a4ff"      # community / selected verbalization


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _color_for(div: float, lo: float, hi: float) -> str:
    """Map divergence → cyan (quiet) → mint (mid) → pink (fork-y)."""
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (div - lo) / (hi - lo)))
    cyan = _hex_to_rgb(CYAN)
    mint = _hex_to_rgb(MINT)
    pink = _hex_to_rgb(PINK)
    if t < 0.5:
        r, g, b = _lerp(cyan, mint, t * 2)
    else:
        r, g, b = _lerp(mint, pink, (t - 0.5) * 2)
    # token highlight is a translucent fill over the dark panel
    return f"rgba({r},{g},{b},0.30)"


def _color_for_swatch(div: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (div - lo) / (hi - lo)))
    cyan = _hex_to_rgb(CYAN)
    mint = _hex_to_rgb(MINT)
    pink = _hex_to_rgb(PINK)
    if t < 0.5:
        r, g, b = _lerp(cyan, mint, t * 2)
    else:
        r, g, b = _lerp(mint, pink, (t - 0.5) * 2)
    return f"rgb({r},{g},{b})"


def render(data: dict) -> str:
    steps = data.get("steps", [])
    prompt = data.get("prompt", "")
    elapsed = data.get("elapsed_s", None)

    divs = [s["divergence"] for s in steps]
    if divs:
        # rescale to p10/p90 so a single peak doesn't wash out the rest
        ds = sorted(divs)
        lo = ds[max(0, int(0.10 * len(ds)))]
        hi = ds[min(len(ds) - 1, int(0.90 * len(ds)))]
        if hi <= lo:
            lo, hi = min(divs), max(divs)
    else:
        lo = hi = 0.0

    token_html_parts: list[str] = []
    for s in steps:
        tok = s["token"]
        display = (
            html.escape(tok)
            .replace("\n", "<br>")
            .replace(" ", "&nbsp;")
        )
        bg = _color_for(s["divergence"], lo, hi)
        verb = s.get("verbalization")
        klass = "tok"
        if verb:
            klass += " selected"
        title_parts = [
            f"i={s['i']}",
            f"d={s['divergence']:.2f}",
            f"r̂={s['r_hat']:.2f}",
            f"reg={s['regime']}",
        ]
        if verb:
            title_parts.append(f"→ {verb[:240]}")
        title = "  ·  ".join(title_parts)
        token_html_parts.append(
            f'<span class="{klass}" style="background:{bg}" data-title="{html.escape(title)}">{display}</span>'
        )

    selected = [s for s in steps if s.get("verbalization")]
    sel_rows = "\n".join(
        f'<tr><td class="num">{s["i"]}</td><td><code>{html.escape(s["token"])}</code></td>'
        f'<td class="num">{s["divergence"]:.2f}</td><td class="num">{s["r_hat"]:.2f}</td>'
        f'<td class="num">{s["regime"]}</td><td class="verb">{html.escape(s.get("verbalization") or "")}</td></tr>'
        for s in selected
    )

    n_tok = len(steps)
    n_sel = len(selected)
    elapsed_s = f"{elapsed:.1f}s" if elapsed is not None else "—"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SRT introspect · adaptive-density reasoning trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: {BG}; --panel: {PANEL}; --panel-alt: {PANEL_ALT};
  --ink: {INK}; --dim: {DIM}; --rule: {RULE};
  --cyan: {CYAN}; --mint: {MINT}; --pink: {PINK};
  --orange: {ORANGE}; --lavender: {LAVENDER};
}}
* {{ box-sizing: border-box; }}
html, body {{ background: var(--bg); color: var(--ink); margin: 0; padding: 0; }}
body {{
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  font-size: 15px; line-height: 1.55;
  background:
    radial-gradient(1200px 600px at 70% -10%, rgba(126,224,255,0.08), transparent 60%),
    radial-gradient(900px 500px at -10% 30%, rgba(255,126,185,0.07), transparent 60%),
    var(--bg);
  min-height: 100vh;
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }}

.eyebrow {{
  font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--cyan); font-weight: 600; margin-bottom: 0.5rem;
}}
h1 {{
  font-size: 1.65rem; margin: 0 0 0.4rem; font-weight: 700; letter-spacing: -0.01em;
}}
h1 .accent {{ color: var(--pink); }}
.sub {{ color: var(--dim); margin: 0 0 1.5rem; font-size: 0.95rem; }}

.explainer {{
  background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1rem 1.2rem; margin-bottom: 1.4rem; color: var(--ink);
  font-size: 0.92rem;
}}
.explainer p {{ margin: 0 0 0.55rem; }}
.explainer p:last-child {{ margin-bottom: 0; }}
.explainer .k {{ color: var(--cyan); font-weight: 600; }}
.explainer .k.pink {{ color: var(--pink); }}
.explainer .k.mint {{ color: var(--mint); }}
.explainer .k.lav {{ color: var(--lavender); }}

.meta-row {{
  display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; margin-bottom: 1rem;
  font-size: 0.82rem; color: var(--dim);
}}
.chip {{
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: var(--panel); border: 1px solid var(--rule);
  padding: 0.22rem 0.6rem; border-radius: 999px;
  font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.78rem;
  color: var(--ink);
}}
.chip .label {{ color: var(--dim); }}

.legend {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1rem;
  font-size: 0.78rem; color: var(--dim); margin-bottom: 1rem;
}}
.legend .grad {{
  display: inline-block; width: 200px; height: 0.55rem; border-radius: 999px;
  background: linear-gradient(90deg, {CYAN}, {MINT}, {PINK});
  vertical-align: middle; box-shadow: 0 0 12px rgba(126,235,192,0.25);
}}
.legend .box {{
  display: inline-block; padding: 0.05rem 0.5rem; border-radius: 4px;
  outline: 1.5px solid var(--lavender); outline-offset: 1px; color: var(--ink);
}}

.prompt {{
  background: var(--panel); border-left: 3px solid var(--cyan);
  padding: 0.75rem 1rem; border-radius: 6px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  white-space: pre-wrap; font-size: 0.84rem; color: var(--ink);
  margin-bottom: 1rem;
}}
.section-label {{
  font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--dim); margin: 1.4rem 0 0.5rem; font-weight: 600;
}}

.response {{
  background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1.4rem 1.5rem; line-height: 2.05; font-size: 1.02rem;
  color: var(--ink);
}}
.tok {{
  display: inline; padding: 1px 1px; border-radius: 3px; position: relative;
  transition: background 80ms ease;
}}
.tok.selected {{
  outline: 1.5px solid var(--lavender); outline-offset: 1px;
  cursor: help; box-shadow: 0 0 10px rgba(184,164,255,0.35);
}}
.tok:hover {{ background: rgba(184,164,255,0.35) !important; }}
.tok:hover::after {{
  content: attr(data-title);
  position: absolute; left: 50%; transform: translateX(-50%);
  top: 1.7em; z-index: 20;
  background: {PANEL_ALT}; color: var(--ink);
  padding: 0.55rem 0.75rem; border-radius: 6px;
  border: 1px solid var(--lavender);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.74rem; line-height: 1.45;
  white-space: pre-wrap; max-width: 460px; min-width: 220px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.45);
  pointer-events: none;
}}

table {{
  border-collapse: collapse; width: 100%; margin-top: 0.6rem; font-size: 0.85rem;
  background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  overflow: hidden;
}}
th, td {{
  text-align: left; padding: 0.5rem 0.75rem;
  border-bottom: 1px solid var(--rule); vertical-align: top;
}}
tr:last-child td {{ border-bottom: none; }}
th {{
  background: var(--panel-alt); font-weight: 600; color: var(--dim);
  font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
}}
td.num {{ font-family: 'JetBrains Mono', ui-monospace, monospace; color: var(--mint); }}
td code {{
  background: var(--bg); padding: 1px 6px; border-radius: 3px;
  font-family: 'JetBrains Mono', ui-monospace, monospace; color: var(--cyan);
  font-size: 0.82rem;
}}
td.verb {{ color: var(--ink); }}

footer {{
  margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--rule);
  font-size: 0.75rem; color: var(--dim); text-align: center;
}}
footer a {{ color: var(--cyan); text-decoration: none; }}
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow">SRT · INTROSPECT</div>
  <h1>Adaptive-density <span class="accent">reasoning trace</span></h1>
  <p class="sub">One forward pass through a frozen LLM &mdash; the SRT adapter watches its own internal state and the activation verbalizer narrates where it bends.</p>

  <div class="explainer">
    <p>Every generated token gets three numbers from the adapter: <span class="k pink">divergence</span> (how hard the metapragmatic heads are working), <span class="k">r̂</span> (the bifurcation network's reflexivity estimate), and a <span class="k">regime</span> label. Token background colour encodes divergence on a <span style="color:{CYAN}">cyan</span> &rarr; <span style="color:{MINT}">mint</span> &rarr; <span style="color:{PINK}">pink</span> scale: cyan = the model is coasting, pink = its meta-state is forking.</p>
    <p>A scheduler then picks <strong>{n_sel}</strong> positions out of {n_tok} so the picks track those divergence peaks (equal-mass quantiles, not equal spacing), and the activation verbalizer decodes the layer-20 hidden state at each pick into English. Hover any <span class="box">boxed token</span> to read what the model was &ldquo;thinking about&rdquo; right there.</p>
  </div>

  <div class="meta-row">
    <span class="chip"><span class="label">tokens</span>{n_tok}</span>
    <span class="chip"><span class="label">verbalizations</span>{n_sel}</span>
    <span class="chip"><span class="label">div range</span>{lo:.2f} → {hi:.2f}</span>
    <span class="chip"><span class="label">elapsed</span>{elapsed_s}</span>
  </div>

  <div class="legend">
    <span>divergence</span>
    <span class="grad" aria-hidden="true"></span>
    <span style="color:{CYAN}">low</span>
    <span>→</span>
    <span style="color:{PINK}">high</span>
    <span style="opacity:.5">·</span>
    <span class="box">box</span>
    <span>= selected for narration (hover)</span>
  </div>

  <div class="section-label">Prompt</div>
  <div class="prompt">{html.escape(prompt)}</div>

  <div class="section-label">Continuation · per-token trace</div>
  <div class="response">{"".join(token_html_parts)}</div>

  <div class="section-label">Selected verbalizations</div>
  <table>
    <thead><tr><th>idx</th><th>token</th><th>div</th><th>r̂</th><th>reg</th><th>verbalization (AV)</th></tr></thead>
    <tbody>{sel_rows}</tbody>
  </table>

  <footer>
    SRT &middot; semiotic-reflexive transformer &middot;
    <a href="https://github.com/space-bacon/SRT">github.com/space-bacon/SRT</a>
  </footer>
</div>
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace_json", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    with open(args.trace_json) as f:
        data = json.load(f)
    out_path = args.out or args.trace_json.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(data))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
