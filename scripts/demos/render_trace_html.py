"""Render a trace JSON (from `trace_demo.py --out-json ...`) as a
self-contained HTML page.

    python scripts/demos/render_trace_html.py artifacts/trace.json \\
        --out artifacts/trace.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _color_for(div: float, lo: float, hi: float) -> str:
    """Map divergence to a yellow→red gradient. lo→#fff8e1, hi→#c62828."""
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (div - lo) / (hi - lo)))
    # lerp from (255,248,225) to (198,40,40)
    r = int(255 + (198 - 255) * t)
    g = int(248 + (40 - 248) * t)
    b = int(225 + (40 - 225) * t)
    return f"rgb({r},{g},{b})"


def render(data: dict) -> str:
    steps = data.get("steps", [])
    prompt = data.get("prompt", "")
    text = data.get("text", "")
    elapsed = data.get("elapsed_s", None)

    divs = [s["divergence"] for s in steps]
    if divs:
        lo, hi = min(divs), max(divs)
    else:
        lo = hi = 0.0

    token_html_parts: list[str] = []
    for s in steps:
        tok = s["token"]
        # Preserve whitespace visibility: leading space -> nbsp, newline -> <br>
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
        # store metadata on data-* for tooltip JS
        title_parts = [
            f"i={s['i']}",
            f"d={s['divergence']:.2f}",
            f"r̂={s['r_hat']:.2f}",
            f"reg={s['regime']}",
        ]
        if verb:
            title_parts.append(f"→ {verb[:240]}")
        title = " | ".join(title_parts)
        token_html_parts.append(
            f'<span class="{klass}" style="background:{bg}" data-title="{html.escape(title)}">{display}</span>'
        )

    selected = [s for s in steps if s.get("verbalization")]
    sel_rows = "\n".join(
        f"<tr><td>{s['i']}</td><td><code>{html.escape(s['token'])}</code></td>"
        f"<td>{s['divergence']:.2f}</td><td>{s['r_hat']:.2f}</td>"
        f"<td>{s['regime']}</td><td>{html.escape(s.get('verbalization') or '')}</td></tr>"
        for s in selected
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SRT introspect trace</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; color:#222; }}
h1 {{ font-size: 1.3rem; margin: 0 0 .5rem; }}
.meta {{ color: #666; font-size: .85rem; margin-bottom: 1rem; }}
.prompt {{ background:#f6f6f6; padding: .6rem .8rem; border-left: 3px solid #999; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: .85rem; margin-bottom: 1rem; }}
.response {{ line-height: 1.9; font-size: 1.05rem; padding: 1rem; border: 1px solid #ddd; border-radius: 6px; }}
.tok {{ display: inline; padding: 0 1px; border-radius: 2px; position: relative; }}
.tok.selected {{ outline: 2px solid #1976d2; outline-offset: 1px; cursor: help; }}
.tok:hover::after {{
    content: attr(data-title);
    position: absolute; left: 0; top: 1.6em;
    background: #222; color: #fff; padding: .4rem .6rem;
    border-radius: 4px; font-size: .75rem; white-space: pre-wrap;
    max-width: 480px; z-index: 10; line-height: 1.3;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; font-size: .85rem; }}
th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eee; vertical-align: top; }}
th {{ background: #fafafa; font-weight: 600; }}
td code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
.legend {{ font-size: .8rem; color:#555; margin-top:.3rem; }}
.legend .swatch {{ display: inline-block; width: 1.6rem; height: .8rem; vertical-align: middle; border-radius: 2px; }}
</style>
</head>
<body>
<h1>SRT introspect — adaptive-density reasoning trace</h1>
<div class="meta">
  {len(steps)} tokens · {len(selected)} verbalizations · divergence range [{lo:.2f}, {hi:.2f}]
  {f"· generated in {elapsed:.1f}s" if elapsed is not None else ""}
</div>
<div class="legend">
  divergence:
  <span class="swatch" style="background:{_color_for(lo, lo, hi)}"></span> low
  <span class="swatch" style="background:{_color_for((lo+hi)/2, lo, hi)}"></span> mid
  <span class="swatch" style="background:{_color_for(hi, lo, hi)}"></span> high
  · <span style="outline:2px solid #1976d2; padding:0 4px;">box</span> = selected for verbalization (hover for narration)
</div>
<div class="prompt">{html.escape(prompt)}</div>
<div class="response">{"".join(token_html_parts)}</div>

<h2 style="font-size:1.05rem; margin-top:2rem;">Selected verbalizations</h2>
<table>
  <thead><tr><th>idx</th><th>token</th><th>div</th><th>r̂</th><th>reg</th><th>verbalization</th></tr></thead>
  <tbody>{sel_rows}</tbody>
</table>
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
