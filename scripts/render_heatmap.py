#!/usr/bin/env python3
"""Render per-token r̂ trace JSONL into a self-contained HTML heatmap.

Reads traces.jsonl produced by scripts/instrument_eval.py and emits a
single-file HTML with per-token color = r̂ (subcritical = blue,
supercritical = red), hover tooltip showing r̂ / r_true / regime /
community.

Usage:
    python scripts/render_heatmap.py \\
        --traces artifacts/instrument/v3/traces.jsonl \\
        --out artifacts/instrument/v3/heatmap.html \\
        --title "SRT-Adapter v3 — per-token r̂"
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Color stops (values clipped to [-1.5, 3.0] then mapped to a blue→white→red ramp)
R_MIN = -1.5
R_MAX = 3.0


def color_for(r: float) -> str:
    """Map r̂ to an RGB color via blue → white → red."""
    r = max(R_MIN, min(R_MAX, r))
    if r < 0:
        # blue → white   (R_MIN..0 → t=0..1)
        t = (r - R_MIN) / (0 - R_MIN)
        rr, gg, bb = int(255 * t + 60 * (1 - t)), int(255 * t + 100 * (1 - t)), int(255)
    else:
        # white → red    (0..R_MAX → t=0..1)
        t = r / R_MAX
        rr, gg, bb = int(255), int(255 * (1 - t) + 60 * t), int(255 * (1 - t) + 60 * t)
    return f"rgb({rr},{gg},{bb})"


REGIME_NAME = {0: "subcritical", 1: "supercritical"}


def render_passage(idx: int, trace: dict) -> str:
    tokens = trace["tokens"]
    r_hat = trace["r_hat"]
    r_true = trace["r_true"]
    r_mask = trace["r_mask"]
    regime = trace["regime_pred"]
    cid_true = trace.get("community_id_true")
    cid_pred = trace.get("community_pred")

    spans: list[str] = []
    for tok, rh, rt, rm, reg in zip(tokens, r_hat, r_true, r_mask, regime):
        # Sanitize token for display (Qwen / SP tokens have ▁ etc.)
        display = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "↵\n")
        display_html = html.escape(display)
        bg = color_for(rh)
        title = (
            f"r̂ = {rh:+.3f}\\n"
            f"r_true = {rt:+.3f}{' (no label)' if not rm else ''}\\n"
            f"regime = {REGIME_NAME.get(reg, '?')}"
        )
        spans.append(
            f'<span class="tok" style="background:{bg}" '
            f'title="{title}">{display_html}</span>'
        )

    header = (
        f'<div class="meta">passage #{idx} · '
        f'community id (true) = {cid_true} · '
        f'community id (predicted prototype) = {cid_pred}</div>'
    )
    return f'<div class="passage">{header}<div class="text">{"".join(spans)}</div></div>'


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       max-width: 980px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
       color: #111; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; color: #555; font-weight: 500; }
.legend { display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0 1.5rem; }
.legend .bar { display: inline-block; width: 320px; height: 14px;
               background: linear-gradient(to right,
                   rgb(60,100,255), rgb(255,255,255), rgb(255,60,60)); border: 1px solid #ccc; }
.legend .label { font-size: 0.85rem; color: #555; }
.passage { margin-bottom: 1.4rem; padding: 0.8rem; background: #fafafa;
           border: 1px solid #eee; border-radius: 6px; }
.passage .meta { font-size: 0.78rem; color: #888; margin-bottom: 0.6rem;
                 font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.passage .text { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                 font-size: 0.92rem; white-space: pre-wrap; word-break: break-word; }
.tok { padding: 1px 2px; border-radius: 2px; cursor: help; }
.intro { color: #444; font-size: 0.95rem; margin-bottom: 1rem; }
"""


def render_html(traces: list[dict], title: str) -> str:
    intro = (
        "Per-token bifurcation parameter r̂ from a frozen Qwen 2.5-7B + "
        "SRT-Adapter. Blue = subcritical (stable shared meaning), "
        "white = near-critical, red = supercritical (meaning has bifurcated "
        "across communities). Hover any token for full diagnostics."
    )
    legend = (
        '<div class="legend">'
        f'<span class="label">r̂ = {R_MIN:+.1f}</span>'
        '<span class="bar"></span>'
        f'<span class="label">r̂ = {R_MAX:+.1f}</span>'
        '</div>'
    )
    body_passages = "\n".join(
        render_passage(i, t) for i, t in enumerate(traces)
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class="intro">{intro}</div>
{legend}
<h2>Passages</h2>
{body_passages}
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="SRT-Adapter — per-token r̂")
    p.add_argument("--limit", type=int, default=None,
                   help="Render only the first N passages")
    args = p.parse_args()

    traces: list[dict] = []
    with args.traces.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))
    if args.limit:
        traces = traces[:args.limit]

    html_str = render_html(traces, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    print(f"Wrote {args.out}  ({len(traces)} passages, {args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
