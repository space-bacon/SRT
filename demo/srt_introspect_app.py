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


def _render_trace_html(result, prompt: str, elapsed: float) -> str:
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

    spans = []
    for s in steps:
        disp = _html.escape(s.token).replace("\n", "<br>").replace(" ", "&nbsp;")
        bg = _div_color(s.divergence, lo, hi)
        klass = "tok selected" if s.verbalization else "tok"
        title = f"i={s.token_idx}  ·  d={s.divergence:.2f}  ·  r̂={s.r_hat:.2f}  ·  reg={s.regime}"
        if s.verbalization:
            title += f"  →  {s.verbalization[:240]}"
        spans.append(
            f'<span class="{klass}" style="background:{bg}" '
            f'data-title="{_html.escape(title)}">{disp}</span>'
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

    return f"""
<div class="srt-trace">
  <div class="srt-meta">
    <span class="chip"><span class="lbl">tokens</span>{len(steps)}</span>
    <span class="chip"><span class="lbl">verbalizations</span>{len(sel)}</span>
    <span class="chip"><span class="lbl">div range</span>{lo:.2f} → {hi:.2f}</span>
    <span class="chip"><span class="lbl">elapsed</span>{elapsed:.1f}s</span>
  </div>
  <div class="srt-legend">
    <span>divergence</span>
    <span class="grad"></span>
    <span style="color:{CYAN}">low</span><span>→</span><span style="color:{PINK}">high</span>
    <span style="opacity:.5">·</span>
    <span class="box">box</span><span>= verbalization (hover)</span>
  </div>
  <div class="srt-prompt">{_html.escape(prompt)}</div>
  <div class="srt-response">{''.join(spans)}</div>
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
</style>
"""


# ---------- callbacks ----------

@_gpu(duration=180)
def cb_generate(prompt: str, max_new: int, budget: int, k: int,
                temperature: float, top_p: float, repetition_penalty: float):
    if not prompt.strip():
        return '<div style="color:#8a9bb8;padding:1rem">(enter a prompt above)</div>'
    t = _get_trace()
    t0 = time.perf_counter()
    result = t.generate(
        prompt,
        max_new_tokens=int(max_new),
        budget=int(budget),
        k=int(k),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )
    elapsed = time.perf_counter() - t0
    return _render_trace_html(result, prompt, elapsed)


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

    with gr.Blocks(theme=theme, title="SRT · introspect", css=f"""
        body, .gradio-container {{ background: {BG} !important; }}
        .gradio-container {{ max-width: 1080px !important; margin: 0 auto; }}
        h1, h2, h3 {{ color: {INK}; }}
        a {{ color: {CYAN}; }}
    """) as app:
        gr.HTML(_TRACE_CSS)
        gr.Markdown(INTRO_MD)

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=(
                        "Q: A retired plumber from Ohio claims he invented a "
                        "self-cooling beer can in 1973. Is this likely true, "
                        "and what would have to be different about "
                        "thermodynamics for it to work?\nA:"
                    ),
                    lines=4,
                )
            with gr.Column(scale=2):
                max_new = gr.Slider(32, 320, value=160, step=8, label="max_new_tokens")
                budget = gr.Slider(2, 24, value=10, step=1, label="verbalization budget (adaptive slots)")
                k = gr.Slider(1, 12, value=6, step=1, label="AV samples per slot (K)")
                with gr.Accordion("Sampling", open=False):
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                    top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="top_p")
                    rep_pen = gr.Slider(1.0, 1.5, value=1.15, step=0.01, label="repetition_penalty")
                go = gr.Button("Generate trace", variant="primary")

        gr.Markdown("### Trace")
        out = gr.HTML('<div style="color:#8a9bb8;padding:1rem">Click <b>Generate trace</b> to start. First run loads ~17&nbsp;GB of weights — give it ~15 s.</div>')

        gr.Examples(
            examples=[
                ["Q: What killed the dinosaurs?\nA:"],
                ["Q: Is the speed of light constant in all reference frames?\nA:"],
                ["Q: Why does pasta water boil over so easily?\nA:"],
                ["Q: Briefly summarize the plot of Hamlet.\nA:"],
                ["Q: What is gradient descent, in two sentences?\nA:"],
                ["Q: Translate to French: 'The trains are on strike again.'\nA:"],
            ],
            inputs=[prompt],
            label="Try one",
        )

        go.click(
            cb_generate,
            inputs=[prompt, max_new, budget, k, temperature, top_p, rep_pen],
            outputs=[out],
        )

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

The AV is honest about its limits: on TruthfulQA-style probes it reaches ρ ≈ 0.26 vs. ground-truth descriptions — *real signal*, but think of each verbalization as **what part of conceptual space the model is in**, not a literal inner monologue.
""")

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
