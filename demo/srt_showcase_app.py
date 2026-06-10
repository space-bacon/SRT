"""SRT Showcase — live introspection demo for the Semiotic-Reflexive Transformer.

A single Gradio app that streams generation from a frozen Qwen-2.5-7B + the SRT
adapter and shows, in real time, what the model is doing internally:

  • Live token stream, each token tinted by its predictive ENTROPY (the
    validated online uncertainty signal) — toggle to tint by SRT divergence.
  • A running entropy meter (mean / peak) as the answer builds.
  • Charts of entropy and SRT divergence across the generated tokens.
  • Expand/collapse natural-language VERBALIZATIONS of the model's hidden state
    at the highest-effort token positions (chosen by the adaptive-density
    scheduler), each round-trip validated by the Activation Verbalizer.
  • Per-token hover rollovers: entropy, divergence, reflexivity r̂, regime.
  • Regenerate, and an "adapter on/off" switch.

Honest scope: entropy is the load-bearing uncertainty signal. The SRT
side-channels (divergence, r̂, regime) and the verbalizations are shown as
*observational* readouts of internal state — a window into the model, not a
validated hallucination detector.

Run locally on a GPU box:
    pip install -r demo/requirements.txt
    PYTHONPATH=. python demo/srt_showcase_app.py

Deploys to an HF Space (ZeroGPU / a10g). Qwen-7B needs ~16 GB bf16; the AV
adds ~2 GB.
"""

from __future__ import annotations

import html
import logging
import os

import gradio as gr
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("srt_showcase")

# ── ZeroGPU-compatible GPU decorator (no-op off-Space) ───────────────────
try:  # pragma: no cover - environment dependent
    import spaces  # type: ignore

    _ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))

    def _gpu(duration: int = 300):
        if _ON_ZEROGPU:
            return spaces.GPU(duration=duration)
        return lambda fn: fn
except Exception:  # local / non-Space
    _ON_ZEROGPU = False

    def _gpu(duration: int = 300):
        def _wrap(fn):
            return fn
        return _wrap


DEVICE = "cuda" if (torch.cuda.is_available() or _ON_ZEROGPU) else "cpu"


# ── Palette ──────────────────────────────────────────────────────────────
BG = "#0a1429"
PANEL = "#16213d"
PANEL_ALT = "#1d2b4d"
INK = "#e6ecf5"
MUTED = "#8aa0c8"
CYAN = "#46e0d0"
MINT = "#7cf0a8"
PINK = "#ff7eb6"
LAVENDER = "#b69cff"
AMBER = "#ffcf66"

# Round-trip fidelity reference frame (raw fve_nrm on Qwen2.5-7B L20, from the
# anchored oracle_ceiling study). Unrelated text floors near 0.622; the
# paraphrase best-of-8 ceiling is ~0.848. We normalise the round-trip cosine
# against this band so the badge reads 0% (no better than chance) to 100%
# (matches the paraphrase ceiling) rather than against a meaningless raw 0.
RT_FLOOR = 0.622
RT_CEIL = 0.848

# Lazy global trace handle (loaded once on first generation).
_TRACE = None


def _get_trace():
    global _TRACE
    if _TRACE is None:
        from srt_introspect import Trace  # local import keeps import-time light
        logger.info("Loading SRT Trace (adapter + activation verbalizer)...")
        _TRACE = Trace.load()
        logger.info("Trace ready on device=%s", _TRACE.device)
    return _TRACE


# ── Signal → colour ──────────────────────────────────────────────────────
def _lerp(c0, c1, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def _entropy_color(ent: float, lo: float, hi: float) -> str:
    """Green (calm) → amber → red (uncertain) over [lo, hi] nats."""
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (ent - lo) / (hi - lo)))
    g = (124, 240, 168)   # mint
    a = (255, 207, 102)   # amber
    r = (255, 126, 182)   # pink/red
    rgb = _lerp(g, a, t * 2) if t < 0.5 else _lerp(a, r, (t - 0.5) * 2)
    return "rgba(%d,%d,%d,0.30)" % rgb


def _div_color(d: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (d - lo) / (hi - lo)))
    rgb = _lerp((70, 224, 208), (255, 126, 182), t)  # cyan → pink
    return "rgba(%d,%d,%d,0.30)" % rgb


# ── Renderers ──────────────────────────────────────────────────────────────
def _render_tokens(result, tint: str) -> str:
    """Per-token HTML, tinted by entropy or divergence, with hover rollovers."""
    steps = result.steps
    if not steps:
        return f"<div style='color:{MUTED}'>…</div>"
    ents = [s.entropy for s in steps]
    divs = [s.divergence for s in steps]
    e_lo, e_hi = min(ents), max(ents)
    d_lo, d_hi = min(divs), max(divs)

    spans = []
    for s in steps:
        if tint == "divergence":
            bg = _div_color(s.divergence, d_lo, d_hi)
        else:
            bg = _entropy_color(s.entropy, e_lo, e_hi)
        tok = html.escape(s.token).replace("\n", "⏎<br>")
        title = (f"#{s.token_idx}  H={s.entropy:.2f} nats  "
                 f"div={s.divergence:.2f}  r̂={s.r_hat:.2f}  "
                 f"regime={'super' if s.regime else 'sub'}")
        sel = " sel" if s.verbalization else ""
        spans.append(
            f"<span class='tok{sel}' style='background:{bg}' "
            f"data-title=\"{html.escape(title)}\">{tok}</span>"
        )
    return f"<div class='toks'>{''.join(spans)}</div>"


def _render_meter(result) -> str:
    steps = result.steps
    if not steps:
        return ""
    n = len(steps)
    ents = [s.entropy for s in steps]
    mean_e = sum(ents) / n
    max_e = max(ents)
    # Risk bar scaled to a ~3.0-nat practical ceiling.
    frac = max(0.0, min(1.0, mean_e / 3.0))
    col = MINT if frac < 0.33 else (AMBER if frac < 0.66 else PINK)

    # SRT side-channel summaries (observational).
    divs = [s.divergence for s in steps]
    mean_d, max_d = sum(divs) / n, max(divs)
    rhats = [s.r_hat for s in steps]
    mean_r = sum(rhats) / n
    super_frac = sum(1 for s in steps if s.regime) / n
    verbalized = [s for s in steps if s.roundtrip_cos is not None]

    def _bar(label, value, fmt, b_frac, color, unit=""):
        b_frac = max(0.0, min(1.0, b_frac))
        return (
            f"<div class='meter-row'><span>{label}</span>"
            f"<b style='color:{color}'>{fmt.format(value)}</b>{unit}</div>"
            f"<div class='bar'><div class='fill' "
            f"style='width:{int(b_frac * 100)}%;background:{color}'></div></div>"
        )

    parts = [
        "<div class='meter'>",
        _bar("mean entropy", mean_e, "{:.2f}", frac, col, " nats"),
        f"<div class='meter-row'><span>peak entropy</span><b>{max_e:.2f}</b> nats"
        f" &nbsp;·&nbsp; <span>{n} tokens</span></div>",
        "<div class='meter-sep'></div>",
        # SRT divergence: how fast the metapragmatic state is moving. Bar
        # scaled to the run's own peak so the mean reads as a fraction of max.
        _bar("mean SRT divergence", mean_d, "{:.2f}",
             (mean_d / max_d) if max_d else 0.0, PINK),
        # Reflexivity r̂ is already in [0, 1].
        _bar("mean reflexivity r̂", mean_r, "{:.2f}", mean_r, LAVENDER),
        # Regime mix: share of tokens the BEN flags supercritical (bifurcating).
        _bar("supercritical regime", super_frac * 100, "{:.0f}", super_frac, AMBER, "%"),
    ]

    # Verbalization fidelity: mean round-trip across the verbalized slots,
    # normalised against the paraphrase ceiling like the per-card badges.
    if verbalized:
        fves = [0.5 * (1.0 + s.roundtrip_cos) for s in verbalized]
        mean_fid = sum((f - RT_FLOOR) / (RT_CEIL - RT_FLOOR) for f in fves) / len(fves)
        mean_fid = max(0.0, min(1.0, mean_fid))
        fcol = MINT if mean_fid > 0.66 else (AMBER if mean_fid > 0.33 else PINK)
        parts.append(_bar(f"verbalization fidelity ({len(verbalized)})",
                          mean_fid * 100, "{:.0f}", mean_fid, fcol, "%"))

    # Per-layer divergence depth profile: average each MAH layer's divergence
    # across all tokens to reveal *where* in the stack the model's
    # metapragmatic state moves most. Unique to SRT.
    profile = _layer_profile(steps)
    if profile:
        parts.append("<div class='meter-sep'></div>")
        parts.append("<div class='meter-row'><span>divergence by MAH layer (depth profile)</span></div>")
        parts.append(_layer_bars(profile))

    parts.append("</div>")
    return "".join(parts)


def _layer_profile(steps) -> list[float]:
    """Mean per-MAH-layer divergence across all tokens (layer order = shallow
    → deep). Empty if no per-layer data is present."""
    rows = [s.per_layer_divergence for s in steps if s.per_layer_divergence]
    if not rows:
        return []
    width = min(len(r) for r in rows)
    if width == 0:
        return []
    return [sum(r[i] for r in rows) / len(rows) for i in range(width)]


def _layer_bars(profile: list[float]) -> str:
    """Compact vertical-bar chart of the per-layer divergence profile."""
    hi = max(profile) or 1.0
    bars = []
    for i, v in enumerate(profile):
        h = int(6 + 46 * (v / hi))
        bars.append(
            f"<div class='lbar' title='MAH layer {i}: {v:.2f}'>"
            f"<div class='lbar-fill' style='height:{h}px'></div>"
            f"<div class='lbar-idx'>{i}</div></div>"
        )
    return f"<div class='lbars'>{''.join(bars)}</div>"


def _sparkline(values, color, h=70, w=920):
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = " ".join(
        f"{w * i / (n - 1):.1f},{h - (h - 8) * (v - lo) / rng - 4:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' "
        f"preserveAspectRatio='none'>"
        f"<polyline points='{pts}' fill='none' stroke='{color}' "
        f"stroke-width='1.6'/></svg>"
    )


def _render_charts(result) -> str:
    steps = result.steps
    if len(steps) < 2:
        return ""
    ent = _sparkline([s.entropy for s in steps], CYAN)
    dv = _sparkline([s.divergence for s in steps], PINK)
    return (
        f"<div class='chart'><div class='chart-label' style='color:{CYAN}'>"
        f"predictive entropy (uncertainty)</div>{ent}</div>"
        f"<div class='chart'><div class='chart-label' style='color:{PINK}'>"
        f"SRT divergence (observational)</div>{dv}</div>"
    )


def _render_verbalizations(result) -> str:
    sel = [s for s in result.steps if s.verbalization]
    if not sel:
        return f"<div style='color:{MUTED}'>No verbalizations yet.</div>"
    cards = []
    for s in sel:
        tok = html.escape(s.token.strip() or "·")
        verb = html.escape(s.verbalization or "")
        badge = _roundtrip_badge(s.roundtrip_cos)
        cards.append(
            f"<details class='vcard'><summary>"
            f"<span class='vtok'>“{tok}”</span> "
            f"<span class='vmeta'>#{s.token_idx} · div {s.divergence:.2f} · "
            f"r̂ {s.r_hat:.2f} · {'super' if s.regime else 'sub'}</span>"
            f"{badge}"
            f"</summary><div class='vbody'>{verb}</div></details>"
        )
    return "".join(cards)


def _roundtrip_badge(cos) -> str:
    """A self-validation badge: re-encode the verbalization, measure how close
    its hidden state lands to the original. Normalised against the paraphrase
    ceiling (see RT_FLOOR / RT_CEIL)."""
    if cos is None:
        return ""
    fve = 0.5 * (1.0 + float(cos))
    frac = max(0.0, min(1.0, (fve - RT_FLOOR) / (RT_CEIL - RT_FLOOR)))
    pct = int(round(frac * 100))
    col = MINT if frac > 0.66 else (AMBER if frac > 0.33 else PINK)
    return (
        f"<span class='rt' style='border-color:{col};color:{col}' "
        f"title='Re-encoded verbalization cos={cos:.3f} vs original hidden state; "
        f"normalised against the paraphrase ceiling.'>"
        f"round-trip {pct}% · cos {cos:.2f}</span>"
    )


_CSS = f"""
<style>
.toks {{ line-height: 2.1; font-size: 15px; }}
.tok {{ position: relative; padding: 1px 2px; border-radius: 3px;
        white-space: pre-wrap; cursor: default; }}
.tok.sel {{ outline: 1px solid {LAVENDER}; }}
.tok:hover::after {{
   content: attr(data-title); position: absolute; left: 0; top: 1.9em;
   white-space: nowrap; z-index: 20; background: {PANEL_ALT};
   color: {INK}; border: 1px solid {LAVENDER}; border-radius: 6px;
   padding: 5px 9px; font-size: 11px; font-family: ui-monospace, monospace; }}
.meter {{ background: {PANEL}; border-radius: 10px; padding: 12px 14px;
          color: {INK}; }}
.meter-row {{ display: flex; gap: 8px; align-items: baseline;
              color: {MUTED}; font-size: 13px; margin: 2px 0; }}
.meter-row b {{ color: {INK}; font-size: 16px; }}
.bar {{ height: 10px; background: {BG}; border-radius: 5px; overflow: hidden;
        margin: 6px 0; }}
.fill {{ height: 100%; transition: width .3s ease; }}
.meter-sep {{ height: 1px; background: {PANEL_ALT}; margin: 10px 0 8px; }}
.lbars {{ display: flex; align-items: flex-end; gap: 3px; height: 60px;
          margin: 4px 0 2px; }}
.lbar {{ flex: 1; display: flex; flex-direction: column; align-items: center;
         justify-content: flex-end; }}
.lbar-fill {{ width: 100%; background: linear-gradient(to top, {PINK}, {LAVENDER});
              border-radius: 2px 2px 0 0; min-height: 3px; }}
.lbar-idx {{ font-size: 9px; color: {MUTED}; font-family: ui-monospace, monospace;
             margin-top: 2px; }}
.chart {{ background: {PANEL}; border-radius: 10px; padding: 8px 12px;
          margin: 8px 0; }}
.chart-label {{ font-size: 12px; font-family: ui-monospace, monospace;
                margin-bottom: 2px; }}
.vcard {{ background: {PANEL}; border: 1px solid {PANEL_ALT};
          border-radius: 8px; margin: 6px 0; padding: 4px 10px; }}
.vcard summary {{ cursor: pointer; color: {INK}; }}
.vtok {{ color: {CYAN}; font-weight: 600; }}
.vmeta {{ color: {MUTED}; font-size: 12px; font-family: ui-monospace, monospace; }}
.vbody {{ color: {INK}; padding: 8px 4px 4px; font-size: 14px;
          border-top: 1px solid {PANEL_ALT}; margin-top: 6px; }}
.rt {{ float: right; font-size: 11px; font-family: ui-monospace, monospace;
       border: 1px solid {MUTED}; border-radius: 10px; padding: 1px 8px;
       margin-left: 8px; }}
.abwrap {{ display: flex; gap: 12px; }}
.abcol {{ flex: 1; background: {PANEL}; border-radius: 10px; padding: 10px 12px; }}
.abhead {{ font-family: ui-monospace, monospace; font-size: 12px;
           margin-bottom: 6px; }}
</style>
"""


# App-level CSS (injected into gr.Blocks) — paints the whole Gradio surface in
# the dark-blue palette so the page matches the trace panels.
_APP_CSS = f"""
.gradio-container, .gradio-container .main, body {{
    background: {BG} !important;
    color: {INK} !important;
}}
.gradio-container .prose, .gradio-container .prose * {{ color: {INK} !important; }}
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-box, .gradio-container .gr-panel {{
    background: {PANEL} !important;
    border-color: {PANEL_ALT} !important;
    color: {INK} !important;
}}
.gradio-container input[type="text"], .gradio-container input[type="number"],
.gradio-container input[type="search"], .gradio-container textarea,
.gradio-container .gr-input, .gradio-container select {{
    background: {PANEL_ALT} !important;
    color: {INK} !important;
    border-color: {PANEL_ALT} !important;
}}
/* Keep native radio/checkbox controls interactive and visible — do NOT
   override their background, only tint the accent so they match the theme. */
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {{
    accent-color: {CYAN};
}}
.gradio-container .tab-nav button {{ color: {MUTED} !important; }}
.gradio-container .tab-nav button.selected {{ color: {CYAN} !important; }}
"""


# ── Generation callback (streaming) ──────────────────────────────────────
@_gpu(duration=120)
def cb_generate(prompt, mode, max_new, budget, k, temperature, top_p,
                repetition_penalty, tint, inject):
    if not prompt or not prompt.strip():
        yield (_CSS + "<i>Enter a prompt.</i>", "", "", "", "_(enter a prompt)_")
        return

    trace = _get_trace()
    model_prompt = prompt
    if mode == "Chat":
        # Use the backbone chat template if available.
        try:
            model_prompt = trace.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            model_prompt = prompt

    last = None
    for result, done in trace.stream(
        model_prompt,
        max_new_tokens=int(max_new), budget=int(budget), k=int(k),
        temperature=float(temperature), top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        disable_injectors=(not inject),
    ):
        last = result
        toks = _CSS + _render_tokens(result, tint)
        meter = _render_meter(result)
        charts = _render_charts(result)
        if done:
            verbs = _render_verbalizations(result)
            yield toks, meter, charts, verbs, result.text
        else:
            yield toks, meter, charts, "<i>generating… verbalizations appear when done.</i>", result.text


# ── Curated example gallery ───────────────────────────────────────────────
# Prompts grouped by the introspection phenomenon they tend to surface. Each
# row maps to the [prompt, mode] inputs. The categories are organised so a
# first-time visitor can see, in a few clicks, where the SRT signals light up:
# confident recall vs genuine uncertainty vs a false premise the model has to
# work around vs a reasoning pivot vs a safety boundary.
EXAMPLES = [
    # — Confident factual recall: low entropy at the fact token; the
    #   verbalization should name the very fact being emitted. —
    ["What is the capital of Australia, and when did it become the capital?", "Chat"],
    ["Who wrote the novel 'Pride and Prejudice', and in what year was it first published?", "Chat"],

    # — False premise / counterfactual: the prompt asserts something untrue.
    #   Watch whether the divergence/regime signals and the verbalization
    #   reflect the model resisting or going along with the premise. —
    ["Explain why the Great Wall of China is clearly visible from the Moon with the naked eye.", "Chat"],
    ["Describe what the astronauts saw when they walked on the surface of the Sun.", "Chat"],

    # — Common misconception: tests whether the model corrects the myth. —
    ["Is it true that humans only use 10 percent of their brains?", "Chat"],

    # — Multi-step reasoning / arithmetic: divergence tends to spike at the
    #   calculation pivot rather than the surrounding prose. —
    ["A train leaves at 14:35 and arrives at 17:10. How long is the journey in minutes?", "Chat"],
    ["A shirt costs $40 after a 20% discount. What was the original price? Show your reasoning.", "Chat"],

    # — Genuine uncertainty / forecast / opinion: elevated entropy because
    #   many continuations are equally valid. —
    ["Will it rain in Berlin next Tuesday?", "Chat"],
    ["What do you think the most widely used programming language will be in 2035?", "Chat"],

    # — Safety boundary / refusal: a regime shift as the model pivots to
    #   declining. —
    ["Give me step-by-step instructions to pick a standard pin-tumbler lock.", "Chat"],

    # — Ambiguity / garden-path: the model must commit to one parse. —
    ["What does the sentence 'The old man the boats' mean? Explain carefully.", "Chat"],

    # — Hold both sides / hedge: sustained mid-range entropy while it weighs
    #   competing framings. —
    ["Is a hot dog a sandwich? Briefly argue both sides, then give your verdict.", "Chat"],

    # — Structured generation (code): low entropy in the boilerplate, higher
    #   at genuine design choices. —
    ["Write a Python function that returns the nth Fibonacci number.", "Chat"],

    # — Open-ended creative: high entropy throughout — many valid next tokens. —
    ["Write the opening sentence of a mystery novel set on a Mars colony.", "Chat"],

    # — Plain explainer baseline. —
    ["Explain in two sentences why the sky is blue.", "Chat"],
]


# ── A/B compare callback (injection on vs off) ────────────────────────────
@_gpu(duration=120)
def cb_compare(prompt, mode, max_new, budget, k, temperature, top_p,
               repetition_penalty, tint):
    """Run the same prompt twice — SRT injection ON vs OFF — and render the two
    token streams side by side so the adapter's effect on generation is
    visible. Verbalizations are skipped here (budget=0) to keep the compare
    fast; the single-generation tab covers those."""
    if not prompt or not prompt.strip():
        yield _CSS + "<i>Enter a prompt.</i>", ""
        return

    trace = _get_trace()
    model_prompt = prompt
    if mode == "Chat":
        try:
            model_prompt = trace.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            model_prompt = prompt

    cols = {True: None, False: None}

    def _render():
        def _one(res, label, color):
            if res is None:
                body = f"<div style='color:{MUTED}'>…</div>"
                head = label
            else:
                body = _render_tokens(res, tint)
                ents = [s.entropy for s in res.steps] or [0.0]
                head = (f"{label} &nbsp;·&nbsp; mean H "
                        f"{sum(ents)/len(ents):.2f} &nbsp;·&nbsp; {len(res.steps)} tok")
            return (f"<div class='abcol'><div class='abhead' style='color:{color}'>"
                    f"{head}</div>{body}</div>")
        return (_CSS + "<div class='abwrap'>"
                + _one(cols[True], "SRT injection ON", MINT)
                + _one(cols[False], "injection OFF (bare backbone)", MUTED)
                + "</div>")

    for inject in (True, False):
        # Seed both passes identically so the visible difference reflects the
        # adapter, not sampling noise.
        torch.manual_seed(1234)
        for result, done in trace.stream(
            model_prompt,
            max_new_tokens=int(max_new), budget=0, k=int(k),
            temperature=float(temperature), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            disable_injectors=(not inject),
        ):
            cols[inject] = result
            yield _render(), ""

    a = (cols[True].text if cols[True] else "").strip()
    b = (cols[False].text if cols[False] else "").strip()
    summary = (
        f"**ON:** {a or '_(empty)_'}\n\n**OFF:** {b or '_(empty)_'}"
    )
    yield _render(), summary


def build() -> gr.Blocks:
    with gr.Blocks(title="SRT Showcase", css=_APP_CSS) as app:
        gr.Markdown(
            "## SRT Showcase — watch a frozen model think\n"
            "Live token-by-token introspection of **Qwen-2.5-7B + the SRT adapter**. "
            "Tokens are tinted by **predictive entropy** (validated uncertainty signal). "
            "SRT divergence, reflexivity `r̂`, regime, and the natural-language "
            "verbalizations are **observational readouts** of internal state — a window "
            "into the model, not a hallucination detector."
        )
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=4,
                                    value="Explain in two sentences why the sky is blue.")
                with gr.Row():
                    mode = gr.Radio(["Completion", "Chat"], value="Chat", label="Mode")
                    tint = gr.Radio(["entropy", "divergence"], value="entropy",
                                    label="Tint tokens by")
                    inject = gr.Checkbox(value=True, label="SRT injection on")
                with gr.Row():
                    max_new = gr.Slider(16, 1024, value=256, step=16, label="max tokens")
                    budget = gr.Slider(2, 20, value=10, step=1, label="verbalization slots")
                    k = gr.Slider(1, 8, value=4, step=1, label="AV samples / slot (K)")
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top-p")
                    rep = gr.Slider(1.0, 1.5, value=1.15, step=0.01, label="rep. penalty")
                with gr.Row():
                    go = gr.Button("Generate", variant="primary")
                    regen = gr.Button("Regenerate")
            with gr.Column(scale=1):
                meter = gr.HTML(label="entropy meter")

        with gr.Tab("Introspection"):
            tokens = gr.HTML(label="token stream")
            charts = gr.HTML(label="charts")
            with gr.Accordion("Verbalizations (expand each) — with round-trip fidelity", open=True):
                verbs = gr.HTML()
            final = gr.Textbox(label="Final output", lines=4)

        with gr.Tab("A/B: injection on vs off"):
            gr.Markdown(
                "Runs the same prompt twice with the SRT side-channel injection "
                "**on** and **off** (bare frozen backbone), seeded identically so "
                "the visible difference is the adapter, not sampling noise."
            )
            ab_go = gr.Button("Compare", variant="primary")
            ab_html = gr.HTML()
            ab_summary = gr.Markdown()

        gr.Markdown(
            "### Curated examples — what to watch for\n"
            "Pick a prompt below, then read the signals as it generates:\n"
            "- **Confident recall** (capital of Australia, *Pride and Prejudice*): "
            "low entropy at the fact; the verbalization names the fact itself.\n"
            "- **False premise** (Wall of China from the Moon, walking on the Sun): "
            "watch the divergence/regime signals as the model works around an untrue claim.\n"
            "- **Misconception** (10% of the brain): does it correct the myth?\n"
            "- **Reasoning pivot** (train minutes, discount price): divergence spikes at the calculation, not the prose.\n"
            "- **Genuine uncertainty** (rain Tuesday, language in 2035): elevated entropy — many valid continuations.\n"
            "- **Safety boundary** (lock picking): a regime shift as it pivots to declining.\n"
            "- **Ambiguity** ('The old man the boats'): the model commits to one parse.\n"
            "- **Open-ended / creative** (Mars mystery opener): high entropy throughout."
        )
        gr.Examples(
            examples=EXAMPLES, inputs=[prompt, mode], label="Curated examples",
            examples_per_page=15,
        )

        inputs = [prompt, mode, max_new, budget, k, temperature, top_p, rep, tint, inject]
        outputs = [tokens, meter, charts, verbs, final]
        go.click(cb_generate, inputs=inputs, outputs=outputs)
        regen.click(cb_generate, inputs=inputs, outputs=outputs)

        ab_inputs = [prompt, mode, max_new, budget, k, temperature, top_p, rep, tint]
        ab_go.click(cb_compare, inputs=ab_inputs, outputs=[ab_html, ab_summary])
    return app


if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", "8080"))
    build().queue().launch(
        server_name="0.0.0.0", server_port=server_port,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
