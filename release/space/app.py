"""SRT-Adapter v8a — Gradio Space.

Loads the v8a adapter on top of a frozen Qwen/Qwen2.5-7B and exposes:
  - per-token reflexivity heatmap (r_hat)
  - per-token P(supercritical)
  - passage-level summary card with verbal verdict
  - reference-distribution percentile vs val_200
  - JSON / Markdown export of the trace
"""

from __future__ import annotations

import html as _html
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

try:
    import spaces  # type: ignore
    HAS_ZEROGPU = True
except Exception:
    HAS_ZEROGPU = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((HERE / "src").resolve()))

from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import (  # noqa: E402
    SRTConfig, MAHConfig, RRMConfig, BENConfig, CommunityConfig, LossConfig,
)

ADAPTER_REPO = "RiverRider/srt-adapter-v8a"
MAX_SEQ_LEN = 512


def build_config(config_path: Path) -> SRTConfig:
    raw = json.loads(config_path.read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(**{
            k: v for k, v in raw["loss"].items()
            if k in LossConfig.__dataclass_fields__
        }),
    )


_state: dict = {
    "model": None, "tok": None, "device": None,
    "ref_r_hat_mean": None, "ref_p_super_frac": None,
    "ref_community_vectors": None, "ref_texts": [],
}


def _ensure_loaded():
    if _state["model"] is not None:
        return
    config_path = Path(hf_hub_download(ADAPTER_REPO, "config.json"))
    adapter_path = Path(hf_hub_download(ADAPTER_REPO, "adapter.safetensors"))
    config = build_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SRTAdapter(config).to(device)
    from safetensors.torch import load_file
    state = load_file(str(adapter_path), device=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(config.backbone_id)
    _state.update(model=model, tok=tok, device=device)
    _build_reference_baseline()


def _build_reference_baseline():
    try:
        path = hf_hub_download(ADAPTER_REPO, "data/val_200.jsonl")
    except Exception as e:  # noqa: BLE001
        print(f"[baseline] skip: {e}", flush=True)
        return
    means, fracs, cvs, texts = [], [], [], []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text") or obj.get("passage") or obj.get("input") or ""
            if not text.strip():
                continue
            try:
                _, r_hat, p_super, _, cv = _score_raw(text)
            except Exception:
                continue
            means.append(float(r_hat.mean()))
            fracs.append(float((p_super > 0.5).mean()))
            cvs.append(cv)
            texts.append(text)
            if len(means) >= 200:
                break
    if means:
        _state["ref_r_hat_mean"] = np.array(means)
        _state["ref_p_super_frac"] = np.array(fracs)
        _state["ref_community_vectors"] = np.stack(cvs, axis=0)
        _state["ref_texts"] = texts
        print(f"[baseline] indexed {len(means)} reference passages "
              f"(r̂ mean median={np.median(means):+.3f})", flush=True)


def _score_raw(text: str):
    model, tok, device = _state["model"], _state["tok"], _state["device"]
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0].tolist())
    r_hat = out.ben_output.r_hat[0].float().cpu().numpy()
    p_super = torch.softmax(out.ben_output.regime_logits[0], dim=-1)[:, 1].float().cpu().numpy()
    div_norms = [d[0].norm(dim=-1).float().cpu().numpy() for d in out.divergences]
    cv = out.community_output.vector[0].float().cpu().numpy()
    return tokens, r_hat, p_super, div_norms, cv


def _score(text: str):
    _ensure_loaded()
    return _score_raw(text)


# ----- formatting -----------------------------------------------------------

# Palette: matches the static landing card (RiverRider/srt-adapter-v8a-demo).
PAL_BG      = "#0f0f1a"
PAL_BG2     = "#181828"
PAL_FG      = "#e8e8f0"
PAL_MUTED   = "#9a9ab0"
PAL_ACCENT  = "#ff5e8a"   # high / warm
PAL_ACCENT2 = "#6e8cff"   # low / cool
PAL_BORDER  = "#2a2a40"


def _ramp(t: float) -> str:
    """Diverging color ramp: accent2 (low) -> bg2 (mid) -> accent (high).
    Matches the static card's SVG visuals."""
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        u = t / 0.5
        a = (0x6e, 0x8c, 0xff)
        b = (0x18, 0x18, 0x28)
    else:
        u = (t - 0.5) / 0.5
        a = (0x18, 0x18, 0x28)
        b = (0xff, 0x5e, 0x8a)
    r = int(a[0] + (b[0] - a[0]) * u)
    g = int(a[1] + (b[1] - a[1]) * u)
    bl = int(a[2] + (b[2] - a[2]) * u)
    return f"rgb({r},{g},{bl})"


def _legend_gradient_svg(vmin: float, vmax: float, label_lo: str = "low",
                         label_hi: str = "high") -> str:
    """Inline SVG color-scale strip used under each heatmap."""
    n, w, h = 24, 220, 8
    stops = "".join(
        f'<rect x="{(w*i)/n:.2f}" y="0" width="{w/n + 0.6:.2f}" '
        f'height="{h}" fill="{_ramp(i/(n-1))}"/>'
        for i in range(n)
    )
    return (
        f'<svg width="{w + 240}" height="{h + 4}" '
        f'style="vertical-align:middle">'
        f'<text x="110" y="{h - 1}" font-size="10" text-anchor="end" '
        f'fill="{PAL_MUTED}">{label_lo} ({vmin:+.2f})</text>'
        f'<g transform="translate(116, 0)">{stops}</g>'
        f'<text x="{116 + w + 6}" y="{h - 1}" font-size="10" '
        f'fill="{PAL_MUTED}">{label_hi} ({vmax:+.2f})</text>'
        f'</svg>'
    )


def _overview_strip_svg(values, vmin: float, vmax: float,
                        max_cells: int = 80) -> str:
    """Position-aware summary strip: one colored cell per chunk of the passage,
    plus a smoothed line on top. Lets you see passage-scale structure at a
    glance, especially for long inputs."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return ""
    span = max(vmax - vmin, 1e-6)
    cells = min(max_cells, n)
    # bin tokens into `cells` chunks
    binned = []
    for i in range(cells):
        a = int(i * n / cells)
        b = max(a + 1, int((i + 1) * n / cells))
        binned.append(float(arr[a:b].mean()))
    width, h = 760, 22
    cw = width / cells
    rects = "".join(
        f'<rect x="{i*cw:.2f}" y="0" width="{cw + 0.6:.2f}" '
        f'height="{h}" fill="{_ramp((v - vmin)/span)}"/>'
        for i, v in enumerate(binned)
    )
    # smoothed polyline overlay
    pts = " ".join(
        f"{(i + 0.5) * cw:.1f},{h - ((v - vmin)/span) * (h - 4) - 2:.1f}"
        for i, v in enumerate(binned)
    )
    return (
        f'<svg viewBox="0 0 {width} {h + 2}" preserveAspectRatio="none" '
        f'style="display:block;width:100%;height:{h + 2}px;'
        f'border:1px solid {PAL_BORDER};border-radius:6px;'
        f'background:{PAL_BG};margin-bottom:8px">'
        f'{rects}'
        f'<polyline points="{pts}" fill="none" '
        f'stroke="{PAL_FG}" stroke-width="1.1" opacity="0.55"/>'
        f'</svg>'
    )


def _heatmap_html(tokens, values, title: str, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        vmax = vmin + 1e-6
    span = vmax - vmin
    mean_v = float(np.mean(values))
    max_v = float(np.max(values))
    overview = _overview_strip_svg(values, vmin, vmax)
    spans = []
    for tok, v in zip(tokens, values):
        display = tok.replace("\u0120", " ").replace("\u2581", " ").replace("\u010a", "\u21b5")
        if not display.strip():
            display = display.replace(" ", "\u00a0")
        norm = float(np.clip((float(v) - vmin) / span, 0.0, 1.0))
        bg = _ramp(norm)
        fg = PAL_FG
        spans.append(
            f'<span title="{_html.escape(str(tok))}: {float(v):+.3f}" '
            f'style="background:{bg};color:{fg};padding:2px 4px;'
            f'margin:1px;border-radius:3px;'
            f'font-family:ui-monospace,Menlo,monospace;'
            f'font-size:13px;display:inline-block;'
            f'border:1px solid rgba(255,255,255,0.04);">'
            f'{_html.escape(display)}</span>'
        )
    stats = (
        f'<span style="color:{PAL_MUTED};font-size:12px">'
        f'mean {mean_v:+.2f} \u00b7 peak {max_v:+.2f} \u00b7 {len(values)} tokens'
        f'</span>'
    )
    legend = _legend_gradient_svg(vmin, vmax)
    return (
        f'<div style="background:{PAL_BG2};border:1px solid {PAL_BORDER};'
        f'border-radius:10px;padding:12px 14px;margin:8px 0;'
        f'color:{PAL_FG};line-height:1.9">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:8px;line-height:1.3">'
        f'<b style="color:{PAL_FG};font-size:13px">{title}</b>{stats}</div>'
        f'{overview}'
        f'<div>{"".join(spans)}</div>'
        f'<div style="margin-top:10px;border-top:1px solid {PAL_BORDER};'
        f'padding-top:8px">{legend}</div>'
        f'</div>'
    )


def _label_for_percentile(pct: int) -> tuple[str, str]:
    """Return (label, plain-English gloss) keyed off the val_200 percentile.

    Calibrated against the live baseline so the labels actually distribute
    rather than collapsing to "high" for nearly every input.
    """
    if pct < 0:
        return "unknown", "(reference baseline not loaded)"
    if pct < 15:
        return ("transparent",
                "the backbone is mostly *denoting* — words point at things "
                "and the model is doing little internal hedging or framing.")
    if pct < 40:
        return ("low-rhetoric",
                "below-average internal hedging. Reads as more direct than "
                "typical web text.")
    if pct < 70:
        return ("typical",
                "internal hedging/framing in the normal range for English "
                "prose. Most text lives here.")
    if pct < 90:
        return ("rhetorical",
                "above-average internal stance — the model is spending "
                "representational work on framing, qualification, or "
                "self-positioning rather than pure denotation.")
    return ("highly rhetorical",
            "top-decile internal stance work. Text of this kind tends to "
            "frame, hedge, intensify, or refer to its own utterance more "
            "than it refers to the world.")


def _percentile(value: float, ref) -> int:
    if ref is None or len(ref) == 0:
        return -1
    return int(round(100.0 * float((ref < value).mean())))


def _top_spans(tokens, r_hat, k: int = 3, win: int = 3):
    if len(tokens) < win:
        return []
    smooth = np.convolve(r_hat, np.ones(win) / win, mode="valid")
    idxs = np.argsort(-smooth)
    seen, results = set(), []
    for i in idxs:
        if any(abs(i - j) < win for j in seen):
            continue
        seen.add(int(i))
        span = tokens[int(i): int(i) + win]
        text = "".join(span).replace("\u0120", " ").replace("\u010a", " ")
        results.append((text.strip(), float(smooth[i])))
        if len(results) >= k:
            break
    return results


def _summary_card(tokens, r_hat, p_super) -> str:
    mean_r = float(r_hat.mean())
    max_r = float(r_hat.max())
    super_frac = float((p_super > 0.5).mean())

    pct_r = _percentile(mean_r, _state.get("ref_r_hat_mean"))
    pct_s = _percentile(super_frac, _state.get("ref_p_super_frac"))
    label, gloss = _label_for_percentile(pct_r)

    pct_s_html = (f"<span style='opacity:0.7;font-size:12px'>"
                  f"&nbsp; ({pct_s}th percentile vs val_200)</span>"
                  if pct_s >= 0 else "")

    spans = _top_spans(tokens, r_hat, k=3, win=3)
    spans_html = ""
    if spans:
        items = "".join(
            f"<li><code>{_html.escape(t)}</code> &nbsp; "
            f"<span style='opacity:0.65'>r̂ ≈ {v:+.2f}</span></li>"
            for t, v in spans
        )
        spans_html = (f"<div style='margin-top:8px'><b>Most contested spans:</b>"
                      f"<ul style='margin:4px 0 0 1.2em'>{items}</ul></div>")

    pct_phrase = (f" — higher than {pct_r}% of the val_200 baseline"
                  if pct_r >= 0 else "")
    verdict = (
        f"<div style='font-size:15px;line-height:1.5'>"
        f"This passage reads as <b>{label}</b>{pct_phrase}.</div>"
        f"<div style='margin-top:6px;opacity:0.85;font-size:13px;line-height:1.5'>"
        f"{gloss}</div>"
        f"<div style='margin-top:6px;opacity:0.7;font-size:12px'>"
        f"mean r̂ {mean_r:+.2f}, peak {max_r:+.2f}; "
        f"{super_frac*100:.0f}% of tokens flagged supercritical.</div>"
    )

    return f"""
    <div style="border:1px solid {PAL_BORDER};border-radius:10px;padding:14px 16px;
                background:{PAL_BG2};margin-bottom:6px;color:{PAL_FG};">
      {verdict}
      <div style="margin-top:10px;font-size:13px">
        <b>supercritical fraction</b>: {super_frac*100:.1f}%{pct_s_html}
        <span style="color:{PAL_MUTED};margin-left:6px">
          (fraction of tokens where the regime classifier flags
          high prompt-sensitivity)
        </span>
      </div>
      {spans_html}
    </div>
    """


INTERNALS_GLOSSARY = f"""
<div style='background:{PAL_BG2};border:1px solid {PAL_BORDER};
            border-radius:10px;padding:12px 14px;margin-bottom:12px;font-size:13px;
            line-height:1.55;color:{PAL_FG}'>
  <b>What you're looking at.</b><br>
  <b>Per-layer divergence (‖d‖)</b> — at MAH layers 7, 14, 21, the L2 norm of
  the divergence vector that the adapter computes from the backbone's residual
  stream. This is the raw signal r̂ is distilled from. Layer 7 ≈ early
  syntactic forking; layer 21 ≈ late discourse-level forking.<br>
  <b>Per-token layer breakdown</b> — at each token, the share of total
  divergence coming from each layer. Honest version: not "contribution to r̂"
  (that's mediated through RRM), just where in the network this token is
  doing its forking work.<br>
  <b>Regime trajectory</b> — P(supercritical) per token, with the 0.5
  decision boundary marked.<br>
  <b>Community vector</b> — 64-D unsupervised embedding from the
  Community Discovery Head (paper §3.2). Geometrically: where this passage
  lives in the discourse-trajectory space the adapter discovered.
</div>
"""


def _div_heatmap_set(tokens, div_norms) -> str:
    blocks = []
    for ln, arr in zip([7, 14, 21], div_norms):
        vmin, vmax = float(arr.min()), float(arr.max())
        blocks.append(_heatmap_html(tokens, arr,
                                    f"Layer {ln} divergence ‖d‖", vmin, vmax))
    return "<div style='display:flex;flex-direction:column;gap:14px'>" \
           + "".join(blocks) + "</div>"


def _layer_breakdown_svg(tokens, div_norms, max_tokens: int = 80) -> str:
    """Stacked bar per token: share of divergence from each of 3 layers."""
    layers = np.stack([np.asarray(d) for d in div_norms], axis=0)  # (3, T)
    totals = layers.sum(axis=0) + 1e-9
    shares = layers / totals  # (3, T)
    T = min(shares.shape[1], max_tokens)
    bw, h = 10, 70
    pad_l, pad_b = 0, 18
    width = bw * T
    colors = ["#6e8cff", "#ff5e8a", "#c9a4ff"]
    bars = []
    for i in range(T):
        y = 0.0
        for li in range(3):
            seg = float(shares[li, i]) * h
            bars.append(
                f'<rect x="{i*bw}" y="{y}" width="{bw-1}" height="{seg:.2f}" '
                f'fill="{colors[li]}" />'
            )
            y += seg
    legend = (
        "<div style='font-size:12px;margin-top:4px;display:flex;gap:14px;opacity:0.85'>"
        "<span><span style='display:inline-block;width:10px;height:10px;"
        "background:#6e8cff;margin-right:4px'></span>layer 7</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;"
        "background:#ff5e8a;margin-right:4px'></span>layer 14</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;"
        "background:#c9a4ff;margin-right:4px'></span>layer 21</span>"
        f"<span style='opacity:0.6'>(first {T} of {shares.shape[1]} tokens)</span>"
        "</div>"
    )
    note = ("<div style='opacity:0.65;font-size:12px;margin-top:6px'>"
            "Each column is one token; height = fraction of that token's "
            "total ‖d‖ contributed by each layer.</div>")
    return (
        "<div style='overflow-x:auto'>"
        f"<svg width='{width}' height='{h+pad_b}' "
        f"style='display:block;background:rgba(255,255,255,0.02);"
        f"border-radius:6px'>"
        f"{''.join(bars)}"
        f"</svg></div>{legend}{note}"
    )


def _regime_timeline_svg(tokens, p_super) -> str:
    arr = np.asarray(p_super)
    T = len(arr)
    if T == 0:
        return ""
    w_per = max(4, min(8, 800 // max(T, 1)))
    width, height, pad = w_per * T, 80, 18
    pts = []
    for i, v in enumerate(arr):
        x = i * w_per + w_per / 2
        y = height - float(np.clip(v, 0, 1)) * height
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ff5e8a" stroke-width="1.6"/>'
    threshold_y = height - 0.5 * height
    threshold = (
        f'<line x1="0" y1="{threshold_y}" x2="{width}" y2="{threshold_y}" '
        f'stroke="#9a9ab0" stroke-dasharray="3,3" stroke-width="1"/>'
    )
    crossings = []
    for i in range(1, T):
        a, b = float(arr[i - 1]), float(arr[i])
        if (a < 0.5) != (b < 0.5):
            x = i * w_per
            crossings.append(
                f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" '
                f'stroke="#6e8cff" stroke-width="0.6" opacity="0.5"/>'
            )
    n_super = int((arr > 0.5).sum())
    legend = (
        f"<div style='font-size:12px;margin-top:4px;opacity:0.8'>"
        f"P(supercritical) over {T} tokens · "
        f"{n_super} tokens above 0.5 · {len(crossings)} regime crossings · "
        f"dashed line = 0.5 boundary</div>"
    )
    return (
        "<div style='overflow-x:auto'>"
        f"<svg width='{width}' height='{height+pad}' "
        f"style='display:block;background:rgba(255,255,255,0.02);border-radius:6px'>"
        f"{threshold}{''.join(crossings)}{polyline}"
        f"</svg></div>{legend}"
    )


def _cv_bar_svg(cv, top_k: int = 16) -> str:
    cv = np.asarray(cv)
    norm = float(np.linalg.norm(cv))
    order = np.argsort(-np.abs(cv))[:top_k]
    vmax = float(np.max(np.abs(cv[order]))) + 1e-9
    bar_h, gap = 14, 4
    width = 320
    centerx = width / 2
    rows = []
    for rank, i in enumerate(order):
        v = float(cv[int(i)])
        seg = (v / vmax) * (width / 2 - 30)
        x = centerx if seg >= 0 else centerx + seg
        color = "#6e8cff" if v >= 0 else "#ff5e8a"
        y = rank * (bar_h + gap)
        rows.append(
            f'<rect x="{x:.1f}" y="{y}" width="{abs(seg):.1f}" height="{bar_h}" fill="{color}"/>'
            f'<text x="{centerx + (width/2-25) + 4 if v >= 0 else centerx - (width/2-25) - 4:.1f}" '
            f'y="{y + bar_h - 3}" font-size="10" fill="#9a9ab0" '
            f'text-anchor="{"start" if v >= 0 else "end"}">d{int(i)} {v:+.2f}</text>'
        )
    height = top_k * (bar_h + gap)
    return (
        f"<div style='font-size:13px;margin-bottom:6px'>"
        f"<b>Community vector</b> · 64-D · L2 norm = {norm:.3f}</div>"
        f"<svg width='{width + 80}' height='{height}' "
        f"style='display:block'>"
        f'<line x1="{centerx}" y1="0" x2="{centerx}" y2="{height}" '
        f'stroke="#2a2a40" stroke-width="1"/>'
        f"{''.join(rows)}"
        f"</svg>"
        f"<div style='opacity:0.65;font-size:12px;margin-top:4px'>"
        f"Top {top_k} dimensions by |value|. Blue = positive, pink = negative.</div>"
    )


def _nearest_neighbors_html(cv, k: int = 3) -> str:
    ref = _state.get("ref_community_vectors")
    texts = _state.get("ref_texts") or []
    if ref is None or len(texts) == 0:
        return ""
    cv_n = cv / (np.linalg.norm(cv) + 1e-9)
    ref_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
    sims = ref_n @ cv_n
    top = np.argsort(-sims)[:k]
    items = []
    for i in top:
        snippet = texts[int(i)].strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "…"
        items.append(
            f"<li style='margin-bottom:6px'>"
            f"<span style='opacity:0.6;font-size:12px'>cos = {float(sims[int(i)]):+.3f}</span>"
            f" &nbsp; <span>{_html.escape(snippet)}</span></li>"
        )
    return (
        "<div style='margin-top:14px'>"
        "<b>Nearest val_200 passages in community space</b>"
        "<div style='opacity:0.65;font-size:12px;margin-bottom:6px'>"
        "Cosine similarity in 64-D community space. Anchors what 'this region "
        "of discourse-space' actually looks like.</div>"
        f"<ul style='margin:4px 0 0 1.2em;padding:0'>{''.join(items)}</ul></div>"
    )


def _div_summary(div_norms) -> str:
    layer_names = ["layer 7", "layer 14", "layer 21"]
    rows = []
    for name, arr in zip(layer_names, div_norms):
        rows.append(
            f"<tr><td style='padding-right:1em'>{name}</td>"
            f"<td style='text-align:right;padding-right:1em'>{float(arr.mean()):.3f}</td>"
            f"<td style='text-align:right'>{float(arr.max()):.3f}</td></tr>"
        )
    return (
        "<table style='border-collapse:collapse;font-size:13px'>"
        "<thead><tr><th style='text-align:left;padding-right:1em'>MAH layer</th>"
        "<th style='text-align:right;padding-right:1em'>mean ‖d‖</th>"
        "<th style='text-align:right'>max ‖d‖</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _internals_html(tokens, r_hat, p_super, div_norms, cv) -> str:
    sections = [
        INTERNALS_GLOSSARY,
        "<h4 style='margin:14px 0 6px'>Per-layer divergence heatmaps</h4>",
        _div_heatmap_set(tokens, div_norms),
        "<h4 style='margin:18px 0 6px'>Per-token layer breakdown</h4>",
        _layer_breakdown_svg(tokens, div_norms),
        "<h4 style='margin:18px 0 6px'>Regime trajectory</h4>",
        _regime_timeline_svg(tokens, p_super),
        "<h4 style='margin:18px 0 6px'>Layer summary</h4>",
        _div_summary(div_norms),
        "<h4 style='margin:18px 0 6px'>Community vector</h4>",
        _cv_bar_svg(cv),
        _nearest_neighbors_html(cv),
    ]
    return "".join(sections)


def _trace_to_json(text, tokens, r_hat, p_super, div_norms, cv) -> str:
    obj = {
        "passage": text,
        "tokens": list(tokens),
        "r_hat": [round(float(x), 4) for x in r_hat],
        "p_supercritical": [round(float(x), 4) for x in p_super],
        "divergence_norms": {
            f"layer_{ln}": [round(float(x), 4) for x in arr]
            for ln, arr in zip([7, 14, 21], div_norms)
        },
        "community_vector": [round(float(x), 4) for x in cv],
    }
    return json.dumps(obj, indent=2)


def _trace_to_markdown(text, tokens, r_hat, p_super) -> str:
    mean_r = float(r_hat.mean())
    super_frac = float((p_super > 0.5).mean())
    spans = _top_spans(tokens, r_hat, k=3, win=3)
    pct_r = _percentile(mean_r, _state.get("ref_r_hat_mean"))
    label, _ = _label_for_percentile(pct_r)
    lines = [
        "## SRT-Adapter v8a trace",
        "",
        f"> {text}",
        "",
        f"- **mean r̂**: {mean_r:+.3f}"
        + (f" ({pct_r}th percentile vs val_200)" if pct_r >= 0 else ""),
        f"- **supercritical fraction**: {super_frac*100:.1f}%",
        f"- **label**: {label}",
    ]
    if spans:
        lines.append("- **most contested spans**: " +
                     ", ".join(f"`{t}` (r̂≈{v:+.2f})" for t, v in spans))
    lines.append("")
    lines.append("Generated by https://huggingface.co/RiverRider/srt-adapter-v8a")
    return "\n".join(lines)


# ----- main inference --------------------------------------------------------

def _do_inference(text: str):
    text = (text or "").strip()
    if not text:
        return ("Please enter a non-empty passage.", "", "", "", "", "", "")
    try:
        tokens, r_hat, p_super, div_norms, cv = _score(text)
    except Exception as e:  # noqa: BLE001
        return (f"<b>Error:</b> {_html.escape(str(e))}", "", "", "", "", "")

    summary_html = _summary_card(tokens, r_hat, p_super)
    rh_html = _heatmap_html(tokens, r_hat, "Per-token reflexivity (r̂)",
                            float(r_hat.min()), float(r_hat.max()))
    ps_html = _heatmap_html(tokens, p_super, "Per-token P(supercritical)", 0.0, 1.0)
    internals_html = _internals_html(tokens, r_hat, p_super, div_norms, cv)
    json_blob = _trace_to_json(text, tokens, r_hat, p_super, div_norms, cv)
    md_blob = _trace_to_markdown(text, tokens, r_hat, p_super)
    return summary_html, rh_html, ps_html, internals_html, json_blob, md_blob



# ----- usage logging --------------------------------------------------------
import os as _os, time as _time, hashlib as _hashlib, threading as _threading
_USAGE_LOG = _os.environ.get("SRT_USAGE_LOG", "/root/srt-adapter/logs/usage.jsonl")
_os.makedirs(_os.path.dirname(_USAGE_LOG), exist_ok=True)
_USAGE_LOCK = _threading.Lock()
_inner_do_inference = _do_inference

def _do_inference(text: str):  # type: ignore[no-redef]
    t0 = _time.time()
    raw = text or ""
    rec = {
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "text_len": len(raw),
        "text_hash": _hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        "text_sample": raw[:160],
    }
    try:
        out = _inner_do_inference(text)
        rec["status"] = "ok"
        return out
    except Exception as e:
        rec["status"] = "err"
        rec["err"] = repr(e)[:200]
        raise
    finally:
        rec["latency_ms"] = int((_time.time() - t0) * 1000)
        try:
            with _USAGE_LOCK, open(_USAGE_LOG, "a") as _f:
                _f.write(json.dumps(rec) + chr(10))
        except Exception:
            pass


if HAS_ZEROGPU:
    _do_inference = spaces.GPU(duration=120)(_do_inference)  # type: ignore


EXAMPLE_GROUPS: list[dict] = [
    {
        "name": "Pure denotation",
        "expected": "Expected: transparent → low-rhetoric (≈ 0–40th percentile).",
        "blurb": "Statements that point at the world. Little hedging, framing, "
                 "or self-reference. The backbone is mostly *denoting*.",
        "items": [
            ["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
            ["The Eiffel Tower is 330 meters tall, including its broadcasting antenna."],
            ["The training loss converged after roughly 30,000 steps on a single A6000."],
            ["SELECT user_id, COUNT(*) FROM events WHERE ts > now() - interval '7 days' GROUP BY 1;"],
        ],
    },
    {
        "name": "Same claim, framed vs neutral",
        "expected": "Expected: a 20–60-percentile gap between each pair.",
        "blurb": "Pairs with roughly the same propositional content but different "
                 "rhetorical packaging. The framed half should score higher.",
        "items": [
            ["Vaccines are safe and effective."],
            ["Vaccine mandates are an obvious public-health win — only the cranks disagree."],
            ["Free speech is a foundational liberal value."],
            ["Free speech is sacred, but only for ideas I agree with."],
            ["Unemployment fell to 3.8% in the latest quarter."],
            ["Unemployment fell to 3.8% — finally, real proof the policy is working."],
        ],
    },
    {
        "name": "Implicature & insinuation",
        "expected": "Expected: rhetorical → highly rhetorical (≈ 70–95th percentile).",
        "blurb": "Sentences that mean more than they say. The model has to do "
                 "extra meta-stance work to keep the implicature alive.",
        "items": [
            ["Of course, the official report concludes there is nothing to investigate."],
            ["The price of bread doubled in three years, but the official inflation index says everything is fine."],
            ["I'm not saying he lied. I'm just saying the timeline is, let's say, interesting."],
        ],
    },
    {
        "name": "Marketing & brand voice",
        "expected": "Expected: rhetorical → highly rhetorical (≈ 75–99th percentile).",
        "blurb": "Pure framing with very little denotation. Often near the top of "
                 "the val_200 distribution.",
        "items": [
            ["Introducing the future of productivity. Reimagined. Refined. For you."],
            ["Our journey began with a simple question: what if banking could feel human again?"],
        ],
    },
    {
        "name": "Bureaucratic & legalese",
        "expected": "Expected: typical → rhetorical (≈ 50–85th percentile).",
        "blurb": "Hedging-dense, agent-suppressed, ritualized prose. Reflexive "
                 "without being persuasive.",
        "items": [
            ["The Committee notes with concern the matters raised herein and reserves the right to revisit them at a future date."],
            ["Notwithstanding any provision to the contrary, the licensee shall be deemed to have accepted such terms upon use."],
        ],
    },
    {
        "name": "First-person & relational",
        "expected": "Expected: rhetorical (≈ 70–90th percentile).",
        "blurb": "Self-referential, stance-dense interior speech. The backbone "
                 "is positioning the speaker as much as denoting anything.",
        "items": [
            ["I think I'm finally beginning to understand why I keep doing this to myself."],
            ["Look, I love you, but I need you to hear what I'm actually trying to say."],
        ],
    },
    {
        "name": "Literary, poetic & meta",
        "expected": "Expected: variable — depends on whether the language is doing world-pointing or self-pointing.",
        "blurb": "Edge cases. Useful for kicking the tires on what the readout "
                 "actually tracks.",
        "items": [
            ["The harbor lights came on one by one, the way memory sometimes does, without warning."],
            ["This sentence is an example of a sentence whose meaning depends on calling itself an example."],
            ["The data is clear: anyone still defending this position is either lying or hasn't read the study."],
        ],
    },
]


DEMO_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff0f4", c100="#ffd6e1", c200="#ffadc4", c300="#ff85a8",
        c400="#ff6f99", c500="#ff5e8a", c600="#e64f78", c700="#bf3f63",
        c800="#992f4d", c900="#731f37", c950="#4d1424",
    ),
    secondary_hue=gr.themes.Color(
        c50="#eef2ff", c100="#dbe3ff", c200="#b8c8ff", c300="#94adff",
        c400="#7e9cff", c500="#6e8cff", c600="#5a73d6", c700="#475bab",
        c800="#354481", c900="#232d57", c950="#11162c",
    ),
    neutral_hue=gr.themes.Color(
        c50="#e8e8f0", c100="#c8c8d6", c200="#9a9ab0", c300="#6e6e85",
        c400="#4a4a60", c500="#2a2a40", c600="#23233a", c700="#1c1c2f",
        c800="#181828", c900="#13131f", c950="#0f0f1a",
    ),
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#0f0f1a",
    body_text_color="#e8e8f0",
    background_fill_primary="#181828",
    background_fill_secondary="#0f0f1a",
    border_color_primary="#2a2a40",
    block_background_fill="#181828",
    block_border_color="#2a2a40",
    block_label_background_fill="#181828",
    block_label_text_color="#9a9ab0",
    block_title_text_color="#e8e8f0",
    input_background_fill="#0f0f1a",
    input_border_color="#2a2a40",
    button_primary_background_fill="linear-gradient(135deg,#ff5e8a,#6e8cff)",
    button_primary_background_fill_hover="linear-gradient(135deg,#ff7099,#7d99ff)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#ff5e8a",
    button_secondary_background_fill="#181828",
    button_secondary_text_color="#e8e8f0",
    button_secondary_border_color="#2a2a40",
    color_accent="#ff5e8a",
    color_accent_soft="#6e8cff",
    link_text_color="#6e8cff",
    link_text_color_hover="#ff5e8a",
)

DEMO_CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.gradio-container, body { background: #0f0f1a !important; color: #e8e8f0 !important; }
footer { display: none !important; }
.tabs > .tab-nav button { color: #9a9ab0 !important; border-bottom: 2px solid transparent !important; }
.tabs > .tab-nav button.selected { color: #e8e8f0 !important; border-bottom-color: #ff5e8a !important; }
"""

SRT_LOGO_SVG = """
<div style='text-align:center;margin:8px 0 18px'>
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 200' role='img'
     aria-label='SRT — Semiotic-Reflexive Transformer'
     style='width:min(380px,80%);height:auto;display:inline-block'>
  <defs>
    <linearGradient id='srtStrokeDemo' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#ff5e8a'/>
      <stop offset='55%' stop-color='#c376c6'/>
      <stop offset='100%' stop-color='#6e8cff'/>
    </linearGradient>
    <linearGradient id='srtFillFaintDemo' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#ff5e8a' stop-opacity='0.06'/>
      <stop offset='100%' stop-color='#6e8cff' stop-opacity='0.06'/>
    </linearGradient>
  </defs>
  <text x='260' y='120'
        font-family="'Didot','GFS Didot','Bodoni 72','Bodoni MT','Playfair Display',Georgia,serif"
        font-style='italic' font-weight='400' font-size='148'
        text-anchor='middle' letter-spacing='6'
        fill='url(#srtFillFaintDemo)'
        stroke='url(#srtStrokeDemo)' stroke-width='1.4'
        paint-order='stroke fill'>SRT</text>
  <line x1='60' y1='148' x2='460' y2='148' stroke='#2a2a40' stroke-width='1'/>
  <text x='260' y='178'
        font-family="'Didot','GFS Didot','Bodoni 72','Bodoni MT','Playfair Display',Georgia,serif"
        font-style='normal' font-weight='400' font-size='14'
        text-anchor='middle' letter-spacing='6.5'
        fill='#9a9ab0'>SEMIOTIC&#8202;&#8211;&#8202;REFLEXIVE&#8202;&#8202;TRANSFORMER</text>
</svg>
<div style='color:#9a9ab0;font-size:13px;margin-top:4px;letter-spacing:0.04em'>v8a &middot; live readouts on a frozen Qwen/Qwen2.5-7B</div>
</div>
"""

with gr.Blocks(title="SRT-Adapter v8a Demo", theme=DEMO_THEME, css=DEMO_CSS) as demo:
    gr.HTML(SRT_LOGO_SVG)
    gr.Markdown(
        "A 14.5M-param adapter that bolts **semiotic awareness** onto a 7B "
        "causal LM without modifying a single backbone parameter. "
        "Paste a passage to get a verdict, per-token reflexivity (r̂), "
        "supercritical-regime probability, and per-layer "
        "divergence norms.\n\n"
        "Model card: [RiverRider/srt-adapter-v8a](https://huggingface.co/RiverRider/srt-adapter-v8a)"
    )
    with gr.Row():
        inp = gr.Textbox(
            label="Passage",
            placeholder="Paste up to ~512 tokens of English text.",
            lines=4,
        )
    with gr.Row():
        btn = gr.Button("Score passage", variant="primary")
        clear = gr.Button("Clear")

    with gr.Accordion("How to read this", open=False):
        gr.Markdown(
            "**What is being measured?** The adapter produces three readouts "
            "per token from the frozen Qwen2.5-7B residual stream:\n\n"
            "- **r̂ (reflexivity)** — a continuous score, roughly z-scored at "
            "training time. Loosely: how strongly the model's internal "
            "representation at this position is *aware of itself as language* "
            "rather than transparently denoting a fact. Trained against a "
            "self-supervised target derived from MAH (multi-axis hessian) "
            "divergence between successive transformer layers.\n"
            "- **P(supercritical)** — a calibrated probability in [0, 1] from "
            "a regime classifier head. Supercritical ≈ the model is operating "
            "in a regime where small perturbations to the prompt yield large "
            "swings in next-token distribution. Subcritical ≈ stable, "
            "low-entropy continuation.\n"
            "- **‖d‖ (divergence norm)** — per-layer norm of the MAH "
            "divergence at layers 7, 14, 21. The signal r̂ is distilled from.\n\n"
            "**What it is *not*.** It is **not** a contestedness detector, "
            "a hallucination detector, or a fact-checker. It is a readout of "
            "*how the backbone is processing the text*. Short factual "
            "statements about contested topics often score low (the model is "
            "just denoting); long fluent rhetoric on a banal topic can score "
            "high (the model is hedging, framing, or self-referencing).\n\n"
            "**Training data.** A self-supervised target derived from MAH "
            "divergence on a corpus of English passages — the adapter is "
            "trained to predict where the backbone is expending "
            "representational work, not to label any external category. "
            "The 200-passage validation set used for the percentiles in "
            "the summary card is `data/val_200.jsonl` on the "
            "[model repo](https://huggingface.co/RiverRider/srt-adapter-v8a/tree/main/data).\n\n"
            "**Reading the heatmap.** Red = high, blue = low. Hover any "
            "token to see its exact value. The 'most contested spans' "
            "section in the summary card is a 3-token rolling mean of r̂, "
            "deduplicated so spans don't overlap.\n\n"
            "**No backbone parameters were modified.** All readouts come "
            "from the 14.5M-param adapter side-network. The full model + "
            "code are on the "
            "[model card](https://huggingface.co/RiverRider/srt-adapter-v8a)."
        )

    summary = gr.HTML(label="Summary")
    with gr.Tab("Heatmaps"):
        rh = gr.HTML(label="r_hat heatmap")
        ps = gr.HTML(label="P(supercritical) heatmap")
    with gr.Tab("Internals"):
        internals = gr.HTML(label="Internals")
    with gr.Tab("Export"):
        gr.Markdown("Copy the trace as JSON (raw vectors) or Markdown (summary).")
        json_box = gr.Code(label="JSON trace", language="json", lines=12)
        md_box = gr.Code(label="Markdown summary", language="markdown", lines=8)

    gr.Markdown("### Curated examples")
    gr.Markdown(
        "Each tab is a register the adapter responds to differently. The "
        "**expected percentile range** is calibrated against the val_200 "
        "baseline — your mileage will vary, that's part of the point.",
    )
    with gr.Tabs():
        for group in EXAMPLE_GROUPS:
            with gr.Tab(group["name"]):
                gr.Markdown(
                    f"*{group['blurb']}*  \n**{group['expected']}**"
                )
                gr.Examples(
                    group["items"], inputs=[inp],
                    label="", examples_per_page=10,
                )

    btn.click(
        _do_inference, inputs=[inp],
        outputs=[summary, rh, ps, internals, json_box, md_box],
    )
    clear.click(
        lambda: ("", "", "", "", "", "", ""),
        outputs=[inp, summary, rh, ps, internals, json_box, md_box],
    )

    gr.Markdown(
        "---\n"
        "**Tip.** Try the curated examples in pairs (e.g. *Free speech is a "
        "foundational liberal value* vs *Free speech is sacred, but only for "
        "ideas I agree with*). Same topic, very different r̂ profiles — that "
        "gap is what the adapter actually learned to see."
    )


if __name__ == "__main__":
    print("prewarming model...", flush=True)
    _ensure_loaded()
    print("warmed.", flush=True)
    demo.queue(max_size=8).launch(share=True)
