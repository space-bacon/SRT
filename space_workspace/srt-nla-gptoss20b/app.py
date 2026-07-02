"""SRT-NLA gpt-oss-20b — full input→output trace demo (HF Space, ZeroGPU).

What it shows, honestly:

  1. FULL TRACE — the model generates a continuation; every (layer, position)
     hidden state across L6/L12/L18/L24 is assigned a "magic number" via the
     4096-code VQ state codebook, rendered as an addressable grid over input
     AND output tokens. Hovering a cell shows the codebook's retrieval
     verbalization (O(1) nearest-centroid decode).
  2. VERBALIZE — for any position, compare the trained Activation Verbalizer
     (best-of-K sampling, oracle rerank) against codebook retrieval, both
     scored with the anisotropy-CENTERED metric against published anchors.
     On this backbone retrieval wins; the demo shows that honestly.
  3. Generated tokens are tinted by predictive entropy (the validated
     uncertainty signal, same as the original SRT Showcase).

Anchors (centered fve, L18): random floor 0.500 · NN retrieval 0.744 ·
replay ceiling 0.999. AV K-curve tops out at 0.642 @ K=64.
"""
from __future__ import annotations

import hashlib
import html
import math
import os

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- ZeroGPU shim ----------------
try:
    import spaces  # type: ignore

    _ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU")) or bool(os.environ.get("SPACE_ID"))

    def _gpu(duration: int = 120):
        if _ON_ZEROGPU:
            return spaces.GPU(duration=duration)
        return lambda fn: fn
except ImportError:  # local dev
    _ON_ZEROGPU = False

    def _gpu(duration: int = 120):
        return lambda fn: fn


BB_ID = "openai/gpt-oss-20b"
AV_REPO = "RiverRider/srt-nla-av-gptoss20b"
ART_REPO = "RiverRider/srt-nla-gptoss20b-artifacts"
LAYERS = [6, 12, 18, 24]
L_MAIN = 18
FLOOR, NN_BASE, REPLAY = 0.500, 0.744, 0.999   # centered fve anchors (L18)
N_PREFIX = 16
MAX_PROMPT_CHARS = 800
MAX_NEW_CAP = 256

_STATE: dict = {}


def _predownload() -> None:
    """Fetch all weights at startup (CPU time) so GPU calls only load from disk."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        snapshot_download(BB_ID, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
        hf_hub_download(AV_REPO, "best_av/best_av.pt")
        hf_hub_download(ART_REPO, "state_codebook_vq.pt", repo_type="dataset")
        hf_hub_download(ART_REPO, "mu_L18.pt", repo_type="dataset")
    except Exception as e:  # noqa: BLE001 — non-fatal; the GPU call retries
        print("predownload warning:", e)


_predownload()


# ---------------- vendored minimal AV ----------------
class MiniAV(nn.Module):
    """Prefix-injection verbalizer matching the released checkpoint layout."""

    def __init__(self, d: int, n_layers: int, bos_embed: torch.Tensor):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.prefix_embeds = nn.Parameter(bos_embed.expand(N_PREFIX, -1).clone())
        self.layer_embed = nn.Embedding(n_layers + 1, d)

    def inject(self, v: torch.Tensor, layer: int, dtype) -> torch.Tensor:
        v32 = v.float()
        if v32.dim() == 1:
            v32 = v32.unsqueeze(0)
        lid = torch.full((v32.size(0),), layer, dtype=torch.long, device=v32.device)
        slot = (self.proj(v32) + self.layer_embed(lid).float()).unsqueeze(1)
        pref = self.prefix_embeds.unsqueeze(0).expand(v32.size(0), -1, -1).float()
        return torch.cat([slot, pref], dim=1).to(dtype)


# ---------------- vendored codebook ----------------
class MiniCodebook:
    def __init__(self, obj: dict, device: str):
        self.mu = obj["mu"].float().to(device)
        self.centroids = obj["centroids"].float().to(device)          # (k, d) centred
        self.text: dict[int, str] = {int(e["code"]): (e["text"] or "") for e in obj["entries"]}
        self.count: dict[int, int] = {int(e["code"]): int(e["count"]) for e in obj["entries"]}
        self.labels: dict[int, str] = {}

    def label(self, code: int) -> str:
        return self.labels.get(int(code), "")

    def encode(self, vs: torch.Tensor) -> torch.Tensor:
        vc = vs.float().to(self.centroids.device) - self.mu
        return torch.cdist(vc, self.centroids).argmin(dim=-1)

    def decode(self, code: int) -> str:
        return self.text.get(int(code), "")


def _load_everything():
    if _STATE.get("ready"):
        return
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BB_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bb = AutoModelForCausalLM.from_pretrained(
        BB_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    d = bb.config.hidden_size
    emb = bb.get_input_embeddings()
    bos_embed = emb(torch.tensor([tok.bos_token_id or 0], device=emb.weight.device)).detach().float().cpu()

    av = MiniAV(d, bb.config.num_hidden_layers, bos_embed[0])
    sd = torch.load(hf_hub_download(AV_REPO, "best_av/best_av.pt"),
                    map_location="cpu", weights_only=False)
    state = sd.get("trainable", sd)
    av.load_state_dict({k: v for k, v in state.items() if k in av.state_dict()}, strict=False)
    av = av.cuda().eval()

    cb_obj = torch.load(hf_hub_download(ART_REPO, "state_codebook_vq.pt", repo_type="dataset"),
                        map_location="cpu", weights_only=False)
    cb = MiniCodebook(cb_obj, "cuda")

    mu18 = torch.load(hf_hub_download(ART_REPO, "mu_L18.pt", repo_type="dataset"),
                      map_location="cpu", weights_only=False).float().cuda()
    try:
        import json as _json
        raw = _json.load(open(hf_hub_download(ART_REPO, "codebook_labels.json",
                                              repo_type="dataset")))
        good = {int(k): v.strip() for k, v in raw.items()
                if v and len(v.split()) >= 2 and not v.strip().isdigit()}
        cb.labels = good
    except Exception as e:  # noqa: BLE001 — labels are an enhancement, not required
        print("labels unavailable:", e)
    try:
        mu_by = torch.load(hf_hub_download(ART_REPO, "mu_by_layer.pt", repo_type="dataset"),
                           map_location="cpu", weights_only=False)
        mu_by = {int(k): v.float().cuda() for k, v in mu_by.items()}
    except Exception:  # noqa: BLE001
        mu_by = {L: mu18 for L in LAYERS}

    _STATE.update(ready=True, tok=tok, bb=bb, av=av, cb=cb, mu18=mu18, mu_by=mu_by, d=d)


def _cen_fve(h: torch.Tensor, v: torch.Tensor, mu: torch.Tensor) -> float:
    return float(0.5 * (1.0 + F.cosine_similarity((h - mu).unsqueeze(0), (v - mu).unsqueeze(0))))


def _code_color(code: int) -> str:
    hue = int(hashlib.md5(str(code).encode()).hexdigest()[:4], 16) % 360
    return f"hsl({hue},62%,30%)"


def _entropy_color(h: float, hmax: float = 4.0) -> str:
    t = max(0.0, min(1.0, h / hmax))
    r, g, b = int(30 + 200 * t), int(140 - 60 * t), int(220 - 160 * t)
    return f"rgb({r},{g},{b})"


def _conf_band(cen: float):
    """(color, label) for a centered round-trip fidelity score."""
    if cen >= NN_BASE:
        return "#7eebc0", "strong (beats retrieval baseline)"
    if cen >= 0.62:
        return "#7ee0ff", "good"
    if cen >= 0.55:
        return "#e8c468", "moderate"
    return "#ff7eb9", "weak"


def _conf_bar(cen: float) -> str:
    pct = max(2.0, min(100.0, (cen - FLOOR) / (REPLAY - FLOOR) * 100))
    col, label = _conf_band(cen)
    return (
        f"<div title='round-trip fidelity (centered) {cen:.3f} - {label}' "
        f"style='background:#16213d;border-radius:4px;height:8px;width:100%;margin-top:4px'>"
        f"<div style='background:{col};width:{pct:.0f}%;height:8px;border-radius:4px'></div></div>"
        f"<div style='font-size:10px;color:{col}'>confidence {cen:.2f} - {label}</div>")


@_gpu(duration=180)
def cb_trace(prompt: str, max_new: int, stride: int):
    prompt = (prompt or "").strip()[:MAX_PROMPT_CHARS]
    if not prompt:
        return "<p>Enter a prompt.</p>", ""
    _load_everything()
    tok, bb, cb, mu_by = _STATE["tok"], _STATE["bb"], _STATE["cb"], _STATE["mu_by"]
    max_new = int(min(max(max_new, 8), MAX_NEW_CAP))

    enc = tok(prompt, return_tensors="pt").to("cuda")
    n_in = enc.input_ids.shape[1]
    with torch.no_grad():
        gen = bb.generate(**enc, max_new_tokens=max_new, do_sample=True,
                          temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
        full = gen[0]
        out = bb(input_ids=full.unsqueeze(0), output_hidden_states=True, use_cache=False)

    logp = F.log_softmax(out.logits[0].float(), dim=-1)
    ent = (-(logp.exp() * logp).sum(-1)).tolist()

    toks = [tok.decode([t]) for t in full.tolist()]
    T = full.shape[0]
    stride = max(1, int(stride))
    cols = list(range(0, T, stride))
    if (T - 1) not in cols:
        cols.append(T - 1)

    # codes + vectors for every displayed (layer, col)
    grid = {}
    vecs = {}
    for L in LAYERS:
        hL = out.hidden_states[L][0][cols].float()
        vecs[L] = hL
        grid[L] = cb.encode(hL).tolist()

    # ---- round-trip confidence for every displayed cell ----------------
    # Re-encode each UNIQUE codebook text once; one forward yields hiddens at
    # every layer, so cell (L, p) is scored against that layer's mu.
    uniq_codes = sorted({c for L in LAYERS for c in grid[L]})[:192]
    code_h = {}
    texts = [(c, (cb.decode(c) or "").strip()) for c in uniq_codes]
    texts = [(c, t) for c, t in texts if t]
    with torch.no_grad():
        for s in range(0, len(texts), 16):
            chunk = texts[s:s + 16]
            e = tok([t for _, t in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=96).to("cuda")
            o = bb(input_ids=e.input_ids, attention_mask=e.attention_mask,
                   output_hidden_states=True, use_cache=False)
            last = (e.attention_mask.sum(1) - 1).clamp(min=0)
            rows_i = torch.arange(len(chunk), device="cuda")
            for L in LAYERS:
                hl = o.hidden_states[L][rows_i, last].float()
                for b, (c, _) in enumerate(chunk):
                    code_h.setdefault(c, {})[L] = hl[b]

    def cell_conf(L: int, j: int):
        c = grid[L][j]
        if c not in code_h:
            return None
        return _cen_fve(code_h[c][L], vecs[L][j], mu_by[L])

    # ---- true NLA: AV-generated decodes on the highest-entropy steps ----
    av = _STATE["av"]
    K_AV = 4
    n_av = min(8, len(cols))
    order = sorted(range(len(cols)),
                   key=lambda j: -(ent[cols[j] - 1] if 0 < cols[j] <= len(ent) else 0.0))
    av_rows = sorted(order[:n_av])
    av_deco: dict[int, tuple[str, float]] = {}
    if av_rows:
        with torch.no_grad():
            v_sel = torch.stack([vecs[L_MAIN][j] for j in av_rows]).cuda()
            inj = av.inject(v_sel.repeat_interleave(K_AV, 0), L_MAIN, torch.bfloat16)
            a = torch.ones(inj.shape[:2], dtype=torch.long, device="cuda")
            ids = bb.generate(inputs_embeds=inj, attention_mask=a, max_new_tokens=48,
                              do_sample=True, temperature=1.0, top_p=1.0,
                              pad_token_id=tok.pad_token_id)
            cand = tok.batch_decode(ids, skip_special_tokens=True)
            cand = [t.strip() or " " for t in cand]
            e2 = tok(cand, return_tensors="pt", padding=True, truncation=True,
                     max_length=96).to("cuda")
            o2 = bb(input_ids=e2.input_ids, attention_mask=e2.attention_mask,
                    output_hidden_states=True, use_cache=False)
            last2 = (e2.attention_mask.sum(1) - 1).clamp(min=0)
            h2 = o2.hidden_states[L_MAIN][torch.arange(len(cand), device="cuda"), last2].float()
            for bi, j in enumerate(av_rows):
                vj = vecs[L_MAIN][j]
                best_t, best_c = "", -1.0
                for k in range(K_AV):
                    cen = _cen_fve(h2[bi * K_AV + k], vj, mu_by[L_MAIN])
                    if cen > best_c:
                        best_c, best_t = cen, cand[bi * K_AV + k]
                if best_t.strip():
                    av_deco[j] = (best_t.strip(), best_c)

    seen = {}
    for L in LAYERS:
        for c in grid[L]:
            seen[c] = seen.get(c, 0) + 1

    # ---- 1. the answer, entropy-tinted ---------------------------------
    spans = []
    for p in range(n_in, T):
        e = ent[p - 1] if p - 1 < len(ent) else 0.0
        spans.append(f"<span title='entropy {e:.2f} nats' "
                     f"style='color:{_entropy_color(e)}'>{html.escape(toks[p])}</span>")
    text_html = (
        f"<div style='background:#101b36;border-radius:10px;padding:12px 16px'>"
        f"<div style='color:#8a9bb8;font-size:12px'>PROMPT</div>"
        f"<div style='margin-bottom:8px'>{html.escape(prompt)}</div>"
        f"<div style='color:#8a9bb8;font-size:12px'>MODEL ANSWER "
        f"<span style='font-size:11px'>(tint: <span style='color:rgb(30,140,220)'>confident</span>"
        f" to <span style='color:rgb(230,80,60)'>uncertain</span>)</span></div>"
        f"<div style='font-size:17px;line-height:1.75'>{''.join(spans)}</div></div>")

    # ---- 2. thought timeline (L18, every displayed step) ---------------
    cards = []
    for j, p in enumerate(cols):
        code = grid[L_MAIN][j]
        thought = (cb.decode(code) or "").strip()
        conf = cell_conf(L_MAIN, j)
        phase = "WRITING" if p >= n_in else "READING"
        ph_col = "#7ee0ff" if p >= n_in else "#b8a4ff"
        recur = "" if seen[code] <= 1 else (
            "<span title='the model re-visits this internal state' "
            "style='color:#7eebc0;font-size:10px'>&#8634; recurring</span>")
        lab = cb.label(code)
        raw_txt = html.escape(thought) if thought else "<i>(no canonical text)</i>"
        if lab:
            body = (
                f"<div style='font-size:10px;color:#8a9bb8'>state label (model-named)</div>"
                f"<div style='font-size:14px;font-weight:600;color:#e6ecf5'>{html.escape(lab)}</div>"
                f"<details style='margin-top:3px'><summary style='cursor:pointer;font-size:10px;"
                f"color:#8a9bb8'>nearest known state (raw retrieval text)</summary>"
                f"<div style='font-size:12px;color:#cdd8ea'>{raw_txt}</div></details>")
        else:
            body = (f"<div style='font-size:10px;color:#8a9bb8'>nearest known state (retrieval)</div>"
                    f"<div style='font-size:13px;color:#cdd8ea'>{raw_txt}</div>")
        if j in av_deco:
            at, ac = av_deco[j]
            acol, albl = _conf_band(ac)
            body += (
                f"<div style='margin-top:6px;border-top:1px dashed #26345a;padding-top:4px'>"
                f"<div style='font-size:10px;color:#e8c468'>&#10022; AV generated decode "
                f"(true NLA, best-of-{K_AV}) &mdash; "
                f"<span style='color:{acol}'>confidence {ac:.2f} - {albl}</span></div>"
                f"<div style='font-size:13px;font-style:italic;color:#e6ecf5'>{html.escape(at)}</div>"
                f"</div>")
        conf_html = _conf_bar(conf) if conf is not None else \
            "<div style='font-size:10px;color:#8a9bb8'>confidence n/a</div>"
        cards.append(
            f"<div style='display:flex;gap:12px;align-items:flex-start;background:#16213d;"
            f"border-left:4px solid {ph_col};border-radius:8px;padding:8px 12px;"
            f"margin-bottom:6px;max-width:860px'>"
            f"<div style='flex:0 0 130px'>"
            f"<div style='font-size:11px;color:{ph_col}'>{phase} - step {p}</div>"
            f"<div style='font-size:15px;font-weight:600;margin:2px 0'>&ldquo;{html.escape(toks[p][:14])}&rdquo;</div>"
            f"<div style='font-size:9px;color:#8a9bb8'>state #{code}"
            f"{(' &middot; ' + html.escape(lab[:24])) if lab else ''}</div>"
            f"<div>{recur}</div>"
            f"</div>"
            f"<div style='flex:1 1 auto;min-width:0'>"
            f"{body}"
            f"</div>"
            f"<div style='flex:0 0 170px'>{conf_html}</div>"
            f"</div>")
    timeline_html = (
        "<h4 style='margin:14px 0 4px'>What the model was thinking, step by step (layer 18)</h4>"
        "<p style='color:#8a9bb8;font-size:12px;margin:0 0 6px'>Each row decodes the hidden "
        "state at that token into language via the state codebook. The confidence bar is the "
        "measured round-trip fidelity of that decode: the text is re-encoded and compared "
        "back to the actual hidden state (centered metric; floor 0.50, retrieval baseline "
        "0.744, ceiling 1.0). Purple = reading your prompt, cyan = writing the answer.</p>"
        f"<div style='padding:6px 0'>{''.join(cards)}</div>")

    # ---- 3. depth map (all layers) --------------------------------------
    head = "<tr><th style='text-align:left;padding:3px 6px'>layer</th>" + "".join(
        f"<th style='padding:2px 3px;font-size:11px;color:#8a9bb8;"
        f"{'border-top:2px solid #7ee0ff;' if p >= n_in else ''}'>"
        f"{html.escape(toks[p][:6])}<br><span style='font-size:9px'>{p}</span></th>"
        for p in cols) + "</tr>"
    rows = []
    for L in reversed(LAYERS):
        cells = []
        for j, p in enumerate(cols):
            code = grid[L][j]
            conf = cell_conf(L, j)
            lab = cb.label(code)
            tip = html.escape((cb.decode(code) or "")[:240]) or "(no canonical text)"
            ctxt = f" - conf {conf:.2f}" if conf is not None else ""
            ltxt = f" [{html.escape(lab)}]" if lab else ""
            cell_lab = (f"<br><span style='font-size:8px;color:#cdd8ea'>"
                        f"{html.escape(lab[:12])}</span>" if lab else "")
            cells.append(
                f"<td title='state #{code}{ltxt}{ctxt} - {tip}' "
                f"style='background:{_code_color(code)};color:#e6ecf5;font-size:10px;"
                f"padding:4px 3px;text-align:center;border-radius:3px;cursor:help;"
                f"{'outline:2px solid #7eebc0;' if seen[code] > 1 else ''}'>"
                f"{code:04d}{cell_lab}</td>")
        rows.append(f"<tr><td style='color:#8a9bb8;padding:3px 6px'>L{L}</td>{''.join(cells)}</tr>")
    n_recur = sum(1 for c, n in seen.items() if n > 1)
    grid_html = (
        "<details style='margin-top:10px'><summary style='cursor:pointer;color:#8a9bb8'>"
        "Depth map - the same trace across all four layers (click to expand)</summary>"
        "<div style='overflow-x:auto;margin-top:6px'>"
        "<table style='border-collapse:separate;border-spacing:2px'>"
        + head + "".join(rows) + "</table></div>"
        f"<p style='color:#8a9bb8;font-size:12px'>Rows are layers (deepest on top). Each cell "
        f"is the state's magic number; hover for its decode + confidence. "
        f"<span style='color:#7eebc0'>Green ring</span> = recurring state ({n_recur} codes "
        f"recur). Cyan column border = generated tokens. Layer roles (external layer-sweep "
        f"finding on this architecture family): early layers separate <i>structure/register</i>, "
        f"later layers separate <i>meaning</i> — so L6 codes track form while L18/L24 codes "
        f"track content.</p></details>")

    return timeline_html + grid_html, text_html


@_gpu(duration=120)
def cb_verbalize(prompt: str, position: int, K: int):
    prompt = (prompt or "").strip()[:MAX_PROMPT_CHARS]
    if not prompt:
        return "<p>Run a trace first / enter a prompt.</p>"
    _load_everything()
    tok, bb, av, cb, mu = _STATE["tok"], _STATE["bb"], _STATE["av"], _STATE["cb"], _STATE["mu18"]

    enc = tok(prompt, return_tensors="pt").to("cuda")
    T = enc.input_ids.shape[1]
    p = int(max(0, min(position, T - 1)))
    with torch.no_grad():
        out = bb(**enc, output_hidden_states=True, use_cache=False)
        v = out.hidden_states[L_MAIN][0, p].float()

        # codebook retrieval
        code = int(cb.encode(v.unsqueeze(0))[0])
        ret_text = cb.decode(code) or "(empty)"
        renc = tok(ret_text, return_tensors="pt").to("cuda")
        rh = bb(**renc, output_hidden_states=True, use_cache=False).hidden_states[L_MAIN][0, -1].float()
        ret_cen = _cen_fve(rh, v, mu)

        # AV best-of-K with oracle rerank
        K = int(max(2, min(K, 16)))
        inj = av.inject(v.unsqueeze(0).expand(K, -1), L_MAIN, torch.bfloat16)
        attn = torch.ones(inj.shape[:2], dtype=torch.long, device="cuda")
        ids = bb.generate(inputs_embeds=inj, attention_mask=attn, max_new_tokens=96,
                          do_sample=True, temperature=1.0, top_p=1.0,
                          pad_token_id=tok.pad_token_id)
        cands = tok.batch_decode(ids, skip_special_tokens=True)
        scored = []
        for t in cands:
            t2 = t.strip() or " "
            e2 = tok(t2, return_tensors="pt").to("cuda")
            h2 = bb(**e2, output_hidden_states=True, use_cache=False).hidden_states[L_MAIN][0, -1].float()
            scored.append((t2, _cen_fve(h2, v, mu)))
        scored.sort(key=lambda r: r[1], reverse=True)

    def badge(cen: float) -> str:
        pct = max(0.0, min(1.0, (cen - FLOOR) / (NN_BASE - FLOOR))) * 100
        col = "#7eebc0" if cen >= NN_BASE else ("#7ee0ff" if pct > 50 else "#ff7eb9")
        return (f"<span style='background:{col};color:#0a1429;border-radius:8px;"
                f"padding:1px 8px;font-size:12px'>centered {cen:.3f} · "
                f"{pct:.0f}% of retrieval baseline</span>")

    tk = html.escape(tok.decode([int(enc.input_ids[0, p])]))
    vlab = cb.label(code)
    vlab_html = (f" &middot; <b>{html.escape(vlab)}</b> "
                 f"<span style='font-size:11px;color:#8a9bb8'>(model-named label)</span>"
                 if vlab else "")
    rows = "".join(
        f"<tr><td style='padding:4px'>{badge(c)}</td>"
        f"<td style='padding:4px'>{html.escape(t)}</td></tr>"
        for t, c in scored[:5])
    return (
        f"<h4>Position {p} (token \u201c{tk}\u201d) \u00b7 layer {L_MAIN}</h4>"
        f"<p><b>Codebook retrieval (recommended on this backbone)</b> — magic number "
        f"<code>{code}</code>{vlab_html} {badge(ret_cen)}<br>"
        f"<span style='font-size:14px'>{html.escape(ret_text)}</span></p>"
        f"<p><b>Activation Verbalizer, best-of-{K} (oracle rerank)</b></p>"
        f"<table>{rows}</table>"
        f"<p style='color:#8a9bb8;font-size:12px'>Anchors (centered fve, L18): random "
        f"floor {FLOOR} · NN retrieval {NN_BASE} · replay {REPLAY}. The released AV "
        f"tops out at 0.642 @ best-of-64 — below retrieval, which is why the codebook "
        f"is the primary decoder here.</p>")


HOWTO_TRACE = """
### How to read the trace

1. **The answer (top).** Your prompt plus the continuation gpt-oss-20b
   generates. Each generated token is tinted by **predictive entropy**:
   <span style='color:rgb(30,140,220)'>blue = confident</span>,
   <span style='color:rgb(230,80,60)'>red = uncertain</span> - the validated
   uncertainty signal.
2. **The thought timeline (middle) - the main event.** One card per step,
   spanning **reading your prompt (purple)** and **writing the answer (cyan)**.
   Each card shows: the token at that step, the model's decoded internal state
   ("what it was thinking"), and a **confidence bar** - the *measured*
   round-trip fidelity of that decode. The decoded text is re-encoded through
   the same frozen model and compared back to the actual hidden state
   (anisotropy-centered cosine; 0.50 = random floor, 0.744 = retrieval
   baseline, 1.0 = perfect). Long bar = trust the thought; short bar = the
   decode is a loose paraphrase of the state.
3. **The recurring mark** flags internal states the model re-visits at several
   steps - the ones that straddle the reading/writing boundary show what
   carries over from understanding your prompt into composing the answer.
4. **The depth map (bottom, collapsed).** The same trace across layers
   L6/L12/L18/L24 as a grid of "magic numbers" (4096-code VQ ids; same
   number = same state, same colour). Hover any cell for its decode +
   confidence. Deepest layer on top.

*Generate tokens* sets the continuation length (up to 256); *position stride*
thins the timeline so long generations stay readable.
"""

HOWTO_VERB = """
### How to read this tab

Pick a **token position in the prompt** (0-indexed; it is clamped to the
prompt length). The app grabs the hidden state at **layer 18** for that
position and decodes it two ways:

- **Codebook retrieval** — O(1) lookup of the state's magic number and its
  canonical text. On gpt-oss-20b this is the stronger decoder.
- **Activation Verbalizer, best-of-K** — K sampled verbalizations from the
  trained adapter; each is re-encoded and the list is ranked by round-trip
  fidelity (oracle rerank — legitimate at deploy time because the target
  vector is available by construction).

Every candidate carries a **centered-fidelity badge**: ½(1+cos(h−μ, v−μ)),
normalized against the published anchors — random floor **0.500**, zero-training
NN-retrieval baseline **0.744**, replay ceiling **0.999**. The percentage is
\"how far above the floor toward the retrieval baseline\";
<span style='color:#7eebc0'>green</span> means it beats retrieval,
<span style='color:#7ee0ff'>cyan</span> respectable,
<span style='color:#ff7eb9'>pink</span> weak. Raw (uncentered) scores are
never shown — on this backbone two *unrelated* states already score 0.837 raw.
"""

ABOUT_MD = f"""
## What this Space is

A frozen **`{BB_ID}`** narrating its own internals, end to end. Sibling of
[srt-showcase](https://huggingface.co/spaces/RiverRider/srt-showcase) (Qwen-2.5-7B),
extended with this backbone's new capabilities: the **full input→output trace**
(every layer × position, not just one vector), **magic-number state indexing**
(a 4096-code VQ codebook makes hidden states addressable integers), and
**retrieval decoding** with anisotropy-centered scoring.

## What is deterministic vs sampled

| Output | Deterministic? | Why |
|---|---|---|
| State codes / labels | yes | computed from hidden states, no sampling |
| Retrieval decodes + confidence | yes | nearest-centroid lookup + greedy re-encode |
| A vs B comparison | yes | forward passes only |
| Model answer | no | sampled (temp 0.7, top-p 0.95) |
| AV generated decodes | no | best-of-K sampling (oracle-reranked) |

## Honest metrics (all published)

**SRT adapter** ([model](https://huggingface.co/RiverRider/srt-adapter-gptoss20b)) —
held-out probe: regime **ECE 0.0009**, **AUROC 0.974**; r̂ Pearson 0.69; community NMI 0.42.

**AV K-curve** ([model](https://huggingface.co/RiverRider/srt-nla-av-gptoss20b)),
centered fve, 50 L18 targets:

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| best-of-K | .541 | .566 | .578 | .589 | .609 | .628 | .642 |

Anchors: floor **0.500** · NN retrieval **0.744** · replay **0.999**
([artifacts](https://huggingface.co/datasets/RiverRider/srt-nla-gptoss20b-artifacts)).

**The negative result is the point:** on gpt-oss-20b the trained verbalizer never
reaches the zero-training retrieval baseline (L18 anisotropy ‖μ‖≈4438, ~80×
Qwen's). So this demo decodes primarily via the codebook and shows the AV
comparison transparently. On Qwen-2.5-7B the same recipe *saturates* the
paraphrase ceiling — verbalizability is backbone-dependent.

Nothing here is a hallucination detector; entropy tinting is the validated
uncertainty signal, everything else is observational.
"""

_CSS = """
.gradio-container {background:#0a1429 !important; color:#e6ecf5;}
table {border-collapse:separate;}
/* examples/dataset tables inherit light-theme text; force readable colors */
.gradio-container .gr-samples-table, .gradio-container .gr-samples-table td,
.gradio-container [class*="samples"] td, .gradio-container [class*="samples"] th,
.gradio-container .dataset td, .gradio-container .dataset th,
.gradio-container [id*="dataset"] td, .gradio-container [id*="dataset"] th,
.gradio-container [id*="dataset"] button, .gradio-container [class*="example"] button,
.gradio-container [class*="example"] td {color:#e6ecf5 !important;}
"""


@_gpu(duration=120)
def cb_compare(pa: str, pb: str):
    """A vs B: does the model's internal STATE IDENTITY change between two
    prompts? Deterministic (forward passes only). Inspired by an external
    replication's finding that discrete state identity, not divergence
    magnitude, carries semantic-role structure (community changed for 50% of
    valid reversals vs 0% of nonsense reversals)."""
    pa = (pa or "").strip()[:MAX_PROMPT_CHARS]
    pb = (pb or "").strip()[:MAX_PROMPT_CHARS]
    if not pa or not pb:
        return "<p>Enter both prompts.</p>"
    _load_everything()
    tok, bb, cb, mu_by = _STATE["tok"], _STATE["bb"], _STATE["cb"], _STATE["mu_by"]
    with torch.no_grad():
        ha, hb = {}, {}
        for p, dest in ((pa, ha), (pb, hb)):
            e = tok(p, return_tensors="pt").to("cuda")
            o = bb(**e, output_hidden_states=True, use_cache=False)
            for L in LAYERS:
                dest[L] = o.hidden_states[L][0, -1].float()
    rows, n_changed = [], 0
    for L in reversed(LAYERS):
        ca = int(cb.encode(ha[L].unsqueeze(0))[0])
        cs = int(cb.encode(hb[L].unsqueeze(0))[0])
        cos = float(F.cosine_similarity((ha[L] - mu_by[L]).unsqueeze(0),
                                        (hb[L] - mu_by[L]).unsqueeze(0)))
        same = ca == cs
        n_changed += 0 if same else 1

        def _desc(code: int) -> str:
            lab = cb.label(code)
            if lab:
                return lab
            raw = (cb.decode(code) or "").strip()
            return ("\u201c" + raw[:60] + ("\u2026" if len(raw) > 60 else "") + "\u201d") \
                if raw else "(no canonical text)"

        la, lb = _desc(ca), _desc(cs)
        chip = ("<span style='color:#7eebc0'>same state</span>" if same
                else "<span style='color:#ff7eb9'>STATE CHANGED</span>")
        rows.append(
            f"<tr><td style='padding:4px 8px;color:#8a9bb8'>L{L}</td>"
            f"<td style='padding:4px 8px'>#{ca}<br><span style='font-size:11px;color:#cdd8ea'>{html.escape(la)}</span></td>"
            f"<td style='padding:4px 8px'>#{cs}<br><span style='font-size:11px;color:#cdd8ea'>{html.escape(lb)}</span></td>"
            f"<td style='padding:4px 8px'>{chip}</td>"
            f"<td style='padding:4px 8px'>{cos:.3f}</td></tr>")
    verdict = (f"<p><b>{n_changed}/{len(LAYERS)} layers changed state identity.</b> "
               "External replication on this architecture family found state-identity "
               "change tracks whether a reversal is semantically VALID (50% change) "
               "vs nonsense (0% change), while divergence magnitude alone does not "
               "discriminate. Centered cosine is the continuous similarity "
               "(1.0 = identical direction, 0 = unrelated).</p>")
    return (
        f"<h4>State identity: A vs B (last token, per layer)</h4>"
        f"<p style='font-size:12px;color:#8a9bb8'>A: {html.escape(pa)}<br>B: {html.escape(pb)}</p>"
        "<table><tr><th>layer</th><th>A state</th><th>B state</th><th>identity</th>"
        "<th>centered cos</th></tr>" + "".join(rows) + "</table>" + verdict
        + "<p style='font-size:11px;color:#8a9bb8'>Deterministic: forward passes only, no sampling.</p>")

with gr.Blocks(css=_CSS, title="SRT-NLA gpt-oss-20b trace") as app:
    gr.Markdown("# SRT-NLA · gpt-oss-20b — full input→output trace\n"
                "Magic-number state grid over every layer & token · codebook retrieval "
                "decoding · AV best-of-K with honest centered scoring")
    with gr.Tab("Full trace"):
        with gr.Accordion("How to read this demo", open=False):
            gr.Markdown(HOWTO_TRACE)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="The key difference between a virus and a bacterium is", lines=2)
        with gr.Row():
            max_new = gr.Slider(8, MAX_NEW_CAP, value=64, step=8, label="Generate tokens")
            stride = gr.Slider(1, 8, value=1, step=1, label="Position stride (1 = decode every token)")
            btn = gr.Button("Trace", variant="primary")
        text_html = gr.HTML()
        grid_html = gr.HTML()
        btn.click(cb_trace, [prompt, max_new, stride], [grid_html, text_html])
        gr.Examples(
            [["The key difference between a virus and a bacterium is", 64, 1],
             ["Q: If a train travels 60 miles in 45 minutes, its average speed is\nA:", 48, 1],
             ["I'm not entirely sure, but I believe the capital of Australia is", 32, 1],
             ["Whether the minimum wage should be raised is a question that", 48, 1],
             ["Yo dude, so like, the mitochondria is basically", 48, 1],
             ["I can't believe you did that. After everything we went through, you just", 48, 1],
             ["I miss her every single day. The house feels so empty now that", 48, 1],
             ["The word 'bank' can mean a financial institution or the side of a river, so", 48, 1],
             ["Some people insist the Earth is flat, but the evidence shows", 48, 1]],
            inputs=[prompt, max_new, stride],
            label=("factual · math · hedged/ambiguous · contested · register · affect-anger · "
                   "affect-sadness · word-sense · contested-factual"))
    with gr.Tab("Verbalize a position"):
        with gr.Accordion("How to read this tab", open=False):
            gr.Markdown(HOWTO_VERB)
        gr.Markdown("Pick a **token position in the prompt** and compare the codebook's "
                    "O(1) retrieval decode vs the AV's best-of-K, both centered-scored.")
        with gr.Row():
            vprompt = gr.Textbox(label="Prompt", value="The key difference between a virus and a bacterium is", lines=2)
            vpos = gr.Number(label="Token position (-clamped to range)", value=8, precision=0)
            vk = gr.Slider(2, 16, value=8, step=2, label="AV best-of-K")
        vbtn = gr.Button("Verbalize", variant="primary")
        vout = gr.HTML()
        vbtn.click(cb_verbalize, [vprompt, vpos, vk], [vout])
    with gr.Tab("Compare A vs B"):
        gr.Markdown(
            "**Does the model's internal state identity change between two prompts?** "
            "Swap a subject and object, flip a meaning, or shift the register — then "
            "see which layers change their magic-number state and by how much "
            "(centered cosine). Deterministic; no sampling.")
        with gr.Row():
            cpa = gr.Textbox(label="Prompt A", value="The Principal defines the principles.", lines=2)
            cpb = gr.Textbox(label="Prompt B", value="The principles define the Principal.", lines=2)
        cbtn = gr.Button("Compare", variant="primary")
        cout = gr.HTML()
        cbtn.click(cb_compare, [cpa, cpb], [cout])
        # NOTE: gr.Examples renders empty cells for two-Textbox example sets in
        # this gradio version; use explicit buttons instead.
        _cmp_pairs = [
            ("chirality", "The Principal defines the principles.", "The principles define the Principal."),
            ("nonsense reversal 1", "The rain wet the ground.", "The ground wet the rain."),
            ("nonsense reversal 2", "The fire melted the ice.", "The ice melted the fire."),
            ("valid reversal 1", "The culture shaped the values.", "The values shaped the culture."),
            ("valid reversal 2", "The teacher inspired the students.", "The students inspired the teacher."),
            ("word-sense (bank)", "The bank raised the interest rate.", "The river bank rose after the rain."),
            ("factual vs contested", "The Earth orbits the Sun.", "The Earth is flat and the Sun orbits it."),
            ("register shift", "The mitochondria is the powerhouse of the cell.",
             "Yo dude, the mitochondria is basically the cell's battery."),
            ("affect: happy vs angry", "I am so happy to see you again, my friend.",
             "I am so furious with you right now."),
            ("affect: sad vs friendly", "I miss her every single day since she left.",
             "It's wonderful catching up with you, this is so much fun."),
            ("word shuffle", "The quick brown fox jumps over the lazy dog.",
             "Dog lazy the over jumps fox brown quick the."),
            ("paraphrase (control)", "The meeting was moved to Thursday afternoon.",
             "They rescheduled the meeting for Thursday afternoon."),
        ]
        gr.Markdown("**Probe pairs** — click to load, then hit Compare:")
        for _i in range(0, len(_cmp_pairs), 4):
            with gr.Row():
                for _name, _a, _b in _cmp_pairs[_i:_i + 4]:
                    _btn = gr.Button(_name, size="sm")
                    _btn.click(lambda a=_a, b=_b: (a, b), None, [cpa, cpb])
    with gr.Tab("About & metrics"):
        gr.Markdown(ABOUT_MD)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
