"""SRT-NLA v1 demo Space — round-trip autoencoder + latent arithmetic.

Two tabs:

Tab 1 — Round-trip autoencoder:
    text  -- AR (frozen Qwen L20) -->  v  --  AV (this model)  -->  text'
    Score: fve_nrm_centered(v, AR(text')), rho_norm against published anchors.
    Best-of-N rerank by AR fidelity.

Tab 2 — Latent arithmetic:
    Encode text A and text B, interpolate v = (1-a)*v_A + a*v_B, verbalize.

Public model: RiverRider/srt-nla-av-v1
Public assets pulled at runtime:
    - best_av.pt    (12.7M adapter weights)
    - config.json   (NLAConfig)
    - mu_l20.pt     (anisotropy mean for centered fve_nrm)
Backbone: Qwen/Qwen2.5-7B (bf16, frozen).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer

MODEL_REPO = "RiverRider/srt-nla-av-v1"

# Published anchors (paper §3, artifacts/nla/oracle_ceiling_30k_v2.json).
RANDOM_FLOOR_CEN = 0.510
PARAPHRASE_CEILING_CEN = 0.799
RHO_DENOM = PARAPHRASE_CEILING_CEN - RANDOM_FLOOR_CEN

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS_DEFAULT = 64

_state: dict[str, Any] = {}


def _strip_adapter_state(sd: Any) -> dict:
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def _setup(device: torch.device) -> None:
    if _state.get("ready"):
        return
    print("[nla] downloading model assets")
    cfg_path = hf_hub_download(MODEL_REPO, "config.json")
    weights_path = hf_hub_download(MODEL_REPO, "best_av.pt")
    mu_path = hf_hub_download(MODEL_REPO, "mu_l20.pt")

    cfg = NLAConfig.from_json(cfg_path)
    print(f"[nla] backbone={cfg.backbone_id} L={cfg.extraction_layer} "
          f"np={cfg.num_prefix_tokens} slots={cfg.num_inject_slots}")

    print("[nla] loading backbone")
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = tok.pad_token_id

    print("[nla] loading adapter")
    av = ActivationVerbalizer(cfg, backbone, tok).to(device).eval()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = _strip_adapter_state(sd)
    missing, unexpected = av.load_state_dict(sd, strict=False)
    print(f"[nla] adapter loaded: missing={len(missing)} unexpected={len(unexpected)}")

    mu_blob = torch.load(mu_path, map_location="cpu", weights_only=False)
    mu = mu_blob["mu"].float().to(device)
    print(f"[nla] mu loaded: ||mu||={float(mu.norm()):.4f}")

    _state.update({
        "ready": True,
        "device": device,
        "cfg": cfg,
        "tok": tok,
        "backbone": backbone,
        "av": av,
        "mu": mu,
    })
    print("[nla] ready")


@torch.no_grad()
def _encode_text(text: str) -> torch.Tensor:
    """Run frozen Qwen forward, return pooled L20 last-token hidden (1, d) fp32."""
    cfg: NLAConfig = _state["cfg"]
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]
    enc = tok(
        text,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt",
    ).to(device)
    out = backbone(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[cfg.extraction_layer]
    last = enc.attention_mask.sum(-1) - 1
    idx = last.clamp(min=0).long()
    rows = torch.arange(h.size(0), device=device)
    return h[rows, idx, :].detach().to(torch.float32)


@torch.no_grad()
def _verbalize(v: torch.Tensor, n: int, max_new: int, temperature: float) -> list[str]:
    """Decode v into `n` candidate strings. v shape (1, d)."""
    av: ActivationVerbalizer = _state["av"]
    tok = _state["tok"]
    if n <= 1:
        ids = av.generate(
            v,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        return tok.batch_decode(ids, skip_special_tokens=True)
    v_rep = v.expand(n, -1).contiguous()
    ids = av.generate(
        v_rep,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
    )
    return tok.batch_decode(ids, skip_special_tokens=True)


def _fve_cen(h: torch.Tensor, v: torch.Tensor) -> float:
    mu = _state["mu"]
    return float(0.5 * (1.0 + F.cosine_similarity(
        (h - mu).float(), (v - mu).float(), dim=-1
    )))


def _rho(cen: float) -> float:
    return (cen - RANDOM_FLOOR_CEN) / RHO_DENOM


def _score_band(rho: float) -> str:
    if rho >= 0.80:
        return "near paraphrase ceiling"
    if rho >= 0.50:
        return "well above NN-retrieval baseline"
    if rho >= 0.20:
        return "above random floor"
    return "near random floor"


# ───────────────────────── Tab 1: round-trip ─────────────────────────

@spaces.GPU(duration=180)
def roundtrip(passage: str, n_samples: int, max_new: int, temperature: float):
    if not passage or not passage.strip():
        return "", "", "Enter a passage."
    device = torch.device("cuda")
    _setup(device)

    v = _encode_text(passage.strip())  # (1, d)

    n = max(1, min(int(n_samples), 16))
    candidates = _verbalize(v, n=n, max_new=int(max_new), temperature=float(temperature))

    # Re-encode each candidate and pick the one maximizing centered fve_nrm.
    scored: list[tuple[float, float, str]] = []
    for text in candidates:
        if not text.strip():
            continue
        h = _encode_text(text)
        cen = _fve_cen(h, v)
        rho = _rho(cen)
        scored.append((cen, rho, text))
    if not scored:
        return "", "", "AV produced only empty strings; try lowering temperature."
    scored.sort(key=lambda r: r[0], reverse=True)

    best_cen, best_rho, best_text = scored[0]

    lines = [
        f"**Best-of-{n}** (reranked by AR fidelity):",
        f"- centered fve_nrm: **{best_cen:.3f}**",
        f"- rho_norm:        **{best_rho:+.2f}**  _( {_score_band(best_rho)} )_",
        "",
        "_Anchors_: random floor `0.510` · NN-retrieval `0.71` ·"
        " paraphrase ceiling `0.799`.",
    ]
    if n > 1:
        lines += ["", f"Candidate spread: cen ∈ [{scored[-1][0]:.3f}, {scored[0][0]:.3f}]"]
        lines += ["", "Other candidates:"]
        for cen, rho, text in scored[1:5]:
            preview = text.strip().replace("\n", " ")[:160]
            lines.append(f"- `cen={cen:.3f}  rho={rho:+.2f}`  {preview}")

    return best_text.strip(), "\n".join(lines), ""


# ───────────────────────── Tab 2: arithmetic ─────────────────────────

@spaces.GPU(duration=180)
def arithmetic(text_a: str, text_b: str, alpha: float, max_new: int):
    if not (text_a and text_a.strip() and text_b and text_b.strip()):
        return "", "Provide both texts."
    device = torch.device("cuda")
    _setup(device)

    v_a = _encode_text(text_a.strip())
    v_b = _encode_text(text_b.strip())
    a = float(alpha)
    v = (1.0 - a) * v_a + a * v_b

    [text] = _verbalize(v, n=1, max_new=int(max_new), temperature=1.0)
    h = _encode_text(text)
    cen_a = _fve_cen(h, v_a)
    cen_b = _fve_cen(h, v_b)
    cen_v = _fve_cen(h, v)
    info = (
        f"**alpha = {a:.2f}**  →  v = (1-α)·A + α·B\n\n"
        f"- centered fve_nrm to **A**: `{cen_a:.3f}`  (rho `{_rho(cen_a):+.2f}`)\n"
        f"- centered fve_nrm to **B**: `{cen_b:.3f}`  (rho `{_rho(cen_b):+.2f}`)\n"
        f"- centered fve_nrm to **mix v**: `{cen_v:.3f}`  (rho `{_rho(cen_v):+.2f}`)\n"
    )
    return text.strip(), info


# ───────────────────────────── UI ────────────────────────────────────

EXAMPLES_RT = [
    ["The Roman Senate convened in the Curia Julia to debate the future of the republic, weighing the consuls' authority against the rising influence of the equestrian class.", 4, 64, 0.9],
    ["A black hole's event horizon marks the boundary beyond which no information can escape; an outside observer sees infalling matter freeze and redshift toward it.", 4, 64, 0.9],
    ["def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])", 4, 64, 0.9],
]

EXAMPLES_AR = [
    [
        "The trial dragged on for weeks; jurors grew restless and the judge's patience wore visibly thin.",
        "After the ceremony, the wedding party spilled into the gardens and danced until dawn.",
        0.5,
        64,
    ],
    [
        "Quantum entanglement implies non-local correlations between particle pairs.",
        "Classical thermodynamics treats heat as a flow between bodies at different temperatures.",
        0.5,
        64,
    ],
]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SRT-NLA v1 — latent autoencoder demo",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SRT-NLA v1 — latent autoencoder demo\n"
            "Round-trip a passage through Qwen2.5-7B's residual stream at L20: "
            "the **encoder** is a frozen Qwen forward, the **decoder** is the "
            "12.7M-parameter NLA verbalizer from "
            "[`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1).\n\n"
            "_Score: centered fve_nrm  =  ½(1 + cos(h−μ, v−μ)),  with anchors "
            "random=0.510, NN-retrieval=0.71, paraphrase ceiling=0.799._\n\n"
            "_First request loads the backbone (~20s on ZeroGPU). Subsequent requests are fast._"
        )
        with gr.Tabs():
            # ---------------- Tab 1 ----------------
            with gr.Tab("Round-trip autoencoder"):
                gr.Markdown(
                    "Pipeline: `text → Qwen L20 hidden v → AV decoder → text' → re-encode → score`.\n"
                    "Best-of-N samples are reranked by AR fidelity (this is how the paper "
                    "gets greedy ρ=0.26 → BoN=0.92)."
                )
                with gr.Row():
                    with gr.Column():
                        passage = gr.Textbox(
                            label="Source passage",
                            lines=6,
                            placeholder="Paste 1-3 sentences...",
                        )
                        with gr.Row():
                            n_samples = gr.Slider(1, 16, value=4, step=1,
                                                  label="Best-of-N samples")
                            max_new = gr.Slider(16, 128, value=MAX_NEW_TOKENS_DEFAULT,
                                                step=16, label="Max new tokens")
                            temperature = gr.Slider(0.3, 1.5, value=0.9, step=0.05,
                                                    label="Temperature")
                        go = gr.Button("Round-trip", variant="primary")
                    with gr.Column():
                        rewrite = gr.Textbox(label="AV verbalization (best of N)",
                                             lines=6)
                        info = gr.Markdown()
                        err = gr.Markdown()
                gr.Examples(examples=EXAMPLES_RT,
                            inputs=[passage, n_samples, max_new, temperature])
                go.click(roundtrip,
                         inputs=[passage, n_samples, max_new, temperature],
                         outputs=[rewrite, info, err])

            # ---------------- Tab 2 ----------------
            with gr.Tab("Latent arithmetic"):
                gr.Markdown(
                    "Encode two passages → v_A, v_B in R^3584. "
                    "Verbalize  `v = (1-α)·v_A + α·v_B`.\n"
                    "α = 0 reproduces A; α = 1 reproduces B; in between the AV "
                    "shows what the L20 mixture decodes to."
                )
                with gr.Row():
                    with gr.Column():
                        text_a = gr.Textbox(label="Passage A", lines=4)
                        text_b = gr.Textbox(label="Passage B", lines=4)
                        alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                          label="α  (0 = A, 1 = B)")
                        max_new_a = gr.Slider(16, 128, value=MAX_NEW_TOKENS_DEFAULT,
                                              step=16, label="Max new tokens")
                        go_a = gr.Button("Verbalize mix", variant="primary")
                    with gr.Column():
                        mix_out = gr.Textbox(label="AV verbalization of mix",
                                             lines=6)
                        info_a = gr.Markdown()
                gr.Examples(examples=EXAMPLES_AR,
                            inputs=[text_a, text_b, alpha, max_new_a])
                go_a.click(arithmetic,
                           inputs=[text_a, text_b, alpha, max_new_a],
                           outputs=[mix_out, info_a])

        gr.Markdown(
            "---\n"
            "**About.** This demo runs the published `srt-nla-av-v1` checkpoint "
            "(12.7M trainable params on top of frozen Qwen/Qwen2.5-7B). "
            "Encoder = frozen Qwen L20 last-token, decoder = injection-prefix "
            "verbalizer (16 learned prefix tokens, 1 inject slot). "
            "All metrics use the published anisotropy mean μ "
            "(`||μ|| = 55.178`, derived from 2000 held-out targets).\n\n"
            "Code & paper: https://github.com/space-bacon/SRT (branch `nla`)."
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=8).launch()
