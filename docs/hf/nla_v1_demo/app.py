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

# `spaces` must be imported BEFORE torch (and before gradio, which imports torch)
# so its fake-CUDA patches install. Otherwise `.to("cuda")` at module level
# triggers a real `torch._C._cuda_init()` which fails outside the GPU slot.
import spaces
import torch
import torch.nn.functional as F
import gradio as gr
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
MAX_NEW_TOKENS_DEFAULT = 128
MAX_NEW_TOKENS_LIMIT = 512  # ZeroGPU 180s budget: ~512 tok × BoN=16 fits

_state: dict[str, Any] = {}


def _strip_adapter_state(sd: Any) -> dict:
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


_paths: dict[str, str] = {}


def _prefetch() -> None:
    """Download every needed asset to disk at module import (no GPU needed).

    This is the ZeroGPU-friendly part of setup: pulls the AV adapter, μ, NLA
    config and the full Qwen2.5-7B snapshot into the HF hub cache so that the
    in-GPU-slot load is just shard mmap + .to('cuda') (~10s) instead of a 14GB
    download (~3 min, which previously consumed the entire 180s GPU budget).
    """
    if _paths:
        return
    from huggingface_hub import snapshot_download
    print("[nla] prefetching AV adapter + μ + config")
    _paths["cfg"] = hf_hub_download(MODEL_REPO, "config.json")
    _paths["weights"] = hf_hub_download(MODEL_REPO, "best_av.pt")
    _paths["mu"] = hf_hub_download(MODEL_REPO, "mu_l20.pt")
    cfg = NLAConfig.from_json(_paths["cfg"])
    _paths["backbone_id"] = cfg.backbone_id
    print(f"[nla] prefetching backbone {cfg.backbone_id} (this is the slow part)")
    snapshot_download(
        cfg.backbone_id,
        allow_patterns=[
            "*.json",
            "*.txt",
            "tokenizer*",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )
    print("[nla] prefetch complete")


def _setup_cpu() -> None:
    """Build model graph + load weights on CPU at module import.

    ZeroGPU does NOT intercept ``torch._C._cuda_init`` at module load, so any
    ``.to('cuda')`` outside an ``@spaces.GPU`` function raises ``Found no NVIDIA
    driver``. We therefore keep everything on CPU here, and move to GPU on the
    first request inside :func:`_ensure_gpu`.
    """
    if _state.get("cpu_ready"):
        return
    cfg = NLAConfig.from_json(_paths["cfg"])
    print(f"[nla] backbone={cfg.backbone_id} L={cfg.extraction_layer} "
          f"np={cfg.num_prefix_tokens} slots={cfg.num_inject_slots}")

    print("[nla] loading backbone on CPU (cached on disk)")
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.backbone_id,
        torch_dtype=torch.bfloat16,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = tok.pad_token_id

    print("[nla] loading adapter on CPU")
    av = ActivationVerbalizer(cfg, backbone, tok).eval()
    sd = torch.load(_paths["weights"], map_location="cpu", weights_only=False)
    sd = _strip_adapter_state(sd)
    missing, unexpected = av.load_state_dict(sd, strict=False)
    print(f"[nla] adapter loaded: missing={len(missing)} unexpected={len(unexpected)}")

    mu_blob = torch.load(_paths["mu"], map_location="cpu", weights_only=False)
    mu = mu_blob["mu"].float()
    print(f"[nla] mu loaded: ||mu||={float(mu.norm()):.4f}")

    _state.update({
        "cpu_ready": True,
        "gpu_ready": False,
        "device": torch.device("cpu"),
        "cfg": cfg,
        "tok": tok,
        "backbone": backbone,
        "av": av,
        "mu": mu,
    })
    print("[nla] cpu-ready (GPU move deferred to first request)")


def _ensure_gpu() -> None:
    """Move model + μ to CUDA. MUST be called from inside @spaces.GPU."""
    if _state.get("gpu_ready"):
        return
    device = torch.device("cuda")
    print("[nla] moving model to GPU (first request)")
    _state["backbone"] = _state["backbone"].to(device)
    _state["av"] = _state["av"].to(device)
    _state["mu"] = _state["mu"].to(device)
    _state["device"] = device
    _state["gpu_ready"] = True
    print("[nla] gpu-ready")


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
    _ensure_gpu()

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
    _ensure_gpu()

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


# ───────────────────────── Tab 3: steering ───────────────────────────

def _resolve_layer_module() -> torch.nn.Module:
    """Return the backbone block whose OUTPUT corresponds to
    `hidden_states[extraction_layer]` from `output_hidden_states=True`.
    With HF convention `hidden_states[0] = embeddings`, the i-th entry is
    the output of `model.model.layers[i-1]`."""
    cfg: NLAConfig = _state["cfg"]
    backbone = _state["backbone"]
    layers = backbone.model.layers  # type: ignore[attr-defined]
    idx = cfg.extraction_layer - 1
    return layers[idx]


def _make_steer_hook(steer_vec: torch.Tensor):
    """Forward hook adding `steer_vec` (shape (d,)) to all positions of the
    block's output hidden states. `steer_vec` must already be on-device and
    in the backbone's dtype."""

    def hook(_module, _inputs, output):
        # Qwen2 block output is a tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            h = output[0]
            h2 = h + steer_vec.to(h.dtype).to(h.device)
            return (h2,) + output[1:]
        h = output
        return h + steer_vec.to(h.dtype).to(h.device)

    return hook


@torch.no_grad()
def _generate_with_steering(prompt: str, steer_vec: torch.Tensor | None,
                            max_new: int) -> str:
    """Run greedy generation on the raw backbone. If `steer_vec` is not None
    it is added to L20 output at every forward pass."""
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]

    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=MAX_INPUT_TOKENS).to(device)

    handle = None
    if steer_vec is not None:
        layer = _resolve_layer_module()
        handle = layer.register_forward_hook(_make_steer_hook(steer_vec))
    try:
        out_ids = backbone.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=int(max_new),
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()
    # Strip the prompt from the decoded text.
    new_ids = out_ids[0, enc.input_ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


@spaces.GPU(duration=180)
def steer(prompt: str, text_a: str, text_b: str, alpha: float, max_new: int):
    """Activation-patching probe.

    Compute v_A = encode(text_A), v_B = encode(text_B) at L20 last-token.
    Add  alpha * (v_B - v_A)  to every L20 hidden state during a real
    greedy Qwen forward pass on `prompt`. Returns the baseline generation
    (no steering) and the steered generation side by side.
    """
    if not (prompt and prompt.strip()):
        return "", "", "Provide a prompt."
    if not (text_a and text_a.strip() and text_b and text_b.strip()):
        return "", "", "Provide both A and B anchors."
    _ensure_gpu()

    v_a = _encode_text(text_a.strip())  # (1, d) fp32
    v_b = _encode_text(text_b.strip())
    direction = (v_b - v_a).squeeze(0)  # (d,) fp32
    dir_norm = float(direction.norm())
    a = float(alpha)
    steer_vec = (a * direction).contiguous()

    baseline = _generate_with_steering(prompt, steer_vec=None, max_new=int(max_new))
    steered = _generate_with_steering(prompt, steer_vec=steer_vec, max_new=int(max_new))

    info = (
        f"**alpha = {a:.2f}** · steering direction = v_B − v_A, "
        f"||v_B − v_A|| = `{dir_norm:.3f}` · ||α·dir|| = `{a*dir_norm:.3f}`\n\n"
        f"Steering vector is added to **every position** of the L{_state['cfg'].extraction_layer} "
        "output during the real Qwen2.5-7B forward pass (no AV involved)."
    )
    return baseline.strip(), steered.strip(), info


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
                            max_new = gr.Slider(16, MAX_NEW_TOKENS_LIMIT, value=MAX_NEW_TOKENS_DEFAULT,
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
                        max_new_a = gr.Slider(16, MAX_NEW_TOKENS_LIMIT, value=MAX_NEW_TOKENS_DEFAULT,
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

            # ---------------- Tab 3 ----------------
            with gr.Tab("Activation steering"):
                gr.Markdown(
                    "Activation-patching probe. Encode two anchors A, B at L20 "
                    "→ steering direction **s = v_B − v_A**. Run a real greedy "
                    "Qwen2.5-7B forward pass on `prompt`, adding **α · s** to "
                    "every token's L20 hidden state. **The AV decoder is not "
                    "used here** — generation comes straight from the backbone. "
                    "Compares baseline (α=0) and steered output side by side."
                )
                with gr.Row():
                    with gr.Column():
                        s_prompt = gr.Textbox(label="Prompt to Qwen", lines=3,
                                              placeholder="e.g. How are you today?")
                        s_text_a = gr.Textbox(label="Anchor A", lines=3)
                        s_text_b = gr.Textbox(label="Anchor B", lines=3)
                        s_alpha = gr.Slider(-2.0, 2.0, value=1.0, step=0.05,
                                            label="α  (steering strength)")
                        s_max = gr.Slider(16, MAX_NEW_TOKENS_LIMIT,
                                          value=MAX_NEW_TOKENS_DEFAULT, step=16,
                                          label="Max new tokens")
                        go_s = gr.Button("Steer", variant="primary")
                    with gr.Column():
                        s_baseline = gr.Textbox(label="Baseline (α = 0)", lines=8)
                        s_steered = gr.Textbox(label="Steered output", lines=8)
                        s_info = gr.Markdown()
                go_s.click(steer,
                           inputs=[s_prompt, s_text_a, s_text_b, s_alpha, s_max],
                           outputs=[s_baseline, s_steered, s_info])

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


# Pre-download all assets + build the model graph on CPU at import. The
# CUDA move is deferred until the first @spaces.GPU call (see _ensure_gpu),
# because ZeroGPU does not stub out torch._C._cuda_init at module-load time.
_prefetch()
_setup_cpu()

if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=8).launch()
