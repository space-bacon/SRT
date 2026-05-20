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


# ───────────── Tab 4: layer-scan + ablation + multi-pair ─────────────

@torch.no_grad()
def _encode_text_at_layer(text: str, layer: int) -> torch.Tensor:
    """Pooled last-token hidden at an arbitrary block. (1, d) fp32.
    layer is 1..N (1 = first block output, N = final block output)."""
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]
    enc = tok(text, truncation=True, max_length=MAX_INPUT_TOKENS,
              return_tensors="pt").to(device)
    out = backbone(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    h = out.hidden_states[layer]  # hidden_states[0]=embed, [layer]=output of layers[layer-1]
    last = enc.attention_mask.sum(-1) - 1
    idx = last.clamp(min=0).long()
    rows = torch.arange(h.size(0), device=device)
    return h[rows, idx, :].detach().to(torch.float32)


def _make_ablate_hook(unit_dir: torch.Tensor):
    """Project the direction out of every position's hidden state.
    unit_dir is (d,) fp32 on-device, unit norm.

    h'_t = h_t − (h_t · d̂) d̂
    """
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        d = unit_dir.to(h.dtype).to(h.device)
        # (B, T, d) · (d,) -> (B, T)
        proj = (h * d).sum(dim=-1, keepdim=True)
        h2 = h - proj * d
        if isinstance(output, tuple):
            return (h2,) + output[1:]
        return h2
    return hook


@torch.no_grad()
def _generate_with_hook(prompt: str, layer: int, hook_fn, max_new: int) -> str:
    """Greedy generation with `hook_fn` (or None) attached to layers[layer-1]."""
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=MAX_INPUT_TOKENS).to(device)
    handle = None
    if hook_fn is not None:
        layer_mod = backbone.model.layers[layer - 1]  # type: ignore[attr-defined]
        handle = layer_mod.register_forward_hook(hook_fn)
    try:
        out_ids = backbone.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=int(max_new), do_sample=False,
            temperature=1.0, top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def _mean_encode(texts: list[str], layer: int) -> torch.Tensor:
    """Difference-of-means anchor: mean of L<layer> last-token hiddens. (d,) fp32."""
    vs = [_encode_text_at_layer(t, layer).squeeze(0) for t in texts]
    return torch.stack(vs, dim=0).mean(dim=0)


@spaces.GPU(duration=180)
def steer_layer(prompt: str, texts_a: str, texts_b: str, alpha: float,
                layer: int, mode: str, max_new: int):
    """Layer-scan + ablation + multi-pair difference-of-means.

    `texts_a`, `texts_b`: pipe-separated lists. Empty entries are dropped.
    `mode`: 'add' = h + α·(μ_B − μ_A); 'ablate' = h − (h·d̂)·d̂.
    `layer`: 1..num_hidden_layers. Direction is computed at the same layer.
    """
    if not (prompt and prompt.strip()):
        return "", "", "Provide a prompt."
    backbone = _state.get("backbone")
    cfg = _state.get("cfg")
    if backbone is None or cfg is None:
        _ensure_gpu()
        backbone = _state["backbone"]; cfg = _state["cfg"]
    _ensure_gpu()
    n_layers = backbone.config.num_hidden_layers  # type: ignore[attr-defined]
    layer = int(layer)
    if not (1 <= layer <= n_layers):
        return "", "", f"layer must be in [1, {n_layers}]"

    A = [t.strip() for t in (texts_a or "").split("|") if t.strip()]
    B = [t.strip() for t in (texts_b or "").split("|") if t.strip()]
    if not A or not B:
        return "", "", "Provide at least one A and one B (pipe-separated for multi-pair)."

    mu_A = _mean_encode(A, layer)
    mu_B = _mean_encode(B, layer)
    direction = mu_B - mu_A
    dir_norm = float(direction.norm())
    if dir_norm < 1e-6:
        return "", "", "v_B − v_A is degenerate (zero norm)."

    baseline = _generate_with_hook(prompt, layer, None, int(max_new))

    if mode == "ablate":
        unit = (direction / dir_norm).contiguous()
        hook_fn = _make_ablate_hook(unit)
        a_used = 0.0
    else:
        steer_vec = (float(alpha) * direction).contiguous()
        def hook_fn(_module, _inputs, output, _sv=steer_vec):
            if isinstance(output, tuple):
                h = output[0]
                return (h + _sv.to(h.dtype).to(h.device),) + output[1:]
            return output + _sv.to(output.dtype).to(output.device)
        a_used = float(alpha)

    steered = _generate_with_hook(prompt, layer, hook_fn, int(max_new))

    info = (
        f"**layer={layer}/{n_layers}** · mode=**{mode}** · "
        f"|A|={len(A)} |B|={len(B)} · "
        f"||μ_B − μ_A|| = `{dir_norm:.3f}` · α·||dir|| = `{a_used*dir_norm:.3f}`\n\n"
        f"{'Ablating' if mode=='ablate' else 'Adding'} the direction "
        f"at every position of `layers[{layer-1}]` output during a real "
        f"Qwen2.5-7B greedy forward pass."
    )
    return baseline.strip(), steered.strip(), info


# ───────────────── Tab 5: geometry (direction quality) ───────────────

@torch.no_grad()
def _all_layer_last_token(text: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """One forward pass → last-token hidden at each requested layer."""
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]
    enc = tok(text, truncation=True, max_length=MAX_INPUT_TOKENS,
              return_tensors="pt").to(device)
    out = backbone(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    last = (enc.attention_mask.sum(-1) - 1).clamp(min=0).long()
    rows = torch.arange(enc.input_ids.size(0), device=device)
    return {L: out.hidden_states[L][rows, last, :].detach().to(torch.float32).squeeze(0)
            for L in layers}


@torch.no_grad()
def _all_layer_all_tokens(text: str, layers: list[int]) -> tuple[dict[int, torch.Tensor], int]:
    """One forward pass → (T, d) hidden at each layer; returns (dict, T)."""
    tok = _state["tok"]
    backbone = _state["backbone"]
    device = _state["device"]
    enc = tok(text, truncation=True, max_length=MAX_INPUT_TOKENS,
              return_tensors="pt").to(device)
    out = backbone(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    T = int(enc.attention_mask.sum().item())
    # hidden_states[L] is (1, T_full, d). Use attention_mask to take valid tokens.
    mask = enc.attention_mask[0].bool()
    return ({L: out.hidden_states[L][0, mask, :].detach().to(torch.float32)
             for L in layers}, T)


@spaces.GPU(duration=180)
def geometry(prompt: str, texts_a: str, texts_b: str, layers_str: str):
    """Per-layer geometric report on the (μ_B − μ_A) direction.

    For each requested layer L:
      - compute μ_A, μ_B over the A/B anchor banks, d_L = μ_B − μ_A
      - report ||d_L||, ||μ_A||, ||μ_B||
      - run `prompt` forward, get (T, d) at L
      - report mean_t |h_t · d̂_L| (alignment magnitude of residual stream
        with the direction, averaged over the prompt's tokens) and
        ||h_last||
    Also report the cosine matrix between d̂_L across the requested layers.
    """
    if not (prompt and prompt.strip()):
        return "Provide a prompt."
    _ensure_gpu()
    backbone = _state["backbone"]
    n_layers = backbone.config.num_hidden_layers  # type: ignore[attr-defined]

    try:
        layers = sorted({int(x.strip()) for x in layers_str.split(",") if x.strip()})
    except Exception:
        return "layers must be a comma-separated list of ints."
    layers = [L for L in layers if 1 <= L <= n_layers]
    if not layers:
        return f"no valid layers (allowed 1..{n_layers})"

    A = [t.strip() for t in (texts_a or "").split("|") if t.strip()]
    B = [t.strip() for t in (texts_b or "").split("|") if t.strip()]
    if not A or not B:
        return "Provide at least one A and one B anchor (pipe-separated)."

    # Compute μ_A, μ_B at each layer via individual forward passes (small N).
    a_stacks: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    b_stacks: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    for t in A:
        d = _all_layer_last_token(t, layers)
        for L in layers: a_stacks[L].append(d[L])
    for t in B:
        d = _all_layer_last_token(t, layers)
        for L in layers: b_stacks[L].append(d[L])

    mu_A = {L: torch.stack(a_stacks[L]).mean(0) for L in layers}
    mu_B = {L: torch.stack(b_stacks[L]).mean(0) for L in layers}
    d    = {L: (mu_B[L] - mu_A[L]) for L in layers}
    d_norm = {L: float(d[L].norm()) for L in layers}
    d_hat  = {L: (d[L] / (d_norm[L] + 1e-9)) for L in layers}

    # Prompt residual stream at each layer.
    h_dict, T = _all_layer_all_tokens(prompt.strip(), layers)
    per_layer = []
    for L in layers:
        h = h_dict[L]                # (T, d) fp32
        # projection scalars per token
        proj = (h * d_hat[L]).sum(dim=-1)        # (T,)
        proj_abs_mean = float(proj.abs().mean())
        proj_signed_mean = float(proj.mean())
        h_norms = h.norm(dim=-1)                 # (T,)
        h_norm_mean = float(h_norms.mean())
        # cosine of last-token h with d_hat
        h_last = h[-1]
        cos_last = float(F.cosine_similarity(h_last.unsqueeze(0),
                                             d_hat[L].unsqueeze(0), dim=-1))
        per_layer.append({
            "L": L,
            "d_norm": d_norm[L],
            "mu_A_norm": float(mu_A[L].norm()),
            "mu_B_norm": float(mu_B[L].norm()),
            "h_norm_mean": h_norm_mean,
            "proj_abs_mean": proj_abs_mean,
            "proj_signed_mean": proj_signed_mean,
            "frac_of_hnorm": proj_abs_mean / max(h_norm_mean, 1e-9),
            "cos_last_token": cos_last,
        })

    # Cross-layer cosine matrix on d_hat.
    cos_matrix = []
    for L1 in layers:
        row = []
        for L2 in layers:
            row.append(float(F.cosine_similarity(d_hat[L1].unsqueeze(0),
                                                 d_hat[L2].unsqueeze(0), dim=-1)))
        cos_matrix.append(row)

    # Render markdown.
    lines = [
        f"**Prompt tokens:** T = {T} · **|A|** = {len(A)} · **|B|** = {len(B)}",
        "",
        "### Per-layer direction quality & residual stream alignment",
        "",
        "| L | ‖d_L‖ | ‖μ_A‖ | ‖μ_B‖ | mean‖h_t‖ | mean<sub>t</sub>|h·d̂| | as frac of ‖h‖ | cos(h_last, d̂) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_layer:
        lines.append(
            f"| {r['L']} | `{r['d_norm']:.2f}` | `{r['mu_A_norm']:.2f}` | "
            f"`{r['mu_B_norm']:.2f}` | `{r['h_norm_mean']:.2f}` | "
            f"`{r['proj_abs_mean']:.3f}` | `{r['frac_of_hnorm']*100:.2f}%` | "
            f"`{r['cos_last_token']:+.4f}` |"
        )
    lines += [
        "",
        "### Cross-layer direction cosine matrix · cos(d̂_L, d̂_L′)",
        "",
        "| L \\ L' | " + " | ".join(str(L) for L in layers) + " |",
        "|" + "---|" * (len(layers) + 1),
    ]
    for L1, row in zip(layers, cos_matrix):
        lines.append(
            "| **" + str(L1) + "** | " +
            " | ".join(f"`{v:+.3f}`" for v in row) + " |"
        )
    return "\n".join(lines)


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

            # ---------------- Tab 4 ----------------
            with gr.Tab("Layer scan / ablation"):
                gr.Markdown(
                    "Generalised activation patching: choose **any layer** "
                    "(1..28 for Qwen2.5-7B), choose **mode** = `add` (h + α·d) "
                    "or `ablate` (project d̂ out of h), and use **multi-pair** "
                    "anchors by separating with `|`. Direction is computed at "
                    "the same layer being patched (difference-of-means: "
                    "`μ_B − μ_A`).\n\n"
                    "_Tab 3 is the single-layer single-pair specialisation of this._"
                )
                with gr.Row():
                    with gr.Column():
                        l_prompt = gr.Textbox(label="Prompt to Qwen", lines=3)
                        l_texts_a = gr.Textbox(
                            label="Anchors A  (pipe-separated for multi-pair)",
                            lines=4,
                            placeholder="text1 | text2 | text3",
                        )
                        l_texts_b = gr.Textbox(
                            label="Anchors B  (pipe-separated for multi-pair)",
                            lines=4,
                            placeholder="text1 | text2 | text3",
                        )
                        l_alpha = gr.Slider(-2.0, 2.0, value=0.10, step=0.01,
                                            label="α  (used only when mode='add')")
                        l_layer = gr.Slider(1, 28, value=20, step=1,
                                            label="Layer (1..28)")
                        l_mode = gr.Radio(["add", "ablate"], value="add",
                                          label="Mode")
                        l_max = gr.Slider(16, MAX_NEW_TOKENS_LIMIT,
                                          value=MAX_NEW_TOKENS_DEFAULT,
                                          step=16, label="Max new tokens")
                        go_l = gr.Button("Run", variant="primary")
                    with gr.Column():
                        l_baseline = gr.Textbox(label="Baseline (no hook)", lines=8)
                        l_steered = gr.Textbox(label="Patched output", lines=8)
                        l_info = gr.Markdown()
                go_l.click(steer_layer,
                           inputs=[l_prompt, l_texts_a, l_texts_b,
                                   l_alpha, l_layer, l_mode, l_max],
                           outputs=[l_baseline, l_steered, l_info])

            # ---------------- Tab 5 ----------------
            with gr.Tab("Geometry"):
                gr.Markdown(
                    "**Geometric report on the (μ_B − μ_A) direction.** "
                    "For each requested layer L this computes "
                    "‖d_L‖, the mean magnitude of the prompt's residual-stream "
                    "projection onto d̂_L (averaged over tokens), and the "
                    "cosine matrix between d̂_L across layers (does the "
                    "direction stay fixed through the network?).\n\n"
                    "_If `mean|h·d̂|` is near zero, the model isn't using "
                    "the direction at that layer — which is exactly why "
                    "ablation does nothing._"
                )
                with gr.Row():
                    with gr.Column():
                        g_prompt = gr.Textbox(label="Prompt", lines=3)
                        g_texts_a = gr.Textbox(
                            label="Anchors A (pipe-separated)", lines=4)
                        g_texts_b = gr.Textbox(
                            label="Anchors B (pipe-separated)", lines=4)
                        g_layers = gr.Textbox(
                            label="Layers (comma-separated, 1..28)",
                            value="4, 8, 12, 16, 20, 24, 28")
                        go_g = gr.Button("Measure", variant="primary")
                    with gr.Column():
                        g_report = gr.Markdown()
                go_g.click(geometry,
                           inputs=[g_prompt, g_texts_a, g_texts_b, g_layers],
                           outputs=[g_report])

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
