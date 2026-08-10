"""Head registry: which linear head reads which backbone at which layer.

A head is a dict: {img: {weight, bias}, txt: {weight, bias},
mu_img, mu_txt, meta}. Score = cos(Wi(x - mu_img) + bi, Wt(t - mu_txt) + bt).
All heads are public on the HF Hub; total download ~44MB each.
"""
from __future__ import annotations

import numpy as np
import torch

HEAD_REGISTRY: dict[str, dict] = {
    # shipped stable head for gemma-4-31B (v3: drift-family trained, best
    # cross-runtime behavior; see RiverRider/srt-sunstone-linear-head card)
    "google/gemma-4-31B-it": {
        "repo": "RiverRider/srt-sunstone-linear-head",
        "file": "sunstone_linear_head_v3_drift.pt",
        "layer": 47,
        "d": 5376,
    },
    # scale-floor head for the 3B tier (identical recipe, 10x smaller host)
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "repo": "RiverRider/srt-nla-gemma4-artifacts",
        "file": "scalefloor/qwen3b_linear_head.pt",
        "layer": 29,
        "d": 2048,
    },
}

# native 4-bit variant (NF4-trained); the bf16 heads also work unchanged
# on quantized backbones within ~1 R@1 point (artifacts/nla/q4/q4_drift.json)
HEAD_REGISTRY["Qwen/Qwen2.5-VL-3B-Instruct+q4"] = {
    "repo": "RiverRider/srt-nla-gemma4-artifacts",
    "file": "q4/qwen3b_q4_head.pt",
    "layer": 29,
    "d": 2048,
}


def load_head(backbone: str, quant4: bool = False) -> dict:
    """Download (cached) and normalize a head for `backbone`.

    Returns numpy arrays ready for the scoring path:
    {W_img, b_img, W_txt, b_txt, mu_img, mu_txt, layer, d, proj_dim}.
    """
    from huggingface_hub import hf_hub_download

    key = backbone + "+q4" if quant4 and backbone + "+q4" in HEAD_REGISTRY \
        else backbone
    if key not in HEAD_REGISTRY:
        raise KeyError(
            f"no head registered for {backbone!r}; known: "
            f"{sorted(HEAD_REGISTRY)}. Custom heads: sunstonenorth.com")
    spec = HEAD_REGISTRY[key]
    raw = torch.load(hf_hub_download(spec["repo"], spec["file"]),
                     map_location="cpu", weights_only=True)
    head = {
        "W_img": raw["img"]["weight"].float().numpy(),
        "b_img": raw["img"]["bias"].float().numpy(),
        "W_txt": raw["txt"]["weight"].float().numpy(),
        "b_txt": raw["txt"]["bias"].float().numpy(),
        "mu_img": raw["mu_img"].float().numpy(),
        "mu_txt": raw["mu_txt"].float().numpy(),
        "layer": spec["layer"],
        "d": spec["d"],
    }
    head["proj_dim"] = head["W_img"].shape[0]
    return head


def project(v: np.ndarray, W: np.ndarray, b: np.ndarray,
            mu: np.ndarray) -> np.ndarray:
    """Center, project, L2-normalize. Works on (d,) or (n, d)."""
    z = (v - mu) @ W.T + b
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / np.maximum(n, 1e-8)
