"""Mean recalibration: the 42KB file that makes cross-runtime honest.

Different runtimes (CUDA bf16, MLX 4-bit, NF4) displace the modality
means of the raw states. The head is robust to everything else, but the
image branch is ~17x more mean-sensitive than text: using a shipped
mu_img on MLX-Q4 image queries costs ~24 R@1 points. The fix is
measured, tiny, and one command: average ~256 raw states from YOUR
runtime over YOUR data, and use those means for centering.

    sunstone-sidecar calibrate --images ./sample_dir --out my_runtime.npz
    side.load_calibration("my_runtime.npz")

Validated: MLX-Q4 vs CUDA bf16 goes 0.964 -> 0.967 head-space R@1 on
text self-retrieval, and image retrieval recovers the full ~24 points
(artifacts/nla/q4/head_space_validation_20260805.json and the v31 arc).
"""
from __future__ import annotations

import numpy as np

RECOMMENDED_N = 256


def measure_means(tap, images=None, texts=None) -> dict:
    """Average RAW (pre-head) states from the caller's runtime.

    images: iterable of PIL images or paths; texts: list of strings.
    Either or both. ~256 samples is enough; means are head-independent
    and survive head swaps.
    """
    out: dict = {}
    if images is not None:
        from PIL import Image
        vs = []
        for it in images:
            if isinstance(it, (str, bytes)) or hasattr(it, "__fspath__"):
                img = Image.open(it).convert("RGB")
            else:
                img = it
            vs.append(np.asarray(tap.image_vector(img), dtype=np.float64))
        if vs:
            out["mu_img"] = np.mean(vs, axis=0).astype(np.float32)
            out["n_img"] = len(vs)
    if texts:
        v = np.asarray(tap.text_vectors(list(texts)), dtype=np.float64)
        out["mu_txt"] = v.mean(0).astype(np.float32)
        out["n_txt"] = len(texts)
    if not out:
        raise ValueError("give images and/or texts to calibrate on")
    return out


def save_calibration(cal: dict, path: str) -> None:
    np.savez(path, **cal)


def load_calibration(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def apply_calibration(head: dict, cal: dict) -> dict:
    """Return a copy of the head with locally-measured means swapped in."""
    h = dict(head)
    if "mu_img" in cal and cal["mu_img"].shape == head["mu_img"].shape:
        h["mu_img"] = cal["mu_img"]
    if "mu_txt" in cal and cal["mu_txt"].shape == head["mu_txt"].shape:
        h["mu_txt"] = cal["mu_txt"]
    return h
