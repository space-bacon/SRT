"""Linear weight-average two adapter checkpoints.

Usage:
    python scripts/soup_adapters.py \
        --ckpt-a artifacts/checkpoints/v18/best_adapter.pt \
        --ckpt-b artifacts/checkpoints/v20/best_adapter.pt \
        --alpha 0.7 \
        --out artifacts/checkpoints/v21b_a070/best_adapter.pt

Output state_dict = alpha * A + (1 - alpha) * B per parameter tensor.
Both checkpoints must share the exact same keys and shapes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def _extract_state_dict(blob: dict | "torch.nn.Module") -> dict:
    if isinstance(blob, dict):
        # Common wrappers: {"adapter": ..., "model": ..., "state_dict": ...}
        for key in ("adapter", "model", "state_dict", "module"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
        return blob
    raise TypeError(f"Unexpected checkpoint type: {type(blob)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--alpha", type=float, required=True, help="Weight on A; B gets (1 - alpha)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a_blob = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    b_blob = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)

    a_sd = _extract_state_dict(a_blob)
    b_sd = _extract_state_dict(b_blob)

    if a_sd.keys() != b_sd.keys():
        only_a = set(a_sd) - set(b_sd)
        only_b = set(b_sd) - set(a_sd)
        raise ValueError(f"Key mismatch. only_a={sorted(only_a)[:5]} only_b={sorted(only_b)[:5]}")

    alpha = args.alpha
    soup: dict[str, torch.Tensor] = {}
    for k, va in a_sd.items():
        vb = b_sd[k]
        if not torch.is_tensor(va) or not torch.is_tensor(vb):
            soup[k] = va
            continue
        if va.shape != vb.shape:
            raise ValueError(f"Shape mismatch for {k}: {va.shape} vs {vb.shape}")
        soup[k] = (alpha * va.float() + (1.0 - alpha) * vb.float()).to(va.dtype)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the wrapper structure of A if it had one.
    if isinstance(a_blob, dict) and any(k in a_blob for k in ("adapter", "model", "state_dict", "module")):
        wrapper_key = next(k for k in ("adapter", "model", "state_dict", "module") if k in a_blob and isinstance(a_blob[k], dict))
        out_blob = dict(a_blob)
        out_blob[wrapper_key] = soup
        torch.save(out_blob, out)
    else:
        torch.save(soup, out)

    print(f"Wrote souped checkpoint: {out} (alpha_v18={alpha:.3f}, alpha_v20={1 - alpha:.3f})")
    print(f"  parameters: {sum(t.numel() for t in soup.values() if torch.is_tensor(t)):,}")


if __name__ == "__main__":
    main()
