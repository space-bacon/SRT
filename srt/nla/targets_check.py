"""Shared precondition checks + fingerprint for NLA target pools.

A whole iteration was wasted in May 2026 training on a degenerate target
pool where every sample collapsed to the same (~all-zeros) activation
because Qwen2.5's ``bos_token_id == eos_token_id`` caused the BOS slot
to be picked as the "last token" by ``sample_targets.py``. Loss
nonetheless decreased and ``fve_nrm`` plateaued at a fake ~0.62
because the AV learned to emit one constant string. Nothing in any
train script noticed.

These helpers fail loudly *before* the optimizer runs if the pool is
degenerate or constant-norm, and emit a one-line fingerprint
(sha + N + per-elem std + norm mean/std) into the train log header so
every run is traceable to the exact data version it consumed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger("nla.targets_check")


def fingerprint_targets(path: Path, pool: torch.Tensor) -> dict:
    """Compute sha256 (first 16 hex) + summary stats of the pool."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha16 = h.hexdigest()[:16]
    pool_f = pool.float()
    per_elem_std = float(pool_f.std(dim=0).mean().item())
    norms = pool_f.norm(dim=-1)
    return {
        "path": str(p),
        "sha16": sha16,
        "n": int(pool.size(0)),
        "d": int(pool.size(1)),
        "per_elem_std": per_elem_std,
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()),
    }


def assert_targets_healthy(
    path: Path,
    pool: torch.Tensor,
    *,
    min_per_elem_std: float = 0.1,
    min_norm_std: float = 1.0,
) -> dict:
    """Fingerprint + log + assert pool is non-degenerate. Returns fingerprint."""
    fp = fingerprint_targets(path, pool)
    logger.info(
        "TARGETS sha=%s N=%d d=%d per_elem_std=%.4f norm=%.3f±%.3f",
        fp["sha16"], fp["n"], fp["d"],
        fp["per_elem_std"], fp["norm_mean"], fp["norm_std"],
    )
    if fp["per_elem_std"] < min_per_elem_std:
        raise RuntimeError(
            f"DEGENERATE target pool {path}: per-elem std={fp['per_elem_std']:.4f} "
            f"< {min_per_elem_std}. Almost certainly a sampling bug (e.g. "
            f"BOS==EOS collapse). Refusing to train."
        )
    if fp["norm_std"] < min_norm_std:
        raise RuntimeError(
            f"CONSTANT-NORM target pool {path}: norm std={fp['norm_std']:.4f} "
            f"< {min_norm_std}. Targets are not diverse. Refusing to train."
        )
    return fp
