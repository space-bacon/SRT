"""Canonical NLA evaluation metrics.

Single source of truth for fve_nrm, the centered (anisotropy-corrected)
variant, and the rho_norm normalization used in paper_nla.md.

Previously these helpers were duplicated in 6+ scripts. New code should
import from here. The duplicates remain in legacy scripts for now to
preserve bit-exact reproducibility of historical artifacts/nla/*.json.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Anchors from artifacts/nla/oracle_ceiling_30k_v2.json (Qwen2.5-7B L20,
# 200 held-out targets, pool=2000). Used to normalize centered fve_nrm
# into rho_norm ∈ [0, 1].
RANDOM_FLOOR_CEN: float = 0.510
PARAPHRASE_CEILING_CEN: float = 0.799
RHO_DENOM: float = PARAPHRASE_CEILING_CEN - RANDOM_FLOOR_CEN  # 0.289


def fve_nrm(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """fve_nrm(h, v) = 0.5 * (1 + cos(h, v))  in [0, 1].

    Linear remap of cosine similarity. Operates on the last dim.
    Shapes: h, v broadcast-compatible; returns batch shape.
    """
    return 0.5 * (1.0 + F.cosine_similarity(h.float(), v.float(), dim=-1))


def fve_nrm_centered(
    h: torch.Tensor, v: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """Anisotropy-corrected fve_nrm: subtract pool mean mu before cosine.

    mu is the (d,) mean activation of the held-out pool at the same
    layer/backbone. Without this correction, Qwen2.5-7B L20 activations
    sit at cos≈0.24 baseline (||mu||≈55) which inflates raw fve_nrm by
    ~0.11 and compresses dynamic range.
    """
    return fve_nrm(h - mu, v - mu)


def rho_norm(cen: torch.Tensor | float) -> torch.Tensor | float:
    """Normalize centered fve_nrm to [0, 1] using paper anchors.

    rho = (cen - random_floor) / (paraphrase_ceiling - random_floor)

    rho = 0 ↔ unrelated; rho = 1 ↔ saturates Qwen paraphrase ceiling.
    """
    return (cen - RANDOM_FLOOR_CEN) / RHO_DENOM


def anisotropy_mu(pool: torch.Tensor) -> torch.Tensor:
    """Compute the anisotropy mean mu = pool.mean(0).

    pool: (N, d) of last-token L20 activations from a held-out target file.
    Returns (d,) float32.
    """
    return pool.float().mean(0)


@dataclass(frozen=True)
class Anchors:
    """Reference points reported in paper_nla.md §3."""

    random_floor_cen: float = RANDOM_FLOOR_CEN
    paraphrase_ceiling_cen: float = PARAPHRASE_CEILING_CEN
    nn_in_pool_cen: float = 0.663  # pool=200
    nn_retrieval_cen: float = 0.714  # pool=2000
    replay_cen: float = 0.968


__all__ = [
    "fve_nrm",
    "fve_nrm_centered",
    "rho_norm",
    "anisotropy_mu",
    "Anchors",
    "RANDOM_FLOOR_CEN",
    "PARAPHRASE_CEILING_CEN",
    "RHO_DENOM",
]
