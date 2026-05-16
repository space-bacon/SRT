"""Losses and metrics for SRT-NLA.

The canonical NLA quality metric is direction MSE on L2-normalised
vectors (``mse_nrm``); ``fve_nrm`` is the fraction of variance explained
relative to a uniform random unit baseline (which has ``E[mse_nrm] = 2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def mse_nrm(v_hat: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance between unit-normalised vectors. Lower is better.

    Both tensors are ``(..., d)``. Returns a scalar (mean over batch).
    """
    a = F.normalize(v_hat, dim=-1)
    b = F.normalize(v_target, dim=-1)
    return ((a - b) ** 2).sum(dim=-1).mean()


def cosine_similarity(v_hat: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity. ``= 1 - mse_nrm/2``."""
    return F.cosine_similarity(v_hat, v_target, dim=-1).mean()


def fraction_variance_explained(
    v_hat: torch.Tensor, v_target: torch.Tensor
) -> torch.Tensor:
    """FVE on L2-normalised vectors.

    Random uniform unit vectors have ``E[||u - v||^2] = 2``, so
    ``fve_nrm = 1 - mse_nrm / 2``. The Qwen2.5-7B NLA reference reports
    ``fve_nrm = 0.752`` at L=20.
    """
    return 1.0 - mse_nrm(v_hat, v_target) / 2.0


def magnitude_penalty(
    v_hat: torch.Tensor, v_target: torch.Tensor
) -> torch.Tensor:
    """Squared difference of L2 norms. Penalises systematic over/under-shoot."""
    return (v_hat.norm(dim=-1) - v_target.norm(dim=-1)).pow(2).mean()


def entropy_floor_hinge(
    token_entropy: torch.Tensor, h_min: float
) -> torch.Tensor:
    """Hinge that fires when per-token entropy drops below ``h_min`` nats."""
    return torch.clamp(h_min - token_entropy, min=0.0).mean()


@dataclass
class RewardComponents:
    mse: torch.Tensor
    mag: torch.Tensor
    kl: torch.Tensor
    entropy_hinge: torch.Tensor
    community: torch.Tensor
    total: torch.Tensor


def nla_reward(
    v_hat: torch.Tensor,
    v_target: torch.Tensor,
    *,
    kl_av_vs_base: torch.Tensor | None = None,
    token_entropy: torch.Tensor | None = None,
    community_consistency: torch.Tensor | None = None,
    lambda_mag: float = 0.1,
    beta_kl: float = 0.05,
    gamma_entropy: float = 0.05,
    delta_community: float = 0.2,
    h_min: float = 1.5,
) -> RewardComponents:
    """5-term NLA reward (higher is better).

    ``r = -mse_nrm - lambda_mag * mag - beta_kl * KL - gamma * H_hinge + delta * comm``

    Any optional term left as ``None`` is treated as zero (so N0/N1 can
    compose the reward incrementally as the surrounding modules come
    online).
    """
    device = v_hat.device
    zero = torch.zeros((), device=device, dtype=v_hat.dtype)

    mse = mse_nrm(v_hat, v_target)
    mag = magnitude_penalty(v_hat, v_target)
    kl = kl_av_vs_base if kl_av_vs_base is not None else zero
    ent = (
        entropy_floor_hinge(token_entropy, h_min)
        if token_entropy is not None
        else zero
    )
    comm = community_consistency if community_consistency is not None else zero

    total = -mse - lambda_mag * mag - beta_kl * kl - gamma_entropy * ent + delta_community * comm
    return RewardComponents(
        mse=mse, mag=mag, kl=kl, entropy_hinge=ent, community=comm, total=total
    )
