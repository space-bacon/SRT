"""Losses and metrics for SRT-NLA.

The canonical NLA quality metric is direction MSE on L2-normalised
vectors (``mse_nrm``); ``fve_nrm`` is the fraction of variance explained
relative to a uniform random unit baseline (which has ``E[mse_nrm] = 2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def mse_nrm(
    v_hat: torch.Tensor, v_target: torch.Tensor, *, reduce: bool = True
) -> torch.Tensor:
    """Squared L2 distance between unit-normalised vectors. Lower is better.

    Both tensors are ``(..., d)``. With ``reduce=True`` (default) returns a
    scalar (mean over leading dims); with ``reduce=False`` returns ``(...,)``
    per-sample values — needed by REINFORCE so the advantage baseline
    ``r - r.mean()`` is not identically zero.
    """
    a = F.normalize(v_hat, dim=-1)
    b = F.normalize(v_target, dim=-1)
    per_sample = ((a - b) ** 2).sum(dim=-1)
    return per_sample.mean() if reduce else per_sample


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
    v_hat: torch.Tensor, v_target: torch.Tensor, *, reduce: bool = True
) -> torch.Tensor:
    """Squared difference of L2 norms. Penalises systematic over/under-shoot."""
    per_sample = (v_hat.norm(dim=-1) - v_target.norm(dim=-1)).pow(2)
    return per_sample.mean() if reduce else per_sample


def entropy_floor_hinge(
    token_entropy: torch.Tensor, h_min: float, *, reduce: bool = True
) -> torch.Tensor:
    """Hinge that fires when per-token entropy drops below ``h_min`` nats."""
    per_sample = torch.clamp(h_min - token_entropy, min=0.0)
    return per_sample.mean() if reduce else per_sample


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
    reduce: bool = True,
) -> RewardComponents:
    """5-term NLA reward (higher is better).

    ``r = -mse_nrm - lambda_mag * mag - beta_kl * KL - gamma * H_hinge + delta * comm``

    Any optional term left as ``None`` is treated as zero (so N0/N1 can
    compose the reward incrementally as the surrounding modules come
    online).

    With ``reduce=False`` every component is per-sample ``(B,)``; with
    ``reduce=True`` (default) every component is a scalar. REINFORCE must
    use ``reduce=False`` so ``advantage = r - r.mean()`` is non-degenerate.
    """
    device = v_hat.device
    dtype = v_hat.dtype
    B = v_hat.shape[0] if v_hat.dim() >= 1 else 1
    zero_shape = () if reduce else (B,)
    zero = torch.zeros(zero_shape, device=device, dtype=dtype)

    def _maybe_per_sample(x: torch.Tensor) -> torch.Tensor:
        if reduce:
            return x.mean() if x.dim() > 0 else x
        # caller passed a scalar but we want per-sample → broadcast.
        return x.expand(B) if x.dim() == 0 else x

    mse = mse_nrm(v_hat, v_target, reduce=reduce)
    mag = magnitude_penalty(v_hat, v_target, reduce=reduce)
    kl = _maybe_per_sample(kl_av_vs_base) if kl_av_vs_base is not None else zero
    ent = (
        entropy_floor_hinge(token_entropy, h_min, reduce=reduce)
        if token_entropy is not None
        else zero
    )
    comm = (
        _maybe_per_sample(community_consistency)
        if community_consistency is not None
        else zero
    )

    total = -mse - lambda_mag * mag - beta_kl * kl - gamma_entropy * ent + delta_community * comm
    return RewardComponents(
        mse=mse, mag=mag, kl=kl, entropy_hinge=ent, community=comm, total=total
    )
