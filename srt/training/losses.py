"""Semiotic loss functions for SRT Adapter.

All losses operate on the adapter's intermediate outputs — divergence vectors,
meta-state, r̂, regime, community assignments.  The CE loss comes directly from
the backbone's native LM head and is not computed here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from srt.adapter import SRTAdapterOutput
from srt.config import LossConfig


def chain_loss(
    divergences: list[torch.Tensor],
    chain_predictor: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Self-supervised chain-of-interpretants loss.

    Each MAH layer's divergence should be predictable from the previous layer's
    divergence.  This is Peirce's chain of interpretants: each interpretation
    leads to the next in a predictable way when meaning is stable.

    Args:
        divergences: list of (B, T, d_div) tensors from successive MAH layers.
        chain_predictor: nn.Linear that predicts next divergence from current.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.

    Returns:
        Scalar loss.
    """
    if len(divergences) < 2:
        return torch.tensor(0.0, device=divergences[0].device)

    loss = torch.tensor(0.0, device=divergences[0].device)
    for i in range(len(divergences) - 1):
        pred = chain_predictor(divergences[i])
        target = divergences[i + 1].detach()
        per_pos = (pred - target).pow(2).mean(dim=-1)  # (B, T)
        if attention_mask is not None:
            mask = attention_mask.to(per_pos.dtype)
            loss = loss + (per_pos * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss + per_pos.mean()
    return loss / (len(divergences) - 1)


def bifurcation_loss(
    r_hat: torch.Tensor,
    r_true: torch.Tensor,
    r_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 loss between predicted and true reflexivity coefficient.

    Uses focal weighting to upweight rare supercritical samples.

    Args:
        r_hat: (B, T) predicted reflexivity from BEN.
        r_true: (B, T) ground-truth reflexivity.
        r_mask: (B, T) bool mask — True where r_true is valid.

    Returns:
        Scalar loss.
    """
    if r_mask.sum() == 0:
        return torch.tensor(0.0, device=r_hat.device)

    r_hat_masked = r_hat[r_mask]
    r_true_masked = r_true[r_mask]

    # Focal weighting: upweight non-zero r_true (rare supercritical signals)
    focal_weight = 1.0 + 3.0 * r_true_masked.abs()
    diff = F.smooth_l1_loss(r_hat_masked, r_true_masked, reduction="none")
    return (diff * focal_weight).mean()


def regime_loss(
    regime_logits: torch.Tensor,
    r_true: torch.Tensor,
    r_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for regime classification.

    Regime is derived from r_true:
      - subcritical (class 0): r_true <= 0
      - supercritical (class 1): r_true > 0

    Args:
        regime_logits: (B, T, 2) from BEN.
        r_true: (B, T) ground-truth reflexivity.
        r_mask: (B, T) bool mask.

    Returns:
        Scalar loss.
    """
    if r_mask.sum() == 0:
        return torch.tensor(0.0, device=regime_logits.device)

    regime_targets = (r_true > 0).long()  # (B, T)
    logits_masked = regime_logits[r_mask]  # (N, 2)
    targets_masked = regime_targets[r_mask]  # (N,)
    return F.cross_entropy(logits_masked, targets_masked)


def divergence_alive_loss(
    divergences: list[torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prevent divergence vectors from collapsing to zero.

    Encourages divergence norms to stay near a target value (1.0).

    Args:
        divergences: list of (B, T, d_div) tensors.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.

    Returns:
        Scalar loss.
    """
    if not divergences:
        return torch.tensor(0.0, device=divergences[0].device if divergences else "cpu")

    total = torch.tensor(0.0, device=divergences[0].device)
    for d in divergences:
        norms = d.norm(dim=-1)  # (B, T)
        if attention_mask is not None:
            mask = attention_mask.to(norms.dtype)
            mean_norm = (norms * mask).sum() / mask.sum().clamp(min=1)
        else:
            mean_norm = norms.mean()
        total = total + (1.0 - mean_norm).abs()
    return total / len(divergences)


def injection_regularization(
    injections: list[torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep injection vectors small (L2 penalty).

    The adapter should make subtle corrections, not override the backbone.

    Args:
        injections: list of (B, T, d_backbone) injection vectors.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.

    Returns:
        Scalar loss.
    """
    if not injections:
        return torch.tensor(0.0, device=injections[0].device if injections else "cpu")

    total = torch.tensor(0.0, device=injections[0].device)
    for inj in injections:
        per_pos = inj.pow(2).mean(dim=-1)  # (B, T)
        if attention_mask is not None:
            mask = attention_mask.to(per_pos.dtype)
            total = total + (per_pos * mask).sum() / mask.sum().clamp(min=1)
        else:
            total = total + per_pos.mean()
    return total / len(injections)


def community_entropy_loss(community_weights: torch.Tensor) -> torch.Tensor:
    """Encourage diverse community usage across the batch.

    Maximizes entropy of the average community assignment distribution.
    Without this, the model might collapse all inputs to one community.

    Args:
        community_weights: (B, K) soft community assignments.

    Returns:
        Scalar loss (lower = more diverse).
    """
    avg_dist = community_weights.mean(dim=0)  # (K,)
    entropy = -(avg_dist * (avg_dist + 1e-8).log()).sum()
    max_entropy = math.log(community_weights.shape[-1])
    return max_entropy - entropy


def compute_total_loss(
    output: SRTAdapterOutput,
    chain_predictor: torch.nn.Module,
    r_true: torch.Tensor | None,
    r_mask: torch.Tensor | None,
    config: LossConfig,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss from adapter output.

    Args:
        output: SRTAdapterOutput from adapter forward pass.
        chain_predictor: the chain predictor module from the adapter.
        r_true: (B, T) ground-truth reflexivity, or None.
        r_mask: (B, T) bool mask for valid r_true positions, or None.
        config: LossConfig with weights.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.

    Returns:
        (total_loss, metrics_dict) where metrics_dict has per-component values.
    """
    device = output.logits.device
    metrics: dict[str, float] = {}
    total = torch.tensor(0.0, device=device)

    # CE loss (from backbone)
    if output.ce_loss is not None:
        total = total + config.ce_weight * output.ce_loss
        metrics["ce"] = output.ce_loss.item()

    # Chain-of-interpretants loss
    if output.divergences:
        l_chain = chain_loss(output.divergences, chain_predictor, attention_mask)
        total = total + config.chain_weight * l_chain
        metrics["chain"] = l_chain.item()

    # Bifurcation + regime losses
    if output.ben_output is not None and r_true is not None and r_mask is not None:
        l_bif = bifurcation_loss(output.ben_output.r_hat, r_true, r_mask)
        total = total + config.bif_weight * l_bif
        metrics["bif"] = l_bif.item()

        l_regime = regime_loss(output.ben_output.regime_logits, r_true, r_mask)
        total = total + config.regime_weight * l_regime
        metrics["regime"] = l_regime.item()

    # Divergence alive
    if output.divergences:
        l_alive = divergence_alive_loss(output.divergences, attention_mask)
        total = total + config.div_alive_weight * l_alive
        metrics["div_alive"] = l_alive.item()

    # Injection regularization
    if output.injections:
        l_inject = injection_regularization(output.injections, attention_mask)
        total = total + config.inject_reg_weight * l_inject
        metrics["inject_reg"] = l_inject.item()

    # Community entropy
    if output.community_output is not None:
        l_comm = community_entropy_loss(output.community_output.weights)
        total = total + config.community_entropy_weight * l_comm
        metrics["comm_entropy"] = l_comm.item()

    metrics["total"] = total.item()
    return total, metrics
