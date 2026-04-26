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
    """Smooth L1 loss between predicted and log-compressed true reflexivity.

    r_true spans [0, ~13] but BEN outputs Tanh ∈ [-1, 1].  Log-compressing
    the target via  sign(r) * log(1 + |r|)  maps the bulk of r_true into
    roughly [-1.5, 1.5], much better aligned with the output range.

    Uses mild focal weighting (1.0×) for supercritical tokens.

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

    # Log-compress targets to match BEN's Tanh output range
    r_true_compressed = r_true_masked.sign() * (1.0 + r_true_masked.abs()).log()

    # Mild focal weighting on compressed scale
    focal_weight = 1.0 + 1.0 * r_true_compressed.abs()
    diff = F.smooth_l1_loss(r_hat_masked, r_true_compressed, reduction="none")
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
    target_norm: float = 1.0,
) -> torch.Tensor:
    """Penalize injection norms that deviate from a target.

    Uses (||inj|| - target)^2 per position, which pulls norms toward the
    target rather than toward zero.  This lets the RRM contribute useful
    signal at norm ~1 while strongly penalizing the 6-8 norms seen in v2.

    Args:
        injections: list of (B, T, d_backbone) injection vectors.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.
        target_norm: desired L2 norm for injection vectors.

    Returns:
        Scalar loss.
    """
    if not injections:
        return torch.tensor(0.0, device=injections[0].device if injections else "cpu")

    total = torch.tensor(0.0, device=injections[0].device)
    for inj in injections:
        norms = inj.norm(dim=-1)  # (B, T) — L2 norm per position
        per_pos = (norms - target_norm).pow(2)  # (B, T)
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


def community_supcon_loss(
    community_vectors: torch.Tensor,
    community_ids: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised contrastive loss on per-sample community vectors.

    The v3 entropy regularizer kept the prototype distribution from collapsing
    to a single mode but produced congruent collapse instead: pairwise cosine
    similarity between learned prototypes converged to ~0.99 and the assignment
    head learned to be near-uniform.  SupCon (Khosla et al. 2020) gives the
    encoder direct gradient pressure to put samples from the same source into
    a tight neighborhood and push different sources apart, which forces
    prototypes to diversify because that is the only way to satisfy the loss.

    v5 note: pass `encoded` (pre-mixing) rather than `vector` here. When the
    assignment head collapses, `vector ≈ prototype[k*]` is constant across
    the batch and this loss is identically `log(B-1)` with zero gradient.

    Args:
        community_vectors: (B, d) per-sample vectors to contrast.
        community_ids: (B,) integer ids — same id = positive pair.
        temperature: softmax temperature.

    Returns:
        (loss, diagnostics) where diagnostics carries the number of positive
        pairs and unique classes seen this batch (for sanity checks).
    """
    B = community_vectors.shape[0]
    device = community_vectors.device
    n_unique = int(community_ids.unique().numel())
    if B < 2:
        return torch.tensor(0.0, device=device), {
            "pos_pairs": 0.0, "unique_classes": float(n_unique),
        }

    # Cast to float32 for numerical stability of the contrastive softmax;
    # the input vector may be bf16 because the adapter modules run in bf16.
    z = F.normalize(community_vectors.float(), dim=-1)
    sim = (z @ z.T) / temperature  # (B, B)

    # Mask: True where i != j and ids match
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = (community_ids.view(-1, 1) == community_ids.view(1, -1)) & ~eye

    # If no positive pairs in this batch, loss is zero (skip rather than NaN)
    pos_count = pos_mask.sum(dim=1)
    n_pos_pairs = float(pos_mask.sum().item())
    if pos_count.sum() == 0:
        return torch.tensor(0.0, device=device), {
            "pos_pairs": 0.0, "unique_classes": float(n_unique),
        }

    # Mask self-similarity from denominator
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)
    # Diagonal entries are -inf; zero them so they don't contaminate the
    # masked sum below (0 * -inf = NaN otherwise).
    log_prob = log_prob.masked_fill(eye, 0.0)

    # Per-anchor average log-prob over its positives; rows with no positives
    # contribute 0 (avoid divide-by-zero with clamp).
    per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1).float()
    valid = pos_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=device), {
            "pos_pairs": 0.0, "unique_classes": float(n_unique),
        }
    return per_anchor[valid].mean(), {
        "pos_pairs": n_pos_pairs, "unique_classes": float(n_unique),
    }


def archetype_supcon_loss(
    community_vectors: torch.Tensor,
    archetype_ids: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised contrastive loss keyed by archetype_id.

    v9 addition. Uses the same supcon kernel as community_supcon_loss but
    masks out anchors with archetype_id == -1 (Reddit-corpus rows that
    carry no archetype label). Designed to be applied to the same
    `community_output.encoded` representation that community_supcon
    operates on, so both signals shape the same encoder geometry.

    Args:
        community_vectors: (B, d) per-sample encoder output.
        archetype_ids: (B,) ints in [1, 33] for archetype rows, -1 for
            Reddit rows. Anchors with id == -1 are dropped.
        temperature: softmax temperature.

    Returns:
        (loss, diagnostics).
    """
    device = community_vectors.device
    valid = archetype_ids >= 0
    n_valid = int(valid.sum().item())
    if n_valid < 2:
        return torch.tensor(0.0, device=device), {
            "pos_pairs": 0.0, "unique_classes": 0.0, "n_valid": float(n_valid),
        }
    sub_vec = community_vectors[valid]
    sub_ids = archetype_ids[valid]
    loss, diag = community_supcon_loss(
        sub_vec, sub_ids, temperature=temperature,
    )
    diag["n_valid"] = float(n_valid)
    return loss, diag


def divergence_supcon_loss(
    divergences: list[torch.Tensor],
    community_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised contrastive loss on per-sample mean divergence vectors.

    v6 extension of the v5 community-SupCon idea applied to MAH divergence:
    the last MAH layer's divergence is mean-pooled (masked) to a per-sample
    vector, then contrasted by community id. Same lesson as v5 — operate on
    a representation that is bijective with the encoder input rather than on
    a quantity that can collapse to a constant across the batch.

    Args:
        divergences: list of (B, T, d_div) tensors from successive MAH layers.
            The last entry is used as the contrastive representation.
        community_ids: (B,) integer ids — same id = positive pair.
        attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.
        temperature: softmax temperature.

    Returns:
        (loss, diagnostics).
    """
    if not divergences:
        device = community_ids.device if community_ids is not None else "cpu"
        return torch.tensor(0.0, device=device), {
            "pos_pairs": 0.0, "unique_classes": 0.0,
        }
    div = divergences[-1]  # (B, T, d_div)
    if attention_mask is not None:
        m = attention_mask.to(div.dtype).unsqueeze(-1)  # (B, T, 1)
        pooled = (div * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    else:
        pooled = div.mean(dim=1)
    # Reuse the same SupCon kernel as community_supcon_loss.
    return community_supcon_loss(pooled, community_ids, temperature=temperature)


def listnet_loss(
    r_hat: torch.Tensor,
    r_true: torch.Tensor,
    r_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet ranking loss for r̂ over each sequence.

    Cross-entropy between softmax(r_true) and softmax(r_hat) treated as
    rankings over the valid positions of each sequence. Complements the
    pointwise smooth-L1 bifurcation loss by giving direct gradient on the
    *ordering* of r̂ within a passage, which is what downstream uses
    (top-k attention probes, heatmap visualization, percentile thresholds).

    Args:
        r_hat: (B, T) predicted reflexivity.
        r_true: (B, T) ground-truth reflexivity (same scale).
        r_mask: (B, T) bool mask.
        temperature: softmax temperature.

    Returns:
        Scalar loss averaged over sequences with >=2 valid positions.
    """
    B, T = r_hat.shape
    device = r_hat.device
    losses: list[torch.Tensor] = []
    # Compress true reflexivity to match r_hat scale (same transform as bif loss).
    r_true_c = r_true.sign() * (1.0 + r_true.abs()).log()
    for b in range(B):
        m = r_mask[b]
        n = int(m.sum())
        if n < 2:
            continue
        rh = r_hat[b][m].float() / temperature
        rt = r_true_c[b][m].float() / temperature
        # Softmax over the valid positions; mask out -inf done implicitly
        # because we only index the valid slice.
        log_p_hat = rh - torch.logsumexp(rh, dim=0)
        p_true = torch.softmax(rt, dim=0)
        losses.append(-(p_true * log_p_hat).sum())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def chain_residual_aux_loss(
    chain_residual: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    target: float = 0.5,
) -> torch.Tensor:
    """Auxiliary penalty pulling mean per-token chain residual toward a target.

    The chain prediction loss already minimizes the residual directly, but
    that signal is averaged across all positions and dimensions. This term
    keeps the per-token residual at a non-trivial level so it remains a
    useful inference-time signal (used by the hallucination probe) rather
    than collapsing to ~0 everywhere as training proceeds. With a small
    weight this acts as a soft floor, not a primary objective.

    Args:
        chain_residual: (B, T) per-token mean chain residual.
        attention_mask: (B, T) padding mask.
        target: desired mean residual on real tokens.

    Returns:
        Scalar loss.
    """
    if attention_mask is not None:
        m = attention_mask.to(chain_residual.dtype)
        mean = (chain_residual * m).sum() / m.sum().clamp(min=1)
    else:
        mean = chain_residual.mean()
    return (mean - target).pow(2)


def compute_total_loss(
    output: SRTAdapterOutput,
    chain_predictor: torch.nn.Module,
    r_true: torch.Tensor | None,
    r_mask: torch.Tensor | None,
    config: LossConfig,
    attention_mask: torch.Tensor | None = None,
    community_ids: torch.Tensor | None = None,
    archetype_ids: torch.Tensor | None = None,
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

        # v6: ListNet ranking loss on r̂ within each sequence.
        if config.listnet_weight > 0:
            l_listnet = listnet_loss(
                output.ben_output.r_hat, r_true, r_mask,
                temperature=config.listnet_temperature,
            )
            total = total + config.listnet_weight * l_listnet
            metrics["listnet"] = l_listnet.item()

    # Divergence alive
    if output.divergences:
        l_alive = divergence_alive_loss(output.divergences, attention_mask)
        total = total + config.div_alive_weight * l_alive
        metrics["div_alive"] = l_alive.item()

        # v6: SupCon on mean-pooled last-MAH divergence.
        if (community_ids is not None
                and config.divergence_supcon_weight > 0):
            l_div_sup, div_sup_diag = divergence_supcon_loss(
                output.divergences, community_ids,
                attention_mask=attention_mask,
                temperature=config.divergence_supcon_temperature,
            )
            total = total + config.divergence_supcon_weight * l_div_sup
            metrics["div_supcon"] = l_div_sup.item()
            metrics["div_supcon_pos_pairs"] = div_sup_diag["pos_pairs"]

    # v6: chain-residual auxiliary floor (keeps inference signal alive).
    if (output.chain_residual_per_token is not None
            and config.chain_residual_aux_weight > 0):
        l_chain_aux = chain_residual_aux_loss(
            output.chain_residual_per_token,
            attention_mask=attention_mask,
            target=config.chain_residual_aux_target,
        )
        total = total + config.chain_residual_aux_weight * l_chain_aux
        metrics["chain_aux"] = l_chain_aux.item()

    # Injection regularization (target-norm penalty)
    if output.injections:
        l_inject = injection_regularization(
            output.injections, attention_mask, target_norm=config.inject_target_norm,
        )
        total = total + config.inject_reg_weight * l_inject
        metrics["inject_reg"] = l_inject.item()

    # Community entropy (only meaningful when there is a soft assignment)
    if output.community_output is not None:
        if output.community_output.weights is not None:
            l_comm = community_entropy_loss(output.community_output.weights)
            total = total + config.community_entropy_weight * l_comm
            metrics["comm_entropy"] = l_comm.item()

        # Community SupCon (v5 — applied to pre-mixing encoder output, not
        # the prototype-weighted vector. The vector is a convex combination
        # of prototypes; if the assignment head collapses to one prototype
        # then `vector` becomes constant across the batch and SupCon's
        # gradient is zero by symmetry — this is exactly what killed v4.
        # `encoded` is the encoder's bijective image of the pooled hidden
        # state, so it always varies per-sample, giving SupCon non-zero
        # gradient even from a degenerate warm-start.)
        if community_ids is not None and config.community_supcon_weight > 0:
            l_supcon, supcon_diag = community_supcon_loss(
                output.community_output.encoded,
                community_ids,
                temperature=config.community_supcon_temperature,
            )
            total = total + config.community_supcon_weight * l_supcon
            metrics["comm_supcon"] = l_supcon.item()
            metrics["comm_supcon_pos_pairs"] = supcon_diag["pos_pairs"]
            metrics["comm_supcon_unique_classes"] = supcon_diag["unique_classes"]

        # v9: archetype-keyed SupCon on the same encoder representation.
        # Skipped silently if no archetype rows are in the batch (n_valid<2).
        if (archetype_ids is not None
                and config.archetype_supcon_weight > 0):
            l_arch, arch_diag = archetype_supcon_loss(
                output.community_output.encoded,
                archetype_ids,
                temperature=config.archetype_supcon_temperature,
            )
            total = total + config.archetype_supcon_weight * l_arch
            metrics["arch_supcon"] = l_arch.item()
            metrics["arch_supcon_pos_pairs"] = arch_diag["pos_pairs"]
            metrics["arch_supcon_n_valid"] = arch_diag["n_valid"]

    metrics["total"] = total.item()
    return total, metrics
