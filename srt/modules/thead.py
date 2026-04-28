"""T-Head: Takens delay-embedding readout (v10 prototype).

Inspired by Haylett (2025, 2026)'s Geofinitism program and MARINA architecture,
which apply Takens (1981)'s delay-embedding theorem to the token stream.

Where MARINA *replaces* attention with delay-coordinate phase-space
reconstruction, the T-Head is the conservative cross-program experiment: keep
the frozen attention backbone of SRT-Adapter v9 and add a delay-embedding
readout over hidden states as an *additional* channel, orthogonal to the MAH's
pairwise interpretant divergence.

The motivation is that the SRT readouts (BEN r_hat, MAH divergence, Community
Head prototype) all operate on per-token state. The trajectory through hidden
state across positions has additional structure (curvature, recurrence,
local-Lyapunov sensitivity) that none of the v9 readouts surface. A
delay-embedding readout extracts two such signals per position:

  - **Local Lyapunov estimate.** Approximates the local divergence rate of
    nearby trajectories by comparing the displacement of a delay-coordinate
    point at time t to the displacement of its k nearest neighbors in the
    embedded space at time t+1. Positive values indicate locally chaotic
    (sensitive) regions; near-zero values indicate locally stable regions.

  - **Recurrence density.** Counts how many other positions in the sequence
    fall within an epsilon-ball of the current position's delay coordinate.
    High recurrence density indicates positions on a low-dimensional
    attractor (a topic the model returns to); low density indicates novel
    excursions.

Both signals are computed from the same delay-coordinate stack and add a
single learned linear readout each.

This module is NOT wired into v9 SRTAdapter. It is a standalone prototype
intended for v10 experiments where it will be added as a parallel readout
alongside MAH/BEN/Community.

References:
  Takens, F. (1981). Detecting strange attractors in turbulence.
  Haylett, K. R. (2025). Finite Tractus.
  Haylett, K. R. (2026). Geofinitism / MARINA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class THeadConfig:
    """T-Head delay-embedding readout configuration."""

    embedding_dim: int = 4  # m in Takens: number of delay coordinates
    delay: int = 2  # tau: lag between successive coordinates (in tokens)
    knn_k: int = 5  # neighbors used for local-Lyapunov estimate
    recurrence_eps: float = 1.0  # eps-ball radius for recurrence density
    project_to: int = 32  # dim of the per-position projection of hidden state


@dataclass
class THeadOutput:
    """Per-token outputs from T-Head."""

    lyapunov: torch.Tensor  # (B, T) local Lyapunov estimate
    recurrence: torch.Tensor  # (B, T) recurrence density (count of neighbors)
    delay_coord: torch.Tensor  # (B, T, m * project_to) raw delay-coord stack


class TakensHead(nn.Module):
    """Delay-embedding readout over a sequence of hidden states.

    Given a hidden-state stream H of shape (B, T, D), the T-Head:
      1. Projects H -> Z of shape (B, T, project_to) via a learned linear.
      2. Builds a delay-coordinate stack X of shape (B, T, m, project_to)
         where X[b, t, j] = Z[b, t - j*tau] (zero-padded for t < j*tau).
      3. Flattens to V of shape (B, T, m * project_to).
      4. Computes per-position local-Lyapunov estimate and recurrence density
         from V.

    The two scalar outputs (lyapunov, recurrence) can be supervised against
    surrogate signals (e.g. CE-loss derivative for Lyapunov; topical
    self-similarity for recurrence) or used as auxiliary inputs to downstream
    heads in v10.

    Notes
    -----
    The neighbor search and recurrence count are O(T^2) in sequence length;
    for long sequences a chunked or random-projection approximation will be
    needed. For prototype validation on the v9 evaluation sequences (all
    <= 512 tokens) the naive implementation is acceptable.
    """

    def __init__(self, cfg: THeadConfig, d_hidden: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.project = nn.Linear(d_hidden, cfg.project_to, bias=False)
        nn.init.normal_(self.project.weight, std=0.02)

        flat_dim = cfg.embedding_dim * cfg.project_to
        # Optional learned linear over the delay-coordinate vector for
        # downstream wiring; v10 will decide whether to keep this raw or
        # project it further.
        self.readout = nn.Linear(flat_dim, flat_dim, bias=False)
        nn.init.eye_(self.readout.weight)

    def _build_delay_stack(self, z: torch.Tensor) -> torch.Tensor:
        """Build (B, T, m, project_to) delay-coordinate stack."""
        b, t, d = z.shape
        m = self.cfg.embedding_dim
        tau = self.cfg.delay
        out = z.new_zeros(b, t, m, d)
        for j in range(m):
            shift = j * tau
            if shift == 0:
                out[:, :, j, :] = z
            elif shift < t:
                out[:, shift:, j, :] = z[:, : t - shift, :]
            # else: leave zeros (positions earlier than t = j*tau)
        return out

    def _local_lyapunov(self, v: torch.Tensor) -> torch.Tensor:
        """Estimate local Lyapunov exponent at each position.

        For each position t, find the k nearest neighbors in the embedded
        space and compare the mean inter-neighbor distance at t to the mean
        inter-neighbor distance at t+1. The log of the ratio approximates the
        local divergence rate (positive = chaotic).
        """
        b, t, d = v.shape
        k = min(self.cfg.knn_k, t - 1)
        if t < 2 or k <= 0:
            return v.new_zeros(b, t)

        # (B, T, T) pairwise distances
        dist = torch.cdist(v, v)  # symmetric, zero diagonal
        # exclude self by setting diagonal to +inf
        eye_mask = torch.eye(t, device=v.device, dtype=torch.bool)
        dist = dist.masked_fill(eye_mask.unsqueeze(0), float("inf"))

        # nearest-neighbor indices (B, T, k)
        nn_idx = dist.topk(k, dim=-1, largest=False).indices

        # next-step indices (clamp at T-1)
        nxt_idx = (nn_idx + 1).clamp(max=t - 1)

        # gather neighbor positions at t and t+1
        # v_nbr_t: (B, T, k, D); v_nbr_t1: (B, T, k, D)
        v_nbr_t = torch.gather(
            v.unsqueeze(1).expand(-1, t, -1, -1), 2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, d),
        )
        v_nbr_t1 = torch.gather(
            v.unsqueeze(1).expand(-1, t, -1, -1), 2,
            nxt_idx.unsqueeze(-1).expand(-1, -1, -1, d),
        )

        # current-position vector at t and t+1
        v_t = v.unsqueeze(2)  # (B, T, 1, D)
        nxt_self = torch.arange(t, device=v.device).clamp(max=t - 1) + 1
        nxt_self = nxt_self.clamp(max=t - 1)
        v_t1 = v[:, nxt_self, :].unsqueeze(2)  # (B, T, 1, D)

        d_t = (v_nbr_t - v_t).norm(dim=-1).mean(dim=-1).clamp_min(1e-8)
        d_t1 = (v_nbr_t1 - v_t1).norm(dim=-1).mean(dim=-1).clamp_min(1e-8)
        return (d_t1 / d_t).log()

    def _recurrence_density(self, v: torch.Tensor) -> torch.Tensor:
        """Count neighbors within eps-ball at each position.

        Returns a (B, T) float tensor with the number of other positions
        within `recurrence_eps` of each position's delay coordinate.
        """
        b, t, _ = v.shape
        if t < 2:
            return v.new_zeros(b, t)
        dist = torch.cdist(v, v)
        eye_mask = torch.eye(t, device=v.device, dtype=torch.bool)
        dist = dist.masked_fill(eye_mask.unsqueeze(0), float("inf"))
        return (dist < self.cfg.recurrence_eps).float().sum(dim=-1)

    def forward(self, hidden: torch.Tensor) -> THeadOutput:
        z = self.project(hidden)  # (B, T, project_to)
        stack = self._build_delay_stack(z)  # (B, T, m, project_to)
        b, t, m, d = stack.shape
        v = self.readout(stack.reshape(b, t, m * d))  # (B, T, m*project_to)
        lyap = self._local_lyapunov(v)
        rec = self._recurrence_density(v)
        return THeadOutput(lyapunov=lyap, recurrence=rec, delay_coord=v)


__all__ = ["THeadConfig", "THeadOutput", "TakensHead"]
