"""Bifurcation Estimation Network (BEN).

Estimates the reflexivity coefficient r̂ ∈ [-1, 1] at each position from
the RRM's accumulated meta-state.  Also classifies semiotic regime:
  - Subcritical (r < 0): sign has stable, conventional meaning
  - Supercritical (r > 0): sign is contested, meaning is actively forking

r̂ is the core output of SRT — it tells you WHERE and HOW MUCH meaning
is under contestation in a given text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from srt.config import BENConfig


@dataclass
class BENOutput:
    """Output from BEN."""

    r_hat: torch.Tensor  # (B, T) reflexivity coefficient (unbounded; supervised on log-compressed r_true)
    regime_logits: torch.Tensor  # (B, T, 2) subcritical/supercritical


class BifurcationEstimationNetwork(nn.Module):
    """Estimates bifurcation from RRM meta-state."""

    def __init__(self, cfg: BENConfig, d_meta: int) -> None:
        super().__init__()

        # r̂ prediction: meta-state → unbounded scalar.
        # v3 used nn.Tanh() here, which capped output at ±1.  The training target
        # is sign(r) * log1p(|r|) and r_true reaches ~12.77 (compressed ~2.55), so
        # the tanh ceiling truncated ~25% of supercritical tokens and capped the
        # achievable Pearson.  The smooth_l1 loss on a log-compressed target is
        # numerically well-behaved without an output activation; we keep the head
        # unbounded and init the final linear with small weights so early outputs
        # start near zero and the supervised gradient does the shaping.
        self.r_head = nn.Sequential(
            nn.Linear(d_meta, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, 1),
        )
        r_out: nn.Linear = self.r_head[-1]  # type: ignore[assignment]
        nn.init.normal_(r_out.weight, std=0.02)
        nn.init.zeros_(r_out.bias)

        # Regime classification: subcritical (0) vs supercritical (1)
        self.regime_head = nn.Sequential(
            nn.Linear(d_meta, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, 2),
        )

    def forward(self, meta_state: torch.Tensor) -> BENOutput:
        """Estimate bifurcation from accumulated meta-state.

        Args:
            meta_state: (B, T, d_meta) from RRM.

        Returns:
            BENOutput with r_hat and regime_logits.
        """
        r_hat = self.r_head(meta_state).squeeze(-1)  # (B, T)
        regime_logits = self.regime_head(meta_state)  # (B, T, 2)
        return BENOutput(r_hat=r_hat, regime_logits=regime_logits)
