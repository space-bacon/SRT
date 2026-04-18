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

    r_hat: torch.Tensor  # (B, T) reflexivity coefficient
    regime_logits: torch.Tensor  # (B, T, 2) subcritical/supercritical


class BifurcationEstimationNetwork(nn.Module):
    """Estimates bifurcation from RRM meta-state."""

    def __init__(self, cfg: BENConfig, d_meta: int) -> None:
        super().__init__()

        # r̂ prediction: meta-state → scalar ∈ [-1, 1]
        self.r_head = nn.Sequential(
            nn.Linear(d_meta, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, 1),
            nn.Tanh(),
        )

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
