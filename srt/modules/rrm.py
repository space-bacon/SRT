"""Reflexive Recurrent Module (RRM).

Tracks per-position semiotic meta-state via a GRU that processes divergence
observations from MAH layers.  At injection points, produces a small gated
correction vector that is added to the backbone's hidden states.

The meta-state h_meta_t represents the model's accumulated awareness of
semiotic divergence at position t.  Each MAH observation updates it:
  h_meta_t^{l+1} = GRU(divergence_t^l, h_meta_t^l)

Injection is gated to ensure corrections are small relative to the backbone's
hidden states:
  injection_t = gate(h_meta_t) * proj(h_meta_t) * scale
"""

from __future__ import annotations

import torch
import torch.nn as nn

from srt.config import RRMConfig


class ReflexiveRecurrentModule(nn.Module):
    """GRU-based reflexive meta-state tracker with gated injection."""

    def __init__(self, cfg: RRMConfig, d_divergence: int, d_backbone: int) -> None:
        super().__init__()
        self.d_meta = cfg.d_meta
        self.inject_scale = cfg.inject_scale

        # Per-position GRU: processes divergence → meta-state
        self.gru = nn.GRUCell(d_divergence, cfg.d_meta)

        # Injection: meta-state → backbone-dim correction
        self.inject_proj = nn.Linear(cfg.d_meta, d_backbone, bias=False)
        nn.init.zeros_(self.inject_proj.weight)  # start with zero injection

        # Scalar gate per position
        self.inject_gate = nn.Sequential(
            nn.Linear(cfg.d_meta, 1),
            nn.Sigmoid(),
        )

    def step(
        self, divergence: torch.Tensor, meta_state: torch.Tensor | None
    ) -> torch.Tensor:
        """Update per-position meta-state with new divergence observation.

        Args:
            divergence: (B, T, d_divergence) from MAH.
            meta_state: (B, T, d_meta) or None for initial state.

        Returns:
            Updated meta-state (B, T, d_meta).
        """
        B, T, d_div = divergence.shape
        div_flat = divergence.reshape(B * T, d_div)

        if meta_state is None:
            meta_flat = torch.zeros(
                B * T, self.d_meta, device=divergence.device, dtype=divergence.dtype
            )
        else:
            meta_flat = meta_state.reshape(B * T, self.d_meta)

        meta_flat = self.gru(div_flat, meta_flat)
        return meta_flat.reshape(B, T, self.d_meta)

    def inject(
        self, meta_state: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Produce gated injection vector for backbone hidden states.

        Args:
            meta_state: (B, T, d_meta) current reflexive awareness.
            hidden_states: (B, T, d_backbone) current hidden states.

        Returns:
            Injection vector (B, T, d_backbone) to add to hidden_states.
        """
        gate = self.inject_gate(meta_state)  # (B, T, 1)
        correction = self.inject_proj(meta_state)  # (B, T, d_backbone)
        return gate * correction * self.inject_scale
