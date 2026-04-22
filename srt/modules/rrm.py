"""Reflexive Recurrent Module (RRM).

Tracks per-position semiotic meta-state via a GRU that processes divergence
observations from MAH layers.  At injection points, produces a FiLM-style
modulation (gamma, beta) that multiplicatively + additively biases the
backbone's hidden states:

    h' = h * (1 + gamma(meta_state)) + beta(meta_state)

The meta-state h_meta_t represents the model's accumulated awareness of
semiotic divergence at position t.  Each MAH observation updates it:
    h_meta_t^{l+1} = GRU(divergence_t^l, h_meta_t^l)

v3 used a single low-rank linear inject (gate * proj * scale) with
zero-initialized projection.  Ablation showed the inject-back arm contributed
exactly nothing (every benchmark metric was identical to four decimal places
with injection forced to zero).  The diagnosis was that the zero init plus
the inject-norm regularizer (which rewarded ||inj|| \u2248 1 regardless of
direction) drove the optimizer to satisfy the norm penalty with arbitrary
directions that were then orthogonal to the gradient signal from the frozen
backbone's CE.

v4 fixes both: FiLM modulation has a non-zero gradient pathway from the first
step (beta is initialized to zero so the forward is identity at init, but
gamma has small Gaussian init so dL/d(gamma_proj) flows immediately when the
downstream MAH layer's divergence is supervised by the bif/regime losses).
The inject-norm regularizer is dropped at the loss layer (LossConfig
inject_reg_weight = 0.0 by default in v4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from srt.config import RRMConfig


class ReflexiveRecurrentModule(nn.Module):
    """GRU-based reflexive meta-state tracker with FiLM-style injection."""

    def __init__(self, cfg: RRMConfig, d_divergence: int, d_backbone: int) -> None:
        super().__init__()
        self.d_meta = cfg.d_meta
        self.inject_scale = cfg.inject_scale
        self.d_backbone = d_backbone

        # Per-position GRU: processes divergence \u2192 meta-state
        self.gru = nn.GRUCell(d_divergence, cfg.d_meta)

        # FiLM projections: meta-state \u2192 (gamma, beta) in backbone-dim.
        # gamma is multiplicative on (1 + gamma); beta is additive.
        # gamma init: small Gaussian (std=0.02) so identity-at-init holds in
        # expectation but gradient flows from the first step.
        # beta init: zeros so identity-at-init is exact, then learns offsets.
        self.gamma_proj = nn.Linear(cfg.d_meta, d_backbone, bias=True)
        self.beta_proj = nn.Linear(cfg.d_meta, d_backbone, bias=True)
        nn.init.normal_(self.gamma_proj.weight, std=0.02)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

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
        """Produce FiLM modulation correction for backbone hidden states.

        Returns the *correction*  (h' - h) = h * gamma + beta, NOT h'.  The
        caller adds this to h to get h'.  This keeps the rest of the adapter
        and the diagnostic logging (injection norm tracking) unchanged: the
        \"injection\" tensor is still the additive correction applied to h.

        Args:
            meta_state: (B, T, d_meta) current reflexive awareness.
            hidden_states: (B, T, d_backbone) current hidden states.

        Returns:
            Correction vector (B, T, d_backbone) to add to hidden_states.
        """
        gamma = self.gamma_proj(meta_state)  # (B, T, d_backbone)
        beta = self.beta_proj(meta_state)  # (B, T, d_backbone)
        # FiLM: h' = h * (1 + gamma) + beta  \u2192  correction = h * gamma + beta
        correction = hidden_states * gamma + beta
        return correction * self.inject_scale
