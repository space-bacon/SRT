"""Metapragmatic Attention Head (MAH).

Detects where meaning diverges across positions by computing the gap between
direct (local) interpretation and contextual (global) interpretation of each
token's hidden state.  This is Peirce's "unlimited semiosis" made computational:
each sign (representamen) receives an interpretation (interpretant) that depends
on the surrounding discourse context.  MAH quantifies where that context
*changes* the interpretation — i.e., where meaning forks.

The divergence vector d_t at position t captures:
  d_t = f(interp_t) - g(attend(interp_{0..t}))
where f is direct projection, g is the contextual output after causal attention.
High ||d_t|| → the sign at position t means something different in context
than it would in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from srt.config import MAHConfig


@dataclass
class MAHOutput:
    """Output from a single MAH layer."""

    divergence: torch.Tensor  # (B, T, d_divergence)
    attention_weights: torch.Tensor | None = None  # (B, H, T, T)


class MetapragmaticAttentionHead(nn.Module):
    """Single MAH layer that reads hidden states and produces divergence vectors."""

    def __init__(self, cfg: MAHConfig, d_backbone: int, d_community: int = 0) -> None:
        super().__init__()
        d_sub = cfg.d_sub

        # Project backbone hidden states → interpretant subspace
        self.interp_proj = nn.Linear(d_backbone, d_sub, bias=False)

        # Optional community conditioning
        self.comm_proj: nn.Module | None = None
        if d_community > 0:
            self.comm_proj = nn.Linear(d_community, d_sub, bias=False)

        # Multi-head self-attention in interpretant subspace
        self.num_heads = cfg.num_heads
        self.head_dim = d_sub // cfg.num_heads
        assert d_sub % cfg.num_heads == 0

        self.q_proj = nn.Linear(d_sub, d_sub, bias=False)
        self.k_proj = nn.Linear(d_sub, d_sub, bias=False)
        self.v_proj = nn.Linear(d_sub, d_sub, bias=False)
        self.out_proj = nn.Linear(d_sub, d_sub, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # Divergence output projection
        self.div_proj = nn.Linear(d_sub, cfg.d_divergence, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        community_vec: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> MAHOutput:
        """Compute divergence from backbone hidden states.

        Args:
            hidden_states: (B, T, d_backbone) from a transformer layer.
            community_vec: (B, d_community) soft community vector.
            causal_mask: (1, 1, T, T) additive causal mask.

        Returns:
            MAHOutput with divergence vectors and optional attention weights.
        """
        B, T, _ = hidden_states.shape

        # Project to interpretant subspace
        interp = self.interp_proj(hidden_states)  # (B, T, d_sub)

        # Community conditioning: shift interpretant space
        if community_vec is not None and self.comm_proj is not None:
            comm_bias = self.comm_proj(community_vec)  # (B, d_sub)
            interp = interp + comm_bias.unsqueeze(1)

        # Multi-head causal self-attention
        q = self.q_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn = attn + causal_mask
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        contextual = (attn_weights @ v).transpose(1, 2).reshape(B, T, -1)
        contextual = self.out_proj(contextual)  # (B, T, d_sub)

        # Divergence = gap between direct and contextual interpretation
        divergence = self.div_proj(interp - contextual)  # (B, T, d_divergence)

        return MAHOutput(divergence=divergence, attention_weights=attn_weights.detach())
