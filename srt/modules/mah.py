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

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        community_vec: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental MAH forward with an external K/V cache.

        Computes divergence for the `T` query positions in `hidden_states`
        (T may be 1 for autoregressive decode, or the full prompt length for
        prefill) while attending over all cached keys/values plus the new ones.

        This is mathematically identical to `forward()` when called once with
        the full sequence and `kv_cache=None`; it just additionally returns the
        per-layer K/V so subsequent single-token steps can attend to the past
        without recomputing it.

        Args:
            hidden_states: (B, T, d_backbone) query positions.
            community_vec: (B, d_community) optional conditioning.
            kv_cache: optional (k_past, v_past), each (B, H, T_past, head_dim).

        Returns:
            (divergence (B, T, d_divergence), (k_full, v_full)).
        """
        B, T, _ = hidden_states.shape

        interp = self.interp_proj(hidden_states)  # (B, T, d_sub)
        if community_vec is not None and self.comm_proj is not None:
            interp = interp + self.comm_proj(community_vec).unsqueeze(1)

        q = self.q_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(interp).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # Use memory-efficient SDPA instead of materializing the full (T, S)
        # attention matrix. At long prefill lengths (tens of thousands of
        # tokens) the explicit `q @ k.T` tensor is tens of GB and OOMs; the
        # fused kernel never materializes it.
        #   - prefill (T > 1, no cache): standard causal self-attention.
        #   - decode  (T == 1): the single query attends to every cached key
        #     (all are in the past), so no causal mask is applied.
        contextual = F.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1),
        )  # (B, H, T, head_dim)
        contextual = contextual.transpose(1, 2).reshape(B, T, -1)
        contextual = self.out_proj(contextual)

        divergence = self.div_proj(interp - contextual)
        return divergence, (k, v)
