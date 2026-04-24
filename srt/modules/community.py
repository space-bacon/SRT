"""Unsupervised Community Discovery Head.

Discovers discourse communities from backbone hidden states without
predefined labels.  A discourse community (in Peirce's framework) is a
group of language users who share interpretive norms — they assign similar
interpretants to the same representamens.

The community head runs at an early backbone layer (before MAH hooks) and
produces a soft assignment over K learned prototypes.  The resulting community
vector conditions how MAH computes divergence, so the same sign can produce
different divergence patterns in different community contexts.

Training signal: the community prototypes are pulled apart by the semiotic
losses — if assigning text to different communities helps the model predict
divergence better, it will learn to separate them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from srt.config import CommunityConfig


@dataclass
class CommunityOutput:
    """Output from community discovery.

    When the head runs in continuous-trajectory mode (cfg.use_prototypes=False,
    v8a), `logits` and `weights` are None and `vector == encoded`.
    """

    logits: torch.Tensor | None  # (B, K) raw assignment scores, or None
    weights: torch.Tensor | None  # (B, K) soft assignment probabilities, or None
    vector: torch.Tensor  # (B, d_community) community embedding (mixture or encoded)
    encoded: torch.Tensor  # (B, d_community) pre-prototype-mixing encoder output


class CommunityDiscoveryHead(nn.Module):
    """Soft clustering of hidden states into discourse communities.

    With cfg.use_prototypes=True (default): pooled hidden state → encoder →
    cosine similarity to K learned prototypes → soft assignment weights →
    weighted mixture of prototypes as the community vector. This is the
    v3–v7 architecture.

    With cfg.use_prototypes=False (v8a): pooled hidden state → encoder →
    the encoder output IS the community vector. No discrete basis. Motivated
    by the v7 PCA finding that prototype tensors barely move from random
    init; the encoder was already doing the discriminative work and the
    soft-argmax over K anchors was throwing information away.
    """

    def __init__(self, cfg: CommunityConfig, d_backbone: int) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        self.use_prototypes = cfg.use_prototypes

        # Encode pooled hidden states → community space
        self.encoder = nn.Sequential(
            nn.Linear(d_backbone, cfg.d_community),
            nn.SiLU(),
        )

        # Learnable community prototypes (only when enabled)
        if cfg.use_prototypes:
            self.prototypes = nn.Embedding(cfg.num_prototypes, cfg.d_community)
        else:
            self.prototypes = None  # type: ignore[assignment]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CommunityOutput:
        """Discover community from hidden states.

        Args:
            hidden_states: (B, T, d_backbone) from an early backbone layer.
            attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.

        Returns:
            CommunityOutput. In prototype mode, logits/weights are populated
            and vector is the prototype-weighted mixture. In trajectory mode
            (use_prototypes=False), logits and weights are None and vector
            equals encoded.
        """
        # Masked mean pool across positions → document-level representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)  # (B, d_backbone)
        encoded = self.encoder(pooled)  # (B, d_community)

        if not self.use_prototypes:
            # v8a: continuous-trajectory mode — no discrete basis.
            return CommunityOutput(
                logits=None, weights=None, vector=encoded, encoded=encoded,
            )

        # Cosine similarity to prototypes
        encoded_norm = F.normalize(encoded, dim=-1)
        proto_norm = F.normalize(self.prototypes.weight, dim=-1)
        logits = (encoded_norm @ proto_norm.T) / self.temperature  # (B, K)

        weights = F.softmax(logits, dim=-1)  # (B, K)
        vector = weights @ self.prototypes.weight  # (B, d_community)

        return CommunityOutput(
            logits=logits, weights=weights, vector=vector, encoded=encoded,
        )
