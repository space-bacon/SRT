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
    """Output from community discovery."""

    logits: torch.Tensor  # (B, K) raw assignment scores
    weights: torch.Tensor  # (B, K) soft assignment probabilities
    vector: torch.Tensor  # (B, d_community) weighted community embedding


class CommunityDiscoveryHead(nn.Module):
    """Soft clustering of hidden states into discourse communities."""

    def __init__(self, cfg: CommunityConfig, d_backbone: int) -> None:
        super().__init__()
        self.temperature = cfg.temperature

        # Encode pooled hidden states → community space
        self.encoder = nn.Sequential(
            nn.Linear(d_backbone, cfg.d_community),
            nn.SiLU(),
        )

        # Learnable community prototypes
        self.prototypes = nn.Embedding(cfg.num_prototypes, cfg.d_community)

    def forward(self, hidden_states: torch.Tensor) -> CommunityOutput:
        """Discover community from hidden states.

        Args:
            hidden_states: (B, T, d_backbone) from an early backbone layer.

        Returns:
            CommunityOutput with soft assignment and community vector.
        """
        # Pool across positions → document-level representation
        pooled = hidden_states.mean(dim=1)  # (B, d_backbone)
        encoded = self.encoder(pooled)  # (B, d_community)

        # Cosine similarity to prototypes
        encoded_norm = F.normalize(encoded, dim=-1)
        proto_norm = F.normalize(self.prototypes.weight, dim=-1)
        logits = (encoded_norm @ proto_norm.T) / self.temperature  # (B, K)

        weights = F.softmax(logits, dim=-1)  # (B, K)
        vector = weights @ self.prototypes.weight  # (B, d_community)

        return CommunityOutput(logits=logits, weights=weights, vector=vector)
