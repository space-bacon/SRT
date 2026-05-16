"""ActivationReconstructor (AR): frozen-backbone hidden-state reader.

Zero learned parameters in the canonical setup — the "reconstruction" is
just the forward pass through layers 0..L of the frozen backbone.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from srt.nla.config import NLAConfig

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ActivationReconstructor(nn.Module):
    """Reads pooled hidden states at ``cfg.extraction_layer`` from a text input.

    Pass an existing ``backbone`` to share weights with an
    ``ActivationVerbalizer`` (recommended — saves 14GB of RAM).
    """

    def __init__(
        self,
        cfg: NLAConfig,
        backbone: AutoModelForCausalLM | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            dtype = _DTYPE_MAP.get(cfg.backbone_dtype, torch.bfloat16)
            logger.info("AR: loading backbone %s (%s)", cfg.backbone_id, cfg.backbone_dtype)
            backbone = AutoModelForCausalLM.from_pretrained(
                cfg.backbone_id, torch_dtype=dtype
            )
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()
        self.backbone = backbone
        self._d = self.backbone.config.hidden_size
        self._num_layers = self.backbone.config.num_hidden_layers
        if not (0 < cfg.extraction_layer <= self._num_layers):
            raise ValueError(
                f"extraction_layer={cfg.extraction_layer} out of range "
                f"(backbone has {self._num_layers} hidden layers)"
            )

    @property
    def d_vector(self) -> int:
        return self.cfg.d_vector or self._d

    def _pool(
        self,
        h_layer: torch.Tensor,
        attention_mask: torch.Tensor | None,
        mode: Literal["last", "mean", "first"],
    ) -> torch.Tensor:
        if mode == "first":
            return h_layer[:, 0]
        if mode == "mean":
            if attention_mask is None:
                return h_layer.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(h_layer.dtype)
            return (h_layer * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # default: "last" non-pad token
        if attention_mask is None:
            return h_layer[:, -1]
        lengths = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
        idx = torch.arange(h_layer.size(0), device=h_layer.device)
        return h_layer[idx, lengths]

    @torch.no_grad()
    def reconstruct(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pool: str | None = None,
    ) -> torch.Tensor:
        """Return pooled hidden state ``(B, d)`` at layer ``cfg.extraction_layer``."""
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states is a tuple of length num_hidden_layers + 1
        h_layer = out.hidden_states[self.cfg.extraction_layer]
        return self._pool(h_layer, attention_mask, pool or self.cfg.pool)  # type: ignore[arg-type]
