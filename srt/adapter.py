"""SRT Adapter — Semiotic awareness bolted onto any frozen causal LM.

The adapter wraps a HuggingFace AutoModelForCausalLM and runs its layers
manually, tapping hidden states at MAH hook points and injecting corrections
at RRM injection points.  The backbone's native embeddings and LM head are
used directly — no bridges, no tied embeddings, no CE degradation.

    model = SRTAdapter(config)
    out = model(input_ids, labels=labels)
    # out.ce_loss  — from backbone's native LM head
    # out.r_hat    — per-position reflexivity estimate
    # out.regime   — subcritical vs supercritical classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from srt.config import SRTConfig
from srt.modules.mah import MetapragmaticAttentionHead, MAHOutput
from srt.modules.rrm import ReflexiveRecurrentModule
from srt.modules.ben import BifurcationEstimationNetwork, BENOutput
from srt.modules.community import CommunityDiscoveryHead, CommunityOutput

logger = logging.getLogger(__name__)


@dataclass
class SRTAdapterOutput:
    """Full output from the SRT adapter."""

    logits: torch.Tensor  # (B, T, V)
    ce_loss: torch.Tensor | None = None  # scalar
    divergences: list[torch.Tensor] = field(default_factory=list)  # [(B, T, d_div)]
    injections: list[torch.Tensor] = field(default_factory=list)  # [(B, T, d_backbone)]
    ben_output: BENOutput | None = None
    community_output: CommunityOutput | None = None
    meta_state: torch.Tensor | None = None  # (B, T, d_meta)


def _make_causal_mask(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create 4D additive causal attention mask."""
    mask = torch.full(
        (seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]  # (1, 1, T, T)


class SRTAdapter(nn.Module):
    """Semiotic-Reflexive Transformer adapter for any causal LM backbone."""

    def __init__(self, config: SRTConfig) -> None:
        super().__init__()
        self.config = config

        # ── Load and freeze backbone ─────────────────────────────────
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        load_dtype = dtype_map.get(config.backbone_dtype, torch.bfloat16)

        logger.info("Loading backbone: %s in %s", config.backbone_id, config.backbone_dtype)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.backbone_id, torch_dtype=load_dtype
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Extract backbone parts (works for LLaMA, Qwen, Mistral, Phi, Gemma)
        inner = self.backbone.model
        self._embed_tokens = inner.embed_tokens
        self._layers = inner.layers
        self._final_norm = inner.norm
        self._lm_head = self.backbone.lm_head
        self._rotary_emb = getattr(inner, "rotary_emb", None)

        d_backbone = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers
        self._d_backbone = d_backbone
        self._num_layers = num_layers

        # Resolve auto layer indices
        config.resolve_layer_indices(num_layers)

        logger.info(
            "Backbone: d=%d, L=%d, MAH@%s, inject@%s, community@%d",
            d_backbone,
            num_layers,
            config.mah_layer_indices,
            config.rrm_inject_indices,
            config.community_layer_idx,
        )

        # ── Community discovery (early layer) ────────────────────────
        self.community_head = CommunityDiscoveryHead(config.community, d_backbone)

        # ── MAH heads (one per hook layer) ───────────────────────────
        self.mah_heads = nn.ModuleList([
            MetapragmaticAttentionHead(
                config.mah, d_backbone, d_community=config.community.d_community
            )
            for _ in config.mah_layer_indices
        ])

        # ── RRM ──────────────────────────────────────────────────────
        self.rrm = ReflexiveRecurrentModule(
            config.rrm, d_divergence=config.mah.d_divergence, d_backbone=d_backbone
        )

        # Chain predictor: predict next divergence from current (self-supervised)
        self.chain_predictor = nn.Linear(
            config.mah.d_divergence, config.mah.d_divergence, bias=False
        )

        # ── BEN ──────────────────────────────────────────────────────
        self.ben = BifurcationEstimationNetwork(config.ben, d_meta=config.rrm.d_meta)

        # Build lookup sets for fast layer-index checking
        self._mah_set = set(config.mah_layer_indices)
        self._inject_set = set(config.rrm_inject_indices)
        self._mah_index_map = {idx: i for i, idx in enumerate(config.mah_layer_indices)}

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        logger.info(
            "SRT Adapter: %s trainable, %s frozen (backbone)",
            f"{trainable:,}",
            f"{frozen:,}",
        )

        # Cast adapter modules to backbone dtype so bf16 hidden states flow
        # through without dtype mismatch (backbone is frozen bf16, adapter
        # modules default to float32)
        for module in [
            self.community_head, self.mah_heads, self.rrm,
            self.chain_predictor, self.ben,
        ]:
            module.to(load_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        forced_community: torch.Tensor | None = None,
    ) -> SRTAdapterOutput:
        """Forward pass: backbone with semiotic taps and injections.

        Args:
            input_ids: (B, T) token ids.
            attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.
            labels: (B, T) target token ids for CE loss. Optional.
            forced_community: (B, d_community) override community vector. Optional.
                When provided, uses this instead of CommunityDiscoveryHead output
                for conditioning MAH heads. Discovery still runs for diagnostics.

        Returns:
            SRTAdapterOutput with logits, losses, and semiotic intermediates.
        """
        device = input_ids.device
        B, T = input_ids.shape

        # 1. Native backbone embeddings
        h = self._embed_tokens(input_ids)

        # 2. Prepare position embeddings
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = None
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(h, position_ids)

        # 3. Causal mask for MAH attention
        mah_causal_mask = _make_causal_mask(T, h.dtype, device)

        # 4. Prepare 4D causal+padding mask for backbone layers
        # Must combine causal mask (T, T) with padding mask (B, T) into (B, 1, T, T)
        # so that SDPA doesn't drop is_causal=True behavior
        causal_4d = _make_causal_mask(T, h.dtype, device)  # (1, 1, T, T)
        backbone_mask = None
        if attention_mask is not None:
            # (B, T) → (B, 1, 1, T) padding mask
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(
                h.dtype
            ).min
            backbone_mask = causal_4d + pad_mask  # (B, 1, T, T)
        else:
            backbone_mask = causal_4d  # (1, 1, T, T) — causal only

        # 5. Layer-by-layer forward with semiotic taps
        divergences: list[torch.Tensor] = []
        injections: list[torch.Tensor] = []
        meta_state: torch.Tensor | None = None
        community_out: CommunityOutput | None = None
        community_vec: torch.Tensor | None = None
        mah_idx = 0

        for layer_i, layer in enumerate(self._layers):
            # Run backbone layer
            layer_kwargs: dict = {"position_ids": position_ids}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            if backbone_mask is not None:
                layer_kwargs["attention_mask"] = backbone_mask

            layer_out = layer(h, **layer_kwargs)
            h = layer_out[0]

            # Community discovery at early layer
            if layer_i == self.config.community_layer_idx and community_out is None:
                community_out = self.community_head(h.detach(), attention_mask)
                # Use forced_community override if provided, else discovered
                community_vec = (
                    forced_community if forced_community is not None
                    else community_out.vector
                )

            # MAH hook: extract divergence
            if layer_i in self._mah_set:
                mah_head = self.mah_heads[self._mah_index_map[layer_i]]
                mah_out = mah_head(h, community_vec=community_vec, causal_mask=mah_causal_mask)
                divergences.append(mah_out.divergence)

                # Update RRM meta-state
                meta_state = self.rrm.step(mah_out.divergence, meta_state)

                # RRM injection (if this is also an injection layer)
                if layer_i in self._inject_set:
                    inj = self.rrm.inject(meta_state, h)
                    h = h + inj
                    injections.append(inj)

        # 6. Final norm + native LM head
        h = self._final_norm(h)
        logits = self._lm_head(h)

        # 7. CE loss (shifted, standard next-token prediction)
        ce_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 8. BEN
        ben_out = None
        if meta_state is not None:
            ben_out = self.ben(meta_state)

        return SRTAdapterOutput(
            logits=logits,
            ce_loss=ce_loss,
            divergences=divergences,
            injections=injections,
            ben_output=ben_out,
            community_output=community_out,
            meta_state=meta_state,
        )

    # Adapter module prefixes for save/load (everything else is backbone)
    _ADAPTER_PREFIXES = (
        "community_head.", "mah_heads.", "rrm.", "chain_predictor.", "ben.",
    )

    def save_adapter(self, path: str) -> None:
        """Save only the trainable adapter weights (not the backbone)."""
        state = {
            k: v for k, v in self.state_dict().items()
            if k.startswith(self._ADAPTER_PREFIXES)
        }
        torch.save(state, path)
        logger.info("Saved adapter weights (%d tensors) to %s", len(state), path)

    def load_adapter(self, path: str) -> None:
        """Load adapter weights (backbone loaded separately from HF)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        # Expected: all non-adapter keys will be "missing" (loaded from HF)
        adapter_missing = [k for k in missing if k.startswith(self._ADAPTER_PREFIXES)]
        if adapter_missing:
            logger.warning("Missing adapter keys: %s", adapter_missing)
        logger.info("Loaded adapter weights from %s", path)

    def trainable_parameters(self):
        """Yield only the trainable (adapter) parameters."""
        return (p for p in self.parameters() if p.requires_grad)
