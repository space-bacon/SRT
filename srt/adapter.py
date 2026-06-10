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
    chain_residual_per_token: torch.Tensor | None = None  # (B, T) mean chain residual


def _make_causal_mask(
    seq_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create 4D additive causal attention mask."""
    mask = torch.full(
        (seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]  # (1, 1, T, T)


def _sample_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Sample a single next-token id from (1, V) logits. Greedy if temp<=0."""
    if not temperature or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens until cumulative prob first exceeds top_p (inclusive).
        cutoff = cumsum - sorted_probs > top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, choice)
    return torch.multinomial(probs, 1)


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
        disable_injectors: bool = False,
    ) -> SRTAdapterOutput:
        """Forward pass: backbone with semiotic taps and injections.

        Args:
            input_ids: (B, T) token ids.
            attention_mask: (B, T) padding mask (1 = real, 0 = pad). Optional.
            labels: (B, T) target token ids for CE loss. Optional.
            forced_community: (B, d_community) override community vector. Optional.
                When provided, uses this instead of CommunityDiscoveryHead output
                for conditioning MAH heads. Discovery still runs for diagnostics.
            disable_injectors: when True, the RRM injection step is skipped
                (h is not updated by `+ inj`). MAH divergences and BEN signals
                are still computed for inspection, but the logits come from
                the frozen backbone alone. Used by the demo to render a
                side-by-side "adapter off" trace.

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
                    if not disable_injectors:
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

        # Per-token chain residual: mean across consecutive divergence pairs of
        # squared error (chain_predictor(div_i) - div_{i+1})^2 averaged over
        # the divergence dim. Shape (B, T). Same quantity that chain_loss
        # reduces to a scalar; surfaced here for inference/probing.
        chain_res = None
        if len(divergences) >= 2:
            B_, T_, _ = divergences[0].shape
            acc = torch.zeros(B_, T_, dtype=divergences[0].dtype,
                              device=divergences[0].device)
            for i in range(len(divergences) - 1):
                pred = self.chain_predictor(divergences[i])
                acc = acc + (pred - divergences[i + 1]).pow(2).mean(dim=-1)
            chain_res = acc / (len(divergences) - 1)

        return SRTAdapterOutput(
            logits=logits,
            ce_loss=ce_loss,
            divergences=divergences,
            injections=injections,
            ben_output=ben_out,
            community_output=community_out,
            meta_state=meta_state,
            chain_residual_per_token=chain_res,
        )

    @torch.no_grad()
    def _cached_step(
        self,
        input_ids: torch.Tensor,
        backbone_cache,
        mah_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
        community_vec: torch.Tensor | None,
        start_pos: int,
        disable_injectors: bool,
        collect_signals: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        """One cached forward over `T` new tokens (prefill: T>1, decode: T=1).

        Mutates `backbone_cache` and `mah_kv` in place. Returns
        (logits (B, T, V), community_vec, signals) where community_vec is
        computed on the prefill call (when the incoming one is None) and passed
        through on decode calls so the discourse basin stays fixed after the
        prompt. When `collect_signals` is True, `signals` is a dict of the SRT
        side-channels at the LAST position (the newest token), each a python
        float, for online hallucination logging / reject-and-resample:
            r_hat        — BEN reflexivity estimate
            regime_super — softmax P(supercritical) from BEN
            div_norm     — L2 norm of the last MAH layer divergence
            chain        — chain residual (mean over consecutive MAH layers)
        Otherwise `signals` is None.
        """
        device = input_ids.device
        B, T = input_ids.shape

        h = self._embed_tokens(input_ids)

        position_ids = torch.arange(
            start_pos, start_pos + T, device=device
        ).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(start_pos, start_pos + T, device=device)

        position_embeddings = None
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(h, position_ids)

        # MAH attention mask is built per-head inside forward_step; here we only
        # need the backbone mask. With a KV cache and a single decode token no
        # mask is required (all cached keys are in the past). For prefill we
        # build the causal mask over the new block against itself.
        backbone_mask = None
        if T > 1:
            backbone_mask = _make_causal_mask(T, h.dtype, device)

        meta_state: torch.Tensor | None = None
        divergences: list[torch.Tensor] = []

        for layer_i, layer in enumerate(self._layers):
            layer_kwargs: dict = {
                "position_ids": position_ids,
                "past_key_value": backbone_cache,
                "use_cache": True,
                "cache_position": cache_position,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            if backbone_mask is not None:
                layer_kwargs["attention_mask"] = backbone_mask

            layer_out = layer(h, **layer_kwargs)
            h = layer_out[0]

            # Community discovery: compute once during prefill, then freeze.
            if layer_i == self.config.community_layer_idx and community_vec is None:
                community_out = self.community_head(h.detach(), None)
                community_vec = community_out.vector

            if layer_i in self._mah_set:
                mah_head = self.mah_heads[self._mah_index_map[layer_i]]
                divergence, new_kv = mah_head.forward_step(
                    h, community_vec=community_vec, kv_cache=mah_kv.get(layer_i)
                )
                mah_kv[layer_i] = new_kv
                divergences.append(divergence)

                meta_state = self.rrm.step(divergence, meta_state)

                if layer_i in self._inject_set:
                    inj = self.rrm.inject(meta_state, h)
                    if not disable_injectors:
                        h = h + inj

        h = self._final_norm(h)
        logits = self._lm_head(h)

        signals = None
        if collect_signals and meta_state is not None:
            ben_out = self.ben(meta_state)  # r_hat (B,T), regime_logits (B,T,2)
            r_hat_last = ben_out.r_hat[0, -1]
            regime_super = F.softmax(ben_out.regime_logits[0, -1], dim=-1)[1]
            div_norm_last = divergences[-1][0, -1].norm()
            chain_last = torch.zeros((), device=device, dtype=h.dtype)
            if len(divergences) >= 2:
                acc = torch.zeros((), device=device, dtype=h.dtype)
                for i in range(len(divergences) - 1):
                    pred = self.chain_predictor(divergences[i][0, -1])
                    acc = acc + (pred - divergences[i + 1][0, -1]).pow(2).mean()
                chain_last = acc / (len(divergences) - 1)
            signals = {
                "r_hat": float(r_hat_last),
                "regime_super": float(regime_super),
                "div_norm": float(div_norm_last),
                "chain": float(chain_last),
            }

        return logits, community_vec, signals

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_ids: set[int] | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        disable_injectors: bool = False,
        return_signals: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
        """KV-cached autoregressive generation through the SRT adapter.

        Uses a backbone `DynamicCache` plus a per-MAH-layer K/V cache so each
        decode step is O(1) in sequence length rather than O(T). The community
        vector is fixed after prefill. Supports batch size 1.

        Args:
            input_ids: (1, T) prompt token ids.
            max_new_tokens: cap on generated tokens.
            eos_token_ids: ids that terminate generation.
            temperature: 0 (or <=0) means greedy.
            top_p: nucleus sampling cutoff (1.0 disables).
            disable_injectors: True serves the bare frozen backbone (control).
            return_signals: when True, also return a per-generated-token list of
                signal dicts for online hallucination analysis. Each dict has:
                  token_id  — the generated token
                  ent       — entropy of the predictive distribution it came from
                  margin    — top-1 minus top-2 probability of that distribution
                  nll       — negative log-prob of the chosen token
                  r_hat, regime_super, div_norm, chain — SRT side-channels at the
                    position where the token is processed (next step)

        Returns:
            (1, n_new) generated token ids, or (ids, signals) if return_signals.
        """
        assert input_ids.shape[0] == 1, "generate() supports batch size 1"
        from transformers.cache_utils import DynamicCache

        device = input_ids.device
        eos_token_ids = eos_token_ids or set()

        backbone_cache = DynamicCache()
        mah_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Prefill.
        logits, community_vec, _ = self._cached_step(
            input_ids, backbone_cache, mah_kv,
            community_vec=None, start_pos=0, disable_injectors=disable_injectors,
        )
        cur_pos = input_ids.shape[1]
        new_ids: list[int] = []
        signal_log: list[dict] = []

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            # Predictive-uncertainty signals of the distribution we sample from.
            pred_sig = {}
            if return_signals:
                probs = F.softmax(next_logits, dim=-1)[0]
                logp = torch.log_softmax(next_logits, dim=-1)[0]
                ent = float(-(probs * logp).sum())
                top2 = torch.topk(probs, 2).values
                margin = float(top2[0] - top2[1])
                pred_sig = {"ent": ent, "margin": margin}

            next_id = _sample_token(next_logits, temperature, top_p)
            tok = int(next_id.item())
            new_ids.append(tok)
            if return_signals:
                pred_sig["token_id"] = tok
                pred_sig["nll"] = float(-torch.log_softmax(next_logits, dim=-1)[0, tok])

            if tok in eos_token_ids:
                if return_signals:
                    signal_log.append(pred_sig)
                break

            logits, community_vec, srt_sig = self._cached_step(
                next_id.view(1, 1), backbone_cache, mah_kv,
                community_vec=community_vec, start_pos=cur_pos,
                disable_injectors=disable_injectors,
                collect_signals=return_signals,
            )
            cur_pos += 1
            if return_signals:
                if srt_sig:
                    pred_sig.update(srt_sig)
                signal_log.append(pred_sig)

        ids = torch.tensor(new_ids, device=device, dtype=torch.long).view(1, -1)
        if return_signals:
            return ids, signal_log
        return ids

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
