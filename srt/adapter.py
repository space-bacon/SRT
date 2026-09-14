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

import contextlib
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


def _make_sliding_window_mask(
    seq_len: int, window: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create a 4D additive sliding-window causal mask.

    Position i may attend to position j iff j <= i (causal) AND i - j < window.
    Used by backbones with alternating sliding/full attention (e.g. gpt-oss),
    where the sliding layers cannot be expressed by the is_causal fast path.
    """
    idx = torch.arange(seq_len, device=device)
    future = idx[None, :] > idx[:, None]                # j > i      → disallow
    too_far = (idx[:, None] - idx[None, :]) >= window   # i - j >= w → disallow
    disallowed = future | too_far
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask.masked_fill_(disallowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]  # (1, 1, T, T)


def _make_sliding_window_mask_cached(
    q_start: int, q_len: int, kv_len: int, window: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Sliding-window causal mask over a KV cache (prefill or decode).

    Query absolute positions are [q_start, q_start + q_len); key absolute
    positions are [0, kv_len). Query q attends to key k iff k <= q AND
    q - k < window. Returns (1, 1, q_len, kv_len).
    """
    q_idx = torch.arange(q_start, q_start + q_len, device=device)
    k_idx = torch.arange(kv_len, device=device)
    future = k_idx[None, :] > q_idx[:, None]
    too_far = (q_idx[:, None] - k_idx[None, :]) >= window
    disallowed = future | too_far
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(disallowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]  # (1, 1, q_len, kv_len)


def _layer_hidden(layer_out) -> torch.Tensor:
    """Extract hidden states from a decoder layer's return value.

    transformers has returned `tuple(hidden, ...)` historically, but some
    versions/architectures return the bare tensor. Indexing a tensor with
    [0] would silently slice the batch dim, so normalize here.
    """
    if isinstance(layer_out, torch.Tensor):
        return layer_out
    return layer_out[0]


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

    def __init__(self, config: SRTConfig, device_map: str | dict | None = None,
                 max_memory: dict | None = None) -> None:
        super().__init__()
        self.config = config

        # ── Load and freeze backbone ─────────────────────────────────
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        load_dtype = dtype_map.get(config.backbone_dtype, torch.bfloat16)

        logger.info("Loading backbone: %s in %s%s", config.backbone_id,
                    config.backbone_dtype,
                    f" (device_map={device_map})" if device_map else "")
        load_kwargs: dict = {"torch_dtype": load_dtype}
        if device_map is not None:
            # Sharded load (multi-GPU / CPU-offload). accelerate dispatches the
            # model and attaches AlignDevicesHooks that move each module's
            # inputs to its execution device, so the manual layer loop works
            # unchanged; we only have to manage the SRT-head boundary (see
            # set_head_device + the tap/inject movement in forward).
            load_kwargs["device_map"] = device_map
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.backbone_id, **load_kwargs
        )
        self._sharded = device_map is not None
        # When set, SRT head modules live on this device and taps are moved
        # to it (and injections moved back). None = single-device behaviour
        # (heads follow the adapter's device; no movement).
        self._head_device: torch.device | None = None
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

        # Alternating sliding-window / full attention (e.g. gpt-oss). When the
        # backbone declares per-layer attention types with a sliding window,
        # the manual loop must hand sliding layers an explicit banded mask;
        # full layers keep the is_causal fast path. All other backbones leave
        # this disabled, so their mask path is unchanged (byte-identical).
        bb_config = self.backbone.config
        self._layer_types: list[str] = list(getattr(bb_config, "layer_types", None) or [])
        self._sliding_window: int = int(getattr(bb_config, "sliding_window", 0) or 0)
        self._has_sliding: bool = bool(
            self._layer_types
            and self._sliding_window
            and any(lt == "sliding_attention" for lt in self._layer_types)
        )

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
        if self._has_sliding:
            n_slide = sum(1 for lt in self._layer_types if lt == "sliding_attention")
            logger.info(
                "Sliding-window attention: window=%d, %d/%d sliding layers",
                self._sliding_window, n_slide, len(self._layer_types),
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

    # ------------------------------------------------------- device plumbing
    def _head_modules(self) -> list[nn.Module]:
        return [self.community_head, self.mah_heads, self.rrm,
                self.chain_predictor, self.ben]

    def set_head_device(self, device: str | torch.device) -> None:
        """Pin the SRT head modules to one device (sharded-backbone mode).

        With a `device_map`-sharded backbone the adapter must NOT be moved
        wholesale with `.to(device)` (that would yank dispatched backbone
        weights off their assigned devices). Instead call this once after
        construction: head modules move to `device`, and forward()/
        _cached_step() route taps to it and injections back to each layer's
        device.
        """
        device = torch.device(device)
        for m in self._head_modules():
            m.to(device)
        self._head_device = device
        logger.info("SRT heads pinned to %s", device)

    def _to_head(self, t: torch.Tensor) -> torch.Tensor:
        """Move a tap to the head device (no-op when not sharded)."""
        if self._head_device is not None and t.device != self._head_device:
            return t.to(self._head_device)
        return t

    @property
    def _embed_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        forced_community: torch.Tensor | None = None,
        disable_injectors: bool = False,
        read_only: bool = False,
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
            read_only: when True, run the backbone layers (and final
                norm/LM head) under `torch.no_grad()` so no backbone
                activation graph is built, while the SRT heads (community,
                MAH, RRM, BEN) still build their own graph from the detached
                taps. Injections are applied with `.detach()` so forward
                behaviour matches deploy time without extending the graph
                into subsequent backbone layers. This is the Phase-A
                training mode for very large backbones: memory cost is
                inference-only regardless of backbone size.
        """
        # With a sharded backbone, route inputs to the embedding's device;
        # accelerate's AlignDevicesHooks then move per-layer inputs as the
        # hidden state crosses device boundaries.
        if self._sharded:
            input_ids = input_ids.to(self._embed_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._embed_device)
        device = input_ids.device
        B, T = input_ids.shape

        # 1. Native backbone embeddings
        h = self._embed_tokens(input_ids)

        # 2. Prepare position embeddings
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = None
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(h, position_ids)

        # 3. Causal mask for MAH attention (lives on the head device when
        # the backbone is sharded; MAH heads consume taps there).
        mah_causal_mask = _make_causal_mask(
            T, h.dtype, self._head_device or device
        )

        # 4. Prepare backbone attention mask.
        # When there is NO padding mask, pass None: transformers' SDPA path
        # then uses is_causal=True, which is exactly what the backbone's own
        # forward does. Passing an explicit additive causal mask instead is
        # mathematically equivalent but takes a different kernel path whose
        # bf16 rounding differs; on deep MoE backbones (94 layers x discrete
        # expert routing) that epsilon amplifies into real logit divergence
        # (seen on Qwen3-235B: 4/34 top-1 flips incl. a 0.885-margin one).
        # The explicit 4D mask is only needed to combine causal + padding.
        backbone_mask = None
        sliding_mask = None
        if attention_mask is not None:
            causal_4d = _make_causal_mask(T, h.dtype, device)  # (1, 1, T, T)
            # (B, T) → (B, 1, 1, T) padding mask
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(
                h.dtype
            ).min
            backbone_mask = causal_4d + pad_mask  # (B, 1, T, T)
            if self._has_sliding:
                sliding_mask = _make_sliding_window_mask(
                    T, self._sliding_window, h.dtype, device
                ) + pad_mask
        elif self._has_sliding:
            # gpt-oss: HF's own forward builds explicit per-layer masks rather
            # than relying on the is_causal fast path, and full vs sliding
            # layers need different masks. Match HF exactly: full-attention
            # layers get a plain causal mask, sliding layers a banded one.
            # (Plain dense backbones keep backbone_mask=None → is_causal.)
            backbone_mask = _make_causal_mask(T, h.dtype, device)
            sliding_mask = _make_sliding_window_mask(
                T, self._sliding_window, h.dtype, device
            )

        # 5. Layer-by-layer forward with semiotic taps
        divergences: list[torch.Tensor] = []
        injections: list[torch.Tensor] = []
        meta_state: torch.Tensor | None = None
        community_out: CommunityOutput | None = None
        community_vec: torch.Tensor | None = None
        mah_idx = 0

        # In read-only mode the backbone runs without building a graph; the
        # SRT heads operate on detached taps and build their own (tiny) graph.
        bb_ctx = torch.no_grad() if read_only else contextlib.nullcontext()

        for layer_i, layer in enumerate(self._layers):
            # Run backbone layer
            layer_kwargs: dict = {"position_ids": position_ids}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            # Per-layer attention mask. On backbones with alternating
            # sliding/full attention (gpt-oss), sliding layers get the banded
            # mask while full layers fall back to backbone_mask (None →
            # is_causal when unpadded). On every other backbone _has_sliding
            # is False, so this is simply backbone_mask as before.
            layer_mask = backbone_mask
            if self._has_sliding and self._layer_types[layer_i] == "sliding_attention":
                layer_mask = sliding_mask
            if layer_mask is not None:
                layer_kwargs["attention_mask"] = layer_mask

            with bb_ctx:
                layer_out = layer(h, **layer_kwargs)
            h = _layer_hidden(layer_out)
            tap = h.detach() if read_only else h
            tap = self._to_head(tap)

            # Community discovery at early layer
            if layer_i == self.config.community_layer_idx and community_out is None:
                community_out = self.community_head(
                    self._to_head(h.detach()),
                    self._to_head(attention_mask) if attention_mask is not None else None,
                )
                # Use forced_community override if provided, else discovered
                community_vec = (
                    forced_community if forced_community is not None
                    else community_out.vector
                )
                if community_vec is not None:
                    community_vec = self._to_head(community_vec)

            # MAH hook: extract divergence
            if layer_i in self._mah_set:
                mah_head = self.mah_heads[self._mah_index_map[layer_i]]
                mah_out = mah_head(tap, community_vec=community_vec, causal_mask=mah_causal_mask)
                divergences.append(mah_out.divergence)

                # Update RRM meta-state
                meta_state = self.rrm.step(mah_out.divergence, meta_state)

                # RRM injection (if this is also an injection layer)
                if layer_i in self._inject_set:
                    inj = self.rrm.inject(meta_state, tap)
                    if not disable_injectors:
                        # In read-only mode the injection is applied detached:
                        # forward behaviour matches deploy time, but the graph
                        # does not extend into subsequent backbone layers.
                        inj_h = inj.detach() if read_only else inj
                        h = h + inj_h.to(h.device)
                    injections.append(inj)

        # 6. Final norm + native LM head
        with bb_ctx:
            h = self._final_norm(h)
            logits = self._lm_head(h)

        # 7. CE loss (shifted, standard next-token prediction).
        # Computed in batch chunks: cross_entropy upcasts logits to fp32, and
        # at large batch (e.g. 128 x 511 x 151936 vocab) the single allocation
        # is ~19 GB — OOMs the last shard GPU. Chunking bounds it to ~2.4 GB.
        ce_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            chunk = 16
            losses = []
            weights = []
            for i in range(0, shift_logits.shape[0], chunk):
                lg = shift_logits[i:i + chunk].contiguous()
                lb = shift_labels[i:i + chunk].contiguous()
                n_valid = (lb != -100).sum()
                if n_valid == 0:
                    continue
                losses.append(F.cross_entropy(
                    lg.view(-1, lg.size(-1)), lb.view(-1), ignore_index=-100,
                ))
                weights.append(n_valid)
            if losses:
                w = torch.stack([x.float() for x in weights])
                ce_loss = (torch.stack(losses) * w).sum() / w.sum()

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
        if self._sharded:
            input_ids = input_ids.to(self._embed_device)
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
        # pass None so SDPA uses its is_causal fast path — numerically
        # identical to the backbone's own forward (see note in forward()).
        backbone_mask = None
        sliding_mask = None
        if self._has_sliding:
            # gpt-oss: match HF's explicit per-layer masks over the KV cache.
            # Full layers get a causal mask (window == kv_len disables the
            # band), sliding layers get the banded window mask.
            kv_len = start_pos + T
            backbone_mask = _make_sliding_window_mask_cached(
                start_pos, T, kv_len, kv_len, h.dtype, device,
            )
            sliding_mask = _make_sliding_window_mask_cached(
                start_pos, T, kv_len, self._sliding_window, h.dtype, device,
            )

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
            layer_mask = backbone_mask
            if self._has_sliding and self._layer_types[layer_i] == "sliding_attention":
                layer_mask = sliding_mask
            if layer_mask is not None:
                layer_kwargs["attention_mask"] = layer_mask

            layer_out = layer(h, **layer_kwargs)
            h = _layer_hidden(layer_out)
            tap = self._to_head(h)

            # Community discovery: compute once during prefill, then freeze.
            if layer_i == self.config.community_layer_idx and community_vec is None:
                community_out = self.community_head(tap.detach(), None)
                community_vec = community_out.vector

            if layer_i in self._mah_set:
                mah_head = self.mah_heads[self._mah_index_map[layer_i]]
                divergence, new_kv = mah_head.forward_step(
                    tap, community_vec=community_vec, kv_cache=mah_kv.get(layer_i)
                )
                mah_kv[layer_i] = new_kv
                divergences.append(divergence)

                meta_state = self.rrm.step(divergence, meta_state)

                if layer_i in self._inject_set:
                    inj = self.rrm.inject(meta_state, tap)
                    if not disable_injectors:
                        h = h + inj.to(h.device)

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
            # Additive fields for introspection/streaming UIs (do not remove the
            # above keys: online_hallucination.py depends on them).
            per_layer = [float(d[0, -1].norm()) for d in divergences]
            signals["per_layer"] = per_layer
            signals["div_total"] = float(sum(per_layer))
            signals["regime"] = int(ben_out.regime_logits[0, -1].argmax())

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
