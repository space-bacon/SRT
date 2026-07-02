"""ActivationVerbalizer (AV): inject a target vector, decode text.

The frozen backbone runs over a prefix consisting of:

    [proj(v) , learned_prefix_embeds..., generated_tokens...]

``proj`` is the only trainable module in N0–N2 (a linear map from the
vector space to the backbone embedding space). After N2 the SRT adapter
replaces ``proj`` with the full 12.7M-parameter adapter pipeline so the
divergence and community channels condition the generation.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla.config import NLAConfig

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ActivationVerbalizer(nn.Module):
    """Injection-prefix language generator.

    ``forward`` returns logits over a teacher-forced sequence (for training).
    ``generate`` returns sampled token ids (for evaluation / RL rollout).
    """

    def __init__(
        self,
        cfg: NLAConfig,
        backbone: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        dtype = _DTYPE_MAP.get(cfg.backbone_dtype, torch.bfloat16)
        if backbone is None:
            logger.info("AV: loading backbone %s (%s)", cfg.backbone_id, cfg.backbone_dtype)
            backbone = AutoModelForCausalLM.from_pretrained(
                cfg.backbone_id, torch_dtype=dtype
            )
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()
        self.backbone = backbone
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_id)
        self.tokenizer = tokenizer

        self._d_embed = self.backbone.config.hidden_size
        self._num_layers = self.backbone.config.num_hidden_layers
        self._backbone_dtype = dtype
        d_vec = cfg.d_vector or self._d_embed

        # Trainable projection from vector space → backbone embedding space.
        # Initialised to identity when shapes match; small random otherwise.
        # NOTE: adapter params are kept in float32 even when the backbone is
        # bf16/fp16 — AdamW on low-precision params is numerically unstable
        # and the eye-init drifts to noise within a few hundred steps. We
        # cast to backbone dtype only at the boundary inside _inject_prefix.
        self.proj = nn.Linear(d_vec, self._d_embed, bias=False)
        with torch.no_grad():
            if d_vec == self._d_embed:
                nn.init.eye_(self.proj.weight)
            else:
                nn.init.xavier_uniform_(self.proj.weight)

        # Multi-position injection: M independent projections of v placed at
        # the first M slots of the prefix. proj_extra holds slots 1..M-1
        # (proj covers slot 0). Initialised small so the model starts near
        # the single-slot baseline and the extra slots add information as
        # training progresses.
        n_inject = max(1, getattr(cfg, "num_inject_slots", 1))
        self._n_inject = n_inject
        if n_inject > 1:
            self.proj_extra = nn.ModuleList(
                [nn.Linear(d_vec, self._d_embed, bias=False) for _ in range(n_inject - 1)]
            )
            with torch.no_grad():
                for lin in self.proj_extra:
                    nn.init.xavier_uniform_(lin.weight, gain=0.1)
        else:
            self.proj_extra = None

        # Trainable extra prefix embeddings that follow the injected vector.
        # These give the model a small "header" before it starts generating
        # describing text.
        #
        # Two modes:
        #   "static" — learned (P, d_embed) tensor, shared across inputs.
        #              Initialised from the backbone's BOS embedding.
        #   "mlp"    — 2-layer MLP maps v → (P, d_embed) per-input. Init
        #              second layer to zeros so prefix starts equal to the
        #              BOS embedding (the bias of layer 2), then drifts.
        # Adapter params are float32 (see note above).
        n_pref = max(0, cfg.num_prefix_tokens)
        self._n_pref = n_pref
        self._prefix_mode = getattr(cfg, "prefix_mode", "static")
        if n_pref > 0:
            bos_id = (
                self.backbone.config.bos_token_id
                or self.tokenizer.bos_token_id
                or 0
            )
            with torch.no_grad():
                embed = self.backbone.get_input_embeddings()
                bos_embed = (
                    embed(torch.tensor([bos_id], device=embed.weight.device))
                    .detach()
                    .clone()
                    .float()
                )  # (1, d) float32
            if self._prefix_mode == "mlp":
                hidden = getattr(cfg, "prefix_mlp_hidden", 256)
                out_dim = n_pref * self._d_embed
                self.prefix_mlp = nn.Sequential(
                    nn.Linear(d_vec, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, out_dim),
                )
                with torch.no_grad():
                    # Zero output weights so prefix = bias (set to tiled BOS)
                    # — preserves warm-start behaviour at init.
                    nn.init.xavier_uniform_(self.prefix_mlp[0].weight)
                    nn.init.zeros_(self.prefix_mlp[0].bias)
                    nn.init.zeros_(self.prefix_mlp[2].weight)
                    self.prefix_mlp[2].bias.copy_(
                        bos_embed.expand(n_pref, -1).reshape(-1)
                    )
                self.register_parameter("prefix_embeds", None)
            else:
                init = bos_embed.expand(n_pref, -1).clone()
                self.prefix_embeds = nn.Parameter(init)
                self.prefix_mlp = None
        else:
            self.register_parameter("prefix_embeds", None)
            self.prefix_mlp = None
        # Optional layer-conditioning embedding. When the AV is trained on
        # activations drawn from *multiple* layers (the full input→output
        # trace), different layers have different scales and semantics; this
        # embedding tells the injection which layer v was read from. It is
        # added to every inject slot and zero-initialised so it starts as a
        # no-op (safe to enable on top of a single-layer warm-start).
        if getattr(cfg, "use_layer_embed", False):
            self.layer_embed = nn.Embedding(self._num_layers + 1, self._d_embed)
            with torch.no_grad():
                nn.init.zeros_(self.layer_embed.weight)
        else:
            self.layer_embed = None

        # Injection scale normalization (see NLAConfig.inject_norm). When
        # "embed", v is unit-normalized and rescaled to the backbone's mean
        # input-embedding row norm before projection, so the injected slot
        # lives at token-embedding scale regardless of the backbone's hidden-
        # state norms.
        self._inject_scale: float | None = None
        if getattr(cfg, "inject_norm", "none") == "embed":
            with torch.no_grad():
                emb_w = self.backbone.get_input_embeddings().weight
                self._inject_scale = float(emb_w.float().norm(dim=-1).mean())
            logger.info("AV inject_norm=embed: scaling v to %.3f", self._inject_scale)
    # ─────────────────────────── helpers ────────────────────────────

    def _inject_prefix(
        self, v: torch.Tensor, layer: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build ``(B, P, d_embed)`` injection prefix from vectors ``(B, d_vec)``.

        Adapter params are float32; the output is cast to the backbone's
        dtype at this boundary so ``inputs_embeds`` matches the frozen
        backbone weights without forcing the optimizer onto low precision.

        ``layer`` selects the layer-conditioning embedding (int, ``(B,)``
        tensor, or ``None`` to fall back to ``cfg.extraction_layer``). Ignored
        when the AV was built without ``use_layer_embed``.
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v32 = v.float()
        if self._inject_scale is not None:
            v32 = F.normalize(v32, dim=-1) * self._inject_scale
        # Slot 0 always uses self.proj; slots 1..M-1 use proj_extra.
        inject_list = [self.proj(v32).unsqueeze(1)]
        if self.proj_extra is not None:
            for lin in self.proj_extra:
                inject_list.append(lin(v32).unsqueeze(1))
        inject = torch.cat(inject_list, dim=1)  # (B, M, d) float32
        if self.layer_embed is not None:
            le = self._layer_embedding(layer, v32.size(0), v32.device)  # (B, d)
            inject = inject + le.unsqueeze(1)
        if self._n_pref == 0:
            return inject.to(self._backbone_dtype)
        if self.prefix_mlp is not None:
            extra = self.prefix_mlp(v32).view(v.size(0), self._n_pref, self._d_embed)
        else:
            extra = self.prefix_embeds.unsqueeze(0).expand(v.size(0), -1, -1)
        return torch.cat([inject, extra], dim=1).to(self._backbone_dtype)

    def _layer_embedding(
        self, layer: int | torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        """Return ``(B, d_embed)`` float32 layer embedding for ``layer``."""
        if layer is None:
            layer = self.cfg.extraction_layer
        if not torch.is_tensor(layer):
            layer_ids = torch.full((batch,), int(layer), dtype=torch.long, device=device)
        else:
            layer_ids = layer.to(device=device, dtype=torch.long)
            if layer_ids.dim() == 0:
                layer_ids = layer_ids.expand(batch)
        layer_ids = layer_ids.clamp(0, self._num_layers)
        return self.layer_embed(layer_ids).float()

    @property
    def prefix_length(self) -> int:
        return self._n_inject + self._n_pref

    # ───────────────────────── generation ───────────────────────────

    @torch.no_grad()
    def generate(
        self,
        v: torch.Tensor,
        max_new_tokens: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        layer: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return generated token ids ``(B, T_new)`` (without the prefix)."""
        inputs_embeds = self._inject_prefix(v, layer=layer)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        out = self.backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=top_p if top_p is not None else self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        # When inputs_embeds is used, HF returns only the new tokens.
        return out

    @torch.no_grad()
    def verbalize(
        self,
        v: torch.Tensor,
        **gen_kwargs,
    ) -> list[str]:
        """Convenience: vectors → text strings."""
        ids = self.generate(v, **gen_kwargs)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
