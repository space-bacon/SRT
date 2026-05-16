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

        # Trainable extra prefix embeddings that follow the injected vector.
        # These give the model a small "header" before it starts generating
        # describing text. Initialised from the backbone's BOS embedding
        # (cast back to float32 for optimizer stability — see note above).
        n_pref = max(0, cfg.num_prefix_tokens)
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
            init = bos_embed.expand(n_pref, -1).clone()
            self.prefix_embeds = nn.Parameter(init)
        else:
            self.register_parameter("prefix_embeds", None)

    # ─────────────────────────── helpers ────────────────────────────

    def _inject_prefix(self, v: torch.Tensor) -> torch.Tensor:
        """Build ``(B, P, d_embed)`` injection prefix from vectors ``(B, d_vec)``.

        Adapter params are float32; the output is cast to the backbone's
        dtype at this boundary so ``inputs_embeds`` matches the frozen
        backbone weights without forcing the optimizer onto low precision.
        """
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v32 = v.float()
        inject = self.proj(v32).unsqueeze(1)  # (B, 1, d) float32
        if self.prefix_embeds is None:
            return inject.to(self._backbone_dtype)
        extra = self.prefix_embeds.unsqueeze(0).expand(v.size(0), -1, -1)
        return torch.cat([inject, extra], dim=1).to(self._backbone_dtype)

    @property
    def prefix_length(self) -> int:
        return 1 + (0 if self.prefix_embeds is None else self.prefix_embeds.size(0))

    # ───────────────────────── generation ───────────────────────────

    @torch.no_grad()
    def generate(
        self,
        v: torch.Tensor,
        max_new_tokens: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Return generated token ids ``(B, T_new)`` (without the prefix)."""
        inputs_embeds = self._inject_prefix(v)
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
