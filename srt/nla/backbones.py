"""Backbone loading helpers for NLA scripts.

Multimodal hosts (gemma-4) do NOT load through ``AutoModelForCausalLM``:
that path maps no weights (checkpoint keys live under
``model.language_model.*``) and silently returns a randomly initialised
text tower — a hard-learned failure mode. ``load_frozen_backbone`` detects
the model type from its config and picks the correct class, and
``text_config``/``hidden_size_of``/``num_layers_of`` resolve the nested
text-tower config that multimodal wrappers use.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

#: model_type values that require the conditional-generation (multimodal)
#: class even for text-only use.
_MULTIMODAL_TYPES = {"gemma4", "gemma4_text_vision"}


def text_config(config):
    """Return the text-tower config (nested for multimodal wrappers)."""
    return getattr(config, "text_config", None) or config


def hidden_size_of(config) -> int:
    return text_config(config).hidden_size


def num_layers_of(config) -> int:
    return text_config(config).num_hidden_layers


def load_frozen_backbone(
    model_id: str,
    dtype: str | torch.dtype = "bfloat16",
    device: str | None = None,
):
    """Load a frozen backbone + tokenizer with the correct class.

    Returns ``(model, tokenizer)``. The model is eval-mode with
    ``requires_grad=False`` on every parameter.
    """
    if isinstance(dtype, str):
        dtype = _DTYPE_MAP[dtype]
    cfg = AutoConfig.from_pretrained(model_id)
    model_type = getattr(cfg, "model_type", "")

    if model_type in _MULTIMODAL_TYPES:
        # gemma-4: Gemma4ForCausalLM does NOT map checkpoint weights
        # (keys are model.language_model.*) — must use the conditional-
        # generation class even for text-only forwards.
        import transformers

        cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
        if cls is None:
            from transformers import AutoModelForImageTextToText as cls  # noqa: N813
        logger.info("loading %s via %s (multimodal host)", model_id, cls.__name__)
        try:
            model = cls.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = cls.from_pretrained(model_id, torch_dtype=dtype)
    else:
        from transformers import AutoModelForCausalLM

        logger.info("loading %s via AutoModelForCausalLM", model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if device is not None:
        model.to(device)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok
