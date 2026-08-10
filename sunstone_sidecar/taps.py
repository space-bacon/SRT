"""State taps: extract the layer-L hidden states the head reads.

Two paths:
- `TransformersTap`: owns a backbone (or attaches to one you already
  loaded) and encodes images/texts with the exact Sunstone conventions.
- `read()` helpers: the zero-marginal path. If your pipeline already ran
  a forward pass with output_hidden_states=True, these read the vector
  out of it with no additional model work.

Conventions (validated across the whole program; do not change lightly):
- image vector = mean of layer-L states over image-token positions,
  prompt = chat template + "Describe this image."
- text vector = layer-L state at the LAST token, BOS enforced when the
  tokenizer has one (bare re-encode drops gemma-4 replay 0.9986 -> 0.615).
"""
from __future__ import annotations

import numpy as np
import torch

DESCRIBE_PROMPT = "Describe this image."
MAX_SEQ_LEN = 64


def read_image_vector(hidden_states, input_ids: torch.Tensor,
                      layer: int, image_token_id: int) -> np.ndarray:
    """Zero-marginal image read from an existing forward pass.

    hidden_states: tuple/list from output_hidden_states=True.
    input_ids: (T,) or (1, T) for the same pass.
    """
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    mask = ids == image_token_id
    if not bool(mask.any()):
        raise ValueError("no image tokens in this pass; was an image attached?")
    return hidden_states[layer][0][mask].float().mean(0).cpu().numpy()


def read_text_vector(hidden_states, attention_mask: torch.Tensor,
                     layer: int) -> np.ndarray:
    """Zero-marginal text read: last-token state per row. Returns (n, d)."""
    h = hidden_states[layer]
    last = attention_mask.sum(-1) - 1
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, last].float().cpu().numpy()


class TransformersTap:
    """Standalone encoder over a transformers backbone (CUDA/CPU/MPS).

    Pass an existing (model, processor) pair with `attach` to avoid a
    second copy of the weights; or let `from_pretrained` load one.
    """

    def __init__(self, model, processor, layer: int,
                 max_seq_len: int = MAX_SEQ_LEN):
        self.model = model
        self.proc = processor
        self.tok = getattr(processor, "tokenizer", processor)
        self.layer = layer
        self.max_seq_len = max_seq_len
        self.image_token_id: int = getattr(model.config, "image_token_id",
                                            None)
        if self.image_token_id is None:
            raise ValueError("backbone config lacks image_token_id; "
                             "is this a vision-language model?")
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(cls, backbone: str, layer: int,
                        dtype=torch.bfloat16, device_map="auto",
                        quant4: bool = False) -> "TransformersTap":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(backbone)
        kw: dict = {"dtype": dtype, "device_map": device_map}
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(backbone, **kw)
        model.eval()
        return cls(model, proc, layer)

    @classmethod
    def attach(cls, model, processor, layer: int) -> "TransformersTap":
        """Attach to a backbone your service already has resident."""
        return cls(model, processor, layer)

    @torch.no_grad()
    def image_vector(self, img) -> np.ndarray:
        """img: PIL.Image (or anything the processor accepts)."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ]}]
        enc = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True, use_cache=False)
        return read_image_vector(out.hidden_states, enc["input_ids"],
                                 self.layer, self.image_token_id)

    @torch.no_grad()
    def text_vectors(self, texts: list[str]) -> np.ndarray:
        tok, bos = self.tok, self.tok.bos_token_id
        ids_list = []
        for t in texts:
            ids = tok(t, truncation=True, max_length=self.max_seq_len,
                      add_special_tokens=True).input_ids
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: self.max_seq_len - 1]
            ids_list.append(ids)
        T = max(len(i) for i in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (
            bos if bos is not None else 0)
        input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_list), T), dtype=torch.long)
        for j, ids in enumerate(ids_list):        # right-pad
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        out = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attn.to(self.device),
            output_hidden_states=True, use_cache=False)
        return read_text_vector(out.hidden_states, attn.to(self.device),
                                self.layer)
