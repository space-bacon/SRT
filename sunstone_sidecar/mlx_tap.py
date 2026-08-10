"""MLX tap: the Apple Silicon tier of the deployment ladder.

The MLX gemma-4 port has a built-in state tap: `capture_layer_ids`
appends the running hidden state at loop entry of block idx, which
equals HF `hidden_states[idx]` (embed scaling and masks handled inside
the model). Validated: local Q4-MLX states through the head agree with
CUDA bf16 at 0.984 R@1 as-is and 1.000 R@1 after the 42KB mean
recalibration (see calibrate.py).

Requires `mlx-vlm` (Apple Silicon only). Same conventions as
taps.TransformersTap: image = mean over image-token states with the
"Describe this image." template; text = BOS-enforced last-token.
"""
from __future__ import annotations

import numpy as np

from .taps import DESCRIBE_PROMPT, MAX_SEQ_LEN


class MLXTap:
    """Standalone encoder over an mlx-vlm backbone (Apple Silicon)."""

    def __init__(self, model, processor, layer: int,
                 max_seq_len: int = MAX_SEQ_LEN):
        self.model = model
        self.proc = processor
        self.tok = getattr(processor, "tokenizer", processor)
        self.layer = layer
        self.max_seq_len = max_seq_len
        cfg = getattr(model, "config", None)
        self.image_token_id = getattr(cfg, "image_token_id", None)
        if self.image_token_id is None:
            raise ValueError("mlx model config lacks image_token_id")
        self._lm = getattr(model, "language_model", None) or model

    @classmethod
    def from_pretrained(cls, model_id: str, layer: int) -> "MLXTap":
        from mlx_vlm import load
        model, processor = load(model_id)
        return cls(model, processor, layer)

    @classmethod
    def attach(cls, model, processor, layer: int) -> "MLXTap":
        return cls(model, processor, layer)

    def image_vector(self, img) -> np.ndarray:
        import mlx.core as mx
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        cfg = getattr(self.model, "config", None)
        prompt = apply_chat_template(self.proc, cfg, DESCRIBE_PROMPT,
                                     num_images=1)
        inputs = prepare_inputs(self.proc, images=[img], prompts=[prompt])
        input_ids = inputs["input_ids"]
        kwargs = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "pixel_values", "attention_mask")}
        res = self.model(input_ids, inputs.get("pixel_values"),
                         capture_layer_ids=[self.layer], **kwargs)
        h = res.hidden_states[0]
        mx.eval(h)
        h = np.array(h[0].astype(mx.float32))
        mask = np.array(input_ids[0]) == self.image_token_id
        if not mask.any():
            raise ValueError("no image tokens in the processed prompt")
        return h[mask].mean(0)

    def text_vectors(self, texts: list[str]) -> np.ndarray:
        import mlx.core as mx

        bos = getattr(self.tok, "bos_token_id", None)
        out = []
        for t in texts:
            ids = self.tok.encode(t)[: self.max_seq_len]
            if bos is not None and ids and ids[0] != bos:
                ids = [bos] + ids[: self.max_seq_len - 1]
            res = self._lm(mx.array([ids]), capture_layer_ids=[self.layer])
            h = res.hidden_states[0]
            mx.eval(h)
            out.append(np.array(h[0, -1, :].astype(mx.float32)))
        return np.stack(out)
