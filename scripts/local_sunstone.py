"""SRT-Sunstone local runtime: gemma-4-31B (4-bit MLX) + linear head on a Mac.

The home-computer realization of the instrumented-appliance architecture
(docs/CROSSMODAL_LINEAR_HEAD.md Tier 3+): one quantized backbone on Apple
Silicon serving chat + captioning natively, with the cross-modal linear
head reading layer-47 hidden states from the same forward passes.

MLX models are Python end-to-end, so the "edge state-tap" that requires
ggml surgery in llama.cpp is a list slice here: we run the text tower's
layers manually up to the tap layer.

Modes:
  --probe            print model structure (find the layer list + config)
  --validate N       encode N captions locally, compare against the HF
                     bf16 calibration cache (centered cosine per row).
                     This is the on-device version of the Q4-drift test.
  --retrieve IMAGE   image -> top-5 captions from the local 5k index,
                     through the shipped linear head
  --chat "PROMPT"    sanity: native generation still works

    python scripts/local_sunstone.py --probe
    python scripts/local_sunstone.py --validate 64
    python scripts/local_sunstone.py --retrieve photo.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
TAP_LAYER = 47          # hidden_states[47] in HF convention = output of block 46
HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEAD_FILE = "sunstone_linear_head.pt"
CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--layer", type=int, default=TAP_LAYER)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--validate", type=int, default=0, metavar="N")
    p.add_argument("--retrieve", type=Path, default=None, metavar="IMAGE")
    p.add_argument("--chat", type=str, default=None)
    p.add_argument("--max-seq-len", type=int, default=64)
    return p.parse_args()


def load_model(model_id: str):
    from mlx_vlm import load

    model, processor = load(model_id)
    return model, processor


def find_text_layers(model):
    """Locate the text tower's decoder layer list, embed fn, and norm."""
    lm = getattr(model, "language_model", None) or model
    inner = getattr(lm, "model", None) or lm
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("could not find decoder layers; run --probe")
    embed = getattr(inner, "embed_tokens", None)
    return inner, layers, embed


def probe(model) -> None:
    def walk(obj, prefix="model", depth=0):
        if depth > 3:
            return
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            tname = type(child).__name__
            if tname in ("method", "builtin_function_or_method", "function"):
                continue
            if hasattr(child, "__len__") and "layers" in name:
                print(f"{prefix}.{name}: list of {len(child)} x "
                      f"{type(child[0]).__name__}")
            elif "nn." in str(type(child)) or hasattr(child, "parameters"):
                print(f"{prefix}.{name}: {tname}")
                walk(child, f"{prefix}.{name}", depth + 1)
    walk(model)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in ("hidden_size", "num_hidden_layers", "image_token_id",
                  "text_config", "model_type"):
            print("config:", k, "=", getattr(cfg, k, None))


def encode_text_states(model, processor, texts: list[str], layer: int,
                       max_seq_len: int):
    """Last-token hidden state at `layer` for each text, BOS-prefixed.

    Uses the MLX port's built-in tap: ``capture_layer_ids`` appends the
    running hidden state at loop entry of block ``idx``, which equals HF
    ``hidden_states[idx]`` (embed scaling and sliding/full masks are
    handled inside the model).
    """
    import mlx.core as mx
    import numpy as np

    tok = getattr(processor, "tokenizer", processor)
    lm = getattr(model, "language_model", None) or model

    out = []
    for t in texts:
        ids = tok.encode(t)[: max_seq_len]
        bos = getattr(tok, "bos_token_id", None) or 2
        if ids[0] != bos:
            ids = [bos] + ids[: max_seq_len - 1]
        res = lm(mx.array([ids]), capture_layer_ids=[layer])
        h = res.hidden_states[0]
        mx.eval(h)
        out.append(np.array(h[0, -1, :].astype(mx.float32)))
    return np.stack(out)


def encode_image_state(model, processor, img, layer: int):
    """Mean hidden_states[layer] over image-token positions (Sunstone
    convention, same as sunstone_server.encode_image_local and the CUDA
    reference in gemma4_procrustes_xmodal.image_v)."""
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    cfg = getattr(model, "config", None)
    img_tok = getattr(cfg, "image_token_id", None)
    if img_tok is None:
        img_tok = 258880
    prompt = apply_chat_template(processor, cfg, "Describe this image.",
                                 num_images=1)
    inputs = prepare_inputs(processor, images=[img], prompts=[prompt])
    input_ids = inputs["input_ids"]
    kwargs = {k: v for k, v in inputs.items()
              if k not in ("input_ids", "pixel_values", "attention_mask")}
    res = model(input_ids, inputs.get("pixel_values"),
                capture_layer_ids=[layer], **kwargs)
    h = res.hidden_states[0]
    mx.eval(h)
    h = np.array(h[0].astype(mx.float32))
    mask = np.array(input_ids[0]) == img_tok
    if not mask.any():
        raise ValueError("no image tokens in the processed prompt")
    return h[mask].mean(0)


def main() -> None:
    args = parse_args()

    if args.chat is not None:
        from mlx_vlm import generate, load

        model, processor = load(args.model)
        print(generate(model, processor, args.chat, max_tokens=200))
        return

    model, processor = load_model(args.model)

    if args.probe:
        probe(model)
        return

    if args.validate:
        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download

        calib = torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                           map_location="cpu", weights_only=True)
        n = args.validate
        # cap5 rows correspond to captions5 (5 per eval image, in order)
        texts = [c for caps in calib["captions5"] for c in caps][:n]
        ref = calib["cap5"].float().numpy()[:n]                    # bf16 HF states
        loc = encode_text_states(model, processor, texts, args.layer,
                                 args.max_seq_len)                 # local Q4 states

        # The linear head this validation exists to serve (2026-08-05:
        # a reader correctly noted the head was declared but never
        # loaded here, so "head-space agreement" was actually raw
        # centered state space; all four arms now reported).
        hd = torch.load(hf_hub_download(HEAD_REPO, HEAD_FILE),
                        map_location="cpu", weights_only=True)
        W = hd["txt"]["weight"].float().numpy()
        b = hd["txt"]["bias"].float().numpy()
        mu_h = hd["mu_txt"].float().numpy()

        def agree_at_1(a, bb):
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            bn = bb / (np.linalg.norm(bb, axis=1, keepdims=True) + 1e-8)
            return float((np.argmax(an @ bn.T, axis=1) == np.arange(len(a))).mean())

        mu_r, mu_l = ref.mean(0), loc.mean(0)
        cos = ((ref - mu_r) * (loc - mu_l)).sum(1) / (
            np.linalg.norm(ref - mu_r, axis=1)
            * np.linalg.norm(loc - mu_l, axis=1) + 1e-8)
        print(f"n={n}  centered cos local-vs-HF: mean={cos.mean():.4f} "
              f"p10={np.percentile(cos, 10):.4f} min={cos.min():.4f}")
        print(f"self-retrieval@1, raw as-is:        "
              f"{agree_at_1(loc, ref):.3f}")
        print(f"self-retrieval@1, mean-centered:    "
              f"{agree_at_1(loc - mu_l, ref - mu_r):.3f}")
        print(f"self-retrieval@1, head (train mu):  "
              f"{agree_at_1((loc - mu_h) @ W.T + b, (ref - mu_h) @ W.T + b):.3f}")
        print(f"self-retrieval@1, head (local mu):  "
              f"{agree_at_1((loc - mu_l) @ W.T + b, (ref - mu_r) @ W.T + b):.3f}")
        print("(protocol: self-retrieval over the full n-pool, local queries "
              "reference; n=5000 is the reportable setting)")
        return

    if args.retrieve is not None:
        import numpy as np
        import torch
        from PIL import Image
        from huggingface_hub import hf_hub_download

        calib = torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                           map_location="cpu", weights_only=True)
        hd = torch.load(hf_hub_download(HEAD_REPO, HEAD_FILE),
                        map_location="cpu", weights_only=True)

        def project(X, side, mu):
            W = hd[side]["weight"].float().numpy()
            b = hd[side]["bias"].float().numpy()
            Z = (X - mu) @ W.T + b
            return Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-8)

        captions = [c for caps in calib["captions5"] for c in caps]
        Zt = project(calib["cap5"].float().numpy(), "txt",
                     hd["mu_txt"].float().numpy())

        img = Image.open(args.retrieve).convert("RGB")
        v = encode_image_state(model, processor, img, args.layer)
        z = project(v[None, :], "img", hd["mu_img"].float().numpy())[0]

        sims = Zt @ z
        for rank, i in enumerate(np.argsort(-sims)[:5], 1):
            print(f"{rank}. [{sims[i]:.4f}] {captions[i]}")


if __name__ == "__main__":
    from mlx_vlm import load  # noqa: F401  (fail fast if mlx-vlm missing)
    main()
