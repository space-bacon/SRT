"""Abort-first check: does Qwen3-Omni give usable states for image, audio and video?

Everything downstream assumes we can pull one vector per item, in any modality,
from a single omni backbone. That assumption has never been tested here, so this
runs before the expensive encode and fails loudly rather than quietly.

Passing means, per modality: states have the right shape, are finite, and two
different items do not collapse to the same vector. The collapse check matters
because a broken modality path often returns the prompt's states unchanged, which
looks fine until every item retrieves identically.

    python scripts/omni_smoke.py --gpu 0
"""
from __future__ import annotations

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--img-dir", default="/root/val2017")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    # Assign, do not setdefault: this box presets HF_HOME=/workspace/.hf_home, so a
    # setdefault silently points at an empty cache and re-downloads 66GB.
    os.environ["HF_HOME"] = os.environ.get("SRT_HF_HOME", "/root/.hf_home")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    print(f"transformers {transformers.__version__}", flush=True)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=False)
    arch = getattr(cfg, "architectures", ["?"])
    print(f"architectures: {arch}", flush=True)

    # Omni models are containers: the top class has no forward, and calling it
    # raises _forward_unimplemented. The thinker is the understanding half that
    # ingests image/audio/video/text; the talker only synthesises speech, which
    # retrieval does not need, so loading the thinker alone is correct and lighter.
    cls = None
    for name in arch:
        thinker = name.replace("ForConditionalGeneration",
                               "ThinkerForConditionalGeneration")
        if thinker != name and hasattr(transformers, thinker):
            cls = getattr(transformers, thinker)
            print(f"using transformers.{thinker} (thinker half)", flush=True)
            break
        cls = getattr(transformers, name, None)
        if cls is not None:
            print(f"using transformers.{name}", flush=True)
            break
    if cls is None:
        from transformers import AutoModel
        cls = AutoModel
        print("falling back to AutoModel", flush=True)

    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=False)
    print(f"processor: {type(proc).__name__}", flush=True)
    for attr in ("image_processor", "feature_extractor", "video_processor",
                 "audio_processor", "tokenizer"):
        print(f"  has {attr}: {hasattr(proc, attr)}", flush=True)

    model = cls.from_pretrained(a.model, dtype=torch.bfloat16,
                                device_map="cuda:0",
                                trust_remote_code=False).eval()
    tcfg = getattr(model.config, "thinker_config", None) or model.config
    text_cfg = getattr(tcfg, "text_config", tcfg)
    n_layer = getattr(text_cfg, "num_hidden_layers", None)
    d_model = getattr(text_cfg, "hidden_size", None)
    print(f"depth {n_layer}, d_model {d_model}", flush=True)

    from PIL import Image
    imgs = sorted(os.listdir(a.img_dir))[:2]
    results = {}

    tcfg = getattr(model.config, "thinker_config", None) or model.config
    TOK = {"image": getattr(tcfg, "image_token_id", None),
           "audio": getattr(tcfg, "audio_token_id", None),
           "video": getattr(tcfg, "video_token_id", None)}
    print(f"content token ids: {TOK}", flush=True)

    def probe(tag, content_a, content_b):
        """Two different items must not produce the same vector.

        Pools over CONTENT tokens only. Averaging all positions lets the shared
        chat template and prompt dominate, which made unrelated items look
        near-identical (cos 0.999) on the first version of this check.
        """
        vecs = []
        for content in (content_a, content_b):
            msg = [{"role": "user", "content": content}]
            try:
                enc = proc.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
            except Exception as e:
                results[tag] = f"FAIL processor: {type(e).__name__}: {e}"
                return
            enc = {k: (v.to("cuda") if hasattr(v, "to") else v)
                   for k, v in enc.items()}
            try:
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True,
                                use_cache=False, return_dict=True)
            except Exception as e:
                results[tag] = f"FAIL forward: {type(e).__name__}: {e}"
                return
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                results[tag] = "FAIL: no hidden_states returned"
                return
            L = len(hs) // 2
            h = hs[L][0]
            tid = TOK.get(tag)
            ids = enc["input_ids"][0]
            mask = (ids == tid) if tid is not None else None
            if mask is not None and mask.any():
                v = h[mask].float().mean(0)
                n_content = int(mask.sum())
            else:
                # Text has no content token; use the last position instead of a
                # mean, so the prompt does not swamp it.
                v = h[-1].float()
                n_content = 1
            vecs.append((v.cpu().numpy(), n_content))
        (v1, n1), (v2, n2) = vecs
        finite = bool(np.isfinite(v1).all() and np.isfinite(v2).all())
        cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        mu = (v1 + v2) / 2.0
        c1, c2 = v1 - mu, v2 - mu
        cen = float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9))
        ok = finite and cos < 0.99
        results[tag] = (f"{'PASS' if ok else 'WEAK'} dim={v1.shape[0]} "
                        f"content_tok={n1}/{n2} raw_cos={cos:.4f} centered={cen:+.4f}")

    probe("text",
          [{"type": "text", "text": "A dog running on a beach."}],
          [{"type": "text", "text": "A plate of pasta on a table."}])

    probe("image",
          [{"type": "image", "image": Image.open(os.path.join(a.img_dir, imgs[0])).convert("RGB")},
           {"type": "text", "text": "Describe this."}],
          [{"type": "image", "image": Image.open(os.path.join(a.img_dir, imgs[1])).convert("RGB")},
           {"type": "text", "text": "Describe this."}])

    # Synthetic audio: two clearly different tones, so a collapse is unambiguous.
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    tone_a = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone_b = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    probe("audio",
          [{"type": "audio", "audio": tone_a}, {"type": "text", "text": "Describe this."}],
          [{"type": "audio", "audio": tone_b}, {"type": "text", "text": "Describe this."}])

    # Synthetic video: 8 frames, two different colour ramps.
    va = [Image.new("RGB", (224, 224), (10 * i, 0, 0)) for i in range(8)]
    vb = [Image.new("RGB", (224, 224), (0, 0, 10 * i)) for i in range(8)]
    probe("video",
          [{"type": "video", "video": va}, {"type": "text", "text": "Describe this."}],
          [{"type": "video", "video": vb}, {"type": "text", "text": "Describe this."}])

    print("\n=== RESULT ===", flush=True)
    for k in ("text", "image", "audio", "video"):
        print(f"  {k:<6} {results.get(k, 'not run')}", flush=True)
    good = [k for k, v in results.items() if str(v).startswith("PASS")]
    print(f"\nusable modalities: {good}", flush=True)


if __name__ == "__main__":
    main()
