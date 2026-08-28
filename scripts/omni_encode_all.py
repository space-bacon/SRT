"""Encode every modality in the manifest through one omni backbone.

Produces one vector per item and one per caption, in the same space, so a later
linear head can be fitted across modalities. Images, audio and video are pooled
over their CONTENT tokens; captions use the last token. Pooling over all
positions instead lets the shared chat template dominate and makes unrelated
items look near-identical, which is what the first smoke test caught.

    python scripts/omni_encode_all.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--manifest", default="/root/omni_manifest.json")
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--layer-frac", type=float, default=0.6)
    p.add_argument("--video-frames", type=int, default=8)
    p.add_argument("--skip-modalities", default="",
                   help="comma list a backbone cannot ingest, e.g. audio for "
                        "gemma-4, which has an audio token id but no audio tower")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--out", default="/root/omni_states.npz")
    return p.parse_args()


def load_audio(path, sr=16000):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, duration=10.0)
    return y.astype("float32")


def load_video(path, n_frames):
    from PIL import Image
    import numpy as np
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        idx = np.linspace(0, len(vr) - 1, n_frames).astype(int)
        return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idx]
    except Exception:
        import av
        with av.open(path) as c:
            frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
        if not frames:
            raise RuntimeError("no frames")
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        return [frames[i] for i in idx]


def main() -> None:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["HF_HOME"] = os.environ.get("SRT_HF_HOME", "/root/.hf_home")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from PIL import Image

    cfg = AutoConfig.from_pretrained(a.model)
    arch = getattr(cfg, "architectures", [""])[0]
    thinker = arch.replace("ForConditionalGeneration", "ThinkerForConditionalGeneration")
    cls = getattr(transformers, thinker, None) or getattr(transformers, arch)
    print(f"loading {cls.__name__}", flush=True)
    proc = AutoProcessor.from_pretrained(a.model)
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16,
                                device_map="cuda:0").eval()

    tcfg = getattr(model.config, "thinker_config", None) or model.config
    text_cfg = getattr(tcfg, "text_config", tcfg)
    n_layer = text_cfg.num_hidden_layers
    d_model = text_cfg.hidden_size
    L = max(1, min(n_layer, round(a.layer_frac * n_layer)))
    TOK = {"image": getattr(tcfg, "image_token_id", None),
           "audio": getattr(tcfg, "audio_token_id", None),
           "video": getattr(tcfg, "video_token_id", None)}
    print(f"depth {n_layer}, d {d_model}, tap L{L}, content tokens {TOK}", flush=True)

    man = json.load(open(a.manifest))
    rows = man["rows"][: a.limit] if a.limit else man["rows"]
    skip = {s for s in a.skip_modalities.split(",") if s}
    if skip:
        before = len(rows)
        rows = [r for r in rows if r["modality"] not in skip]
        print(f"skipping {sorted(skip)}: {before} -> {len(rows)} rows", flush=True)
    n = len(rows)
    item = np.zeros((n, d_model), np.float16)
    text = np.zeros((n, d_model), np.float16)
    ok = np.zeros(n, bool)
    modality = [r["modality"] for r in rows]

    tok = getattr(proc, "tokenizer", proc)

    @torch.no_grad()
    def states(content, tag):
        msg = [{"role": "user", "content": content}]
        enc = proc.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
        # Pixel/audio tensors arrive fp32 while the model is bf16, and some
        # vision towers do not cast internally.
        enc = {k: (v.to("cuda", model.dtype)
                   if hasattr(v, "is_floating_point") and v.is_floating_point()
                   else v.to("cuda") if hasattr(v, "to") else v)
               for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False,
                    return_dict=True)
        h = out.hidden_states[L][0]
        tid = TOK.get(tag)
        if tid is not None:
            m = enc["input_ids"][0] == tid
            if m.any():
                return h[m].float().mean(0).cpu().numpy()
        return h[-1].float().cpu().numpy()

    t0 = time.time()
    per_mod = {}
    for i, r in enumerate(rows):
        mod, key = r["modality"], r["key"]
        try:
            if mod == "image":
                media = {"type": "image",
                         "image": Image.open(os.path.join(a.img_dir, key)).convert("RGB")}
            elif mod == "audio":
                media = {"type": "audio", "audio": load_audio(key)}
            else:
                media = {"type": "video", "video": load_video(key, a.video_frames)}
            item[i] = states([media, {"type": "text", "text": "Describe this."}], mod)
            text[i] = states([{"type": "text", "text": r["caption"]}], "text")
            ok[i] = True
        except Exception as e:
            if per_mod.get(f"{mod}_err", 0) < 3:
                print(f"  [{mod}] {type(e).__name__}: {str(e)[:110]}", flush=True)
            per_mod[f"{mod}_err"] = per_mod.get(f"{mod}_err", 0) + 1
        per_mod[mod] = per_mod.get(mod, 0) + 1

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            r_ = (i + 1) / el
            print(f"  {i+1}/{n} {r_:.2f}/s ok={int(ok[:i+1].sum())} "
                  f"eta {(n-i-1)/r_/60:.1f}m", flush=True)
        if (i + 1) % 1000 == 0:
            np.savez(a.out.replace(".npz", ".partial.npz"),
                     item=item, text=text, ok=ok, modality=modality, done=i + 1)

    np.savez_compressed(a.out, item=item, text=text, ok=ok,
                        modality=np.array(modality), layer=L,
                        d_model=d_model, model=a.model)
    print(f"\nencoded {int(ok.sum())}/{n}", flush=True)
    for m in ("image", "audio", "video"):
        sel = [j for j in range(n) if modality[j] == m]
        if sel:
            print(f"  {m}: {int(ok[sel].sum())}/{len(sel)} ok "
                  f"({per_mod.get(m+'_err',0)} errors)", flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
