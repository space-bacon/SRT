"""Encode COCO images and captions through one backbone, for the joint space.

One process per GPU, one backbone each. Writes an npz of image and text states
at several depths so the joint fit can choose, rather than betting on one layer:
today's depth-probe result says the best layer is a property of the probe, not
of the backbone alone.

Layers are given as FRACTIONS of depth because these four models have different
layer counts, so an absolute index would tap a different relative position in
each and quietly confound the comparison.

    python scripts/omni_encode.py --model google/gemma-4-31B-it --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

BATCH_TXT = 16
MAX_SEQ = 64
CHUNK = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--img-dir", default="/root/val2017")
    p.add_argument("--caps", default="/root/annotations/captions_val2017.json")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--layer-fracs", default="0.4,0.6,0.75")
    p.add_argument("--out", default=None)
    return p.parse_args()


def pairs(caps_path: str, n: int):
    ann = json.load(open(caps_path))
    first: dict[int, str] = {}
    for c in sorted(ann["annotations"], key=lambda x: x["id"]):
        first.setdefault(c["image_id"], c["caption"].strip())
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    ids = sorted(i for i in first if i in id2file)[:n]
    return [id2file[i] for i in ids], [first[i] for i in ids]


def main() -> None:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    # Assign, do not setdefault: this box presets HF_HOME to an empty cache.
    os.environ["HF_HOME"] = os.environ.get("SRT_HF_HOME", "/root/.hf_home")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

    tag = a.model.split("/")[-1]
    out = a.out or f"/root/states_{tag}.npz"
    files, caps = pairs(a.caps, a.n)
    print(f"[{tag}] {len(files)} pairs on gpu {a.gpu}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=False)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=False)
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False).eval()

    n_layer = getattr(getattr(model.config, "text_config", model.config),
                      "num_hidden_layers", None)
    d_model = getattr(getattr(model.config, "text_config", model.config),
                      "hidden_size")
    fracs = [float(x) for x in a.layer_fracs.split(",")]
    layers = sorted({max(1, min(n_layer, round(f * n_layer))) for f in fracs})
    print(f"[{tag}] depth {n_layer}, d_model {d_model}, tapping {layers}", flush=True)

    tok = getattr(proc, "tokenizer", proc)
    img_tok = getattr(model.config, "image_token_id", None)
    img = np.zeros((len(files), len(layers), d_model), np.float16)
    txt = np.zeros((len(caps), len(layers), d_model), np.float16)

    t0 = time.time()
    with torch.no_grad():
        for i, fn in enumerate(files):
            im = Image.open(os.path.join(a.img_dir, fn)).convert("RGB")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": "Describe this image."}]}]
            enc = proc.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to("cuda")
            o = model(**enc, output_hidden_states=True, use_cache=False)
            ids = enc["input_ids"][0]
            mask = (ids == img_tok) if img_tok is not None else None
            for li, L in enumerate(layers):
                h = o.hidden_states[L][0]
                # Mean over image-token positions; fall back to all positions if
                # this processor does not expose a distinct image token id.
                v = h[mask].float().mean(0) if mask is not None and mask.any() \
                    else h.float().mean(0)
                img[i, li] = v.cpu().numpy()
            if (i + 1) % 500 == 0:
                r = (i + 1) / (time.time() - t0)
                print(f"[{tag}] img {i+1}/{len(files)} {r:.2f}/s "
                      f"eta {(len(files)-i-1)/r/60:.1f}m", flush=True)
            if (i + 1) % CHUNK == 0:
                np.savez(out.replace(".npz", ".partial.npz"),
                         img=img, txt=txt, done=i + 1, layers=layers)

        bos = tok.bos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else bos
        t0 = time.time()
        for i in range(0, len(caps), BATCH_TXT):
            chunk = caps[i:i + BATCH_TXT]
            idl = []
            for s in chunk:
                x = tok(s, truncation=True, max_length=MAX_SEQ,
                        add_special_tokens=True).input_ids
                if bos is not None and x and x[0] != bos:
                    x = [bos] + x[: MAX_SEQ - 1]
                idl.append(x)
            T = max(len(x) for x in idl)
            ii = torch.full((len(chunk), T), pad, dtype=torch.long)
            am = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, x in enumerate(idl):
                ii[j, : len(x)] = torch.tensor(x)
                am[j, : len(x)] = 1
            o = model(input_ids=ii.cuda(), attention_mask=am.cuda(),
                      output_hidden_states=True, use_cache=False)
            last = (am.sum(-1) - 1).cuda()
            rows = torch.arange(len(chunk)).cuda()
            for li, L in enumerate(layers):
                txt[i:i + len(chunk), li] = (
                    o.hidden_states[L][rows, last].float().cpu().numpy()
                    .astype(np.float16))
            if (i // BATCH_TXT) % 50 == 0:
                done = i + len(chunk)
                r = done / (time.time() - t0)
                print(f"[{tag}] txt {done}/{len(caps)} {r:.1f}/s", flush=True)

    np.savez_compressed(out, img=img, txt=txt, layers=np.array(layers),
                        d_model=d_model, n_layer=n_layer, model=a.model)
    p = out.replace(".npz", ".partial.npz")
    if os.path.exists(p):
        os.remove(p)
    print(f"[{tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
