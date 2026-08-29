#!/usr/bin/env python3
"""Encode radiographs keeping several layers and several poolings per image.

A 1024px chest film becomes 256 image tokens, so one token covers roughly a
64x64 block of the original. Mean-pooling those 256 into one vector is fine for
cardiomegaly, which is a property of the whole field, and close to fatal for a
nodule, which is a handful of pixels averaged against everything else. The
first probe showed exactly that ordering: diffuse findings near 0.85, focal
findings near 0.65.

So keep the poolings that a focal lesion could survive.

  mean   the current baseline, every image token averaged
  max    per dimension across tokens, so a lesion firing on a few tokens is not
         diluted by the 250 that show clear lung
  top16  mean of the sixteen tokens with the largest norm, a middle ground that
         is less brittle than max but still local

Layers are taken as fractions of depth, because the best layer is a property of
the probe rather than of the backbone, and one pass gets all of them.

The text tower is skipped: this probe never uses it.

    python scripts/cxr_encode_pool.py --gpu 0 --manifest ... --out ...
"""
import argparse
import json
import os
import time

import numpy as np

POOLS = ["mean", "max", "top16"]
TOPK = 16
CHUNK = 4000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--img-dir", default="/root/cxr14")
    p.add_argument("--layer-fracs", default="0.4,0.6,0.8")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["HF_HOME"] = os.environ.get("SRT_HF_HOME", "/root/.hf_home")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    rows = [r for r in json.load(open(a.manifest))["rows"]
            if r["modality"] == "image"]
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()

    tc = getattr(model.config, "text_config", model.config)
    n_layer, d = tc.num_hidden_layers, tc.hidden_size
    layers = sorted({max(1, min(n_layer, round(float(f) * n_layer)))
                     for f in a.layer_fracs.split(",")})
    img_tok = getattr(model.config, "image_token_id", None)
    print(f"depth {n_layer}, d {d}, layers {layers}, pools {POOLS}", flush=True)

    n = len(rows)
    item = np.zeros((n, len(layers), len(POOLS), d), np.float16)
    ok = np.zeros(n, bool)

    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                im = Image.open(os.path.join(a.img_dir, r["key"])).convert("RGB")
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": "Describe this image."}]}]
                enc = proc.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to("cuda")
                out = model(**enc, output_hidden_states=True, use_cache=False)
                mask = enc["input_ids"][0] == img_tok
                if not mask.any():
                    continue
                for li, L in enumerate(layers):
                    h = out.hidden_states[L][0][mask].float()
                    k = min(TOPK, h.shape[0])
                    top = h[h.norm(dim=-1).topk(k).indices].mean(0)
                    for pi, v in enumerate((h.mean(0), h.max(0).values, top)):
                        item[i, li, pi] = v.cpu().numpy()
                ok[i] = True
            except Exception as e:
                print(f"  skip {r['key']}: {type(e).__name__}", flush=True)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{n} {rate:.2f}/s ok={int(ok.sum())} "
                      f"eta {(n-i-1)/rate/60:.1f}m", flush=True)
            if (i + 1) % CHUNK == 0:
                np.savez(a.out.replace(".npz", ".partial.npz"),
                         item=item, ok=ok, layers=layers, pools=POOLS)

    np.savez(a.out, item=item, ok=ok, layers=np.array(layers),
             pools=np.array(POOLS), d_model=d, model=a.model)
    print(f"\n{int(ok.sum())}/{n} encoded -> {a.out}")


if __name__ == "__main__":
    main()
