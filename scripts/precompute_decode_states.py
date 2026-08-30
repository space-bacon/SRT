#!/usr/bin/env python3
"""Precompute what the 31B contributes, so the demo can stop paying for it live.

The Space currently loads a 4-bit gemma-4-31B on ZeroGPU, runs it to produce one
5376-dimensional vector, and then does the interesting part, a 0.6B reading that
vector, in milliseconds. It spends its whole budget on the boring half and pays
twice for it: a cold start and a queue on every visit, and a quantisation the
card admits changes the reader's wording.

walk-the-space solved this already. Precompute the expensive half, ship a small
reader, run on cpu-basic, and the demo becomes something you play rather than
something you request. This produces that precomputed half at FULL precision,
which also retires the NF4 caveat.

Everything needed is already in the Space repo, so there is no COCO download and
no manifest to rebuild: 1000 indexed gallery images plus 8 examples deliberately
held outside the index.

Also caches the host's own caption. The exhibit compares what gemma says with
what the 0.6B reader recovers from the vector alone, and that comparison has to
survive gemma not being present at runtime.

Layers beyond 47 are stored for a depth exhibit, with a caveat that belongs on
the face of it: the verbalizer was TRAINED at layer 47, so reading other depths
measures how far they sit from layer 47 in the reader's frame. It does not
measure where meaning lives, and must not be captioned that way.

    python scripts/precompute_decode_states.py --out /root/decode_states.npz
"""
import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

SPACE = "RiverRider/0.6b-decodes-31b"
HOST = "google/gemma-4-31B-it"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--space", default=SPACE)
    p.add_argument("--model", default=HOST)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 24, 36, 47, 56])
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="/root/decode_states.npz")
    return p.parse_args()


def main():
    a = parse_args()
    local = snapshot_download(a.space, repo_type="space",
                              allow_patterns=["gallery/*", "examples/*"])
    items = []
    for sub in ("gallery", "examples"):
        d = os.path.join(local, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((sub, f, os.path.join(d, f)))
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} images from {a.space}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model)
    host = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    img_tok = host.config.image_token_id
    n_layers = (getattr(host.config, "num_hidden_layers", None)
                or host.config.text_config.num_hidden_layers)
    bad = [l for l in a.layers if l > n_layers]
    if bad:
        raise SystemExit(f"layers {bad} exceed the model's {n_layers}")
    print(f"  {n_layers} layers, taking {a.layers}", flush=True)

    V = np.zeros((len(items), len(a.layers), host.config.text_config.hidden_size
                  if hasattr(host.config, "text_config")
                  else host.config.hidden_size), np.float16)
    caps, meta = {}, []
    for i, (sub, name, path) in enumerate(items):
        im = Image.open(path).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = host.generate(**enc, max_new_tokens=a.max_new_tokens,
                                do_sample=False, output_hidden_states=True,
                                return_dict_in_generate=True)
        mask = enc["input_ids"][0] == img_tok
        for j, l in enumerate(a.layers):
            # hidden_states[0] is the prefill, so this predates the first token.
            V[i, j] = out.hidden_states[0][l][0][mask].mean(0).float().cpu().numpy()
        caps[name] = proc.decode(out.sequences[0][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        meta.append({"file": name, "set": sub})
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)}", flush=True)

    np.savez_compressed(a.out, states=V, layers=np.array(a.layers),
                        files=np.array([m["file"] for m in meta]),
                        sets=np.array([m["set"] for m in meta]))
    json.dump(caps, open(a.out.replace(".npz", "_captions.json"), "w"), indent=1)
    print(f"\nwrote {a.out}  {V.shape}  {os.path.getsize(a.out) / 1e6:.1f} MB")
    print(f"wrote {a.out.replace('.npz', '_captions.json')}  {len(caps)} captions")


if __name__ == "__main__":
    main()
