"""Ask gemma-4 (generative head, greedy) to caption image files.

Used for the autostereogram contrast: the raw stereogram vs the stereo-decoded
figure. The base model reads the flat raster, so it should fail / confabulate on
the raw stereogram and correctly name the recovered figure.

Usage (venv transformers>=5):
    python scripts/gemma_caption.py \
        --images artifacts/nla/gemma4/stereo/stereogram.png \
                 artifacts/nla/gemma4/stereo/disparity.png \
                 artifacts/nla/gemma4/stereo/control.png \
        --prompt "What is in this picture? Answer in one short sentence." \
        --out artifacts/nla/gemma4/stereo_captions.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--prompt", default="What is in this picture? Answer in one short sentence.")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--out", default="artifacts/nla/gemma4/stereo_captions.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from PIL import Image
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor

    proc = AutoProcessor.from_pretrained(MID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()

    results = {}
    for path in args.images:
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": args.prompt}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        text = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results[os.path.basename(path)] = text
        print(f"\n=== {os.path.basename(path)} ===\n{text}", flush=True)

    json.dump({"model": MID, "prompt": args.prompt, "captions": results},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
