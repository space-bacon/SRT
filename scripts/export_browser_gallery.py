"""Export the browser-rung gallery: the images a 0.6B text tower searches.

The image states were produced by a 27B tower on a datacenter GPU. Projecting
them through the browser head turns 1.6GB of hidden states into a few MB of
head-space vectors, which is the only part a visitor's browser ever sees. The
27B model is not shipped, downloaded, or called at query time.

Runs on the box that holds the states.

    python scripts/export_browser_gallery.py \
        --states /root/qwen38_coco_ls/images.pt \
        --captions /root/qwen38_coco_ls/captions.pt \
        --head browser_text_head_qwen3_0.6B.pt --layer 52 --out /root/browser_gallery
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--layer", type=int, default=52)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    head = torch.load(a.head, map_location="cpu", weights_only=True)
    W = head["img"]["weight"].float().numpy()
    b = head["img"]["bias"].float().numpy()
    mu = head["mu_img"].float().numpy()

    st = torch.load(a.states, map_location="cpu", weights_only=False)
    if a.layer not in st["layers"]:
        raise SystemExit(f"layer {a.layer} not in {st['layers']}")
    slot = st["layers"].index(a.layer)
    V = st["base"][:, slot, :].float().numpy()
    files = [str(f) for f in st["files"]]

    Z = (V - mu) @ W.T + b
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    np.savez_compressed(a.out / "gallery.npz",
                        Z_img=Z.astype(np.float16), files=np.array(files))
    print(f"gallery: {Z.shape} from layer {a.layer}")

    cap = torch.load(a.captions, map_location="cpu", weights_only=False)
    pairs = [{"text": t, "owner": o} for t, o in zip(cap["texts"], cap["owner"])]
    (a.out / "captions.json").write_text(json.dumps({
        "image_layer": a.layer,
        "text_tower": head["meta"]["text_tower"],
        "text_layer": head["meta"]["text_layer"],
        "n_images": len(files),
        "pairs": pairs,
    }))
    print(f"captions: {len(pairs)} for {len(set(p['owner'] for p in pairs))} images")


if __name__ == "__main__":
    main()
