#!/usr/bin/env python
"""Export MLX-Q4 L47 image states for the 1000 captioned eval images via
the running sunstone server (/encode_image). Output feeds
encode_ref_cuda.py --mlx-img-states.

Images are fetched one at a time from the public COCO server (or read
from --img-dir if present), posted to the loopback endpoint, and the
result is checkpointed every 25 images so the run survives interruption
and interleaves politely with live demo traffic (each encode is one
short GenSlot).

    .venv/bin/python scripts/export_mlx_image_states.py --out mlx_img_q4.npz
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download

CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"
SERVER = "http://127.0.0.1:8765"
COCO_URL = "http://images.cocodataset.org/val2017/{name}"
IMG_N = 1000


def fetch_image(name: str, img_dir: str | None) -> bytes:
    if img_dir:
        path = os.path.join(img_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    r = requests.get(COCO_URL.format(name=name), timeout=60)
    r.raise_for_status()
    return r.content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mlx_img_q4.npz")
    p.add_argument("--img-dir", default=None,
                   help="local val2017 dir (falls back to COCO http)")
    p.add_argument("--checkpoint-every", type=int, default=25)
    args = p.parse_args()

    calib = torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                       map_location="cpu", weights_only=True)
    names = [f.rsplit("/", 1)[-1] for f in calib["files"][:IMG_N]]

    states: list[np.ndarray] = []
    start = 0
    part = args.out + ".part.npz"
    if os.path.exists(part):
        z = np.load(part)
        states = list(z["states"])
        start = len(states)
        print(f"resuming from checkpoint: {start} done")

    print(f"{len(names)} images via {SERVER}/encode_image (from {start})")
    for i in range(start, len(names)):
        blob = fetch_image(names[i], args.img_dir)
        r = requests.post(f"{SERVER}/encode_image",
                          files={"image": (names[i], io.BytesIO(blob),
                                           "image/jpeg")}, timeout=600)
        r.raise_for_status()
        states.append(np.array(r.json()["state"], dtype=np.float32))
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(names):
            np.savez_compressed(part, states=np.stack(states))
            print(f"  {i + 1}/{len(names)} (checkpointed)")

    X = np.stack(states)
    np.savez_compressed(args.out, states=X)
    os.remove(part)
    print(f"saved {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
