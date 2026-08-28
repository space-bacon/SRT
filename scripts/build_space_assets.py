#!/usr/bin/env python3
"""Assemble the gallery the Space ships with: thumbnails plus the projected
vectors that were computed from bf16 states.

The Space cannot encode 1000 images at startup, so the index travels with it
and only the visitor's own picture is encoded live.

    python scripts/build_space_assets.py --n 1000 --out /root/space_assets
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/root/.hf_home")

import numpy as np
from PIL import Image

IMG_DIR = "/root/val2017"
CACHE = "/root/live_gallery.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--out", default="/root/space_assets")
    a = ap.parse_args()

    z = np.load(CACHE, allow_pickle=True)
    files, vecs = list(z["files"])[:a.n], z["vecs"][:a.n]

    thumbs = os.path.join(a.out, "gallery")
    os.makedirs(thumbs, exist_ok=True)
    total = 0
    for i, f in enumerate(files, 1):
        im = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
        im.thumbnail((a.px, a.px), Image.LANCZOS)
        dst = os.path.join(thumbs, f)
        im.save(dst, "JPEG", quality=82, optimize=True)
        total += os.path.getsize(dst)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    np.savez_compressed(os.path.join(a.out, "gallery.npz"),
                        files=np.array(files), vecs=vecs.astype(np.float32))
    npz = os.path.getsize(os.path.join(a.out, "gallery.npz"))
    print(f"\n{len(files)} thumbnails  {total/1e6:.1f} MB")
    print(f"gallery.npz  {npz/1e6:.1f} MB  vecs {vecs.shape}")


if __name__ == "__main__":
    main()
