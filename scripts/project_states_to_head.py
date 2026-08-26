#!/usr/bin/env python
"""Project raw gemma-4-31B L47 image states into the Lab's own 1024-d space.

The Lab serves `sunstone_linear_head_v3_drift.pt`, and its gallery is that
head's projection of 123,287 COCO images, L2 normalized. The reader currently
attached to the Lab reads RAW 5376-d states instead, which is a different
space, so it cannot say what is at a point on a map of the Lab's gallery.

This writes the training inputs for a reader that lives in the Lab's space:
the same projection, the same normalization, from the same raw states the
gallery was built from. A reader trained on a different representation than it
is served would underperform silently, which is the whole reason to do the
projection here rather than approximately.

    python scripts/project_states_to_head.py \
        --states /root/full_img_vecs.npy --out /root/sunstone_img_vecs.npy
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--states", default="/root/full_img_vecs.npy")
    p.add_argument("--head-repo", default="RiverRider/srt-sunstone-linear-head")
    p.add_argument("--head-file", default="sunstone_linear_head_v3_drift.pt")
    p.add_argument("--side", choices=("img", "txt"), default="img")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--out", default="/root/sunstone_img_vecs.npy")
    a = p.parse_args()

    from huggingface_hub import hf_hub_download

    d = torch.load(hf_hub_download(a.head_repo, a.head_file),
                   map_location="cpu", weights_only=True)
    W = d[a.side]["weight"].float().numpy()
    b = d[a.side]["bias"].float().numpy()
    mu = d[f"mu_{a.side}"].float().numpy()
    print(f"head {a.side}: {W.shape[1]} -> {W.shape[0]}, meta {d.get('meta')}",
          flush=True)

    X = np.load(a.states, mmap_mode="r")
    print(f"states {X.shape} {X.dtype}", flush=True)
    if X.shape[1] != W.shape[1]:
        raise SystemExit(f"state width {X.shape[1]} != head input {W.shape[1]}")

    out = np.empty((X.shape[0], W.shape[0]), np.float32)
    for i in range(0, len(X), a.batch):
        v = (X[i:i + a.batch].astype(np.float32) - mu) @ W.T + b
        # The gallery ships L2 normalized and the runtime does not renormalize,
        # so the reader must be trained on unit vectors too.
        out[i:i + a.batch] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        if i % (a.batch * 4) == 0:
            print(f"  {min(i + a.batch, len(X))}/{len(X)}", flush=True)

    np.save(a.out, out)
    print(f"wrote {a.out} {out.shape}", flush=True)
    print(f"norm check: mean {np.linalg.norm(out[:1000], axis=1).mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
