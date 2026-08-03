"""Build the full-COCO head-space image gallery for the Sunstone server.

Downloads the 24 banked train2017 encoding chunks (RiverRider/
srt-nla-gemma4-artifacts, ~5GB one-time, HF-cached) plus the val2017
calibration cache, projects every image state through the shipped linear
head, and writes one compact search index:

    artifacts/local/gallery_full.npz
        Z_img  [N, 1024] fp16   L2-normalized head-space vectors
        files  [N] str          COCO basenames
        split  [N] str          "train2017" | "val2017"

N is ~122k. The server loads this at startup when present and search
becomes one [N, 1024] @ [1024] matmul, same as before, just 24x the pool.

    .venv/bin/python scripts/build_full_gallery.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("gallery")

HEAD_REPO = "RiverRider/srt-sunstone-linear-head"
HEAD_FILE = "sunstone_linear_head.pt"
ART_REPO = "RiverRider/srt-nla-gemma4-artifacts"
VAL_FILE = "procrustes/encoded_L47_n5000.pt"
OUT = Path("artifacts/local/gallery_full.npz")


def project(v: np.ndarray, W: np.ndarray, b: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z = (v - mu) @ W.T + b
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    head = torch.load(hf_hub_download(HEAD_REPO, HEAD_FILE),
                      map_location="cpu", weights_only=True)
    W = head["img"]["weight"].float().numpy()
    b = head["img"]["bias"].float().numpy()
    mu = head["mu_img"].float().numpy()

    zs, files, splits = [], [], []

    val = torch.load(hf_hub_download(ART_REPO, VAL_FILE),
                     map_location="cpu", weights_only=True)
    zs.append(project(val["img"].float().numpy(), W, b, mu).astype(np.float16))
    files += [Path(f).name for f in val["files"]]
    splits += ["val2017"] * len(val["files"])
    log.info("val2017: %d images", len(val["files"]))

    chunks = sorted(f for f in list_repo_files(ART_REPO) if "train_pairs/chunk_" in f)
    log.info("train2017: %d chunks", len(chunks))
    for i, name in enumerate(chunks):
        ch = torch.load(hf_hub_download(ART_REPO, name),
                        map_location="cpu", weights_only=True)
        zs.append(project(ch["img"].float().numpy(), W, b, mu).astype(np.float16))
        files += [Path(f).name for f in ch["files"]]
        splits += ["train2017"] * len(ch["files"])
        log.info("chunk %d/%d done (total %d images)", i + 1, len(chunks), len(files))

    Z = np.concatenate(zs, axis=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, Z_img=Z, files=np.array(files), split=np.array(splits))
    log.info("wrote %s: Z_img %s fp16, %d files", OUT, Z.shape, len(files))


if __name__ == "__main__":
    main()
