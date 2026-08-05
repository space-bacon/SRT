#!/usr/bin/env python
"""Export MLX-Q4 L47 states for the 5000 calib captions via the running
sunstone server (/encode_texts). Output feeds encode_ref_cuda.py --mlx-states.

    .venv/bin/python scripts/export_mlx_states.py --out mlx_q4_states.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download

CALIB_REPO = "RiverRider/srt-nla-gemma4-artifacts"
CALIB_FILE = "procrustes/encoded_L47_n5000.pt"
SERVER = "http://127.0.0.1:8765"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mlx_q4_states.npz")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    calib = torch.load(hf_hub_download(CALIB_REPO, CALIB_FILE),
                       map_location="cpu", weights_only=True)
    texts = [c for caps in calib["captions5"] for c in caps]
    print(f"{len(texts)} captions via {SERVER}/encode_texts")

    chunks = []
    for i in range(0, len(texts), args.batch):
        r = requests.post(f"{SERVER}/encode_texts",
                          json={"texts": texts[i:i + args.batch],
                                "max_seq_len": 64}, timeout=1800)
        r.raise_for_status()
        chunks.append(np.array(r.json()["states"], dtype=np.float32))
        print(f"  {i + len(chunks[-1])}/{len(texts)}")
    X = np.concatenate(chunks)
    np.savez_compressed(args.out, states=X)
    print(f"saved {X.shape} -> {args.out}")


if __name__ == "__main__":
    main()
