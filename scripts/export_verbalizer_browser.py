#!/usr/bin/env python
"""Export a verbalizer prefix for the browser runtime.

The trainer stores a pickled torch checkpoint. The browser wants safetensors it
can mmap, at half precision, with the centring statistics alongside the weights
rather than in a separate file that can drift away from them.

Layout matches what the Rust side reads:
    net0.weight (hidden, d_in)      net0.bias (hidden)
    net2.weight (n_tok*d_model, hidden)   net2.bias (n_tok*d_model)
    mu (d_in)   sd ()   meta in the header
"""
from __future__ import annotations

import argparse
import json

import torch
from safetensors.torch import save_file


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    return p.parse_args()


def main():
    a = parse()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    dt = torch.float16 if a.dtype == "f16" else torch.float32

    p = ck["prefix"]
    tensors = {
        "net0.weight": p["net.0.weight"].to(dt).contiguous(),
        "net0.bias": p["net.0.bias"].to(dt).contiguous(),
        "net2.weight": p["net.2.weight"].to(dt).contiguous(),
        "net2.bias": p["net.2.bias"].to(dt).contiguous(),
        # Centring stays f32: it is 4 KB and it is applied before the first
        # matmul, where half precision would cost more than it saves.
        "mu": torch.as_tensor(ck["mu"]).float().contiguous(),
        "sd": torch.tensor(float(ck["sd"])),
    }
    meta = {
        "d_in": str(ck["d_in"]),
        "d_model": str(ck["d_model"]),
        "n_tok": str(ck["n_tok"]),
        "hidden": str(ck.get("hidden", 2048)),
        "dtype": a.dtype,
    }
    save_file(tensors, a.out, metadata=meta)
    n = sum(t.numel() for k, t in tensors.items() if k.startswith("net"))
    print(json.dumps(meta), f"-> {a.out}  ({n / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
