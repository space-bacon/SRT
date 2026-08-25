#!/usr/bin/env python
"""PyTorch fp16 reference for the browser read-out, at deployment scale.

The published anchor table was measured on an earlier 4,000-image head against
a 1,000-image pool. This scores the SHIPPED head against the SHIPPED gallery so
the recovery fraction quoted for the deployment is the deployment's own.

Projection matches the runtime exactly (rust/srt-geometry/src/head.rs):
out = W (v - mu) + b, then L2 normalise. Pooling is mean over tokens, tap L28.

    python browser_anchor_reference.py \
        --head ../BlackWindow/wasm/dist/head.safetensors \
        --index artifacts/local/browser/gallery_123k_v3.srtidx \
        --replay artifacts/local/browser/replay_123k.json \
        --out artifacts/nla/q4/browser_fp16_reference_123k.json
"""
from __future__ import annotations

import argparse
import json
import struct

import numpy as np
import torch
from safetensors.torch import load_file


def load_index(path: str) -> tuple[np.ndarray, list[str]]:
    raw = open(path, "rb").read()
    dim, n = struct.unpack_from("<II", raw, 8)
    off = 16
    scales = np.frombuffer(raw, "<f4", n, off); off += 4 * n
    q = np.frombuffer(raw, np.int8, n * dim, off).reshape(n, dim); off += n * dim
    keys = []
    for _ in range(n):
        (l,) = struct.unpack_from("<I", raw, off); off += 4
        keys.append(raw[off:off + l].decode()); off += l
    z = q.astype(np.float32) * scales[:, None]
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z, keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    replay = json.load(open(a.replay))
    caps = replay["captions"]
    gal, keys = load_index(a.index)
    print(f"gallery {gal.shape}, captions {len(caps)}", flush=True)

    h = load_file(a.head)
    W = h["txt.weight"].float().numpy()
    b = h["txt.bias"].float().numpy()
    mu = h["mu_txt"].float().numpy()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()

    qs = np.zeros((len(caps), W.shape[0]), np.float32)
    with torch.no_grad():
        for i in range(0, len(caps), a.batch):
            chunk = [c["text"] for c in caps[i:i + a.batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=64).to(dev)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[a.layer].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = ((hs * m).sum(1) / m.sum(1)).cpu().numpy()
            v = (pooled - mu) @ W.T + b
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            qs[i:i + len(chunk)] = v
            if i % (a.batch * 20) == 0:
                print(f"  {i}/{len(caps)}", flush=True)

    gold = np.array([c["gold"] for c in caps])
    ranks = np.empty(len(caps), np.int64)
    for i in range(0, len(caps), 256):
        s = qs[i:i + 256] @ gal.T
        g = s[np.arange(len(s)), gold[i:i + 256]]
        ranks[i:i + 256] = (s > g[:, None]).sum(1)
    res = {
        "runtime": f"pytorch fp16 {a.model} L{a.layer} on {dev}",
        "head": a.head, "index": a.index,
        "n_images": len(keys), "n_captions": len(caps),
        "t2i": {f"r@{k}": round(float((ranks < k).mean()), 4) for k in (1, 5, 10)},
        "median_rank": float(np.median(ranks)) + 1.0,
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
