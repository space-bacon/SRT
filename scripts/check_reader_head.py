#!/usr/bin/env python3
"""Check a reader head against its published evidence, with nothing but this file.

Two checks, both against files on the RiverRider/srt-reader-heads and
RiverRider/srt-browser-head-118k repositories:

  parity   re-encode the eight probe captions with the encoder, apply the head,
           and compare the embedding, the projection and the top-5 gallery rows
           with the recorded fixture (parity_<reader>.json)
  replay   score the 5,001 held-out captions (replay_123k.json) against every
           row of the 123,287-row gallery and report t2i R@1/5/10 and median
           rank, the numbers in the model card and reader_swap_123k.json

    pip install sentence-transformers safetensors huggingface_hub numpy torch
    python check_head.py --reader bge-small            # parity + replay
    python check_head.py --reader e5-base --skip-replay

The files are fetched from the Hub on first use (the gallery is 130 MB).
"""
from __future__ import annotations

import argparse
import json
import struct
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer

HEADS = "RiverRider/srt-reader-heads"
GALLERY = "RiverRider/srt-browser-head-118k"
READERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: "),
    "gte-base": ("thenlper/gte-base", ""),
}
MAGIC_F16, MAGIC_I8 = b"SRTIDX01", b"SRTIDX02"


def read_srtidx(path):
    b = open(path, "rb").read()
    magic, (dim, n) = b[:8], struct.unpack("<II", b[8:16])
    off = 16
    if magic == MAGIC_I8:
        scale = np.frombuffer(b[off:off + 4 * n], dtype="<f4"); off += 4 * n
        q = np.frombuffer(b[off:off + n * dim], dtype=np.int8).reshape(n, dim); off += n * dim
        Z = q.astype(np.float32) * scale[:, None]
    elif magic == MAGIC_F16:
        Z = np.frombuffer(b[off:off + 2 * n * dim], dtype="<f2").reshape(n, dim).astype(np.float32); off += 2 * n * dim
    else:
        raise SystemExit(f"bad magic {magic!r}")
    keys = []
    for _ in range(n):
        (L,) = struct.unpack("<I", b[off:off + 4]); off += 4
        keys.append(b[off:off + L].decode()); off += L
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(Z), keys


def project(head, e):
    p = (e - head["mu_txt"].float()) @ head["txt.weight"].float().T + head["txt.bias"].float()
    return torch.nn.functional.normalize(p, dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="bge-small", choices=list(READERS))
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    repo_id, prefix = READERS[a.reader]

    head = load_file(hf_hub_download(HEADS, f"text_head_{a.reader}_v3gallery.safetensors"))
    enc = SentenceTransformer(repo_id, device=a.device)
    enc.max_seq_length = 64
    encode = lambda texts: torch.tensor(enc.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=256, convert_to_numpy=True))
    print(f"reader {repo_id}  head {a.reader}  d_txt {head['txt.weight'].shape[1]} -> {head['txt.weight'].shape[0]}")

    Zg, keys = read_srtidx(hf_hub_download(GALLERY, "gallery_123k_v3.srtidx"))
    print(f"gallery {len(keys):,} rows x {Zg.shape[1]}")

    # parity: the fixture's eight probes, stage by stage
    try:
        fx = json.load(open(hf_hub_download(HEADS, f"parity_{a.reader}.json")))
    except Exception:
        fx = None
        print("parity: no fixture for this reader on the Hub (fixtures exist for bge-small and e5-base)")
    if fx:
        texts = [p["text"] for p in fx["probes"]]
        e = encode(texts); p = project(head, e)
        emb_gap = max(float((e[i] - torch.tensor(fx["probes"][i]["embedding"])).abs().max()) for i in range(len(texts)))
        proj_cos = min(float(torch.nn.functional.cosine_similarity(p[i], torch.nn.functional.normalize(torch.tensor(fx["probes"][i]["projection"]), dim=0), dim=0)) for i in range(len(texts)))
        top5 = (p @ Zg.T).topk(5, dim=1).indices
        same1 = sum(int(top5[i, 0]) == fx["probes"][i]["top5_rows"][0] for i in range(len(texts)))
        same5 = sum(sorted(top5[i].tolist()) == sorted(fx["probes"][i]["top5_rows"]) for i in range(len(texts)))
        print(f"parity: max |embedding gap| {emb_gap:.2e}  min projection cosine {proj_cos:.6f}  top-1 identical {same1}/{len(texts)}  top-5 identical {same5}/{len(texts)}")

    if a.skip_replay:
        return
    rp = json.load(open(hf_hub_download(HEADS, "replay_123k.json")))
    caps = [c["text"] for c in rp["captions"]]; gold = torch.tensor([c["gold"] for c in rp["captions"]])
    t0 = time.time()
    zt = project(head, encode(caps))
    S = zt @ Zg.T
    gold_s = S[torch.arange(len(gold)), gold]
    rank = (S > gold_s[:, None]).sum(1) + 1
    print(f"replay: {len(caps):,} captions vs {len(keys):,} rows in {time.time() - t0:.0f}s  "
          f"t2i R@1 {(rank <= 1).float().mean():.4f}  R@5 {(rank <= 5).float().mean():.4f}  R@10 {(rank <= 10).float().mean():.4f}  median rank {int(rank.float().median())}")
    ref = rp.get("pytorch_t2i", {})
    if ref:
        print(f"the Qwen3-0.6B tower on the same set: R@1 {ref['r@1']}  R@5 {ref['r@5']}  R@10 {ref['r@10']}  median rank 36")


if __name__ == "__main__":
    main()
