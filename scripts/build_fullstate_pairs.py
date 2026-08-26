#!/usr/bin/env python
"""Assemble large-model image records and their COCO captions.

Three sources, none of which costs a re-encode:

    gemma31b  procrustes/train_pairs/chunk_*.pt, gemma-4-31B layer-47, 5376-d
    qwen38    raw118k/shard*_chunk*.npz, Qwen3.8-27B, 5120-d
    gallery   the shipped .srtidx itself, 1024-d

The first two are raw hidden states. The third is the head's projection of
them, which is what the browser already holds for all 123,287 images: reading
it costs no extra download, so every image in the gallery becomes describable
rather than a curated handful. It is a lossy summary of the raw state, so it
is a narrower claim, and the vectors are dequantized from int8 and L2
normalized here exactly as the runtime does it, because a verbalizer trained
on a different representation than it is served would silently underperform.

The chunks carry the caption's STATE but not its words, so caption text comes
from the COCO annotations, joined on file basename.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("gemma31b", "qwen38", "gallery"), default="gemma31b")
    p.add_argument("--repo", default=None)
    p.add_argument("--chunks", type=int, default=24)
    p.add_argument("--index", default="/root/gallery_123k_v3.srtidx")
    p.add_argument("--key-prefix", default="train2017/")
    p.add_argument("--cache", default="/root/.hf_home")
    p.add_argument("--coco", default="/root/annotations/captions_train2017.json")
    p.add_argument("--out-vecs", default="/root/full_img_vecs.npy")
    p.add_argument("--out-caps", default="/root/full_caps.json")
    return p.parse_args()


def iter_chunks(a):
    """Yield (label, states, filenames) per published chunk."""
    from huggingface_hub import hf_hub_download

    if a.source == "gallery":
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent))
        from eval_shared_space_verbalizer import load_index

        gal, pos = load_index(a.index)      # already dequantized and normalized
        rows = [(k, i) for k, i in pos.items() if k.startswith(a.key_prefix)]
        rows.sort(key=lambda kv: kv[1])
        idx = np.array([i for _, i in rows])
        yield a.index, gal[idx], [k.split("/", 1)[1] for k, _ in rows]
        return

    if a.source == "gemma31b":
        repo = a.repo or "RiverRider/srt-nla-gemma4-artifacts"
        for i in range(a.chunks):
            path = hf_hub_download(repo, f"procrustes/train_pairs/chunk_{i:03d}.pt",
                                   cache_dir=a.cache)
            d = torch.load(path, map_location="cpu", weights_only=True)
            yield f"chunk_{i:03d}", d["img"].float().numpy(), list(d["files"])
    else:
        repo = a.repo or "RiverRider/srt-qwen38-coco-states"
        for shard in (0, 1):
            for c in range(15):
                name = f"raw118k/shard{shard}_chunk{c:04d}.npz"
                path = hf_hub_download(repo, name, repo_type="dataset",
                                       cache_dir=a.cache)
                z = np.load(path, allow_pickle=True)
                # Bind once: indexing an NpzFile re-reads the whole array.
                raw, files = z["raw"], z["files"]
                yield name, raw.astype(np.float32), [str(f) for f in files]


def main():
    a = parse()

    print("reading COCO captions", flush=True)
    coco = json.load(open(a.coco))
    by_id = defaultdict(list)
    for ann in coco["annotations"]:
        by_id[ann["image_id"]].append(ann["caption"].strip())
    # COCO file names are the zero-padded image id, so the basename is the key.
    print(f"  {len(by_id)} images have captions", flush=True)

    vecs, caps, images, skipped, seen = [], [], [], 0, set()
    for name, states, files in iter_chunks(a):
        keep = []
        for row, f in enumerate(files):
            stem = Path(f).stem
            c = by_id.get(int(stem))
            # The qwen38 shards overlap at the edges, so drop repeats.
            if not c or stem in seen:
                skipped += 1
                continue
            seen.add(stem)
            keep.append(row)
            caps.append(c)
            images.append(f"{stem}.jpg")
        vecs.append(states[keep])
        print(f"  {name}: kept {len(keep)}/{len(files)}", flush=True)

    V = np.concatenate(vecs, 0)
    np.save(a.out_vecs, V)
    json.dump({"captions": caps, "images": images}, open(a.out_caps, "w"))
    print(f"{V.shape[0]} images x {V.shape[1]} dims, "
          f"{sum(len(c) for c in caps)} captions, {skipped} skipped", flush=True)
    print(f"wrote {a.out_vecs} and {a.out_caps}", flush=True)


if __name__ == "__main__":
    main()
