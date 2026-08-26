#!/usr/bin/env python
"""Assemble the raw large-model states and their COCO captions.

The published procrustes chunks already hold what the re-encode would have
cost: gemma-4-31B layer-47 image states at full width, 5376-d, for every
train2017 image. They carry the caption's STATE but not its words, so the
caption text comes from the COCO annotations and is joined on file basename.

Output is the pair of files the verbalizer trainer already reads, so the only
thing that changes downstream is the width of the vector: 5376 raw dimensions
instead of the 1024 that survive the browser head.
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
    p.add_argument("--repo", default="RiverRider/srt-nla-gemma4-artifacts")
    p.add_argument("--chunks", type=int, default=24)
    p.add_argument("--cache", default="/root/.hf_home")
    p.add_argument("--coco", default="/root/annotations/captions_train2017.json")
    p.add_argument("--out-vecs", default="/root/full_img_vecs.npy")
    p.add_argument("--out-caps", default="/root/full_caps.json")
    return p.parse_args()


def main():
    from huggingface_hub import hf_hub_download

    a = parse()

    print("reading COCO captions", flush=True)
    coco = json.load(open(a.coco))
    by_id = defaultdict(list)
    for ann in coco["annotations"]:
        by_id[ann["image_id"]].append(ann["caption"].strip())
    # COCO file names are the zero-padded image id, so the basename is the key.
    print(f"  {len(by_id)} images have captions", flush=True)

    vecs, caps, images, missing = [], [], [], 0
    for i in range(a.chunks):
        path = hf_hub_download(a.repo, f"procrustes/train_pairs/chunk_{i:03d}.pt",
                               cache_dir=a.cache)
        d = torch.load(path, map_location="cpu", weights_only=True)
        img, files = d["img"], d["files"]
        keep = []
        for row, f in enumerate(files):
            stem = Path(f).stem
            c = by_id.get(int(stem))
            if not c:
                missing += 1
                continue
            keep.append(row)
            caps.append(c)
            images.append(f"{stem}.jpg")
        vecs.append(img[keep].float().numpy())
        print(f"  chunk {i:03d}: kept {len(keep)}/{len(files)}", flush=True)

    V = np.concatenate(vecs, 0)
    np.save(a.out_vecs, V)
    json.dump({"captions": caps, "images": images}, open(a.out_caps, "w"))
    print(f"{V.shape[0]} images x {V.shape[1]} dims, "
          f"{sum(len(c) for c in caps)} captions, {missing} unmatched", flush=True)
    print(f"wrote {a.out_vecs} and {a.out_caps}", flush=True)


if __name__ == "__main__":
    main()
