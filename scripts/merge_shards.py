#!/usr/bin/env python3
"""Reassemble the per-GPU shards of one vendor into a single states file.

The manifest is split into contiguous shards, so concatenating them in shard
order reproduces the original row order exactly and the downstream probe can go
on using the unsplit manifest.

    python scripts/merge_shards.py --tag gemma4 --shards 4
"""
import argparse

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--prefix", default="/root/cxr14")
    return p.parse_args()


def main():
    a = parse_args()
    parts = [np.load(f"{a.prefix}_{a.tag}_s{i}.npz", allow_pickle=True)
             for i in range(a.shards)]
    n0 = len(parts[0]["ok"])
    out = {}
    for k in parts[0].files:
        # Per-row arrays are concatenated; everything else describes the run and
        # is identical across shards. Keeps this working for both the plain
        # item/text encoder and the multi-layer pooling one.
        if getattr(parts[0][k], "shape", ()) and parts[0][k].shape[0] == n0:
            out[k] = np.concatenate([z[k] for z in parts])
        else:
            out[k] = parts[0][k]
    path = f"{a.prefix}_{a.tag}.npz"
    np.savez(path, **out)
    ok = out["ok"]
    print(f"{a.tag}: {len(ok)} rows, {int(ok.sum())} ok, "
          f"item {out['item'].shape} -> {path}")


if __name__ == "__main__":
    main()
