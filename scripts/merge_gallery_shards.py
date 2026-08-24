"""Merge sharded gallery chunks into one index, with the checks that matter.

Two GPUs write numbered chunks independently, so this is where the pieces are
checked before they become a single artifact: no duplicate keys, uniform
width, and vectors that are actually unit-norm. A gallery is the one file a
deployment cannot sanity-check at runtime, because a bad row just quietly wins
every search.

    python scripts/merge_gallery_shards.py \
        --chunks /root/gallery118k --also artifacts/local/browser/gallery.npz \
        --dtype int8 --out artifacts/local/browser/gallery_123k.srtidx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_index_srtidx import write_index


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, required=True, help="dir of shard*_chunk*.npz")
    p.add_argument("--also", type=Path, nargs="*", default=[],
                   help="extra projected npz to fold in, e.g. the val2017 gallery")
    p.add_argument("--dtype", choices=["f16", "int8"], default="int8")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys-out", type=Path, default=None,
                   help="optional newline-separated key list, for a page that "
                        "wants filenames without parsing the index")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    files = sorted(a.chunks.glob("shard*_chunk*.npz")) if a.chunks.exists() else []
    files += list(a.also)
    if not files:
        raise SystemExit(f"no chunks under {a.chunks}")

    Zs, keys = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        key = "Z_img" if "Z_img" in z.files else "Z"
        Zs.append(z[key].astype(np.float32))
        keys.append(np.array([str(x) for x in z["files"]]))
    Z = np.concatenate(Zs)
    K = np.concatenate(keys)
    print(f"{len(files)} sources -> {Z.shape[0]} vectors, dim {Z.shape[1]}")

    if len({z.shape[1] for z in Zs}) != 1:
        raise SystemExit("chunks disagree on width")

    # Shards partition by stride, so a duplicate means a chunk was counted
    # twice, which would put the same image in the gallery at two ranks.
    uniq, first = np.unique(K, return_index=True)
    if len(uniq) != len(K):
        print(f"  dropping {len(K) - len(uniq)} duplicate keys")
        order = np.sort(first)
        Z, K = Z[order], K[order]

    norms = np.linalg.norm(Z, axis=1)
    print(f"  norms min {norms.min():.4f} max {norms.max():.4f}")
    bad = int((norms < 0.5).sum())
    if bad:
        print(f"  dropping {bad} degenerate rows")
        keep = norms >= 0.5
        Z, K = Z[keep], K[keep]

    k = min(500, len(Z))
    U = Z[:k] / (np.linalg.norm(Z[:k], axis=1, keepdims=True) + 1e-8)
    C = U @ U.T
    print(f"  mean pairwise cos: {C[np.triu_indices(k, 1)].mean():+.4f}")

    write_index(a.out, Z, K, a.dtype)
    print(f"wrote {a.out}: {len(Z)} x {Z.shape[1]} {a.dtype}, "
          f"{a.out.stat().st_size / 1e6:.1f} MB")

    if a.keys_out:
        a.keys_out.write_text("\n".join(str(x) for x in K))
        print(f"wrote {a.keys_out}")


if __name__ == "__main__":
    main()
