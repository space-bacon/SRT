"""Write a head-space gallery to the SRTIDX01 format the Rust index reads.

Two input shapes, because two things exist in this repo:

  --projected  an npz that already holds head-space vectors (Z_img/files),
               e.g. artifacts/local/gallery_full.npz
  --states     an npz of raw backbone states plus --head, projected here

The output is a flat file a browser can fetch and use with no parser: a
16-byte header, fp16 rows, then length-prefixed keys. Search over it is a dot
product, which is the whole reason the static tier needs no model.

    python scripts/export_index_srtidx.py \
        --projected artifacts/local/gallery_full.npz \
        --split val2017 --out artifacts/local/gallery_val2017.srtidx
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC_F16 = b"SRTIDX01"
MAGIC_I8 = b"SRTIDX02"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--projected", type=Path, nargs="+",
                     help="one or more npz of head-space vectors (Z_img or Z) and files; "
                          "several are concatenated, which is how sharded encodes merge")
    src.add_argument("--states", type=Path, help="npz with raw states + files/texts")
    p.add_argument("--head", type=Path, help="head .pt, required with --states")
    p.add_argument("--modality", choices=["img", "txt"], default="img")
    p.add_argument("--split", default=None, help="keep only rows with this split value")
    p.add_argument("--start", type=int, default=0, help="skip this many rows")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--dtype", choices=["f16", "int8"], default="f16",
                   help="int8 costs nothing measurable and quarters resident memory; "
                        "use it for anything a browser has to hold")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_projected(paths: list[Path], split: str | None):
    Zs, keys = [], []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        key = "Z_img" if "Z_img" in z.files else "Z"
        Z = z[key].astype(np.float32)
        k = np.array([str(f) for f in z["files"]])
        if split is not None:
            if "split" not in z.files:
                raise SystemExit(f"{path} has no split array")
            m = z["split"] == split
            Z, k = Z[m], k[m]
        Zs.append(Z)
        keys.append(k)
    prov = f"{len(paths)} file(s), first {paths[0].name}"
    return np.concatenate(Zs), np.concatenate(keys), prov


def load_states(path: Path, head_path: Path, modality: str):
    import torch

    z = np.load(path, allow_pickle=True)
    S = z["states"].astype(np.float32)
    name_field = "files" if "files" in z.files else "texts"
    keys = np.array([str(f) for f in z[name_field]])
    d = torch.load(head_path, map_location="cpu", weights_only=True)
    W = d[modality]["weight"].float().numpy()
    b = d[modality]["bias"].float().numpy()
    mu = d[f"mu_{modality}"].float().numpy()
    Z = (S - mu) @ W.T + b
    return Z, keys, f"{path.name} via {head_path.name}[{modality}]"


def write_index(out: Path, Z: np.ndarray, keys: np.ndarray, dtype: str = "f16") -> None:
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    n, dim = Z.shape
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        if dtype == "f16":
            f.write(MAGIC_F16)
            f.write(struct.pack("<II", dim, n))
            f.write(Z.astype(np.float16).tobytes(order="C"))
        else:
            # Symmetric per-row scale; rows are unit-norm so one scale suffices.
            scale = np.abs(Z).max(axis=1, keepdims=True) / 127.0
            scale = np.maximum(scale, 1e-12)
            q = np.clip(np.rint(Z / scale), -127, 127).astype(np.int8)
            f.write(MAGIC_I8)
            f.write(struct.pack("<II", dim, n))
            f.write(scale.astype(np.float32).ravel().tobytes(order="C"))
            f.write(q.tobytes(order="C"))
        for k in keys:
            enc = str(k).encode("utf-8")
            f.write(struct.pack("<I", len(enc)))
            f.write(enc)


def main() -> None:
    a = parse_args()
    if a.states:
        if not a.head:
            raise SystemExit("--states requires --head")
        Z, keys, prov = load_states(a.states, a.head, a.modality)
    else:
        Z, keys, prov = load_projected(a.projected, a.split)

    if a.start:
        Z, keys = Z[a.start :], keys[a.start :]
    if a.limit:
        Z, keys = Z[: a.limit], keys[: a.limit]

    write_index(a.out, Z, keys, a.dtype)
    mb = a.out.stat().st_size / 1e6
    print(f"wrote {a.out}: {Z.shape[0]} x {Z.shape[1]} {a.dtype}, {mb:.1f} MB")
    print(f"  source: {prov}")

    # Anisotropy of the index itself. Head-space vectors should be close to
    # isotropic; a high value here means the projection is not doing its job
    # and every retrieval score will read as suspiciously high.
    k = min(500, Z.shape[0])
    U = Z[:k] / (np.linalg.norm(Z[:k], axis=1, keepdims=True) + 1e-8)
    C = U @ U.T
    print(f"  mean pairwise cos: {C[np.triu_indices(k, 1)].mean():+.4f}")


if __name__ == "__main__":
    main()
