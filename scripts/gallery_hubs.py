"""Find the hubs of a head-space gallery: the images nearest its centre.

High-dimensional retrieval has hubs, rows that are close to almost everything.
A query carrying no signal lands near the centroid and returns them, which is
what a visitor sees when they ask a model with no visual content in the
question. Those images are not a bug and not an answer, and it is worth being
able to name them.

    python scripts/gallery_hubs.py --index artifacts/local/browser/gallery_20k_v3.srtidx \
        --out artifacts/nla/q4/gallery_hubs_20k.json

Add --thumbs to also pull the hub thumbnails out of the packed mirror, which
is the only way to find out what they actually are.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

THUMBS = "https://huggingface.co/datasets/RiverRider/srt-coco-thumbs/resolve/main"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--thumbs", type=Path, default=None,
                   help="directory to write hub thumbnails into, fetched by byte range")
    return p.parse_args()


def load(path: Path):
    raw = path.read_bytes()
    magic = raw[:8]
    dim, n = struct.unpack_from("<II", raw, 8)
    off = 16
    if magic == b"SRTIDX02":
        scales = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
        off += 4 * n
        q = np.frombuffer(raw, dtype=np.int8, count=n * dim, offset=off).reshape(n, dim)
        off += n * dim
        Z = q.astype(np.float32) * scales[:, None]
    elif magic == b"SRTIDX01":
        Z = np.frombuffer(raw, dtype="<f2", count=n * dim, offset=off).reshape(n, dim)
        Z = Z.astype(np.float32)
        off += n * dim * 2
    else:
        raise SystemExit(f"{path}: not an srtidx ({magic!r})")
    keys = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4
        keys.append(raw[off:off + ln].decode())
        off += ln
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return Z, keys, magic.decode()


def fetch_thumb(row: int, out: Path) -> int:
    hdr = subprocess.run(
        ["curl", "-sSL", "-r", f"{8 * row}-{8 * row + 7}", f"{THUMBS}/thumbs_offsets.bin"],
        capture_output=True, check=True).stdout
    pos, ln = struct.unpack("<II", hdr)
    if not ln:
        return 0
    img = subprocess.run(
        ["curl", "-sSL", "-r", f"{pos}-{pos + ln - 1}", f"{THUMBS}/thumbs.bin"],
        capture_output=True, check=True).stdout
    (out / f"hub_{row}.jpg").write_bytes(img)
    return len(img)


def main() -> None:
    a = parse_args()
    Z, keys, magic = load(a.index)
    mu = Z.mean(0)
    mu_norm = float(np.linalg.norm(mu))
    cos = Z @ (mu / (mu_norm + 1e-12))
    order = np.argsort(-cos)[: a.k]

    report = {
        "index": str(a.index),
        "format": magic,
        "rows": int(Z.shape[0]),
        "dim": int(Z.shape[1]),
        # 0 would be perfectly isotropic. The head keeps this far below the
        # backbone's, but not at zero, and what is left is what hubs live in.
        "mu_norm": round(mu_norm, 4),
        "cos_to_centre": {
            "mean": round(float(cos.mean()), 4),
            "max": round(float(cos.max()), 4),
            "min": round(float(cos.min()), 4),
        },
        "hubs": [{"row": int(i), "key": keys[i], "cos": round(float(cos[i]), 4)} for i in order],
    }

    if a.thumbs:
        a.thumbs.mkdir(parents=True, exist_ok=True)
        for h in report["hubs"]:
            h["thumb_bytes"] = fetch_thumb(h["row"], a.thumbs)
        print(f"thumbnails written to {a.thumbs}")

    print(f"{magic}  {report['rows']} x {report['dim']}   ||mu|| = {mu_norm:.4f}")
    print(f"cos to centre: mean {cos.mean():.3f}  max {cos.max():.3f}  min {cos.min():.3f}")
    for h in report["hubs"]:
        print(f"  {h['cos']:.3f}  row {h['row']:>6}  {h['key']}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
