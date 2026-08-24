"""What does a 123x bigger gallery cost in retrieval?

Every number reported for the browser rung so far used a 1,000-image gallery.
The deployed one has 123,287, which is 122,287 more chances to outrank the
right answer. Recall must fall, and by how much is the difference between a
demo that feels sharp and one that feels vague.

Gold is unchanged: each caption still owns exactly one image. Only the
distractor pool grows.

    python scripts/gallery_scale_cost.py
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BROWSER = ROOT / "artifacts/local/browser"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path,
                   default=ROOT / "checkpoints/gemma4_readout/browser_text_head_qwen3_0.6B.pt")
    p.add_argument("--index", type=Path, default=BROWSER / "gallery_123k.srtidx")
    p.add_argument("--replay", type=Path, default=BROWSER / "replay.json")
    p.add_argument("--states", type=Path, default=BROWSER / "ref_text_states.npy")
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts/nla/q4/gallery_scale_cost.json")
    return p.parse_args()


def read_srtidx(path: Path):
    b = path.read_bytes()
    magic = b[:8]
    dim, count = struct.unpack("<II", b[8:16])
    off = 16
    if magic == b"SRTIDX02":
        scale = np.frombuffer(b[off:off + count * 4], dtype="<f4").reshape(-1, 1)
        off += count * 4
        q = np.frombuffer(b[off:off + dim * count], dtype=np.int8).reshape(count, dim)
        off += dim * count
        Z = q.astype(np.float32) * scale
    elif magic == b"SRTIDX01":
        Z = np.frombuffer(b[off:off + dim * count * 2], dtype="<f2").reshape(count, dim)
        Z = Z.astype(np.float32)
        off += dim * count * 2
    else:
        raise SystemExit(f"bad magic {magic!r}")
    keys = []
    for _ in range(count):
        (n,) = struct.unpack("<I", b[off:off + 4])
        off += 4
        keys.append(b[off:off + n].decode())
        off += n
    return Z, keys


def unit(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    a = parse_args()
    replay = json.loads(a.replay.read_text())
    eval_files = replay["files"]
    caps = replay["captions"]

    head = torch.load(a.head, map_location="cpu", weights_only=True)
    Wt = head["txt"]["weight"].float().numpy()
    bt = head["txt"]["bias"].float().numpy()
    mt = head["mu_txt"].float().numpy()
    Z_txt = unit((np.load(a.states) - mt) @ Wt.T + bt)

    Z, keys = read_srtidx(a.index)
    print(f"index {Z.shape[0]} x {Z.shape[1]}")

    # Map each eval caption's gold image to its row in the big index. Keys
    # carry their split, the replay does not, so match on basename.
    row_of = {k.split("/")[-1]: i for i, k in enumerate(keys)}
    gold_full, keep = [], []
    for i, c in enumerate(caps):
        f = eval_files[c["gold"]]
        if f in row_of:
            gold_full.append(row_of[f])
            keep.append(i)
    gold_full = np.array(gold_full)
    Z_txt = Z_txt[keep]
    print(f"{len(keep)} of {len(caps)} captions have their gold image in the index")

    # Small gallery: the 1,000 eval images only, as everything so far reported.
    small_rows = np.array([row_of[f] for f in eval_files if f in row_of])
    small_pos = {r: i for i, r in enumerate(small_rows)}
    gold_small = np.array([small_pos[g] for g in gold_full])

    out = {"n_captions": len(keep), "variants": {}}
    for name, sub, gold in (
        (f"eval_only_{len(small_rows)}", Z[small_rows], gold_small),
        (f"full_{Z.shape[0]}", Z, gold_full),
    ):
        S = Z_txt @ sub.T
        # rank of gold = how many distractors score at least as high
        gold_score = S[np.arange(len(gold)), gold][:, None]
        rank = (S > gold_score).sum(axis=1)
        r = {f"r@{k}": round(float((rank < k).mean()), 4) for k in (1, 5, 10, 50)}
        out["variants"][name] = {"gallery": int(sub.shape[0]), "t2i": r,
                                 "median_rank": float(np.median(rank) + 1)}
        print(f"{name:22s} gallery {sub.shape[0]:>7} "
              + "  ".join(f"R@{k} {r[f'r@{k}']:.4f}" for k in (1, 5, 10, 50))
              + f"  median rank {np.median(rank) + 1:.0f}")

    s = out["variants"][f"eval_only_{len(small_rows)}"]["t2i"]["r@1"]
    f = out["variants"][f"full_{Z.shape[0]}"]["t2i"]["r@1"]
    out["r@1_cost_of_scale"] = round(f - s, 4)
    out["note"] = ("same captions and same gold, only the distractor pool grows; "
                   "the full gallery is what the demo actually searches")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nR@1 cost of going to full scale: {f - s:+.4f}\nwrote {a.out}")


if __name__ == "__main__":
    main()
