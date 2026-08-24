"""What does compressing the gallery cost in retrieval?

At 123K images an f32 index is 504MB resident, which no phone will hold. f16
halves it and int8 halves it again, but only a measurement says whether that
is free. Rows are L2-normalized, so int8 uses a symmetric per-row scale.

Run against the PyTorch reference so the answer is about the storage format
and nothing else.

    python scripts/gallery_precision_cost.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BROWSER = ROOT / "artifacts/local/browser"
N_EVAL = 1000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path,
                   default=ROOT / "checkpoints/gemma4_readout/browser_text_head_qwen3_0.6B.pt")
    p.add_argument("--gallery", type=Path, default=BROWSER / "gallery.npz")
    p.add_argument("--replay", type=Path, default=BROWSER / "replay.json")
    p.add_argument("--states", type=Path, default=BROWSER / "ref_text_states.npy",
                   help="cached PyTorch text states; built by browser_rung_reference")
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts/nla/q4/gallery_precision_cost.json")
    return p.parse_args()


def unit(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def as_f16(Z):
    return Z.astype(np.float16).astype(np.float32)


def as_int8(Z):
    """Symmetric per-row int8, the format a browser would actually hold."""
    scale = np.abs(Z).max(axis=1, keepdims=True) / 127.0
    q = np.clip(np.rint(Z / np.maximum(scale, 1e-12)), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def recall(S, gold):
    order = np.argsort(-S, axis=1)
    rank = np.array([int(np.where(order[i] == gold[i])[0][0]) for i in range(len(gold))])
    return {f"r@{k}": round(float((rank < k).mean()), 4) for k in (1, 5, 10)}


def main() -> None:
    a = parse_args()
    replay = json.loads(a.replay.read_text())
    caps = replay["captions"]
    gold = np.array([c["gold"] for c in caps])

    if not a.states.exists():
        raise SystemExit(
            f"missing {a.states}. Re-run browser_rung_reference.py, which now caches "
            "the encoded text states so this comparison costs no GPU time.")

    head = torch.load(a.head, map_location="cpu", weights_only=True)
    Wt = head["txt"]["weight"].float().numpy()
    bt = head["txt"]["bias"].float().numpy()
    mt = head["mu_txt"].float().numpy()
    Z_txt = unit((np.load(a.states) - mt) @ Wt.T + bt)

    g = np.load(a.gallery, allow_pickle=True)
    Z_img = unit(g["Z_img"][-N_EVAL:].astype(np.float32))

    variants = {
        "f32": Z_img,
        "f16": as_f16(Z_img),
        "int8_per_row": as_int8(Z_img),
    }
    bytes_per_vec = {"f32": 4 * 1024, "f16": 2 * 1024, "int8_per_row": 1024 + 4}

    out = {"n_images": int(Z_img.shape[0]), "n_captions": len(caps), "variants": {}}
    base = None
    for name, Z in variants.items():
        r = recall(Z_txt @ Z.T, gold)
        if base is None:
            base = r["r@1"]
        mb_123k = bytes_per_vec[name] * 123_287 / 1e6
        out["variants"][name] = {
            "t2i": r,
            "delta_r@1_vs_f32": round(r["r@1"] - base, 4),
            "bytes_per_vector": bytes_per_vec[name],
            "gallery_123k_mb": round(mb_123k, 1),
        }
        print(f"{name:14s} R@1 {r['r@1']:.4f}  R@5 {r['r@5']:.4f}  R@10 {r['r@10']:.4f}  "
              f"delta {r['r@1'] - base:+.4f}  123K gallery {mb_123k:.0f} MB")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
