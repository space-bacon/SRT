"""How many things can the shared space tell apart, and does that ceiling bind?

A reader asked whether meaning here is built from a bounded set of units. Variance
rank answers a different question: it measures where the spread sits, not how many
items the space separates. A space can carry most of its variance in a few dozen
directions and still discriminate hundreds of thousands of points.

So this measures capacity directly. Fixed text queries, retrieval against pools
that grow from 1,000 to 123,287 real images, all through the shipped head. If the
alphabet were exhausted, rank would grow in proportion to the pool and the
fraction would stay flat. If discrimination has headroom, the fraction falls.

    .venv/bin/python scripts/space_capacity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
HEAD = ROOT / "release/srt-sunstone-linear-head/sunstone_linear_head_v3_drift.pt"
GALLERY = ROOT / "artifacts/local/gallery_full.npz"
OUT = ROOT / "artifacts/nla/q4/space_capacity.json"
POOLS = [1_000, 4_000, 16_000, 64_000, 123_287]
SEED = 0


def project(v: np.ndarray, W: np.ndarray, b: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z = (v - mu) @ W.T + b
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    head = torch.load(HEAD, map_location="cpu", weights_only=True)
    calib = torch.load(
        hf_hub_download("RiverRider/srt-nla-gemma4-artifacts", "procrustes/encoded_L47_n5000.pt"),
        map_location="cpu", weights_only=True)

    z = np.load(GALLERY, allow_pickle=False)
    Z_img = z["Z_img"].astype(np.float32)
    pos = {f: i for i, f in enumerate(z["files"])}
    gold = np.array([pos[Path(f).name] for f in calib["files"]])

    W_i = head["img"]["weight"].float().numpy()
    b_i = head["img"]["bias"].float().numpy()
    mu_i = head["mu_img"].float().numpy()

    # Control first: re-projecting the calibration images must land exactly on the
    # shipped gallery rows. This is what localises any failure to the text side.
    mine = project(calib["img"].float().numpy(), W_i, b_i, mu_i)
    ctrl = float((mine * Z_img[gold]).sum(1).mean())
    print(f"control: reprojected images vs shipped gallery rows, cos {ctrl:.4f}")
    assert ctrl > 0.999, "image path disagrees with the shipped gallery, stop here"

    # cap0, not cap5: cap5 is not row-aligned to files and retrieves at chance,
    # while cap0 gives median rank 4 in a 5k pool.
    Z_txt = project(calib["cap0"].float().numpy(),
                    head["txt"]["weight"].float().numpy(),
                    head["txt"]["bias"].float().numpy(),
                    head["mu_txt"].float().numpy())

    rng = np.random.default_rng(SEED)
    n_gal = Z_img.shape[0]
    results = {}
    for N in POOLS:
        # Every pool holds the correct image plus N-1 real distractors, so the
        # only thing changing across rows is how much company the answer keeps.
        ranks = np.empty(len(gold), dtype=np.int64)
        for qi in range(len(gold)):
            if N >= n_gal:
                sub = np.arange(n_gal)
                tgt = gold[qi]
            else:
                sub = rng.choice(n_gal, size=N - 1, replace=False)
                sub = sub[sub != gold[qi]]
                sub = np.concatenate([[gold[qi]], sub])
                tgt = gold[qi]
            s = Z_img[sub] @ Z_txt[qi]
            ranks[qi] = int((s > s[np.where(sub == tgt)[0][0]]).sum()) + 1
        med = float(np.median(ranks))
        results[str(N)] = {
            "median_rank": med,
            "median_rank_fraction_of_pool": med / N,
            "r@1": float((ranks == 1).mean()),
            "r@10": float((ranks <= 10).mean()),
        }
        print(f"pool {N:>7,}  median {med:>7.1f}  = {med / N:.5f} of pool  "
              f"r@1 {results[str(N)]['r@1']:.3f}", flush=True)

    Xc = Z_img - Z_img.mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam /= lam.sum()
    out = {
        "question": "does discrimination in the shared space saturate as the pool grows",
        "queries": int(len(gold)),
        "query_note": "5,000 held-out COCO first captions, datacenter L47 states, shipped head",
        "image_control_cos": ctrl,
        "pools": results,
        "participation_ratio": float(1.0 / (lam ** 2).sum()),
        "dims_for_99pct_variance": int(np.searchsorted(np.cumsum(lam), 0.99)) + 1,
        "dims_total": int(Z_img.shape[1]),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
