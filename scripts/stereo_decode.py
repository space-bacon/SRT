"""Simulate binocular fusion to decode an autostereogram.

A single-image autostereogram hides its figure purely in the horizontal
DISPARITY between repeated pattern columns. A human recovers it by verging the
eyes so each eye lands on a neighbouring copy of the pattern (an alignment shift
of ~one repeat period), then the visual cortex fuses them by solving the
stereo-correspondence problem. Regions at a different depth were drawn with a
different repeat period, so their correspondence shift differs and they pop out
in relief.

This script performs exactly that correspondence search: for each pixel it slides
the image against a horizontally shifted copy of itself and picks the shift that
minimises a local colour mismatch. The resulting shift map is the depth, so the
hidden figure is recovered from the flat raster that a 2D encoder reads as noise.

Outputs (to <out>/):
  disparity.png  - recovered depth/shift map (grey; brighter = nearer)
  figure.png     - denoised, thresholded silhouette of the recovered figure

Usage:
    python scripts/stereo_decode.py --image artifacts/nla/gemma4/stereo/stereogram.png \
        --out artifacts/nla/gemma4/stereo --smin 45 --smax 80
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="artifacts/nla/gemma4/stereo/stereogram.png")
    p.add_argument("--out", default="artifacts/nla/gemma4/stereo")
    p.add_argument("--smin", type=int, default=45, help="min vergence shift (px)")
    p.add_argument("--smax", type=int, default=80, help="max vergence shift (px)")
    p.add_argument("--winx", type=int, default=17, help="horizontal match window")
    p.add_argument("--winy", type=int, default=5, help="vertical match window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    H, W, _ = img.shape

    # For every candidate vergence shift s, the local colour mismatch between the
    # image and its s-shifted copy. Low mismatch = the two eyes fuse at that shift.
    best = np.full((H, W), np.inf, dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    for s in range(args.smin, args.smax + 1):
        shifted = np.empty_like(img)
        shifted[:, s:] = img[:, : W - s]
        shifted[:, :s] = img[:, :s]  # left border: no correspondence, neutralise
        mism = np.abs(img - shifted).sum(axis=2)
        mism[:, :s] = np.inf  # invalid region has no left neighbour
        # average the mismatch over a local window (patch correspondence)
        finite = np.isfinite(mism)
        m = np.where(finite, mism, 0.0)
        sm = uniform_filter(m, size=(args.winy, args.winx), mode="nearest")
        cnt = uniform_filter(finite.astype(np.float32), size=(args.winy, args.winx),
                             mode="nearest")
        sm = np.where(cnt > 0, sm / np.maximum(cnt, 1e-6), np.inf)
        upd = sm < best
        best[upd] = sm[upd]
        depth[upd] = s

    # depth = recovered separation. Background has one period, the raised figure a
    # smaller one. Convert to a "nearer = brighter" map, denoise, normalise.
    depth = median_filter(depth, size=7)
    valid = depth[:, args.smax:]  # drop the un-decodable left strip
    lo, hi = np.percentile(valid, 5), np.percentile(valid, 95)
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    near = 1.0 - norm  # smaller separation (figure) -> brighter
    near[:, : args.smax] = 0.0
    Image.fromarray((near * 255).astype(np.uint8)).save(os.path.join(args.out, "disparity.png"))

    # threshold into a clean silhouette (figure vs ground)
    thr = near > 0.5
    thr = median_filter(thr.astype(np.uint8), size=9) > 0
    fig = np.zeros((H, W), dtype=np.uint8)
    fig[thr] = 255
    Image.fromarray(fig).save(os.path.join(args.out, "figure.png"))
    frac = float(thr.mean())
    print(f"decoded {args.image}: shift range [{args.smin},{args.smax}], "
          f"figure covers {frac:.1%} of frame -> disparity.png + figure.png", flush=True)


if __name__ == "__main__":
    main()
