"""Generate a random-dot autostereogram (Magic-Eye) that hides a known shape,
plus a plain visible control of the same shape.

The hidden figure of an autostereogram lives ONLY in the horizontal disparity
between repeated dot columns. There is, by construction, no 2D luminance/colour
cue to the shape in the flat image: recovering it requires binocular stereopsis
(matching a column to its shifted copy). A vision encoder that reads the flat
raster sees repeating coloured noise, not the figure. That is exactly the point
of the test.

Outputs:
  <out>/control.png    - the shape as a plain white silhouette on black
  <out>/stereogram.png - a colour random-dot autostereogram hiding the SAME shape

Uses the Thimbleby/Inglis same-pixel-linking algorithm.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def heart_depth(h: int, w: int) -> np.ndarray:
    """Return a depth map in [0,1]: 1 = nearest (raised), 0 = background."""
    yy, xx = np.mgrid[0:h, 0:w]
    # normalise to the classic heart curve domain
    x = (xx - w / 2) / (w * 0.32)
    y = (h * 0.52 - yy) / (h * 0.32)
    inside = (x * x + y * y - 1.0) ** 3 - x * x * (y ** 3) <= 0.0
    return inside.astype(np.float32)


def separation(z: np.ndarray, eye: int, mu: float) -> np.ndarray:
    return np.round((1.0 - mu * z) * eye / (2.0 - mu * z)).astype(np.int32)


def make_stereogram(depth: np.ndarray, eye: int = 140, mu: float = 0.32,
                    seed: int = 0) -> np.ndarray:
    h, w = depth.shape
    rng = np.random.default_rng(seed)
    # vivid magic-eye palette
    palette = np.array([
        [235, 64, 52], [240, 147, 43], [247, 220, 60], [76, 209, 55],
        [18, 203, 196], [52, 152, 219], [155, 89, 182], [253, 121, 168],
        [232, 67, 147], [0, 184, 148], [253, 203, 110], [108, 92, 231],
    ], dtype=np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sep = separation(depth, eye, mu)
    for y in range(h):
        same = np.arange(w)
        for x in range(w):
            s = int(sep[y, x])
            left = x - s // 2
            right = left + s
            if 0 <= left and right < w:
                # link right pixel to left pixel (constraint chaining)
                k = right
                while same[k] != k:
                    k = same[k]
                same[right] = k if k != right else left
                same[right] = left
        for x in range(w):
            if same[x] == x:
                img[y, x] = palette[rng.integers(len(palette))]
            else:
                img[y, x] = img[y, same[x]]
    return img


def make_control(depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[depth > 0.5] = np.array([255, 255, 255], dtype=np.uint8)
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/nla/gemma4/stereo")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--eye", type=int, default=140)
    p.add_argument("--mu", type=float, default=0.32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    depth = heart_depth(args.size, args.size)
    Image.fromarray(make_control(depth)).save(os.path.join(args.out, "control.png"))
    stereo = make_stereogram(depth, eye=args.eye, mu=args.mu, seed=args.seed)
    Image.fromarray(stereo).save(os.path.join(args.out, "stereogram.png"))
    # a labelled hint of the separation band for the caption (not shown to model)
    print(f"wrote control.png + stereogram.png to {args.out} "
          f"(size {args.size}, eye {args.eye}, mu {args.mu})", flush=True)


if __name__ == "__main__":
    main()
