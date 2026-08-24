"""Export a published SRT head to safetensors, with reference projections.

The Rust port in `rust/srt-geometry` has to produce the same numbers as the
Python path that generated every result in the papers. This writes both halves
of that check: the head in a format Rust can load, and a fixture of real states
with their expected projections.

`cargo test -p srt-geometry` fails if the port drifts. That is the point: a
second implementation of the read-out geometry is only worth having if it is
pinned to the first.

    python scripts/export_head_safetensors.py \
        --head checkpoints/gemma4_readout/gemma_v6_head.pt \
        --states artifacts/nla/q4/... .npz \
        --out rust/srt-geometry/tests/fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--states", type=Path, default=None,
                   help="npz with a 'states' array to draw fixture inputs from")
    p.add_argument("--modality", choices=["img", "txt"], default="txt",
                   help="which side the fixture states belong to")
    p.add_argument("--n-fixture", type=int, default=8)
    p.add_argument("--sides", default="img,txt",
                   help="which projections to include; a text-only runtime "
                        "searches a gallery that was projected elsewhere")
    p.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = torch.load(args.head, map_location="cpu", weights_only=True)

    # v2 heads may carry a non-linear image side under "img_state", which has
    # no safetensors equivalent here and is never shipped anyway.
    tensors = {
        "txt.weight": d["txt"]["weight"].float().contiguous(),
        "txt.bias": d["txt"]["bias"].float().contiguous(),
        "mu_txt": d["mu_txt"].float().contiguous(),
    }
    if isinstance(d.get("img"), dict) and "weight" in d["img"]:
        tensors["img.weight"] = d["img"]["weight"].float().contiguous()
        tensors["img.bias"] = d["img"]["bias"].float().contiguous()
        tensors["mu_img"] = d["mu_img"].float().contiguous()
    sides = {s.strip() for s in args.sides.split(",") if s.strip()}
    keep = {k: v for k, v in tensors.items() if k.split(".")[0].replace("mu_", "") in sides}
    if args.dtype == "f16":
        keep = {k: v.half().contiguous() for k, v in keep.items()}

    from safetensors.torch import save_file
    st_path = args.out / "head.safetensors"
    save_file(keep, str(st_path))
    print(f"wrote {st_path} ({sorted(sides)}, {args.dtype}, "
          f"{st_path.stat().st_size / 1e6:.1f} MB)")

    if args.states is None:
        return

    z = np.load(args.states, allow_pickle=True)
    states = z["states"][: args.n_fixture].astype(np.float32)
    side, mu_key = args.modality, f"mu_{'img' if args.modality == 'img' else 'txt'}"
    W = tensors[f"{side}.weight"].numpy()
    b = tensors[f"{side}.bias"].numpy()
    mu = tensors[mu_key].numpy()
    proj = (states - mu) @ W.T + b
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8

    fixture = {
        "modality": "Text" if side == "txt" else "Image",
        "d_in": int(states.shape[1]),
        "d_out": int(proj.shape[1]),
        "states": states.tolist(),
        "expected": proj.tolist(),
        "source_head": str(args.head),
        "note": ("expected = normalize((state - mu) @ W.T + b), the projection "
                 "every published number in this program was computed through"),
    }
    fx = args.out / "projection_fixture.json"
    fx.write_text(json.dumps(fixture))
    print(f"wrote {fx} ({len(states)} states, {states.shape[1]} -> {proj.shape[1]})")


if __name__ == "__main__":
    main()
