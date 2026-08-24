"""End-to-end retrieval reference: the numbers the Rust port must reproduce.

Parity on a single projection is necessary but weak. This runs the whole path
the deployment runs, image state -> head -> index -> ranked captions, on real
SugarCrepe/COCO data, and writes both the result and a fixture the Rust
integration test replays.

If the Rust and Python R@k disagree, the port is wrong, and the difference
shows up here rather than in a demo.

    python scripts/retrieval_reference.py \
        --cell qwen3b --out-fixtures rust/srt-geometry/tests/fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_index_srtidx import write_index

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]

CELLS = {
    "qwen3b": {
        "head": "checkpoints/gemma4_readout/qwen3b_v6_head.pt",
        "img": "artifacts/nla/q4/qwen3b_caches/sc_img_states_qwen3b.npz",
        "txt": "artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz",
    },
    "gemma": {
        "head": "checkpoints/gemma4_readout/gemma_v6_head.pt",
        "img": "sugarcrepe/img_states.npz",
        "txt": "sugarcrepe/txt_states.npz",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", choices=sorted(CELLS), default="qwen3b")
    p.add_argument("--n-queries", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out-fixtures", type=Path,
                   default=ROOT / "rust/srt-geometry/tests/fixtures")
    p.add_argument("--out-result", type=Path,
                   default=ROOT / "artifacts/nla/q4/retrieval_reference.json")
    return p.parse_args()


def load(cell: dict):
    head = torch.load(ROOT / cell["head"], map_location="cpu", weights_only=True)
    h = {
        "Wi": head["img"]["weight"].float().numpy(),
        "bi": head["img"]["bias"].float().numpy(),
        "mi": head["mu_img"].float().numpy(),
        "Wt": head["txt"]["weight"].float().numpy(),
        "bt": head["txt"]["bias"].float().numpy(),
        "mt": head["mu_txt"].float().numpy(),
    }
    iz = np.load(ROOT / cell["img"], allow_pickle=True)
    img = {str(f): iz["states"][i].astype(np.float32)
           for i, f in enumerate(iz["files"])}
    tz = np.load(ROOT / cell["txt"], allow_pickle=True)
    ts, tn = tz["states"], tz["texts"]
    txt = {str(t): ts[i].astype(np.float32) for i, t in enumerate(tn)}
    return h, img, txt


def unit(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    a = parse_args()
    cell = CELLS[a.cell]
    h, img, txt = load(cell)

    # One positive caption per image, deduplicated: captions recur across
    # splits, and letting a duplicate into the gallery makes rank-1 ambiguous.
    pairs: dict[str, str] = {}
    for s in SPLITS:
        with open(ROOT / "sugarcrepe" / f"{s}.json") as f:
            for it in json.load(f).values():
                fn, cap = it["filename"], it["caption"]
                if fn in img and cap in txt:
                    pairs.setdefault(fn, cap)

    captions, seen = [], set()
    for cap in pairs.values():
        if cap not in seen:
            seen.add(cap)
            captions.append(cap)
    cap_row = {c: i for i, c in enumerate(captions)}

    Zt = unit((np.stack([txt[c] for c in captions]) - h["mt"]) @ h["Wt"].T + h["bt"])
    files = sorted(pairs)
    Zi = unit((np.stack([img[f] for f in files]) - h["mi"]) @ h["Wi"].T + h["bi"])

    # Score against the fp16 index that actually ships, not the fp32 vectors it
    # was made from. The two disagree on near-ties, and reporting the fp32
    # number would be reporting a gallery no deployment ever holds. No second
    # normalization here: the reader does not renormalize either, so doing it
    # would reintroduce the same mismatch in the other direction.
    Zt_idx = Zt.astype(np.float16).astype(np.float32)

    S = Zi @ Zt_idx.T
    order = np.argsort(-S, axis=1)
    gold = np.array([cap_row[pairs[f]] for f in files])
    rank = np.array([int(np.where(order[i] == gold[i])[0][0]) for i in range(len(files))])

    result = {
        "cell": a.cell,
        "n_images": len(files),
        "n_captions": len(captions),
        "i2t": {f"r@{k}": round(float((rank < k).mean()), 4) for k in (1, 5, 10)},
        "median_rank": float(np.median(rank) + 1),
        "note": "one positive caption per image, captions deduplicated across splits",
    }

    # Shuffled control: the same scorer against permuted gold. Anything near
    # this floor is not retrieval, it is the shape of the score distribution.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    shuf = np.array([int(np.where(order[i] == gold[perm[i]])[0][0]) for i in range(len(files))])
    result["shuffled_control"] = {f"r@{k}": round(float((shuf < k).mean()), 4) for k in (1, 5, 10)}

    a.out_result.parent.mkdir(parents=True, exist_ok=True)
    a.out_result.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # Fixtures for the Rust replay.
    a.out_fixtures.mkdir(parents=True, exist_ok=True)
    idx_path = a.out_fixtures / "captions.srtidx"
    write_index(idx_path, Zt, np.array(captions))

    q = np.linspace(0, len(files) - 1, a.n_queries).astype(int)
    fixture = {
        "cell": a.cell,
        "topk": a.topk,
        "queries": [
            {
                "file": files[i],
                "state": img[files[i]].tolist(),
                "gold": pairs[files[i]],
                "expect": [
                    {"key": captions[j], "score": float(S[i, j])}
                    for j in order[i, : a.topk]
                ],
            }
            for i in q
        ],
        "i2t": result["i2t"],
    }
    fx = a.out_fixtures / "retrieval_fixture.json"
    fx.write_text(json.dumps(fixture))
    print(f"wrote {idx_path} and {fx} ({len(q)} queries)")


if __name__ == "__main__":
    main()
