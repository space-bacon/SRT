#!/usr/bin/env python
"""Does the swap-split result survive stripping the provenance formatting?

Round 16. The reviewer showed the generated foil ends with a period and the
human caption often does not, with zero counter-examples in 7,511 pairs, and
priced a surface-only rule at a third of our swap_obj ablation margin.

Three text sources, same head and same images:

    cached  the shipped states, encoded months ago
    raw     re-encoded here from unmodified captions
    norm    re-encoded here with the trailing period stripped and the first
            character lowercased

raw-vs-cached is the re-encode floor. norm-vs-raw is the punctuation effect
with that floor divided out, which is why both are encoded on one box in one
session rather than comparing against the shipped cache alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sugarcrepe_controls import (  # noqa: E402
    CELLS, CLEAN5, SPLITS, ROOT, acc, ci95, gather, load_cell, load_items,
    proj_img, proj_txt,
)
from sugarcrepe_encode_norm import normalise  # noqa: E402


def load_txt(path: Path, key_norm: bool) -> dict:
    z = np.load(path)
    states, texts = z["states"], z["texts"]
    return {(normalise(t) if key_norm else t): states[i].astype(np.float32)
            for i, t in enumerate(texts)}


def arms_for(items, img, txt, h, seeds: int, key_norm: bool) -> dict:
    """real / deranged / mean-image, per split, for one text source."""
    get = (lambda it, k: txt.get(normalise(it[k]))) if key_norm else \
          (lambda it, k: txt.get(it[k]))
    out = {}
    per_split_rows = {}
    for s in SPLITS:
        vi, vp, vn = [], [], []
        for it in items[s]:
            a, p, n = img.get(it["filename"]), get(it, "caption"), get(it, "negative_caption")
            if a is None or p is None or n is None:
                continue
            vi.append(a), vp.append(p), vn.append(n)
        per_split_rows[s] = (np.stack(vi), np.stack(vp), np.stack(vn))

    mean_img = {s: proj_img(per_split_rows[s][0].mean(0, keepdims=True), h)
                for s in SPLITS}
    for s in SPLITS:
        vi, vp, vn = per_split_rows[s]
        n = len(vi)
        zi, zp, zn = proj_img(vi, h), proj_txt(vp, h), proj_txt(vn, h)
        sh = []
        for k in range(seeds):
            g = np.random.default_rng(1000 + k)
            while True:
                perm = g.permutation(n)
                if n < 2 or not (perm == np.arange(n)).any():
                    break
            sh.append(acc(zi[perm], zp, zn))
        out[s] = {
            "n": n,
            "real": acc(zi, zp, zn),
            "shuffle_pair": float(np.mean(sh)),
            "mean_image": acc(np.repeat(mean_img[s], n, 0), zp, zn),
            "ci95_real": ci95(acc(zi, zp, zn), n),
        }
    for k in ("real", "shuffle_pair", "mean_image"):
        out[f"clean5_{k}"] = float(np.mean([out[s][k] for s in CLEAN5]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=list(CELLS))
    p.add_argument("--txt-raw", required=True)
    p.add_argument("--txt-norm", required=True)
    p.add_argument("--work-dir", default="sugarcrepe")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = CELLS[a.cell]
    h, img, txt_cached = load_cell(cfg)
    items = load_items(ROOT / a.work_dir)

    sources = {
        "cached": (txt_cached, False),
        "raw": (load_txt(Path(a.txt_raw), False), False),
        "norm": (load_txt(Path(a.txt_norm), True), True),
    }
    res = {name: arms_for(items, img, t, h, a.seeds, kn)
           for name, (t, kn) in sources.items()}

    print(f"cell {a.cell}\n")
    for arm in ("real", "mean_image", "shuffle_pair"):
        print(f"--- {arm} ---")
        print(f"{'split':12s} {'cached':>8} {'raw':>8} {'norm':>8} "
              f"{'norm-raw':>9} {'raw-cached':>11}")
        for s in CLEAN5 + [f"clean5"]:
            key = s if s in CLEAN5 else f"clean5_{arm}"
            c, r, nm = (res["cached"][key][arm] if s in CLEAN5 else res["cached"][key],
                        res["raw"][key][arm] if s in CLEAN5 else res["raw"][key],
                        res["norm"][key][arm] if s in CLEAN5 else res["norm"][key])
            print(f"{s:12s} {c:8.4f} {r:8.4f} {nm:8.4f} {nm - r:+9.4f} {r - c:+11.4f}")
        print()

    json.dump({"cell": a.cell, "sources": res}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
