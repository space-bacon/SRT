#!/usr/bin/env python
"""Lay the room out flat, so it can be looked at.

The gallery is 123,287 points in 1024 dimensions. A map is that room projected
to two, which is a lie in the usual way projections are, but a useful one: what
survives is the neighbourhood structure, and the neighbourhoods are the whole
point. Nearby on the map means nearby in the model's reading.

Also names the regions, by verbalizing the mean of each cluster. Those labels
are not annotations anybody wrote; they are the reader saying what is there.

    python scripts/build_space_map.py --n 40000 --clusters 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_shared_space_verbalizer import load_index  # noqa: E402
from train_shared_space_verbalizer import Prefix  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--npz", default=None,
                   help="gallery_full.npz (Z_img) instead of a .srtidx; this is "
                        "the form the Lab's own gallery is stored in")
    p.add_argument("--ckpt", default="checkpoints/fullstate_verbalizer/gallery_verbalizer.pt")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=40000, help="points to lay out")
    p.add_argument("--clusters", type=int, default=24, help="regions to name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/local/browser/space_map.npz")
    p.add_argument("--out-labels", default="artifacts/local/browser/space_map_labels.json")
    return p.parse_args()


def main():
    import umap
    from sklearn.cluster import KMeans
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse()
    if a.npz:
        z = np.load(a.npz, allow_pickle=True)
        gal = z["Z_img"].astype(np.float32)
    else:
        gal, _ = load_index(a.index)
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(gal), min(a.n, len(gal)), replace=False)
    sel.sort()
    X = gal[sel].astype(np.float32)
    print(f"laying out {len(sel):,} of {len(gal):,} points", flush=True)

    # cosine, because the room is a sphere: every row was L2 normalised.
    xy = umap.UMAP(n_neighbors=25, min_dist=0.12, metric="cosine",
                   random_state=a.seed, verbose=True).fit_transform(X)
    xy = np.asarray(xy, dtype=np.float32)
    xy -= xy.mean(0)
    xy /= np.abs(xy).max() + 1e-8          # into [-1, 1] for easy rendering
    print("layout done", flush=True)

    # Regions, named by the reader rather than by us.
    km = KMeans(n_clusters=a.clusters, random_state=a.seed, n_init=4).fit(xy)
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
    pre.load_state_dict(ck["prefix"])
    pre.eval()

    @torch.no_grad()
    def say(v, max_new=24):
        u = v / (np.linalg.norm(v) + 1e-8)
        x = torch.tensor((u - ck["mu"]) / ck["sd"], dtype=torch.float32).unsqueeze(0)
        soft = pre(x)
        out = lm.generate(inputs_embeds=soft, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id,
                          attention_mask=torch.ones(soft.shape[:2], dtype=torch.long))
        t = tok.decode(out[0], skip_special_tokens=True).strip()
        seen, kept = set(), []
        for s in (p.strip() for p in t.split(".")):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                kept.append(s)
        return (kept[0] + "." if kept else t)

    labels = []
    for c in range(a.clusters):
        m = km.labels_ == c
        centre_xy = xy[m].mean(0)
        centre_v = X[m].mean(0)
        labels.append({
            "x": float(centre_xy[0]), "y": float(centre_xy[1]),
            "n": int(m.sum()), "text": say(centre_v),
        })
        print(f"  region {c:2d} ({m.sum():5d} photos): {labels[-1]['text']}", flush=True)

    np.savez_compressed(a.out, xy=xy, rows=sel.astype(np.int32))
    json.dump({"regions": labels, "n_points": int(len(sel)),
               "n_gallery": int(len(gal))}, open(a.out_labels, "w"), indent=1)
    print(f"wrote {a.out} and {a.out_labels}")


if __name__ == "__main__":
    main()
