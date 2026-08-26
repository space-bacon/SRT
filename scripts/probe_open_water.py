#!/usr/bin/env python
"""Is a point on the map genuinely between things, or is it just a photograph?

The Lab's map invites you to click "where there is nothing". That phrase does
not survive contact with the code. Clicking calls `map_vector_at(x, y, k=48)`,
which averages the 1024-d vectors of the 48 nearest MAPPED photographs, so
every click reads an average of real photographs and none of them read nothing.

An earlier version of this check asked whether any map dots sat within 0.05 of
the clicked coordinate. That was the wrong measurement three times over: the
radius was invented, UMAP coordinates preserve neighbourhood rather than
distance, and the instrument averages 48 neighbours regardless of how far away
they are.

What actually distinguishes arithmetic from retrieval is measurable:

  novelty    how well the averaged vector matches its BEST match anywhere in
             the gallery, calibrated against how well real photographs match
             their own nearest neighbour. A point whose best match is as good
             as a typical photograph's neighbour is, for practical purposes,
             that photograph. A point whose best match is much weaker sits
             where no photograph sits.

  mixedness  where the 48 contributing photographs come from. Half from each
             parent region is a blend. Forty-seven from one is that one.

  novel word a word in the sentence that appears in NEITHER parent's name.
             "Snowboard" is in neither "skiing" nor "skateboarding".

    python scripts/probe_open_water.py --map artifacts/local/space_map.npz \
        --labels artifacts/local/space_map_labels.json \
        --gallery artifacts/local/gallery_full.npz --ckpt <reader.pt>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_shared_space_verbalizer import Prefix  # noqa: E402

STOP = {"a", "an", "the", "of", "on", "in", "with", "and", "is", "are", "at",
        "to", "next", "some", "his", "her", "its", "that", "this", "it",
        "there", "down", "up", "for", "from", "by", "near", "sitting",
        "standing", "man", "woman", "person", "people", "group", "young"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="artifacts/local/space_map.npz")
    p.add_argument("--labels", default="artifacts/local/space_map_labels.json")
    p.add_argument("--gallery", default="artifacts/local/gallery_full.npz")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--k", type=int, default=48, help="must match the server's map_vector_at")
    p.add_argument("--baseline-n", type=int, default=2000)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse()
    m = np.load(a.map)
    xy, rows = m["xy"].astype(np.float32), m["rows"].astype(np.int64)
    regions = json.load(open(a.labels))["regions"]
    Z = np.load(a.gallery, allow_pickle=False)["Z_img"].astype(np.float32)
    print(f"map {xy.shape[0]:,} points, gallery {Z.shape[0]:,}, k={a.k}", flush=True)

    # How well does a REAL photograph match its nearest neighbour? Without this
    # the novelty number is a bare cosine with nothing to be compared against.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(Z), size=min(a.baseline_n, len(Z)), replace=False)
    nn = np.empty(len(sample), dtype=np.float32)
    for i in range(0, len(sample), 256):
        blk = sample[i:i + 256]
        s = Z[blk] @ Z.T
        s[np.arange(len(blk)), blk] = -1.0          # exclude self
        nn[i:i + len(blk)] = s.max(1)
    print(f"gallery nearest-neighbour cosine: median {np.median(nn):.4f}, "
          f"5th pct {np.percentile(nn, 5):.4f}, 95th {np.percentile(nn, 95):.4f}",
          flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
    pre.load_state_dict(ck["prefix"])
    pre.eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    cent = np.array([[r["x"], r["y"]] for r in regions], dtype=np.float32)

    def say(v):
        x = torch.tensor((v - ck["mu"]) / ck["sd"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            soft = pre(x)
            out = lm.generate(inputs_embeds=soft, do_sample=False,
                              attention_mask=torch.ones(soft.shape[:2], dtype=torch.long),
                              max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
        t = tok.decode(out[0], skip_special_tokens=True).strip()
        seen, kept = set(), []
        for s in (p.strip() for p in t.split(".")):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                kept.append(s)
        return ". ".join(kept[:2]) + "." if kept else t

    def words(s):
        return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}

    def probe(ai, bi):
        ra, rb = regions[ai], regions[bi]
        mid = np.array([(ra["x"] + rb["x"]) / 2, (ra["y"] + rb["y"]) / 2], np.float32)
        d = ((xy - mid) ** 2).sum(1)
        near = np.argpartition(d, a.k)[:a.k]
        v = Z[rows[near]].mean(0)
        v /= np.linalg.norm(v) + 1e-8

        sims = Z @ v
        best = float(sims.max())
        pct = float((nn < best).mean() * 100)          # where it sits in the NN distribution

        owner = np.argmin(((xy[near][:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        frac_a = float((owner == ai).mean())
        frac_b = float((owner == bi).mean())
        pair = Z[rows[near]]
        cross = float((pair @ pair.T)[np.triu_indices(len(pair), 1)].mean())

        text = say(v)
        novel = sorted(words(text) - words(ra["text"]) - words(rb["text"]))
        return {
            "between": [ra["text"], rb["text"]],
            "midpoint": [float(mid[0]), float(mid[1])],
            "sentence": text,
            "novelty_best_cos": round(best, 4),
            "novelty_pct_of_real_nn": round(pct, 1),
            "frac_from_parent_a": round(frac_a, 3),
            "frac_from_parent_b": round(frac_b, 3),
            "frac_from_elsewhere": round(1 - frac_a - frac_b, 3),
            "mean_pairwise_cos_of_48": round(cross, 4),
            "novel_words": novel,
        }

    def find(kw):
        hits = [i for i, r in enumerate(regions) if kw.lower() in r["text"].lower()]
        if not hits:
            raise SystemExit(f"no region matching {kw!r}")
        return max(hits, key=lambda i: regions[i]["n"])

    # Every distinct pair would be 276 probes; these are the ones the post uses.
    names = [("skiing", "skateboard"), ("kite", "frisbee"),
             ("living room", "cat sitting on a bed"), ("kitchen", "plate of food"),
             ("giraffes", "elephants"), ("tennis", "baseball"),
             ("motorcycle", "bus is parked"), ("jetliner", "train traveling")]
    out = []
    for ka, kb in names:
        try:
            r = probe(find(ka), find(kb))
        except SystemExit as e:
            print(f"  skip {ka}+{kb}: {e}", flush=True)
            continue
        out.append(r)
        print(f"\n{ka} + {kb}  ({r['midpoint'][0]:+.2f},{r['midpoint'][1]:+.2f})", flush=True)
        print(f"  -> {r['sentence']}", flush=True)
        print(f"  novelty  best cos {r['novelty_best_cos']:.4f}  "
              f"= {r['novelty_pct_of_real_nn']:.0f}th pct of real neighbour cosines", flush=True)
        print(f"  mix      {r['frac_from_parent_a']:.0%} / {r['frac_from_parent_b']:.0%} parents, "
              f"{r['frac_from_elsewhere']:.0%} elsewhere, 48-set cohesion {r['mean_pairwise_cos_of_48']:.3f}",
              flush=True)
        print(f"  novel    {r['novel_words']}", flush=True)

    if a.out:
        json.dump({
            "question": "is a clicked point on the map arithmetic or retrieval",
            "k": a.k,
            "note": "every click averages the k nearest MAPPED photographs, so no click "
                    "reads empty space. Novelty is the averaged vector's best match in the "
                    "gallery, expressed as a percentile of the cosine real photographs "
                    "achieve with their own nearest neighbour. Low percentile means the "
                    "point sits where no photograph sits.",
            "gallery_nn_cosine": {
                "median": round(float(np.median(nn)), 4),
                "p5": round(float(np.percentile(nn, 5)), 4),
                "p95": round(float(np.percentile(nn, 95)), 4),
                "n": int(len(nn)),
            },
            "map": a.map, "gallery": a.gallery, "ckpt": a.ckpt,
            "probes": out,
        }, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
