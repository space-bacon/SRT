#!/usr/bin/env python3
"""Their protocol, our models: sampled intra- and inter-model similarity.

The greedy run answered the wrong question. Artificial Hivemind (arXiv:2510.22954)
samples at top-p 0.9, T=1.0, fifty responses per query. Greedy returns the mode of
a distribution; they measured the distribution. Comparing our greedy number to
their sampled band confounds decoding with post-training, and greedy cannot touch
their headline claim at all, because intra-model similarity needs more than one
sample to exist.

Greedy also flatters nobody here: it drives base models into degenerate loops
(gpt-oss repeating "So the total number of servings is 1"), which is a decode
artifact being scored as if it were content.

So: same 12 open-weight models, same 60 prompts, K samples each under their
settings. Three arms, all on one footing.

  intra-model  same model, same prompt, different samples   (their 79% above 0.8)
  inter-model  different models, same prompt                (their 0.71 to 0.82)
  floor        different prompts                            (their 0.1 to 0.2)

Scored with MiniLM, whose different-prompt floor lands near theirs, so the scale
is comparable. Whether these base models reach their band is the open question,
and a shortfall now means post-training rather than decoding.

    python scripts/hivemind_sampled_protocol.py --k 8 --prompts 60
"""
import argparse
import gc
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openweight_transport_atlas import MODELS
from hivemind_mechanism_link import STEMS, embed, spearman, pval

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
ATLAS = "artifacts/nla/atlas/openweight_transport_atlas.json"
GEN_DIR = "artifacts/nla/atlas/samples"
# Their stated settings.
TOP_P, TEMP = 0.9, 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8, help="samples per prompt per model")
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--max-gb", type=float, default=28.0)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_sampled_protocol.json")
    return p.parse_args()


def sample(tag, mid, dtype, prompts, k, max_new):
    path = f"{GEN_DIR}/{tag}.json"
    if os.path.isfile(path):
        r = json.load(open(path))
        if len(r) == len(prompts) and len(r[0]) == k:
            print(f"  {tag:14s} cached")
            return r
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mod = AutoModelForCausalLM.from_pretrained(mid, dtype=getattr(torch, dtype)).to(DEV).eval()
    torch.manual_seed(0)
    out = []
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            b = tok([pr], return_tensors="pt").to(DEV)
            g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                             top_p=TOP_P, temperature=TEMP, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
            out.append([tok.decode(g[j][b["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip() for j in range(k)])
            print(f"\r  {tag:14s} {i + 1}/{len(prompts)}", end="", flush=True)
    os.makedirs(GEN_DIR, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\r  {tag:14s} done")
    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def main():
    a = parse_args()
    prompts = STEMS[:a.prompts]
    atlas = json.load(open(ATLAS))
    roster = [m for m in MODELS if m[3] <= a.max_gb and m[0] in atlas["matrix"]]
    print(f"device {DEV}  {len(roster)} models  {len(prompts)} prompts  "
          f"K={a.k}  top_p={TOP_P} T={TEMP}\n")

    S = {}
    for tag, mid, lab, gb, dt in roster:
        try:
            S[tag] = sample(tag, mid, dt, prompts, a.k, a.max_new)
        except Exception as e:
            print(f"  {tag:14s} SKIP {type(e).__name__}: {str(e)[:60]}")
    tags = list(S)

    flat = [t for tg in tags for row in S[tg] for t in row]
    print(f"\nembedding {len(flat)} samples")
    E = embed(flat).reshape(len(tags), len(prompts), a.k, -1)

    # intra: same model, same prompt, across samples
    iu, ju = zip(*itertools.combinations(range(a.k), 2))
    intra = {}
    for m, tg in enumerate(tags):
        v = (E[m][:, iu, :] * E[m][:, ju, :]).sum(-1).mean(1)   # per prompt
        intra[tg] = {"mean": round(float(v.mean()), 4),
                     "frac_prompts_above_0.8": round(float((v > 0.8).mean()), 4)}

    # inter: different models, same prompt, mean over all sample pairs
    inter, tr, pairs = [], [], []
    M = E.mean(2)  # sample-mean per model per prompt is a fair pair summary
    for i, j in itertools.combinations(range(len(tags)), 2):
        s = float((E[i][:, :, None, :] * E[j][:, None, :, :]).sum(-1).mean())
        inter.append(s)
        tr.append(0.5 * (atlas["matrix"][tags[i]][tags[j]]
                         + atlas["matrix"][tags[j]][tags[i]]))
        pairs.append((tags[i], tags[j]))
    inter, tr = np.array(inter), np.array(tr)

    rng = np.random.default_rng(0)
    F = E.reshape(-1, E.shape[-1])
    pi = rng.integers(0, len(F), 20000); qi = rng.integers(0, len(F), 20000)
    per = len(prompts) * a.k
    ok = ((pi % per) // a.k) != ((qi % per) // a.k)
    floor = float((F[pi[ok]] * F[qi[ok]]).sum(-1).mean())

    meta = atlas["models"]
    cross = [k for k, (i, j) in enumerate(itertools.combinations(range(len(tags)), 2))
             if meta[tags[i]]["lab"].split("/")[0] != meta[tags[j]]["lab"].split("/")[0]]
    keep = [k for k, p in enumerate(pairs) if set(p) != {"qwen25_7b", "qwen25_7b_f32"}]
    im = np.array([intra[t]["mean"] for t in tags])

    res = {
        "protocol": f"top_p={TOP_P}, temperature={TEMP}, K={a.k}, {a.max_new} new tokens",
        "matches": "arXiv:2510.22954 sampling settings",
        "n_models": len(tags), "n_prompts": len(prompts),
        "intra_model": {
            "per_model": intra,
            "mean_across_models": round(float(im.mean()), 4),
            "range": [round(float(im.min()), 4), round(float(im.max()), 4)],
            "paper_claim": "79% of queries above 0.8"},
        "inter_model": {
            "mean": round(float(inter.mean()), 4),
            "range": [round(float(inter.min()), 4), round(float(inter.max()), 4)],
            "cross_lab_mean": round(float(inter[cross].mean()), 4),
            "paper_range": [0.71, 0.82]},
        "floor_different_prompt": round(floor, 4),
        "link_transport_vs_inter": {
            "all": {"n": len(pairs), "rho": round(spearman(tr, inter), 4),
                    "p": round(pval(spearman(tr, inter), len(pairs)), 4)},
            "minus_anchor": {"n": len(keep), "rho": round(spearman(tr[keep], inter[keep]), 4),
                             "p": round(pval(spearman(tr[keep], inter[keep]), len(keep)), 4)},
            "cross_lab": {"n": len(cross), "rho": round(spearman(tr[cross], inter[cross]), 4),
                          "p": round(pval(spearman(tr[cross], inter[cross]), len(cross)), 4)}},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'model':<16}{'intra':>8}{'frac>0.8':>10}")
    for t in tags:
        print(f"  {t:<16}{intra[t]['mean']:>6.4f}{intra[t]['frac_prompts_above_0.8']:>10.2f}")
    r = res
    print(f"\nintra-model  mean {r['intra_model']['mean_across_models']}  "
          f"range {r['intra_model']['range']}   paper: 79% of queries > 0.8")
    print(f"inter-model  mean {r['inter_model']['mean']}  "
          f"cross-lab {r['inter_model']['cross_lab_mean']}   paper: 0.71 - 0.82")
    print(f"floor        {r['floor_different_prompt']}                    paper: 0.1 - 0.2")
    print("\nSPEARMAN transport vs sampled inter-model")
    for k, v in r["link_transport_vs_inter"].items():
        print(f"  {k:14s} n={v['n']:3d}  rho {v['rho']:+.4f}  p {v['p']:.4f}"
              f"{'' if v['p'] < 0.05 else '   n.s.'}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
