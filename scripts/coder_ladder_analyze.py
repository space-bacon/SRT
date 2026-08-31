#!/usr/bin/env python3
"""The section-5 scaling curve: does the chat-template effect hold across 66x?

paper_hivemind section 5 measured six matched base/instruct pairs between 0.36B
and 2B and found instruction tuning worth +0.0786 intra-model similarity while
the same weights through a chat template were worth +0.3623. The standing
objection is that this is a small-model artifact.

This computes the same quantities at every rung of the Qwen2.5-Coder ladder,
0.5B to 32B, one lab, one recipe, one tokenizer. Per rung:

  base raw      the pretrained model, bare prompt
  inst raw      tuned weights, bare prompt        -> tuning alone
  inst chat     tuned weights, own template       -> tuning + format
  inst shared   generic markers, not native       -> frame without tuned tokens

A flat curve kills the small-model objection. A bending curve is a scaling law
for homogeneity, which is a better result than the one we set out to get.

Floors are different-prompt pairs computed through the identical machinery, and
every arm is scored with the same encoder, so levels are comparable across rungs.

    python scripts/coder_ladder_analyze.py
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

GEN = "artifacts/nla/coder_ladder"
SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.1, "7B": 7.6, "14B": 14.8, "32B": 32.8}
SCORER = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", default=GEN)
    p.add_argument("--prefix", default="coder")
    p.add_argument("--sizes", nargs="*", default=SIZES)
    p.add_argument("--params", nargs="*", type=float, default=None)
    p.add_argument("--family", default="Qwen2.5-Coder, one lab, one recipe, one tokenizer")
    p.add_argument("--domain", default="HumanEval prompts")
    p.add_argument("--out", default="artifacts/nla/coder_ladder/scaling_curve.json")
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def embed(texts, batch):
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(SCORER)
    mod = AutoModel.from_pretrained(SCORER).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    X = torch.cat(out).double()
    return torch.nn.functional.normalize(X, dim=1).numpy()


def arm_stats(E, k, rng):
    """intra over sample pairs, and a different-prompt floor through the same path."""
    n = E.shape[0]
    iu, ju = zip(*itertools.combinations(range(k), 2))
    v = (E[:, iu, :] * E[:, ju, :]).sum(-1).mean(1)
    F = E.reshape(-1, E.shape[-1])
    p = rng.integers(0, len(F), 20000)
    q = rng.integers(0, len(F), 20000)
    ok = (p // k) != (q // k)
    return (round(float(v.mean()), 4),
            round(float((v > 0.8).mean()), 4),
            round(float((F[p[ok]] * F[q[ok]]).sum(-1).mean()), 4))


def main():
    a = parse_args()
    GEN, SIZES = a.gen_dir, a.sizes
    PARAMS = (dict(zip(a.sizes, a.params)) if a.params
              else {s: globals()["PARAMS"][s] for s in a.sizes})
    files = {f[:-5]: f"{GEN}/{f}" for f in os.listdir(GEN) if f.endswith(".json")
             and "__" in f}
    if not files:
        sys.exit(f"no generations in {GEN}")
    print(f"found {len(files)} generation files")

    rows, texts, index = {}, [], []
    for key, path in sorted(files.items()):
        g = json.load(open(path))
        rows[key] = g
        for pi, row in enumerate(g):
            for s in row:
                texts.append(s if s.strip() else " ")
                index.append(key)
    print(f"embedding {len(texts):,} generations with {SCORER.split('/')[-1]}")
    X = embed(texts, a.batch)

    rng = np.random.default_rng(0)
    off, stats = 0, {}
    for key in sorted(rows):
        g = rows[key]
        n, k = len(g), len(g[0])
        E = X[off:off + n * k].reshape(n, k, -1)
        off += n * k
        m, frac, floor = arm_stats(E, k, rng)
        stats[key] = {"intra": m, "frac_above_0.8": frac, "floor": floor,
                      "n_prompts": n, "k": k}

    res = {"question": "does the chat-template effect hold across model scale",
           "family": a.family,
           "domain": a.domain, "scorer": SCORER,
           "per_arm": stats, "curve": {}}

    print(f"\n{'size':<7}{'params':>8}{'base raw':>10}{'inst raw':>10}"
          f"{'inst chat':>11}{'shared':>9}{'tuning':>9}{'format':>9}")
    for s in SIZES:
        b = stats.get(f"{a.prefix}{s}_base__raw", {}).get("intra")
        ir = stats.get(f"{a.prefix}{s}_inst__raw", {}).get("intra")
        ic = stats.get(f"{a.prefix}{s}_inst__chat", {}).get("intra")
        sh = stats.get(f"{a.prefix}{s}_inst__shared", {}).get("intra")
        if None in (b, ir, ic):
            print(f"{s:<7}{PARAMS[s]:>8.1f}   incomplete")
            continue
        res["curve"][s] = {
            "params_b": PARAMS[s], "base_raw": b, "inst_raw": ir,
            "inst_chat": ic, "inst_shared": sh,
            "tuning_gain": round(ir - b, 4),
            "format_gain": round(ic - ir, 4),
            "total_gain": round(ic - b, 4),
            "frac_above_0.8_chat": stats.get(f"{a.prefix}{s}_inst__chat", {}).get("frac_above_0.8"),
            "floor_chat": stats.get(f"{a.prefix}{s}_inst__chat", {}).get("floor")}
        c = res["curve"][s]
        print(f"{s:<7}{PARAMS[s]:>8.1f}{b:>10.4f}{ir:>10.4f}{ic:>11.4f}"
              f"{(sh if sh else float('nan')):>9.4f}"
              f"{c['tuning_gain']:>+9.4f}{c['format_gain']:>+9.4f}")

    if len(res["curve"]) >= 3:
        xs = np.log10([v["params_b"] for v in res["curve"].values()])
        fg = np.array([v["format_gain"] for v in res["curve"].values()])
        tg = np.array([v["tuning_gain"] for v in res["curve"].values()])
        sl_f = float(np.polyfit(xs, fg, 1)[0])
        sl_t = float(np.polyfit(xs, tg, 1)[0])
        res["trend"] = {
            "format_gain_slope_per_decade": round(sl_f, 4),
            "tuning_gain_slope_per_decade": round(sl_t, 4),
            "format_gain_range": [round(float(fg.min()), 4), round(float(fg.max()), 4)],
            "paper_small_model_format_gain": 0.3623,
            "reading": ("a slope near zero means the format effect is scale-free and "
                        "the small-model objection to section 5 fails; a clearly "
                        "negative slope means it decays with scale and the paper needs "
                        "to say so")}
        print(f"\nformat gain vs log10(params): slope {sl_f:+.4f} per decade, "
              f"range {fg.min():.4f} to {fg.max():.4f}")
        print(f"tuning gain vs log10(params): slope {sl_t:+.4f} per decade")
        print(f"paper's small-model format gain was +0.3623")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
