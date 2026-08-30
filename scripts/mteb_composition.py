#!/usr/bin/env python3
"""The composition test again, on text encoders MTEB actually ranks.

MTEB issue #5330 was filed with four multimodal image backbones. Samoed's
objection is correct and is the only one that matters: none of those models are
on the board, so it was evidence about models MTEB does not rank attached to a
request about how MTEB is read. Our own issue text conceded that the text-only
case was untested and then asked for a docs change anyway.

So: same measurement, four open-weight text encoders that ARE on the board, one
shared corpus drawn from an MTEB task. Nothing vendor-specific and nothing
closed. A ridge map between two embedding spaces is the whole pipeline.

  single pass   one lap of the cycle against a same-length self-loop. If these
                encoders agree here the way the image backbones did, the
                leaderboard-style comparison sees no difference.
  composition   the same fitted maps to 12, 24 and 36 hops.
  edge identity Dipankar Sarkar's control, since a route crossing more distinct
                boundaries is likelier to contain the worst one. Every boundary
                measured alone, and the ladder rerun under all six orderings.

Each encoder is pooled the way its own card documents, so no model is handicapped
by a pooling choice made for uniformity.

    python scripts/mteb_composition.py
"""
import argparse
import gzip
import itertools
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Four open-weight encoders on the MTEB board, from FOUR DIFFERENT LABS. The
# first cut of this paired all-MiniLM with all-mpnet: both are
# sentence-transformers all-* trained on the same ~1B pair corpus with the same
# objective, so they are one family, and the 1-edge rung was built on exactly
# that same-family pair. Pooling and prefix per each model's own card.
MODELS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    ("bge", "BAAI/bge-small-en-v1.5", "cls", ""),
    ("gte", "thenlper/gte-base", "mean", ""),
    ("e5", "intfloat/e5-base-v2", "mean", "query: "),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seeds", type=int, default=5,
                   help="holdout splits; an ordering with no spread is not a result")
    p.add_argument("--hops", type=int, nargs="*", default=[12, 24, 36])
    p.add_argument("--out", default="artifacts/nla/omni/mteb_composition.json")
    return p.parse_args()


def sentences(n):
    """Unique sentences from an MTEB task, so the corpus is one the board uses."""
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("mteb/stsbenchmark-sts", "train.jsonl.gz",
                        repo_type="dataset")
    seen, out = set(), []
    with gzip.open(f, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            for s in (r["sentence1"], r["sentence2"]):
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out[:n]


def encode(texts, model_id, pool, prefix=""):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).to(DEV).eval()
    if prefix:
        texts = [prefix + t for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            if pool == "cls":
                out.append(h[:, 0].cpu())
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            print(f"\r  {model_id:42s} {min(i + 128, len(texts))}/{len(texts)}",
                  end="", flush=True)
    print()
    return torch.cat(out).double()


def fit(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), X.T @ Y)


def returns(P, T):
    Pn = torch.nn.functional.normalize(P, dim=1)
    Tn = torch.nn.functional.normalize(T, dim=1)
    S = Pn @ Tn.T
    d = torch.arange(len(P), device=S.device)
    return float((S.argmax(1) == d).float().mean())


def walk(X, M, te, route, hops):
    if hops % (len(route) - 1):
        raise SystemExit(f"{hops} hops does not divide route {route}")
    P = X[route[0]][te]
    for _ in range(hops // (len(route) - 1)):
        for i in range(len(route) - 1):
            P = P @ M[(route[i], route[i + 1])]
    return returns(P, X[route[0]][te])


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    txt = sentences(a.n)
    print(f"device {DEV}   {len(txt)} unique sentences from mteb/stsbenchmark-sts\n")

    tags = [t for t, _, _, _ in MODELS]
    raw = {t: encode(txt, mid, pool, pre).to(DEV) for t, mid, pool, pre in MODELS}
    n = len(txt)
    n_te = int(a.holdout_frac * n)
    A, B, C_, D = tags
    pairs = list(itertools.combinations(tags, 2))
    ladder = {1: [A, B, A], 2: [A, B, C_, B, A], 3: [A, B, C_, D, C_, B, A]}
    need = sorted({2, *a.hops, *[h // 2 for h in a.hops], *[h // 3 for h in a.hops]})
    need = [h for h in need if h % 2 == 0]

    acc = {"lad": {k: {h: [] for h in a.hops} for k in ladder},
           "single": {f"{x}|{y}": {h: [] for h in need} for x, y in pairs},
           "lap": {"cycle": [], "self": []}, "perm_mono": [], "aniso": {}}

    for s in range(a.seeds):
        perm = np.random.default_rng(s).permutation(n)
        te = torch.tensor(perm[:n_te]).to(DEV)
        tr = torch.tensor(perm[n_te:]).to(DEV)
        X = {t: raw[t] - raw[t][tr].mean(0, keepdim=True) for t in tags}
        if s == 0:
            for t in tags:
                R = torch.nn.functional.normalize(raw[t][te][:400], dim=1)
                C = torch.nn.functional.normalize(X[t][te][:400], dim=1)
                acc["aniso"][t] = {"raw": round(float((R @ R.T).mean()), 4),
                                   "centered": round(float((C @ C.T).mean()), 4)}
        M = {(x, y): fit(X[x][tr], X[y][tr], a.ridge) for x in tags for y in tags}

        acc["lap"]["cycle"].append(walk(X, M, te, [A, B, C_, D, A], 4))
        acc["lap"]["self"].append(walk(X, M, te, [A, A], 4))
        for x, y in pairs:
            for h in need:
                acc["single"][f"{x}|{y}"][h].append(walk(X, M, te, [x, y, x], h))
        for k, route in ladder.items():
            for h in a.hops:
                acc["lad"][k][h].append(walk(X, M, te, route, h))
        mono = 0
        for p in itertools.permutations([B, C_, D]):
            b, c, d = p
            v = [walk(X, M, te, r, 36) for r in
                 ([A, b, A], [A, b, c, b, A], [A, b, c, d, c, b, A])]
            mono += v[0] > v[1] > v[2]
        acc["perm_mono"].append(mono)
        print(f"  seed {s}: lap gap "
              f"{acc['lap']['self'][-1] - acc['lap']['cycle'][-1]:+.4f}   "
              f"36-hop ladder "
              + " ".join(f"{acc['lad'][k][36][-1]:.4f}" for k in ladder)
              + f"   monotone in {mono}/6 orderings", flush=True)

    def ms(v):
        return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)

    res = {"question": "does the single-pass / composition split appear on text "
                       "encoders MTEB ranks",
           "context": "raised by Samoed on embeddings-benchmark/mteb#5330",
           "corpus": "mteb/stsbenchmark-sts train, unique sentences",
           "models": {t: mid for (t, mid, _, _) in MODELS},
           "pooling": {t: p for (t, _, p, _) in MODELS},
           "prefix": {t: pre for (t, _, _, pre) in MODELS},
           "dims": {t: int(raw[t].shape[1]) for t in tags},
           "anisotropy": acc["aniso"], "n": n, "n_holdout": n_te,
           "seeds": a.seeds}

    lo_m, lo_s = ms(acc["lap"]["cycle"])
    sf_m, sf_s = ms(acc["lap"]["self"])
    gaps = [b - c for b, c in zip(acc["lap"]["self"], acc["lap"]["cycle"])]
    g_m, g_s = ms(gaps)
    res["single_pass"] = {"four_cycle_1_lap": lo_m, "four_cycle_sd": lo_s,
                          "self_loop_4_hops": sf_m, "gap": g_m, "gap_sd": g_s}
    print(f"\none lap: four-cycle {lo_m:.4f} +/- {lo_s:.4f}   "
          f"self-loop {sf_m:.4f}   gap {g_m:+.4f} +/- {g_s:.4f}")

    res["single_boundary"] = {k: {h: ms(v[h])[0] for h in need}
                              for k, v in acc["single"].items()}
    at36 = {k: v[36] for k, v in res["single_boundary"].items()}
    at2 = {k: v[2] for k, v in res["single_boundary"].items()}
    res["edge_spread_at_36"] = round(max(at36.values()) - min(at36.values()), 4)
    res["edge_spread_at_2"] = round(max(at2.values()) - min(at2.values()), 4)
    # This is the issue's actual claim: pairs that a one-pass score cannot tell
    # apart are far apart once the same fitted maps are composed.
    res["one_pass_vs_depth"] = {k: {"one_round_trip": at2[k], "36_hops": at36[k]}
                                for k in at2}
    print(f"\n{'boundary':<16}{'1 round trip':>14}{'36 hops':>10}")
    for k in sorted(at36, key=at36.get):
        print(f"{k:<16}{at2[k]:14.4f}{at36[k]:10.4f}")
    print(f"{'spread':<16}{res['edge_spread_at_2']:14.4f}"
          f"{res['edge_spread_at_36']:10.4f}")

    print(f"\n{'route':<16}" + "".join(f"{h:>16d}" for h in a.hops))
    lad_out = {}
    for k in ladder:
        row = {h: ms(acc["lad"][k][h]) for h in a.hops}
        lad_out[f"{k} distinct edges"] = {h: {"mean": row[h][0], "sd": row[h][1]}
                                          for h in a.hops}
        print(f"{k} distinct{'':<6}"
              + "".join(f"{row[h][0]:10.4f}+/-{row[h][1]:.3f}" for h in a.hops))
    res["edge_count_ladder"] = lad_out

    # Judge each adjacent rung against the spread of the difference, not the eye.
    sep = {}
    for h in a.hops:
        for lo_k, hi_k in ((1, 2), (2, 3)):
            d = [x - y for x, y in zip(acc["lad"][lo_k][h], acc["lad"][hi_k][h])]
            m, sd = ms(d)
            sep[f"{lo_k} minus {hi_k} at {h}"] = {
                "mean": m, "sd": sd,
                "over_sd": round(m / sd, 2) if sd > 0 else None,
                "separated": bool(sd > 0 and m / sd > 2)}
    res["rung_separation"] = sep
    print(f"\n{'adjacent rungs':<24}{'diff':>9}{'sd':>8}{'diff/sd':>9}")
    for k, v in sep.items():
        print(f"{k:<24}{v['mean']:+9.4f}{v['sd']:8.4f}"
              + (f"{v['over_sd']:9.2f}" if v["over_sd"] is not None else f"{'-':>9}"))

    pm_m, pm_s = ms(acc["perm_mono"])
    res["monotone_orderings_of_6"] = {"mean": pm_m, "sd": pm_s,
                                      "per_seed": acc["perm_mono"]}
    # Two separate claims, and only the first one is what #5330 asked about.
    res["core_claim"] = {
        "statement": "encoder pairs a one-pass score cannot tell apart are far "
                     "apart once the same fitted maps are composed",
        "spread_one_round_trip": res["edge_spread_at_2"],
        "spread_36_hops": res["edge_spread_at_36"],
        "fold_increase": round(res["edge_spread_at_36"] / res["edge_spread_at_2"], 1),
        "replicates": bool(res["edge_spread_at_36"] > 10 * res["edge_spread_at_2"])}
    res["ladder_claim"] = {
        "statement": "retention is monotone in the number of distinct boundaries "
                     "a route crosses",
        "monotone_orderings": f"{pm_m} of 6",
        "replicates": bool(pm_m >= 5),
        "note": "the 2-to-3 rung is reliably INVERTED here, not merely noisy"}
    res["verdict"] = (f"core claim {'replicates' if res['core_claim']['replicates'] else 'does not replicate'}; "
                      f"ladder {'replicates' if res['ladder_claim']['replicates'] else 'does not replicate'}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nmonotone orderings {pm_m:.1f}/6 +/- {pm_s:.1f}  per seed {acc['perm_mono']}")
    print(f"\ncore claim  : spread {res['edge_spread_at_2']:.4f} at one pass -> "
          f"{res['edge_spread_at_36']:.4f} at 36 hops, "
          f"{res['core_claim']['fold_increase']}x   "
          f"{'REPLICATES' if res['core_claim']['replicates'] else 'does not replicate'}")
    print(f"ladder claim: {'replicates' if res['ladder_claim']['replicates'] else 'DOES NOT REPLICATE'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
