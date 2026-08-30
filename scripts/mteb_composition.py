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

# All four are open-weight and on the MTEB leaderboard. Pooling per model card.
MODELS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "mean"),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2", "mean"),
    ("bge", "BAAI/bge-small-en-v1.5", "cls"),
    ("gte", "thenlper/gte-base", "mean"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--ridge", type=float, default=1e-2)
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


def encode(texts, model_id, pool):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).to(DEV).eval()
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

    tags = [t for t, _, _ in MODELS]
    raw = {}
    for t, mid, pool in MODELS:
        raw[t] = encode(txt, mid, pool).to(DEV)

    n = len(txt)
    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)

    X, aniso = {}, {}
    for t in tags:
        R = torch.nn.functional.normalize(raw[t][te][:400], dim=1)
        aniso[t] = round(float((R @ R.T).mean()), 4)
        X[t] = raw[t] - raw[t][tr].mean(0, keepdim=True)
        C = torch.nn.functional.normalize(X[t][te][:400], dim=1)
        aniso[t] = {"raw": aniso[t], "centered": round(float((C @ C.T).mean()), 4)}
    print(f"\n{'model':<10}{'dim':>6}{'raw cos':>10}{'centered':>10}")
    for t in tags:
        print(f"{t:<10}{X[t].shape[1]:>6}{aniso[t]['raw']:>10.4f}"
              f"{aniso[t]['centered']:>10.4f}")

    M = {(x, y): fit(X[x][tr], X[y][tr], a.ridge) for x in tags for y in tags}
    A, B, C_, D = tags
    res = {"question": "does the single-pass / composition split appear on text "
                       "encoders MTEB ranks",
           "context": "raised by Samoed on embeddings-benchmark/mteb#5330: the "
                      "original filing used models MTEB does not rank",
           "corpus": "mteb/stsbenchmark-sts train, unique sentences",
           "models": {t: mid for (t, mid, _) in MODELS},
           "pooling": {t: p for (t, _, p) in MODELS},
           "dims": {t: int(X[t].shape[1]) for t in tags},
           "anisotropy": aniso, "n": n, "n_holdout": n_te}

    # 1. one lap, which is what a leaderboard-style comparison would see
    cyc = walk(X, M, te, [A, B, C_, D, A], 4)
    self4 = walk(X, M, te, [A, A], 4)
    res["single_pass"] = {"four_cycle_1_lap": round(cyc, 4),
                          "self_loop_4_hops": round(self4, 4),
                          "gap": round(self4 - cyc, 4)}
    print(f"\none lap:  four-cycle {cyc:.4f}   self-loop {self4:.4f}"
          f"   gap {self4 - cyc:+.4f}")

    # 2. every boundary alone, so edge identity is visible
    single = {}
    for x, y in itertools.combinations(tags, 2):
        single[f"{x}|{y}"] = {h: round(walk(X, M, te, [x, y, x], h), 4)
                              for h in sorted({*a.hops, *[h // 2 for h in a.hops],
                                               *[h // 3 for h in a.hops]})
                              if h % 2 == 0}
    res["single_boundary"] = single
    at36 = {k: v[36] for k, v in single.items() if 36 in v}
    print(f"\nsingle boundary at 36 hops: "
          + ", ".join(f"{k} {v:.4f}" for k, v in sorted(at36.items(), key=lambda x: x[1])))
    res["edge_spread_at_36"] = round(max(at36.values()) - min(at36.values()), 4)

    # 3. the ladder
    ladder = {1: [A, B, A], 2: [A, B, C_, B, A], 3: [A, B, C_, D, C_, B, A]}
    print(f"\n{'route':<16}" + "".join(f"{h:>10d}" for h in a.hops))
    lad = {}
    for k, route in ladder.items():
        row = {h: round(walk(X, M, te, route, h), 4) for h in a.hops}
        lad[f"{k} distinct edges"] = row
        print(f"{k} distinct{'':<6}" + "".join(f"{row[h]:10.4f}" for h in a.hops))
    res["edge_count_ladder"] = lad
    res["ladder_monotone"] = all(
        lad["1 distinct edges"][h] > lad["2 distinct edges"][h] >
        lad["3 distinct edges"][h] for h in a.hops)

    # 4. scramble which boundary sits in which slot
    print(f"\nladder under every ordering, 36 hops")
    perms, n_mono = {}, 0
    for p in itertools.permutations([B, C_, D]):
        b, c, d = p
        v = [walk(X, M, te, r, 36) for r in
             ([A, b, A], [A, b, c, b, A], [A, b, c, d, c, b, A])]
        mono = v[0] > v[1] > v[2]
        n_mono += mono
        perms[", ".join(p)] = {"1": round(v[0], 4), "2": round(v[1], 4),
                               "3": round(v[2], 4), "monotone": mono}
        print(f"  {', '.join(p):<26}" + "".join(f"{x:9.4f}" for x in v)
              + f"   {'yes' if mono else 'NO'}")
    res["ladder_under_permutations_36"] = perms
    res["monotone_permutations"] = f"{n_mono} of {len(perms)}"

    res["verdict"] = ("replicates on MTEB text encoders"
                      if res["ladder_monotone"] and n_mono == len(perms)
                      else "does NOT replicate on MTEB text encoders")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nmonotone in {n_mono} of {len(perms)} orderings")
    print(f"verdict: {res['verdict']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
