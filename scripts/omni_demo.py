#!/usr/bin/env python3
"""Search one vendor's gallery with another vendor's text encoder.

Runs on CPU against the published states, so the cross-vendor result is
checkable without the four GPUs and four 24-31B hosts that produced it.

  python scripts/omni_demo.py                       # download from the Hub
  python scripts/omni_demo.py --local artifacts/nla/omni
  python scripts/omni_demo.py --query "a man riding a horse on the beach"
"""
import argparse
import json
import pathlib
import re

import numpy as np

REPO = "RiverRider/srt-omni-crossvendor-states"
VENDORS = {
    "qwen3omni": ("omni_states_s*.npz", "omni_manifest_s*.json"),
    "gemma4": ("gemma4_states_s*.npz", "xv_manifest_s*.json"),
    "mistral": ("mistral_states_s*.npz", "img_s*.json"),
    "aria": ("aria_states_s*.npz", "img_s*.json"),
}


def shard_key(f):
    return int(re.search(r"_s(\d+)\.", str(f)).group(1))


def fetch(root, vendor):
    """Load one vendor's rows, keyed by manifest key so vendors can be aligned."""
    states_pat, man_pat = VENDORS[vendor]
    if root:
        base = pathlib.Path(root)
        sf = sorted(base.glob(states_pat), key=shard_key)
        mf = sorted(list(base.glob(man_pat)) + list((base / "manifests").glob(man_pat)),
                    key=shard_key)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(REPO, repo_type="dataset")
        sn = sorted([f for f in files if f.startswith("states/")
                     and pathlib.Path(f).match("*" + states_pat)], key=shard_key)
        mn = sorted([f for f in files if f.startswith("manifests/")
                     and pathlib.Path(f).match("*" + man_pat)], key=shard_key)
        sf = [pathlib.Path(hf_hub_download(REPO, f, repo_type="dataset")) for f in sn]
        mf = [pathlib.Path(hf_hub_download(REPO, f, repo_type="dataset")) for f in mn]

    item, text, keys, caps = [], [], [], []
    for s, m in zip(sf, mf):
        z = np.load(s)
        rows = json.load(open(m))["rows"]
        ok = z["ok"]
        item.append(z["item"][ok])
        text.append(z["text"][ok])
        keys += [r["key"] for r, k in zip(rows, ok) if k]
        caps += [r["caption"] for r, k in zip(rows, ok) if k]
    return (np.concatenate(item).astype(np.float32),
            np.concatenate(text).astype(np.float32), np.array(keys), np.array(caps))


def fit(Xi, Ta, Tb, dim, epochs, lr):
    """One item tower and one text tower per vendor, into a shared space.

    Each vendor needs its own tower because hidden sizes differ (5376 vs 2560),
    and all pairs are trained together so the space is shared rather than two
    spaces that happen to sit side by side.
    """
    mus = [Xi.mean(0, keepdims=True), Ta.mean(0, keepdims=True), Tb.mean(0, keepdims=True)]
    A, Pa, Pb = Xi - mus[0], Ta - mus[1], Tb - mus[2]
    rng = np.random.default_rng(0)
    W = [rng.normal(0, 0.02, (M.shape[1], dim)).astype(np.float32) for M in (A, Pa, Pb)]
    n = len(A)
    for _ in range(epochs):
        Z = []
        for M, w in zip((A, Pa, Pb), W):
            Y = M @ w
            Z.append(Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8))
        grads = [np.zeros_like(w) for w in W]
        for ti in (1, 2):
            S = Z[0] @ Z[ti].T / 0.05
            S -= S.max(1, keepdims=True)
            G = np.exp(S)
            G /= G.sum(1, keepdims=True)
            G[np.arange(n), np.arange(n)] -= 1.0
            G /= n
            grads[0] += A.T @ (G @ Z[ti])
            grads[ti] += (Pa if ti == 1 else Pb).T @ (G.T @ Z[0])
        for i in range(3):
            W[i] -= lr * grads[i]
    return W, mus


def project(X, W, mu):
    Y = (X - mu) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="Directory of states, else the Hub.")
    ap.add_argument("--gallery", default="gemma4", choices=list(VENDORS))
    ap.add_argument("--query-vendor", default="aria", choices=list(VENDORS))
    ap.add_argument("--query", default=None, help="Caption text to look up.")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--holdout", type=int, default=1000)
    a = ap.parse_args()

    print(f"loading {a.gallery} (gallery) and {a.query_vendor} (queries)")
    Ig, Tg, Kg, Cg = fetch(a.local, a.gallery)
    Iq, Tq, Kq, Cq = fetch(a.local, a.query_vendor)

    pos = {k: i for i, k in enumerate(Kq)}
    shared = [k for k in Kg if k in pos]
    ig = np.array([i for i, k in enumerate(Kg) if k in pos])
    iq = np.array([pos[k] for k in shared])
    print(f"aligned on {len(shared)} shared keys")

    n = len(shared)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    te, tr = perm[:min(a.holdout, n // 4)], perm[min(a.holdout, n // 4):]

    # Both vendors' towers are fitted together on the train split; the holdout
    # is what decides whether the cross direction survives.
    W, mus = fit(Ig[ig][tr], Tg[ig][tr], Tq[iq][tr], a.dim, a.epochs, a.lr)

    G = project(Ig[ig][te], W[0], mus[0])
    caps = Cg[ig][te]

    def rank_of(Q):
        S = Q @ G.T
        d = np.arange(len(Q))
        return (S > S[d, d][:, None]).sum(1) + 1

    res = {}
    for label, T, w, mu in ((f"within ({a.gallery})", Tg[ig][te], W[1], mus[1]),
                            (f"CROSS  ({a.query_vendor})", Tq[iq][te], W[2], mus[2])):
        r = rank_of(project(T, w, mu))
        res[label] = (r == 1).mean()
        print(f"  {label:22s} r@1 {(r == 1).mean():.4f}  "
              f"r@10 {(r <= 10).mean():.4f}  median {np.median(r):.0f}")
    w_, c_ = list(res.values())
    print(f"  retention (cross/within) {c_ / w_:.3f}")
    print("  note: a plain numpy fit, weaker than the torch fit behind the"
          " published numbers; the cross/within comparison is the point.")

    if a.query:
        sims = [len(set(a.query.lower().split()) & set(c.lower().split())) for c in caps]
        hit = int(np.argmax(sims))
        print(f"\nclosest held-out caption to your query:\n  {caps[hit]}")
        q = project(Tq[iq][te][hit:hit + 1], W[2], mus[2])
        top = np.argsort(-(q @ G.T)[0])[:5]
        print(f"\ntop 5 {a.gallery} gallery items for that caption, "
              f"queried by {a.query_vendor}:")
        for j, t in enumerate(top, 1):
            print(f"  {j}. {'<-- correct' if t == hit else '           '} {caps[t][:70]}")


if __name__ == "__main__":
    main()
