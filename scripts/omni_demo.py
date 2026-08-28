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


def fit(mats, dim, epochs, lr, batch=512):
    """One tower per input, every item/text pair trained together.

    Each vendor needs its own tower because hidden sizes differ (2048 to 5376).
    Inputs are centred and scaled to unit RMS first, otherwise a shared lr
    trains the towers at different speeds and that shows up as a fake vendor
    difference.
    """
    mus = [M.mean(0, keepdims=True) for M in mats]
    C = [M - mu for M, mu in zip(mats, mus)]
    scales = [float(np.sqrt((M ** 2).sum(1).mean())) + 1e-8 for M in C]
    C = [M / s for M, s in zip(C, scales)]
    rng = np.random.default_rng(0)
    W = [rng.normal(0, 0.02, (M.shape[1], dim)).astype(np.float32) for M in C]
    items, texts = (0, 2), (1, 3)  # [itemA, textA, itemB, textB]
    n = len(C[0])
    for _ in range(epochs):
        for s in range(0, n, batch):
            sl = slice(s, min(s + batch, n))
            B = [M[sl] for M in C]
            b = len(B[0])
            if b < 8:
                continue
            Z = []
            for M, w in zip(B, W):
                Y = M @ w
                Z.append(Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8))
            g = [np.zeros_like(w) for w in W]
            for i in items:
                for t in texts:
                    S = Z[i] @ Z[t].T / 0.05
                    S -= S.max(1, keepdims=True)
                    G = np.exp(S)
                    G /= G.sum(1, keepdims=True)
                    G[np.arange(b), np.arange(b)] -= 1.0
                    G /= b
                    g[i] += B[i].T @ (G @ Z[t])
                    g[t] += B[t].T @ (G.T @ Z[i])
            for k in range(4):
                W[k] -= lr * g[k]
    return W, mus, scales


def project(X, W, mu, scale):
    Y = ((X - mu) / scale) @ W
    return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="Directory of states, else the Hub.")
    ap.add_argument("--gallery", default="gemma4", choices=list(VENDORS))
    ap.add_argument("--query-vendor", default="aria", choices=list(VENDORS))
    ap.add_argument("--query", default=None, help="Caption text to look up.")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--holdout", type=int, default=800)
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
    hold = min(a.holdout, n // 4)
    te, tr = perm[:hold], perm[hold:]

    # [item_A, text_A, item_B, text_B]. Both cross directions get measured,
    # because a single direction is far too noisy at this fit quality to say
    # anything: it lands anywhere from 0.65 to 1.36 depending on the pair.
    W, mus, sc = fit([Ig[ig][tr], Tg[ig][tr], Iq[iq][tr], Tq[iq][tr]],
                     a.dim, a.epochs, a.lr)

    held = [Ig[ig][te], Tg[ig][te], Iq[iq][te], Tq[iq][te]]
    Z = [project(M, w, mu, s) for M, w, mu, s in zip(held, W, mus, sc)]
    caps = Cg[ig][te]
    names = [a.gallery, a.query_vendor]

    def r1(item, text):
        S = Z[text] @ Z[item].T
        d = np.arange(len(S))
        return float(((S > S[d, d][:, None]).sum(1) + 1 == 1).mean())

    for i in (0, 1):
        for j in (0, 1):
            kind = "within" if i == j else "CROSS "
            print(f"  {kind} {names[i]:>10s} gallery <- {names[j]:<10s} text "
                  f"r@1 {r1(i * 2, j * 2 + 1):.4f}")
    within = np.mean([r1(0, 1), r1(2, 3)])
    cross = np.mean([r1(0, 3), r1(2, 1)])
    print(f"  mean cross {cross:.4f}  mean within {within:.4f}  "
          f"retention {cross / within:.3f}")
    print("  note: a plain numpy fit, much weaker than the torch fit behind the"
          " published retention 0.988, 95% CI [0.955, 1.023].")

    if a.query:
        sims = [len(set(a.query.lower().split()) & set(c.lower().split())) for c in caps]
        hit = int(np.argmax(sims))
        if sims[hit] == 0:
            print("\nno held-out caption shares a word with that query")
            return
        print(f"\nclosest held-out caption to your query:\n  {caps[hit]}")
        top = np.argsort(-(Z[3][hit:hit + 1] @ Z[0].T)[0])[:5]
        print(f"\ntop 5 {a.gallery} gallery items for that caption, "
              f"queried by {a.query_vendor}:")
        for j, t in enumerate(top, 1):
            print(f"  {j}. {'<-- correct' if t == hit else '           '} {caps[t][:70]}")


if __name__ == "__main__":
    main()
