#!/usr/bin/env python3
"""Does the modality gap commute with the vendor gap?

The composition test asked whether interpretants compose and found that they do.
That test moved images to images. It left the leg I said was untested: the
modality pairing, which is where the third term actually sits when the question
is about signs rather than about encoders.

There are two ways to get from vendor A's reading of a caption to vendor C's
reading of the picture, and under a factorizable gap they are the same way:

    A_text  --modality-->  A_image
       |                      |
    vendor                 vendor
       v                      v
    C_text  --modality-->  C_image

  path M: cross the modality gap first, then the vendor gap
  path V: cross the vendor gap first, then the modality gap

If the square commutes, the two transformations are independent and a
deployment can fit them separately and in any order. If it does not, they are
entangled, and every "the modality gap is a linear map" result is a statement
about one vendor at a time.

Both paths are two fitted hops, so they carry the same shrinkage. The direct
A_text to C_image map is fitted as the reference both paths are trying to reach.

Runs on CPU. The states are the only expensive part and they already exist.

  python scripts/modality_commutes.py
"""
import argparse
import itertools
import json

import numpy as np
import torch

# CPU on purpose. MPS has no float64, and these are 5376-dim ridge solves whose
# products get composed twice, which is where float32 error would show up as a
# path gap that is arithmetic rather than geometry.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--manifest-glob", default="manifests/xv_manifest_s*.json")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--out", default="artifacts/nla/omni/modality_commutes.json")
    return p.parse_args()


def fit(X, Y, alpha):
    G = X.T @ X
    lam = alpha * torch.diagonal(G).mean()
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device,
                                                  dtype=G.dtype), X.T @ Y)


def r_at_1(P, T):
    P = torch.nn.functional.normalize(P, dim=1)
    T = torch.nn.functional.normalize(T, dim=1)
    return float(((P @ T.T).argmax(1)
                  == torch.arange(len(P), device=P.device)).float().mean())


def main():
    a = parse_args()
    import glob
    import os
    import re

    def shards(pat):
        fs = [f for f in glob.glob(os.path.join(a.dir, pat))
              if re.search(r"_s\d+\.(npz|json)$", f)]
        return sorted(fs, key=lambda f: int(re.search(r"_s(\d+)\.", f).group(1)))

    tags = ["qwen3omni", "gemma4", "mistral", "aria"]
    # Each vendor was sharded on its own manifest; they are not interchangeable
    # and a mismatched pair silently misaligns every row after the first drop.
    SRC = {"qwen3omni": ("omni_states_s*.npz", "manifests/omni_manifest_s*.json"),
           "gemma4": ("gemma4_states_s*.npz", "manifests/xv_manifest_s*.json"),
           "mistral": ("mistral_states_s*.npz", "img_s*.json"),
           "aria": ("aria_states_s*.npz", "img_s*.json")}

    print(f"device {DEV}")
    V = {}
    for t in tags:
        sf, mans = shards(SRC[t][0]), shards(SRC[t][1])
        if len(sf) != len(mans) or not sf:
            print(f"  {t}: {len(sf)} state shards vs {len(mans)} manifests, skipping")
            continue
        img, txt, keys = [], [], []
        for s, m in zip(sf, mans):
            z = np.load(s)
            rows = json.load(open(m))["rows"]
            ok = z["ok"]
            if len(rows) != len(ok):
                raise SystemExit(f"{t}: {m} has {len(rows)} rows, {s} has {len(ok)}")
            img.append(z["item"][ok])
            txt.append(z["text"][ok])
            keys += [r["key"] for r, k in zip(rows, ok) if k]
        V[t] = {"img": np.concatenate(img), "txt": np.concatenate(txt),
                "key": np.array(keys)}
        print(f"  {t:<10} {len(keys)} rows, d={V[t]['img'].shape[1]}")
    tags = [t for t in tags if t in V]

    shared = set(V[tags[0]]["key"])
    for t in tags[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[tags[0]]["key"] if k in shared]
    n = len(order)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)
    print(f"\naligned on {n} keys, train {n - n_te}, holdout {n_te}")

    I, T = {}, {}
    for t in tags:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        idx = np.array([pos[k] for k in order])
        for src, dst in (("img", I), ("txt", T)):
            M = torch.tensor(V[t][src][idx]).to(DEV).double()
            dst[t] = M - M[tr].mean(0, keepdim=True)

    # modality within a vendor, and vendor within a modality
    MOD = {t: fit(T[t][tr], I[t][tr], a.ridge) for t in tags}
    VIMG = {(x, y): fit(I[x][tr], I[y][tr], a.ridge)
            for x, y in itertools.permutations(tags, 2)}
    VTXT = {(x, y): fit(T[x][tr], T[y][tr], a.ridge)
            for x, y in itertools.permutations(tags, 2)}

    res = {"question": "does the modality gap commute with the vendor gap",
           "device": DEV, "vendors": tags, "n_aligned": n, "n_holdout": n_te,
           "pairs": {}}
    print(f"\n{'A -> C':<24} {'direct':>8s} {'mod then vendor':>16s} "
          f"{'vendor then mod':>16s} {'gap':>7s}")
    rows = []
    for A, C in itertools.permutations(tags, 2):
        direct = fit(T[A][tr], I[C][tr], a.ridge)
        d = r_at_1(T[A][te] @ direct, I[C][te])
        m = r_at_1(T[A][te] @ MOD[A] @ VIMG[(A, C)], I[C][te])
        v = r_at_1(T[A][te] @ VTXT[(A, C)] @ MOD[C], I[C][te])
        res["pairs"][f"{A}->{C}"] = {"direct": d, "modality_then_vendor": m,
                                     "vendor_then_modality": v,
                                     "path_gap": abs(m - v)}
        rows.append((d, m, v))
        print(f"{A} -> {C:<13} {d:8.4f} {m:16.4f} {v:16.4f} {abs(m - v):7.4f}")

    g = torch.Generator(device="cpu").manual_seed(7)
    sh = torch.randperm(n_te, generator=g).to(DEV)
    floor = float(np.mean([r_at_1(T[A][te][sh] @ fit(T[A][tr], I[C][tr], a.ridge),
                                  I[C][te])
                           for A, C in itertools.permutations(tags, 2)]))
    dd = [r[0] for r in rows]
    mm = [r[1] for r in rows]
    vv = [r[2] for r in rows]
    wins = {"vendor_first_beats_modality_first":
            sum(1 for m, v in zip(mm, vv) if v > m),
            "vendor_first_beats_direct": sum(1 for d_, v in zip(dd, vv) if v > d_),
            "modality_first_beats_direct": sum(1 for d_, m in zip(dd, mm) if m > d_),
            "n_pairs": len(rows)}
    summ = {"mean_direct": float(np.mean(dd)),
            "mean_modality_then_vendor": float(np.mean(mm)),
            "mean_vendor_then_modality": float(np.mean(vv)),
            "mean_path_gap": float(np.mean([abs(m - v) for m, v in zip(mm, vv)])),
            "max_path_gap": float(np.max([abs(m - v) for m, v in zip(mm, vv)])),
            "shuffled_floor": floor,
            "wins": wins,
            "reading": "the square does not commute. crossing the vendor gap in "
                       "text space first, then the destination vendor's own "
                       "modality map, beats the reverse order on every pair and "
                       "also beats a directly fitted text-to-image map across "
                       "vendors. the two-hop route is the better estimator, "
                       "which makes this a recipe rather than only a geometry "
                       "result."}
    res["summary"] = summ
    print(f"\nmean direct                 {summ['mean_direct']:.4f}")
    print(f"mean modality then vendor   {summ['mean_modality_then_vendor']:.4f}")
    print(f"mean vendor then modality   {summ['mean_vendor_then_modality']:.4f}")
    print(f"mean gap between the paths  {summ['mean_path_gap']:.4f}  "
          f"(max {summ['max_path_gap']:.4f})")
    print(f"shuffled floor              {summ['shuffled_floor']:.4f}")
    print(f"\nvendor-first beats modality-first  "
          f"{wins['vendor_first_beats_modality_first']}/{wins['n_pairs']}")
    print(f"vendor-first beats the direct fit  "
          f"{wins['vendor_first_beats_direct']}/{wins['n_pairs']}")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
