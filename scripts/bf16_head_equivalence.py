"""Check whether the shipped head survives a bf16 export, before claiming it does.

The cards quote a 22 MB head, which is the bf16 size of an artifact only ever
shipped as 42 MB fp32. This measures what bf16 actually costs on real L47
states, so the smaller number can be backed by a file instead of arithmetic.

CPU only, by design: a GPU job is usually running and this is a matvec.
"""
from __future__ import annotations

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn.functional as F

HEAD = "/root/sunstone_linear_head_v3_drift.pt"
CACHE = "/root/depth_states_all9.npz"
CACHE_LAYERS = [4, 12, 20, 28, 36, 44, 47, 54, 60]
OUT = "/root/bf16_head_equivalence.json"


def project(state, W, b, mu):
    return F.normalize((torch.from_numpy(state).float() - mu) @ W.T + b, dim=1)


def main() -> None:
    d = torch.load(HEAD, map_location="cpu", weights_only=True)
    z = np.load(CACHE, allow_pickle=False)
    li = CACHE_LAYERS.index(47)
    img, txt = z["img"][:, li], z["txt"][:, li]
    n = len(img)
    print(f"{n} L47 image/caption pairs", flush=True)

    res = {"n_pairs": n, "params": 11022848,
           "question": "does the shipped fp32 head survive a bf16 export"}

    for side, state in (("img", img), ("txt", txt)):
        W32, b32 = d[side]["weight"], d[side]["bias"]
        mu = d[f"mu_{side}"]
        # bf16 round-trip: what a bf16-serialised head computes at inference.
        W16 = W32.to(torch.bfloat16).float()
        b16 = b32.to(torch.bfloat16).float()
        mu16 = mu.to(torch.bfloat16).float()
        a = project(state, W32, b32, mu)
        c = project(state, W16, b16, mu16)
        cos = (a * c).sum(1)
        res[side] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                     "max_abs_weight_delta": float((W32 - W16).abs().max())}
        print(f"  {side}: cos mean {cos.mean():.6f} min {cos.min():.6f}", flush=True)

    # Retrieval is what the head is for, so agreement there is the real test.
    Wi, bi, mui = d["img"]["weight"], d["img"]["bias"], d["mu_img"]
    Wt, bt, mut = d["txt"]["weight"], d["txt"]["bias"], d["mu_txt"]
    A32 = project(img, Wi, bi, mui)
    B32 = project(txt, Wt, bt, mut)
    A16 = project(img, Wi.to(torch.bfloat16).float(), bi.to(torch.bfloat16).float(),
                  mui.to(torch.bfloat16).float())
    B16 = project(txt, Wt.to(torch.bfloat16).float(), bt.to(torch.bfloat16).float(),
                  mut.to(torch.bfloat16).float())
    dg = torch.arange(n)
    for tag, A, B in (("fp32", A32, B32), ("bf16", A16, B16)):
        S = B @ A.T
        rank = (S > S[dg, dg][:, None]).sum(1) + 1
        res[f"t2i_{tag}"] = {"r@1": float((rank == 1).float().mean()),
                             "r@10": float((rank <= 10).float().mean()),
                             "median_rank": float(rank.float().median())}
        print(f"  t2i {tag}: r@1 {res[f't2i_{tag}']['r@1']:.4f} "
              f"median {res[f't2i_{tag}']['median_rank']:.0f}", flush=True)

    res["delta_r@1"] = res["t2i_bf16"]["r@1"] - res["t2i_fp32"]["r@1"]
    res["note"] = (f"pool is {n} images, not the shipped 123,287 gallery, so these "
                   "are agreement figures rather than deployment numbers")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\ndelta r@1 {res['delta_r@1']:+.4f} -> wrote {OUT}")


if __name__ == "__main__":
    main()
