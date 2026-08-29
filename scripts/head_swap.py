#!/usr/bin/env python3
"""Swap the caption tower for an unrelated encoder and see if cross and within move together.

I claimed retention is close to insensitive to the vendor boundary because both
of its terms are limited by the caption head: within-vendor text-to-image sits
at 0.1050 while vendor-to-vendor image agreement sits at 0.8024.

Dipankar Sarkar's test settles it, and it is the right one. Hold the image side
fixed and vary the head. Put an unrelated off-the-shelf text encoder in as the
caption tower and refit. If cross and within both scale by roughly the same
factor, the head is the shared bottleneck. If they move apart, retention is not
head-limited and my reframe does not hold.

The swapped head is a small general-purpose sentence encoder with no connection
to any of the four vendors, which is the point: it shares no training run, no
tokenizer and no architecture with the image towers it is asked to serve.

Everything is centred on train rows and every retrieval is on the holdout.

  python scripts/head_swap.py --domain roco
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["qwen3omni", "gemma4", "mistral", "aria"]
HEAD = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--domain", default="roco")
    p.add_argument("--head", default=HEAD)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=None)
    return p.parse_args()


def encode_captions(caps, model_id):
    """Mean-pooled sentence embeddings from an encoder unrelated to every vendor."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(caps), 256):
            b = tok(caps[i:i + 256], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            print(f"\r  encoding captions {min(i + 256, len(caps))}/{len(caps)}",
                  end="", flush=True)
    print()
    return torch.cat(out).double()


def infonce_towers(Timg, Ttxt, tr, te, dim, epochs, lr, seed=0):
    """One linear tower each side, trained with InfoNCE, scored by r@1.

    float32 and a sampled batch: unlike the composed ridge walks, nothing here
    is multiplied by itself dozens of times, and the full 4000x4000 logits
    matrix in float64 costs far more than the result is worth.
    """
    g = torch.Generator().manual_seed(seed)
    Ti, Tt = Timg.float(), Ttxt.float()
    Wi = torch.nn.Linear(Ti.shape[1], dim).to(DEV)
    Wt = torch.nn.Linear(Tt.shape[1], dim).to(DEV)
    opt = torch.optim.AdamW(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)
    bs = min(1024, len(tr))
    lab = torch.arange(bs, device=DEV)
    for _ in range(epochs):
        sel = tr[torch.randperm(len(tr), generator=g)[:bs]]
        a = torch.nn.functional.normalize(Wi(Ti[sel]), dim=1)
        b = torch.nn.functional.normalize(Wt(Tt[sel]), dim=1)
        logits = a @ b.T * 20.0
        loss = (torch.nn.functional.cross_entropy(logits, lab)
                + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = torch.nn.functional.normalize(Wi(Ti[te]), dim=1)
        b = torch.nn.functional.normalize(Wt(Tt[te]), dim=1)
        S = b @ a.T
        return float((S.argmax(1) == torch.arange(len(te), device=DEV)).float().mean())


def main():
    a = parse_args()
    a.out = a.out or os.path.join(a.dir, f"head_swap_{a.domain}.json")
    rows = json.load(open(os.path.join(a.dir, f"{a.domain}_manifest.json")))["rows"]

    print(f"device {DEV}   domain {a.domain}   swapped head {a.head}")
    V = {}
    for t in TAGS:
        f = os.path.join(a.dir, "states", f"{a.domain}_{t}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ok = z["ok"]
        V[t] = {"img": z["item"][ok], "txt": z["text"][ok],
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k]),
                "cap": [r["caption"] for r, k in zip(rows, ok) if k]}
    tags = [t for t in TAGS if t in V]
    shared = set(V[tags[0]]["key"])
    for t in tags[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[tags[0]]["key"] if k in shared]
    n = len(order)
    perm = np.random.default_rng(0).permutation(n)
    n_te = int(a.holdout_frac * n)
    te = torch.tensor(perm[:n_te]).to(DEV)
    tr = torch.tensor(perm[n_te:]).to(DEV)
    print(f"  aligned {n}, train {n - n_te}, holdout {n_te}")

    cap_by_key = dict(zip(V[tags[0]]["key"], V[tags[0]]["cap"]))
    caps = [cap_by_key[k] for k in order]
    SWAP = encode_captions(caps, a.head).to(DEV)
    SWAP = SWAP - SWAP[tr].mean(0, keepdim=True)

    I, T = {}, {}
    for t in tags:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        idx = np.array([pos[k] for k in order])
        for src, dst in (("img", I), ("txt", T)):
            M = torch.tensor(V[t][src][idx]).to(DEV).double()
            dst[t] = M - M[tr].mean(0, keepdim=True)

    res = {"question": "does swapping the caption tower move cross and within by "
                       "the same factor",
           "credit": "test proposed by Dipankar Sarkar",
           "domain": a.domain, "vendors": tags, "swapped_head": a.head,
           "n_holdout": n_te, "native": {}, "swapped": {}}

    print(f"\n{'':<26}{'native head':>13}{'swapped head':>14}{'ratio':>8}")
    for label, pairs in (("within", [(t, t) for t in tags]),
                         ("cross", list(itertools.permutations(tags, 2)))):
        nat = [infonce_towers(I[x], T[y], tr, te, a.dim, a.epochs, a.lr)
               for x, y in pairs]
        swp = [infonce_towers(I[x], SWAP, tr, te, a.dim, a.epochs, a.lr)
               for x, _ in pairs]
        res["native"][label] = round(float(np.mean(nat)), 4)
        res["swapped"][label] = round(float(np.mean(swp)), 4)
        r = np.mean(swp) / np.mean(nat) if np.mean(nat) else float("nan")
        res[f"{label}_ratio"] = round(float(r), 4)
        print(f"{label + ' r@1':<26}{np.mean(nat):13.4f}{np.mean(swp):14.4f}{r:8.3f}")

    gap = abs(res["within_ratio"] - res["cross_ratio"])
    res["ratio_gap"] = round(gap, 4)
    res["verdict"] = ("head-limited" if gap < 0.15 else "not head-limited")
    res["reading"] = (
        "the two terms scaling by the same factor when the head is replaced is "
        "what 'both limited by the caption head' predicts. the terms moving "
        "apart would mean retention is not head-limited and the reframe should "
        "be withdrawn.")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwithin scales {res['within_ratio']}, cross scales {res['cross_ratio']}, "
          f"gap {gap:.4f}")
    print(f"verdict: {res['verdict']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
