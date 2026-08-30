#!/usr/bin/env python3
"""Head swap, redone with one unrelated text tower per vendor slot.

The first version of this test (scripts/head_swap.py) is degenerate and its
result should not be used. It replaced all four caption towers with a SINGLE
shared encoder, so on the swapped side the pair (image x, text y) lost its
dependence on y entirely:

    swp = mean(infonce(I[x], SWAP) for x, _ in pairs)

The y is discarded. Swapped-within and swapped-cross then estimate the same
quantity, and they did: 0.0835 against 0.0833, a difference of 0.0002. With the
numerator shared, the ratio comparison collapses to cross_native / within_native,
which §3.1 already reported directly. More refits could not have fixed that.

Fixed here by giving each vendor slot its OWN unrelated text encoder, so
swapped-within is (image x, swapped text x') and swapped-cross is
(image x, swapped text y') with x' != y'. The within/cross structure survives the
swap, which is what the test needs in order to say anything.

  both ratios move together  -> the caption head is the shared bottleneck
  they move apart            -> retention is not head-limited, reframe withdrawn

The four swapped encoders share no training run, tokenizer or architecture with
any image tower they are asked to serve. Slot assignment is fixed and arbitrary;
it is recorded in the artifact.

    python scripts/head_swap_multi.py --domain roco --seeds 5
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["qwen3omni", "gemma4", "mistral", "aria"]

# One unrelated encoder per vendor slot, four different labs, none connected to
# any image tower here. Pooling and prefix per each model's own card.
SWAP_FOR = {
    "qwen3omni": ("sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    "gemma4": ("BAAI/bge-small-en-v1.5", "cls", ""),
    "mistral": ("thenlper/gte-base", "mean", ""),
    "aria": ("intfloat/e5-base-v2", "mean", "query: "),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="artifacts/nla/omni")
    p.add_argument("--domain", default="roco")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default=None)
    return p.parse_args()


def encode(caps, model_id, pool, prefix):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).to(DEV).eval()
    if prefix:
        caps = [prefix + c for c in caps]
    out = []
    with torch.no_grad():
        for i in range(0, len(caps), 128):
            b = tok(caps[i:i + 128], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            if pool == "cls":
                out.append(h[:, 0].cpu())
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            print(f"\r  {model_id:42s} {min(i + 128, len(caps))}/{len(caps)}",
                  end="", flush=True)
    print()
    return torch.cat(out).double()


def infonce(Timg, Ttxt, tr, te, dim, epochs, lr, seed):
    # Seed the global RNG too: Linear() draws its init from it, so seeding only
    # the batch generator left repeat calls non-reproducible.
    torch.manual_seed(seed)
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
    a.out = a.out or os.path.join(a.dir, f"head_swap_multi_{a.domain}.json")
    rows = json.load(open(os.path.join(a.dir, f"{a.domain}_manifest.json")))["rows"]
    print(f"device {DEV}   domain {a.domain}")

    V = {}
    for t in TAGS:
        z = np.load(os.path.join(a.dir, "states", f"{a.domain}_{t}.npz"))
        ok = z["ok"]
        V[t] = {"img": z["item"][ok], "txt": z["text"][ok],
                "key": np.array([r["key"] for r, k in zip(rows, ok) if k]),
                "cap": [r["caption"] for r, k in zip(rows, ok) if k]}
    shared = set(V[TAGS[0]]["key"])
    for t in TAGS[1:]:
        shared &= set(V[t]["key"])
    order = [k for k in V[TAGS[0]]["key"] if k in shared]
    n = len(order)
    n_te = int(a.holdout_frac * n)
    print(f"  aligned {n}, holdout {n_te}")

    cap_by_key = dict(zip(V[TAGS[0]]["key"], V[TAGS[0]]["cap"]))
    caps = [cap_by_key[k] for k in order]
    SW = {t: encode(caps, *SWAP_FOR[t]).to(DEV) for t in TAGS}

    I, T = {}, {}
    for t in TAGS:
        pos = {k: i for i, k in enumerate(V[t]["key"])}
        idx = np.array([pos[k] for k in order])
        for src, dst in (("img", I), ("txt", T)):
            dst[t] = torch.tensor(V[t][src][idx]).to(DEV).double()

    within = [(t, t) for t in TAGS]
    cross = list(itertools.permutations(TAGS, 2))
    acc = {k: [] for k in ("nat_w", "nat_c", "swp_w", "swp_c")}

    for s in range(a.seeds):
        perm = np.random.default_rng(s).permutation(n)
        te = torch.tensor(perm[:n_te]).to(DEV)
        tr = torch.tensor(perm[n_te:]).to(DEV)
        Ic = {t: I[t] - I[t][tr].mean(0, keepdim=True) for t in TAGS}
        Tc = {t: T[t] - T[t][tr].mean(0, keepdim=True) for t in TAGS}
        Sc = {t: SW[t] - SW[t][tr].mean(0, keepdim=True) for t in TAGS}
        f = lambda X, Y: infonce(X, Y, tr, te, a.dim, a.epochs, a.lr, s)
        acc["nat_w"].append(np.mean([f(Ic[x], Tc[y]) for x, y in within]))
        acc["nat_c"].append(np.mean([f(Ic[x], Tc[y]) for x, y in cross]))
        acc["swp_w"].append(np.mean([f(Ic[x], Sc[y]) for x, y in within]))
        acc["swp_c"].append(np.mean([f(Ic[x], Sc[y]) for x, y in cross]))
        print(f"  seed {s}: native w {acc['nat_w'][-1]:.4f} c {acc['nat_c'][-1]:.4f}"
              f"   swapped w {acc['swp_w'][-1]:.4f} c {acc['swp_c'][-1]:.4f}",
              flush=True)

    def ms(v):
        return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)

    rw = [w / n_ for w, n_ in zip(acc["swp_w"], acc["nat_w"])]
    rc = [c / n_ for c, n_ in zip(acc["swp_c"], acc["nat_c"])]
    gaps = [w - c for w, c in zip(rw, rc)]
    gm, gs = ms(gaps)

    res = {"question": "does swapping the caption tower move within and cross by "
                       "the same factor",
           "credit": "test proposed by Dipankar Sarkar; spread requirement his too",
           "supersedes": {
               "file": f"head_swap_{a.domain}.json",
               "why": "that version used ONE shared swapped encoder, so the "
                      "swapped side lost its dependence on the text vendor and "
                      "swapped-within and swapped-cross estimated the same "
                      "quantity (0.0835 vs 0.0833). The ratio test collapsed to "
                      "cross_native / within_native. Not underpowered, degenerate."},
           "domain": a.domain, "vendors": TAGS,
           "swapped_head_per_slot": {t: SWAP_FOR[t][0] for t in TAGS},
           "slot_assignment": "fixed and arbitrary",
           "n_holdout": n_te, "seeds": a.seeds,
           "native": {"within": ms(acc["nat_w"]), "cross": ms(acc["nat_c"])},
           "swapped": {"within": ms(acc["swp_w"]), "cross": ms(acc["swp_c"])},
           "within_ratio": ms(rw), "cross_ratio": ms(rc),
           "ratio_gap": gm, "ratio_gap_sd": gs,
           "gap_over_sd": round(gm / gs, 2) if gs > 0 else None}

    # The control on the control: if these two are equal the swap is degenerate
    # again and nothing below means anything.
    dw = res["swapped"]["within"][0] - res["swapped"]["cross"][0]
    res["swapped_arms_differ_by"] = round(dw, 4)
    res["swap_is_nondegenerate"] = bool(abs(dw) > 0.002)

    print(f"\n{'':<20}{'native':>10}{'swapped':>10}{'ratio':>10}{'sd':>8}")
    print(f"{'within':<20}{res['native']['within'][0]:10.4f}"
          f"{res['swapped']['within'][0]:10.4f}{res['within_ratio'][0]:10.4f}"
          f"{res['within_ratio'][1]:8.4f}")
    print(f"{'cross':<20}{res['native']['cross'][0]:10.4f}"
          f"{res['swapped']['cross'][0]:10.4f}{res['cross_ratio'][0]:10.4f}"
          f"{res['cross_ratio'][1]:8.4f}")
    print(f"\nswapped arms differ by {dw:+.4f}  "
          f"({'non-degenerate, the swap keeps the vendor structure' if res['swap_is_nondegenerate'] else 'STILL DEGENERATE, do not read on'})")
    print(f"ratio gap {gm:+.4f} +/- {gs:.4f}"
          + (f"   gap/sd {res['gap_over_sd']}" if res["gap_over_sd"] else ""))

    res["verdict"] = (
        "degenerate, unreadable" if not res["swap_is_nondegenerate"] else
        "head-limited: both terms scale together" if res["gap_over_sd"] is not None
        and abs(res["gap_over_sd"]) < 2 else
        "NOT head-limited: the terms move apart")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"verdict: {res['verdict']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
