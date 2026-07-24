"""Non-linear cross-modal alignment ceiling on gemma-4 L47 (Sunstone §11.6.4b).

Procrustes (orthogonal) is refuted: no rigid rotation beats per-modality
centering. This probe asks how much a *trained* map can recover beyond
centering, bounding the recoverable shared cross-modal structure. Three
rungs, same eval protocol as scripts/gemma4_procrustes_xmodal.py:

  1. baseline   — centered cosine (published zero-training pipeline)
  2. linear     — unconstrained linear maps, symmetric InfoNCE
  3. mlp        — two-layer GELU MLPs, symmetric InfoNCE

InfoNCE (in-batch negatives, both directions) is the right objective:
it optimizes ranking directly. MSE would repeat Procrustes' failure of
optimizing absolute similarity while destroying discrimination.

Runs entirely from the cached pair encodings produced by the Procrustes
run (encoded_L47_n5000.pt) — no backbone load, minutes on one GPU.

Protocol hygiene: the 1000 eval images are NEVER touched during
training or model selection. Early stopping uses a 500-pair validation
fold carved from the 4000-pair fit split. Controls: shuffled-pairs
training (must collapse to the floor) and a train-size curve.

    python scripts/gemma4_mlp_align.py \
        --encodings /workspace/procrustes/encoded_L47_n5000.pt \
        --out artifacts/nla/gemma4/procrustes/mlp_align.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encodings", type=Path, required=True,
                   help="encoded_L47_n5000.pt from gemma4_procrustes_xmodal.py")
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=500,
                   help="validation pairs carved from the END of the fit split")
    p.add_argument("--train-sizes", type=int, nargs="+", default=[1000, 2000, 3500],
                   help="train-size curve; last = headline")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--proj-dim", type=int, default=1024,
                   help="shared output space dimension")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-3,
                   help="validated on the synthetic smoke: 1e-4 undertrains "
                        "badly, 2e-3 fully recovers a planted non-linear link")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def make_heads(kind: str, d: int, hidden: int, out_dim: int) -> nn.ModuleDict:
    def head() -> nn.Module:
        if kind == "linear":
            return nn.Linear(d, out_dim, bias=True)
        return nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
        )
    return nn.ModuleDict({"img": head(), "txt": head()})


def retrieval_eval(q: torch.Tensor, pool: torch.Tensor,
                   hit_sets: list[set[int]], ks=(1, 5, 10)) -> dict:
    sims = F.normalize(q, dim=-1) @ F.normalize(pool, dim=-1).T
    order = sims.argsort(dim=-1, descending=True)
    out = {f"r@{k}": 0.0 for k in ks}
    ranks = []
    for i, hits in enumerate(hit_sets):
        row = order[i].tolist()
        first = next(r for r, c in enumerate(row) if c in hits)
        ranks.append(first + 1)
        for k in ks:
            if first < k:
                out[f"r@{k}"] += 1.0
    for k in ks:
        out[f"r@{k}"] = round(out[f"r@{k}"] / len(hit_sets), 4)
    out["median_rank"] = float(torch.tensor(ranks, dtype=torch.float).median())
    return out


def train_heads(
    kind: str,
    Xtr: torch.Tensor, Ytr: torch.Tensor,          # centered train pairs (n, d)
    Xval: torch.Tensor, Yval: torch.Tensor,        # centered val pairs
    *, hidden: int, proj_dim: int, tau: float, lr: float,
    batch_size: int, max_epochs: int, patience: int, device: str,
) -> tuple[nn.ModuleDict, dict]:
    d = Xtr.size(1)
    heads = make_heads(kind, d, hidden, proj_dim).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.size(0)
    best_val, best_state, bad, epochs_run = -1.0, None, 0, 0
    val_hits = [{i} for i in range(Xval.size(0))]

    for epoch in range(max_epochs):
        heads.train()
        perm = torch.randperm(n)
        for j in range(0, n, batch_size):
            idx = perm[j : j + batch_size]
            if idx.numel() < 8:
                continue
            zi = F.normalize(heads["img"](Xtr[idx]), dim=-1)
            zt = F.normalize(heads["txt"](Ytr[idx]), dim=-1)
            logits = zi @ zt.T / tau
            labels = torch.arange(idx.numel(), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.T, labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
        heads.eval()
        with torch.no_grad():
            v = retrieval_eval(heads["img"](Xval), heads["txt"](Yval), val_hits)
        epochs_run = epoch + 1
        if v["r@1"] > best_val:
            best_val, bad = v["r@1"], 0
            best_state = {k: p.detach().clone() for k, p in heads.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        heads.load_state_dict(best_state)
    heads.eval()
    return heads, {"val_r@1": best_val, "epochs": epochs_run}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    enc = torch.load(args.encodings, map_location="cpu", weights_only=True)
    img, cap0, cap5 = enc["img"].float(), enc["cap0"].float(), enc["cap5"].float()
    N, d = img.shape
    n_eval = args.n_eval
    n_fit = N - n_eval
    n_train_max = n_fit - args.n_val

    # Same split geometry as the Procrustes run. Eval tail is untouchable.
    mu_x, mu_y = img[:n_fit].mean(0), cap0[:n_fit].mean(0)
    Xf, Yf = (img[:n_fit] - mu_x).to(device), (cap0[:n_fit] - mu_y).to(device)
    Xval, Yval = Xf[n_train_max:], Yf[n_train_max:]
    X_eval = (img[n_fit:] - mu_x).to(device)
    pool5 = (cap5 - mu_y).to(device)
    t2i_q = (cap0[n_fit:] - mu_y).to(device)
    hit_sets = [set(range(5 * i, 5 * i + 5)) for i in range(n_eval)]
    t2i_hits = [{i} for i in range(n_eval)]

    results: dict = {
        "encodings": str(args.encodings), "d": d, "n_fit": n_fit,
        "n_val": args.n_val, "n_eval": n_eval, "seed": args.seed,
        "objective": "symmetric InfoNCE, in-batch negatives",
        "tau": args.temperature, "proj_dim": args.proj_dim,
        "hidden": args.hidden, "runs": {},
    }

    # Rung 1: centered-cosine baseline (identity heads).
    results["runs"]["baseline_centered"] = {
        "i2t": retrieval_eval(X_eval, pool5, hit_sets),
        "t2i": retrieval_eval(t2i_q, X_eval, t2i_hits),
    }
    print("baseline_centered", results["runs"]["baseline_centered"]["i2t"], flush=True)

    def run(name: str, kind: str, n_train: int, shuffle: bool = False) -> None:
        g = torch.Generator().manual_seed(args.seed)
        Xtr, Ytr = Xf[:n_train].clone(), Yf[:n_train].clone()
        if shuffle:
            Ytr = Ytr[torch.randperm(n_train, generator=g)]
        heads, fit_info = train_heads(
            kind, Xtr, Ytr, Xval, Yval,
            hidden=args.hidden, proj_dim=args.proj_dim, tau=args.temperature,
            lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, device=device,
        )
        with torch.no_grad():
            i2t = retrieval_eval(heads["img"](X_eval), heads["txt"](pool5), hit_sets)
            t2i = retrieval_eval(heads["txt"](t2i_q), heads["img"](X_eval), t2i_hits)
        rec = {"kind": kind, "n_train": n_train, "shuffle": shuffle,
               **fit_info, "i2t": i2t, "t2i": t2i}
        results["runs"][name] = rec
        print(f"{name:24s} val_r@1={fit_info['val_r@1']:.3f} "
              f"i2t {i2t}  t2i r@1={t2i['r@1']:.3f}", flush=True)

    headline_n = min(args.train_sizes[-1], n_train_max)
    # Rung 2: unconstrained linear (is the gap linear-but-not-orthogonal?)
    run(f"linear_n{headline_n}", "linear", headline_n)
    # Rung 3: MLP train-size curve + shuffled control
    for n_train in args.train_sizes:
        run(f"mlp_n{min(n_train, n_train_max)}", "mlp", min(n_train, n_train_max))
    run(f"mlp_shuffled_n{headline_n}", "mlp", headline_n, shuffle=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
