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
                   help="encoded_L47_n5000.pt from gemma4_procrustes_xmodal.py "
                        "(supplies the untouched val2017 eval split, and the "
                        "train pool when --train-encodings is absent)")
    p.add_argument("--train-encodings", type=Path, nargs="*", default=None,
                   help="optional chunk .pt files (img/cap0) from "
                        "gemma4_encode_pairs.py (e.g. COCO train2017). When "
                        "given, ALL training/val pairs come from these and "
                        "--encodings is used only for the eval split.")
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
    p.add_argument("--mean-jitter", type=float, default=0.0,
                   help="mean-invariance augmentation: each batch adds one "
                        "random constant vector per modality with norm "
                        "~U(0, R) to the centered inputs, simulating a "
                        "cross-runtime mean displacement. Motivation: the "
                        "MLX-Q4 runtime displaces both modality means by "
                        "~22 units and the img branch is ~17x more "
                        "mean-sensitive than txt (24 vs 1.5 i2t/t2i points "
                        "at matched norm; artifacts/nla/q4/"
                        "mean_displacement_20260805.json). R=30 covers the "
                        "observed displacement with margin.")
    p.add_argument("--drift-residuals-img", type=Path, default=None,
                   help="npz with 'residuals' [N,d]: measured per-vector "
                        "cross-runtime drift (mlx - cuda) for image states. "
                        "v3 recipe (round 5): train against the measured "
                        "drift family, not isotropic noise. Isotropic "
                        "jitter armored random directions 3.0x but the "
                        "real direction only 1.47x "
                        "(artifacts/nla/q4/round5_v2_direction_20260806.json).")
    p.add_argument("--drift-residuals-txt", type=Path, default=None,
                   help="same, for text states")
    p.add_argument("--drift-prob", type=float, default=0.5,
                   help="per-batch probability of applying drift-residual "
                        "augmentation (per-sample residual draws, scaled "
                        "~U(0, drift-scale))")
    p.add_argument("--drift-scale", type=float, default=1.5,
                   help="max residual scale multiplier")
    p.add_argument("--hard-neg-states", type=Path, default=None,
                   help=".pt with 'states' [N*K,d]: L47 states of "
                        "compositional hard-negative captions, ROW-ALIGNED "
                        "with the training pairs (K per row, K-major). v4/v5 "
                        "recipe (SugarCrepe diagnostic): perturbed captions "
                        "join the InfoNCE denominator so the head cannot "
                        "project the word-order axis away.")
    p.add_argument("--hard-neg-k", type=int, default=1,
                   help="negatives per training row in --hard-neg-states")
    p.add_argument("--hard-neg-weight", type=float, default=1.0,
                   help="multiplier on the hard-negative logits: the "
                        "compositionality/clean-retrieval trade dial "
                        "(>1 pushes harder, <1 gentler)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-head", type=Path, default=None,
                   help="persist the trained heads + per-run centering means "
                        "for the HEADLINE LINEAR run to this .pt (ship-able "
                        "artifact: {'img','txt' state_dicts, 'mu_img','mu_txt', "
                        "'meta'})")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def make_heads(kind: str, d_img: int, d_txt: int, hidden: int,
               out_dim: int) -> nn.ModuleDict:
    def head(d: int) -> nn.Module:
        if kind == "linear":
            return nn.Linear(d, out_dim, bias=True)
        return nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
        )
    return nn.ModuleDict({"img": head(d_img), "txt": head(d_txt)})


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
    mean_jitter: float = 0.0,
    drift_img: torch.Tensor | None = None,        # [N,d] measured residuals
    drift_txt: torch.Tensor | None = None,
    drift_prob: float = 0.5, drift_scale: float = 1.5,
    hard_neg: torch.Tensor | None = None,         # [n, K, d] row-aligned
    hard_neg_weight: float = 1.0,
) -> tuple[nn.ModuleDict, dict]:
    d_img, d_txt = Xtr.size(1), Ytr.size(1)
    heads = make_heads(kind, d_img, d_txt, hidden, proj_dim).to(device)
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
            xb, yb = Xtr[idx], Ytr[idx]
            if drift_img is not None and torch.rand(1).item() < drift_prob:
                # measured-drift-family augmentation (v3): per-sample
                # residual draws simulate "this input arrived from the
                # other runtime" (round 5: isotropic jitter armors random
                # directions 3.0x but the real one only 1.47x)
                ri = drift_img[torch.randint(len(drift_img), (idx.numel(),))]
                xb = xb + ri.to(xb.device) * (torch.rand(idx.numel(), 1,
                                              device=xb.device) * drift_scale)
            if drift_txt is not None and torch.rand(1).item() < drift_prob:
                rt = drift_txt[torch.randint(len(drift_txt), (idx.numel(),))]
                yb = yb + rt.to(yb.device) * (torch.rand(idx.numel(), 1,
                                              device=yb.device) * drift_scale)
            if mean_jitter > 0:
                # one constant displacement per modality per batch: the
                # augmentation a cross-runtime mean shift actually applies
                ri = torch.randn(d_img, device=xb.device)
                rt = torch.randn(d_txt, device=yb.device)
                ri = ri / ri.norm() * (torch.rand(1, device=xb.device) * mean_jitter)
                rt = rt / rt.norm() * (torch.rand(1, device=yb.device) * mean_jitter)
                xb = xb + ri
                yb = yb + rt
            zi = F.normalize(heads["img"](xb), dim=-1)
            zt = F.normalize(heads["txt"](yb), dim=-1)
            logits = zi @ zt.T / tau
            labels = torch.arange(idx.numel(), device=device)
            if hard_neg is not None:
                # v5: K compositional negatives per sample, full-pool
                # inclusion — every image in the batch sees every negative
                # caption as a candidate, so same-nouns-different-syntax
                # becomes a training error batch-wide.
                B = idx.numel()
                nb = hard_neg[idx].reshape(
                    B * hard_neg.size(1), hard_neg.size(-1)).to(xb.device)
                zn = F.normalize(heads["txt"](nb), dim=-1)
                extra = (zi @ zn.T) * hard_neg_weight / tau     # [B, B*K]
                loss = 0.5 * (F.cross_entropy(
                                  torch.cat([logits, extra], dim=1), labels)
                              + F.cross_entropy(logits.T, labels))
            else:
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

    import numpy as np
    drift_img = drift_txt = None
    if args.drift_residuals_img is not None:
        drift_img = torch.from_numpy(
            np.load(args.drift_residuals_img)["residuals"]).float()
        print(f"drift residuals img: {tuple(drift_img.shape)}", flush=True)
    if args.drift_residuals_txt is not None:
        drift_txt = torch.from_numpy(
            np.load(args.drift_residuals_txt)["residuals"]).float()
        print(f"drift residuals txt: {tuple(drift_txt.shape)}", flush=True)

    hard_neg_all = None
    if args.hard_neg_states is not None:
        hard_neg_all = torch.load(args.hard_neg_states, map_location="cpu",
                                  weights_only=True)["states"].float()
        hard_neg_all = hard_neg_all.view(-1, args.hard_neg_k,
                                         hard_neg_all.size(-1))
        print(f"hard negatives: {tuple(hard_neg_all.shape)} "
              f"(K={args.hard_neg_k}, weight={args.hard_neg_weight})",
              flush=True)

    enc = torch.load(args.encodings, map_location="cpu", weights_only=True)
    img, cap0, cap5 = enc["img"].float(), enc["cap0"].float(), enc["cap5"].float()
    N, d_img = img.shape
    d_txt = cap0.size(1)
    n_eval = args.n_eval
    n_fit = N - n_eval

    # Train/val pool: external chunks (train2017) or the val2017 fit split.
    if args.train_encodings:
        imgs, caps = [], []
        for pth in args.train_encodings:
            ch = torch.load(pth, map_location="cpu", weights_only=True)
            imgs.append(ch["img"].float())
            caps.append(ch["cap0"].float())
        img_tr, cap_tr = torch.cat(imgs), torch.cat(caps)
        train_src = [str(p) for p in args.train_encodings]
    else:
        img_tr, cap_tr = img[:n_fit], cap0[:n_fit]
        train_src = [str(args.encodings) + "[:n_fit]"]
    n_pool = img_tr.size(0)
    n_train_max = n_pool - args.n_val

    # Raw (uncentered) eval tensors; each run centers by ITS training means.
    img_tr, cap_tr = img_tr.to(device), cap_tr.to(device)
    X_eval_raw = img[n_fit:].to(device)
    pool5_raw = cap5.to(device)
    t2i_q_raw = cap0[n_fit:].to(device)
    hit_sets = [set(range(5 * i, 5 * i + 5)) for i in range(n_eval)]
    t2i_hits = [{i} for i in range(n_eval)]

    results: dict = {
        "encodings": str(args.encodings), "train_encodings": train_src,
        "d_img": d_img, "d_txt": d_txt, "n_train_pool": n_pool,
        "n_val": args.n_val, "n_eval": n_eval, "seed": args.seed,
        "objective": "symmetric InfoNCE, in-batch negatives",
        "tau": args.temperature, "proj_dim": args.proj_dim,
        "hidden": args.hidden, "runs": {},
    }

    # Rung 1: centered-cosine baseline, published convention (val2017 fit means).
    # Undefined for mixed-tower runs (d_img != d_txt): no shared space
    # exists without a trained map, which is the point of the control.
    if d_img == d_txt:
        mu_x0, mu_y0 = img[:n_fit].mean(0).to(device), cap0[:n_fit].mean(0).to(device)
        results["runs"]["baseline_centered"] = {
            "i2t": retrieval_eval(X_eval_raw - mu_x0, pool5_raw - mu_y0, hit_sets),
            "t2i": retrieval_eval(t2i_q_raw - mu_y0, X_eval_raw - mu_x0, t2i_hits),
        }
        print("baseline_centered", results["runs"]["baseline_centered"]["i2t"], flush=True)
    else:
        results["runs"]["baseline_centered"] = None
        print("baseline_centered skipped (d_img != d_txt)", flush=True)

    def run(name: str, kind: str, n_train: int, shuffle: bool = False,
            save_to: Path | None = None) -> None:
        g = torch.Generator().manual_seed(args.seed)
        # Center everything by THIS run's training-pool means.
        mu_x = img_tr[:n_train].mean(0)
        mu_y = cap_tr[:n_train].mean(0)
        Xtr = img_tr[:n_train] - mu_x
        Ytr = cap_tr[:n_train] - mu_y
        Xval = img_tr[n_train_max:] - mu_x
        Yval = cap_tr[n_train_max:] - mu_y
        hard_neg = None
        if hard_neg_all is not None:
            if len(hard_neg_all) < n_train:
                raise SystemExit(f"hard-neg rows {len(hard_neg_all)} < "
                                 f"n_train {n_train}; alignment broken")
            hard_neg = (hard_neg_all[:n_train] - mu_y.cpu()).to(device)
        if shuffle:
            Ytr = Ytr[torch.randperm(n_train, generator=g)]
        heads, fit_info = train_heads(
            kind, Xtr, Ytr, Xval, Yval,
            hidden=args.hidden, proj_dim=args.proj_dim, tau=args.temperature,
            lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, device=device,
            mean_jitter=args.mean_jitter,
            drift_img=drift_img, drift_txt=drift_txt,
            drift_prob=args.drift_prob, drift_scale=args.drift_scale,
            hard_neg=hard_neg, hard_neg_weight=args.hard_neg_weight,
        )
        with torch.no_grad():
            i2t = retrieval_eval(heads["img"](X_eval_raw - mu_x),
                                 heads["txt"](pool5_raw - mu_y), hit_sets)
            t2i = retrieval_eval(heads["txt"](t2i_q_raw - mu_y),
                                 heads["img"](X_eval_raw - mu_x), t2i_hits)
        rec = {"kind": kind, "n_train": n_train, "shuffle": shuffle,
               **fit_info, "i2t": i2t, "t2i": t2i}
        results["runs"][name] = rec
        print(f"{name:24s} val_r@1={fit_info['val_r@1']:.3f} "
              f"i2t {i2t}  t2i r@1={t2i['r@1']:.3f}", flush=True)
        if save_to is not None:
            torch.save({
                "img": {k: v.cpu() for k, v in heads["img"].state_dict().items()},
                "txt": {k: v.cpu() for k, v in heads["txt"].state_dict().items()},
                "mu_img": mu_x.cpu(), "mu_txt": mu_y.cpu(),
                "meta": {"kind": kind, "n_train": n_train,
                         "d_img": d_img, "d_txt": d_txt,
                         "proj_dim": args.proj_dim, "hidden": args.hidden,
                         "tau": args.temperature, "seed": args.seed,
                         "i2t": i2t, "t2i": t2i, **fit_info},
            }, save_to)
            print(f"saved head -> {save_to}", flush=True)

    headline_n = min(args.train_sizes[-1], n_train_max)
    # Both kinds along the train-size curve + shuffled controls at headline.
    for n_train in args.train_sizes:
        n_t = min(n_train, n_train_max)
        run(f"linear_n{n_t}", "linear", n_t,
            save_to=args.save_head if n_t == headline_n else None)
        run(f"mlp_n{n_t}", "mlp", n_t)
    run(f"mlp_shuffled_n{headline_n}", "mlp", headline_n, shuffle=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
