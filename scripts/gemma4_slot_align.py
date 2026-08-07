"""Multi-vector (slot) head training: the post-width-null experiment.

Same recipe as gemma4_mlp_align.py (symmetric InfoNCE, drift-residual
augmentation, compositional hard negatives full-pool in the
denominator) with ONE change: the image side keeps S=5 slots per image
(4 spatial bands + global mean, gemma4_encode_slots.py) and every
image-text similarity is the MAX over slots of the per-slot cosine.

Motivation: the 0.705 SugarCrepe wall survived weight sweeps, family
unions, pressure-matching, and a 4x width increase. The slot pilot
showed the information the wall hides is spatial: raw L47 bands
separate swap_obj +6.5 over the global mean. Max-over-slots lets a
caption match the region that contains its subject-object arrangement
while the global slot preserves whole-image retrieval.

Centering: per-slot-index means (mu_img is [S, d]).

    python gemma4_slot_align.py \
        --encodings $SNAP/procrustes/encoded_L47_n5000.pt \
        --train-chunks $SNAP/procrustes/train_pairs/chunk_*.pt \
        --slot-chunks slot_chunks/slot_chunk_*.pt \
        --eval-slots eval_slots.pt \
        --drift-residuals-img drift_res_img_train.npz \
        --drift-residuals-txt drift_res_txt_train.npz \
        --hard-neg-states comp_negs/neg_states_v6.pt --hard-neg-k 4 \
        --save-head v10_slot_head.pt --out mlp_align_v10_slot.json
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
    p.add_argument("--encodings", type=Path, required=True)
    p.add_argument("--train-chunks", type=Path, nargs="+", required=True,
                   help="pair chunks (supplies cap0 text states)")
    p.add_argument("--slot-chunks", type=Path, nargs="+", required=True,
                   help="row-aligned slot chunks from gemma4_encode_slots.py")
    p.add_argument("--eval-slots", type=Path, required=True)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-train", type=int, default=117000)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--drift-residuals-img", type=Path, default=None)
    p.add_argument("--drift-residuals-txt", type=Path, default=None)
    p.add_argument("--drift-prob", type=float, default=0.5)
    p.add_argument("--drift-scale", type=float, default=1.5)
    p.add_argument("--hard-neg-states", type=Path, default=None)
    p.add_argument("--hard-neg-k", type=int, default=1)
    p.add_argument("--hard-neg-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-head", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def slot_sim(zi: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """zi [B,S,p] normalized, zt [C,p] normalized -> [B,C] max over S."""
    return torch.einsum("bsp,cp->bsc", zi, zt).amax(dim=1)


def retrieval_eval_slots(q_slots: torch.Tensor, pool: torch.Tensor,
                         hit_sets, ks=(1, 5, 10)) -> dict:
    """i2t: image slot queries [N,S,p] vs caption pool [C,p]."""
    sims = slot_sim(F.normalize(q_slots, dim=-1), F.normalize(pool, dim=-1))
    return _rank(sims, hit_sets, ks)


def retrieval_eval_t2i(q: torch.Tensor, pool_slots: torch.Tensor,
                       hit_sets, ks=(1, 5, 10)) -> dict:
    """t2i: caption queries [N,p] vs image slot pool [C,S,p]."""
    sims = slot_sim(F.normalize(pool_slots, dim=-1),
                    F.normalize(q, dim=-1)).T          # [N, C]
    return _rank(sims, hit_sets, ks)


def _rank(sims, hit_sets, ks) -> dict:
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
        print(f"hard negatives: {tuple(hard_neg_all.shape)}", flush=True)

    # ---- load training pool: slots (image) + cap0 (text), aligned ----
    slots_l, caps_l = [], []
    for spath, cpath in zip(sorted(args.slot_chunks),
                            sorted(args.train_chunks)):
        sc = torch.load(spath, map_location="cpu", weights_only=True)
        ch = torch.load(cpath, map_location="cpu", weights_only=True)
        assert list(sc["files"]) == list(ch["files"]), \
            f"file mismatch {spath} vs {cpath}"
        slots_l.append(sc["slots"].float())
        caps_l.append(ch["cap0"].float())
    slots_tr = torch.cat(slots_l)                      # [N, S, d]
    cap_tr = torch.cat(caps_l)                         # [N, d]
    N, S, d = slots_tr.shape
    print(f"train pool: slots {tuple(slots_tr.shape)} caps "
          f"{tuple(cap_tr.shape)}", flush=True)

    # ---- eval tensors ----
    enc = torch.load(args.encodings, map_location="cpu", weights_only=True)
    cap0, cap5 = enc["cap0"].float(), enc["cap5"].float()
    ev = torch.load(args.eval_slots, map_location="cpu", weights_only=True)
    eval_slots = ev["slots"].float().to(device)        # [1000, S, d]
    n_eval = args.n_eval
    n_fit = cap0.size(0) - n_eval
    pool5_raw = cap5.to(device)
    t2i_q_raw = cap0[n_fit:].to(device)
    hit_sets = [set(range(5 * i, 5 * i + 5)) for i in range(n_eval)]
    t2i_hits = [{i} for i in range(n_eval)]

    n_train = min(args.n_train, N - args.n_val)
    mu_x = slots_tr[:n_train].mean(0)                  # [S, d] per-slot means
    mu_y = cap_tr[:n_train].mean(0)                    # [d]

    Xtr = (slots_tr[:n_train] - mu_x).to(device)
    Ytr = (cap_tr[:n_train] - mu_y).to(device)
    Xval = (slots_tr[N - args.n_val:] - mu_x).to(device)
    Yval = (cap_tr[N - args.n_val:] - mu_y).to(device)
    val_hits = [{i} for i in range(Xval.size(0))]
    hard_neg = None
    if hard_neg_all is not None:
        assert len(hard_neg_all) >= n_train, "hard-neg alignment broken"
        hard_neg = (hard_neg_all[:n_train] - mu_y).to(device)

    heads = nn.ModuleDict({"img": nn.Linear(d, args.proj_dim, bias=True),
                           "txt": nn.Linear(d, args.proj_dim, bias=True)}
                          ).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    tau = args.temperature
    best_val, best_state, bad = -1.0, None, 0

    for epoch in range(args.max_epochs):
        heads.train()
        perm = torch.randperm(n_train)
        for j in range(0, n_train, args.batch_size):
            idx = perm[j : j + args.batch_size]
            if idx.numel() < 8:
                continue
            xb, yb = Xtr[idx], Ytr[idx]                # [B,S,d], [B,d]
            if drift_img is not None and torch.rand(1).item() < args.drift_prob:
                ri = drift_img[torch.randint(len(drift_img), (idx.numel(),))]
                xb = xb + (ri.to(device) * (torch.rand(idx.numel(), 1,
                           device=device) * args.drift_scale)).unsqueeze(1)
            if drift_txt is not None and torch.rand(1).item() < args.drift_prob:
                rt = drift_txt[torch.randint(len(drift_txt), (idx.numel(),))]
                yb = yb + rt.to(device) * (torch.rand(idx.numel(), 1,
                                           device=device) * args.drift_scale)
            zi = F.normalize(heads["img"](xb), dim=-1)  # [B,S,p]
            zt = F.normalize(heads["txt"](yb), dim=-1)  # [B,p]
            logits = slot_sim(zi, zt) / tau             # [B,B]
            labels = torch.arange(idx.numel(), device=device)
            if hard_neg is not None:
                B = idx.numel()
                nb = hard_neg[idx].reshape(B * hard_neg.size(1), d)
                zn = F.normalize(heads["txt"](nb), dim=-1)
                extra = slot_sim(zi, zn) * args.hard_neg_weight / tau
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
            v = retrieval_eval_slots(heads["img"](Xval), heads["txt"](Yval),
                                     val_hits)
        if v["r@1"] > best_val:
            best_val, bad = v["r@1"], 0
            best_state = {k: p.detach().clone()
                          for k, p in heads.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch} val_r@1={v['r@1']:.3f} "
                  f"best={best_val:.3f}", flush=True)

    heads.load_state_dict(best_state)
    heads.eval()
    with torch.no_grad():
        Xe = eval_slots - mu_x.to(device)
        i2t = retrieval_eval_slots(heads["img"](Xe),
                                   heads["txt"](pool5_raw - mu_y.to(device)),
                                   hit_sets)
        t2i = retrieval_eval_t2i(heads["txt"](t2i_q_raw - mu_y.to(device)),
                                 heads["img"](Xe), t2i_hits)
    print(f"slot_linear val_r@1={best_val:.3f} i2t {i2t} "
          f"t2i r@1={t2i['r@1']:.3f}", flush=True)

    if args.save_head is not None:
        torch.save({
            "img": {k: v.cpu() for k, v in heads["img"].state_dict().items()},
            "txt": {k: v.cpu() for k, v in heads["txt"].state_dict().items()},
            "mu_img": mu_x.cpu(), "mu_txt": mu_y.cpu(),
            "meta": {"kind": "slot_linear", "n_train": n_train, "d": d,
                     "n_slots": S, "pooling": "max_over_slots",
                     "proj_dim": args.proj_dim, "tau": tau,
                     "seed": args.seed, "val_r@1": best_val,
                     "i2t": i2t, "t2i": t2i},
        }, args.save_head)
        print(f"saved head -> {args.save_head}", flush=True)

    args.out.write_text(json.dumps({
        "n_train": n_train, "n_slots": S, "d": d,
        "proj_dim": args.proj_dim, "tau": tau,
        "objective": "symmetric InfoNCE, max-over-slots image similarity",
        "val_r@1": best_val, "i2t": i2t, "t2i": t2i}, indent=2))
    print(f"wrote {args.out}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
