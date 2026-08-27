"""Does the object/arrangement spectrum survive a fitted probe?

The layer sweep we published, and sent to a correspondent, scored SugarCrepe
splits with a head-free centred cosine and concluded that object identity is a
late property while word order peaks mid stack. The fitted depth profile then
showed that a cosine mis-ranks layers outright: it puts L47 first of nine where
a fitted head puts it fourth, behind L20. So that conclusion is owed a re-test
with a probe that can follow a rotation.

Fitting on the SugarCrepe images themselves is impossible, 1,560 is far too few.
Instead the head is fitted per layer on the val2017 images that SugarCrepe does
NOT use, then applied to the SugarCrepe pairs. No image appears on both sides.

    python scripts/sugarcrepe_fitted_depth.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj",
          "replace_rel", "swap_att", "swap_obj"]
CACHE_LAYERS = [4, 12, 20, 28, 36, 44, 47, 54, 60]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sc-cache", default="/root/sugarcrepe/states_4_12_20_28_36_44_47_54_60.npz")
    p.add_argument("--fit-cache", default="/root/depth_states_all9.npz")
    p.add_argument("--caps", default="/root/annotations/captions_val2017.json")
    p.add_argument("--data-dir", default="/root/sugarcrepe")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--out", default="/root/sugarcrepe_fitted_depth.json")
    return p.parse_args()


def val_file_order(caps_path: str, n: int = 5000) -> list[str]:
    """Rebuild the file list the fit cache was written with, which stored no names."""
    ann = json.load(open(caps_path))
    first: dict[int, str] = {}
    for c in sorted(ann["annotations"], key=lambda x: x["id"]):
        first.setdefault(c["image_id"], c["caption"].strip())
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    ids = sorted(i for i in first if i in id2file)[:n]
    return [id2file[i] for i in ids]


def fit(I_tr, T_tr, dim, epochs, lr, tau):
    mu_i, mu_t = I_tr.mean(0, keepdim=True), T_tr.mean(0, keepdim=True)
    Wi = torch.nn.Linear(I_tr.shape[1], dim).cuda()
    Wt = torch.nn.Linear(T_tr.shape[1], dim).cuda()
    opt = torch.optim.Adam(list(Wi.parameters()) + list(Wt.parameters()), lr=lr)
    n = len(I_tr)
    for _ in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 8:
                continue
            a = F.normalize(Wi(I_tr[idx] - mu_i), dim=1)
            b = F.normalize(Wt(T_tr[idx] - mu_t), dim=1)
            lg = a @ b.T / tau
            lab = torch.arange(len(idx), device="cuda")
            loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return Wi, Wt, mu_i, mu_t


def main() -> None:
    a = parse_args()
    sc = np.load(a.sc_cache, allow_pickle=True)
    sc_files = list(sc["img_files"])
    sc_txts = list(sc["txts"])
    fi = {f: i for i, f in enumerate(sc_files)}
    ti = {t: i for i, t in enumerate(sc_txts)}

    fit_z = np.load(a.fit_cache, allow_pickle=True)
    fit_files = val_file_order(a.caps, fit_z["img"].shape[0])
    held = set(sc_files)
    keep = [i for i, f in enumerate(fit_files) if f not in held]
    print(f"fit pool {len(keep)} val2017 images, none of them SugarCrepe's {len(sc_files)}")

    data = {s: json.load(open(f"{a.data_dir}/{s}.json")) for s in SPLITS}
    rows = []
    for s, split in data.items():
        for it in split.values():
            if it["filename"] in fi and it["caption"] in ti:
                rows.append((s, fi[it["filename"]], ti[it["caption"]],
                             ti[it["negative_caption"]]))
    print(f"{len(rows)} triples")

    def cosrow(A, B):
        return (F.normalize(A, dim=1) * F.normalize(B, dim=1)).sum(1)

    results = {}
    for li, L in enumerate(CACHE_LAYERS):
        I_fit = torch.tensor(fit_z["img"][keep, li], dtype=torch.float32, device="cuda")
        T_fit = torch.tensor(fit_z["txt"][keep, li], dtype=torch.float32, device="cuda")
        Wi, Wt, mu_i, mu_t = fit(I_fit, T_fit, a.dim, a.epochs, a.lr, a.tau)

        I = torch.tensor(sc["img"][:, li], dtype=torch.float32, device="cuda")
        T = torch.tensor(sc["txt"][:, li], dtype=torch.float32, device="cuda")
        Ic, Tc = I - I.mean(0, keepdim=True), T - T.mean(0, keepdim=True)
        with torch.no_grad():
            Ip, Tp = Wi(I - mu_i), Wt(T - mu_t)

        raw, fitd = {}, {}
        for s in SPLITS:
            sel = [r for r in rows if r[0] == s]
            gi = [r[1] for r in sel]
            gp = [r[2] for r in sel]
            gn = [r[3] for r in sel]
            raw[s] = float((cosrow(Ic[gi], Tc[gp]) > cosrow(Ic[gi], Tc[gn])).float().mean())
            fitd[s] = float((cosrow(Ip[gi], Tp[gp]) > cosrow(Ip[gi], Tp[gn])).float().mean())
        results[f"L{L}"] = {
            "raw_macro": round(float(np.mean([raw[s] for s in SPLITS])), 4),
            "fitted_macro": round(float(np.mean([fitd[s] for s in SPLITS])), 4),
            "raw": {k: round(v, 4) for k, v in raw.items()},
            "fitted": {k: round(v, 4) for k, v in fitd.items()},
            "raw_identity_minus_order": round(raw["replace_obj"] - raw["swap_obj"], 4),
            "fitted_identity_minus_order": round(fitd["replace_obj"] - fitd["swap_obj"], 4),
        }
        r = results[f"L{L}"]
        print(f"L{L:<3d} raw macro {r['raw_macro']:.3f} id-order {r['raw_identity_minus_order']:+.3f}"
              f"   fitted macro {r['fitted_macro']:.3f} id-order "
              f"{r['fitted_identity_minus_order']:+.3f}", flush=True)

    with open(a.out, "w") as f:
        json.dump({"question": "does the object/arrangement depth spectrum survive a fitted probe",
                   "fit_pool": len(keep), "triples": len(rows),
                   "note": "head fitted on val2017 images SugarCrepe does not use",
                   "results": results}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
