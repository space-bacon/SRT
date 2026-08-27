"""Is the L60 collapse the forward pass discarding, or the readout losing reach?

A correspondent read our depth sweep as the forward pass throwing away what it
does not need for the next token, with the projection record surviving only in
the layers before the output. The sweep cannot settle that, because it scored
raw states with a centred cosine. A cosine finds structure only in the basis it
is handed, so "the information is gone" and "the information moved somewhere a
cosine cannot follow" produce the same reading.

This fits a linear readout at each depth instead. Train it on one set of images
and score held-out ones, same splits and same metric as the head-free sweep, so
the only thing that changes is whether the readout was fitted. A fitted map that
recovers at L60 what cosine missed means the record is still there.

Reuses the states the layer sweep already cached, so no re-encoding.

    .venv/bin/python scripts/l60_fitted_readout.py
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATES = Path("/tmp/states_4_12_20_28_36_44_47_54_60.npz")
WORK = Path("/tmp/sugarcrepe_data")
OUT = ROOT / "artifacts/nla/q4/l60_fitted_readout.json"
SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj",
          "replace_rel", "swap_att", "swap_obj"]
DATA_URL = ("https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/"
            "main/data/{split}.json")
LAYERS = [4, 12, 20, 28, 36, 44, 47, 54, 60]
K = 128          # PCA dims per modality, fitted on train images only
RIDGE = 1.0
SEED = 0


def fetch() -> dict[str, dict]:
    WORK.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    data = {}
    for s in SPLITS:
        p = WORK / f"{s}.json"
        if not p.exists():
            with urllib.request.urlopen(DATA_URL.format(split=s), context=ctx) as r:
                p.write_bytes(r.read())
        data[s] = json.loads(p.read_text())
    return data


def cosrow(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A * B).sum(1)


def main() -> None:
    z = np.load(STATES, allow_pickle=False)
    files, txts = list(z["img_files"]), list(z["txts"])
    fi = {f: i for i, f in enumerate(files)}
    ti = {t: i for i, t in enumerate(txts)}
    data = fetch()

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(files))
    n_tr = int(0.6 * len(files))
    train_img = set(order[:n_tr].tolist())

    # Triples are assigned by their image, so no image appears on both sides.
    rows = []
    for sname, split in data.items():
        for it in split.values():
            if it["filename"] in fi and it["caption"] in ti:
                rows.append((sname, fi[it["filename"]],
                             ti[it["caption"]], ti[it["negative_caption"]]))
    tr = [r for r in rows if r[1] in train_img]
    te = [r for r in rows if r[1] not in train_img]
    print(f"{len(rows)} triples, {len(tr)} train / {len(te)} test, "
          f"{len(train_img)} train images of {len(files)}")

    results = {}
    for li, L in enumerate(LAYERS):
        I = z["img"][:, li].astype(np.float32)
        T = z["txt"][:, li].astype(np.float32)
        mi, mt = I.mean(0), T.mean(0)
        Ic, Tc = I - mi, T - mt

        # Head-free arm, the sweep's own metric restricted to the test images.
        raw = {}
        for sname in SPLITS:
            sel = [r for r in te if r[0] == sname]
            if not sel:
                continue
            vi = Ic[[r[1] for r in sel]]
            hit = (cosrow(vi, Tc[[r[2] for r in sel]])
                   > cosrow(vi, Tc[[r[3] for r in sel]])).mean()
            raw[sname] = float(hit)

        # Fitted arm. PCA bases come from train rows only, then a ridge map from
        # image space to caption space learned on train positives.
        tr_i = sorted({r[1] for r in tr})
        tr_t = sorted({r[2] for r in tr})
        Ui = np.linalg.svd(Ic[tr_i], full_matrices=False)[2][:K].T
        Ut = np.linalg.svd(Tc[tr_t], full_matrices=False)[2][:K].T
        A = Ic[[r[1] for r in tr]] @ Ui
        B = Tc[[r[2] for r in tr]] @ Ut
        W = np.linalg.solve(A.T @ A + RIDGE * np.eye(K), A.T @ B)

        fit = {}
        for sname in SPLITS:
            sel = [r for r in te if r[0] == sname]
            if not sel:
                continue
            vi = (Ic[[r[1] for r in sel]] @ Ui) @ W
            hit = (cosrow(vi, Tc[[r[2] for r in sel]] @ Ut)
                   > cosrow(vi, Tc[[r[3] for r in sel]] @ Ut)).mean()
            fit[sname] = float(hit)

        results[f"L{L}"] = {
            "raw_macro": round(float(np.mean([raw[s] for s in SPLITS])), 4),
            "fitted_macro": round(float(np.mean([fit[s] for s in SPLITS])), 4),
            "raw": {k: round(v, 4) for k, v in raw.items()},
            "fitted": {k: round(v, 4) for k, v in fit.items()},
        }
        print(f"L{L:<3d} head-free {results[f'L{L}']['raw_macro']:.4f}   "
              f"fitted {results[f'L{L}']['fitted_macro']:.4f}", flush=True)

    OUT.write_text(json.dumps({
        "question": "is the L60 collapse lost information or a readout that lost reach",
        "method": f"PCA-{K} per modality on train images, ridge map image->caption",
        "test_note": "held-out images, same splits and metric as the head-free sweep",
        "results": results,
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
