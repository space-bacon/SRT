"""Quantization-drift eval for the cross-modal linear head (Pi-tier gate #2).

The Qwen2.5-VL-3B head was trained on bf16 states. Edge deployment runs
the backbone 4-bit quantized, which perturbs the states. This script
measures whether the head survives, at three levels of repair:

  1. bf16_head_asis      — bf16-trained head + bf16 means applied to Q4
                           states unchanged (pure drift measurement)
  2. bf16_head_mu_recal  — same head weights, but means recomputed from
                           Q4 states (cheapest possible recalibration)
  3. (full retrain is run separately via gemma4_mlp_align.py on Q4
      train chunks; this script reports 1, 2, and the Q4 zero-training
      baseline for context)

Caveat recorded in the output: bitsandbytes NF4 on GPU is a proxy for
llama.cpp Q4 on CPU. Same weight precision class, different kernels;
final Pi validation should re-run on-device.

    python scripts/q4_drift_eval.py \
        --bf16-head /workspace/scalefloor/qwen3b_linear_head.pt \
        --q4-encodings /workspace/q4/encoded_L29_n5000.pt \
        --out /workspace/q4/q4_drift.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "gemma4_mlp_align", _HERE / "gemma4_mlp_align.py")
_MA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_MA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16-head", type=Path, required=True,
                   help="head .pt saved by gemma4_mlp_align --save-head "
                        "(trained on bf16 states)")
    p.add_argument("--q4-encodings", type=Path, required=True,
                   help="encoded_L*.pt produced with --quant4")
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.bf16_head, map_location="cpu", weights_only=True)
    meta = ck["meta"]
    d, proj = meta["d"], meta["proj_dim"]
    head_i, head_t = nn.Linear(d, proj), nn.Linear(d, proj)
    head_i.load_state_dict(ck["img"])
    head_t.load_state_dict(ck["txt"])
    head_i, head_t = head_i.to(device).eval(), head_t.to(device).eval()
    mu_x_bf16 = ck["mu_img"].to(device)
    mu_y_bf16 = ck["mu_txt"].to(device)

    enc = torch.load(args.q4_encodings, map_location="cpu", weights_only=True)
    img, cap0, cap5 = enc["img"].float(), enc["cap0"].float(), enc["cap5"].float()
    n_eval = args.n_eval
    n_fit = img.size(0) - n_eval
    X_eval = img[n_fit:].to(device)
    pool5 = cap5.to(device)
    t2i_q = cap0[n_fit:].to(device)
    hit_sets = [set(range(5 * i, 5 * i + 5)) for i in range(n_eval)]
    t2i_hits = [{i} for i in range(n_eval)]

    # Q4-derived means (fit split of the Q4 encodings).
    mu_x_q4 = img[:n_fit].mean(0).to(device)
    mu_y_q4 = cap0[:n_fit].mean(0).to(device)

    results: dict = {
        "bf16_head": str(args.bf16_head),
        "bf16_reference_i2t": meta.get("i2t"),
        "q4_encodings": str(args.q4_encodings),
        "caveat": "bnb NF4 on GPU as proxy for llama.cpp Q4 on CPU; "
                  "final validation on-device",
        "state_drift": {},
        "runs": {},
    }

    # Descriptive drift: how far did the states move? (needs matched rows —
    # only meaningful if a bf16 encoding of the same rows exists; report the
    # mean-shift magnitude instead, which is always available.)
    results["state_drift"]["mu_img_shift_cos"] = float(
        torch.nn.functional.cosine_similarity(mu_x_bf16, mu_x_q4, dim=0))
    results["state_drift"]["mu_txt_shift_cos"] = float(
        torch.nn.functional.cosine_similarity(mu_y_bf16, mu_y_q4, dim=0))

    with torch.no_grad():
        def run(name, mu_x, mu_y, use_head):
            if use_head:
                q = head_i(X_eval - mu_x)
                pool = head_t(pool5 - mu_y)
                tq = head_t(t2i_q - mu_y)
                tpool = head_i(X_eval - mu_x)
            else:
                q, pool = X_eval - mu_x, pool5 - mu_y
                tq, tpool = t2i_q - mu_y, X_eval - mu_x
            rec = {"i2t": _MA.retrieval_eval(q, pool, hit_sets),
                   "t2i": _MA.retrieval_eval(tq, tpool, t2i_hits)}
            results["runs"][name] = rec
            print(f"{name:22s} i2t {rec['i2t']}  t2i r@1={rec['t2i']['r@1']:.3f}",
                  flush=True)

        run("baseline_q4_centered", mu_x_q4, mu_y_q4, use_head=False)
        run("bf16_head_asis", mu_x_bf16, mu_y_bf16, use_head=True)
        run("bf16_head_mu_recal", mu_x_q4, mu_y_q4, use_head=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
