#!/usr/bin/env python3
"""Test 2 (cross-backbone invariance): metapragmatic load on gemma-4-31B-it.

The Metapragmatic-load conjecture (paper_nla.md §12.5) claims E[D] = D0 +
alpha*U_ref + beta*U_com is a property of the *interpretant geometry* of a
frozen decoder, not an artefact of any one backbone. Test 1 established the
law on Qwen2.5-7B (srt-adapter-v1.0 and v8a). Test 2 asks whether the SAME
2x2 dissociation reappears on a different backbone family read out by an
independently trained side-channel.

We use the SRT-Sunstone read-out for a frozen google/gemma-4-31B-it
(RiverRider/Gemma-4-31B-it-SRT-Sunstone, readout_selected.pt). That read-out
carries a community head (layer 8) and THREE Metapragmatic Attention Heads
(layers 15/30/45) with divergence projections — exactly the load channel the
dissociation isolates. It has no BEN/RRM, so this test is purely about MAH
divergence (load), which is the axis the dissociation found robust; the r_hat
(commitment) channel was already null on Qwen and is not part of the claim.

For each concept in the balanced 2x2 battery
(data/probes/contestedness_dissociation_v1.jsonl: 10 concepts per cell,
concrete{0,1} x contested{0,1}) we run ONE forward pass on the uniform stem
"{Concept}, properly understood, is", read the community vector at layer 8,
condition the three MAH heads on it, and record the divergence L2 norm at the
final shared token (`div_last`) per layer. Primary `div_last` is the last MAH
layer (L45), matching the Qwen convention (o.divergences[-1]).

The analysis reproduces the corrected dissociation contrasts and, because raw
divergence magnitudes are not comparable across backbones, reports them
z-scored *within* each battery so the standardized effects line up against
Qwen. If the abstractness main effect and the contested>consensus interaction
within the abstract cell survive on gemma-4, the conjecture graduates from a
single-backbone regularity toward a backbone-invariant law.

Usage (box, transformers>=5 for Gemma4ForConditionalGeneration):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/mah_crossbackbone_gemma4.py \
        --out-readouts artifacts/nla/coupling/dissociation_gemma4_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/dissociation_gemma4_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


# ------------------------------------------------------------------------- stats

def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, z) -> float:
    """Rank partial correlation of x,y controlling for z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        beta = np.linalg.lstsq(b1, a, rcond=None)[0]
        return a - b1 @ beta

    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def perm_p_meandiff(y: np.ndarray, group: np.ndarray, nperm: int, rng) -> tuple[float, float]:
    """Two-sided permutation p for the difference in means y[group==1]-y[group==0]."""
    y = np.asarray(y, float)
    g = np.asarray(group).astype(bool)
    obs = y[g].mean() - y[~g].mean()
    c = 0
    for _ in range(nperm):
        gp = rng.permutation(g)
        if abs(y[gp].mean() - y[~gp].mean()) >= abs(obs) - 1e-12:
            c += 1
    return float(obs), (c + 1) / (nperm + 1)


# --------------------------------------------------------------------- analysis

def analyze(rows: list[dict], metric: str, nperm: int, rng) -> dict:
    """2x2 dissociation contrasts on `metric`, raw and z-scored within battery."""
    y = np.array([r[metric] for r in rows], float)
    conc = np.array([r["concrete"] for r in rows])
    cont = np.array([r["contested"] for r in rows])
    z = (y - y.mean()) / (y.std() + 1e-12)

    def cell(c, k, arr):
        return arr[(conc == c) & (cont == k)]

    cells = {f"concrete{c}_contested{k}": {
        "n": int(((conc == c) & (cont == k)).sum()),
        "mean": float(cell(c, k, y).mean()),
        "z_mean": float(cell(c, k, z).mean()),
    } for c in (1, 0) for k in (0, 1)}

    # main effect of abstractness (abstract = concrete 0)
    abs_delta, abs_p = perm_p_meandiff(y, (conc == 0).astype(int), nperm, rng)
    abs_delta_z = float(z[conc == 0].mean() - z[conc == 1].mean())
    abs_d = cohen_d(y[conc == 0], y[conc == 1])

    # contestedness interaction, within each concreteness level
    ma = conc == 0
    mc = conc == 1
    ca_delta, ca_p = perm_p_meandiff(y[ma], cont[ma], nperm, rng)
    ca_delta_z = float(z[ma & (cont == 1)].mean() - z[ma & (cont == 0)].mean())
    ca_d = cohen_d(y[ma & (cont == 1)], y[ma & (cont == 0)])
    cc_delta, cc_p = perm_p_meandiff(y[mc], cont[mc], nperm, rng)
    cc_delta_z = float(z[mc & (cont == 1)].mean() - z[mc & (cont == 0)].mean())
    cc_d = cohen_d(y[mc & (cont == 1)], y[mc & (cont == 0)])

    # standardized additive fit: z(metric) ~ U_ref + U_com, U_ref=abstract, U_com=contested
    Uref = (1 - conc).astype(float)
    Ucom = cont.astype(float)
    X = np.c_[np.ones(len(y)), (Uref - Uref.mean()) / Uref.std(),
              (Ucom - Ucom.mean()) / Ucom.std()]
    beta = np.linalg.lstsq(X, z, rcond=None)[0]

    # length control (contested items tend shorter; averaged/last divergence can inflate)
    ntok = np.array([r["n_tokens"] for r in rows], float)

    return {
        "metric": metric,
        "cells": cells,
        "main_abstractness": {
            "delta_raw": abs_delta, "delta_z": abs_delta_z,
            "cohen_d": abs_d, "perm_p": abs_p,
        },
        "contest_within_abstract": {
            "delta_raw": ca_delta, "delta_z": ca_delta_z,
            "cohen_d": ca_d, "perm_p": ca_p,
        },
        "contest_within_concrete": {
            "delta_raw": cc_delta, "delta_z": cc_delta_z,
            "cohen_d": cc_d, "perm_p": cc_p,
        },
        "std_beta": {"intercept": float(beta[0]), "U_ref": float(beta[1]),
                     "U_com": float(beta[2])},
        "contest_vs_ntokens_spearman": spearman(cont, ntok),
        "contest_partial_given_ntokens": partial_spearman(cont, y, ntok),
    }


# -------------------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/Gemma-4-31B-it-SRT-Sunstone")
    ap.add_argument("--ckpt-file", default="readout_selected.pt")
    ap.add_argument("--probe", default="data/probes/contestedness_dissociation_v1.jsonl")
    ap.add_argument("--out-readouts",
                    default="artifacts/nla/coupling/dissociation_gemma4_readouts.jsonl")
    ap.add_argument("--out-summary",
                    default="artifacts/nla/coupling/dissociation_gemma4_summary.json")
    ap.add_argument("--qwen-readouts",
                    default="artifacts/nla/coupling/dissociation_v1_readouts.jsonl",
                    help="Qwen v1.0 dissociation readouts for a side-by-side comparison")
    ap.add_argument("--backbone-path", default=None,
                    help="Local dir to load the backbone from (bypasses the hub); "
                         "defaults to the backbone id in the readout config")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    from srt.config import CommunityConfig, MAHConfig
    from srt.modules.community import CommunityDiscoveryHead
    from srt.modules.mah import MetapragmaticAttentionHead

    dev = args.device

    # ---- load read-out heads -------------------------------------------------
    ckpt = torch.load(hf_hub_download(args.repo, args.ckpt_file),
                      map_location="cpu", weights_only=False)
    cc = ckpt["config"]
    comm_L = int(cc["community_layer"])
    mah_layers = [int(x) for x in cc["mah_layers"]]
    d_backbone = int(cc["d_backbone"])
    d_community = int(cc["d_community"])
    mid = cc["backbone"]
    load_id = args.backbone_path or mid
    heads = ckpt["heads"]
    print(f"[ckpt] step {ckpt.get('step')} backbone={mid} load_from={load_id} "
          f"community@{comm_L} mah@{mah_layers} d={d_backbone}", flush=True)

    # community head lives under the "0." prefix; MAH heads under "1.<i>."
    community = CommunityDiscoveryHead(CommunityConfig(d_community=d_community),
                                       d_backbone).to(dev).eval()
    community.load_state_dict(
        {k[len("0."):]: v for k, v in heads.items() if k.startswith("0.")})
    community.float()

    mah_cfg = MAHConfig()  # d_sub=512, d_divergence=256, num_heads=4 — matches shapes
    mah_heads = []
    for i in range(len(mah_layers)):
        h = MetapragmaticAttentionHead(mah_cfg, d_backbone, d_community=d_community)
        sd = {k[len(f"1.{i}."):]: v for k, v in heads.items() if k.startswith(f"1.{i}.")}
        miss, unexp = h.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f"MAH {i} load mismatch miss={miss} unexp={unexp}"
        mah_heads.append(h.to(dev).eval().float())
    print(f"[heads] community + {len(mah_heads)} MAH heads loaded", flush=True)

    # ---- backbone ------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(load_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Gemma4ForConditionalGeneration.from_pretrained(
        load_id, dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    last_layer = mah_layers[-1]

    # ---- readouts ------------------------------------------------------------
    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    print(f"[probe] {len(items)} concepts (2x2 dissociation)", flush=True)

    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with out_r.open("w") as f:
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len).to(dev)
            hs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                       output_hidden_states=True, use_cache=False).hidden_states
            assert len(hs) > last_layer, (
                f"backbone has {len(hs)} hidden states, need index {last_layer}")

            comm = community(hs[comm_L].float(), attention_mask=enc.attention_mask)
            cvec = comm.vector  # (1, d_community)

            T = int(enc.input_ids.shape[1])
            causal = torch.full((T, T), float("-inf"), device=dev).triu(1).view(1, 1, T, T)

            per_layer_norm = {}
            for i, L in enumerate(mah_layers):
                dv = mah_heads[i](hs[L].float(), community_vec=cvec,
                                  causal_mask=causal).divergence  # (1, T, d_div)
                per_layer_norm[L] = dv.norm(dim=-1)[0].float().cpu()  # (T,)

            last = per_layer_norm[last_layer]
            row = {
                "id": it["id"], "concept": it["concept"], "domain": it["domain"],
                "tier": it["tier"], "concrete": int(it["concrete"]),
                "contested": int(it["contested"]), "contest_score": it["contest_score"],
                "n_tokens": T,
                "div_last": float(last[-1]),
                "div_mean": float(last.mean()),
                "div_peak": float(last.max()),
                **{f"div_last_L{L}": float(per_layer_norm[L][-1]) for L in mah_layers},
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {it['concept']:<26} conc={row['concrete']} cont={row['contested']} "
                  f"div_last={row['div_last']:.3f} "
                  + " ".join(f"L{L}={row[f'div_last_L{L}']:.2f}" for L in mah_layers),
                  flush=True)

    # ---- analysis ------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    gemma = {m: analyze(rows, m, args.nperm, rng)
             for m in ["div_last"] + [f"div_last_L{L}" for L in mah_layers]}

    summary = {
        "backbone": mid, "repo": args.repo, "step": ckpt.get("step"),
        "n_concepts": len(rows), "mah_layers": mah_layers,
        "hypothesis": "abstractness (U_ref) and contestedness-within-abstract (U_com) "
                      "raise MAH divergence on a second backbone family",
        "gemma4": gemma,
    }

    # ---- optional Qwen side-by-side (merge 2x2 flags by id) -------------------
    qpath = Path(args.qwen_readouts)
    if qpath.exists():
        flags = {it["id"]: (int(it["concrete"]), int(it["contested"])) for it in items}
        qrows = []
        for l in qpath.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r["id"] in flags:
                r = dict(r)
                r["concrete"], r["contested"] = flags[r["id"]]
                qrows.append(r)
        if len(qrows) == len(items):
            summary["qwen_v1_div_last"] = analyze(qrows, "div_last", args.nperm, rng)

    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    # ---- print ---------------------------------------------------------------
    def show(name, a):
        ma = a["main_abstractness"]
        ca = a["contest_within_abstract"]
        cci = a["contest_within_concrete"]
        print(f"\n[{name}] {a['metric']}")
        print(f"  main abstractness   dz={ma['delta_z']:+.3f} d={ma['cohen_d']:+.2f} "
              f"p={ma['perm_p']:.4g}")
        print(f"  contest|abstract    dz={ca['delta_z']:+.3f} d={ca['cohen_d']:+.2f} "
              f"p={ca['perm_p']:.4g}")
        print(f"  contest|concrete    dz={cci['delta_z']:+.3f} d={cci['cohen_d']:+.2f} "
              f"p={cci['perm_p']:.4g}")
        print(f"  std beta  U_ref={a['std_beta']['U_ref']:+.3f}  "
              f"U_com={a['std_beta']['U_com']:+.3f}")

    print("\n=== Test 2: cross-backbone metapragmatic load (gemma-4-31B-it) ===")
    show("gemma4", gemma["div_last"])
    for L in mah_layers:
        show(f"gemma4 L{L}", gemma[f"div_last_L{L}"])
    if "qwen_v1_div_last" in summary:
        show("qwen v1.0 (ref)", summary["qwen_v1_div_last"])
    print(f"\nwrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
