#!/usr/bin/env python3
"""Test 3 (causal U_com): force the community index and measure interpretant movement.

Tests 1, 2 and 4 established that metapragmatic load D rises with community
contestedness, but contestedness was always a *curated* ordinal, never a
causally measured community-decoding disagreement. This script closes that gap
by actively intervening on the community index at decode time, using the one
released adapter that carries a trained discrete community codebook:
RiverRider/zooL4nD3r-v0.1 (a full SRT adapter on frozen Qwen2.5-7B with 32
trained, near-orthogonal community prototypes AND community-conditioned MAH
heads).

The conjecture predicts: forcing the community index should move the
interpretant ONLY where U_com is high. So for each concept in the 2x2
dissociation battery we run the adapter once per community prototype j,
overriding the discovered community with `forced_community = prototype_j`, and
read the MAH divergence (interpretant) at the final shared token. The
concept's causally measured U_com is the DISPERSION of that interpretant across
the 32 communities:

    U_com^causal(concept) = RMS spread of the divergence vector d_j over the
                            32 forced communities.

A concept whose interpretant barely moves when you change the community has low
community underdetermination; one that swings a lot has high U_com. The
predictions are (i) U_com^causal is larger for curated-contested than
curated-consensus concepts, (ii) it tracks the curated contest score (so the
ordinal is validated against a causal measure), and (iii) the baseline load D
tracks U_com^causal (the load is driven by the causally measured community
disagreement).

Usage (box, PYTHONPATH=. so srt/ imports):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/causal_community_forcing.py \
        --repo RiverRider/zooL4nD3r-v0.1 \
        --probe data/probes/contestedness_dissociation_v1.jsonl \
        --out-readouts artifacts/nla/coupling/causal_forcing_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/causal_forcing_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import (
    BENConfig, CommunityConfig, LossConfig, MAHConfig, RRMConfig, SRTConfig,
)


def build_config(config_path: Path) -> SRTConfig:
    raw = json.loads(config_path.read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(**{k: v for k, v in raw["loss"].items()
                           if k in LossConfig.__dataclass_fields__}),
    )


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


def pearson(x, y) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def perm_p_meandiff(y, group, nperm, rng) -> tuple[float, float]:
    y = np.asarray(y, float)
    g = np.asarray(group).astype(bool)
    obs = y[g].mean() - y[~g].mean()
    c = 0
    for _ in range(nperm):
        gp = rng.permutation(g)
        if abs(y[gp].mean() - y[~gp].mean()) >= abs(obs) - 1e-12:
            c += 1
    return float(obs), (c + 1) / (nperm + 1)


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


# -------------------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/srt-adapter-v1.0")
    ap.add_argument("--probe", default="data/probes/contestedness_dissociation_v1.jsonl")
    ap.add_argument("--anchor-source", default="auto", choices=["auto", "prototypes", "corpus"],
                    help="auto: use trained prototypes if present, else corpus")
    ap.add_argument("--anchor-corpus", default="data/val_200.jsonl",
                    help="neutral probe text to elicit the model's own community vectors")
    ap.add_argument("--n-anchors", type=int, default=24)
    ap.add_argument("--out-readouts",
                    default="artifacts/nla/coupling/causal_forcing_readouts.jsonl")
    ap.add_argument("--out-summary",
                    default="artifacts/nla/coupling/causal_forcing_summary.json")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = build_config(Path(hf_hub_download(args.repo, "config.json")))
    print(f"[load] backbone={cfg.backbone_id} use_prototypes={cfg.community.use_prototypes} "
          f"K={cfg.community.num_prototypes}", flush=True)
    model = SRTAdapter(cfg).to(args.device)
    state = load_file(hf_hub_download(args.repo, "adapter.safetensors"), device=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model.eval()

    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # the model's own communities. zooL4nD3r has a trained prototype codebook;
    # v1.0 is trajectory-mode (no codebook), so we elicit its communities by
    # running its OWN community head over neutral probe text and clustering the
    # resulting vectors. Either way the communities come from the model, not
    # from external labels.
    ph = model.community_head.prototypes
    trained_protos = (ph is not None
                      and args.anchor_source in ("auto", "prototypes"))
    if trained_protos:
        anchors_t = ph.weight.detach().float().cpu()
        src = "trained prototype codebook"
    else:
        texts = [json.loads(x)["text"] for x in
                 Path(args.anchor_corpus).read_text().splitlines() if x.strip()]
        vecs = []
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            oc = model(input_ids=e.input_ids.to(args.device),
                       attention_mask=e.attention_mask.to(args.device))
            vecs.append(oc.community_output.vector[0].float().cpu().numpy())
        V = np.stack(vecs)
        rng0 = np.random.default_rng(args.seed)
        C = V[rng0.choice(len(V), args.n_anchors, replace=False)].copy()
        for _ in range(40):
            dd = (V ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * V @ C.T
            a = dd.argmin(1)
            newC = np.stack([V[a == j].mean(0) if (a == j).any() else C[j]
                             for j in range(args.n_anchors)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        anchors_t = torch.tensor(C, dtype=torch.float32)
        src = f"{args.n_anchors} k-means anchors from {len(texts)} probe passages"
    # match the MAH comm_proj dtype (bfloat16); _to_head moves device only, not dtype
    head_dtype = next(model.mah_heads[0].comm_proj.parameters()).dtype
    protos = anchors_t.to(device=args.device, dtype=head_dtype)
    K = protos.shape[0]
    pf = anchors_t.numpy()
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-9)
    offdiag = (pn @ pn.T)[~np.eye(K, dtype=bool)]
    print(f"[communities] K={K} ({src}); pairwise cos mean={offdiag.mean():+.3f}",
          flush=True)

    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    print(f"[probe] {len(items)} concepts", flush=True)

    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_r.open("w") as f:
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len)
            ids = enc.input_ids.to(args.device)
            am = enc.attention_mask.to(args.device)

            # baseline: discovered community
            o = model(input_ids=ids, attention_mask=am)
            base_div = o.divergences[-1].norm(dim=-1)[0].float().cpu()  # (T,)
            base_last = float(base_div[-1])

            # force each community prototype; collect the last-token divergence VECTOR
            dvecs = []
            dlast = []
            for j in range(K):
                fc = protos[j:j + 1].to(args.device)  # (1, d_community)
                oj = model(input_ids=ids, attention_mask=am, forced_community=fc)
                dj = oj.divergences[-1][0, -1, :].float().cpu().numpy()  # (d_div,)
                dvecs.append(dj)
                dlast.append(float(np.linalg.norm(dj)))
            dvecs = np.stack(dvecs)              # (K, d_div)
            dlast = np.array(dlast)              # (K,)

            # causal U_com = RMS movement of the interpretant across communities
            dbar = dvecs.mean(0)
            ucom_causal = float(np.sqrt(((dvecs - dbar) ** 2).sum(1).mean()))
            ucom_norm_std = float(dlast.std())   # scalar-norm version
            row = {
                "id": it["id"], "concept": it["concept"], "domain": it["domain"],
                "tier": it["tier"], "concrete": int(it["concrete"]),
                "contested": int(it["contested"]), "contest_score": it["contest_score"],
                "n_tokens": int(ids.shape[1]),
                "div_last_baseline": base_last,
                "ucom_causal": ucom_causal,
                "ucom_norm_std": ucom_norm_std,
                "div_last_forced_mean": float(dlast.mean()),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {it['concept']:<26} conc={row['concrete']} cont={row['contested']} "
                  f"D_base={base_last:.3f} Ucom^causal={ucom_causal:.3f}", flush=True)

    # ---- analysis -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    conc = np.array([r["concrete"] for r in rows])
    cont = np.array([r["contested"] for r in rows])
    U = np.array([r["ucom_causal"] for r in rows], float)
    Db = np.array([r["div_last_baseline"] for r in rows], float)
    cs = np.array([r["contest_score"] for r in rows], float)

    def cell(c, k):
        return U[(conc == c) & (cont == k)]

    cells = {f"concrete{c}_contested{k}": float(cell(c, k).mean())
             for c in (1, 0) for k in (0, 1)}

    # (i) contested vs consensus on U_com^causal
    ca_delta, ca_p = perm_p_meandiff(U[conc == 0], cont[conc == 0], args.nperm, rng)
    cc_delta, cc_p = perm_p_meandiff(U[conc == 1], cont[conc == 1], args.nperm, rng)
    all_delta, all_p = perm_p_meandiff(U, cont, args.nperm, rng)

    summary = {
        "repo": args.repo, "backbone": cfg.backbone_id, "n_concepts": len(rows),
        "K_communities": int(K), "codebook_pairwise_cos_mean": float(offdiag.mean()),
        "hypothesis": "forcing the community index moves the interpretant (MAH "
                      "divergence) more where U_com is high (contested concepts)",
        "ucom_causal_cells": cells,
        "contested_vs_consensus": {
            "all": {"delta": all_delta, "cohen_d": cohen_d(U[cont == 1], U[cont == 0]),
                    "perm_p": all_p},
            "within_abstract": {"delta": ca_delta, "perm_p": ca_p,
                                "cohen_d": cohen_d(cell(0, 1), cell(0, 0))},
            "within_concrete": {"delta": cc_delta, "perm_p": cc_p,
                                "cohen_d": cohen_d(cell(1, 1), cell(1, 0))},
        },
        "validate_ordinal": {
            "spearman_ucom_causal_vs_contest_score": spearman(cs, U),
            "spearman_ucom_causal_vs_baseline_D": spearman(Db, U),
            "spearman_baseline_D_vs_contest_score": spearman(cs, Db),
        },
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print("\n=== Test 3: causal community forcing (zooL4nD3r, K=%d communities) ===" % K)
    print("U_com^causal (interpretant movement) cell means:")
    print(f"  concrete: consensus {cells['concrete1_contested0']:.3f}  "
          f"contested {cells['concrete1_contested1']:.3f}")
    print(f"  abstract: consensus {cells['concrete0_contested0']:.3f}  "
          f"contested {cells['concrete0_contested1']:.3f}")
    cv = summary["contested_vs_consensus"]
    print(f"contested-consensus U_com^causal: all d={cv['all']['cohen_d']:+.2f} "
          f"p={cv['all']['perm_p']:.4g} | within-abstract d={cv['within_abstract']['cohen_d']:+.2f} "
          f"p={cv['within_abstract']['perm_p']:.4g}")
    vo = summary["validate_ordinal"]
    print("validate curated ordinal against causal measure:")
    print(f"  U_com^causal ~ curated contest_score: rho={vo['spearman_ucom_causal_vs_contest_score']:+.3f}")
    print(f"  baseline D    ~ U_com^causal:          rho={vo['spearman_ucom_causal_vs_baseline_D']:+.3f}")
    print(f"  baseline D    ~ curated contest_score: rho={vo['spearman_baseline_D_vs_contest_score']:+.3f}")
    print(f"\nwrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
