#!/usr/bin/env python3
"""Powered coupling run: curated contestedness vs token-sequence-time order parameter.

Fixes the two power problems in the v8a separatrix reanalysis
(scripts/coupling_transmission_fasttime.py, which was null but underpowered):

  1. range on the transmission axis: the probe battery
     (data/probes/contestedness_battery_v1.jsonl) spans mundane/consensus concepts
     (a bicycle, the number seven) through deeply contested ones (freedom, God,
     consciousness), each with a curated contestedness score in [0, 1].
  2. range on the order-parameter axis: because the battery includes consensus
     concepts, BEN r_hat is no longer forced supercritical on every item.

For each concept we run ONE forward pass of the SRT adapter on a uniform prompt
stem ("<concept>, properly understood, is") and read the token-sequence-time
statistics on the stem: BEN r_hat (mean, max), P(supercritical), and MAH last-layer
divergence (mean, peak). We then test the coupling hypothesis: curated
transmission-time contestedness should track the within-pass reflexivity r_hat.

Coupling = Spearman rho across concepts, permutation null, tertile effect size,
domain breakdown. The uniform stem holds everything but the concept fixed.

Usage (box):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/contestedness_coupling_run.py \
        --repo RiverRider/srt-adapter-v1.0 \
        --probe data/probes/contestedness_battery_v1.jsonl \
        --out-readouts artifacts/nla/coupling/contestedness_v1_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/contestedness_v1_summary.json
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


# ------------------------------------------------------------------ adapter load

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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, z) -> float:
    """Rank partial correlation of x,y controlling for z (residualize ranks on z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        beta = np.linalg.lstsq(b1, a, rcond=None)[0]
        return a - b1 @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def perm_p(x, y, nperm, rng) -> tuple[float, float]:
    rx, ry = rankdata(x), rankdata(y)
    rho0 = pearson(rx, ry)
    if not np.isfinite(rho0):
        return rho0, float("nan")
    c = sum(abs(pearson(rx, rng.permutation(ry))) >= abs(rho0) - 1e-12
            for _ in range(nperm))
    return rho0, (c + 1) / (nperm + 1)


# -------------------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/srt-adapter-v1.0")
    ap.add_argument("--probe", default="data/probes/contestedness_battery_v1.jsonl")
    ap.add_argument("--out-readouts", default="artifacts/nla/coupling/contestedness_v1_readouts.jsonl")
    ap.add_argument("--out-summary", default="artifacts/nla/coupling/contestedness_v1_summary.json")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = build_config(Path(hf_hub_download(args.repo, "config.json")))
    print(f"[load] backbone={cfg.backbone_id} dtype={cfg.backbone_dtype} "
          f"use_prototypes={cfg.community.use_prototypes}")
    model = SRTAdapter(cfg).to(args.device)
    state = load_file(hf_hub_download(args.repo, "adapter.safetensors"), device=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] adapter tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    print(f"[probe] {len(items)} concepts")

    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_r.open("w") as f:
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len)
            o = model(input_ids=enc.input_ids.to(args.device),
                      attention_mask=enc.attention_mask.to(args.device))
            r = o.ben_output.r_hat[0].float().cpu()
            psuper = torch.softmax(o.ben_output.regime_logits[0], dim=-1)[:, 1].float().cpu()
            div = o.divergences[-1].norm(dim=-1)[0].float().cpu()
            row = {
                "id": it["id"], "concept": it["concept"], "domain": it["domain"],
                "tier": it["tier"], "contest_score": it["contest_score"],
                "n_tokens": int(enc.input_ids.shape[1]),
                "r_hat_mean": float(r.mean()), "r_hat_max": float(r.max()),
                "psuper_mean": float(psuper.mean()), "psuper_max": float(psuper.max()),
                "div_mean": float(div.mean()), "div_peak": float(div.max()),
                "r_hat_last": float(r[-1]), "psuper_last": float(psuper[-1]),
                "div_last": float(div[-1]),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {it['concept']:<28} tier{it['tier']} "
                  f"contest={it['contest_score']:.2f}  r_hat={row['r_hat_mean']:+.3f} "
                  f"psuper={row['psuper_mean']:.3f} div={row['div_mean']:.3f}")

    contest = np.array([r["contest_score"] for r in rows])
    domains = [r["domain"] for r in rows]
    rng = np.random.default_rng(args.seed)

    def couple(key):
        y = np.array([r[key] for r in rows])
        rho, p = perm_p(contest, y, args.nperm, rng)
        return {"metric": key, "spearman_rho": rho, "perm_p_two_sided": p}

    couplings = {k: couple(k) for k in
                 ("r_hat_mean", "r_hat_max", "r_hat_last", "psuper_mean", "psuper_last",
                  "div_mean", "div_peak", "div_last")}

    # length-controlled partials for the divergence channel (n_tokens is the confound:
    # contested concepts are shorter, and averaged divergence inflates on short prompts).
    ntok = np.array([r["n_tokens"] for r in rows])
    length_controlled = {
        k: {"partial_rho_given_n_tokens": partial_spearman(
            contest, np.array([r[k] for r in rows]), ntok)}
        for k in ("div_mean", "div_peak", "div_last")
    }
    length_controlled["contest_vs_n_tokens"] = spearman(contest, ntok)

    # tertile effect on the primary axis (r_hat_mean by contestedness)
    order = np.argsort(contest)
    k = len(rows) // 3
    yv = np.array([r["r_hat_mean"] for r in rows])
    tertile = {
        "k_per_group": int(k),
        "r_hat_low_contest": float(yv[order[:k]].mean()),
        "r_hat_high_contest": float(yv[order[-k:]].mean()),
        "delta": float(yv[order[-k:]].mean() - yv[order[:k]].mean()),
    }
    # tier means
    tiers = {}
    for t in sorted(set(r["tier"] for r in rows)):
        m = np.array([r["r_hat_mean"] for r in rows if r["tier"] == t])
        tiers[f"tier{t}"] = {"n": int(len(m)), "r_hat_mean": float(m.mean())}
    # domain breakdown of the primary coupling
    dom = {}
    for d in sorted(set(domains)):
        idx = [i for i, dd in enumerate(domains) if dd == d]
        dom[d] = {"n": len(idx),
                  "spearman_rho": (spearman(contest[idx], yv[idx]) if len(idx) >= 4 else None)}

    summary = {
        "repo": args.repo, "n_concepts": len(rows),
        "hypothesis": "curated transmission-time contestedness tracks within-pass r_hat",
        "couplings": couplings, "length_controlled_divergence": length_controlled,
        "tertile_primary": tertile,
        "tier_means_r_hat": tiers, "domain_breakdown_r_hat_mean": dom,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print("\n=== coupling (contestedness vs fast-time) ===")
    for k2, c in couplings.items():
        print(f"  {k2:<12} rho={c['spearman_rho']:+.3f}  p={c['perm_p_two_sided']:.4g}")
    print("--- divergence, length-controlled (partial rho | n_tokens) ---")
    print(f"  contest vs n_tokens rho={length_controlled['contest_vs_n_tokens']:+.3f}")
    for k2 in ("div_mean", "div_peak", "div_last"):
        print(f"  {k2:<12} partial_rho={length_controlled[k2]['partial_rho_given_n_tokens']:+.3f}")
    print(f"tiers r_hat: " + "  ".join(f"{t}={v['r_hat_mean']:+.3f}(n{v['n']})"
                                       for t, v in tiers.items()))
    print(f"tertile r_hat: high={tertile['r_hat_high_contest']:+.3f} "
          f"low={tertile['r_hat_low_contest']:+.3f} delta={tertile['delta']:+.3f}")
    print(f"wrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
