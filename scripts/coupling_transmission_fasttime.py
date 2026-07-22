#!/usr/bin/env python3
"""Coupling test: transmission-time contestedness vs token-sequence-time order parameter.

Pure reanalysis (no model run) of a committed separatrix readouts file. Tests the
open hypothesis from the SRT / geomorph collaboration (see the "Token-sequence time
versus transmission time" section of the Erik correspondence): a concept whose sign
forks across discourse communities (high transmission-time contestedness) should carry
a higher within-pass reflexivity order parameter r_hat on the shared prompt stem
(token-sequence time). Ontogeny reading the shadow of phylogeny.

Two clocks, both already recorded per concept in the readouts:

  - transmission time (slow / phylogeny proxy): how far the technical (consensus)
    and mystical (contested) framings of a concept sit apart in CENTERED community
    space.  contest = 1 - cos_centered(v_technical, v_mystical).  A bedrock
    (history-of-science) framing gives a specificity control: mere framing change
    that is not genuine contestation should couple to r_hat more weakly.

  - token-sequence time (fast / ontogeny): BEN r_hat and MAH peak divergence read on
    the PROMPT ALONE, before either framing is committed.

Coupling = Spearman rho across concepts, with a permutation null and a domain-wise
breakdown.  All community-vector cosines are reported raw AND centered (pooled mean
removed), per the standing anisotropy rule for this backbone.

Input:  artifacts/separatrix/v8a/readouts.jsonl   (committed; 61 concepts, Qwen2.5-7B + v8a)
Output: artifacts/nla/coupling/separatrix_v8a.json
        artifacts/nla/figures/coupling_contest_vs_rhat.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- stats

def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks (1-indexed) with average-tie handling; no scipy dependency."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
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
    denom = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def perm_p(x: np.ndarray, y: np.ndarray, nperm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Two-sided permutation p-value for Spearman rho by shuffling ranks of y."""
    rx = rankdata(x)
    ry = rankdata(y)
    rho0 = pearson(rx, ry)
    if not np.isfinite(rho0):
        return rho0, float("nan")
    count = 0
    for _ in range(nperm):
        if abs(pearson(rx, rng.permutation(ry))) >= abs(rho0) - 1e-12:
            count += 1
    return rho0, (count + 1) / (nperm + 1)


# ------------------------------------------------------------------------- vectors

def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", default="artifacts/separatrix/v8a/readouts.jsonl")
    ap.add_argument("--out-json", default="artifacts/nla/coupling/separatrix_v8a.json")
    ap.add_argument("--out-fig", default="artifacts/nla/figures/coupling_contest_vs_rhat.png")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.readouts).read_text().splitlines() if l.strip()]

    # Pool all community vectors (all framings, all concepts) for the centering mean.
    pool = []
    for r in rows:
        for name in ("prompt_only", "technical", "mystical", "bedrock"):
            v = r["branches"].get(name, {}).get("community_vector")
            if v:
                pool.append(v)
    mu = np.asarray(pool, float).mean(axis=0)

    concepts, domains = [], []
    contest_raw, contest_cen, contest_bedrock_cen, spread_cen = [], [], [], []
    r_hat, mah_peak = [], []

    for r in rows:
        b = r["branches"]
        vt = b.get("technical", {}).get("community_vector")
        vm = b.get("mystical", {}).get("community_vector")
        vb = b.get("bedrock", {}).get("community_vector")
        rp = b.get("prompt_only", {}).get("r_hat_mean_prompt")
        mp = b.get("prompt_only", {}).get("mah_peak_full")
        if not (vt and vm and vb and rp is not None and mp is not None):
            continue
        vt, vm, vb = np.asarray(vt, float), np.asarray(vm, float), np.asarray(vb, float)
        vtc, vmc, vbc = vt - mu, vm - mu, vb - mu

        concepts.append(r.get("concept", r.get("id")))
        domains.append(r.get("domain", "?"))
        # transmission-time contestedness (higher = technical & mystical framings
        # live in more separated discourse-community subspaces)
        contest_raw.append(1.0 - cos(vt, vm))
        contest_cen.append(1.0 - cos(vtc, vmc))
        contest_bedrock_cen.append(1.0 - cos(vtc, vbc))  # specificity control
        # 3-way spread of the framings in centered community space
        spread_cen.append(np.mean([
            1.0 - cos(vtc, vmc), 1.0 - cos(vtc, vbc), 1.0 - cos(vmc, vbc)
        ]))
        # token-sequence-time order parameter (on the prompt stem alone)
        r_hat.append(float(rp))
        mah_peak.append(float(mp))

    n = len(concepts)
    contest_raw = np.array(contest_raw)
    contest_cen = np.array(contest_cen)
    contest_bedrock_cen = np.array(contest_bedrock_cen)
    spread_cen = np.array(spread_cen)
    r_hat = np.array(r_hat)
    mah_peak = np.array(mah_peak)
    rng = np.random.default_rng(args.seed)

    def couple(x, y, label):
        rho, p = perm_p(x, y, args.nperm, rng)
        return {"label": label, "spearman_rho": rho, "perm_p_two_sided": p, "n": n}

    couplings = {
        "contest_cen__r_hat": couple(contest_cen, r_hat, "centered contestedness vs r_hat (PRIMARY)"),
        "contest_cen__mah_peak": couple(contest_cen, mah_peak, "centered contestedness vs MAH peak divergence"),
        "contest_raw__r_hat": couple(contest_raw, r_hat, "raw contestedness vs r_hat"),
        "spread_cen__r_hat": couple(spread_cen, r_hat, "3-way centered framing spread vs r_hat"),
        "contest_bedrock_cen__r_hat": couple(contest_bedrock_cen, r_hat, "bedrock(consensus) contestedness vs r_hat (CONTROL)"),
    }

    # Tertile effect size on the primary axis
    order = np.argsort(contest_cen)
    k = n // 3
    bottom = r_hat[order[:k]]
    top = r_hat[order[-k:]]
    tertile = {
        "k_per_group": int(k),
        "mean_r_hat_low_contest": float(bottom.mean()),
        "mean_r_hat_high_contest": float(top.mean()),
        "delta": float(top.mean() - bottom.mean()),
    }

    # Domain breakdown of the primary coupling
    dom = {}
    for d in sorted(set(domains)):
        idx = [i for i, dd in enumerate(domains) if dd == d]
        if len(idx) >= 4:
            dom[d] = {"n": len(idx),
                      "spearman_rho": spearman(contest_cen[idx], r_hat[idx])}
        else:
            dom[d] = {"n": len(idx), "spearman_rho": None}

    result = {
        "source": args.readouts,
        "n_concepts": n,
        "centering_mu_norm": float(np.linalg.norm(mu)),
        "hypothesis": ("higher transmission-time contestedness (technical vs mystical "
                       "framing separation in centered community space) couples to higher "
                       "token-sequence-time order parameter r_hat on the shared prompt"),
        "means": {
            "contest_cen": float(contest_cen.mean()),
            "contest_bedrock_cen": float(contest_bedrock_cen.mean()),
            "r_hat": float(r_hat.mean()),
            "mah_peak": float(mah_peak.mean()),
        },
        "couplings": couplings,
        "tertile_primary": tertile,
        "domain_breakdown_primary": dom,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    # -------------------------------------------------------------------- figure
    plt.rcParams.update({
        "figure.dpi": 140, "font.size": 11,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    doms = sorted(set(domains))
    cmap = plt.get_cmap("tab10")
    for di, d in enumerate(doms):
        idx = [i for i, dd in enumerate(domains) if dd == d]
        ax.scatter(contest_cen[idx], r_hat[idx], s=34, color=cmap(di % 10),
                   alpha=0.85, edgecolor="white", linewidth=0.5, label=f"{d} (n={len(idx)})")
    # least-squares guide line
    m, c = np.polyfit(contest_cen, r_hat, 1)
    xs = np.linspace(float(contest_cen.min()), float(contest_cen.max()), 50)
    ax.plot(xs, m * xs + c, color="#333333", lw=1.4, ls="--")
    rho = couplings["contest_cen__r_hat"]["spearman_rho"]
    p = couplings["contest_cen__r_hat"]["perm_p_two_sided"]
    ax.set_title(f"Transmission-time contestedness vs token-sequence-time r\u0302\n"
                 f"Spearman \u03c1 = {rho:+.3f}  (permutation p = {p:.3g}, n = {n})",
                 fontsize=11)
    ax.set_xlabel("contestedness  1 \u2212 cos_centered(technical, mystical community vectors)")
    ax.set_ylabel("BEN r\u0302 on prompt (order parameter)")
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")

    # -------------------------------------------------------------------- console
    print(f"n={n} concepts  |mu|={np.linalg.norm(mu):.2f}")
    for key, c in couplings.items():
        print(f"  {c['label']:<52}  rho={c['spearman_rho']:+.3f}  p={c['perm_p_two_sided']:.3g}")
    print(f"tertile: r_hat high-contest {tertile['mean_r_hat_high_contest']:.3f} "
          f"vs low-contest {tertile['mean_r_hat_low_contest']:.3f} "
          f"(delta {tertile['delta']:+.3f})")
    print(f"wrote {out_json}")
    print(f"wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
