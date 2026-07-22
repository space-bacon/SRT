#!/usr/bin/env python3
"""Test 4 (diachronic): does metapragmatic load D drift with, and track, U_com
across 1770-1964 newspaper text?

The Metapragmatic-load conjecture (paper_nla.md §12.5) predicts that a sign
whose community underdetermination U_com rose over historical (transmission)
time should show rising divergence D. The text route avoids the two failures
of the vision route: the community-prototype channel is not degenerate on
text, and there is no fixed-token image compression.

Source: dell-research-harvard/AmericanStories, per-year OCR article tarballs
faro_YYYY.tar.gz spanning 1770-1964 (a single consistent corpus). We stream
each year (so the multi-GB late years only download the prefix we read) and
sample up to --per-year articles, filtered for a minimum length so OCR
fragments are dropped. Every article is a real dated passage; the tarball year
is its date.

Per article we read the community vector at L8, condition the three MAH heads
on it, and record:
  * D          - MAH divergence L2 norm (last layer L45), mean-pooled over the
                 article's tokens (primary) and at the last token.
  * ucom_entropy - entropy of the article's community assignment over 32
                 prototypes (how many discourse communities the passage spans;
                 the per-passage U_com proxy).
  * ucom_spread  - std of the per-token community-encoder codes across the
                 article (a second, continuous dispersion proxy).

Analyses:
  1. STRUCTURAL (the law in naturalistic historical prose): does D track
     ucom_entropy / ucom_spread across all articles, controlling passage
     length? This is the diachronic analogue of Test 1/2's coupling.
  2. DIACHRONIC drift: do per-article and per-year-mean D and U_com trend with
     year (Spearman vs year)? Rising U_com with rising D across eras is the
     conjecture's transmission-time prediction.
Both U_com proxies are variance-checked before the couplings are trusted (the
vision run's proxy had saturated at log(32)).

Usage (box, transformers>=5):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/diachronic_metapragmatic_gemma4.py --per-year 80 \
        --out-readouts artifacts/nla/coupling/diachronic_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/diachronic_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import get_token

DATASET = "dell-research-harvard/AmericanStories"


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


def partial_spearman(x, y, z) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        beta = np.linalg.lstsq(b1, a, rcond=None)[0]
        return a - b1 @ beta

    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def perm_p_spearman(x, y, nperm, rng) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rho0 = pearson(rx, ry)
    if not np.isfinite(rho0):
        return float("nan")
    c = sum(abs(pearson(rx, rng.permutation(ry))) >= abs(rho0) - 1e-12
            for _ in range(nperm))
    return (c + 1) / (nperm + 1)


# --------------------------------------------------------------------- streaming

def stream_year_articles(year: int, token: str | None, per_year: int,
                         min_chars: int, max_chars: int):
    """Yield up to `per_year` article texts from faro_YEAR.tar.gz by streaming.

    tarfile 'r|gz' reads sequentially, so requests stops pulling bytes once we
    break; the multi-GB late years only transfer the prefix we consume.
    """
    import requests

    url = hf_hub_url(DATASET, f"faro_{year}.tar.gz", repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    n = 0
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120,
                          allow_redirects=True) as r:
            if r.status_code != 200:
                return
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for member in tf:
                    if n >= per_year:
                        break
                    if not member.name.endswith(".json"):
                        continue
                    try:
                        d = json.loads(tf.extractfile(member).read())
                    except Exception:
                        continue
                    for art in d.get("full articles", []):
                        if n >= per_year:
                            break
                        txt = (art.get("article") or "").strip()
                        txt = re.sub(r"\s+", " ", txt)
                        if len(txt) < min_chars:
                            continue
                        yield txt[:max_chars]
                        n += 1
    except Exception as e:
        print(f"  [year {year}] stream error: {str(e)[:80]}", flush=True)


# -------------------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/Gemma-4-31B-it-SRT-Sunstone")
    ap.add_argument("--ckpt-file", default="readout_selected.pt")
    ap.add_argument("--per-year", type=int, default=80)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--year-start", type=int, default=1770)
    ap.add_argument("--year-end", type=int, default=1964)
    ap.add_argument("--out-readouts",
                    default="artifacts/nla/coupling/diachronic_readouts.jsonl")
    ap.add_argument("--out-summary",
                    default="artifacts/nla/coupling/diachronic_summary.json")
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    from srt.config import CommunityConfig, MAHConfig
    from srt.modules.community import CommunityDiscoveryHead
    from srt.modules.mah import MetapragmaticAttentionHead

    dev = args.device

    # ---- read-out heads (same loader as Test 2) -----------------------------
    ckpt = torch.load(hf_hub_download(args.repo, args.ckpt_file),
                      map_location="cpu", weights_only=False)
    cc = ckpt["config"]
    comm_L = int(cc["community_layer"])
    mah_layers = [int(x) for x in cc["mah_layers"]]
    d_backbone = int(cc["d_backbone"])
    d_community = int(cc["d_community"])
    mid = cc["backbone"]
    heads = ckpt["heads"]
    print(f"[ckpt] step {ckpt.get('step')} backbone={mid} community@{comm_L} "
          f"mah@{mah_layers}", flush=True)

    community = CommunityDiscoveryHead(CommunityConfig(d_community=d_community),
                                       d_backbone).to(dev).eval()
    community.load_state_dict(
        {k[len("0."):]: v for k, v in heads.items() if k.startswith("0.")})
    community.float()

    mah_cfg = MAHConfig()
    mah_heads = []
    for i in range(len(mah_layers)):
        h = MetapragmaticAttentionHead(mah_cfg, d_backbone, d_community=d_community)
        sd = {k[len(f"1.{i}."):]: v for k, v in heads.items() if k.startswith(f"1.{i}.")}
        miss, unexp = h.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f"MAH {i} load mismatch"
        mah_heads.append(h.to(dev).eval().float())
    last_i = len(mah_layers) - 1
    print(f"[heads] community + {len(mah_heads)} MAH heads loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Gemma4ForConditionalGeneration.from_pretrained(
        mid, dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    protos = F.normalize(community.prototypes.weight, dim=-1)

    def article_readout(text: str) -> dict | None:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_len)
        ids = enc.input_ids.to(dev)
        am = enc.attention_mask.to(dev)
        T = int(ids.shape[1])
        if T < 8:
            return None
        hs = model(input_ids=ids, attention_mask=am,
                   output_hidden_states=True, use_cache=False).hidden_states

        co = community(hs[comm_L].float(), attention_mask=am)
        cvec = co.vector
        # per-passage U_com proxies
        enc_tok = community.encoder(hs[comm_L].float()[0])  # (T, dc)
        enc_n = F.normalize(enc_tok, dim=-1)
        w = F.softmax((enc_n @ protos.T) / community.temperature, dim=-1)  # (T, K)
        page_assign = w.mean(0)
        ucom_entropy = float(-(page_assign * (page_assign + 1e-9).log()).sum())
        ucom_spread = float(enc_n.std(0).mean())

        causal = torch.full((T, T), float("-inf"), device=dev).triu(1).view(1, 1, T, T)
        d_mean, d_last = {}, {}
        for i, L in enumerate(mah_layers):
            dv = mah_heads[i](hs[L].float(), community_vec=cvec,
                              causal_mask=causal).divergence
            nrm = dv.norm(dim=-1)[0]
            d_mean[L] = float(nrm.mean())
            d_last[L] = float(nrm[-1])
        return {
            "n_tokens": T,
            "D": d_mean[mah_layers[last_i]],
            "D_last": d_last[mah_layers[last_i]],
            **{f"D_L{L}": d_mean[L] for L in mah_layers},
            "ucom_entropy": ucom_entropy,   # prototype channel (degenerate, kept for record)
            "ucom_spread": ucom_spread,     # continuous per-token code spread
            "encoded": co.encoded[0].float().cpu().tolist(),  # 64-d community code
        }

    # ---- available years ----------------------------------------------------
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    yr_size = {}
    for s in info.siblings:
        m = re.match(r"faro_(\d{4})\.tar\.gz", s.rfilename)
        if m:
            yr_size[int(m.group(1))] = s.size or 0
    years = [y for y in sorted(yr_size)
             if args.year_start <= y <= args.year_end and yr_size[y] > 1000]
    print(f"[years] {len(years)} available in [{args.year_start},{args.year_end}]",
          flush=True)

    token = get_token()
    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with out_r.open("w") as f:
        for yi, year in enumerate(years):
            got = 0
            for txt in stream_year_articles(year, token, args.per_year,
                                            args.min_chars, args.max_chars):
                r = article_readout(txt)
                if r is None:
                    continue
                r["year"] = year
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                got += 1
            f.flush()
            if yi % 5 == 0 or got == 0:
                med_d = np.median([x["D"] for x in rows]) if rows else 0
                print(f"  [{yi+1}/{len(years)}] year {year}: +{got} arts "
                      f"(total {len(rows)}, medD={med_d:.3f})", flush=True)

    print(f"[done] {len(rows)} articles across {len(years)} years", flush=True)

    # ---- analysis -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    D = np.array([r["D"] for r in rows], float)
    Uent = np.array([r["ucom_entropy"] for r in rows], float)
    Uspr = np.array([r["ucom_spread"] for r in rows], float)
    ntok = np.array([r["n_tokens"] for r in rows], float)
    yr = np.array([r["year"] for r in rows], float)

    # ---- U_com from corpus-discovered community geometry --------------------
    # The prototype channel is degenerate (ucom_entropy saturates at log K), but
    # the continuous encoder discriminates communities (cross_modal kNN 0.64).
    # We recover the community structure the encoder learned by clustering the
    # L2-normalised encoded vectors, and define per-passage U_com a priori as
    # the ambiguity of its soft assignment to those clusters (entropy) and the
    # closeness of its two nearest clusters (small margin = between communities).
    E = np.array([r["encoded"] for r in rows], float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    K = int(protos.shape[0])

    def kmeans(X, k, iters=30):
        C = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
            a = d.argmin(1)
            newC = np.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        return C

    C = kmeans(E, K)
    dist = (E ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * E @ C.T  # (N,K)
    tau = float(np.median(dist.min(1))) + 1e-6
    soft = np.exp(-dist / tau)
    soft = soft / soft.sum(1, keepdims=True)
    ucom_cluster = -(soft * np.log(soft + 1e-12)).sum(1)          # assignment entropy
    ds = np.sort(dist, axis=1)
    ucom_margin = -(ds[:, 1] - ds[:, 0])                          # -(2nd - 1st nearest)

    # variance check (vision + text prototype proxy saturated at log(32))
    def vstat(a):
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}
    proxy_variance = {
        "ucom_entropy_prototype": vstat(Uent),
        "ucom_spread": vstat(Uspr),
        "ucom_cluster_entropy": vstat(ucom_cluster),
        "ucom_cluster_margin": vstat(ucom_margin),
        "log_K_ceiling": float(np.log(K)),
    }

    def couple(name, yv):
        return {
            "metric": name,
            "spearman": spearman(D, yv),
            "partial_given_ntok": partial_spearman(D, yv, ntok),
            "perm_p": perm_p_spearman(D, yv, args.nperm, rng),
        }

    structural = {
        "D_vs_ucom_cluster_entropy": couple("ucom_cluster_entropy", ucom_cluster),
        "D_vs_ucom_cluster_margin": couple("ucom_cluster_margin", ucom_margin),
        "D_vs_ucom_spread": couple("ucom_spread", Uspr),
        "D_vs_ntok(control)": {"spearman": spearman(D, ntok)},
    }

    # per-year means for the diachronic curve
    ymeans = {}
    for y in sorted(set(int(v) for v in yr)):
        mask = yr == y
        ymeans[y] = {"n": int(mask.sum()), "D": float(D[mask].mean()),
                     "ucom_cluster": float(ucom_cluster[mask].mean()),
                     "ucom_spread": float(Uspr[mask].mean())}
    yy = np.array(sorted(ymeans))
    diachronic = {
        "per_article": {
            "year_vs_D": spearman(yr, D),
            "year_vs_ucom_cluster": spearman(yr, ucom_cluster),
            "year_vs_ucom_spread": spearman(yr, Uspr),
            "year_vs_D_partial_ntok": partial_spearman(yr, D, ntok),
        },
        "per_year_mean": {
            "year_vs_meanD": spearman(yy, np.array([ymeans[y]["D"] for y in yy])),
            "year_vs_mean_ucom_cluster": spearman(
                yy, np.array([ymeans[y]["ucom_cluster"] for y in yy])),
        },
    }
    decade: dict = {}
    for i, r in enumerate(rows):
        decade.setdefault((r["year"] // 10) * 10, []).append(i)
    decade_means = {int(dc): {
        "n": len(idx),
        "D": float(D[idx].mean()),
        "ucom_cluster": float(ucom_cluster[idx].mean()),
        "ucom_spread": float(Uspr[idx].mean()),
    } for dc, idx in sorted(decade.items())}

    summary = {
        "backbone": mid, "repo": args.repo, "dataset": DATASET,
        "n_articles": len(rows), "n_years": len(set(int(v) for v in yr)),
        "year_range": [int(yr.min()), int(yr.max())],
        "mah_layers": mah_layers, "n_clusters": K,
        "proxy_variance": proxy_variance,
        "structural_coupling": structural,
        "diachronic": diachronic,
        "decade_means": decade_means,
        "per_year_mean": {int(y): ymeans[y] for y in ymeans},
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print("\n=== Test 4: diachronic metapragmatic load (AmericanStories 1770-1964) ===")
    print(f"n={len(rows)} articles, {len(set(int(v) for v in yr))} years "
          f"[{int(yr.min())}-{int(yr.max())}]")
    print("proxy variance (must be > ~0 to be usable):")
    print(f"  prototype entropy std={proxy_variance['ucom_entropy_prototype']['std']:.4f}"
          f" (DEGENERATE if ~0; ceiling logK={np.log(K):.3f})")
    print(f"  cluster entropy   std={proxy_variance['ucom_cluster_entropy']['std']:.4f}")
    print(f"  code spread       std={proxy_variance['ucom_spread']['std']:.4f}")
    s = structural
    print("structural coupling (the law in historical prose):")
    for key in ("D_vs_ucom_cluster_entropy", "D_vs_ucom_cluster_margin",
                "D_vs_ucom_spread"):
        c = s[key]
        print(f"  {key:<28} rho={c['spearman']:+.3f} "
              f"partial|ntok={c['partial_given_ntok']:+.3f} p={c['perm_p']:.4g}")
    print(f"  control D~ntok rho={s['D_vs_ntok(control)']['spearman']:+.3f}")
    d = diachronic
    print("diachronic drift:")
    print(f"  per-article  year~D={d['per_article']['year_vs_D']:+.3f} "
          f"year~ucom_cluster={d['per_article']['year_vs_ucom_cluster']:+.3f}")
    print(f"  per-year-mean year~meanD={d['per_year_mean']['year_vs_meanD']:+.3f} "
          f"year~mean_ucom={d['per_year_mean']['year_vs_mean_ucom_cluster']:+.3f}")
    print("decade means (D / ucom_cluster / spread):")
    for dc, m in decade_means.items():
        print(f"  {dc}s n={m['n']:<5} D={m['D']:.3f} "
              f"Uclu={m['ucom_cluster']:.3f} Uspr={m['ucom_spread']:.4f}")
    print(f"\nwrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
