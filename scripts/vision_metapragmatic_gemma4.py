#!/usr/bin/env python3
"""Cross-modal metapragmatic load: does D track U_com on newspaper page IMAGES?

Test 2 established that the community-underdetermination coupling
(beta_{U_com}) is substrate-invariant across two backbone families read out
by two independently trained side-channels. This script asks the next
question the Metapragmatic-load conjecture implies: is the coupling
MODALITY-invariant? The SRT-Sunstone read-out for frozen google/gemma-4-31B-it
transfers to images with zero image training (cross_modal_readout.py: image ->
discourse community, kNN 0.64). Here we point the SAME community head (L8) and
the SAME three MAH divergence heads (L15/30/45) at scanned newspaper PAGE
IMAGES and test, per page:

    does metapragmatic divergence D track the page's community dispersion?

D is the MAH divergence L2 norm (last layer L45, matching Test 2), mean-pooled
over the image soft-token positions. The U_com proxy is the entropy of the
page's community assignment: we run the community encoder per image token,
soft-assign each to the 32 prototypes, average the assignment across the page,
and take its entropy. A page whose tokens spread across many discourse
communities has high U_com; one that collapses onto a single community has low
U_com. The conjecture predicts a positive D~U_com coupling in this modality
too.

The material form of a newspaper page (typography, layout, print convention)
is part of an era's semiotic infrastructure, not noise on top of it, so we do
not try to scrub the visual-era signal. Instead the primary test is the
STRUCTURAL coupling across the pooled corpus (era supplies range on the axis),
controlled for raw image size (n image tokens) so a positive result is not
merely "busier page." Era (1901 vs 1920-1950, the range cleanly available as
embedded images) is a secondary diachronic probe.

Usage (box, transformers>=5, datasets+pillow):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/vision_metapragmatic_gemma4.py --n-per-era 60 \
        --out-readouts artifacts/nla/coupling/vision_metaprag_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/vision_metaprag_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download


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
    """Rank partial correlation of x,y controlling for z."""
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


# ----------------------------------------------------------------------- images

def load_pages(n_per_era: int) -> list[dict]:
    """Load embedded newspaper page images from davanstrien chronicling-america.

    Returns a list of {image: PIL.Image, era: str, year: int}. Only sources
    with embedded image bytes are used (loc.gov URL fetch is unreliable), so
    the diachronic span here is 1901 vs 1920-1950.
    """
    import pandas as pd
    from PIL import Image

    api = HfApi()
    specs = [
        ("davanstrien/chronicling-america-1901-sample", "1901"),
        ("davanstrien/chronicling-america-1920-1950", "1920-1950"),
    ]
    pages: list[dict] = []
    for rid, era in specs:
        pqs = [f for f in api.list_repo_files(rid, repo_type="dataset")
               if f.endswith(".parquet")]
        taken = 0
        for pq in pqs:
            if taken >= n_per_era:
                break
            df = pd.read_parquet(hf_hub_download(rid, pq, repo_type="dataset"))
            for _, r in df.iterrows():
                if taken >= n_per_era:
                    break
                img = r.get("image")
                if not isinstance(img, dict) or not img.get("bytes"):
                    continue
                # year from whatever date field exists
                year = None
                for key in ("date", "issue_date"):
                    v = r.get(key)
                    if v is not None:
                        s = str(v)
                        for tok in s.replace("-", " ").replace("/", " ").split():
                            if tok.isdigit() and len(tok) == 4:
                                year = int(tok)
                                break
                    if year:
                        break
                if year is None:
                    year = 1901 if era == "1901" else 1935
                try:
                    im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                except Exception:
                    continue
                pages.append({"image": im, "era": era, "year": year})
                taken += 1
        print(f"[data] {rid}: {taken} pages", flush=True)
    return pages


# -------------------------------------------------------------------------- main

@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/Gemma-4-31B-it-SRT-Sunstone")
    ap.add_argument("--ckpt-file", default="readout_selected.pt")
    ap.add_argument("--n-per-era", type=int, default=60)
    ap.add_argument("--out-readouts",
                    default="artifacts/nla/coupling/vision_metaprag_readouts.jsonl")
    ap.add_argument("--out-summary",
                    default="artifacts/nla/coupling/vision_metaprag_summary.json")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

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
    print(f"[ckpt] step {ckpt.get('step')} backbone={mid} "
          f"community@{comm_L} mah@{mah_layers}", flush=True)

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

    # ---- backbone -----------------------------------------------------------
    proc = AutoProcessor.from_pretrained(mid)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        mid, dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    img_tok = model.config.image_token_id

    protos = F.normalize(community.prototypes.weight, dim=-1)  # (K, d_community)

    def page_readout(image) -> dict | None:
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": "."}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        img_mask = (enc["input_ids"] == img_tok)  # (1, T)
        idx = img_mask[0].bool()
        n_img = int(idx.sum())
        if n_img < 8:
            return None

        # community: page vector (conditions MAH) + per-token dispersion (U_com proxy)
        co = community(hs[comm_L].float(), attention_mask=img_mask.long())
        cvec = co.vector  # (1, d_community)
        enc_tok = community.encoder(hs[comm_L].float()[0][idx])  # (n_img, d_community)
        enc_n = F.normalize(enc_tok, dim=-1)
        w = F.softmax((enc_n @ protos.T) / community.temperature, dim=-1)  # (n_img, K)
        page_assign = w.mean(0)  # (K,)
        ucom_entropy = float(-(page_assign * (page_assign + 1e-9).log()).sum())
        code_spread = float(enc_n.std(0).mean())

        # MAH divergence pooled over image positions, per layer
        T = int(enc["input_ids"].shape[1])
        causal = torch.full((T, T), float("-inf"), device=dev).triu(1).view(1, 1, T, T)
        d_by_layer = {}
        for i, L in enumerate(mah_layers):
            dv = mah_heads[i](hs[L].float(), community_vec=cvec,
                              causal_mask=causal).divergence  # (1, T, d_div)
            nrm = dv.norm(dim=-1)[0]  # (T,)
            d_by_layer[L] = float(nrm[idx].mean())
        return {
            "n_img_tok": n_img,
            "D": d_by_layer[mah_layers[last_i]],
            **{f"D_L{L}": d_by_layer[L] for L in mah_layers},
            "ucom_entropy": ucom_entropy,
            "code_spread": code_spread,
        }

    # ---- run over pages -----------------------------------------------------
    pages = load_pages(args.n_per_era)
    print(f"[data] total {len(pages)} pages", flush=True)
    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with out_r.open("w") as f:
        for k, pg in enumerate(pages):
            r = page_readout(pg["image"])
            if r is None:
                continue
            r.update({"era": pg["era"], "year": pg["year"]})
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  [{k+1}/{len(pages)}] {pg['era']} y{pg['year']} "
                      f"D={r['D']:.3f} Ucom={r['ucom_entropy']:.3f} "
                      f"nimg={r['n_img_tok']}", flush=True)

    # ---- analysis -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    D = np.array([r["D"] for r in rows], float)
    U = np.array([r["ucom_entropy"] for r in rows], float)
    spread = np.array([r["code_spread"] for r in rows], float)
    ntok = np.array([r["n_img_tok"] for r in rows], float)

    def couple(y_name, yv):
        return {
            "metric": y_name,
            "spearman_D": spearman(D, yv),
            "partial_D_given_ntok": partial_spearman(D, yv, ntok),
            "perm_p": perm_p_spearman(D, yv, args.nperm, rng),
        }

    coupling = {
        "D_vs_ucom_entropy": couple("ucom_entropy", U),
        "D_vs_code_spread": couple("code_spread", spread),
        "D_vs_ntok(control)": {"spearman_D": spearman(D, ntok)},
        "ucom_vs_ntok(control)": {"spearman": spearman(U, ntok)},
    }
    per_layer = {f"D_L{L}_vs_ucom": {
        "spearman": spearman(np.array([r[f"D_L{L}"] for r in rows]), U),
        "partial_given_ntok": partial_spearman(
            np.array([r[f"D_L{L}"] for r in rows]), U, ntok),
    } for L in mah_layers}

    eras = sorted(set(r["era"] for r in rows))
    era_means = {e: {
        "n": int(sum(1 for r in rows if r["era"] == e)),
        "D_mean": float(np.mean([r["D"] for r in rows if r["era"] == e])),
        "ucom_mean": float(np.mean([r["ucom_entropy"] for r in rows if r["era"] == e])),
        "ntok_mean": float(np.mean([r["n_img_tok"] for r in rows if r["era"] == e])),
    } for e in eras}

    summary = {
        "backbone": mid, "repo": args.repo, "step": ckpt.get("step"),
        "n_pages": len(rows), "mah_layers": mah_layers,
        "hypothesis": "per-page MAH divergence D tracks community dispersion "
                      "(U_com proxy) on newspaper page images (modality invariance)",
        "coupling": coupling,
        "per_layer": per_layer,
        "era_means": era_means,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print("\n=== Cross-modal metapragmatic load (gemma-4 vision, newspaper pages) ===")
    c = coupling["D_vs_ucom_entropy"]
    print(f"D ~ U_com(entropy):  rho={c['spearman_D']:+.3f}  "
          f"partial|ntok={c['partial_D_given_ntok']:+.3f}  p={c['perm_p']:.4g}")
    cs = coupling["D_vs_code_spread"]
    print(f"D ~ code_spread:     rho={cs['spearman_D']:+.3f}  "
          f"partial|ntok={cs['partial_D_given_ntok']:+.3f}  p={cs['perm_p']:.4g}")
    print(f"controls: D~ntok={coupling['D_vs_ntok(control)']['spearman_D']:+.3f}  "
          f"Ucom~ntok={coupling['ucom_vs_ntok(control)']['spearman']:+.3f}")
    for e, m in era_means.items():
        print(f"  era {e:<10} n={m['n']:<3} D={m['D_mean']:.3f} "
              f"Ucom={m['ucom_mean']:.3f} ntok={m['ntok_mean']:.0f}")
    print(f"\nwrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
