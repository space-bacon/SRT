#!/usr/bin/env python3
"""Test 4 (diachronic, concept-anchored): does a sign's metapragmatic load D
rise over historical time as it becomes more contested?

The conjecture is about SIGNS, not passages. The whole-passage community
dispersion proxy failed because it measures topical breadth (how many
communities a page's content spans), not the sign-level contestedness U_com
that Test 1/2 established drives D. So here we hold the SIGN fixed and vary the
era: for a fixed set of contested signs (liberty, justice, slavery, race, ...)
and stable concrete controls (river, harvest, horse, ...), we find each sign in
real dated AmericanStories articles (1770-1964) and measure D at the sign's
own token positions in its historical context.

Prediction: contested signs show rising D over time (their community
underdetermination grew as usage fragmented across discourse communities);
concrete controls stay flat. The control group absorbs era-general confounds
(OCR quality, length, generic language drift), so the difference-in-differences
(contested drift minus control drift) isolates the U_com-specific diachronic
signal.

D is the MAH divergence L2 norm (last layer L45), mean-pooled over the sign's
subword tokens at its first occurrence in the article. Each article is
forwarded once; every matched sign in it is read off the same hidden states.

Usage (box, transformers>=5):
    HF_HOME=/workspace/.hf_home PYTHONPATH=. /venv/main/bin/python \
        scripts/diachronic_concept_gemma4.py --scan-per-year 400 --cap 15 \
        --out-readouts artifacts/nla/coupling/diachronic_concept_readouts.jsonl \
        --out-summary  artifacts/nla/coupling/diachronic_concept_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, get_token, hf_hub_url

DATASET = "dell-research-harvard/AmericanStories"

# Era-appropriate signs present across 1770-1964. Contested = abstract,
# politically/morally loaded, plausibly drifting in community-contestedness.
CONTESTED = [
    "liberty", "freedom", "justice", "rights", "equality", "tyranny", "virtue",
    "slavery", "union", "democracy", "republic", "property", "religion", "race",
    "sovereignty", "treason", "patriotism", "nation", "authority", "reason",
]
# Control = concrete, mundane, stable denotation across the whole range.
CONTROL = [
    "river", "harvest", "weather", "horse", "cattle", "wheat", "timber", "bread",
    "water", "road", "bridge", "market", "rain", "winter", "morning", "harbor",
    "ship", "field", "garden", "iron",
]


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
    if len(x) < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def perm_p_diff(ya, yb, xa, xb, nperm, rng) -> tuple[float, float]:
    """Permutation p for (spearman(xa,ya) - spearman(xb,yb)); shuffle group labels."""
    obs = spearman(xa, ya) - spearman(xb, yb)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    na = len(xa)
    c = 0
    for _ in range(nperm):
        idx = rng.permutation(len(x))
        ia, ib = idx[:na], idx[na:]
        s = spearman(x[ia], y[ia]) - spearman(x[ib], y[ib])
        if abs(s) >= abs(obs) - 1e-12:
            c += 1
    return float(obs), (c + 1) / (nperm + 1)


# --------------------------------------------------------------------- streaming

def stream_year_articles(year, token, scan_per_year, min_chars, max_chars):
    import requests
    url = hf_hub_url(DATASET, f"faro_{year}.tar.gz", repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    n = 0
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            if r.status_code != 200:
                return
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for member in tf:
                    if n >= scan_per_year:
                        break
                    if not member.name.endswith(".json"):
                        continue
                    try:
                        d = json.loads(tf.extractfile(member).read())
                    except Exception:
                        continue
                    for art in d.get("full articles", []):
                        if n >= scan_per_year:
                            break
                        txt = re.sub(r"\s+", " ", (art.get("article") or "").strip())
                        if len(txt) >= min_chars:
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
    ap.add_argument("--scan-per-year", type=int, default=400)
    ap.add_argument("--cap", type=int, default=15,
                    help="max article hits per (concept, decade)")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--max-seq-len", type=int, default=160)
    ap.add_argument("--year-start", type=int, default=1770)
    ap.add_argument("--year-end", type=int, default=1964)
    ap.add_argument("--out-readouts",
                    default="artifacts/nla/coupling/diachronic_concept_readouts.jsonl")
    ap.add_argument("--out-summary",
                    default="artifacts/nla/coupling/diachronic_concept_summary.json")
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

    ckpt = torch.load(hf_hub_download(args.repo, args.ckpt_file),
                      map_location="cpu", weights_only=False)
    cc = ckpt["config"]
    comm_L = int(cc["community_layer"])
    mah_layers = [int(x) for x in cc["mah_layers"]]
    d_backbone = int(cc["d_backbone"])
    d_community = int(cc["d_community"])
    mid = cc["backbone"]
    heads = ckpt["heads"]
    last_L = mah_layers[-1]
    print(f"[ckpt] step {ckpt.get('step')} backbone={mid} mah@{mah_layers}", flush=True)

    community = CommunityDiscoveryHead(CommunityConfig(d_community=d_community),
                                       d_backbone).to(dev).eval()
    community.load_state_dict(
        {k[len("0."):]: v for k, v in heads.items() if k.startswith("0.")})
    community.float()
    mah_heads = []
    for i in range(len(mah_layers)):
        h = MetapragmaticAttentionHead(MAHConfig(), d_backbone, d_community=d_community)
        sd = {k[len(f"1.{i}."):]: v for k, v in heads.items() if k.startswith(f"1.{i}.")}
        h.load_state_dict(sd, strict=False)
        mah_heads.append(h.to(dev).eval().float())
    print(f"[heads] community + {len(mah_heads)} MAH heads loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Gemma4ForConditionalGeneration.from_pretrained(
        mid, dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    concepts = {c: 1 for c in CONTESTED}
    concepts.update({c: 0 for c in CONTROL})
    patt = {c: re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE) for c in concepts}

    def article_concept_D(text: str, want: list[str]) -> dict:
        """Forward the article once; return {concept: D_mean} for wanted concepts
        whose token span is found."""
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc.input_ids.to(dev)
        am = enc.attention_mask.to(dev)
        T = int(ids.shape[1])
        if T < 8:
            return {}
        hs = model(input_ids=ids, attention_mask=am,
                   output_hidden_states=True, use_cache=False).hidden_states
        cvec = community(hs[comm_L].float(), attention_mask=am).vector
        causal = torch.full((T, T), float("-inf"), device=dev).triu(1).view(1, 1, T, T)
        dv = mah_heads[-1](hs[last_L].float(), community_vec=cvec,
                           causal_mask=causal).divergence
        nrm = dv.norm(dim=-1)[0].float().cpu().numpy()  # (T,)

        out = {}
        low = text.lower()
        for c in want:
            m = patt[c].search(text)
            if not m:
                continue
            cs, ce = m.start(), m.end()
            toks = [i for i, (a, b) in enumerate(offsets) if b > cs and a < ce and b > a]
            if not toks:
                continue
            out[c] = {"D": float(np.mean([nrm[i] for i in toks])),
                      "D_last": float(nrm[toks[-1]]),
                      "concept_ntok": len(toks), "n_tokens": T}
        return out

    # ---- available years ----------------------------------------------------
    api = HfApi()
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)
    yrs = []
    for s in info.siblings:
        m = re.match(r"faro_(\d{4})\.tar\.gz", s.rfilename)
        if m and args.year_start <= int(m.group(1)) <= args.year_end and (s.size or 0) > 1000:
            yrs.append(int(m.group(1)))
    years = sorted(yrs)
    print(f"[years] {len(years)} available", flush=True)

    token = get_token()
    cap_count: dict = {}   # (concept, decade) -> n
    rows: list[dict] = []
    out_r = Path(args.out_readouts)
    out_r.parent.mkdir(parents=True, exist_ok=True)

    with out_r.open("w") as f:
        for yi, year in enumerate(years):
            dec = (year // 10) * 10
            got = 0
            for txt in stream_year_articles(year, token, args.scan_per_year,
                                            args.min_chars, args.max_chars):
                low = txt.lower()
                want = [c for c in concepts
                        if (c in low) and cap_count.get((c, dec), 0) < args.cap
                        and patt[c].search(txt)]
                if not want:
                    continue
                res = article_concept_D(txt, want)
                for c, r in res.items():
                    if cap_count.get((c, dec), 0) >= args.cap:
                        continue
                    row = {"concept": c, "contested": concepts[c], "year": year,
                           "decade": dec, **r}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    cap_count[(c, dec)] = cap_count.get((c, dec), 0) + 1
                    got += 1
            f.flush()
            if yi % 5 == 0 or got:
                print(f"  [{yi+1}/{len(years)}] {year}: +{got} (total {len(rows)})",
                      flush=True)

    print(f"[done] {len(rows)} concept observations", flush=True)

    # ---- analysis -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    yr = np.array([r["year"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    con = np.array([r["contested"] for r in rows])
    mc = con == 1
    mk = con == 0

    contested_drift = spearman(yr[mc], D[mc])
    control_drift = spearman(yr[mk], D[mk])
    did, did_p = perm_p_diff(D[mc], D[mk], yr[mc], yr[mk], args.nperm, rng)

    # per-concept within-sign drift
    per_concept = {}
    for c in concepts:
        idx = [i for i, r in enumerate(rows) if r["concept"] == c]
        if len(idx) >= 8:
            per_concept[c] = {"contested": concepts[c], "n": len(idx),
                              "year_vs_D": spearman(yr[idx], D[idx]),
                              "D_mean": float(D[idx].mean())}

    # decade means by group
    decs = sorted(set(int(r["decade"]) for r in rows))
    decade_group = {}
    for dd in decs:
        gi_c = [i for i, r in enumerate(rows) if r["decade"] == dd and r["contested"] == 1]
        gi_k = [i for i, r in enumerate(rows) if r["decade"] == dd and r["contested"] == 0]
        decade_group[dd] = {
            "contested_n": len(gi_c),
            "contested_D": float(D[gi_c].mean()) if gi_c else None,
            "control_n": len(gi_k),
            "control_D": float(D[gi_k].mean()) if gi_k else None,
        }

    summary = {
        "backbone": mid, "dataset": DATASET, "n_obs": len(rows),
        "n_contested_obs": int(mc.sum()), "n_control_obs": int(mk.sum()),
        "year_range": [int(yr.min()), int(yr.max())],
        "contested_signs": CONTESTED, "control_signs": CONTROL,
        "diachronic": {
            "contested_year_vs_D": contested_drift,
            "control_year_vs_D": control_drift,
            "difference_in_differences": did,
            "did_perm_p": did_p,
        },
        "per_concept_drift": per_concept,
        "decade_means_by_group": decade_group,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print("\n=== Test 4 (concept-anchored diachronic): D drift of contested vs control signs ===")
    print(f"n={len(rows)} obs ({int(mc.sum())} contested / {int(mk.sum())} control) "
          f"[{int(yr.min())}-{int(yr.max())}]")
    print(f"  contested  year~D = {contested_drift:+.3f}")
    print(f"  control    year~D = {control_drift:+.3f}")
    print(f"  diff-in-diff      = {did:+.3f}  (perm p={did_p:.4g})")
    print("decade means (contested D / control D):")
    for dd, g in decade_group.items():
        cD = f"{g['contested_D']:.3f}" if g["contested_D"] is not None else "  -  "
        kD = f"{g['control_D']:.3f}" if g["control_D"] is not None else "  -  "
        print(f"  {dd}s  contested={cD}(n{g['contested_n']:<3}) control={kD}(n{g['control_n']})")
    top = sorted(per_concept.items(), key=lambda kv: -kv[1]["year_vs_D"])
    print("top rising signs (year~D):")
    for c, v in top[:6]:
        print(f"  {c:<12} {'CON' if v['contested'] else 'ctl'} "
              f"year~D={v['year_vs_D']:+.3f} n={v['n']}")
    print(f"\nwrote {out_r}\nwrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
