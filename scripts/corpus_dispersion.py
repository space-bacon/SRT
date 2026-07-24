#!/usr/bin/env python3
"""Corpus-grounded contestedness probe.

Contestedness is a distributional property: a word is contested when the same
token is used to mean genuinely different things across real usages. A single
forward pass cannot see that. So gather many occurrences of each word from the
AmericanStories newspaper corpus, embed each occurrence *in context* (the word's
own contextual hidden state), and measure how much that cloud of embeddings
splinters (dispersion).

Four classes, on purpose:
  contested   - freedom, justice, ...   (disputed meaning)
  polysemous  - bank, spring, iron, ... (many senses, but nobody fights)
  concrete    - river, horse, ...       (one stable referent)
  abstract    - number, distance, ...   (abstract but uncontested)

The key question is not "contested > concrete" (too easy, and could be mere
polysemy) but "contested > polysemous": does the contestedness signal survive
after ordinary sense-multiplicity is controlled for?

    /venv/main/bin/python scripts/corpus_dispersion.py --per-word 90
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_url
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET = "dell-research-harvard/AmericanStories"

TARGETS = {
    # clean contested: dominant newspaper sense IS the value-laden/political one
    "freedom": "contested", "liberty": "contested", "justice": "contested",
    "rights": "contested", "democracy": "contested", "equality": "contested",
    "morality": "contested", "patriotism": "contested", "religion": "contested",
    "sovereignty": "contested",
    # mixed: contested concept but dominant CORPUS sense is uncontested
    # (race=competition, property=real estate, virtue='by virtue of', progress='in progress')
    "race": "mixed", "property": "mixed", "nation": "mixed", "progress": "mixed",
    "labor": "mixed", "virtue": "mixed", "character": "mixed", "union": "mixed",
    # polysemous controls (many senses, not fought over)
    "bank": "polysemous", "spring": "polysemous", "plant": "polysemous",
    "iron": "polysemous", "seal": "polysemous", "ring": "polysemous",
    "saw": "polysemous", "match": "polysemous",
    # concrete controls
    "river": "concrete", "horse": "concrete", "cotton": "concrete",
    "railroad": "concrete", "mountain": "concrete", "bridge": "concrete",
    "wagon": "concrete", "engine": "concrete", "coal": "concrete", "timber": "concrete",
    # abstract uncontested controls
    "number": "abstract", "distance": "abstract", "weather": "abstract",
    "winter": "abstract", "temperature": "abstract", "height": "abstract",
    "weight": "abstract", "speed": "abstract", "length": "abstract",
}

WORD_RE = {w: re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in TARGETS}
SENT_RE = re.compile(r"[^.!?]*[.!?]")


def stream_year_articles(year, scan, min_chars=300, max_chars=4000):
    import requests
    url = hf_hub_url(DATASET, f"faro_{year}.tar.gz", repo_type="dataset")
    tok_env = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok_env}"} if tok_env else {}
    n = 0
    try:
        with requests.get(url, headers=headers, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"[year {year}] http {r.status_code}", flush=True)
                return
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for m in tf:
                    if n >= scan:
                        break
                    if not m.name.endswith(".json"):
                        continue
                    try:
                        d = json.loads(tf.extractfile(m).read())
                    except Exception:
                        continue
                    for art in d.get("full articles", []):
                        if n >= scan:
                            break
                        txt = re.sub(r"\s+", " ", (art.get("article") or "").strip())
                        if len(txt) >= min_chars:
                            yield txt[:max_chars]
                            n += 1
    except Exception as e:
        print(f"[year {year}] stream error: {str(e)[:80]}", flush=True)


def sentences(text):
    return [s.strip() for s in SENT_RE.findall(text) if 20 <= len(s.strip()) <= 320]


ERAS = {
    "1850-66": [1850, 1858, 1866],
    "1874-90": [1874, 1882, 1890],
    "1898-1914": [1898, 1906, 1914],
    "1922-54": [1922, 1938, 1954],
}


def run_diachronic(args):
    import random

    occ = {era: {w: [] for w in TARGETS} for era in ERAS}
    for era, yrs in ERAS.items():
        for y in yrs:
            if all(len(occ[era][w]) >= args.per_era for w in TARGETS):
                break
            for txt in stream_year_articles(y, args.scan_per_year):
                got = set()
                for s in sentences(txt):
                    sl = s.lower()
                    for w in TARGETS:
                        if w in got or len(occ[era][w]) >= args.per_era or w not in sl:
                            continue
                        if WORD_RE[w].search(s):
                            occ[era][w].append(s)
                            got.add(w)          # one sentence per article per word
                if all(len(occ[era][w]) >= args.per_era for w in TARGETS):
                    break
            print(f"[{era} y{y}] " + " ".join(f"{w}={len(occ[era][w])}" for w in TARGETS), flush=True)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16, output_hidden_states=True, device_map="cuda"
    ).eval()

    @torch.no_grad()
    def word_vec(sentence, word):
        enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=96)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        hs = model(**enc).hidden_states[args.layer][0]
        m = WORD_RE[word].search(sentence)
        idxs = [i for i, (a, b) in enumerate(offs) if b > a and a < m.end() and b > m.start()]
        return hs[idxs].float().mean(0).cpu() if idxs else None

    vecs = {era: {w: [] for w in TARGETS} for era in ERAS}
    for era in ERAS:
        for w in TARGETS:
            for s in occ[era][w]:
                v = word_vec(s, w)
                if v is not None:
                    vecs[era][w].append(v)

    flat = [v for era in ERAS for w in TARGETS for v in vecs[era][w]]
    owner = [w for era in ERAS for w in TARGETS for _ in range(len(vecs[era][w]))]
    alln = torch.stack(flat)
    alln = alln / alln.norm(dim=1, keepdim=True).clamp_min(1e-8)
    random.seed(0)
    N = alln.shape[0]
    cross, tries = [], 0
    while len(cross) < 8000 and tries < 60000:
        i, j = random.randrange(N), random.randrange(N)
        tries += 1
        if owner[i] != owner[j]:
            cross.append(float(alln[i] @ alln[j]))
    baseline = sum(cross) / len(cross)

    def disp(vs):
        if len(vs) < 30:
            return None
        X = torch.stack(vs)
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
        n = X.shape[0]
        G = X @ X.T
        iu = torch.triu_indices(n, n, offset=1)
        return (1.0 - float(G[iu[0], iu[1]].mean())) / (1.0 - baseline + 1e-8)

    eras = list(ERAS.keys())
    ei = {e: i for i, e in enumerate(eras)}

    def slope(pts):
        n = len(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    print(f"\nanisotropy baseline = {baseline:.3f}\n")
    print(f"{'word':13s} {'class':11s}" + "".join(f"{e:>10s}" for e in eras) + "    slope")
    print("-" * 80)
    rows = []
    for w in TARGETS:
        ds = [disp(vecs[e][w]) for e in eras]
        pts = [(ei[e], d) for e, d in zip(eras, ds) if d is not None]
        cells = "".join((f"{d:10.3f}" if d is not None else f"{'-':>10s}") for d in ds)
        if len(pts) < 3:
            print(f"{w:13s} {TARGETS[w]:11s}{cells}    (thin)")
            continue
        s = slope(pts)
        rows.append((w, TARGETS[w], s))
        print(f"{w:13s} {TARGETS[w]:11s}{cells}   {s:+.4f}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    by = defaultdict(list)
    for w, c, s in rows:
        by[c].append(s)
    print("\nmean dispersion slope per class:")
    for c in ["contested", "mixed", "polysemous", "concrete", "abstract"]:
        if by[c]:
            print(f"  {c:11s} {mean(by[c]):+.4f}  (n={len(by[c])})")

    test_rows = [(w, c, s) for (w, c, s) in rows if c != "mixed"]
    labels = [1 if c == "contested" else 0 for _, c, _ in test_rows]
    slopes = [s for _, _, s in test_rows]

    def did(lab):
        con = [s for s, l in zip(slopes, lab) if l == 1]
        oth = [s for s, l in zip(slopes, lab) if l == 0]
        return mean(con) - mean(oth)

    obs = did(labels)
    random.seed(0)
    P, cnt = 10000, 0
    for _ in range(P):
        perm = labels[:]
        random.shuffle(perm)
        if abs(did(perm)) >= abs(obs):
            cnt += 1
    pval = (cnt + 1) / (P + 1)
    print(f"\nDiD (contested - rest) slope = {obs:+.4f}   permutation p = {pval:.4f}"
          f"   (n_contested={sum(labels)}, n_rest={len(labels) - sum(labels)})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outp = args.out.replace(".json", "_diachronic.json")
    with open(outp, "w") as f:
        json.dump({"baseline": baseline, "eras": eras, "did": obs, "p": pval,
                   "rows": [{"word": w, "cls": c, "slope": s} for w, c, s in rows]}, f, indent=2)
    print("\nwrote", outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1880,1900,1920,1940")
    ap.add_argument("--scan-per-year", type=int, default=1500)
    ap.add_argument("--per-word", type=int, default=90)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", default="artifacts/nla/corpus_dispersion.json")
    ap.add_argument("--diachronic", action="store_true", help="per-era dispersion rise, DiD contested vs controls")
    ap.add_argument("--per-era", type=int, default=55)
    args = ap.parse_args()
    if args.diachronic:
        return run_diachronic(args)
    years = [int(y) for y in args.years.split(",")]

    occ = {w: [] for w in TARGETS}

    def need():
        return any(len(v) < args.per_word for v in occ.values())

    for y in years:
        if not need():
            break
        for txt in stream_year_articles(y, args.scan_per_year):
            for s in sentences(txt):
                sl = s.lower()
                for w in TARGETS:
                    if len(occ[w]) >= args.per_word or w not in sl:
                        continue
                    if WORD_RE[w].search(s):
                        occ[w].append((y, s))
            if not need():
                break
        print(f"[year {y}] " + " ".join(f"{w}={len(occ[w])}" for w in TARGETS), flush=True)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16, output_hidden_states=True, device_map="cuda"
    ).eval()

    @torch.no_grad()
    def word_vec(sentence, word):
        enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=96)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        hs = model(**enc).hidden_states[args.layer][0]
        m = WORD_RE[word].search(sentence)
        lo, hi = m.start(), m.end()
        idxs = [i for i, (a, b) in enumerate(offs) if b > a and a < hi and b > lo]
        if not idxs:
            return None
        return hs[idxs].float().mean(0).cpu()

    vecs = {w: [] for w in TARGETS}
    for w in TARGETS:
        for (_, s) in occ[w]:
            v = word_vec(s, w)
            if v is not None:
                vecs[w].append(v)

    import random

    normed = {}
    for w in TARGETS:
        if len(vecs[w]) >= 12:
            X = torch.stack(vecs[w])
            normed[w] = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)

    alln = torch.cat(list(normed.values()))
    owner = [w for w in normed for _ in range(normed[w].shape[0])]
    N = alln.shape[0]
    random.seed(0)
    cross, tries = [], 0
    while len(cross) < 8000 and tries < 60000:
        i, j = random.randrange(N), random.randrange(N)
        tries += 1
        if owner[i] != owner[j]:
            cross.append(float(alln[i] @ alln[j]))
    baseline = sum(cross) / len(cross)          # model anisotropy floor

    rows = []
    for w, Xn in normed.items():
        n = Xn.shape[0]
        G = Xn @ Xn.T
        iu = torch.triu_indices(n, n, offset=1)
        selfsim = float(G[iu[0], iu[1]].mean())          # avg cos among own occurrences
        disp = (1.0 - selfsim) / (1.0 - baseline + 1e-8)  # anisotropy-adjusted spread
        rows.append((w, TARGETS[w], n, selfsim, disp))
    rows.sort(key=lambda r: -r[4])

    print(f"\nanisotropy baseline (cross-word cos) = {baseline:.3f}")
    print("\nword         class        n    selfsim   disp")
    print("-" * 48)
    for w, c, n, si, d in rows:
        print(f"{w:11s} {c:11s} {n:4d}   {si:.3f}   {d:.3f}")

    cls = defaultdict(list)
    for w, c, n, si, d in rows:
        cls[c].append(d)
    print("\nper-class mean adjusted dispersion:")
    for c in ["contested", "polysemous", "abstract", "concrete"]:
        if cls[c]:
            print(f"  {c:11s} {sum(cls[c]) / len(cls[c]):.3f}  (n={len(cls[c])})")
    if cls["contested"] and cls["polysemous"]:
        gap = sum(cls["contested"]) / len(cls["contested"]) - sum(cls["polysemous"]) / len(cls["polysemous"])
        print(f"\ncontested - polysemous = {gap:+.3f}  "
              f"(>0 => contested words disperse beyond mere polysemy)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([{"word": w, "cls": c, "n": n, "selfsim": si, "disp": d}
                   for w, c, n, si, d in rows], f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
