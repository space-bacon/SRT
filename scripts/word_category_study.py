"""Word-category signal-dissociation study for the SRT adapter on gpt-oss-20b.

Replicates (and extends) an external contributor's finding that the seven SRT
signals separate word categories, on our own controlled word list. Adds two
things the original lacked:

  1. A FREQUENCY-MATCHED control ANOVA (common-tier words only), since one-word
     prompts confound category with word frequency / tokenization length, which
     drive entropy/nll directly.
  2. A DISCRETE companion analysis: per-layer mutual information between word
     category and the VQ codebook code, i.e. do the five categories occupy
     distinct state basins at the form layers (L6/L12) vs the meaning layers
     (L18/L24)?

For each word we build a bare prompt (optional BOS), generate a short
continuation, and collect the seven per-token signals from the adapter's
`generate(return_signals=True)` path:
    backbone:  entropy (ent), margin, nll
    SRT:       r_hat, regime_super, div_norm, chain
We also run one forward pass per word to read hidden states at L6/L12/L18/L24
and assign a codebook code per layer (for the basin analysis).

Signals are aggregated per category across all generated tokens, then:
  - one-way ANOVA (F, p) per signal, all words
  - one-way ANOVA per signal, common-tier only (frequency control)
  - category means table
  - per-layer MI(category; code) + top basins per category

Output: artifacts/nla/gptoss20b/word_category_study.json

Usage (full ~40-45 min on one RTX PRO 6000):
    python scripts/word_category_study.py \
        --adapter checkpoints/gptoss20b_phaseAB_v2/best_adapter.pt \
        --words data/word_categories.json \
        --out artifacts/nla/gptoss20b/word_category_study.json

Smoke test (a few words, few tokens):
    python scripts/word_category_study.py ... --max-words 3 --max-new 16
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

SIGNALS = ["r_hat", "regime_super", "div_norm", "chain", "ent", "margin", "nll"]
GEN_KEYMAP = {"ent": "ent", "margin": "margin", "nll": "nll",
              "r_hat": "r_hat", "regime_super": "regime_super",
              "div_norm": "div_norm", "chain": "chain"}
LAYERS = (6, 12, 18, 24)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--words", default="data/word_categories.json")
    p.add_argument("--backbone", default="openai/gpt-oss-20b")
    p.add_argument("--out", default="artifacts/nla/gptoss20b/word_category_study.json")
    p.add_argument("--codebook-repo", default="RiverRider/srt-nla-gptoss20b-artifacts")
    p.add_argument("--max-new", type=int, default=32,
                   help="tokens to generate per word (signal samples per word)")
    p.add_argument("--max-words", type=int, default=0,
                   help="cap words per category (0 = all) for smoke tests")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-bos", action="store_true",
                   help="do not prepend BOS to the bare word")
    p.add_argument("--no-basin", action="store_true",
                   help="skip the codebook basin companion analysis")
    return p.parse_args()


def find_backbone(model: SRTAdapter):
    """The SRTAdapter wraps a frozen backbone; discover the HF model robustly."""
    for attr in ("backbone", "model", "base_model", "lm", "frozen_backbone"):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "config") and hasattr(obj, "forward"):
            return obj
    # last resort: search modules for something with hidden_states support
    for m in model.modules():
        if hasattr(m, "config") and getattr(m.config, "num_hidden_layers", None):
            return m
    raise RuntimeError("could not locate wrapped backbone on SRTAdapter")


# ---------------- ANOVA (one-way, no scipy dependency) ----------------
def one_way_anova(groups: list[list[float]]) -> tuple[float, float, int]:
    groups = [g for g in groups if len(g) > 1]
    k = len(groups)
    if k < 2:
        return float("nan"), float("nan"), 0
    n = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / n
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    df_b, df_w = k - 1, n - k
    if df_w <= 0 or ss_within <= 0:
        return float("inf"), 0.0, n
    F = (ss_between / df_b) / (ss_within / df_w)
    p = _f_sf(F, df_b, df_w)
    return F, p, n


def _f_sf(F: float, d1: int, d2: int) -> float:
    """Survival function of the F distribution via the regularized incomplete beta."""
    if F <= 0:
        return 1.0
    x = d2 / (d2 + d1 * F)
    return _betai(d2 / 2.0, d1 / 2.0, x)


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c
        if abs(1.0 - d * c) < 1e-8:
            break
    return front * (f - 1.0)


def mutual_information(cats: list[str], codes: list[int]) -> float:
    n = len(cats)
    if n == 0:
        return 0.0
    pc = Counter(cats)
    pk = Counter(codes)
    pjk = Counter(zip(cats, codes))
    mi = 0.0
    for (c, k), njk in pjk.items():
        pjoint = njk / n
        mi += pjoint * math.log(pjoint / ((pc[c] / n) * (pk[k] / n)) + 1e-30)
    return mi / math.log(2)  # bits


def main() -> None:
    args = parse_args()
    data = json.load(open(args.words))
    categories = data["_meta"]["categories"]

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype="bfloat16")
    model = SRTAdapter(cfg).to("cuda", dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    model.load_adapter(args.adapter)
    model.eval()
    bb = find_backbone(model)
    print("backbone:", type(bb).__name__, "L", bb.config.num_hidden_layers, flush=True)

    # codebook + per-layer mu for the basin analysis
    cb = mu_by = None
    if not args.no_basin:
        from huggingface_hub import hf_hub_download
        cbo = torch.load(hf_hub_download(args.codebook_repo, "state_codebook_vq.pt",
                                         repo_type="dataset"),
                         map_location="cuda", weights_only=False)
        cb_mu, cb_cent = cbo["mu"].float().cuda(), cbo["centroids"].float().cuda()
        mu_by = torch.load(hf_hub_download(args.codebook_repo, "mu_by_layer.pt",
                                           repo_type="dataset"),
                           map_location="cuda", weights_only=False)
        mu_by = {int(k): v.float().cuda() for k, v in mu_by.items()}

        def encode(v: torch.Tensor) -> int:
            return int(torch.cdist((v - cb_mu).unsqueeze(0), cb_cent).argmin())
        cb = encode

    bos = [] if args.no_bos else [tok.bos_token_id]
    per_signal: dict[str, dict[str, list[float]]] = {
        s: defaultdict(list) for s in SIGNALS}
    per_signal_common: dict[str, dict[str, list[float]]] = {
        s: defaultdict(list) for s in SIGNALS}
    basin_cats: list[str] = []
    basin_codes: dict[int, list[int]] = {L: [] for L in LAYERS}
    word_means = []  # per-word row for the released table

    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for cat in categories:
        words = data[cat]
        if args.max_words:
            words = words[:args.max_words]
        for entry in words:
            w, tier = entry["w"], entry["tier"]
            ids = torch.tensor([bos + tok.encode(w, add_special_tokens=False)],
                               device="cuda")
            with torch.no_grad():
                _, sigs = model.generate(
                    ids, max_new_tokens=args.max_new, eos_token_ids=eos,
                    temperature=args.temperature, return_signals=True)
            row = {"word": w, "cat": cat, "tier": tier, "n": len(sigs)}
            for s in SIGNALS:
                vals = [d[GEN_KEYMAP[s]] for d in sigs if GEN_KEYMAP[s] in d]
                if not vals:
                    continue
                per_signal[s][cat].extend(vals)
                if tier == "common":
                    per_signal_common[s][cat].extend(vals)
                row[s] = sum(vals) / len(vals)
            word_means.append(row)

            if cb is not None:
                with torch.no_grad():
                    out = bb(ids, output_hidden_states=True, use_cache=False)
                for L in LAYERS:
                    h = out.hidden_states[L][0, -1].float()
                    # codebook.encode subtracts the global pool mu internally;
                    # pass the raw last-token hidden state at this layer.
                    basin_codes[L].append(cb(h))
                basin_cats.append(cat)
        print(f"done {cat} ({len(words)} words)", flush=True)

    # ---------------- ANOVA ----------------
    def anova_table(store):
        out = {}
        for s in SIGNALS:
            groups = [store[s][c] for c in categories if store[s][c]]
            F, p, n = one_way_anova(groups)
            out[s] = {"F": F, "p": p, "n": n}
        return out

    means = {s: {c: (sum(per_signal[s][c]) / len(per_signal[s][c])
                     if per_signal[s][c] else None)
                 for c in categories} for s in SIGNALS}

    result = {
        "backbone": args.backbone, "adapter": args.adapter,
        "max_new": args.max_new, "categories": categories,
        "anova_all": anova_table(per_signal),
        "anova_common_tier": anova_table(per_signal_common),
        "category_means": means,
        "n_tokens_per_cat": {c: len(per_signal["ent"][c]) for c in categories},
        "word_means": word_means,
    }

    if cb is not None:
        basin = {}
        for L in LAYERS:
            codes = basin_codes[L]
            mi = mutual_information(basin_cats, codes)
            # top basin per category
            per_cat_top = {}
            for c in categories:
                cc = Counter(k for cat_, k in zip(basin_cats, codes) if cat_ == c)
                per_cat_top[c] = cc.most_common(3)
            basin[str(L)] = {"mi_bits": mi, "n_codes": len(set(codes)),
                             "top_codes_per_cat": per_cat_top}
        result["basin_analysis"] = basin

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # console summary
    print("\n=== ANOVA (all words) ===")
    for s in SIGNALS:
        a = result["anova_all"][s]
        print(f"  {s:13s} F={a['F']:8.2f}  p={a['p']:.2e}  n={a['n']}")
    print("\n=== ANOVA (common tier only, frequency control) ===")
    for s in SIGNALS:
        a = result["anova_common_tier"][s]
        print(f"  {s:13s} F={a['F']:8.2f}  p={a['p']:.2e}  n={a['n']}")
    print("\n=== category means ===")
    hdr = "  " + "signal".ljust(13) + "".join(c[:10].rjust(12) for c in categories)
    print(hdr)
    for s in SIGNALS:
        row = "  " + s.ljust(13)
        for c in categories:
            v = means[s][c]
            row += (f"{v:12.3f}" if v is not None else " " * 12)
        print(row)
    if cb is not None:
        print("\n=== basin MI(category; code) per layer ===")
        for L in LAYERS:
            print(f"  L{L}: {result['basin_analysis'][str(L)]['mi_bits']:.3f} bits")


if __name__ == "__main__":
    main()
