"""Polyseme minimal-pair study: CS vs food/ordinary senses, paired design.

Follow-up to an external contributor's bite/nibble domain study (Stage-3
signals d up to 1.49, CS vs food). Their design compared free-form CS
prompts against free-form food prompts, which confounds domain with
length, frequency, and syntax. This study removes the confound by
construction: every prompt pair shares an identical frame and differs in
exactly ONE domain-carrying word, while the polyseme itself (nibble,
chunk, cookie, word, stream, menu) is held fixed in surface form.

Sign convention (stated once, used everywhere):
    d_paired = mean(CS_i - FOOD_i) / sd(CS_i - FOOD_i)
    d > 0  means the signal is HIGHER in the CS-sense context.

For each prompt we generate a short continuation with
``adapter.generate(..., return_signals=True)`` and aggregate the seven
per-token signals (r_hat, regime_super, div_norm, chain, ent, margin,
nll) over generated tokens, exactly as scripts/word_category_study.py
does. Paired stats per signal: d_paired, t (paired), sign-flip
permutation p (exact over 2^n or 20k samples).

Usage (GPU box):
    python scripts/polyseme_minimal_pairs.py \\
        --adapter checkpoints/adapter_v3/best_adapter.pt \\
        --backbone Qwen/Qwen2.5-7B \\
        --out artifacts/probes/polyseme_minimal_pairs.json

List the prompt pairs without loading any model:
    python scripts/polyseme_minimal_pairs.py --list
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random

SIGNALS = ["r_hat", "regime_super", "div_norm", "chain", "ent", "margin", "nll"]

# Each frame contains {polyseme} fixed and a single {swap} slot whose two
# fillers (cs, food) are matched in register and, where possible, length.
FRAMES: list[dict] = [
    # nibble (CS: 4 bits, half a byte)
    {"p": "nibble", "frame": "The first nibble was stored carefully in the {swap}.",
     "cs": "register", "food": "lunchbox"},
    {"p": "nibble", "frame": "Each nibble should be checked against the {swap} before use.",
     "cs": "checksum", "food": "recipe"},
    {"p": "nibble", "frame": "A single nibble from the {swap} was all it took.",
     "cs": "payload", "food": "pastry"},
    {"p": "nibble", "frame": "She logged every nibble that passed through the {swap}.",
     "cs": "decoder", "food": "kitchen"},
    # chunk (CS: memory/data chunk)
    {"p": "chunk", "frame": "The next chunk was loaded into the {swap}.",
     "cs": "buffer", "food": "bowl"},
    {"p": "chunk", "frame": "One chunk at a time came out of the {swap}.",
     "cs": "parser", "food": "oven"},
    {"p": "chunk", "frame": "They split the {swap} into a chunk of manageable size.",
     "cs": "archive", "food": "brisket"},
    {"p": "chunk", "frame": "A corrupted chunk ruined the whole {swap}.",
     "cs": "download", "food": "casserole"},
    # cookie (CS: HTTP cookie)
    {"p": "cookie", "frame": "Every cookie was carefully stored in the {swap}.",
     "cs": "browser", "food": "jar"},
    {"p": "cookie", "frame": "The cookie expired before anyone could {swap} it.",
     "cs": "refresh", "food": "taste"},
    {"p": "cookie", "frame": "He inspected the cookie set by the {swap}.",
     "cs": "server", "food": "bakery"},
    {"p": "cookie", "frame": "Deleting the cookie from the {swap} solved the problem.",
     "cs": "session", "food": "pantry"},
    # word (CS: fixed-width data word)
    {"p": "word", "frame": "The {swap} handled one word at a time.",
     "cs": "processor", "food": "toddler"},
    {"p": "word", "frame": "Each word was aligned neatly in the {swap}.",
     "cs": "memory", "food": "notebook"},
    {"p": "word", "frame": "A missing word halted the entire {swap}.",
     "cs": "pipeline", "food": "lecture"},
    # stream (CS: data stream)
    {"p": "stream", "frame": "They monitored the stream for unexpected {swap}.",
     "cs": "packets", "food": "fish"},
    {"p": "stream", "frame": "The stream dried up when the {swap} failed.",
     "cs": "socket", "food": "spring"},
    {"p": "stream", "frame": "Filtering the stream removed most of the {swap}.",
     "cs": "noise", "food": "silt"},
    # menu (CS: UI menu)
    {"p": "menu", "frame": "The {swap} menu offered too many options.",
     "cs": "dropdown", "food": "dinner"},
    {"p": "menu", "frame": "She navigated the menu looking for the {swap}.",
     "cs": "settings", "food": "dessert"},
]


def build_pairs() -> list[dict]:
    pairs = []
    for i, f in enumerate(FRAMES):
        pairs.append({
            "pair_id": i, "polyseme": f["p"],
            "cs": f["frame"].replace("{swap}", f["cs"]),
            "food": f["frame"].replace("{swap}", f["food"]),
        })
    return pairs


def paired_stats(cs: list[float], food: list[float],
                 n_perm: int = 20000, seed: int = 0) -> dict:
    diffs = [a - b for a, b in zip(cs, food)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else float("nan")
    d_paired = mean_d / sd if sd and sd > 0 else float("nan")
    t = mean_d / (sd / math.sqrt(n)) if sd and sd > 0 else float("nan")
    # sign-flip permutation on the mean difference
    rng = random.Random(seed)
    obs = abs(mean_d)
    if n <= 16:  # exact
        hits, total = 0, 0
        for signs in itertools.product((1, -1), repeat=n):
            m = abs(sum(s * x for s, x in zip(signs, diffs)) / n)
            hits += m >= obs - 1e-12
            total += 1
        p = hits / total
    else:
        hits = 0
        for _ in range(n_perm):
            m = abs(sum(x if rng.random() < 0.5 else -x for x in diffs) / n)
            hits += m >= obs - 1e-12
        p = (hits + 1) / (n_perm + 1)
    return {"n_pairs": n, "mean_cs": sum(cs) / n, "mean_food": sum(food) / n,
            "d_paired": d_paired, "t_paired": t, "p_signflip": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None,
                   help="local .pt adapter checkpoint (overrides --adapter-repo)")
    ap.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0",
                   help="HF repo with config.json + adapter.safetensors "
                        "(the Stage-3 signals the contributor used)")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (deterministic; recommended for paired stats)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true",
                    help="print the prompt pairs and exit (no model)")
    ap.add_argument("--out", default="artifacts/probes/polyseme_minimal_pairs.json")
    args = ap.parse_args()

    pairs = build_pairs()
    if args.list:
        for pr in pairs:
            print(f"[{pr['pair_id']:02d}] ({pr['polyseme']})")
            print(f"   CS  : {pr['cs']}")
            print(f"   FOOD: {pr['food']}")
        print(f"\n{len(pairs)} pairs; convention: d>0 means CS > FOOD")
        return

    import torch
    from transformers import AutoTokenizer
    from srt.adapter import SRTAdapter
    from srt.config import SRTConfig

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.adapter:
        cfg = SRTConfig(backbone_id=args.backbone)
        model = SRTAdapter(cfg).to(device).eval()
        ck = torch.load(args.adapter, map_location=device, weights_only=False)
        state = ck.get("adapter", ck.get("trainable", ck)) if isinstance(ck, dict) else ck
        model.load_state_dict(state, strict=False)
        print(f"loaded adapter {args.adapter}")
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = SRTConfig.from_json(hf_hub_download(args.adapter_repo, "config.json"))
        cfg.backbone_id = args.backbone
        model = SRTAdapter(cfg).to(device).eval()
        sd = load_file(hf_hub_download(args.adapter_repo, "adapter.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded HF adapter {args.adapter_repo} "
              f"({len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)})")
    tok = AutoTokenizer.from_pretrained(args.backbone)

    def collect(prompt: str) -> dict[str, float]:
        enc = tok(prompt, return_tensors="pt").to(device)
        eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
        with torch.no_grad():
            _, sigs = model.generate(
                enc.input_ids, max_new_tokens=args.max_new,
                eos_token_ids=eos, temperature=args.temperature,
                return_signals=True)
        agg = {}
        for s in SIGNALS:
            vals = [float(d[s]) for d in sigs if s in d]
            agg[s] = sum(vals) / len(vals) if vals else float("nan")
        return agg

    rows = []
    for pr in pairs:
        rec = {"pair_id": pr["pair_id"], "polyseme": pr["polyseme"],
               "cs_prompt": pr["cs"], "food_prompt": pr["food"],
               "cs": collect(pr["cs"]), "food": collect(pr["food"])}
        rows.append(rec)
        print(f"[{pr['pair_id']:02d}] {pr['polyseme']:>7}  "
              f"div cs={rec['cs']['div_norm']:.3f} food={rec['food']['div_norm']:.3f}",
              flush=True)

    stats = {}
    for s in SIGNALS:
        cs_vals = [r["cs"][s] for r in rows]
        fd_vals = [r["food"][s] for r in rows]
        stats[s] = paired_stats(cs_vals, fd_vals, seed=args.seed)
    # per-polyseme breakdown for div_norm (the contributor's headline signal)
    by_p = {}
    for p in sorted({r["polyseme"] for r in rows}):
        sub = [r for r in rows if r["polyseme"] == p]
        by_p[p] = paired_stats([r["cs"]["div_norm"] for r in sub],
                               [r["food"]["div_norm"] for r in sub], seed=args.seed)

    result = {
        "convention": "d_paired > 0 means signal HIGHER in CS-sense context; "
                      "d = mean(CS-FOOD)/sd(CS-FOOD) over pairs",
        "backbone": args.backbone, "adapter": args.adapter,
        "max_new": args.max_new, "temperature": args.temperature,
        "n_pairs": len(rows), "stats": stats,
        "div_norm_by_polyseme": by_p, "rows": rows,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"{'signal':>14} {'d_paired':>9} {'t':>8} {'p':>10}")
    for s in SIGNALS:
        st = stats[s]
        print(f"{s:>14} {st['d_paired']:>9.3f} {st['t_paired']:>8.2f} {st['p_signflip']:>10.4g}")


if __name__ == "__main__":
    main()
