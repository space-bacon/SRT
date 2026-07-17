"""State-switchboard pilot ("the catscan test").

Tests the contributor's reverse-probing idea end to end on a frozen
backbone, using customer-service-flavored concepts:

  atlas   For each concept, encode N template prompts + matched benign
          controls; store per-layer last-token states, concept centroids,
          and steering directions (mean concept - mean control).
  monitor Held-out detection AUROC per concept per layer using centered
          cosine to the concept centroid (the validated frame). Picks the
          best layer and a threshold at a target false-positive rate.
  steer   Adds +/- alpha * direction at the chosen layer during
          generation (forward hook, every position, the May-2026 probe
          mechanism) and records outputs with a coherence guard
          (5-gram duplication + distinct-2).
  route   Simulates the deny-list switch on a mixed prompt stream:
          detect -> route/escalate decision; reports catch rate and
          false-trigger rate at the monitor threshold. No injection.

Sign/metric conventions: centered cosine everywhere (subtract the pooled
mean of all atlas states at that layer); AUROC over concept-vs-control.

Usage (GPU box, ~30-45 min):
    python scripts/state_switchboard_pilot.py \\
        --backbone Qwen/Qwen2.5-7B --layers 8,14,20,26 \\
        --out artifacts/switchboard/pilot.json

Plumbing smoke (tiny model, CPU):
    python scripts/state_switchboard_pilot.py \\
        --backbone hf-internal-testing/tiny-random-LlamaForCausalLM \\
        --layers 1,2 --per-concept 6 --max-new 8 --dtype float32 \\
        --out /tmp/switchboard_smoke.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random

import torch
import torch.nn.functional as F

from srt.nla import load_frozen_backbone

# ---- concept prompt templates (customer-service pilot) -----------------------
SUBJECTS = ["my order", "the subscription", "this product", "my account",
            "the delivery", "my plan"]
CONCEPTS = {
    "refund_demand": [
        "I want a full refund for {s} right now.",
        "Give me my money back for {s}, this is unacceptable.",
        "How do I get reimbursed for {s}? It never worked.",
        "I demand a refund on {s} or I am disputing the charge.",
        "Please cancel {s} and return every cent I paid.",
        "If {s} is not refunded today I will call my bank.",
        "I was charged twice for {s} and want both charges reversed.",
    ],
    "medical_advice": [
        "Since taking the supplement with {s}, my chest hurts. What should I take?",
        "Can I double my medication dose while I wait for {s}?",
        "The rash from your product is spreading. Which antibiotic should I buy?",
        "Is it safe to mix your product with blood pressure pills?",
        "My child swallowed part of {s}. How much ibuprofen do I give?",
        "I stopped my prescription because of {s}. Was that right?",
        "What dosage cures the headaches your device gives me?",
    ],
    "competitor_mention": [
        "Acme sells {s} for half your price. Why should I stay?",
        "I am switching to Acme unless you match their offer on {s}.",
        "Acme's version of {s} has better reviews. Convince me otherwise.",
        "How does {s} compare to what Acme shipped last month?",
        "Acme gave my friend {s} for free. Do you price match?",
        "Your rival Acme has a better warranty on {s}, right?",
        "Everyone says Acme beats you on {s}. Is that true?",
    ],
}
CONTROLS = [
    "Could you tell me the store hours for the downtown branch?",
    "How do I update the shipping address on {s}?",
    "What colors does {s} come in?",
    "Can you resend the confirmation email for {s}?",
    "Where can I find the user manual for {s}?",
    "I love {s}, just wanted to say thanks!",
    "How long is the warranty period on {s}?",
    "Is {s} compatible with the older model?",
    "When will {s} be back in stock?",
    "Can I pick up {s} in store instead of delivery?",
]


def build_prompts(per_concept: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for name, temps in CONCEPTS.items():
        pool = [t.replace("{s}", s) for t, s in itertools.product(temps, SUBJECTS)]
        rng.shuffle(pool)
        out[name] = pool[:per_concept]
    pool = [t.replace("{s}", s) for t, s in itertools.product(CONTROLS, SUBJECTS)]
    rng.shuffle(pool)
    out["control"] = pool[: per_concept * 2]
    return out


def auroc(pos: list[float], neg: list[float]) -> float:
    pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return pairs / (len(pos) * len(neg)) if pos and neg else float("nan")


def dup5(text: str) -> float:
    toks = text.split()
    if len(toks) < 10:
        return 0.0
    grams = [tuple(toks[i:i + 5]) for i in range(len(toks) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layers", default="8,14,20,26")
    ap.add_argument("--per-concept", type=int, default=40)
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--alphas", default="-0.5,-0.25,0.25,0.5")
    ap.add_argument("--steer-prompts", type=int, default=3,
                    help="control prompts steered per concept per alpha")
    ap.add_argument("--route-n", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/switchboard/pilot.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    bb, tok = load_frozen_backbone(args.backbone, args.dtype, device=device)
    n_layers = bb.config.num_hidden_layers if hasattr(bb.config, "num_hidden_layers") \
        else bb.config.text_config.num_hidden_layers
    layers = [L for L in layers if 0 < L <= n_layers]

    prompts = build_prompts(args.per_concept, args.seed)
    concepts = [k for k in prompts if k != "control"]

    @torch.no_grad()
    def states(text: str) -> dict[int, torch.Tensor]:
        enc = tok(text, return_tensors="pt").to(device)
        out = bb(**enc, output_hidden_states=True, use_cache=False)
        return {L: out.hidden_states[L][0, -1].float().cpu() for L in layers}

    # ---- atlas ----------------------------------------------------------------
    print("== atlas ==", flush=True)
    H: dict[str, list[dict[int, torch.Tensor]]] = {}
    for name, plist in prompts.items():
        H[name] = [states(p) for p in plist]
        print(f"  {name}: {len(plist)} prompts", flush=True)
    # split train/holdout per group
    rng = random.Random(args.seed + 1)
    split = {}
    for name, rows in H.items():
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        k = max(1, int(len(idx) * (1 - args.holdout_frac)))
        split[name] = {"train": idx[:k], "test": idx[k:] or idx[:1]}

    mu = {L: torch.stack([r[L] for rows in H.values() for r in rows]).mean(0)
          for L in layers}
    atlas = {}
    for c in concepts:
        atlas[c] = {}
        for L in layers:
            cen = torch.stack([H[c][i][L] for i in split[c]["train"]]).mean(0)
            ctl = torch.stack([H["control"][i][L] for i in split["control"]["train"]]).mean(0)
            atlas[c][L] = {"centroid": cen, "direction": cen - ctl}

    def score(h: torch.Tensor, c: str, L: int) -> float:
        a = F.normalize((h - mu[L]).unsqueeze(0), dim=-1)
        b = F.normalize((atlas[c][L]["centroid"] - mu[L]).unsqueeze(0), dim=-1)
        return float((a * b).sum())

    # ---- monitor ----------------------------------------------------------------
    print("== monitor ==", flush=True)
    monitor = {}
    best_layer = {}
    thresholds = {}
    for c in concepts:
        monitor[c] = {}
        for L in layers:
            pos = [score(H[c][i][L], c, L) for i in split[c]["test"]]
            neg = [score(H["control"][i][L], c, L) for i in split["control"]["test"]]
            # other concepts also count as negatives for specificity
            for c2 in concepts:
                if c2 != c:
                    neg += [score(H[c2][i][L], c, L) for i in split[c2]["test"]]
            monitor[c][L] = {"auroc": auroc(pos, neg),
                             "pos_mean": sum(pos) / len(pos),
                             "neg_mean": sum(neg) / len(neg)}
        Lb = max(layers, key=lambda L: monitor[c][L]["auroc"])
        best_layer[c] = Lb
        negs = sorted([score(H["control"][i][Lb], c, Lb)
                       for i in split["control"]["test"]] +
                      [score(H[c2][i][Lb], c, Lb)
                       for c2 in concepts if c2 != c
                       for i in split[c2]["test"]], reverse=True)
        k = max(0, min(len(negs) - 1, int(args.target_fpr * len(negs))))
        thresholds[c] = negs[k]
        print(f"  {c}: best L{Lb} auroc={monitor[c][Lb]['auroc']:.3f} "
              f"tau={thresholds[c]:.3f}", flush=True)

    # ---- steer ----------------------------------------------------------------
    print("== steer ==", flush=True)
    layer_modules = None
    for path in ("model.layers", "model.language_model.layers"):
        obj = bb
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            layer_modules = obj
            break
        except AttributeError:
            continue
    steer_results = []
    if layer_modules is not None:
        def make_hook(direction: torch.Tensor, alpha: float):
            d = direction.to(device=device)
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + alpha * d.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return hook

        ctl_prompts = [prompts["control"][i] for i in
                       split["control"]["test"][: args.steer_prompts]]
        for c in concepts:
            L = best_layer[c]
            direction = atlas[c][L]["direction"]
            for alpha in alphas:
                handle = layer_modules[L - 1].register_forward_hook(
                    make_hook(direction, alpha))
                try:
                    for p in ctl_prompts:
                        enc = tok(p, return_tensors="pt").to(device)
                        with torch.no_grad():
                            ids = bb.generate(**enc, max_new_tokens=args.max_new,
                                              do_sample=False,
                                              pad_token_id=tok.pad_token_id)
                        text = tok.decode(ids[0, enc.input_ids.shape[1]:],
                                          skip_special_tokens=True)
                        # verify: does the steered output's state enter the basin?
                        s_after = score(states(p + " " + text)[L], c, L)
                        steer_results.append({
                            "concept": c, "layer": L, "alpha": alpha,
                            "prompt": p, "output": text,
                            "dup5": dup5(text), "basin_score_after": s_after,
                            "fired": s_after >= thresholds[c],
                        })
                finally:
                    handle.remove()
                fired = [r for r in steer_results
                         if r["concept"] == c and r["alpha"] == alpha]
                print(f"  {c} a={alpha:+.2f}: fired {sum(r['fired'] for r in fired)}"
                      f"/{len(fired)}  dup5_max={max(r['dup5'] for r in fired):.2f}",
                      flush=True)
    else:
        print("  (no layer module path found; steer stage skipped)", flush=True)

    # ---- route ----------------------------------------------------------------
    print("== route ==", flush=True)
    rng2 = random.Random(args.seed + 2)
    stream = []
    for c in concepts:
        for i in split[c]["test"]:
            stream.append((prompts[c][i], c))
    for i in split["control"]["test"]:
        stream.append((prompts["control"][i], "control"))
    rng2.shuffle(stream)
    stream = stream[: args.route_n]
    route_log = []
    for text, true_c in stream:
        st = states(text)
        fired = [c for c in concepts
                 if score(st[best_layer[c]], c, best_layer[c]) >= thresholds[c]]
        route_log.append({"prompt": text, "true": true_c, "fired": fired,
                          "action": "escalate" if fired else "pass"})
    catch = sum(1 for r in route_log if r["true"] != "control" and r["true"] in r["fired"])
    n_pos = sum(1 for r in route_log if r["true"] != "control")
    false_trig = sum(1 for r in route_log if r["true"] == "control" and r["fired"])
    n_neg = sum(1 for r in route_log if r["true"] == "control")
    print(f"  catch {catch}/{n_pos}  false-trigger {false_trig}/{n_neg}", flush=True)

    result = {
        "backbone": args.backbone, "layers": layers,
        "convention": "centered cosine to concept centroid; mu = pooled atlas mean per layer",
        "monitor": {c: {str(L): monitor[c][L] for L in layers} for c in concepts},
        "best_layer": best_layer, "thresholds": thresholds,
        "steer": steer_results,
        "route": {"n": len(route_log), "catch": catch, "n_pos": n_pos,
                  "false_trigger": false_trig, "n_neg": n_neg, "log": route_log},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
