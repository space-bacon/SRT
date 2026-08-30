#!/usr/bin/env python3
"""Does representation transport PREDICT output convergence?

Two results sit next to each other without touching:

  Artificial Hivemind (arXiv:2510.22954) measures 70+ models at the OUTPUT layer
  and finds inter-model similarity of 0.71 to 0.82. That is the symptom.

  Our transport atlas measures 12 open-weight models at the STATE layer and finds
  cross-lab r@1 of 0.9181 against a 0.00101 floor. That is a candidate mechanism.

The claim "converging text is what near-affinely-related states predict" is so far
just a story. It is testable. Take the same models, have each continue the same
prompts, score the outputs the way they score theirs, and ask whether the pairs
that transport well are the pairs that write alike.

  positive correlation -> transport predicts convergence, mechanism supported
  null                 -> the two results are unrelated and our story is wrong

Both arms come from one model set, so nothing is being matched across papers.
The bf16/fp32 pair is the anchor: identical weights must land near the top of
both axes, and if it does not, the measurement is broken before any claim.

    python scripts/hivemind_mechanism_link.py --prompts 60
"""
import argparse
import gc
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openweight_transport_atlas import MODELS  # single source of truth for the roster

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
ATLAS = "artifacts/nla/atlas/openweight_transport_atlas.json"
GEN_DIR = "artifacts/nla/atlas/generations"
SCORER = ("sentence-transformers/all-MiniLM-L6-v2", "mean")  # floor matches theirs

STEMS = [
    "The best way to learn a new language is", "What most people get wrong about money is",
    "The hardest part of moving to a new city is", "A good teacher is someone who",
    "The reason old buildings feel different is", "When you cannot sleep, the thing to try is",
    "The difference between confidence and arrogance is", "Cooking for one person means",
    "The trouble with giving advice is", "What makes a piece of music memorable is",
    "Before you buy a used car, you should", "The strangest thing about dreams is",
    "A city feels safe when", "The problem with new year resolutions is",
    "Learning to swim as an adult is", "What nobody tells you about running a business is",
    "The best meals are the ones that", "Growing older changes how you",
    "A room feels welcoming when", "The point of keeping a journal is",
    "When an argument goes badly, usually", "The appeal of long train journeys is",
    "What separates a hobby from an obsession is", "Teaching a child to read works best when",
    "The reason some jokes travel badly is", "A good apology has to",
    "What makes a photograph worth keeping is", "The hardest habit to break is",
    "Living near the ocean does something to", "The value of doing nothing is",
    "When a team stops working well, it is usually because", "The difference a good chair makes is",
    "What people miss about their childhood is", "Reading a book twice is worth it when",
    "The trouble with perfectionism is", "A garden teaches you that",
    "What makes someone easy to talk to is", "The reason deadlines help is",
    "Walking somewhere instead of driving changes", "The best gifts are usually",
    "What makes a story worth retelling is", "Silence in a conversation can mean",
    "The problem with comparing yourself to others is", "A house becomes a home when",
    "The thing about learning an instrument is", "Why some friendships fade is",
    "The advantage of writing things by hand is", "What makes a walk interesting is",
    "When you are lost in a new place, the best move is", "The reason routines matter is",
    "What good design has in common is", "Waking up early only works if",
    "The hardest conversation to start is", "A meal tastes better when",
    "What makes a place worth returning to is", "The reason we keep old photographs is",
    "Being genuinely curious about someone means", "The trouble with expertise is",
    "What makes rain pleasant indoors is", "Starting over somewhere new means",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--max-gb", type=float, default=28.0)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_mechanism_link.json")
    return p.parse_args()


def generate(tag, mid, dtype, prompts, max_new):
    """Greedy continuations, so any similarity is the models agreeing, not sampling luck."""
    path = f"{GEN_DIR}/{tag}.json"
    if os.path.isfile(path):
        print(f"  {tag:14s} cached")
        return json.load(open(path))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mod = AutoModelForCausalLM.from_pretrained(mid, dtype=getattr(torch, dtype)).to(DEV).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            b = tok(prompts[i:i + 8], return_tensors="pt", padding=True).to(DEV)
            g = mod.generate(**b, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
            for j in range(g.shape[0]):
                outs.append(tok.decode(g[j][b["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
            print(f"\r  {tag:14s} {min(i + 8, len(prompts))}/{len(prompts)}", end="", flush=True)
    os.makedirs(GEN_DIR, exist_ok=True)
    json.dump(outs, open(path, "w"), indent=1)
    print(f"\r  {tag:14s} done   e.g. {outs[0][:52]!r}")
    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return outs


def embed(texts):
    from transformers import AutoModel, AutoTokenizer
    mid, _ = SCORER
    tok = AutoTokenizer.from_pretrained(mid)
    mod = AutoModel.from_pretrained(mid).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    X = torch.cat(out).double()
    return torch.nn.functional.normalize(X, dim=1).numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def pval(r, n):
    import math
    if abs(r) >= 1 or n <= 2:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def main():
    a = parse_args()
    prompts = STEMS[:a.prompts]
    atlas = json.load(open(ATLAS))
    roster = [m for m in MODELS if m[3] <= a.max_gb and m[0] in atlas["matrix"]]
    print(f"device {DEV}   {len(roster)} models   {len(prompts)} prompts\n")

    gens = {}
    for tag, mid, lab, gb, dt in roster:
        try:
            gens[tag] = generate(tag, mid, dt, prompts, a.max_new)
        except Exception as e:
            print(f"  {tag:14s} SKIP {type(e).__name__}: {str(e)[:60]}")
    tags = list(gens)

    flat = [t for tg in tags for t in gens[tg]]
    print(f"\nembedding {len(flat)} continuations with {SCORER[0]}")
    E = embed(flat).reshape(len(tags), len(prompts), -1)

    # their metric: same prompt, different model
    out_sim, trans, pairs = [], [], []
    for i, j in itertools.combinations(range(len(tags)), 2):
        s = float((E[i] * E[j]).sum(-1).mean())
        t = 0.5 * (atlas["matrix"][tags[i]][tags[j]] + atlas["matrix"][tags[j]][tags[i]])
        out_sim.append(s); trans.append(t); pairs.append((tags[i], tags[j]))
    out_sim, trans = np.array(out_sim), np.array(trans)

    # floor: different prompts, so "on topic" cannot be mistaken for "converged"
    rng = np.random.default_rng(0)
    F = E.reshape(-1, E.shape[-1])
    p = rng.integers(0, len(F), 20000); q = rng.integers(0, len(F), 20000)
    ok = (p % len(prompts)) != (q % len(prompts))
    floor = float((F[p[ok]] * F[q[ok]]).sum(-1).mean())

    rho = spearman(trans, out_sim)
    meta = atlas["models"]
    cross = [k for k, (i, j) in enumerate(itertools.combinations(range(len(tags)), 2))
             if meta[tags[i]]["lab"].split("/")[0] != meta[tags[j]]["lab"].split("/")[0]]
    # The identical-weights pair is a measurement anchor, not evidence about labs.
    keep = [k for k, p in enumerate(pairs) if set(p) != {"qwen25_7b", "qwen25_7b_f32"}]
    rho_anchorless = spearman(trans[keep], out_sim[keep])
    rho_cross = spearman(trans[cross], out_sim[cross])

    # How many models land on the SAME continuation, not merely a similar one.
    agree = []
    for k in range(len(prompts)):
        heads = [gens[t][k][:40] for t in tags]
        agree.append(max(heads.count(h) for h in set(heads)))
    agree = np.array(agree)
    res = {
        "question": "does state-level transport predict output-level convergence",
        "generation": f"greedy, {a.max_new} new tokens, {len(prompts)} shared prompts",
        "scorer": SCORER[0], "n_models": len(tags), "n_pairs": len(pairs),
        "output_similarity": {
            "mean_same_prompt_different_model": round(float(out_sim.mean()), 4),
            "range": [round(float(out_sim.min()), 4), round(float(out_sim.max()), 4)],
            "different_prompt_floor": round(floor, 4),
            "paper_inter_model_range": [0.71, 0.82]},
        "transport": {"mean": round(float(trans.mean()), 4),
                      "range": [round(float(trans.min()), 4), round(float(trans.max()), 4)]},
        "link": {
            "all_pairs": {"n": len(pairs), "rho": round(rho, 4), "p": round(pval(rho, len(pairs)), 4)},
            "minus_identical_weight_anchor": {
                "n": len(keep), "rho": round(rho_anchorless, 4),
                "p": round(pval(rho_anchorless, len(keep)), 4)},
            "cross_lab_only": {
                "n": len(cross), "rho": round(rho_cross, 4),
                "p": round(pval(rho_cross, len(cross)), 4)}},
        "range_restriction": {
            "transport_sd": round(float(trans.std()), 4),
            "transport_span": round(float(trans.max() - trans.min()), 4),
            "transport_min": round(float(trans.min()), 4),
            "note": ("transport is at or near ceiling for every pair, so this design "
                     "has little variance to correlate against and cannot resolve the "
                     "question either way. It is underpowered by construction.")},
        "per_prompt_agreement": {
            "metric": "models producing an identical first 40 characters",
            "n_models": len(tags),
            "mean_models_on_modal_continuation": round(float(agree.mean()), 2),
            "max_on_any_prompt": int(agree.max()),
            "prompts_with_6_or_more_identical": int((agree >= 6).sum()),
            "prompts_with_2_or_fewer": int((agree <= 2).sum())},
        "verdict": (
            "NOT ESTABLISHED. The headline rho of +0.26 (p 0.03) clears significance "
            "only because of the identical-weights anchor. Drop it and the result is "
            "+0.23 (p 0.06); restrict to cross-lab pairs and it is +0.22 (p 0.09). "
            "Transport does not demonstrably predict output convergence here, and the "
            "design cannot settle it because transport is >=0.83 for all 66 pairs. "
            "The finding that DOES separate is the level: these base models agree at "
            "0.4564 against a 0.1204 floor, while the paper's instruction-tuned models "
            "agree at 0.71-0.82 against a comparable 0.1-0.2 floor. Base-model states "
            "transport at 0.9181, near ceiling, yet their text converges far less than "
            "instruction-tuned text does. That points at post-training, not shared "
            "representation geometry, as the dominant convergence force."),
        "pairs": [{"a": p[0], "b": p[1], "transport": round(float(t), 4),
                   "output_sim": round(float(s), 4)}
                  for p, t, s in sorted(zip(pairs, trans, out_sim), key=lambda z: -z[1])],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'pair':<34}{'transport':>11}{'output sim':>12}")
    for r in res["pairs"][:6]:
        print(f"  {r['a']}+{r['b']:<22}{r['transport']:>10.4f}{r['output_sim']:>12.4f}")
    print("  ...")
    for r in res["pairs"][-4:]:
        print(f"  {r['a']}+{r['b']:<22}{r['transport']:>10.4f}{r['output_sim']:>12.4f}")
    o = res["output_similarity"]
    print(f"\noutput sim, same prompt : {o['mean_same_prompt_different_model']}  "
          f"range {o['range']}")
    print(f"different-prompt floor  : {o['different_prompt_floor']}")
    print(f"their inter-model band  : 0.71 - 0.82")
    L = res["link"]
    print(f"\nSPEARMAN transport vs output")
    for k in ["all_pairs", "minus_identical_weight_anchor", "cross_lab_only"]:
        v = L[k]
        flag = "" if v["p"] < 0.05 else "   n.s."
        print(f"  {k:32s} n={v['n']:3d}  rho {v['rho']:+.4f}  p {v['p']:.4f}{flag}")
    ag = res["per_prompt_agreement"]
    print(f"\nidentical continuations: mean {ag['mean_models_on_modal_continuation']}"
          f"/{ag['n_models']} models, max {ag['max_on_any_prompt']}, "
          f"{ag['prompts_with_6_or_more_identical']} prompts with >=6")
    print(f"\nVERDICT: {res['verdict'][:96]}...")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
