#!/usr/bin/env python3
"""Does the chat frame collapse expert routing?

Sections 5 and 5.1 established that the assistant turn structure raises output
similarity sharply, and located the effect in the frame rather than the persona.
That is still an argument from text. A mixture-of-experts model lets us look at
the computation itself.

gpt-oss-20b routes each token to 4 of 32 experts per layer. If the chat frame
narrows which experts fire, then the homogeneity has a visible parameter-level
signature: every prompt, whatever it asks, gets handled by the same subset of the
network. That would be a mechanism rather than a correlation, and it is only
observable with open weights.

Same 60 stems, two conditions, forward passes only:

  raw   the bare stem
  chat  the stem through the model's own template

Measured per MoE layer, over all real tokens:

  usage entropy      distribution over the 32 experts. Lower means fewer experts
                     carry the work.
  effective experts  exp(entropy), the count an equally-spread router would need
  top-4 coverage     share of routing mass taken by the four busiest experts
  per-token entropy  how decisive each individual routing decision is
  Jaccard            overlap between the expert sets the two conditions favour

This is one model, so it is a case study and not a population claim. It can show
that the frame changes routing; it cannot show that every MoE behaves this way.

    python scripts/hivemind_moe_routing.py --prompts 60
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind_mechanism_link import STEMS

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
MODEL = "/Volumes/LaCie/SRT-ARCHIVE/host-models/openai__gpt-oss-20b"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--prompts", type=int, default=60)
    p.add_argument("--out", default="artifacts/nla/atlas/hivemind_moe_routing.json")
    return p.parse_args()


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum(axis=axis).mean()) if axis == -1 else None


def collect(mod, tok, prompts, mode, n_exp, topk):
    """Accumulate expert usage and per-token routing entropy, per MoE layer."""
    usage, tok_ent, n_layers = None, [], None
    with torch.no_grad():
        for i, stem in enumerate(prompts):
            if mode == "chat":
                text = tok.apply_chat_template([{"role": "user", "content": stem}],
                                               tokenize=False, add_generation_prompt=True)
                b = tok([text], return_tensors="pt", add_special_tokens=False).to(DEV)
            else:
                b = tok([stem], return_tensors="pt").to(DEV)
            out = mod(**b, output_router_logits=True)
            rl = getattr(out, "router_logits", None)
            if rl is None:
                raise RuntimeError("model did not return router_logits")
            if usage is None:
                n_layers = len(rl)
                usage = np.zeros((n_layers, n_exp), dtype=np.float64)
                tok_ent = [[] for _ in range(n_layers)]
            for L, logits in enumerate(rl):
                x = logits.float().reshape(-1, n_exp)
                p = torch.softmax(x, dim=-1).cpu().numpy()
                # routing mass actually dispatched: top-k renormalized, as at inference
                idx = np.argpartition(-p, topk - 1, axis=-1)[:, :topk]
                m = np.zeros_like(p)
                np.put_along_axis(m, idx, np.take_along_axis(p, idx, -1), -1)
                m /= np.clip(m.sum(-1, keepdims=True), 1e-12, None)
                usage[L] += m.sum(0)
                q = np.clip(p, 1e-12, None)
                tok_ent[L].append(float((-(q * np.log(q)).sum(-1)).mean()))
            print(f"\r  {mode:4s} {i + 1}/{len(prompts)}", end="", flush=True)
    print()
    usage = usage / np.clip(usage.sum(-1, keepdims=True), 1e-12, None)
    return usage, np.array([np.mean(t) for t in tok_ent])


def summarize(usage, tok_ent, n_exp, topk):
    ent = -(np.clip(usage, 1e-12, None) * np.log(np.clip(usage, 1e-12, None))).sum(-1)
    srt = -np.sort(-usage, axis=-1)
    return {
        "n_moe_layers": int(usage.shape[0]),
        "usage_entropy_mean": round(float(ent.mean()), 4),
        "usage_entropy_max_possible": round(float(np.log(n_exp)), 4),
        "effective_experts_mean": round(float(np.exp(ent).mean()), 4),
        "top4_coverage_mean": round(float(srt[:, :topk].sum(-1).mean()), 4),
        "per_token_routing_entropy": round(float(tok_ent.mean()), 4),
        "per_layer_effective_experts": [round(float(x), 2) for x in np.exp(ent)],
    }


def main():
    a = parse_args()
    prompts = STEMS[:a.prompts]
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model)
    n_exp = getattr(cfg, "num_local_experts", None) or cfg.num_experts
    topk = getattr(cfg, "num_experts_per_tok", None) or cfg.experts_per_token
    print(f"device {DEV}   {a.model.split('/')[-1]}   {n_exp} experts, top-{topk}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(DEV).eval()
    print(f"loaded in {time.time() - t0:.0f}s\n")

    res = {"model": a.model, "n_experts": int(n_exp), "experts_per_token": int(topk),
           "n_prompts": len(prompts),
           "scope": "one MoE model, a case study rather than a population claim",
           "arms": {}}
    keep = {}
    for mode in ("raw", "chat"):
        u, te = collect(mod, tok, prompts, mode, n_exp, topk)
        keep[mode] = u
        res["arms"][mode] = summarize(u, te, n_exp, topk)

    r, c = keep["raw"], keep["chat"]
    top = lambda m: [set(np.argsort(-m[L])[:topk].tolist()) for L in range(m.shape[0])]
    tr, tc = top(r), top(c)
    jac = [len(x & y) / len(x | y) for x, y in zip(tr, tc)]
    res["comparison"] = {
        "delta_usage_entropy": round(res["arms"]["chat"]["usage_entropy_mean"]
                                     - res["arms"]["raw"]["usage_entropy_mean"], 4),
        "delta_effective_experts": round(res["arms"]["chat"]["effective_experts_mean"]
                                         - res["arms"]["raw"]["effective_experts_mean"], 4),
        "delta_top4_coverage": round(res["arms"]["chat"]["top4_coverage_mean"]
                                     - res["arms"]["raw"]["top4_coverage_mean"], 4),
        "delta_per_token_entropy": round(res["arms"]["chat"]["per_token_routing_entropy"]
                                         - res["arms"]["raw"]["per_token_routing_entropy"], 4),
        "top_expert_jaccard_mean": round(float(np.mean(jac)), 4),
        "layers_with_identical_top4": int(sum(1 for j in jac if j == 1.0)),
    }
    d = res["comparison"]
    res["verdict"] = (
        f"The chat frame changes effective experts by {d['delta_effective_experts']:+.2f} "
        f"out of {n_exp} and top-{topk} coverage by {d['delta_top4_coverage']:+.4f}. "
        + ("Routing CONCENTRATES under the frame, which is a parameter-level signature "
           "of the convergence." if d["delta_effective_experts"] < -0.5 else
           "Routing does not concentrate meaningfully, so the frame changes what the "
           "model says without narrowing which experts say it."))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"\n{'arm':<8}{'usage H':>10}{'eff experts':>13}{'top4 cov':>10}{'tok H':>9}")
    for m in ("raw", "chat"):
        v = res["arms"][m]
        print(f"  {m:<6}{v['usage_entropy_mean']:>10.4f}{v['effective_experts_mean']:>13.4f}"
              f"{v['top4_coverage_mean']:>10.4f}{v['per_token_routing_entropy']:>9.4f}")
    print(f"  {'max':<6}{res['arms']['raw']['usage_entropy_max_possible']:>10.4f}"
          f"{float(n_exp):>13.4f}")
    print(f"\ntop-{topk} expert Jaccard raw vs chat: {d['top_expert_jaccard_mean']}")
    print(f"layers with identical top-{topk}: {d['layers_with_identical_top4']}"
          f"/{res['arms']['raw']['n_moe_layers']}")
    print(f"\n{res['verdict']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
