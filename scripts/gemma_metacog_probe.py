"""Metacognition probe layer-sweep on gemma-4-31B (the best US model on the
ginigen Metacognition leaderboard).

The leaderboard's adapters read the FINAL hidden layer -> P(wrong). Our gpt-oss
result (scripts/metacog_probe.py) showed a MID layer beats the last layer. This
sweeps candidate layers on gemma-4-31B and reports the AUROC of a ginigen-style
MLP probe at each, to find where the error signal is most linearly available.

TriviaQA (rc.nocontext, validation), greedy free-form answers graded against
aliases; probe (LayerNorm -> d/4 -> GELU -> 1) trained per layer on a 67/33
split; mean-logprob self-confidence baseline.

Output: artifacts/nla/gemma4/metacog_layer_sweep.json
Usage (venv transformers>=5):
    python scripts/gemma_metacog_probe.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--out", default="artifacts/nla/gemma4/metacog_layer_sweep.json")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--gen-bs", type=int, default=16)
    p.add_argument("--enc-bs", type=int, default=8)
    p.add_argument("--layers", default="", help="comma list; default = sweep 8 evenly")
    return p.parse_args()


def norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", "", s).strip()


def auroc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, r = 0.0, 1
    for _, y in pairs:
        if y == 1:
            rank_sum += r
        r += 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import random
    random.seed(0); torch.manual_seed(0)
    from transformers import Gemma4ForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = Gemma4ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") \
        else model.config.text_config.num_hidden_layers
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") \
        else model.config.text_config.hidden_size
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted(set([L] + [round(L * f) for f in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)])))
    print(f"gemma-4 L={L} d={d} | sweeping layers {layers}", flush=True)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    idx = random.sample(range(len(ds)), args.n)
    rows = [{"q": ds[i]["question"],
             "aliases": [a.lower() for a in ds[i]["answer"]["normalized_aliases"]]}
            for i in idx]

    prompts = [f"Answer the trivia question with just the answer.\nQ: {r['q']}\nA:"
               for r in rows]
    answers, logps = [], []
    for s in range(0, len(prompts), args.gen_bs):
        enc = tok(prompts[s:s + args.gen_bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=16, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               return_dict_in_generate=True, output_scores=True)
        new = o.sequences[:, enc.input_ids.shape[1]:]
        lp = torch.stack(o.scores, dim=1).log_softmax(-1)
        for b in range(new.shape[0]):
            ids = new[b]; keep = ids != tok.pad_token_id
            tl = lp[b, torch.arange(ids.shape[0]), ids][keep]
            logps.append(float(tl.mean()) if int(keep.sum()) else -20.0)
            answers.append(tok.decode(ids, skip_special_tokens=True).split("\n")[0].strip())
        if s % (args.gen_bs * 10) == 0:
            print(f"  gen {s + new.shape[0]}/{len(prompts)}", flush=True)

    correct = []
    for r, a in zip(rows, answers):
        na = norm(a)
        correct.append(int(bool(na) and any(
            norm(al) == na or (norm(al) and norm(al) in na) for al in r["aliases"])))
    acc = sum(correct) / len(correct)
    print(f"accuracy {acc:.3f} ({sum(correct)}/{len(correct)})", flush=True)

    # hidden states at "Q..\nA: {ans}" last token, all swept layers
    texts = [f"Q: {r['q']}\nA: {a or 'unknown'}" for r, a in zip(rows, answers)]
    H = {li: torch.empty(len(texts), d) for li in layers}
    for s in range(0, len(texts), args.enc_bs):
        enc = tok(texts[s:s + args.enc_bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        for li in layers:
            H[li][s:s + enc.input_ids.shape[0]] = out.hidden_states[li][:, -1].float().cpu()
        if s % (args.enc_bs * 20) == 0:
            print(f"  enc {s + enc.input_ids.shape[0]}/{len(texts)}", flush=True)

    n = len(rows)
    perm = list(range(n)); random.shuffle(perm)
    tr, te = perm[:int(n * 0.67)], perm[int(n * 0.67):]
    y = torch.tensor([1 - c for c in correct], dtype=torch.float32)

    res = {"model": args.model, "n": n, "accuracy": acc, "text_layers": L,
           "baseline_logp_auroc": auroc([-logps[i] for i in te],
                                        [1 - correct[i] for i in te]),
           "layer_auroc": {}}
    for li in layers:
        X = H[li]
        probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 4), nn.GELU(),
                              nn.Dropout(0.1), nn.Linear(d // 4, 1)).cuda()
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtr, ytr = X[tr].cuda(), y[tr].cuda()
        for _ in range(60):
            probe.train()
            loss = nn.functional.binary_cross_entropy_with_logits(
                probe(Xtr).squeeze(-1), ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(probe(X[te].cuda()).squeeze(-1)).cpu()
        a = auroc(p_te.tolist(), [1 - correct[i] for i in te])
        res["layer_auroc"][str(li)] = a
        print(f"  L{li} ({li/L:.0%} depth): AUROC {a:.3f}", flush=True)

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    best = max(res["layer_auroc"], key=lambda k: res["layer_auroc"][k])
    print(f"baseline logp AUROC {res['baseline_logp_auroc']:.3f}")
    print(f"best layer L{best} ({int(best)/L:.0%}) AUROC {res['layer_auroc'][best]:.3f}"
          f"  | last layer L{L} AUROC {res['layer_auroc'].get(str(L),'?')}")


if __name__ == "__main__":
    main()
