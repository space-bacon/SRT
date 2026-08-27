"""Do typologically unrelated languages converge structurally, or drift?

The MaL convergence argument rests on an empirical premise about natural
language: that Mandarin, Arabic and Finnish "share only the shallowest semantic
universals", with no deep structural convergence across independently evolving
linguistic lineages. Mathematics converging where language does not is what the
constitutive posit is introduced to explain, so the premise carries the weight.

This measures it. FLORES-200 is sentence-aligned across all four languages, so
the same proposition exists in each. Encode each translation in a frozen
backbone, centre each language by its own mean because anisotropy differs by
language, and ask whether a sentence's translation is its nearest neighbour in
the other language. Chance is 1/N.

CONFOUND, stated up front: one model saw all four languages in training, so any
alignment may be imposed by joint training rather than recovered from the
languages. This measures how much shared structure is recoverable, which bounds
the premise from one side without settling its cause.

LIMIT: FLORES is gated and its ungated mirrors are script-based, which datasets 5
no longer runs, so this uses opus-100 with English as the pivot. Each pair has
its own sentence pool, so Mandarin against Finnish directly is not measured here.

    python scripts/crosslingual_convergence.py --n 400
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

BACKBONE = "google/gemma-4-31B-it"
LAYER = 47
# His three named languages, each against English.
PAIRS = [("en-zh", "en", "zh", "English", "Mandarin"),
         ("ar-en", "en", "ar", "English", "Arabic"),
         ("en-fi", "en", "fi", "English", "Finnish")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--backbone", default=BACKBONE)
    p.add_argument("--out", default="/root/crosslingual_convergence.json")
    return p.parse_args()


def load(n: int) -> dict:
    from datasets import load_dataset
    out = {}
    for cfg, a_code, b_code, a_name, b_name in PAIRS:
        d = load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        rows = [(r["translation"][a_code], r["translation"][b_code]) for r in d]
        # Very short lines carry little structure to converge on.
        rows = [(x, y) for x, y in rows if len(x.split()) >= 5 and len(y) >= 10][:n]
        out[f"{a_name}-{b_name}"] = rows
        print(f"  {a_name}-{b_name}: {len(rows)} pairs, e.g. {rows[0][0][:50]!r}", flush=True)
    return out


@torch.no_grad()
def encode(sents: list[str], tok, model, layer: int) -> np.ndarray:
    vecs = []
    for i in range(0, len(sents), 16):
        chunk = sents[i:i + 16]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=96).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last = enc["attention_mask"].sum(1) - 1
        h = out.hidden_states[layer][torch.arange(len(chunk)), last]
        vecs.append(h.float().cpu().numpy())
    return np.concatenate(vecs)


def retrieval(A: np.ndarray, B: np.ndarray) -> dict:
    """A[i] and B[i] are the same proposition. Is B[i] the nearest to A[i]?"""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    S = A @ B.T
    d = np.arange(len(A))
    rank = (S > S[d, d][:, None]).sum(1) + 1
    return {"r@1": float((rank == 1).mean()),
            "r@10": float((rank <= 10).mean()),
            "median_rank": float(np.median(rank))}


def main() -> None:
    a = parse_args()
    pairs = load(a.n)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        a.backbone, dtype=torch.bfloat16, device_map="auto").eval()

    rng = np.random.default_rng(0)
    results, floors = {}, {}
    for key, rows in pairs.items():
        A = encode([r[0] for r in rows], tok, model, a.layer)
        B = encode([r[1] for r in rows], tok, model, a.layer)
        # Centre each language by its own mean. A shared centre would confound
        # this, since anisotropy differs by language.
        Ac, Bc = A - A.mean(0, keepdims=True), B - B.mean(0, keepdims=True)
        results[key] = {"centred": retrieval(Ac, Bc), "raw": retrieval(A, B),
                        "n": len(rows)}
        floors[key] = retrieval(Ac, Bc[rng.permutation(len(Bc))])["r@1"]
        print(f"  {key:18s} centred r@1 {results[key]['centred']['r@1']:.3f}  "
              f"median {results[key]['centred']['median_rank']:.0f}  "
              f"raw r@1 {results[key]['raw']['r@1']:.3f}  "
              f"shuffled {floors[key]:.4f}", flush=True)

    out = {
        "question": "do independently evolved languages converge structurally",
        "premise_tested": ("MaL 5.1: Mandarin, Arabic and Finnish 'share only the "
                           "shallowest semantic universals'"),
        "confound": ("one model saw all these languages, so alignment may be imposed "
                     "by joint training rather than recovered from the languages"),
        "limit": "English pivot only; Mandarin against Finnish is not measured here",
        "backbone": a.backbone, "layer": a.layer, "n_sentences": a.n,
        "chance_r@1": 1.0 / a.n,
        "pairs": results, "shuffled_floor_r@1": floors,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
