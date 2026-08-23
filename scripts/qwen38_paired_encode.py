"""Paired base-vs-abliterated hidden-state encode, both models resident.

Box 2 has two Blackwells, so `Qwen/Qwen3.8-27B` and its abliterated sibling both
fit (55.6 GB each in bf16). Holding both in one process and pushing the SAME
batch through each removes the re-encode floor from the comparison entirely:
identical tokenization, identical padding, identical batch composition. The
base-vs-abliterated delta is then a property of the weights, which is the whole
question.

Measured re-encode floor when this is NOT done: 9 to 20 item flips per 7,511 on
SugarCrepe (fresh box, same protocol). The abliteration effect could easily be
that size, so the paired design is not a nicety.

    python qwen38_paired_encode.py --texts caps.json --out states.pt
    python qwen38_paired_encode.py --layer-sweep --texts caps.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BASE = "Qwen/Qwen3.8-27B"
ABL = "JonathanColetti/Qwen3.8-27B-Uncensored"
MAX_SEQ = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--key", default=None, help="key if the json is a dict")
    p.add_argument("--base", default=BASE)
    p.add_argument("--abl", default=ABL)
    p.add_argument("--layer", type=int, default=None,
                   help="single layer to keep; default = all (sweep mode)")
    p.add_argument("--layer-sweep", action="store_true",
                   help="keep every layer for a limited number of texts")
    p.add_argument("--sweep-n", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load(name: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16).to(device)
    model.eval()
    return proc, model


def batch_ids(chunk: list[str], tok):
    bos = tok.bos_token_id
    ids_list = []
    for t in chunk:
        ids = tok(t, truncation=True, max_length=MAX_SEQ,
                  add_special_tokens=True).input_ids
        if bos is not None and ids[0] != bos:
            ids = [bos] + ids[: MAX_SEQ - 1]
        ids_list.append(ids)
    T = max(len(x) for x in ids_list)
    pad = tok.pad_token_id if tok.pad_token_id is not None else bos
    input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
    attn = torch.zeros((len(ids_list), T), dtype=torch.long)
    for j, ids in enumerate(ids_list):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        attn[j, : len(ids)] = 1
    return input_ids, attn


@torch.no_grad()
def last_token_states(model, input_ids, attn, device, layers):
    out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                output_hidden_states=True, use_cache=False)
    last = (attn.sum(-1) - 1).to(device)
    rows = torch.arange(input_ids.size(0), device=device)
    return torch.stack([out.hidden_states[l][rows, last].float().cpu()
                        for l in layers], dim=1)          # [B, n_layers, d]


def main() -> None:
    args = parse_args()
    data = json.loads(args.texts.read_text())
    texts = data[args.key] if args.key else data
    if args.layer_sweep:
        texts = texts[: args.sweep_n]
    print(f"{len(texts)} texts, batch {args.batch}", flush=True)

    proc_b, model_b = load(args.base, "cuda:0")
    proc_a, model_a = load(args.abl, "cuda:1")
    tok_b = getattr(proc_b, "tokenizer", proc_b)
    tok_a = getattr(proc_a, "tokenizer", proc_a)

    n_layers = model_b.config.text_config.num_hidden_layers + 1
    layers = list(range(n_layers)) if (args.layer_sweep or args.layer is None) \
        else [args.layer]
    print(f"hidden layers {n_layers}, keeping {len(layers)}", flush=True)

    # Tokenizers must agree or the comparison is confounded by tokenization.
    probe = texts[:256]
    assert all(tok_b(t).input_ids == tok_a(t).input_ids for t in probe), \
        "base and abliterated tokenizers disagree; comparison is not paired"

    rows_b, rows_a = [], []
    for i in range(0, len(texts), args.batch):
        ids, attn = batch_ids(texts[i:i + args.batch], tok_b)
        rows_b.append(last_token_states(model_b, ids, attn, "cuda:0", layers))
        rows_a.append(last_token_states(model_a, ids, attn, "cuda:1", layers))
        if (i // args.batch) % 20 == 0:
            print(f"  {min(i + args.batch, len(texts))}/{len(texts)}", flush=True)

    B = torch.cat(rows_b)
    A = torch.cat(rows_a)
    out = args.out or Path(
        "qwen38_layer_sweep.pt" if args.layer_sweep else "qwen38_paired.pt")
    torch.save({"base": B, "abl": A, "layers": layers, "texts": texts,
                "base_model": args.base, "abl_model": args.abl}, out)
    print(f"saved base {tuple(B.shape)} abl {tuple(A.shape)} -> {out}", flush=True)

    # Per-layer displacement geometry (round-5 protocol from the public review).
    print(f"\n{'layer':>5s} {'cos':>7s} {'||d||':>9s} {'||mu_b||':>9s} {'rho':>7s}")
    for k, l in enumerate(layers):
        b, a = B[:, k], A[:, k]
        cos = torch.nn.functional.cosine_similarity(b, a, dim=-1).mean().item()
        mu_b, mu_a = b.mean(0), a.mean(0)
        d = (mu_a - mu_b).norm().item()
        med_q = b.norm(dim=-1).median().item()
        print(f"{l:5d} {cos:7.4f} {d:9.3f} {mu_b.norm().item():9.3f} "
              f"{d / max(med_q, 1e-9):7.4f}")


if __name__ == "__main__":
    main()
