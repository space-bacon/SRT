#!/usr/bin/env python3
"""Transport atlas: the inter-model matrix at the layer they cannot reach.

Artificial Hivemind (arXiv:2510.22954) measures similarity between 70+ models by
embedding their TEXT, because almost every model it studies is a closed endpoint.
That establishes that outputs agree. It cannot say why, because output cosine has
no access to the thing doing the agreeing.

Open weights remove that limit. For any pair of models we can ask the sharper
question directly: how much of model B's internal state is a linear function of
model A's? Fit a ridge map on a train split, then on held-out items map A's state
through it and retrieve the matching item among B's states.

Read the matrix as we read every transport result. The diagonal is the self-map,
meaning the same ridge machinery from a model back to itself, and it is the
ceiling that the retrieval task itself allows. Off-diagonal is transport. The
distance between them is what crossing a vendor boundary costs. The floor is a
shuffled control, not an assumption.

States are centered before every solve and every cosine, because a raw cosine on
an anisotropic space is not interpretable and the size of that correction is not
knowable in advance.

    python scripts/openweight_transport_atlas.py --n 5000
"""
import argparse
import gc
import glob
import itertools
import json
import os

import numpy as np
import torch

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")

LACIE = "/Volumes/LaCie/SRT-ARCHIVE/host-models/"

# tag, source, lab, weights_gb, dtype
MODELS = [
    ("qwen25_05b", "Qwen/Qwen2.5-0.5B", "Alibaba", 1.0, "float32"),
    ("qwen3_06b", "Qwen/Qwen3-0.6B-Base", "Alibaba", 1.4, "float32"),
    ("gemma2_2b", "google/gemma-2-2b", "Google", 9.8, "float32"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B", "Meta", 2.5, "float32"),
    ("llama32_3b", "meta-llama/Llama-3.2-3B", "Meta", 6.0, "float32"),
    ("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", 2.2, "float32"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B", "AI2", 2.4, "float32"),
    ("smollm2_360m", "HuggingFaceTB/SmolLM2-360M", "HuggingFace", 0.7, "float32"),
    ("pythia_410m", "EleutherAI/pythia-410m", "EleutherAI", 0.9, "float32"),
    # Frontier open weights off the archive drive. 64 GB of unified memory takes
    # bf16 to roughly 26 GB; past that needs an MLX 4-bit port.
    ("qwen25_7b", LACIE + "Qwen__Qwen2.5-7B", "Alibaba", 14, "bfloat16"),
    ("gptoss_20b", LACIE + "openai__gpt-oss-20b", "OpenAI", 26, "bfloat16"),
    ("mistral_24b", LACIE + "mistralai__Mistral-Small-3.1-24B-Instruct-2503", "Mistral", 45, "bfloat16"),
    ("aria_25b", LACIE + "rhymes-ai__Aria", "rhymes-ai", 47, "bfloat16"),
    ("qwen38_27b", LACIE + "Qwen__Qwen3.8-27B", "Alibaba", 52, "bfloat16"),
    ("gemma4_31b", LACIE + "google__gemma-4-31B-it", "Google", 58, "bfloat16"),
    ("qwen3omni_30b", LACIE + "Qwen__Qwen3-Omni-30B-A3B-Instruct", "Alibaba", 66, "bfloat16"),
    # Precision control: identical weights, different dtype. Its transport entry
    # in the matrix is how much of any cost is numerics and not a lab boundary.
    ("qwen25_7b_f32", LACIE + "Qwen__Qwen2.5-7B", "Alibaba/fp32", 28, "float32"),
]
STATES = "artifacts/nla/atlas/states"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--layer-frac", type=float, default=0.6)
    p.add_argument("--max-gb", type=float, default=26.0,
                   help="skip weights larger than this; 26 is the bf16 ceiling here")
    p.add_argument("--out", default="artifacts/nla/atlas/openweight_transport_atlas.json")
    return p.parse_args()


def load_texts(n):
    import pyarrow.parquet as pq
    f = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--phiyodr--coco2017/snapshots/*/data/*.parquet"))[0]
    rows = pq.read_table(f, columns=["captions"]).to_pylist()
    return [r["captions"][0].strip() for r in rows[:n]]


def extract(tag, mid, texts, layer_frac, dtype):
    """Mean-pooled hidden state at a fixed relative depth."""
    path = f"{STATES}/{tag}.npz"
    if os.path.isfile(path):
        z = np.load(path)
        print(f"  {tag:14s} cached  d={z['X'].shape[1]}  layer {int(z['layer'])}")
        return z["X"], int(z["layer"])
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModel.from_pretrained(mid, dtype=getattr(torch, dtype)).to(DEV).eval()
    L = mod.config.num_hidden_layers
    layer = max(1, int(round(layer_frac * L)))
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True,
                    max_length=48, return_tensors="pt").to(DEV)
            h = mod(**b, output_hidden_states=True).hidden_states[layer]
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
            print(f"\r  {tag:14s} {min(i + 64, len(texts))}/{len(texts)}", end="", flush=True)
    X = np.concatenate(out).astype(np.float32)
    os.makedirs(STATES, exist_ok=True)
    np.savez_compressed(path, X=X, layer=layer, model=mid)
    print(f"\r  {tag:14s} done    d={X.shape[1]}  layer {layer}/{L}")
    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return X, layer


def ridge_r1(A, B, n_tr, rng):
    """Fit A->B on train, retrieve the matching held-out item among B's states."""
    Atr, Ate = A[:n_tr], A[n_tr:]
    Btr, Bte = B[:n_tr], B[n_tr:]
    amu, bmu = Atr.mean(0), Btr.mean(0)          # center on TRAIN only
    Atr, Ate, Btr, Bte = Atr - amu, Ate - amu, Btr - bmu, Bte - bmu
    G = Atr.T @ Atr
    lam = 1e-3 * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0], dtype=G.dtype), Atr.T @ Btr)
    P = Ate @ W

    def nrm(M):
        return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    S = nrm(P) @ nrm(Bte).T
    hit = (S.argmax(1) == np.arange(len(Bte)))
    perm = rng.permutation(len(Bte))             # shuffled floor, same machinery
    floor = float((S[perm].argmax(1) == np.arange(len(Bte))).mean())
    return float(hit.mean()), floor


def main():
    a = parse_args()
    todo = [m for m in MODELS if m[3] <= a.max_gb]
    texts = load_texts(a.n)
    n_tr = len(texts) - a.holdout
    print(f"device {DEV}   {len(texts)} texts   train {n_tr}  holdout {a.holdout}")
    print(f"chance r@1 = {1 / a.holdout:.4f}\n")

    S, meta = {}, {}
    for tag, mid, lab, gb, dt in todo:
        try:
            X, layer = extract(tag, mid, texts, a.layer_frac, dt)
            S[tag] = X.astype(np.float64)
            meta[tag] = {"model": mid, "lab": lab, "dim": X.shape[1],
                         "layer": layer, "weights_gb": gb, "dtype": dt}
        except Exception as e:
            print(f"  {tag:14s} SKIP  {type(e).__name__}: {str(e)[:70]}")

    tags = list(S)
    rng = np.random.default_rng(0)
    print(f"\n{len(tags)} models, {len(set(meta[t]['lab'] for t in tags))} labs\n")

    mat, floors = {}, []
    for src in tags:
        mat[src] = {}
        for dst in tags:
            r1, fl = ridge_r1(S[src], S[dst], n_tr, rng)
            mat[src][dst] = round(r1, 4)
            floors.append(fl)
            print(f"\r  {src} -> {dst}          ", end="", flush=True)
    print("\r" + " " * 44)

    diag = [mat[t][t] for t in tags]
    off = [(s, d, mat[s][d]) for s, d in itertools.permutations(tags, 2)]
    cross = [(s, d, v) for s, d, v in off if meta[s]["lab"] != meta[d]["lab"]]
    within = [(s, d, v) for s, d, v in off if meta[s]["lab"] == meta[d]["lab"]]
    ov = [v for _, _, v in off]

    res = {
        "question": "how much of one open-weight model's state is a linear function of another's",
        "why": ("arXiv:2510.22954 measures 70+ models at the output layer because they "
                "are closed. Open weights let us measure the mechanism instead."),
        "n_texts": len(texts), "n_train": n_tr, "n_holdout": a.holdout,
        "corpus": "COCO val2017 captions, one per image",
        "metric": "held-out retrieval r@1 through a ridge map, centered on train mean",
        "chance_r1": round(1 / a.holdout, 5),
        "models": meta, "matrix": mat,
        "summary": {
            "mean_self_map_diagonal": round(float(np.mean(diag)), 4),
            "mean_transported_off_diagonal": round(float(np.mean(ov)), 4),
            "transport_cost": round(float(np.mean(diag) - np.mean(ov)), 4),
            "mean_shuffled_floor": round(float(np.mean(floors)), 5),
            "mean_within_lab": round(float(np.mean([v for _, _, v in within])), 4) if within else None,
            "mean_cross_lab": round(float(np.mean([v for _, _, v in cross])), 4),
            "within_minus_cross_lab": round(
                float(np.mean([v for _, _, v in within]) - np.mean([v for _, _, v in cross])), 4)
            if within else None,
            "best_pair": max(off, key=lambda t: t[2]),
            "worst_pair": min(off, key=lambda t: t[2]),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    w = max(len(t) for t in tags) + 1
    print(f"{'src \\ dst':<{w}}" + "".join(f"{t[:9]:>10}" for t in tags))
    for s in tags:
        print(f"{s:<{w}}" + "".join(f"{mat[s][d]:10.4f}" for d in tags))
    s = res["summary"]
    print(f"\nself-map diagonal   {s['mean_self_map_diagonal']}")
    print(f"transported         {s['mean_transported_off_diagonal']}")
    print(f"transport cost      {s['transport_cost']}")
    print(f"shuffled floor      {s['mean_shuffled_floor']}   (chance {res['chance_r1']})")
    print(f"within-lab          {s['mean_within_lab']}")
    print(f"cross-lab           {s['mean_cross_lab']}   difference {s['within_minus_cross_lab']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
