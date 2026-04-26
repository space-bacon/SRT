"""Cross-backbone community-vector probe (LOO k-NN recall@1).

For each backbone, extract the mean-pooled hidden state at a target layer for
every val_200 sample, then compute leave-one-out 1-nearest-neighbor recall@1
over the 35 coarse community labels. This isolates the question:

    "Is the discourse-community signal that v8a's CDH lights up specific to
    Qwen, or is it latent in other 7B causal LMs as well?"

We do NOT need any adapter or training to answer this -- v8a's CDH at
layer 4 simply mean-pools and projects, so the headline number ought to be
recoverable from raw layer-4 features (paper's 0.484 figure is built on top
of this same signal, just refined through 64-D community vector contrastive
learning).

Usage:
    python scripts/cross_backbone_probe.py \
        --val-data data/val_200.jsonl \
        --backbones Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.3 \
        --layers 4 8 12 \
        --out artifacts/cross_backbone.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # h: (1, T, D); mask: (1, T)
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def loo_knn_recall_at_1(feats: np.ndarray, labels: np.ndarray) -> float:
    # cosine sim, leave-one-out
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    pred = labels[sim.argmax(axis=1)]
    return float((pred == labels).mean())


@torch.no_grad()
def extract(backbone_id: str, samples: list[dict], layers: list[int],
            max_len: int, device: str) -> dict[int, np.ndarray]:
    print(f"\n=== {backbone_id} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        backbone_id, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers_eff = [l for l in layers if l < n_layers]
    print(f"  loaded in {time.time()-t0:.1f}s, n_layers={n_layers}, "
          f"using layers {layers_eff}")
    out = {l: [] for l in layers_eff}
    for i, ex in enumerate(samples):
        enc = tok(ex["text"], return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        h = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        # h is tuple of length n_layers+1 (embedding + per-layer)
        for l in layers_eff:
            v = mean_pool(h[l + 1], enc.attention_mask)[0].float().cpu().numpy()
            out[l].append(v)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] elapsed={time.time()-t0:.1f}s")
    out_arr = {l: np.stack(out[l]) for l in layers_eff}
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data", default="data/val_200.jsonl")
    ap.add_argument("--backbones", nargs="+",
                    default=["Qwen/Qwen2.5-7B",
                             "mistralai/Mistral-7B-v0.3",
                             "meta-llama/Meta-Llama-3-8B"])
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[4, 8, 12, 16])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="artifacts/cross_backbone.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    samples = [json.loads(l) for l in Path(args.val_data).read_text().splitlines()]
    labels = np.array([s["community_id"] for s in samples])
    n_classes = len(set(labels.tolist()))
    chance = 1.0 / n_classes
    print(f"val_200: n={len(samples)}  classes={n_classes}  chance={chance:.3f}")

    results = {
        "n_samples": len(samples),
        "n_classes": n_classes,
        "chance_recall_at_1": chance,
        "backbones": {},
    }
    for backbone in args.backbones:
        try:
            feats = extract(backbone, samples, args.layers, args.max_len, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {backbone}: {e}")
            results["backbones"][backbone] = {"error": str(e)}
            continue
        per_layer = {}
        for layer, X in feats.items():
            r1 = loo_knn_recall_at_1(X, labels)
            per_layer[str(layer)] = {
                "recall_at_1": r1,
                "x_chance": r1 / chance,
                "feature_dim": int(X.shape[1]),
            }
            print(f"  layer {layer:>2}  D={X.shape[1]:>5}  "
                  f"recall@1={r1:.3f}  ({r1/chance:.1f}x chance)")
        results["backbones"][backbone] = {"per_layer": per_layer}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
