"""Holonomy as a blind candidate-selection signal on HumanEval.

Transport a candidate's hidden state around a closed cross-lab loop
A -> B -> C -> A using ridge maps, and measure how far it fails to come back.
The hypothesis is that a candidate represented consistently across independently
trained models returns near itself, while a model-specific artifact accumulates
defect. Unlike best-of-K oracle scoring this needs no target and no execution,
and unlike cross-model agreement it runs only model A at inference.

Maps are fit on arms disjoint from the evaluated arms, so no candidate under
evaluation contributes to the transport it is scored by.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LOOP = [
    ("qwen25_coder7b", "Qwen/Qwen2.5-Coder-7B-Instruct", "bfloat16"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B", "bfloat16"),
    ("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "bfloat16"),
]
FIT_ARMS = ["coder14B_inst__chat", "coder14B_inst__shared",
            "coder3B_inst__chat", "coder3B_inst__shared"]
EVAL_ARMS = ["coder7B_inst__chat", "coder7B_inst__shared",
             "coder7B_inst__shared_persona"]


def encode(tag, mid, dtype, texts, cache, layer_frac=0.6, max_len=256, batch=32):
    path = os.path.join(cache, f"{tag}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if len(z["X"]) == len(texts):
            print(f"  {tag:16s} cached d={z['X'].shape[1]}", flush=True)
            return z["X"]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModel.from_pretrained(mid, dtype=getattr(torch, dtype)).to(DEV).eval()
    L = mod.config.num_hidden_layers
    layer = max(1, int(round(layer_frac * L)))
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(DEV)
            h = mod(**b, output_hidden_states=True).hidden_states[layer]
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
            print(f"\r  {tag:16s} {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    X = np.concatenate(out).astype(np.float32)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, X=X, layer=layer, model=mid)
    print(f"\r  {tag:16s} done d={X.shape[1]} layer {layer}/{L}   ", flush=True)
    del mod
    torch.cuda.empty_cache()
    return X


def ridge(A, B):
    """Fit centered A -> centered B. Returns W and the two train means."""
    A, B = A.astype(np.float64), B.astype(np.float64)
    amu, bmu = A.mean(0), B.mean(0)
    Ac, Bc = A - amu, B - bmu
    G = Ac.T @ Ac
    lam = 1e-3 * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Ac.T @ Bc)
    return W, amu, bmu


def auroc(score, label):
    """P(score higher for label==1). label 1 = candidate is WRONG."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = score.argsort().argsort().astype(float) + 1
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="artifacts/nla/coder_ladder")
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--cache", default="artifacts/nla/holonomy/states")
    ap.add_argument("--out", default="artifacts/nla/holonomy/results.json")
    ap.add_argument("--fit-arms", nargs="*", default=FIT_ARMS)
    ap.add_argument("--eval-arms", nargs="*", default=EVAL_ARMS)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()

    from code_select import execute_file

    probs = json.load(open(os.path.join(HERE, a.probs)))

    def load(arms):
        texts, owner = [], []
        for arm in arms:
            rows = json.load(open(os.path.join(HERE, a.gen_dir, arm + ".json")))
            k = min(len(c) for c in rows)
            rows = [c[:k] for c in rows]
            for i, cands in enumerate(rows):
                for c in cands:
                    texts.append(c)
                    owner.append((arm, i))
        return texts, owner

    fit_texts, _ = load(a.fit_arms)
    ev_texts, ev_owner = load(a.eval_arms)
    print(f"fit {len(fit_texts)} candidates ({len(a.fit_arms)} arms)  "
          f"eval {len(ev_texts)} ({len(a.eval_arms)} arms)", flush=True)

    cache = os.path.join(HERE, a.cache)
    XF, XE = {}, {}
    for tag, mid, dt in LOOP:
        XF[tag] = encode(tag + "__fit", mid, dt, fit_texts, cache)
        XE[tag] = encode(tag + "__eval", mid, dt, ev_texts, cache)

    tags = [t for t, _, _ in LOOP]
    # Ground truth for the evaluated arms.
    ok = {}
    for arm in a.eval_arms:
        rows = json.load(open(os.path.join(HERE, a.gen_dir, arm + ".json")))
        k = min(len(c) for c in rows)
        ok[arm] = execute_file([c[:k] for c in rows], probs, a.timeout, a.workers)
        print(f"  {arm:34s} pass {ok[arm].mean():.4f}", flush=True)

    routes = {
        "loop3_ABCA": [0, 1, 2, 0],
        "loop2_ABA": [0, 1, 0],
        "loop2_ACA": [0, 2, 0],
    }
    res = {"n_fit": len(fit_texts), "n_eval": len(ev_texts),
           "loop": tags, "fit_arms": a.fit_arms, "eval_arms": a.eval_arms,
           "routes": {}}

    for name, route in routes.items():
        mats = []
        for s, d in zip(route, route[1:]):
            W, _, _ = ridge(XF[tags[s]], XF[tags[d]])
            mats.append(W)
        mu = {t: XF[t].mean(0).astype(np.float64) for t in tags}
        H = XE[tags[route[0]]].astype(np.float64) - mu[tags[route[0]]]
        P = H
        for W in mats:
            P = P @ W

        def nrm(M):
            return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
        defect = 1.0 - (nrm(P) * nrm(H)).sum(1)

        wrong, sel, sel_hi, floor, orc = [], [], [], [], []
        off = 0
        for arm in a.eval_arms:
            n, k = ok[arm].shape
            d = defect[off:off + n * k].reshape(n, k)
            off += n * k
            wrong.extend((~ok[arm]).ravel().tolist())
            for i in range(n):
                sel.append(bool(ok[arm][i, int(d[i].argmin())]))
                sel_hi.append(bool(ok[arm][i, int(d[i].argmax())]))
            floor.append(ok[arm].mean())
            orc.append(ok[arm].any(1).mean())
        au = auroc(defect, wrong)
        # If defect merely tracks verbosity, picking the longest candidate matches it.
        sel_len, off = [], 0
        for arm in a.eval_arms:
            n, k = ok[arm].shape
            L = np.array([len(t) for t in ev_texts[off:off + n * k]]).reshape(n, k)
            off += n * k
            for i in range(n):
                sel_len.append(bool(ok[arm][i, int(L[i].argmax())]))
        r = {
            "auroc_defect_predicts_wrong": au,
            "auroc_low_defect_predicts_wrong": 1.0 - au,
            "select_argmin_defect": float(np.mean(sel)),
            "select_argmax_defect": float(np.mean(sel_hi)),
            "select_argmax_length_control": float(np.mean(sel_len)),
            "defect_vs_length_pearson": float(np.corrcoef(
                defect, [len(t) for t in ev_texts])[0, 1]),
            "floor_pass1": float(np.mean(floor)),
            "oracle_passk": float(np.mean(orc)),
            "defect_mean": float(defect.mean()),
            "defect_std": float(defect.std()),
        }
        r["argmin_minus_floor"] = r["select_argmin_defect"] - r["floor_pass1"]
        r["argmax_minus_floor"] = r["select_argmax_defect"] - r["floor_pass1"]
        r["argmax_minus_length_control"] = (r["select_argmax_defect"]
                                            - r["select_argmax_length_control"])
        # Null: pick one candidate uniformly at random per problem, 10k times.
        rng = np.random.default_rng(0)
        stack = np.concatenate([ok[arm] for arm in a.eval_arms], 0)
        idx = rng.integers(0, stack.shape[1], size=(10000, stack.shape[0]))
        null = stack[np.arange(stack.shape[0])[None, :], idx].mean(1)
        r["null_mean"] = float(null.mean())
        r["null_p95"] = float(np.percentile(null, 95))
        r["p_value_argmax"] = float((null >= r["select_argmax_defect"]).mean())
        res["routes"][name] = r
        np.savez_compressed(os.path.join(HERE, os.path.dirname(a.out),
                                         f"defect_{name}.npz"), defect=defect)
        print(f"  {name:12s} defect {r['defect_mean']:.4f}  auroc(low=wrong) "
              f"{r['auroc_low_defect_predicts_wrong']:.4f}  argmin {r['select_argmin_defect']:.4f}  "
              f"argmax {r['select_argmax_defect']:.4f}  len-ctrl "
              f"{r['select_argmax_length_control']:.4f}  floor {r['floor_pass1']:.4f}  "
              f"oracle {r['oracle_passk']:.4f}  r_len {r['defect_vs_length_pearson']:+.3f}", flush=True)

    lens = np.array([len(t) for t in ev_texts], float)
    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
