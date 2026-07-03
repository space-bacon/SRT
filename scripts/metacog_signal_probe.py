"""Failure-separation probe on Metacognition-Bench: can a mid-layer / SRT signal
flag gemma-4's OWN trap failures better than its verbalized confidence?

Given the judged baseline (metacog_bench_gemma4.json: 286/300 pass, failures
concentrated in G_PivotDetection), we teacher-force each of gemma-4's own answers
back through the frozen model and extract, over the answer span:
  predictive (confidence baseline): nll_mean/max, entropy_mean/max, margin_mean
  SRT read-out (from readout_selected.pt): MAH divergence norm, community entropy
  hidden states at a layer sweep -> diagonal-LDA LOOCV probe

Label = the judge's `correct == False` (the model fell for the trap). We report
AUROC of each scalar signal (unsupervised, no fitting -> honest at small n_fail)
and a LOOCV linear probe per layer, both globally and on the pivot-detection
subset. This directly tests the SRT thesis against the honest-null TriviaQA
finding (there, logp beat every probe on easy QA; traps are the hard regime).

Usage (venv transformers>=5):
    python scripts/metacog_signal_probe.py \
        --judged artifacts/nla/gemma4/metacog_bench_gemma4.rows.judged.jsonl \
        --ckpt checkpoints/gemma4_readout/readout_selected.pt \
        --out artifacts/nla/gemma4/metacog_signal_probe.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--judged", required=True, help="rows.judged.jsonl (task+answer+verdict)")
    p.add_argument("--ckpt", default="checkpoints/gemma4_readout/readout_selected.pt")
    p.add_argument("--out", default="artifacts/nla/gemma4/metacog_signal_probe.json")
    p.add_argument("--layers", default="15,30,45,60")
    p.add_argument("--n", type=int, default=0, help="limit (0 = all)")
    return p.parse_args()


def auroc(scores, labels) -> float:
    order = np.argsort(scores)
    labels = np.asarray(labels)[order]
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.arange(1, len(labels) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def loocv_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Diagonal-covariance LDA direction, leave-one-out. No sklearn."""
    n = len(y)
    if y.sum() < 2 or (n - y.sum()) < 2:
        return float("nan")
    s = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        Xt, yt = X[m], y[m]
        mu1, mu0 = Xt[yt == 1].mean(0), Xt[yt == 0].mean(0)
        var = Xt.var(0) + 1e-6
        w = (mu1 - mu0) / var
        s[i] = float((X[i] - (mu1 + mu0) / 2) @ w)
    return auroc(s, y)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from srt.config import SRTConfig
    from srt.modules.mah import MetapragmaticAttentionHead
    from srt.modules.community import CommunityDiscoveryHead

    rows = [json.loads(l) for l in open(args.judged) if l.strip()]
    if args.n:
        rows = rows[: args.n]
    n_fail = sum(1 for r in rows if not r["verdict"]["correct"])
    print(f"{len(rows)} rows | {n_fail} failures", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cc = ck["config"]
    comm_L, mah_L, d, dc = (cc["community_layer"], cc["mah_layers"][-1],
                            cc["d_backbone"], cc["d_community"])
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(MID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = SRTConfig(backbone_id=MID)
    community = CommunityDiscoveryHead(cfg.community, d).cuda().eval()
    mah = MetapragmaticAttentionHead(cfg.mah, d, d_community=dc).cuda().eval()
    community.load_state_dict({k[2:]: v for k, v in ck["heads"].items() if k.startswith("0.")})
    # mah_heads live under "1.<idx>." — take the last MAH layer's weights
    mah_idx = len(cc["mah_layers"]) - 1
    mah.load_state_dict({k[len(f"1.{mah_idx}."):]: v for k, v in ck["heads"].items()
                         if k.startswith(f"1.{mah_idx}.")})

    def causal(T):
        m = torch.full((T, T), float("-inf"), device="cuda")
        return torch.triu(m, diagonal=1).view(1, 1, T, T)

    feats: dict[str, list] = {k: [] for k in
                              ["nll_mean", "nll_max", "ent_mean", "ent_max",
                               "margin_mean", "div_norm_mean", "comm_entropy"]}
    hid: dict[int, list] = {L: [] for L in layers}
    labels, ticos = [], []

    @torch.no_grad()
    def process(prompt: str, answer: str):
        pids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=True,
                                       return_tensors="pt").to("cuda")
        fids = tok.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": answer}],
            add_generation_prompt=False, tokenize=True, return_tensors="pt").to("cuda")
        start = pids.shape[1]
        if fids.shape[1] <= start + 1:
            return None
        out = model(input_ids=fids, output_hidden_states=True, use_cache=False)
        # predictive signals over answer tokens
        logits = out.logits[0, start - 1:-1].float()          # predicts answer toks
        tgt = fids[0, start:]
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp[torch.arange(len(tgt)), tgt]
        probs = logp.exp()
        ent = -(probs * logp).sum(-1)
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        # answer-span mask over the full sequence for the read-out heads
        amask = torch.zeros(1, fids.shape[1], device="cuda")
        amask[0, start:] = 1
        hs = out.hidden_states
        co = community(hs[comm_L].float(), attention_mask=amask)
        mo = mah(hs[mah_L].float(), community_vec=co.vector, causal_mask=causal(fids.shape[1]))
        dv = mo.divergence[0, start:]                          # (A, d_div)
        div_norm = dv.norm(dim=-1).mean()
        cw = co.weights[0] if co.weights is not None else None
        comm_ent = (-(cw * (cw + 1e-9).log()).sum()) if cw is not None else torch.zeros((), device="cuda")
        return {
            "nll_mean": nll.mean().item(), "nll_max": nll.max().item(),
            "ent_mean": ent.mean().item(), "ent_max": ent.max().item(),
            "margin_mean": margin.mean().item(), "div_norm_mean": div_norm.item(),
            "comm_entropy": comm_ent.item(),
            "hid": {L: hs[L][0, start:].float().mean(0).cpu().numpy() for L in layers},
        }

    for i, r in enumerate(rows, 1):
        res = process(r["prompt"], r["answer"])
        if res is None:
            continue
        for k in feats:
            feats[k].append(res[k])
        for L in layers:
            hid[L].append(res["hid"][L])
        labels.append(0 if r["verdict"]["correct"] else 1)
        ticos.append(r.get("ticos_type"))
        if i % 25 == 0:
            print(f"  processed {i}/{len(rows)}", flush=True)

    y = np.array(labels)
    ticos = np.array(ticos)
    pivot = ticos == "G_PivotDetection"

    def scalar_aurocs(mask):
        yy = y[mask]
        d = {}
        for k, v in feats.items():
            vv = np.array(v)[mask]
            a = auroc(vv, yy)
            # signal may predict failure in either direction; report the informative one
            d[k] = round(max(a, 1 - a), 4) if a == a else None
        return d

    def probe_aurocs(mask):
        yy = y[mask]
        return {str(L): (round(loocv_auroc(np.stack(hid[L])[mask], yy), 4)
                         if loocv_auroc(np.stack(hid[L])[mask], yy) == loocv_auroc(np.stack(hid[L])[mask], yy) else None)
                for L in layers}

    result = {
        "model": MID, "n": int(len(y)), "n_fail": int(y.sum()),
        "note": "AUROC for predicting the model's OWN trap failure (1=failed). "
                "Scalars folded to max(a,1-a). Small n_fail -> high variance.",
        "global": {
            "n_fail": int(y.sum()),
            "scalar_auroc": scalar_aurocs(np.ones_like(y, dtype=bool)),
            "hidden_probe_loocv_auroc": probe_aurocs(np.ones_like(y, dtype=bool)),
        },
        "pivot_detection": {
            "n": int(pivot.sum()), "n_fail": int(y[pivot].sum()),
            "scalar_auroc": scalar_aurocs(pivot),
            "hidden_probe_loocv_auroc": probe_aurocs(pivot),
        },
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
