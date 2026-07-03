"""Checkpoint selection sweep for the gemma-4 SRT read-out adapter.

Train loss is not a usable selector here: it is dominated by which communities
land in each grouped batch (see the noisy supcon column in training logs). This
scores EVERY retained step-tagged checkpoint on held-out community assignment and
reports the val-vs-step curve + the winner.

Efficiency: the frozen backbone's hidden states are checkpoint-independent, so we
run gemma-4 ONCE over the held-out set, cache the taps at {community_layer,
mah_layer}, then only swap the (cheap) head weights per checkpoint.

Usage:
    python scripts/select_gemma4_readout.py --ckpt-dir checkpoints/gemma4_readout \
        --corpus data/phase1_heldout.jsonl --out artifacts/nla/gemma4/readout_selection.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints/gemma4_readout")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="artifacts/nla/gemma4/readout_selection.json")
    p.add_argument("--per-comm", type=int, default=60)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--knn-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(1); torch.manual_seed(1)

    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from srt.config import SRTConfig
    from srt.modules.mah import MetapragmaticAttentionHead
    from srt.modules.community import CommunityDiscoveryHead

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*_step*.pt")),
                   key=lambda p: int(re.search(r"_step(\d+)\.pt$", p).group(1)))
    if not ckpts:
        raise SystemExit(f"no step-tagged checkpoints in {args.ckpt_dir}")
    print(f"found {len(ckpts)} checkpoints: "
          f"{[int(re.search(r'_step(\d+)', c).group(1)) for c in ckpts]}", flush=True)

    cc = torch.load(ckpts[0], map_location="cpu", weights_only=False)["config"]
    comm_L, mah_Ls = cc["community_layer"], cc["mah_layers"]
    mah_L = mah_Ls[-1]
    d, dc = cc["d_backbone"], cc["d_community"]

    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bb = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in bb.parameters():
        p.requires_grad_(False)

    # ---- balanced held-out set, fixed ref/query split ----
    by_comm: dict[int, list[str]] = defaultdict(list)
    with open(args.corpus) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text") and r.get("community_id") is not None:
                c = int(r["community_id"])
                if len(by_comm[c]) < args.per_comm:
                    by_comm[c].append(r["text"])
    comms = sorted(c for c, v in by_comm.items() if len(v) >= 4)
    n_comm = len(comms)
    chance = 1.0 / n_comm
    print(f"held-out: {n_comm} communities, chance {chance:.4f}", flush=True)

    # ---- cache backbone taps ONCE (checkpoint-independent) ----
    # Store per-passage padded hidden states (CPU) + mask + label + role.
    cache = []  # list of dict(hc=(T,d), hm=(T,d), mask=(T,), lbl, role)
    @torch.no_grad()
    def tap(texts):
        out = []
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_seq_len).to("cuda")
            hs = bb(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True, use_cache=False).hidden_states
            hc = hs[comm_L].float().cpu()
            hm = hs[mah_L].float().cpu()
            am = enc.attention_mask.cpu()
            for b in range(len(chunk)):
                out.append((hc[b], hm[b], am[b]))
        return out

    for ci, c in enumerate(comms):
        texts = by_comm[c][:]
        random.shuffle(texts)
        half = len(texts) // 2
        taps = tap(texts)
        for j, t in enumerate(taps):
            cache.append({"hc": t[0], "hm": t[1], "mask": t[2], "lbl": ci,
                          "role": "ref" if j < half else "qry"})
        print(f"  cached [{ci+1}/{n_comm}] comm {c}: {len(texts)}", flush=True)

    def causal_mask(T):
        m = torch.full((T, T), float("-inf"), device="cuda")
        return torch.triu(m, diagonal=1).view(1, 1, T, T)

    # ---- score each checkpoint from the cache ----
    cfg = SRTConfig(backbone_id=MID)

    def score_ckpt(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        community = CommunityDiscoveryHead(cfg.community, d).cuda().eval()
        mah_heads = torch.nn.ModuleList([
            MetapragmaticAttentionHead(cfg.mah, d, d_community=dc) for _ in mah_Ls]).cuda().eval()
        heads = torch.nn.ModuleList([community, mah_heads,
                                     torch.nn.Linear(cfg.mah.d_divergence,
                                                     cfg.mah.d_divergence, bias=False).cuda()])
        heads.load_state_dict(ck["heads"])
        ce, de, lbl, role = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(cache), args.batch):
                chunk = cache[i:i + args.batch]
                T = max(x["hc"].shape[0] for x in chunk)
                hc = torch.zeros(len(chunk), T, d); hm = torch.zeros(len(chunk), T, d)
                am = torch.zeros(len(chunk), T)
                for b, x in enumerate(chunk):
                    t = x["hc"].shape[0]
                    hc[b, :t] = x["hc"]; hm[b, :t] = x["hm"]; am[b, :t] = x["mask"]
                    lbl.append(x["lbl"]); role.append(x["role"])
                hc = hc.cuda(); hm = hm.cuda(); am = am.cuda()
                co = community(hc, attention_mask=am)
                mo = mah_heads[-1](hm, community_vec=co.vector, causal_mask=causal_mask(T))
                m = am.unsqueeze(-1)
                dpool = (mo.divergence * m).sum(1) / m.sum(1).clamp(min=1)
                ce.append(co.encoded.float().cpu()); de.append(dpool.float().cpu())
        ce = torch.cat(ce); de = torch.cat(de)
        lbl = torch.tensor(lbl); role = role

        def acc(emb):
            emb = F.normalize(emb, dim=-1)
            refm = torch.tensor([r == "ref" for r in role])
            qrym = ~refm
            ref, reflbl = emb[refm], lbl[refm]
            qry, qrylbl = emb[qrym], lbl[qrym]
            cent = F.normalize(torch.stack(
                [ref[reflbl == k].mean(0) for k in range(n_comm)]), dim=-1)
            c_top1 = ((qry @ cent.T).argmax(1) == qrylbl).float().mean().item()
            sims = qry @ ref.T
            knn = reflbl[sims.topk(args.knn_k, dim=1).indices]
            k_acc = (torch.mode(knn, dim=1).values == qrylbl).float().mean().item()
            return c_top1, k_acc

        c1, ck_ = acc(ce)
        d1, dk_ = acc(de)
        del community, mah_heads, heads
        torch.cuda.empty_cache()
        return {"community_top1": c1, "community_knn": ck_,
                "mah_top1": d1, "mah_knn": dk_}

    curve = []
    for path in ckpts:
        step = int(re.search(r"_step(\d+)\.pt$", path).group(1))
        r = score_ckpt(path)
        r["step"] = step
        curve.append(r)
        print(f"step {step:5d} | community top1 {r['community_top1']:.4f} "
              f"knn {r['community_knn']:.4f} | mah top1 {r['mah_top1']:.4f} "
              f"knn {r['mah_knn']:.4f} | chance {chance:.4f}", flush=True)

    best = max(curve, key=lambda r: r["community_top1"])
    print(f"\nBEST community checkpoint: step {best['step']} "
          f"top1 {best['community_top1']:.4f} ({best['community_top1']/chance:.1f}x chance)",
          flush=True)
    res = {"chance": chance, "n_communities": n_comm, "n_query": None,
           "best_step": best["step"], "curve": curve}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
