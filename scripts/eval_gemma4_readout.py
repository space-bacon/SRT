"""Held-out validation for the gemma-4 SRT read-out adapter.

The trainer only logs training loss; this measures the thing that must actually
generalize (and then transfer cross-modally): does the trained community encoder
assign UNSEEN passages to the correct discourse community?

Protocol (all on held-out text disjoint from training):
  - reference split -> per-community centroid of community-encoder embeddings
  - query split -> nearest-centroid (cosine) top-1 + kNN(k=5) accuracy
  - chance = 1 / n_communities
Also scores MAH last-layer divergence (mean-pooled) the same way, to check the
dsup objective produced a community-discriminative divergence.

Usage:
    python scripts/eval_gemma4_readout.py --ckpt checkpoints/gemma4_readout/best_readout.pt \
        --corpus data/phase1_heldout.jsonl --out artifacts/nla/gemma4/readout_val.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus", required=True, help="held-out jsonl (NOT trained on)")
    p.add_argument("--out", default="artifacts/nla/gemma4/readout_val.json")
    p.add_argument("--per-comm", type=int, default=60,
                   help="passages per community (split half reference / half query)")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--knn-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(1); torch.manual_seed(1)

    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from srt.config import SRTConfig
    from srt.modules.mah import MetapragmaticAttentionHead
    from srt.modules.community import CommunityDiscoveryHead

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cc = ck["config"]
    comm_L, mah_Ls = cc["community_layer"], cc["mah_layers"]
    d, dc = cc["d_backbone"], cc["d_community"]
    print(f"ckpt step {ck.get('step')} | community@{comm_L} MAH@{mah_Ls} d={d}", flush=True)

    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bb = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in bb.parameters():
        p.requires_grad_(False)

    cfg = SRTConfig(backbone_id=MID)
    community = CommunityDiscoveryHead(cfg.community, d).cuda()
    mah_heads = torch.nn.ModuleList([
        MetapragmaticAttentionHead(cfg.mah, d, d_community=dc) for _ in mah_Ls]).cuda()
    heads = torch.nn.ModuleList([community, mah_heads,
                                 torch.nn.Linear(cfg.mah.d_divergence,
                                                 cfg.mah.d_divergence, bias=False).cuda()])
    heads.load_state_dict(ck["heads"])
    community.eval(); mah_heads.eval()

    # ---- load balanced held-out set ----
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

    def causal_mask(T: int) -> torch.Tensor:
        m = torch.full((T, T), float("-inf"), device="cuda")
        return torch.triu(m, diagonal=1).view(1, 1, T, T)

    @torch.no_grad()
    def embed(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (community_encoded, mah_divergence_pooled) for a list of texts."""
        ce, de = [], []
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_seq_len).to("cuda")
            ids, attn = enc.input_ids, enc.attention_mask
            hs = bb(input_ids=ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False).hidden_states
            co = community(hs[comm_L].float(), attention_mask=attn)
            cmask = causal_mask(ids.shape[1])
            mo = mah_heads[-1](hs[mah_Ls[-1]].float(), community_vec=co.vector,
                               causal_mask=cmask)
            m = attn.unsqueeze(-1).float()
            dpool = (mo.divergence * m).sum(1) / m.sum(1).clamp(min=1)
            ce.append(co.encoded.float().cpu()); de.append(dpool.float().cpu())
        return torch.cat(ce), torch.cat(de)

    # reference / query split per community
    ref_c, ref_lbl, qry_c, qry_lbl = [], [], [], []
    ref_d, qry_d = [], []
    for ci, c in enumerate(comms):
        texts = by_comm[c][:]
        random.shuffle(texts)
        half = len(texts) // 2
        ce, de = embed(texts)
        ref_c.append(ce[:half]); ref_d.append(de[:half])
        qry_c.append(ce[half:]); qry_d.append(de[half:])
        ref_lbl += [ci] * half; qry_lbl += [ci] * (len(texts) - half)
        print(f"  [{ci+1}/{n_comm}] comm {c}: {len(texts)} passages", flush=True)

    def score(ref, reflbl, qry, qrylbl, tag):
        ref = F.normalize(torch.cat(ref), dim=-1)
        qry = F.normalize(torch.cat(qry), dim=-1)
        reflbl = torch.tensor(reflbl); qrylbl = torch.tensor(qrylbl)
        # centroids
        cent = torch.stack([ref[reflbl == k].mean(0) for k in range(n_comm)])
        cent = F.normalize(cent, dim=-1)
        cent_pred = (qry @ cent.T).argmax(1)
        cent_acc = (cent_pred == qrylbl).float().mean().item()
        # kNN
        sims = qry @ ref.T
        knn = sims.topk(args.knn_k, dim=1).indices
        knn_lbl = reflbl[knn]
        knn_pred = torch.mode(knn_lbl, dim=1).values
        knn_acc = (knn_pred == qrylbl).float().mean().item()
        print(f"{tag}: centroid-top1 {cent_acc:.4f} | kNN@{args.knn_k} {knn_acc:.4f} "
              f"| chance {chance:.4f} | lift {cent_acc/chance:.1f}x", flush=True)
        return {"centroid_top1": cent_acc, "knn_acc": knn_acc,
                "chance": chance, "lift": cent_acc / chance}

    res = {
        "ckpt": args.ckpt, "step": ck.get("step"), "n_communities": n_comm,
        "n_query": len(qry_lbl),
        "community_encoder": score(ref_c, ref_lbl, qry_c, qry_lbl, "community-encoder"),
        "mah_divergence": score(ref_d, ref_lbl, qry_d, qry_lbl, "mah-divergence"),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
