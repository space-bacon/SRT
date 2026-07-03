"""Train an SRT read-out adapter on frozen gemma-4-31B (multimodal omni model).

The read-out heads consume the residual stream, which in gemma-4 is modality-
unified (image soft-tokens flow through the same 60-layer text tower — see
scripts/cross_modal_semiosis.py). We train the heads on TEXT (labelled discourse
corpus) so that, per the semiotic hypothesis, they transfer to image signs.

This uses the EXTRACT-AND-TRAIN path: the frozen backbone runs under no_grad with
output_hidden_states, we tap hidden states at {community_layer, mah_layers}, and
train the standalone read-out heads on them. No manual-loop port of gemma-4's
forward is needed (verified: MAH/community/chain are pure functions of cached
hidden states).

v1 scope: community (SupCon on community_id) + MAH divergence + chain
(self-supervised) + divergence SupCon. r_hat/regime are DEFERRED because their
per-token r_true/chain_labels are aligned to the original SRT tokenizer, not
gemma-4's (community_id is per-passage, so tokenization-agnostic).

Grouped sampling (C communities x M passages per batch) is mandatory: SupCon has
zero gradient without same-community pairs in the batch (hard-learned, v5).

Output: checkpoints/gemma4_readout/best_readout.pt
Usage (venv transformers>=5):
    python scripts/train_gemma4_readout.py --corpus data/phase1_slice.jsonl \
        --steps 3000 --comm-per-batch 8 --per-comm 4
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="checkpoints/gemma4_readout/best_readout.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--comm-per-batch", type=int, default=8)
    p.add_argument("--per-comm", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--chain-weight", type=float, default=0.1)
    p.add_argument("--div-norm-weight", type=float, default=0.1)
    p.add_argument("--div-target-norm", type=float, default=1.0)
    p.add_argument("--max-passages", type=int, default=200000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(0); torch.manual_seed(0)

    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
    from srt.config import SRTConfig
    from srt.modules.mah import MetapragmaticAttentionHead
    from srt.modules.community import CommunityDiscoveryHead
    from srt.training.losses import (community_supcon_loss, chain_loss,
                                     community_entropy_loss, divergence_supcon_loss)

    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bb = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in bb.parameters():
        p.requires_grad_(False)

    cfg = SRTConfig(backbone_id=MID)
    L = bb.config.text_config.num_hidden_layers          # 60
    d = bb.config.text_config.hidden_size                # 5376
    cfg.resolve_layer_indices(L)
    comm_L = cfg.community_layer_idx
    mah_Ls = cfg.mah_layer_indices
    print(f"gemma-4 L={L} d={d} | community@{comm_L} MAH@{mah_Ls}", flush=True)

    dc = cfg.community.d_community
    community = CommunityDiscoveryHead(cfg.community, d).cuda()
    mah_heads = torch.nn.ModuleList([
        MetapragmaticAttentionHead(cfg.mah, d, d_community=dc) for _ in mah_Ls]).cuda()
    chain_predictor = torch.nn.Linear(cfg.mah.d_divergence, cfg.mah.d_divergence,
                                      bias=False).cuda()
    heads = torch.nn.ModuleList([community, mah_heads, chain_predictor])
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"trainable head params: {n_params/1e6:.2f}M", flush=True)

    # ---- load + index corpus by community ----
    by_comm: dict[int, list[dict]] = defaultdict(list)
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            if i >= args.max_passages:
                break
            r = json.loads(line)
            if r.get("text") and r.get("community_id") is not None:
                by_comm[int(r["community_id"])].append(
                    {"text": r["text"], "cid": int(r["community_id"])})
    comms = [c for c, v in by_comm.items() if len(v) >= args.per_comm]
    n = sum(len(by_comm[c]) for c in comms)
    print(f"loaded {n} passages across {len(comms)} communities "
          f"(>= {args.per_comm} each)", flush=True)

    def sample_batch() -> list[dict]:
        chosen = random.sample(comms, min(args.comm_per_batch, len(comms)))
        rows = []
        for c in chosen:
            rows.extend(random.sample(by_comm[c], args.per_comm))
        return rows

    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    W = cfg.loss

    def causal_mask(T: int) -> torch.Tensor:
        m = torch.full((T, T), float("-inf"), device="cuda")
        return torch.triu(m, diagonal=1).view(1, 1, T, T)

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rows = sample_batch()
        texts = [r["text"] for r in rows]
        cids = torch.tensor([r["cid"] for r in rows], device="cuda")
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_seq_len).to("cuda")
        ids, attn = enc.input_ids, enc.attention_mask
        with torch.no_grad():
            hs = bb(input_ids=ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False).hidden_states
        cmask = causal_mask(ids.shape[1])
        comm_out = community(hs[comm_L].float(), attention_mask=attn)
        # Detach the community vector before conditioning MAH: the community head
        # is the prize (its transfer to image signs is the goal), so MAH/chain
        # gradients must not flow back into it and corrupt it.
        comm_vec = comm_out.vector.detach()
        divs = []
        for i, Li in enumerate(mah_Ls):
            mo = mah_heads[i](hs[Li].float(), community_vec=comm_vec,
                              causal_mask=cmask)
            divs.append(mo.divergence)

        l_sup, diag = community_supcon_loss(comm_out.encoded, cids,
                                            temperature=W.community_supcon_temperature)
        l_chain = chain_loss(divs, chain_predictor, attention_mask=attn)
        l_dsup, _ = divergence_supcon_loss(divs, cids, attention_mask=attn,
                                           temperature=W.divergence_supcon_temperature)
        l_ent = (community_entropy_loss(comm_out.weights)
                 if comm_out.weights is not None else torch.zeros((), device="cuda"))
        # Target-norm penalty caps divergence magnitude so the self-supervised
        # chain MSE cannot run away as dsup makes divergences more discriminative.
        # Norm over the feature dim on the raw (nonzero) divergence, then mask —
        # taking sqrt of masked-to-zero padded positions gives inf-gradient NaNs.
        am = attn.float()
        dnorms = [((dv.norm(dim=-1) * am).sum() / am.sum().clamp(min=1)) for dv in divs]
        l_dnorm = sum((dn - args.div_target_norm).pow(2) for dn in dnorms) / len(dnorms)
        loss = (W.community_supcon_weight * l_sup
                + args.chain_weight * l_chain
                + W.divergence_supcon_weight * l_dsup
                + args.div_norm_weight * l_dnorm
                + W.community_entropy_weight * l_ent)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            rate = step / (time.time() - t0)
            print(f"step {step:5d} loss {loss.item():.3f} | supcon {l_sup.item():.3f} "
                  f"(pos {diag['pos_pairs']}, cls {diag['unique_classes']}) | "
                  f"chain {l_chain.item():.4f} | dsup {l_dsup.item():.3f} | "
                  f"dnorm {l_dnorm.item():.3f} | "
                  f"{rate:.2f} it/s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if loss.item() < best:
                best = loss.item()
            torch.save({"heads": heads.state_dict(),
                        "config": {"community_layer": comm_L, "mah_layers": mah_Ls,
                                   "d_backbone": d, "d_community": dc,
                                   "backbone": MID},
                        "step": step, "loss": loss.item()}, args.out)
    print(f"saved {args.out} (best loss {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
