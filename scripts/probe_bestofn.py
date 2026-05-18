#!/usr/bin/env python
"""Best-of-N upper bound probe.

For each of M held-out target vectors v:
  - sample K rollouts at temperature T from the warm-started AV
  - compute fve_nrm reward for each
  - report greedy reward, mean sampled reward, top-1 of K, top-5 of K mean

This tells us the ceiling for any expert-iteration / RL approach on this
checkpoint.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("probe_bestofn")


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def fve_nrm(h, v):
    return 1.0 - ((normalize(h) - normalize(v)) ** 2).sum(-1) / 2.0


@torch.no_grad()
def extract_last_hidden(backbone, gen_ids, gen_attn, layer):
    out = backbone(
        input_ids=gen_ids,
        attention_mask=gen_attn,
        output_hidden_states=True,
        use_cache=False,
    )
    h = out.hidden_states[layer]  # (B, T, d)
    # last valid token per row
    last = gen_attn.sum(-1) - 1  # (B,)
    idx = last.clamp(min=0).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, idx, :]  # (B, d)


@torch.no_grad()
def rollout_and_reward(av, backbone, v, K, T, temperature, layer, eos_id, pad_id):
    B = v.size(0)
    # replicate v K times → (B*K, d)
    v_rep = v.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    gen_ids = av.generate(
        v_rep,
        max_new_tokens=T,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-6),
        top_p=1.0,
    )
    # build attn mask: positions up to (and including) first EOS valid
    eos_mask = (gen_ids == eos_id)
    first_eos = torch.where(
        eos_mask.any(-1),
        eos_mask.float().argmax(-1),
        torch.full((gen_ids.size(0),), gen_ids.size(1) - 1, device=gen_ids.device),
    )
    arange = torch.arange(gen_ids.size(1), device=gen_ids.device).unsqueeze(0)
    attn = (arange <= first_eos.unsqueeze(1)).long()
    h_last = extract_last_hidden(backbone, gen_ids, attn, layer)
    rew = fve_nrm(h_last, v_rep)  # (B*K,)
    return rew.view(B, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-vectors", type=int, default=64)
    ap.add_argument("--samples-per-v", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-prefix-tokens", type=int, default=0)
    ap.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    ap.add_argument("--prefix-mlp-hidden", type=int, default=256)
    ap.add_argument("--num-inject-slots", type=int, default=1)
    ap.add_argument("--batch-vectors", type=int, default=2,
                    help="how many v's to process per generation call (B). K samples each.")
    ap.add_argument("--out", default="artifacts/nla/probe_bestofn.json")
    args = ap.parse_args()

    device = "cuda"
    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = pad_id

    cfg = NLAConfig(
        backbone_id=args.backbone,
        num_prefix_tokens=args.num_prefix_tokens,
        extraction_layer=args.layer,
        prefix_mode=args.prefix_mode,
        prefix_mlp_hidden=args.prefix_mlp_hidden,
        num_inject_slots=args.num_inject_slots,
    )
    av = ActivationVerbalizer(cfg, backbone, tok).to(device).eval()

    log.info("warm-start from %s", args.init_from)
    sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                log.info("unwrapped checkpoint key=%s", k)
                break
    log.info("top-level keys (first 6): %s", list(sd.keys())[:6])
    missing, unexpected = av.load_state_dict(sd, strict=False)
    log.info("warm-start: missing=%d unexpected=%d", len(missing), len(unexpected))

    log.info("loading targets %s", args.targets)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=False)
    acts = tgt["activations"]  # list of (T, d)
    pool = [a[-1].float() for a in acts]  # last-token hidden
    pool = torch.stack(pool, 0)  # (N, d)
    log.info("pool %s", tuple(pool.shape))

    # take first M v's for reproducibility
    M = args.num_vectors
    V = pool[:M].to(device)

    K = args.samples_per_v
    T = args.rollout_len
    B = args.batch_vectors

    log.info("probing M=%d K=%d T=%d temp=%.2f", M, K, T, args.temperature)

    # greedy reward per v
    greedy_rew = []
    sampled_rew_all = []  # (M, K)
    for i in range(0, M, B):
        v = V[i:i + B]
        # greedy
        g = rollout_and_reward(av, backbone, v, K=1, T=T, temperature=0.0,
                               layer=args.layer, eos_id=eos_id, pad_id=pad_id)
        greedy_rew.append(g.squeeze(1).cpu())
        # sampled K
        s = rollout_and_reward(av, backbone, v, K=K, T=T,
                               temperature=args.temperature,
                               layer=args.layer, eos_id=eos_id, pad_id=pad_id)
        sampled_rew_all.append(s.cpu())
        log.info("vec %d-%d  greedy_mean=%.4f  sampled_mean=%.4f  sampled_max=%.4f",
                 i, i + v.size(0), g.mean().item(), s.mean().item(), s.max().item())

    greedy_rew = torch.cat(greedy_rew, 0)  # (M,)
    sampled_rew = torch.cat(sampled_rew_all, 0)  # (M, K)

    # stats
    top1 = sampled_rew.max(dim=1).values  # (M,)
    top5 = sampled_rew.topk(min(5, K), dim=1).values.mean(dim=1)  # (M,)
    mean_sampled = sampled_rew.mean(dim=1)  # (M,)

    summary = {
        "M": M,
        "K": K,
        "T": T,
        "temperature": args.temperature,
        "greedy_mean": float(greedy_rew.mean()),
        "greedy_p50": float(greedy_rew.median()),
        "sampled_mean": float(mean_sampled.mean()),
        "sampled_max_mean": float(top1.mean()),
        "sampled_max_p50": float(top1.median()),
        "sampled_top5_mean": float(top5.mean()),
        "greedy_per_v": greedy_rew.tolist(),
        "top1_per_v": top1.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=== SUMMARY ===")
    log.info("greedy mean=%.4f  p50=%.4f", summary["greedy_mean"], summary["greedy_p50"])
    log.info("sampled  mean=%.4f", summary["sampled_mean"])
    log.info("best-of-%d  mean=%.4f  p50=%.4f", K,
             summary["sampled_max_mean"], summary["sampled_max_p50"])
    log.info("top5-of-%d mean=%.4f", K, summary["sampled_top5_mean"])
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
