"""REINFORCE / policy-gradient SFT for SRT-NLA.

Trains AV to generate sequences that reproduce v_target under the
*discrete-eval* objective. Reward = fve_nrm(backbone(generated_ids), v_target).

Pipeline per batch:
  1. Sample rollout (no_grad) via AV.generate(do_sample=True).
  2. Compute reward (no_grad): forward(gen_ids) -> h@L@last -> fve_nrm.
  3. Compute log p(gen_ids | v) with a single grad forward on
     [proj(v), embed(gen_ids[:-1])]. Gradient flows through proj only.
  4. Loss = -mean( advantage * sum_t log p(a_t) ) - beta * entropy.

Memory: one grad forward over ~P+T tokens — fits easily on a 96GB card.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from srt.nla import ActivationVerbalizer, NLAConfig

logger = logging.getLogger("train_nla_pg")

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--num-prefix-tokens", type=int, default=0)
    p.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    p.add_argument("--prefix-mlp-hidden", type=int, default=256)
    p.add_argument("--num-inject-slots", type=int, default=1)
    p.add_argument("--kl-beta", type=float, default=0.0,
                   help="KL(pi || pi_init) penalty on rollout tokens; 0 disables")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rollout-len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-v", type=int, default=4,
                   help="independent rollouts per target — provides REINFORCE baseline")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--entropy-beta", type=float, default=0.001)
    p.add_argument("--ce-weight", type=float, default=0.0,
                   help="optional CE-on-gold-tokens regularizer for stability")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-vectors", type=int, default=256)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def _load_targets(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return torch.stack([a[-1] for a in obj["activations"]])


def _load_pairs(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _fve_nrm(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    hn = F.normalize(h.float(), dim=-1)
    vn = F.normalize(v.float(), dim=-1)
    return 1.0 - ((hn - vn) ** 2).sum(dim=-1) / 2.0


def _last_h(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    lengths = attn.sum(dim=-1)
    last_idx = (lengths - 1).clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPE[args.dtype]

    cfg = NLAConfig(
        backbone_id=args.backbone,
        backbone_dtype=args.dtype,
        extraction_layer=args.layer,
        num_prefix_tokens=args.num_prefix_tokens,
        prefix_mode=args.prefix_mode,
        prefix_mlp_hidden=args.prefix_mlp_hidden,
        num_inject_slots=args.num_inject_slots,
    )

    logger.info("loading backbone %s", args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    eos_id = tok.eos_token_id

    av = ActivationVerbalizer(cfg, backbone=backbone, tokenizer=tok).to(device)
    trainable = [p for p in av.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("AV num_prefix=%d trainable=%s", args.num_prefix_tokens, f"{n_trainable:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.init_from is not None:
        logger.info("warm-start AV from %s", args.init_from)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["trainable"] if isinstance(ckpt, dict) and "trainable" in ckpt else ckpt
        own = av.state_dict()
        compat = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = av.load_state_dict(compat, strict=False)
        logger.info("warm-start: %d/%d keys (missing=%d unexpected=%d)",
                    len(compat), len(state), len(missing), len(unexpected))

    # Snapshot the post-warm-start (or freshly-initialised) policy for KL
    # regularisation. Frozen, no_grad, used only to score actions sampled by
    # the live AV. This is the standard "KL to reference" PPO/RLHF trick to
    # prevent the mode collapse seen in earlier PG runs (n1f/n1g).
    av_init = copy.deepcopy(av)
    for p in av_init.parameters():
        p.requires_grad = False
    av_init.eval()

    pool = _load_targets(args.targets).to(device)
    pairs = _load_pairs(args.pairs)
    logger.info("loaded %d pairs; target pool N=%d", len(pairs), pool.size(0))

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_pairs = [pairs[i] for i in val_idx[: args.val_vectors]]
    logger.info("train pairs=%d  val pairs=%d", len(train_idx), len(val_pairs))

    embed_matrix = backbone.get_input_embeddings().weight  # (V,d) frozen, bf16
    pad_id = tok.pad_token_id

    steps_per_epoch = max(1, (len(train_idx) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = None
    if args.lr_schedule == "cosine":
        warmup = max(0, args.warmup_steps)

        def _lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / float(warmup)
            progress = (s - warmup) / max(1, total_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        logger.info("cosine LR: total_steps=%d warmup=%d peak=%g", total_steps, warmup, args.lr)

    @torch.no_grad()
    def gen_and_reward(v_rep: torch.Tensor, sample: bool, temp: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample T tokens and compute reward via re-embed forward.

        Returns (gen_ids (B,T), gen_attn (B,T), reward (B,)).
        """
        gen_ids = av.generate(
            v_rep, max_new_tokens=args.rollout_len,
            do_sample=sample, temperature=temp, top_p=1.0,
        )
        B_, T_ = gen_ids.shape
        is_eos = gen_ids == eos_id
        pos = torch.arange(T_, device=device).unsqueeze(0).expand(B_, T_)
        first_eos = torch.where(
            is_eos.any(dim=-1, keepdim=True),
            is_eos.float().argmax(dim=-1, keepdim=True).long(),
            torch.full((B_, 1), T_, device=device, dtype=torch.long),
        )
        gen_attn = (pos <= first_eos).long()
        out = backbone(input_ids=gen_ids, attention_mask=gen_attn,
                       output_hidden_states=True, use_cache=False)
        h = _last_h(out.hidden_states[args.layer], gen_attn)
        rew = _fve_nrm(h, v_rep)
        return gen_ids, gen_attn, rew

    @torch.no_grad()
    def run_val() -> float:
        av.eval()
        fve_sum = 0.0
        n = 0
        for i in range(0, len(val_pairs), args.batch_size):
            batch = val_pairs[i : i + args.batch_size]
            v = pool[[r["target_idx"] for r in batch]].to(device)
            _gid, _att, rew = gen_and_reward(v, sample=False, temp=1.0)
            fve_sum += float(rew.sum().item())
            n += v.size(0)
        av.train()
        return fve_sum / max(n, 1)

    def logprobs_with_grad(v_rep: torch.Tensor, gen_ids: torch.Tensor, gen_attn: torch.Tensor):
        """Compute log p(gen_ids | proj(v)) and entropy per (B,T). Grad through proj only."""
        return _logprobs(av, v_rep, gen_ids, gen_attn, with_grad=True)

    def _logprobs(av_mod, v_rep: torch.Tensor, gen_ids: torch.Tensor,
                  gen_attn: torch.Tensor, with_grad: bool):
        B_, T_ = gen_ids.shape
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            prefix = av_mod._inject_prefix(v_rep)  # (B, P, d)
            P_ = prefix.size(1)
            gen_in = gen_ids[:, :-1]
            tok_emb = F.embedding(gen_in, embed_matrix).to(prefix.dtype).detach()
            full_in = torch.cat([prefix, tok_emb], dim=1)
            prefix_attn = torch.ones(B_, P_, dtype=torch.long, device=device)
            full_attn = torch.cat([prefix_attn, gen_attn[:, :-1]], dim=1)
            out = backbone(inputs_embeds=full_in, attention_mask=full_attn,
                           output_hidden_states=False, use_cache=False)
            logits = out.logits[:, P_ - 1 : P_ - 1 + T_, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            nll = -log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1)
        return nll, ent

    best_val = -1.0
    best_path = args.out / "best_av.pt"
    bad = 0
    step = 0
    t0 = time.time()
    log_loss = 0.0
    log_rew = 0.0
    log_n = 0

    K = args.samples_per_v
    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        shuffled = torch.randperm(len(train_idx)).tolist()
        epoch_pairs = [pairs[train_idx[i]] for i in shuffled]
        for i in range(0, len(epoch_pairs), args.batch_size):
            batch = epoch_pairs[i : i + args.batch_size]
            v = pool[[r["target_idx"] for r in batch]].to(device)  # (B, d)
            B_ = v.size(0)
            # Replicate v across K samples per target for baselining.
            v_rep = v.unsqueeze(1).expand(B_, K, -1).reshape(B_ * K, -1)  # (B*K, d)

            # 1+2. sample + reward (no_grad)
            gen_ids, gen_attn, reward = gen_and_reward(v_rep, sample=True, temp=args.temperature)
            # baseline: per-v mean reward across K samples
            rew_mat = reward.view(B_, K)
            baseline = rew_mat.mean(dim=1, keepdim=True)
            adv = (rew_mat - baseline).reshape(B_ * K).detach()

            # 3. log probs (grad)
            nll, ent = logprobs_with_grad(v_rep, gen_ids, gen_attn)
            # mask out positions beyond first EOS
            mask = gen_attn.float()
            # sum log prob over valid tokens per sequence
            log_p_seq = -(nll * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            ent_per_seq = (ent * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            pg_loss = -(adv * log_p_seq).mean()
            ent_loss = -args.entropy_beta * ent_per_seq.mean()
            loss = pg_loss + ent_loss
            # KL-to-reference (init snapshot). Approx token-level KL using
            # sampled-action log-prob diff: log pi_cur(a) - log pi_init(a).
            kl_val = 0.0
            if args.kl_beta > 0:
                with torch.no_grad():
                    nll_init, _ = _logprobs(av_init, v_rep, gen_ids, gen_attn, with_grad=False)
                kl_tok = (-nll) - (-nll_init.detach())  # log_cur - log_init at action
                kl_per_seq = (kl_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                kl_loss = args.kl_beta * kl_per_seq.mean()
                loss = loss + kl_loss
                kl_val = float(kl_per_seq.mean().item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            log_loss += float(loss.item()) * B_
            log_rew += float(reward.mean().item()) * B_
            log_n += B_

            if step % args.log_every == 0:
                logger.info(
                    "ep%d step %d  loss=%.4f  mean_rew=%.4f  adv_std=%.4f  ent=%.3f  kl=%.4f  (%.1fs)",
                    epoch, step, log_loss / log_n, log_rew / log_n,
                    float(adv.std().item()), float(ent_per_seq.mean().item()),
                    kl_val,
                    time.time() - t0,
                )
                log_loss = 0.0
                log_rew = 0.0
                log_n = 0

            if step % args.val_every == 0:
                v_fve = run_val()
                logger.info("val @ %d  disc_fve(greedy)=%.4f", step, v_fve)
                if v_fve > best_val:
                    best_val = v_fve
                    bad = 0
                    state = {n: p.detach().cpu() for n, p in av.named_parameters() if p.requires_grad}
                    torch.save({"trainable": state, "step": step, "val_disc": v_fve,
                                "cfg": vars(args)}, best_path)
                    logger.info("  new best %.4f  saved %s", v_fve, best_path)
                else:
                    bad += 1
                    logger.info("  no improvement (%d/%d)", bad, args.patience)
                    if args.patience > 0 and bad >= args.patience:
                        logger.info("early stop")
                        return

    v_fve = run_val()
    logger.info("final disc_fve=%.4f  best=%.4f", v_fve, best_val)


if __name__ == "__main__":
    main()
