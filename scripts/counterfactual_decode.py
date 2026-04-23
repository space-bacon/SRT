#!/usr/bin/env python3
"""Counterfactual community decoding (Tier-1 differentiator B).

For each prompt, greedy-decode N continuation tokens K times — once with
each of K community prototypes spliced in via SRTAdapter.forward()'s
forced_community argument. Same prompt, same backbone, only the discourse
community conditioning changes.

Output: a self-contained HTML table (rows=prompts, columns=communities)
plus a JSONL dump for downstream analysis. Quantitative metrics:

  - per-prompt token disagreement rate across communities at each position
  - per-prompt mean r̂ during generation under each community
  - per-prompt next-token KL between community-conditioned distributions

Usage:
    python scripts/counterfactual_decode.py \\
        --adapter checkpoints/adapter_v5/best_adapter.pt \\
        --prompts probes/inputs.txt \\
        --communities 6,13,18,19,21,31 \\
        --max-new-tokens 40 \\
        --out artifacts/counterfactual/v5_step17000.html
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt.cf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path,
                   help="One prompt per line.")
    p.add_argument("--communities", default=None,
                   help="Comma-separated community ids. Default: all prototypes.")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--out", required=True, type=Path,
                   help="Output HTML path. Sibling .jsonl is also written.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def cf_decode_one(
    model: SRTAdapter,
    tokenizer,
    prompt_ids: torch.Tensor,
    forced_community: torch.Tensor | None,
    max_new_tokens: int,
    max_seq_len: int,
) -> dict:
    """Greedy-decode one prompt under one community condition.

    Returns:
        {
            "generated_ids": list[int],
            "generated_text": str,
            "next_token_logits": list[Tensor]  per-step last-position logits (cpu, fp32),
            "r_hat_per_step": list[float]      mean r̂ over the *generated* token only,
        }
    """
    device = prompt_ids.device
    cur = prompt_ids.clone()  # (1, T)
    generated: list[int] = []
    step_logits: list[torch.Tensor] = []
    step_r_hat: list[float] = []

    for _ in range(max_new_tokens):
        if cur.size(1) >= max_seq_len:
            break

        out = model(
            input_ids=cur,
            attention_mask=torch.ones_like(cur),
            labels=None,
            forced_community=forced_community,
        )

        last_logits = out.logits[0, -1, :].float().cpu()
        next_id = int(last_logits.argmax())
        generated.append(next_id)
        step_logits.append(last_logits)

        if out.ben_output is not None:
            step_r_hat.append(float(out.ben_output.r_hat[0, -1].float()))

        next_tok = torch.tensor([[next_id]], device=device, dtype=cur.dtype)
        cur = torch.cat([cur, next_tok], dim=1)

        if next_id == tokenizer.eos_token_id:
            break

    return {
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "next_token_logits": step_logits,
        "r_hat_per_step": step_r_hat,
    }


def disagreement_rate(per_community: list[dict]) -> float:
    """Fraction of generated positions where at least two communities pick
    different next-token argmaxes. Compared up to the shortest run."""
    if len(per_community) < 2:
        return 0.0
    L = min(len(d["generated_ids"]) for d in per_community)
    if L == 0:
        return 0.0
    diffs = 0
    for i in range(L):
        ids = {d["generated_ids"][i] for d in per_community}
        if len(ids) > 1:
            diffs += 1
    return diffs / L


def mean_pairwise_kl(per_community: list[dict]) -> float:
    """Mean per-step symmetric KL between every pair of community
    next-token distributions, averaged over positions and pairs."""
    if len(per_community) < 2:
        return 0.0
    L = min(len(d["next_token_logits"]) for d in per_community)
    if L == 0:
        return 0.0

    total = 0.0
    n_pairs = 0
    for i in range(L):
        # stack distributions for this position: (K, V)
        log_p = torch.stack([
            F.log_softmax(d["next_token_logits"][i], dim=-1)
            for d in per_community
        ])
        p = log_p.exp()
        K = log_p.size(0)
        for a in range(K):
            for b in range(a + 1, K):
                kl_ab = (p[a] * (log_p[a] - log_p[b])).sum().item()
                kl_ba = (p[b] * (log_p[b] - log_p[a])).sum().item()
                total += 0.5 * (kl_ab + kl_ba)
                n_pairs += 1
    return total / max(n_pairs, 1) if n_pairs else 0.0


def render_html(rows: list[dict], community_ids: list[int], title: str) -> str:
    style = """
    body { font-family: -apple-system, system-ui, sans-serif; margin: 24px;
           background: #fafafa; color: #222; }
    h1 { font-size: 18px; margin-bottom: 8px; }
    .meta { color: #666; font-size: 12px; margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; background: white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    th, td { border: 1px solid #e0e0e0; padding: 8px 10px; vertical-align: top;
             font-size: 13px; }
    th { background: #f0f4f8; text-align: left; }
    .prompt { width: 22%; font-style: italic; color: #444; }
    .cont { width: 13%; }
    .stats { font-size: 11px; color: #666; margin-top: 8px; }
    .gen { color: #c0392b; }
    """
    out = [f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{style}</style></head><body>"]
    out.append(f"<h1>{html.escape(title)}</h1>")
    out.append("<div class='meta'>Same prompt + same backbone, "
               "only the community-conditioning vector changes. "
               "Greedy-decoded continuations under each prototype.</div>")

    out.append("<table><thead><tr>")
    out.append("<th class='prompt'>prompt</th>")
    for cid in community_ids:
        out.append(f"<th class='cont'>community {cid}</th>")
    out.append("</tr></thead><tbody>")

    for row in rows:
        out.append("<tr>")
        out.append(f"<td class='prompt'>{html.escape(row['prompt'])}<div class='stats'>"
                   f"disagree={row['disagreement_rate']:.2f} · "
                   f"meanKL={row['mean_pairwise_kl']:.3f}</div></td>")
        for cid in community_ids:
            cell = row["per_community"][cid]
            txt = html.escape(cell["generated_text"])
            mean_r = (sum(cell["r_hat_per_step"]) / max(len(cell["r_hat_per_step"]), 1)
                      if cell["r_hat_per_step"] else 0.0)
            out.append(f"<td class='cont'><span class='gen'>{txt}</span>"
                       f"<div class='stats'>r̂={mean_r:.2f}</div></td>")
        out.append("</tr>")

    out.append("</tbody></table></body></html>")
    return "".join(out)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    config = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    logger.info("Building model on %s ...", device)
    model = SRTAdapter(config).to(device)
    state = torch.load(args.adapter, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded adapter: missing=%d unexpected=%d",
                len(missing), len(unexpected))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve community ids
    prototypes = model.community_head.prototypes.weight  # (K, d_community)
    K = prototypes.size(0)
    if args.communities:
        community_ids = [int(x) for x in args.communities.split(",")]
        for cid in community_ids:
            if cid < 0 or cid >= K:
                raise SystemExit(f"community id {cid} out of range [0,{K})")
    else:
        community_ids = list(range(K))
    logger.info("Decoding under %d communities: %s", len(community_ids), community_ids)

    prompts = [
        line.rstrip("\n")
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d prompts", len(prompts))

    rows: list[dict] = []
    for pi, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=args.max_seq_len // 2)
        prompt_ids = enc["input_ids"].to(device)

        per_community: dict[int, dict] = {}
        for cid in community_ids:
            forced = prototypes[cid].unsqueeze(0).to(device)  # (1, d_community)
            res = cf_decode_one(
                model, tokenizer, prompt_ids,
                forced_community=forced,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=args.max_seq_len,
            )
            per_community[cid] = res

        # Stats across the K runs
        runs = list(per_community.values())
        d_rate = disagreement_rate(runs)
        kl = mean_pairwise_kl(runs)

        rows.append({
            "prompt": prompt,
            "disagreement_rate": d_rate,
            "mean_pairwise_kl": kl,
            "per_community": per_community,
        })
        logger.info("[%d/%d] disagree=%.2f  meanKL=%.3f  prompt=%r",
                    pi + 1, len(prompts), d_rate, kl, prompt[:60])

    # Write HTML
    title = f"SRT-Adapter counterfactual community decoding — {args.adapter.name}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(rows, community_ids, title), encoding="utf-8")

    # Write JSONL (drop logits, keep ids + text + r̂)
    jsonl_path = args.out.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "prompt": row["prompt"],
                "disagreement_rate": row["disagreement_rate"],
                "mean_pairwise_kl": row["mean_pairwise_kl"],
                "per_community": {
                    str(cid): {
                        "generated_ids": d["generated_ids"],
                        "generated_text": d["generated_text"],
                        "r_hat_per_step": d["r_hat_per_step"],
                    }
                    for cid, d in row["per_community"].items()
                },
            }) + "\n")

    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"Wrote {jsonl_path} ({jsonl_path.stat().st_size} bytes)")
    n = len(rows)
    print(f"Mean disagreement rate: {sum(r['disagreement_rate'] for r in rows)/n:.3f}")
    print(f"Mean pairwise KL:       {sum(r['mean_pairwise_kl'] for r in rows)/n:.3f}")


if __name__ == "__main__":
    main()
