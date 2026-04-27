#!/usr/bin/env python3
"""Per-position trajectory probe.

Same battery as `interiority_probe.py`, but dumps per-token (not mean-pooled)
readouts so we can ask: is the per-channel signal a token-local spike (e.g.
at a metaphorical word) or a prompt-global lift?

For each prompt, writes a JSON row with arrays of length T (real tokens):
  - tokens (decoded as strings)
  - r_hat (T,)
  - p_supercritical (T,)
  - regime_entropy (T,)
  - inj_norm_per_layer (n_inj_layers, T)
  - div_norm_per_layer (n_mah_layers, T)
  - chain_residual (T,)
The community vector is a single 64-D coordinate per prompt (the head pools
internally), so we keep one vector per row.

Output JSONL is larger than the mean-pooled probe: 275 prompts * ~30 tokens *
~10 floats per token \u2248 1 MB serialised.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trajectory")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--battery", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    log.info("device=%s dtype=%s", device, args.dtype)

    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    cfg.community.use_prototypes = False
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = [json.loads(l) for l in Path(args.battery).read_text().splitlines() if l.strip()]
    log.info("loaded %d prompts; MAH layers=%s inject layers=%s",
             len(items), cfg.mah_layer_indices, cfg.rrm_inject_indices)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_path.open("w") as fout:
        for item in items:
            enc = tok(item["text"], return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len, padding=False)
            ids = enc["input_ids"].to(device)
            att = enc["attention_mask"].to(device)
            T = int(att.sum())
            token_strs = tok.convert_ids_to_tokens(ids[0][:T].tolist())

            o = model(input_ids=ids, attention_mask=att)
            m = att.bool()[0]

            row: dict = {
                "id": item["id"],
                "label": item["label"],
                "n_tokens": T,
                "tokens": token_strs,
                "mah_layers": list(cfg.mah_layer_indices),
                "inject_layers": list(cfg.rrm_inject_indices),
            }

            # divergence norms per MAH layer, per token
            div_per_layer = []
            for div in o.divergences:
                norms = div.norm(dim=-1)[0]              # (T_pad,)
                div_per_layer.append(norms[m].float().cpu().tolist())
            row["div_norm_per_layer"] = div_per_layer

            # injection norms per RRM layer, per token
            inj_per_layer = []
            for inj in o.injections:
                norms = inj.norm(dim=-1)[0]
                inj_per_layer.append(norms[m].float().cpu().tolist())
            row["inj_norm_per_layer"] = inj_per_layer

            # BEN per token
            if o.ben_output is not None:
                row["r_hat"] = o.ben_output.r_hat[0][m].float().cpu().tolist()
                logits = o.ben_output.regime_logits[0][m].float().cpu()
                probs = torch.softmax(logits, dim=-1)
                row["p_supercritical"] = probs[:, 1].tolist()
                logp = torch.log_softmax(logits, dim=-1)
                row["regime_entropy"] = (-(probs * logp).sum(dim=-1)).tolist()

            # chain residual per token
            if o.chain_residual_per_token is not None:
                row["chain_residual"] = (
                    o.chain_residual_per_token[0][m].float().cpu().tolist()
                )

            # community vector: single 64-D coord per prompt (head pools)
            if o.community_output is not None:
                v = o.community_output.vector[0].float().cpu()
                row["community_norm"] = float(v.norm())
                row["community_vector"] = v.tolist()

            fout.write(json.dumps(row) + "\n")
            fout.flush()

            n_done += 1
            if n_done % 25 == 0:
                log.info("processed %d/%d", n_done, len(items))

    log.info("done: %d rows -> %s", n_done, out_path)


if __name__ == "__main__":
    main()
