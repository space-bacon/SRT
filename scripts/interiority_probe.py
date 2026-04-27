#!/usr/bin/env python3
"""Interiority probe — extract per-layer adapter readouts on the probe battery.

For each prompt in the battery, runs a forward pass on the SRT adapter and
records (averaged over real tokens, ignoring padding):

  - per-MAH-layer ‖divergence‖, mean & std across token positions
  - per-RRM-layer ‖injection‖
  - mean ben.r_hat
  - mean P(supercritical) from regime_logits
  - mean chain_residual_per_token
  - community_output.weights argmax + entropy
  - top-1 community weight value

Output: artifacts/interiority/v1/readouts.jsonl  (one row per prompt)

Usage (vast.ai A6000):

    python3 scripts/interiority_probe.py \\
        --adapter /root/srt-adapter/release/srt-adapter-v8a/adapter.pt \\
        --battery /root/srt-adapter/data/probes/probe_battery_v1.jsonl \\
        --output /root/srt-adapter/artifacts/interiority/v1/readouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
log = logging.getLogger("interiority")


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
    # v8a was trained with use_prototypes=False (encoder IS the community vector).
    # Honor that here so we don't compute softmax against untrained random
    # prototypes (which would just give entropy = ln K).
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

            o = model(input_ids=ids, attention_mask=att)

            # mask: real tokens, shape (T,)
            m = att.bool()[0]

            row: dict = {
                "id": item["id"],
                "label": item["label"],
                "n_tokens": T,
                "mah_layers": list(cfg.mah_layer_indices),
                "inject_layers": list(cfg.rrm_inject_indices),
            }

            # divergence norms per MAH layer
            div_means, div_stds = [], []
            for div in o.divergences:
                norms = div.norm(dim=-1)[0]  # (T,)
                vals = norms[m].float().cpu()
                div_means.append(float(vals.mean()))
                div_stds.append(float(vals.std()))
            row["div_norm_mean"] = div_means
            row["div_norm_std"] = div_stds

            # injection norms per RRM layer
            inj_means = []
            for inj in o.injections:
                norms = inj.norm(dim=-1)[0]
                vals = norms[m].float().cpu()
                inj_means.append(float(vals.mean()))
            row["inj_norm_mean"] = inj_means

            # BEN
            if o.ben_output is not None:
                r_hat = o.ben_output.r_hat[0][m].float().cpu()
                row["r_hat_mean"] = float(r_hat.mean())
                row["r_hat_std"] = float(r_hat.std())
                logits = o.ben_output.regime_logits[0][m].float().cpu()
                p_super = torch.softmax(logits, dim=-1)[:, 1]
                row["p_supercritical_mean"] = float(p_super.mean())
                row["regime_entropy_mean"] = float(
                    -(torch.softmax(logits, dim=-1) *
                      torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                )

            # community: in v8a the encoder output IS the community vector;
            # report norm + raw vector for centroid / between-class analysis.
            if o.community_output is not None:
                v = o.community_output.vector[0].float().cpu()  # (d_comm,)
                row["community_norm"] = float(v.norm())
                row["community_vector"] = v.tolist()
                # legacy (only populated if use_prototypes=True; left here
                # for compatibility with older readouts)
                if o.community_output.weights is not None:
                    w = o.community_output.weights[0].float().cpu()
                    row["community_argmax"] = int(w.argmax())
                    row["community_top1_weight"] = float(w.max())
                    eps = 1e-9
                    row["community_entropy"] = float(-(w * (w + eps).log()).sum())

            # chain residual
            if o.chain_residual_per_token is not None:
                cr = o.chain_residual_per_token[0][m].float().cpu()
                row["chain_residual_mean"] = float(cr.mean())

            fout.write(json.dumps(row) + "\n")
            fout.flush()

            n_done += 1
            if n_done % 25 == 0:
                log.info("processed %d/%d", n_done, len(items))

    log.info("done: %d rows -> %s", n_done, out_path)


if __name__ == "__main__":
    main()
