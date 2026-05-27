"""Compare v22c_a050 adapter encoder vs raw frozen-Qwen L28 mean-pool, head-to-head,
on the same 5-task MTEB-STS subset, using identical Spearman computation.

The point: confirm/refute whether the v22c_a050 adapter (paper's MTEB-STS SOTA
checkpoint) actually adds STS-relevant signal on top of the raw frozen backbone.

Usage::

    python scripts/evals/mteb_sts_v22c_vs_raw.py \\
        --repo RiverRider/srt-adapter-v22c_a050 \\
        --tasks biosses stsb sickr sts17 sts22 \\
        --batch-size 8 --max-len 128
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import (
    BENConfig,
    CommunityConfig,
    LossConfig,
    MAHConfig,
    RRMConfig,
    SRTConfig,
)

logger = logging.getLogger("mteb_sts_v22c_vs_raw")


TASKS = {
    "biosses": ("mteb/biosses-sts", None, "test", "sentence1", "sentence2", "score"),
    "stsb": ("mteb/stsbenchmark-sts", None, "test", "sentence1", "sentence2", "score"),
    "sickr": ("mteb/sickr-sts", None, "test", "sentence1", "sentence2", "score"),
    "sts17": ("mteb/sts17-crosslingual-sts", "en-en", "test", "sentence1", "sentence2", "score"),
    "sts22": ("mteb/sts22-crosslingual-sts", "en", "test", "sentence1", "sentence2", "score"),
}


def build_config(p: Path) -> SRTConfig:
    raw = json.loads(p.read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(
            **{k: v for k, v in raw["loss"].items() if k in LossConfig.__dataclass_fields__}
        ),
    )


def load_adapter(repo: str, device: str, config_repo: str | None = None):
    # Some checkpoint repos (e.g. v22c_a050) ship only best_adapter.pt without
    # a config.json. Fall back to a sibling repo that publishes the canonical
    # architecture config (defaults to v1.0 = v15a, same arch family).
    cfg_repo = config_repo or repo
    try:
        cfg_path = Path(hf_hub_download(cfg_repo, "config.json"))
    except Exception:
        logger.warning("no config.json in %s; falling back to RiverRider/srt-adapter-v1.0", cfg_repo)
        cfg_path = Path(hf_hub_download("RiverRider/srt-adapter-v1.0", "config.json"))
    cfg = build_config(cfg_path)
    # v22c_a050 ships best_adapter.pt; v1.0 ships adapter.safetensors.
    try:
        w_path = hf_hub_download(repo, "adapter.safetensors")
        from safetensors.torch import load_file
        state = load_file(w_path, device="cpu")
    except Exception:
        w_path = hf_hub_download(repo, "best_adapter.pt")
        state = torch.load(w_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    model = SRTAdapter(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("adapter loaded: %d missing, %d unexpected", len(missing), len(unexpected))
    model.eval()
    return cfg, model


@torch.no_grad()
def encode_both(model, tok, sentences, *, device, batch_size, max_len, raw_layer):
    """Single forward pass returns (v22c_emb, raw_L_emb)."""
    v22c_chunks = []
    raw_chunks = []
    n = len(sentences)
    for i in range(0, n, batch_size):
        batch = sentences[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        # --- v22c_a050 published path ---
        encoded = out.community_output.encoded  # (B, T, d_c) or (B, d_c)
        attn = enc["attention_mask"]
        if encoded.dim() == 3:
            mask = attn.unsqueeze(-1).to(encoded.dtype)
            enc_vec = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        else:
            enc_vec = encoded
        v22c_chunks.append(F.normalize(enc_vec, p=2, dim=-1).to(torch.float32).cpu().numpy())
        # --- raw backbone L_raw mean-pool ---
        # The SRTAdapter forwards the frozen backbone with output_hidden_states; pull the
        # cached hidden states off the underlying model.
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            # fall back: re-run the frozen backbone directly (cheaper than alternatives)
            be = model.backbone(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                                output_hidden_states=True,
                                use_cache=False)
            hs = be.hidden_states
        h = hs[raw_layer]
        mask = attn.unsqueeze(-1).to(h.dtype)
        raw_vec = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        raw_chunks.append(raw_vec.to(torch.float32).cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info("  encode %d / %d", min(i + batch_size, n), n)
    return np.concatenate(v22c_chunks, 0), np.concatenate(raw_chunks, 0)


def cosine_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(1)


def run_task(name, model, tok, args, device):
    repo, cfg, split, c1, c2, cs = TASKS[name]
    logger.info("[%s] loading %s / %s / %s", name, repo, cfg or "-", split)
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    s1 = [str(x) for x in ds[c1]]
    s2 = [str(x) for x in ds[c2]]
    gold = np.asarray(ds[cs], dtype=np.float64)
    t0 = time.time()
    v1, r1 = encode_both(model, tok, s1, device=device, batch_size=args.batch_size,
                         max_len=args.max_len, raw_layer=args.raw_layer)
    v2, r2 = encode_both(model, tok, s2, device=device, batch_size=args.batch_size,
                         max_len=args.max_len, raw_layer=args.raw_layer)
    dt = time.time() - t0
    row = {"task": name, "n": int(len(gold)), "seconds": dt}
    for label, e1, e2 in (("v22c_a050", v1, v2), (f"raw_L{args.raw_layer}_mean", r1, r2)):
        sims = cosine_rows(e1, e2)
        rho = float(spearmanr(sims, gold).statistic)
        r = float(pearsonr(sims, gold).statistic)
        row[f"{label}_spearman_x100"] = rho * 100
        row[f"{label}_pearson_x100"] = r * 100
        logger.info("[%s] %-22s rho_x100=%6.2f  pearson_x100=%6.2f",
                    name, label, rho * 100, r * 100)
    return row


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="RiverRider/srt-adapter-v22c_a050")
    p.add_argument("--raw-layer", type=int, default=28,
                   help="layer for raw mean-pool comparison (default 28 = final hidden)")
    p.add_argument("--tasks", nargs="+", default=["biosses", "stsb", "sickr", "sts17", "sts22"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("artifacts/mteb_sts_v22c_vs_raw.json"))
    args = p.parse_args()
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("device=%s  repo=%s  raw_layer=%d", device, args.repo, args.raw_layer)
    cfg, model = load_adapter(args.repo, device)
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    for t in args.tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t}")
        rows.append(run_task(t, model, tok, args, device))

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "repo": args.repo,
        "raw_layer": args.raw_layer,
        "backbone": cfg.backbone_id,
        "tasks": rows,
        "mean_v22c_a050_spearman_x100": _mean("v22c_a050_spearman_x100"),
        "mean_raw_spearman_x100": _mean(f"raw_L{args.raw_layer}_mean_spearman_x100"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    logger.info("=" * 60)
    logger.info("MEAN v22c_a050           rho_x100 = %.2f", summary["mean_v22c_a050_spearman_x100"])
    logger.info("MEAN raw L%d mean-pool   rho_x100 = %.2f",
                args.raw_layer, summary["mean_raw_spearman_x100"])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
