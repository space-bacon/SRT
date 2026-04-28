#!/usr/bin/env python3
"""Separatrix-illusion probe evaluator.

Runs the SRT adapter over data/probes/separatrix_illusion_v1.jsonl and tests
predictions (1), (3), (4) from data/probes/SEPARATRIX_ILLUSION_README.md:

  (1) The Community Head should place `technical_continuation` and
      `mystical_continuation` in different prototype clusters; the
      `bedrock_continuation` should land near `technical`.

      Operationalisation: cosine similarity between community vectors of
      (technical, mystical) should be LOWER than (technical, bedrock).
      Reported per item and as a Wilcoxon signed-rank test across items.

  (3) BEN r_hat on the prompt (alone) should be high regardless of which
      continuation was attached — validating the activity-vs-alignment
      decomposition (\u00a76.5).

      Operationalisation: report mean r_hat over the prompt tokens for
      prompt_only branch. The 'high' threshold is set relative to the BEN's
      mean output on a generic-text reference set (passed via --ref-rhat).

  (4) MAH peak divergence on the (technical || mystical) sequence should
      exceed peak divergence on the (technical || bedrock) sequence.

      Operationalisation: For each item, compute peak ||MAH divergence|| over
      a sliding window of size --mah-window for both pairs; report median
      delta and Wilcoxon test.

Prediction (2) (counterfactual decoding under a forced prototype) is left
for a separate script — it requires an inference-time prototype-forcing
codepath that is not yet wired into v8a.

Outputs: artifacts/separatrix/v9/<run>/{readouts.jsonl, summary.json}.

Usage
-----

    python3 scripts/separatrix_eval.py \\
        --adapter /root/srt-adapter/release/srt-adapter-v8a/adapter.pt \\
        --probe data/probes/separatrix_illusion_v1.jsonl \\
        --output artifacts/separatrix/v8a/readouts.jsonl \\
        --ref-rhat 0.0 \\
        --max-seq-len 128

Local smoke test (no GPU; will fail on Qwen-7B without GPU):

    python3 scripts/separatrix_eval.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("separatrix_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter")
    p.add_argument("--probe", default="data/probes/separatrix_illusion_v1.jsonl")
    p.add_argument("--output", default="artifacts/separatrix/v8a/readouts.jsonl")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--mah-window", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    p.add_argument("--ref-rhat", type=float, default=0.0,
                   help="Reference r_hat for generic text (for prediction (3) framing).")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny self-test: exercises the math on synthetic data.")
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def sliding_peak(x: torch.Tensor, window: int) -> float:
    """Peak of mean over sliding window (1-D tensor)."""
    if x.numel() == 0:
        return float("nan")
    if x.numel() < window:
        return float(x.mean().item())
    pooled = F.avg_pool1d(x.view(1, 1, -1).float(), kernel_size=window, stride=1)
    return float(pooled.max().item())


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test (two-sided, no ties handling for ties=0).

    Returns z-statistic and approximate two-sided p-value via normal
    approximation. Adequate for n >= 20.
    """
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    if n == 0:
        return {"n": 0, "z": float("nan"), "p_two_sided": float("nan"),
                "median_delta": 0.0, "mean_delta": 0.0,
                "frac_positive": 0.0}
    abs_sorted = sorted(enumerate(nz), key=lambda kv: abs(kv[1]))
    # ranks (average for ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_sorted[j + 1][1]) == abs(abs_sorted[i][1]):
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[abs_sorted[k][0]] = avg_rank
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sigma if sigma > 0 else float("nan")
    # two-sided normal approx
    if math.isnan(z):
        p = float("nan")
    else:
        p = math.erfc(abs(z) / math.sqrt(2))
    return {
        "n": n,
        "z": z,
        "p_two_sided": p,
        "median_delta": sorted(deltas)[len(deltas) // 2],
        "mean_delta": sum(deltas) / len(deltas),
        "frac_positive": sum(1 for d in deltas if d > 0) / len(deltas),
    }


def self_test() -> int:
    """Exercise summary math on synthetic data; no model load."""
    log.info("self-test: verifying summary math on synthetic readouts")
    # Synthetic: 50 items where (tech, mystic) cosine is consistently lower
    # than (tech, bedrock) by ~0.2; MAH peak on tech-mystic > tech-bedrock.
    torch.manual_seed(0)
    K = 50
    cos_tm, cos_tb = [], []
    mah_tm, mah_tb = [], []
    rhat_prompt = []
    for _ in range(K):
        v_tech = torch.randn(64)
        v_mystic = v_tech + torch.randn(64) * 1.5  # noisy
        v_bedrock = v_tech + torch.randn(64) * 0.3  # close to tech
        cos_tm.append(cosine(v_tech, v_mystic))
        cos_tb.append(cosine(v_tech, v_bedrock))
        mah_tm.append(sliding_peak(torch.rand(50) + 0.4, 8))
        mah_tb.append(sliding_peak(torch.rand(50), 8))
        rhat_prompt.append(float(torch.randn(1).item()) + 0.5)

    delta_cos = [tb - tm for tb, tm in zip(cos_tb, cos_tm)]  # expect > 0
    delta_mah = [tm - tb for tm, tb in zip(mah_tm, mah_tb)]  # expect > 0
    cos_test = wilcoxon_signed_rank(delta_cos)
    mah_test = wilcoxon_signed_rank(delta_mah)
    log.info("cos(tech,bedrock) - cos(tech,mystic): median=%+.3f frac+=%.2f p=%.3g",
             cos_test["median_delta"], cos_test["frac_positive"], cos_test["p_two_sided"])
    log.info("mah_peak(t-m) - mah_peak(t-b): median=%+.3f frac+=%.2f p=%.3g",
             mah_test["median_delta"], mah_test["frac_positive"], mah_test["p_two_sided"])
    log.info("mean r_hat(prompt) over %d items: %.3f",
             K, sum(rhat_prompt) / len(rhat_prompt))
    assert cos_test["frac_positive"] > 0.7, "cosine direction wrong"
    assert mah_test["frac_positive"] > 0.7, "MAH direction wrong"
    log.info("self-test PASSED")
    return 0


@torch.no_grad()
def run_branch(
    model: SRTAdapter,
    tok,
    text: str,
    device: torch.device,
    max_seq_len: int,
    n_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Run adapter forward on one text; return per-branch summary stats.

    If n_prompt_tokens is provided, BEN r_hat and MAH divergence are
    additionally summarised over the *continuation* slice [n_prompt_tokens:].
    """
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_seq_len, padding=False)
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    T = int(att.sum())
    o = model(input_ids=ids, attention_mask=att)

    out: dict[str, Any] = {"n_tokens": T}

    # Community vector (use_prototypes=False in v8a: vector == encoded)
    if o.community_output is not None:
        v = o.community_output.vector[0].float().cpu()  # (d_comm,)
        out["community_vector"] = v.tolist()
        out["community_norm"] = float(v.norm())

    # BEN r_hat: prompt mean and continuation mean
    if o.ben_output is not None:
        r = o.ben_output.r_hat[0].float().cpu()  # (T,)
        out["r_hat_mean_full"] = float(r.mean())
        if n_prompt_tokens is not None and n_prompt_tokens < T:
            out["r_hat_mean_prompt"] = float(r[:n_prompt_tokens].mean())
            out["r_hat_mean_continuation"] = float(r[n_prompt_tokens:].mean())
        else:
            out["r_hat_mean_prompt"] = float(r.mean())

    # MAH peak divergence: take last MAH layer, ||div||, sliding window peak.
    if o.divergences:
        div_norms = o.divergences[-1].norm(dim=-1)[0].float().cpu()  # (T,)
        out["mah_peak_full"] = sliding_peak(div_norms, window=8)
        if n_prompt_tokens is not None and n_prompt_tokens < T:
            out["mah_peak_continuation"] = sliding_peak(
                div_norms[n_prompt_tokens:], window=8
            )

    return out


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not args.adapter:
        log.error("--adapter is required (or use --self-test)")
        return 2

    device = get_device(args.device)
    log.info("device=%s dtype=%s backbone=%s", device, args.dtype, args.backbone)

    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype=args.dtype)
    cfg.community.use_prototypes = False  # v8a default; consistent with interiority_probe.py
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    items = [json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip()]
    if args.limit > 0:
        items = items[: args.limit]
    log.info("loaded %d separatrix items", len(items))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_name("summary.json")

    rows = []
    cos_tm, cos_tb, cos_mb = [], [], []
    mah_tm_peaks, mah_tb_peaks = [], []
    rhat_prompts = []

    with out_path.open("w") as fout:
        for item in items:
            prompt = item["prompt"]
            n_prompt = int(tok(prompt, add_special_tokens=False,
                               return_tensors="pt")["input_ids"].shape[1])
            branches = {
                "prompt_only": prompt,
                "technical": prompt + item["technical_continuation"],
                "mystical": prompt + item["mystical_continuation"],
                "bedrock": prompt + item["bedrock_continuation"],
            }
            row: dict[str, Any] = {
                "id": item["id"], "concept": item["concept"],
                "domain": item["domain"], "n_prompt_tokens": n_prompt,
                "branches": {},
            }
            for name, text in branches.items():
                row["branches"][name] = run_branch(
                    model, tok, text, device, args.max_seq_len,
                    n_prompt_tokens=n_prompt if name != "prompt_only" else None,
                )
            # per-item derived metrics
            v_t = torch.tensor(row["branches"]["technical"].get("community_vector", []))
            v_m = torch.tensor(row["branches"]["mystical"].get("community_vector", []))
            v_b = torch.tensor(row["branches"]["bedrock"].get("community_vector", []))
            if v_t.numel() and v_m.numel() and v_b.numel():
                row["cos_tech_mystic"] = cosine(v_t, v_m)
                row["cos_tech_bedrock"] = cosine(v_t, v_b)
                row["cos_mystic_bedrock"] = cosine(v_m, v_b)
                cos_tm.append(row["cos_tech_mystic"])
                cos_tb.append(row["cos_tech_bedrock"])
                cos_mb.append(row["cos_mystic_bedrock"])

            mah_tm = row["branches"]["mystical"].get("mah_peak_continuation")
            mah_tb = row["branches"]["bedrock"].get("mah_peak_continuation")
            if mah_tm is not None and mah_tb is not None:
                mah_tm_peaks.append(mah_tm)
                mah_tb_peaks.append(mah_tb)

            r_prompt = row["branches"]["prompt_only"].get("r_hat_mean_prompt")
            if r_prompt is not None:
                rhat_prompts.append(r_prompt)

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            rows.append(row)

    # Summary tests
    delta_cos = [b - m for b, m in zip(cos_tb, cos_tm)]  # expect > 0 if pred (1) holds
    delta_mah = [m - b for m, b in zip(mah_tm_peaks, mah_tb_peaks)]  # expect > 0 if pred (4) holds
    cos_test = wilcoxon_signed_rank(delta_cos) if delta_cos else {}
    mah_test = wilcoxon_signed_rank(delta_mah) if delta_mah else {}
    summary = {
        "n_items": len(rows),
        "adapter": args.adapter,
        "backbone": args.backbone,
        "ref_rhat": args.ref_rhat,
        "prediction_1_community_separation": {
            "description": "cos(tech, bedrock) - cos(tech, mystic) > 0",
            "mean_cos_tech_mystic": (sum(cos_tm) / len(cos_tm)) if cos_tm else None,
            "mean_cos_tech_bedrock": (sum(cos_tb) / len(cos_tb)) if cos_tb else None,
            "mean_cos_mystic_bedrock": (sum(cos_mb) / len(cos_mb)) if cos_mb else None,
            "wilcoxon": cos_test,
        },
        "prediction_3_rhat_on_prompt": {
            "description": "r_hat on prompt > ref_rhat (prompts are interpretively dense)",
            "mean_rhat_prompt": (sum(rhat_prompts) / len(rhat_prompts)) if rhat_prompts else None,
            "n_prompts_above_ref": sum(1 for r in rhat_prompts if r > args.ref_rhat),
        },
        "prediction_4_mah_peak": {
            "description": "MAH peak on tech+mystic > MAH peak on tech+bedrock",
            "mean_mah_peak_mystic": (sum(mah_tm_peaks) / len(mah_tm_peaks)) if mah_tm_peaks else None,
            "mean_mah_peak_bedrock": (sum(mah_tb_peaks) / len(mah_tb_peaks)) if mah_tb_peaks else None,
            "wilcoxon": mah_test,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote %d rows -> %s", len(rows), out_path)
    log.info("wrote summary -> %s", summary_path)
    log.info("pred (1) frac+=%.2f p=%.3g | pred (4) frac+=%.2f p=%.3g",
             cos_test.get("frac_positive", float("nan")),
             cos_test.get("p_two_sided", float("nan")),
             mah_test.get("frac_positive", float("nan")),
             mah_test.get("p_two_sided", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
