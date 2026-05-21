"""Cross-backbone K-curve overlay (centred fve_nrm vs K) for SRT-NLA Stage 4.

Reads the three rerank_eval JSON files (Qwen v1, Llama-3.2-3B, Gemma-2-2B)
and renders one figure with three log-K curves on a shared y-axis,
plus per-backbone paraphrase-ceiling and random-floor reference lines.

Usage:
    python scripts/plot_cross_backbone_kcurve.py \
        --out artifacts/nla/plots/cross_backbone_kcurve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


BACKBONES = [
    {
        "label": "Qwen-2.5-7B (L20)",
        "rerank": "artifacts/nla/rerank_eval_ce_seq64_np16_v2.json",
        "oracle": "artifacts/nla/oracle_ceiling_30k_v2.json",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "label": "Llama-3.2-3B (L20)",
        "rerank": "artifacts/nla/llama32_3B/rerank_eval.json",
        "oracle": "artifacts/nla/llama32_3B/oracle_ceiling.json",
        "color": "#2ca02c",
        "marker": "s",
    },
    {
        "label": "Gemma-2-2B (L19)",
        "rerank": "artifacts/nla/gemma2_2B/rerank_eval_M200_K32.json",
        "oracle": "artifacts/nla/gemma2_2B/oracle_ceiling_M200.json",
        "color": "#d62728",
        "marker": "^",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/nla/plots/cross_backbone_kcurve.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=160)

    for b in BACKBONES:
        rer = json.loads(Path(b["rerank"]).read_text())
        ora = json.loads(Path(b["oracle"]).read_text())

        kc = rer["kcurve_oracle_top1_cen"]
        ks = sorted(int(k) for k in kc.keys())
        ys = [kc[str(k)] for k in ks]

        mu = rer.get("anisotropy_mu_norm", float("nan"))
        ax.plot(
            ks, ys,
            label=f"{b['label']}  ‖μ‖={mu:.1f}",
            color=b["color"], marker=b["marker"], lw=1.6, ms=6,
        )

        # paraphrase ceiling and random floor (centred)
        # field names differ slightly between Qwen v1 (older oracle) and the new ones
        para = ora.get("paraphrase_best_cen_fve_mean", ora.get("paraphrase_cen_fve_mean"))
        floor = ora.get("random_floor_cen_fve_mean", ora.get("random_floor_cen_fve"))
        if para is not None:
            ax.axhline(para, color=b["color"], lw=0.8, ls="--", alpha=0.6)
        if floor is not None:
            ax.axhline(floor, color=b["color"], lw=0.8, ls=":", alpha=0.45)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("K (best-of-K oracle rerank)")
    ax.set_ylabel("centred fve_nrm  (½(1 + cos(h − μ, v − μ)))")
    ax.set_title("SRT-NLA Stage 4: cross-backbone K-curve\n"
                 "dashed = paraphrase ceiling (per backbone), dotted = random floor")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_ylim(0.45, 0.90)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
