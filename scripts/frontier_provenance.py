#!/usr/bin/env python3
"""Record what the frontier trio actually is, so the numbers can be audited.

Every figure quoted about these models so far, the sizes, the download times, the
card capacity, existed only in terminal scrollback. That is the same failure that
left "110,704 generations" and the -0.0704 decay slope sitting in a published
card with nothing behind them: a number that only exists in prose is a number
nobody can recheck, including us.

Sizes come from the Hub API rather than from the plan note that first listed
them, because the note was written from memory and one of its figures (185.5 vs
185.6 GB) was already slightly wrong.

    python scripts/frontier_provenance.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import subprocess

OUT = pathlib.Path("artifacts/nla/frontier/provenance.json")
REPOS = [
    ("deepseek_v4_flash", "deepseek-ai/DeepSeek-V4-Flash", "DeepSeek"),
    ("qwen38_flash_next", "Qwen/Qwen3.8-Flash-Next-FP8", "Alibaba"),
    ("glm53_flash", "zai-org/GLM-5.3-Flash", "Zhipu"),
]
HUBS = [h for h in (os.environ.get("HF_HOME", "") and
                    os.path.join(os.environ["HF_HOME"], "hub"),
                    "/workspace/.hf_home/hub", "/root/.hf_home/hub") if h]


def snap(repo):
    stem = f"models--{repo.replace('/', '--')}/snapshots/*/"
    for hub in HUBS:
        g = glob.glob(f"{hub}/{stem}")
        if g:
            return g[0]
    return None


def gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return {}
    rows = [r.split(", ") for r in out.splitlines() if r.strip()]
    total = sum(int(r[2]) for r in rows)
    return {
        "count": len(rows),
        "name": rows[0][1] if rows else None,
        "mib_each": int(rows[0][2]) if rows else None,
        "mib_total": total,
        "gb_total": round(total * 1048576 / 1e9, 1),
    }


def main() -> int:
    from huggingface_hub import HfApi
    api = HfApi()

    members, total_gb = {}, 0.0
    for tag, repo, lab in REPOS:
        rec = {"repo": repo, "lab": lab}
        try:
            info = api.model_info(repo, files_metadata=True)
            gb = sum(f.size or 0 for f in info.siblings) / 1e9
            rec |= {"hub_gb": round(gb, 1), "n_files": len(info.siblings),
                    "sha": info.sha}
            total_gb += gb
        except Exception as e:
            rec["hub_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        p = snap(repo)
        rec["local"] = bool(p)
        if p:
            rec["path"] = p
        members[tag] = rec

    g = gpus()
    biggest = max((m.get("hub_gb", 0) for m in members.values()), default=0)
    res = {
        "question": "what are the frontier trio, and what does the hardware allow",
        "source": "sizes from the Hugging Face Hub API, not from the plan note",
        "members": members,
        "total_gb": round(total_gb, 1),
        "gpus": g,
        "fit": {
            "largest_model_gb": round(biggest, 1),
            "fits_on_one_card": bool(g.get("mib_each")) and
                                biggest * 1e9 < g["mib_each"] * 1048576,
            "cards_needed_largest": (int(biggest * 1e9 // (g["mib_each"] * 1048576)) + 1)
                                    if g.get("mib_each") else None,
            "note": "sequential over all cards: two-cards-each leaves too little "
                    "for KV cache on the 185.6 GB member",
        },
        "status": "generation not yet run; this file records inputs only",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    for tag, m in members.items():
        print(f"  {tag:20s} {m.get('hub_gb', '?'):>7} GB  "
              f"{'local' if m['local'] else 'not downloaded'}  {m['repo']}")
    print(f"\n  total {res['total_gb']} GB   gpus {g.get('count')} x "
          f"{g.get('mib_each')} MiB = {g.get('gb_total')} GB")
    print(f"  largest {res['fit']['largest_model_gb']} GB, "
          f"fits on one card: {res['fit']['fits_on_one_card']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
