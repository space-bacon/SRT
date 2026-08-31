#!/usr/bin/env python3
"""Fetch the three frontier code models, sequentially.

673 GB total against 1.2 TB free, so order matters and so does not running three
200 GB pulls at once into the same disk.

Written as a file rather than piped through a heredoc on purpose: the first
attempt died with SyntaxError because escaped quotes inside an f-string inside a
remote heredoc do not survive the trip, and it exited without fetching a byte
while looking like it had started.

    python scripts/download_frontier.py            # all three
    python scripts/download_frontier.py --only deepseek
"""
from __future__ import annotations

import argparse
import os
import time

REPOS = {
    "deepseek": ("deepseek-ai/DeepSeek-V4-Flash", 159.6),
    "qwen38": ("Qwen/Qwen3.8-Flash-Next-FP8", 185.6),
    "glm53": ("zai-org/GLM-5.3-Flash", 328.4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(REPOS), default=sorted(REPOS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/root/.hf_home")
    from huggingface_hub import snapshot_download

    # Largest last: the two smaller ones become usable while GLM is still landing.
    order = sorted(a.only, key=lambda k: REPOS[k][1])
    print(f"fetching {len(order)}: {', '.join(order)} "
          f"({sum(REPOS[k][1] for k in order):.0f} GB)", flush=True)

    failed = []
    for key in order:
        repo, gb = REPOS[key]
        t = time.time()
        print(f"=== {repo} ({gb} GB) start {time.strftime('%H:%M:%S')}", flush=True)
        try:
            path = snapshot_download(repo, max_workers=a.workers)
            print(f"=== {repo} OK {(time.time() - t) / 60:.1f} min -> {path}", flush=True)
        except Exception as e:
            print(f"=== {repo} FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
            failed.append(repo)

    print(f"DOWNLOADS-DONE ok={len(order) - len(failed)} failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
