#!/usr/bin/env python3
"""Resolve exact base/instruct repo IDs and total download size before pulling ~1 TB."""
import sys

from huggingface_hub import HfApi

api = HfApi()

CAND = [
    ("Ministral-3-3B", ["mistralai/Ministral-3-3B-Base-2512",
                        "mistralai/Ministral-3-3B-Instruct-2512"]),
    ("Ministral-3-8B", ["mistralai/Ministral-3-8B-Base-2512",
                        "mistralai/Ministral-3-8B-Instruct-2512"]),
    ("Ministral-3-14B", ["mistralai/Ministral-3-14B-Base-2512",
                         "mistralai/Ministral-3-14B-Instruct-2512"]),
    ("Qwen3.6-27B", ["Qwen/Qwen3.6-27B-Base", "Qwen/Qwen3.6-27B"]),
    ("Qwen3.8-27B", ["Qwen/Qwen3.8-27B-Base", "Qwen/Qwen3.8-27B"]),
    ("Qwen3.6-35B-A3B", ["Qwen/Qwen3.6-35B-A3B-Base", "Qwen/Qwen3.6-35B-A3B"]),
    ("Ornith-1.5-9B", ["ornith-ai/Ornith-1.5-9B-Base", "ornith-ai/Ornith-1.5-9B"]),
    ("Ornith-1.5-35B", ["ornith-ai/Ornith-1.5-35B-A3B-Base",
                        "ornith-ai/Ornith-1.5-35B-A3B"]),
    ("GLM-4.7-Flash", ["zai-org/GLM-4.7-Flash-Base", "zai-org/GLM-4.7-Flash"]),
    ("Qwen3-Coder-Next", ["Qwen/Qwen3-Coder-Next-Base", "Qwen/Qwen3-Coder-Next"]),
]


def size_of(repo):
    try:
        info = api.model_info(repo, expand=["safetensors"])
        st = getattr(info, "safetensors", None)
        return st.total if st and getattr(st, "total", None) else None
    except Exception:
        return None


def main():
    total = 0.0
    found = []
    header = "%-18s %-46s %9s %9s" % ("pair", "repo", "params", "bf16 GB")
    print(header)
    print("-" * len(header))
    for name, repos in CAND:
        ok = True
        for repo in repos:
            n = size_of(repo)
            if n is None:
                print("%-18s %-46s %9s" % (name, repo, "MISSING"))
                ok = False
            else:
                gb = n * 2 / 1e9
                total += gb
                print("%-18s %-46s %8.0fB %9.0f" % (name, repo, n / 1e9, gb))
        if ok:
            found.append(name)
    print("\ncomplete pairs: %d of %d" % (len(found), len(CAND)))
    print("total bf16 for complete pairs: %,.0f GB".replace("%,", "%") % total)


if __name__ == "__main__":
    main()
