#!/usr/bin/env python3
"""Declare the host models our instruments read, so the Hub links them back.

`base_model` plus `base_model_relation` puts a repo in the host's model tree,
under Adapters or Quantizations. A Space's `models` list puts it in "Spaces
using this model" on each host's page. Both are how anyone browsing
google/gemma-4-31B-it or Qwen/Qwen3-0.6B finds this work at all.

Every relation here is literal. The heads and towers are fitted on states
pulled out of the named hosts and do nothing without them, and the NF4 repo is
that host quantised.

    python scripts/link_hosts.py            # report only
    python scripts/link_hosts.py --apply
"""
import argparse

from huggingface_hub import HfApi, get_token, metadata_update

OMNI = ["Qwen/Qwen3-Omni-30B-A3B-Instruct", "google/gemma-4-31B-it",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "rhymes-ai/Aria"]
GEMMA, Q06, Q27 = "google/gemma-4-31B-it", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.8-27B"

MODELS = {
    "RiverRider/gemma-4-31B-it-nf4": {
        "base_model": GEMMA, "base_model_relation": "quantized",
        "library_name": "transformers"},
    "RiverRider/srt-verbalizer-v1": {
        "base_model": [Q06, GEMMA], "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "text-generation"},
    "RiverRider/srt-sunstone-linear-head": {
        "base_model": [GEMMA, Q06], "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "feature-extraction"},
    "RiverRider/srt-nla-av-gemma4": {
        "base_model": [GEMMA], "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "feature-extraction"},
    "RiverRider/srt-nla-gemma4-artifacts": {
        "base_model": [GEMMA], "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "feature-extraction"},
    "RiverRider/srt-omni-shared-tower": {
        "base_model": OMNI, "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "feature-extraction"},
    "RiverRider/srt-omni-xvendor-towers": {
        "base_model": OMNI, "base_model_relation": "adapter",
        "library_name": "pytorch", "pipeline_tag": "feature-extraction"},
}

SPACES = {
    "RiverRider/0.6b-decodes-31b": {
        "models": [GEMMA, Q06, "RiverRider/gemma-4-31B-it-nf4",
                   "RiverRider/srt-verbalizer-v1",
                   "RiverRider/srt-browser-head-118k"]},
    "RiverRider/srt-omni-demo": {
        "models": OMNI + ["RiverRider/srt-omni-xvendor-towers",
                          "RiverRider/srt-omni-shared-tower"],
        "datasets": ["RiverRider/srt-omni-crossvendor-states",
                     "RiverRider/srt-omni-manifest"]},
    "RiverRider/srt-sunstone": {
        "models": [GEMMA, "RiverRider/Gemma-4-31B-it-SRT-Sunstone"]},
    "RiverRider/walk-the-space": {
        "models": [Q27, Q06, "RiverRider/srt-sunstone-linear-head"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    api = HfApi(token=get_token())

    for repo, meta in list(MODELS.items()) + list(SPACES.items()):
        rtype = "model" if repo in MODELS else "space"
        print(f"\n{rtype:6s} {repo}")
        for k, v in meta.items():
            print(f"    {k}: {v}")
        if a.apply:
            metadata_update(repo, meta, repo_type=rtype, overwrite=True,
                            token=get_token(),
                            commit_message="declare host models")
            print("    updated")


if __name__ == "__main__":
    main()
