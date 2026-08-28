#!/usr/bin/env python3
"""Pre-quantise the host to NF4 and push it, so a Space does not have to pull
62 GB and quantise on every cold start.

The config here must stay identical to scripts/probe_4bit_states.py, because
that is the configuration whose states were measured against the reader and the
retrieval head. A different quantiser is a different model as far as the reader
is concerned.

    python scripts/export_nf4_host.py --repo RiverRider/gemma-4-31B-it-nf4
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/root/.hf_home")

import torch

SRC = "google/gemma-4-31B-it"
OUT = "/root/gemma4_31b_nf4"

CARD = """---
license: gemma
base_model: google/gemma-4-31B-it
pipeline_tag: image-text-to-text
tags: [bitsandbytes, nf4, 4-bit, gemma, srt]
---

# gemma-4-31B-it, NF4

`google/gemma-4-31B-it` quantised to 4-bit NF4 with bitsandbytes, double
quantisation on, bfloat16 compute. Nothing else is changed. Use of this model
is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

It exists so that a Space can load the host in about 17 GB instead of pulling
62 GB and quantising at startup.

## Why this exact quantiser

We read the layer-47 hidden state out of this model and hand it to a separate
0.6B decoder and a linear retrieval head, both of which were fitted on bf16
states. So the question is whether quantisation moves the state out of the
distribution they expect. We measured it on 100 COCO val2017 images.

Anisotropy on this backbone is 0.9874, so raw cosine is not usable: matched
bf16/NF4 pairs read 0.9986 while *mismatched* pairs, different pictures across
the two precisions, read 0.9867. Centered on the pool mean, matched pairs are
**0.9631** (worst 0.9056) against a mismatched floor of **0.0237**.

| | bf16 | NF4 |
|---|---|---|
| distance to the reader's training mean | 13.43 | 14.51 |
| self-retrieval into the 2000-image bf16 index, r@1 | 1.000 | **1.000** |
| top-10 neighbour agreement with bf16 | 1.000 | 0.704 |

Retrieval survives: an NF4 query lands on its own image at rank 1 in 100 of
100 cases. Deeper ranking does move, so the two indices are not
interchangeable. Verbalisation is content-stable but not token-stable: 8 of 8
sampled pairs name the same subject and 2 of 8 are identical strings, so any
quoted example sentence should say which precision produced it.

AWQ and GGUF builds of this model exist and are probably fine, but they were
not the ones we measured.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/gemma-4-31B-it-nf4")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoProcessor,
                              BitsAndBytesConfig)
    print("quantising", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SRC, device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)).eval()

    print("saving", flush=True)
    model.save_pretrained(OUT, safe_serialization=True)
    AutoProcessor.from_pretrained(SRC).save_pretrained(OUT)
    open(os.path.join(OUT, "README.md"), "w").write(CARD)

    gb = sum(os.path.getsize(os.path.join(OUT, f))
             for f in os.listdir(OUT)) / 1e9
    print(f"{OUT}  {gb:.1f} GB", flush=True)

    if a.push:
        from huggingface_hub import HfApi, get_token
        api = HfApi(token=get_token())
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        print(f"uploading to {a.repo}", flush=True)
        api.upload_large_folder(repo_id=a.repo, repo_type="model",
                                folder_path=OUT)
        print("done", flush=True)


if __name__ == "__main__":
    main()
