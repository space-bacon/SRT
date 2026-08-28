#!/usr/bin/env python3
"""Does NF4 quantisation of the host preserve the layer-47 state well enough
to read?

This decides whether the live demo can ever run on free Hugging Face hardware.
bf16 gemma-4-31B is 60 GB and needs a dedicated A100. NF4 is about 17 GB and
fits ZeroGPU. The reader and the retrieval head were both fitted on bf16
states, so the question is whether an NF4 state still lands where they expect.

Encodes the same images twice, once per precision, and compares. Raw cosine on
this backbone is not interpretable, so every similarity is reported centered on
the pool mean alongside the raw anisotropy it is measured against.

    python scripts/probe_4bit_states.py --n 100 --show 8
"""
import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/root/.hf_home")

import gc
import time

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

HOST = "google/gemma-4-31B-it"
LAYER = 47
HEAD = "/root/head_full_L47.pt"
IMG_DIR = "/root/val2017"
CACHE = "/root/live_gallery.npz"
OUT = "artifacts/nla/quant_4bit_states.json"


class Prefix(torch.nn.Module):
    def __init__(self, d_in, d_model, n_tok, hidden=2048):
        super().__init__()
        self.n_tok, self.d_model = n_tok, d_model
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n_tok * d_model))

    def forward(self, v):
        return self.net(v).view(v.size(0), self.n_tok, self.d_model)


def load_host(four_bit):
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(HOST)
    kw = dict(device_map="cuda")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(HOST, **kw).eval()
    return proc, model


@torch.no_grad()
def encode(proc, model, files, tag):
    img_tok = model.config.image_token_id
    out, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        im = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": "Describe this image."}]}]
        enc = proc.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        h = model(**enc, output_hidden_states=True,
                  use_cache=False).hidden_states[LAYER][0].float()
        out.append(h[enc["input_ids"][0] == img_tok].mean(0).cpu())
        if i % 25 == 0:
            print(f"  {tag} {i}/{len(files)}  "
                  f"{i/(time.time()-t0):.2f}/s", flush=True)
    return torch.stack(out)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def cos_rows(A, B):
    A = A / A.norm(dim=1, keepdim=True)
    B = B / B.norm(dim=1, keepdim=True)
    return (A * B).sum(1)


def offdiag_mean(A):
    A = A / A.norm(dim=1, keepdim=True)
    S = A @ A.T
    n = S.shape[0]
    return ((S.sum() - S.diag().sum()) / (n * (n - 1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    z = np.load(CACHE, allow_pickle=True)
    gal_files, gal = list(z["files"]), torch.tensor(z["vecs"])
    files = gal_files[:a.n]
    print(f"{len(files)} images, gallery of {len(gal_files)} bf16 vectors\n",
          flush=True)

    proc, model = load_host(False)
    V16 = encode(proc, model, files, "bf16")
    del model, proc
    free()

    proc, model = load_host(True)
    V4 = encode(proc, model, files, "nf4")
    del model, proc
    free()

    mu16, mu4 = V16.mean(0), V4.mean(0)
    raw = cos_rows(V16, V4)
    cen = cos_rows(V16 - mu16, V4 - mu4)
    aniso16, aniso4 = offdiag_mean(V16), offdiag_mean(V4)
    aniso16c = offdiag_mean(V16 - mu16)

    # Floor: how similar are DIFFERENT images across the two precisions? If the
    # matched pairs do not beat this, the agreement is anisotropy, not signal.
    perm = torch.randperm(len(files))
    floor_raw = cos_rows(V16, V4[perm]).mean().item()
    floor_cen = cos_rows(V16 - mu16, (V4 - mu4)[perm]).mean().item()

    ck = torch.load(hf_hub_download("RiverRider/srt-verbalizer-v1",
                                    "gemma4_31b_L47.pt"),
                    map_location="cpu", weights_only=False)
    mu_t = torch.tensor(ck["mu"]).float().flatten()
    sd_t = torch.tensor(ck["sd"]).float().flatten()
    d16 = (V16 - mu_t).norm(dim=1)
    d4 = (V4 - mu_t).norm(dim=1)

    h = torch.load(HEAD, map_location="cpu", weights_only=False)
    Wi = h["img"]["weight"].float()
    bi = h["img"]["bias"].float()
    mu_img = h["mu_img"].float().flatten()

    def project(V):
        Q = (V - mu_img) @ Wi.T + bi
        return Q / Q.norm(dim=1, keepdim=True)

    Q16, Q4 = project(V16), project(V4)
    ranks, overlap = [], []
    for i in range(len(files)):
        s4 = gal @ Q4[i]
        ranks.append(int((s4 > s4[i]).sum().item()) + 1)
        n16 = set(torch.argsort(-(gal @ Q16[i]))[:11].tolist()) - {i}
        n4 = set(torch.argsort(-s4)[:11].tolist()) - {i}
        overlap.append(len(n16 & n4) / 10.0)
    ranks = np.array(ranks)

    print("\nloading reader", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    lm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32).eval().to("cuda")
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048))
    pre.load_state_dict(ck["prefix"])
    pre = pre.eval().to("cuda")

    @torch.no_grad()
    def say(v):
        soft = pre(((v - mu_t) / sd_t).unsqueeze(0).to("cuda"))
        g = lm.generate(inputs_embeds=soft,
                        attention_mask=torch.ones(soft.shape[:2],
                                                  dtype=torch.long,
                                                  device="cuda"),
                        max_new_tokens=40, do_sample=False)
        return tok.decode(g[0], skip_special_tokens=True).strip()

    pairs = [{"file": files[i], "bf16": say(V16[i]), "nf4": say(V4[i])}
             for i in range(min(a.show, len(files)))]
    agree = sum(p["bf16"] == p["nf4"] for p in pairs)

    res = {
        "n": len(files), "gallery": len(gal_files),
        "cos_bf16_vs_nf4": {
            "raw_mean": raw.mean().item(), "raw_min": raw.min().item(),
            "centered_mean": cen.mean().item(),
            "centered_min": cen.min().item(),
            "raw_floor_mismatched": floor_raw,
            "centered_floor_mismatched": floor_cen},
        "anisotropy": {"bf16_raw": aniso16, "nf4_raw": aniso4,
                       "bf16_centered": aniso16c},
        "norms": {
            "state_bf16": V16.norm(dim=1).mean().item(),
            "state_nf4": V4.norm(dim=1).mean().item(),
            "dist_to_reader_mu_bf16": d16.mean().item(),
            "dist_to_reader_mu_nf4": d4.mean().item(),
            "reader_sd_mean": sd_t.mean().item()},
        "retrieval_nf4_query_into_bf16_index": {
            "self_r_at_1": float((ranks == 1).mean()),
            "self_r_at_10": float((ranks <= 10).mean()),
            "median_rank": float(np.median(ranks)),
            "neighbour_overlap_at_10": float(np.mean(overlap))},
        "verbalizer": {"shown": len(pairs), "identical": agree,
                       "pairs": pairs},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\n{'='*68}\nNF4 vs bf16 layer-{LAYER} states, n={len(files)}\n{'='*68}")
    print(f"cosine, matched pairs   raw {raw.mean():.4f}   "
          f"centered {cen.mean():.4f}  (worst {cen.min():.4f})")
    print(f"cosine, mismatched      raw {floor_raw:.4f}   "
          f"centered {floor_cen:.4f}   <- the floor")
    print(f"anisotropy              bf16 {aniso16:.4f}   nf4 {aniso4:.4f}   "
          f"bf16 centered {aniso16c:.4f}")
    print(f"distance to reader mu   bf16 {d16.mean():.2f}   "
          f"nf4 {d4.mean():.2f}   (sd mean {sd_t.mean():.2f})")
    print(f"nf4 query -> bf16 index r@1 {(ranks==1).mean():.3f}   "
          f"r@10 {(ranks<=10).mean():.3f}   median rank "
          f"{np.median(ranks):.0f}   neighbour overlap@10 "
          f"{np.mean(overlap):.3f}")
    print(f"verbalizer identical    {agree}/{len(pairs)}\n")
    for p in pairs:
        print(f"  {p['file']}")
        print(f"    bf16: {p['bf16']}")
        print(f"    nf4 : {p['nf4']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
