#!/usr/bin/env python
"""Is the shared space made by the backbone, or by the head?

The verbalizer reads 1024-d shared-space vectors and speaks them. It was
trained on IMAGE states. The obvious question is whether it is reading the
backbone's semantics or something narrower, and there is a clean way to ask:
gemma-4-31B produced, at the same layer 47 and in the same 5376 dimensions, a
state for each image AND a state for that image's own caption. Same model,
same layer, same width, same scene. Only the modality differs.

If layer 47 is a modality-agnostic semantic space, the caption's state should
speak roughly the same sentence as the image's. If it does not, then whatever
makes image and text comparable in the shipped demo is not the backbone: it is
the trained head that projects both into 1024-d.

The naive version of this test is unfair and would prove nothing. Raw states
are strongly anisotropic (||mu|| = 115 against a mean radius of 13), and the
image and caption clouds sit at different offsets, so feeding caption states
through the image's centring frame guarantees failure for a trivial reason.
So the caption arm is run twice: once in the image frame, and once centred and
scaled by the caption cloud's OWN statistics. The second is the honest test. If
re-centring rescues it, this is an offset artifact. If it does not, the
separation is real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_shared_space_verbalizer import load_index  # noqa: E402
from train_shared_space_verbalizer import Prefix  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/fullstate_verbalizer/fullstate_verbalizer_eos.pt")
    p.add_argument("--repo", default="RiverRider/srt-nla-gemma4-artifacts")
    p.add_argument("--chunk", default="procrustes/train_pairs/chunk_000.pt")
    # The shipped fp16 text head. gallery_123k_v3.srtidx pairs with THIS head and
    # no other; the 4,000-image head in artifacts/local/browser scores at chance
    # against it while still producing perfectly plausible sentences.
    p.add_argument("--head-repo", default="RiverRider/srt-browser-head-118k")
    p.add_argument("--head-file", default="head_v3.safetensors")
    p.add_argument("--head", default=None, help="local override for --head-file")
    p.add_argument("--caps", default=None,
                   help="full_caps.json, for the gold-caption control")
    p.add_argument("--index", default="artifacts/local/browser/gallery_123k_v3.srtidx")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="artifacts/nla/verbalizer/modality_separation.json")
    return p.parse_args()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download

    a = parse()
    dev = (a.device if a.device != "auto"
           else "cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {dev}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    held = set(ck["images"])

    d = torch.load(hf_hub_download(a.repo, a.chunk), map_location="cpu", weights_only=True)
    files = list(d["files"])
    # Only score images the verbalizer never trained on.
    rows = [i for i, f in enumerate(files) if f"{Path(f).stem}.jpg" in held][: a.n]
    print(f"{len(rows)} held-out images in this chunk", flush=True)
    img = d["img"][rows].float().numpy()
    cap = d["cap0"][rows].float().numpy()
    stems = [Path(files[i]).stem for i in rows]

    gal, pos = load_index(a.index)
    gold_rows = np.array([pos[f"train2017/{s}.jpg"] for s in stems])

    # Geometry first, and centred, because raw cosines on a cloud this
    # anisotropic are not interpretable.
    mu_i, mu_c = img.mean(0), cap.mean(0)

    def cos(x, y):
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return (x * y).sum(1)

    roll = np.roll(np.arange(len(img)), 1)
    geom = {
        "raw_cos_img_to_its_caption": round(float(cos(img, cap).mean()), 4),
        "raw_cos_img_to_other_caption": round(float(cos(img, cap[roll]).mean()), 4),
        "centred_cos_img_to_its_caption": round(float(cos(img - mu_i, cap - mu_c).mean()), 4),
        "centred_cos_img_to_other_caption": round(float(cos(img - mu_i, cap[roll] - mu_c).mean()), 4),
        "raw_anisotropy_within_images": round(float(cos(img, img[roll]).mean()), 4),
        "raw_anisotropy_within_captions": round(float(cos(cap, cap[roll]).mean()), 4),
        "cos_between_cloud_means": round(float(cos(mu_i[None], mu_c[None])[0]), 4),
        "norm_mu_img": round(float(np.linalg.norm(mu_i)), 2),
        "norm_mu_cap": round(float(np.linalg.norm(mu_c)), 2),
    }
    for k, v in geom.items():
        print(f"  {k}: {v}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048)).to(dev)
    pre.load_state_dict(ck["prefix"])
    pre.eval()

    h = load_file(a.head or hf_hub_download(a.head_repo, a.head_file))
    W = h["txt.weight"].float().numpy()
    b = h["txt.bias"].float().numpy()
    mu_txt = h["mu_txt"].float().numpy()

    @torch.no_grad()
    def project(texts):
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True,
                      truncation=True, max_length=64).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[a.layer].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = ((hs * m).sum(1) / m.sum(1)).cpu().numpy()
            v = (pooled - mu_txt) @ W.T + b
            out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
        return np.concatenate(out)

    @torch.no_grad()
    def speak(v):
        said = []
        for i in range(0, len(v), 16):
            soft = pre(torch.tensor(v[i:i + 16], device=dev, dtype=torch.float32))
            gen = model.generate(
                inputs_embeds=soft, max_new_tokens=a.max_new, do_sample=False,
                attention_mask=torch.ones(soft.shape[:2], dtype=torch.long, device=dev),
                pad_token_id=tok.eos_token_id)
            said += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
        return said

    img_frame = lambda x: (x - ck["mu"]) / ck["sd"]                       # noqa: E731
    own_frame = lambda x: (x - x.mean(0)) / (                             # noqa: E731
        float(np.linalg.norm(x - x.mean(0), axis=1).mean()) or 1.0)

    arms = {
        "image_state": img_frame(img),
        "caption_state_image_frame": img_frame(cap),
        "caption_state_own_frame": own_frame(cap),
        "image_state_own_frame": own_frame(img),
    }

    results = {}

    # Run this BEFORE anything expensive. Without it a wrong head reads as a
    # finding: the sentences stay fluent and only the retrieval collapses,
    # which is indistinguishable from the result this probe is looking for.
    if a.caps:
        meta = json.load(open(a.caps))
        by_stem = {Path(im).stem: c for im, c in zip(meta["images"], meta["captions"])}
        gold = [by_stem[s][0] for s in stems]
        q = project(gold)
        s = q @ gal.T
        ranks = (s > s[np.arange(len(s)), gold_rows][:, None]).sum(1)
        results["gold_caption_harness_control"] = {
            "r@1": round(float((ranks < 1).mean()), 4),
            "r@10": round(float((ranks < 10).mean()), 4),
            "median_rank": float(np.median(ranks)) + 1,
            "sample": gold[:3],
        }
        r = results["gold_caption_harness_control"]
        print(f"{'gold (harness control)':28s} R@1 {r['r@1']:.4f}  "
              f"median {r['median_rank']:.0f}", flush=True)
        if r["median_rank"] > 5000:
            raise SystemExit(
                f"harness control failed: human captions retrieve their own "
                f"images at median {r['median_rank']:.0f} of {len(gal)}. The "
                f"head, the index and the gold rows are not the same system. "
                f"Nothing else in this run would mean anything.")

    for name, v in arms.items():
        said = speak(v.astype(np.float32))
        q = project(said)
        s = q @ gal.T
        ranks = (s > s[np.arange(len(s)), gold_rows][:, None]).sum(1)
        empty = sum(1 for t in said if len(t.strip()) < 3)
        results[name] = {
            "r@1": round(float((ranks < 1).mean()), 4),
            "r@10": round(float((ranks < 10).mean()), 4),
            "median_rank": float(np.median(ranks)) + 1,
            "degenerate_outputs": empty,
            "sample": said[:5],
        }
        r = results[name]
        print(f"{name:28s} R@1 {r['r@1']:.4f}  median {r['median_rank']:.0f}", flush=True)
        print(f"    {said[0]!r}", flush=True)

    doc = {
        "question": "does layer 47 of the 31B hold a modality-agnostic space, "
                    "or does the trained head create the shared space",
        "design": "same model, same layer, same 5376 dims, same images: the "
                  "verbalizer was trained on image states and is here handed "
                  "the caption states of the very same scenes",
        "n_scored": len(rows),
        "gallery": len(gal),
        "chance_median_rank": len(gal) // 2,
        "geometry": geom,
        "arms": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
