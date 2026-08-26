#!/usr/bin/env python
"""Score the Lab-space reader with the Lab's own instrument.

The reader is handed a 1024-d point in the sunstone head's space, says what is
there, and the sentence is re-encoded and retrieved against the gallery. If the
words carry the record, the image the point came from comes back.

Two conventions have to match the deployment exactly or the numbers are noise:

    text   BOS-prefixed, LAST token, layer 47. gemma-4 is BOS-sensitive and a
           bare re-encode drops replay 0.9986 -> 0.615 (scripts/sunstone_server
           .py encode_text_local).
    image  mean over image-token positions, already baked into the stored
           states, then the head projection and L2 normalization.

Arms, in the order they run:

    gold        human caption through the same encode-and-retrieve path. This
                is the HARNESS CONTROL and it runs first: if human captions
                cannot retrieve their own images, the head, the gallery and
                the gold rows are not the same system and nothing else in the
                run means anything.
    real        the image's own point
    foreign     another held-out image's point, so a caption prior alone
                cannot pass
    mean        the mean of held-out points: pure typicality
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/root/sunstone_verbalizer.pt")
    p.add_argument("--vecs", default="/root/sunstone_img_vecs.npy")
    p.add_argument("--caps", default="/root/full_caps.json")
    p.add_argument("--head-repo", default="RiverRider/srt-sunstone-linear-head")
    p.add_argument("--head-file", default="sunstone_linear_head_v3_drift.pt")
    p.add_argument("--tower", default="google/gemma-4-31B-it")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--max-seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="/root/sunstone_verb_eval.json")
    return p.parse_args()


def main():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from train_shared_space_verbalizer import Prefix
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download

    a = parse()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    gal = np.load(a.vecs).astype(np.float32)          # already L2 normalized
    meta = json.load(open(a.caps))
    test = list(ck["test_idx"])[: a.n]
    print(f"gallery {gal.shape}, scoring {len(test)} held-out points", flush=True)

    d = torch.load(hf_hub_download(a.head_repo, a.head_file),
                   map_location="cpu", weights_only=True)
    Wt = d["txt"]["weight"].float().numpy()
    bt = d["txt"]["bias"].float().numpy()
    mt = d["mu_txt"].float().numpy()

    tok_t = AutoTokenizer.from_pretrained(a.tower)
    tower = AutoModelForCausalLM.from_pretrained(
        a.tower, dtype=torch.bfloat16, device_map="cuda").eval()
    bos = getattr(tok_t, "bos_token_id", None) or 2

    @torch.no_grad()
    def project(texts):
        out = []
        for i in range(0, len(texts), a.batch):
            chunk = texts[i:i + a.batch]
            ids_list = []
            for t in chunk:
                ids = tok_t(t, truncation=True, max_length=a.max_seq,
                            add_special_tokens=True).input_ids
                if not ids or ids[0] != bos:
                    ids = [bos] + ids[: a.max_seq - 1]
                ids_list.append(ids)
            T = max(len(x) for x in ids_list)
            pad = tok_t.pad_token_id if tok_t.pad_token_id is not None else 0
            inp = torch.full((len(ids_list), T), pad, dtype=torch.long)
            att = torch.zeros((len(ids_list), T), dtype=torch.long)
            for j, ids in enumerate(ids_list):
                inp[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
            res = tower(input_ids=inp.to(dev), attention_mask=att.to(dev),
                        output_hidden_states=True, use_cache=False)
            last = (att.sum(-1) - 1).to(dev)
            rows = torch.arange(len(chunk), device=dev)
            h = res.hidden_states[a.layer][rows, last].float().cpu().numpy()
            v = (h - mt) @ Wt.T + bt
            out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
            if i % (a.batch * 8) == 0:
                print(f"  encode {i}/{len(texts)}", flush=True)
        return np.concatenate(out)

    def rank_of(q, gold):
        s = q @ gal.T
        return (s > s[np.arange(len(s)), gold][:, None]).sum(1)

    gold_rows = np.array(test)
    results = {}

    # Harness control, first and fatal.
    gold_caps = [meta["captions"][i][0] for i in test]
    r = rank_of(project(gold_caps), gold_rows)
    results["gold_caption_harness_control"] = {
        "r@1": round(float((r < 1).mean()), 4),
        "r@10": round(float((r < 10).mean()), 4),
        "median_rank": float(np.median(r)) + 1,
    }
    print(f"gold  R@1 {results['gold_caption_harness_control']['r@1']:.4f} "
          f"median {results['gold_caption_harness_control']['median_rank']:.0f}",
          flush=True)
    if results["gold_caption_harness_control"]["median_rank"] > len(gal) * 0.02:
        raise SystemExit("harness control failed: human captions do not retrieve "
                         "their own images. Nothing else here would mean anything.")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    reader = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
    pre = Prefix(ck["d_in"], ck["d_model"], ck["n_tok"], ck.get("hidden", 2048)).to(dev)
    pre.load_state_dict(ck["prefix"]); pre.eval()

    @torch.no_grad()
    def speak(v):
        v = (v - ck["mu"]) / ck["sd"]
        said = []
        for i in range(0, len(v), 16):
            soft = pre(torch.tensor(v[i:i + 16], device=dev, dtype=torch.float32))
            gen = reader.generate(
                inputs_embeds=soft, max_new_tokens=a.max_new, do_sample=False,
                attention_mask=torch.ones(soft.shape[:2], dtype=torch.long, device=dev),
                pad_token_id=tok.eos_token_id)
            said += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
        return said

    real = gal[gold_rows]
    for name, v in (("real", real),
                    ("foreign", np.roll(real, 1, axis=0)),
                    ("mean", np.repeat(real.mean(0, keepdims=True), len(test), 0))):
        said = speak(v.astype(np.float32))
        r = rank_of(project(said), gold_rows)
        results[name] = {
            "r@1": round(float((r < 1).mean()), 4),
            "r@10": round(float((r < 10).mean()), 4),
            "median_rank": float(np.median(r)) + 1,
            "sample": said[:5],
        }
        print(f"{name:8s} R@1 {results[name]['r@1']:.4f} "
              f"median {results[name]['median_rank']:.0f}\n    {said[0]!r}", flush=True)

    json.dump({"question": "can a 0.6B say what is at a point in the Lab's own space",
               "gallery": int(gal.shape[0]),
               "gallery_note": "train2017 sunstone-head projections, L2 normalized, "
                               "the same representation the Lab serves",
               "n_scored": len(test),
               "text_convention": f"BOS-prefixed last-token L{a.layer}, gemma-4 is BOS-sensitive",
               "head": a.head_file, "reader_backbone": "Qwen/Qwen3-0.6B",
               "n_prefix_tokens": ck["n_tok"], "arms": results},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
