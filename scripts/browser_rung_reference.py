"""PyTorch reference for the browser rung, and the caption set Rust replays.

A 0.6B text tower searches a gallery of images that a 27B tower encoded. This
runs that in PyTorch at fp16 so there is something to compare the browser's
quantized candle path against.

The comparison is the point. Same head, same gallery, same captions, two
different runtimes. If retrieval survives the swap, the read-out is a property
of the representation. If it does not, the 42KB recalibration is what closes
the gap, and that is measurable here rather than assertable.

    python scripts/browser_rung_reference.py --n-images 1000 --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BROWSER = ROOT / "artifacts/local/browser"
MAX_SEQ = 64
N_EVAL_IMAGES = 1000  # the head's own held-out split: the LAST 1000 images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path,
                   default=ROOT / "checkpoints/gemma4_readout/browser_text_head_qwen3_0.6B.pt")
    p.add_argument("--gallery", type=Path, default=BROWSER / "gallery.npz")
    p.add_argument("--captions", type=Path, default=BROWSER / "captions.json")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--n-images", type=int, default=1000,
                   help="gallery is restricted to these, captions follow")
    p.add_argument("--n-anchors", type=int, default=200,
                   help="held-out captions used to measure a runtime's text anchor")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, default=BROWSER / "reference.json")
    p.add_argument("--out-replay", type=Path, default=BROWSER / "replay.json")
    return p.parse_args()


def unit(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


@torch.no_grad()
def encode(model, tok, texts, layer, device, batch):
    """Byte-for-byte the tokenization the head was trained with.

    Qwen3 has no BOS by default, so the training script prepended one
    explicitly. A head fit on BOS-prefixed states reads a different
    distribution without it, and the resulting drop looks like a runtime
    problem rather than a preprocessing one.
    """
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids_list = []
        for t in chunk:
            ids = tok(t, truncation=True, max_length=MAX_SEQ,
                      add_special_tokens=True).input_ids
            bos = tok.bos_token_id
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: MAX_SEQ - 1]
            ids_list.append(ids)
        T = max(len(x) for x in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_list), T), dtype=torch.long)
        for j, ids in enumerate(ids_list):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        res = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False)
        last = (attn.sum(-1) - 1).to(device)
        rows = torch.arange(len(chunk), device=device)
        out.append(res.hidden_states[layer][rows, last].float().cpu().numpy())
        if (i // batch) % 20 == 0:
            print(f"  {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out)


def main() -> None:
    a = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    head = torch.load(a.head, map_location="cpu", weights_only=True)
    Wt = head["txt"]["weight"].float().numpy()
    bt = head["txt"]["bias"].float().numpy()
    mt = head["mu_txt"].float().numpy()

    g = np.load(a.gallery, allow_pickle=True)
    all_files = [str(f) for f in g["files"]]
    # The head was fit on the first n-1000 images. Scoring on those measures
    # memorisation, and it shows: on the training images i2t R@1 reads 0.998.
    ev = slice(len(all_files) - N_EVAL_IMAGES, len(all_files))
    files = all_files[ev]
    Z_img = g["Z_img"][ev].astype(np.float32)
    keep = set(files)
    row_of = {f: i for i, f in enumerate(files)}

    meta = json.loads(a.captions.read_text())
    pairs = [p for p in meta["pairs"] if p["owner"] in keep]
    texts = [p["text"] for p in pairs]
    owner = np.array([row_of[p["owner"]] for p in pairs])

    # Recalibration anchors come from images the head was fit on, never from
    # the captions being scored. Measuring an anchor on the eval set and then
    # reporting recall on it is transductive, and it would inflate exactly the
    # number this experiment exists to establish.
    anchor_pool = [p["text"] for p in meta["pairs"] if p["owner"] not in keep]
    rng0 = np.random.default_rng(0)
    anchors = [anchor_pool[i] for i in
               rng0.choice(len(anchor_pool), size=min(a.n_anchors, len(anchor_pool)),
                           replace=False)]
    print(f"{len(files)} images, {len(texts)} captions, {len(anchors)} held-out anchors, "
          f"layer {a.layer}")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16).to(a.device).eval()

    S_txt = encode(model, tok, texts, a.layer, a.device, a.batch)
    np.save(a.out_replay.parent / "ref_text_states.npy", S_txt)
    Z_txt = unit((S_txt - mt) @ Wt.T + bt)

    # fp16 index, because that is what ships
    Z_idx = unit(Z_img).astype(np.float16).astype(np.float32)
    S = Z_txt @ Z_idx.T

    t2i_rank = np.array([int(np.where(np.argsort(-S[i]) == owner[i])[0][0])
                         for i in range(len(texts))])
    # i2t: for each image, best caption among all
    Si = Z_idx @ Z_txt.T
    i2t_rank = []
    for r in range(len(files)):
        gold = np.where(owner == r)[0]
        if len(gold) == 0:
            continue
        order = np.argsort(-Si[r])
        pos = np.where(np.isin(order, gold))[0]
        i2t_rank.append(int(pos[0]))
    i2t_rank = np.array(i2t_rank)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(texts))
    shuf = np.array([int(np.where(np.argsort(-S[i]) == owner[perm[i]])[0][0])
                     for i in range(len(texts))])

    result = {
        "runtime": f"pytorch fp16 {a.model} L{a.layer} on {a.device}",
        "n_images": len(files),
        "n_captions": len(texts),
        "t2i": {f"r@{k}": round(float((t2i_rank < k).mean()), 4) for k in (1, 5, 10)},
        "i2t": {f"r@{k}": round(float((i2t_rank < k).mean()), 4) for k in (1, 5, 10)},
        "shuffled_t2i": {f"r@{k}": round(float((shuf < k).mean()), 4) for k in (1, 5, 10)},
        "text_state_mean_norm": round(float(np.linalg.norm(S_txt, axis=1).mean()), 3),
    }
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # What Rust replays: the caption strings, their gold image, and this
    # runtime's own text-anchor so the recalibration comparison is possible.
    a.out_replay.write_text(json.dumps({
        "model": a.model,
        "layer": a.layer,
        "files": files,
        "captions": [{"text": t, "gold": int(o)} for t, o in zip(texts, owner)],
        "anchor_captions": anchors,
        "anchor_note": "drawn from training-split images, disjoint from the eval gallery",
        "pytorch_t2i": result["t2i"],
        "pytorch_mu_txt": S_txt.mean(0).tolist(),
    }))
    print(f"wrote {a.out_replay}")


if __name__ == "__main__":
    main()
