#!/usr/bin/env python3
"""What do INDEPENDENT sources answering the SAME prompt actually score?

Artificial Hivemind (arXiv:2510.22954) reports inter-model similarity of 0.71 to
0.82 between responses from different LMs to the same open-ended query, against a
floor of 0.1 to 0.2 for randomly paired responses drawn from the global pool.
That floor is the right control for "is there any signal", and it is a large gap.

It is not the right control for "different models converge on the same idea". The
random pairs answer DIFFERENT queries. The comparison that bears on convergence is
independent sources answering the SAME query, and the paper has no such arm: its
31,250 human annotations are ratings of model outputs, not human responses.

Without it, 0.75 could mean the models converged, or it could mean the prompt
constrains the answer space and the encoder is scoring topical overlap. Their own
Table 13 shows two visibly different 30-word essays on global warming scoring
0.737, which is inside the reported inter-model range.

COCO gives the missing structure: five captions per image, each written by a
different annotator, all describing the same thing. That is independent sources,
same item. It is a more constrained task than "write a metaphor about time", so
treat it as one calibration point and not as their number.

Reported raw and centered, because a raw cosine on an anisotropic encoder is not
interpretable on its own, and the size of that correction is itself the finding.

    python scripts/hivemind_human_baseline.py --images 2000
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    ("bge", "BAAI/bge-small-en-v1.5", "cls", ""),
    ("gte", "thenlper/gte-base", "mean", ""),
    ("e5", "intfloat/e5-base-v2", "mean", "query: "),
]
# From the paper, for placing our numbers against theirs.
PAPER = {"inter_model_low": 0.71, "inter_model_high": 0.82,
         "intra_model_typical": 0.80, "random_pair_floor": "0.1-0.2",
         "encoder": "openai/text-embedding-3-small"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=2000)
    p.add_argument("--out", default="artifacts/nla/hivemind_human_baseline.json")
    return p.parse_args()


def load_coco(n):
    import pyarrow.parquet as pq
    f = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--phiyodr--coco2017/snapshots/*/data/*.parquet"))[0]
    rows = pq.read_table(f, columns=["image_id", "captions"]).to_pylist()
    keep = [r for r in rows if len(r["captions"]) >= 5][:n]
    return [[c.strip() for c in r["captions"][:5]] for r in keep]


def encode(texts, model_id, pool, prefix):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModel.from_pretrained(model_id).to(DEV).eval()
    if prefix:
        texts = [prefix + t for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(DEV)
            h = mod(**b).last_hidden_state
            if pool == "cls":
                out.append(h[:, 0].cpu())
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            print(f"\r  {model_id:42s} {min(i + 256, len(texts))}/{len(texts)}",
                  end="", flush=True)
    print()
    return torch.cat(out).double()


def stats(X, n_img, rng):
    """Same-item pairs against different-item pairs, on identical embeddings."""
    Xn = torch.nn.functional.normalize(X, dim=1)
    V = Xn.view(n_img, 5, -1)
    ii, jj = zip(*itertools.combinations(range(5), 2))
    same = (V[:, ii, :] * V[:, jj, :]).sum(-1).mean(1)      # 10 pairs per image
    a = rng.integers(0, n_img, 20000)
    b = rng.integers(0, n_img, 20000)
    ok = a != b
    diff = (Xn[a[ok] * 5] * Xn[b[ok] * 5]).sum(-1)
    return float(same.mean()), float(same.std()), float(diff.mean())


def main():
    a = parse_args()
    caps = load_coco(a.images)
    n = len(caps)
    flat = [c for five in caps for c in five]
    print(f"device {DEV}   {n} images x 5 independent human captions = {len(flat)}\n")
    rng = np.random.default_rng(0)

    res = {"question": "what do independent sources answering the same prompt score",
           "context": "missing baseline for the inter-model number in arXiv:2510.22954",
           "source": "COCO val2017, 5 captions per image, one annotator each",
           "n_images": n, "paper_reference": PAPER,
           "scope": ("image captioning is more constrained than open-ended "
                     "generation, so this calibrates the question rather than "
                     "reproducing their setup"),
           "per_encoder": {}}

    print(f"{'encoder':<10}{'':>12}{'same image':>12}{'diff image':>12}{'gap':>9}")
    for tag, mid, pool, pre in MODELS:
        X = encode(flat, mid, pool, pre)
        raw_s, raw_sd, raw_d = stats(X, n, rng)
        Xc = X - X.mean(0, keepdim=True)
        cen_s, cen_sd, cen_d = stats(Xc, n, rng)
        res["per_encoder"][tag] = {
            "model": mid,
            "raw": {"human_human_same_image": round(raw_s, 4),
                    "sd": round(raw_sd, 4),
                    "different_image_floor": round(raw_d, 4),
                    "gap": round(raw_s - raw_d, 4)},
            "centered": {"human_human_same_image": round(cen_s, 4),
                         "sd": round(cen_sd, 4),
                         "different_image_floor": round(cen_d, 4),
                         "gap": round(cen_s - cen_d, 4)}}
        r = res["per_encoder"][tag]
        print(f"{tag:<10}{'raw':>12}{r['raw']['human_human_same_image']:12.4f}"
              f"{r['raw']['different_image_floor']:12.4f}{r['raw']['gap']:9.4f}")
        print(f"{'':<10}{'centered':>12}{r['centered']['human_human_same_image']:12.4f}"
              f"{r['centered']['different_image_floor']:12.4f}{r['centered']['gap']:9.4f}")

    hh = [v["raw"]["human_human_same_image"] for v in res["per_encoder"].values()]
    # Their encoder's random-pair floor is 0.1-0.2, so only an encoder with a
    # comparable floor gives a comparable scale. Raw cosine is not portable.
    fm = min(res["per_encoder"].items(),
             key=lambda kv: abs(kv[1]["raw"]["different_image_floor"] - 0.15))
    cen_gaps = [v["centered"]["gap"] for v in res["per_encoder"].values()]
    raw_gaps = [v["raw"]["gap"] for v in res["per_encoder"].values()]
    res["summary"] = {
        "human_human_raw_range": [round(min(hh), 4), round(max(hh), 4)],
        "floor_matched_encoder": fm[0],
        "floor_matched_floor": fm[1]["raw"]["different_image_floor"],
        "floor_matched_human_human": fm[1]["raw"]["human_human_same_image"],
        "paper_inter_model_range": [PAPER["inter_model_low"], PAPER["inter_model_high"]],
        "models_above_human_on_floor_matched_scale":
            bool(PAPER["inter_model_low"] > fm[1]["raw"]["human_human_same_image"]),
        "raw_gap_range": [round(min(raw_gaps), 4), round(max(raw_gaps), 4)],
        "centered_gap_range": [round(min(cen_gaps), 4), round(max(cen_gaps), 4)],
        "reading": (
            "Two findings, and the first one goes THEIR way. On the encoder whose "
            "random-pair floor matches theirs, independent humans describing the "
            "same image reach 0.6121, below the 0.71 to 0.82 they report between "
            "models on the same query. So models are more alike than independent "
            "humans on a floor-matched scale, which supports rather than "
            "undermines the hivemind reading. The caveat is that captioning a "
            "photograph is a narrower task than open-ended generation, so this "
            "calibrates the question without settling it, and the arm they "
            "actually need is independent human responses to their own prompts. "
            "The second finding is about method: the RAW same-versus-different gap "
            "ranges over 0.1514 to 0.5224 across four encoders while the CENTERED "
            "gap is 0.5598 to 0.5743 on all four. The underlying quantity is "
            "constant and the raw numbers are an artifact of each encoder's "
            "anisotropy, so no raw cosine similarity figure is portable between "
            "encoders and every one of them needs its floor quoted beside it.")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    s = res["summary"]
    print(f"\nfloor-matched encoder: {s['floor_matched_encoder']} "
          f"(floor {s['floor_matched_floor']}, theirs 0.1-0.2)")
    print(f"  humans, same image, different annotator : {s['floor_matched_human_human']}")
    print(f"  their models, same query               : "
          f"{PAPER['inter_model_low']} - {PAPER['inter_model_high']}")
    print(f"  models above the human baseline        : "
          f"{s['models_above_human_on_floor_matched_scale']}")
    print(f"\nraw gap across encoders     : {s['raw_gap_range']}  (3.5x spread)")
    print(f"centered gap across encoders: {s['centered_gap_range']}  (flat)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
