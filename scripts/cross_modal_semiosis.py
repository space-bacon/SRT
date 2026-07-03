"""Cross-modal interpretant convergence on gemma-4-31B (a native omni model).

Tests the core semiotic claim of the SRT program: the residual-stream read-out
is modality-agnostic, so an image-sign and a word-sign for the same referent
build a *converging interpretant* as depth increases. If true, matched
(image, word) pairs align in the residual stream and mismatched pairs do not,
and the alignment grows with layer depth.

Method (forward passes only, no trained adapter):
  - image_rep[L]  = mean over the image soft-token positions of hidden_states[L]
  - word_rep[L]   = last-token hidden state of the bare concept word
  - per-modality centering (subtract each modality's pool mean at layer L, since
    image-space and word-space have different anisotropy)
  - per layer: matched centred cosine (image_i vs word[concept_i]),
    mismatched centred cosine (image_i vs word[other]), the gap, and
    retrieval@1 (does the correct concept word top the ranking for each image)

Output: artifacts/nla/gemma4/cross_modal_semiosis.json

Usage (venv with transformers>=5 + torchvision):
    python scripts/cross_modal_semiosis.py --per-class 10
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"

# label-name -> cleaner concept word (identity for the rest)
WORD_FIX = {"automobile": "car"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/nla/gemma4/cross_modal_semiosis.json")
    p.add_argument("--dataset", default="uoft-cs/cifar10")
    p.add_argument("--config", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--per-class", type=int, default=10)
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from datasets import load_dataset

    proc = AutoProcessor.from_pretrained(MID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.text_config.num_hidden_layers
    img_tok = model.config.image_token_id
    tok = proc.tokenizer

    ds = load_dataset(args.dataset, args.config or None, split=args.split)
    label_names = ds.features["label"].names
    img_key = "image" if "image" in ds.features else "img"
    concept_word = {i: WORD_FIX.get(n, n) for i, n in enumerate(label_names)}
    print(f"{len(ds)} images | {len(label_names)} labels: {label_names}", flush=True)

    # gather up to per-class images per label
    by_label: dict[int, list] = {}
    for row in ds:
        lb = int(row["label"])
        by_label.setdefault(lb, [])
        if len(by_label[lb]) < args.per_class:
            by_label[lb].append(row[img_key])
        if all(len(by_label.get(i, [])) >= args.per_class for i in range(len(label_names))):
            break
    concepts = sorted(by_label)
    words = [concept_word[c] for c in concepts]
    print("concepts:", words, flush=True)

    @torch.no_grad()
    def image_reps(image) -> torch.Tensor:  # (L+1, d)
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": "."}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = (enc["input_ids"][0] == img_tok)
        return torch.stack([h[0][mask].float().mean(0) for h in out.hidden_states])

    @torch.no_grad()
    def word_reps(word: str) -> torch.Tensor:  # (L+1, d)
        ids = tok("a photo of a " + word, return_tensors="pt").input_ids.to("cuda")
        out = model.model.language_model(ids, output_hidden_states=True, use_cache=False) \
            if hasattr(model.model, "language_model") else model(ids, output_hidden_states=True, use_cache=False)
        return torch.stack([h[0, -1].float() for h in out.hidden_states])

    # collect reps
    word_R = {c: word_reps(concept_word[c]) for c in concepts}  # concept -> (L+1,d)
    img_R: list[tuple[int, torch.Tensor]] = []
    for c in concepts:
        for im in by_label[c]:
            img_R.append((c, image_reps(im)))
        print(f"  encoded images for '{concept_word[c]}'", flush=True)

    n_layers = L + 1
    # per-modality centering means per layer
    mu_word = torch.stack([torch.stack([word_R[c][li] for c in concepts]).mean(0)
                           for li in range(n_layers)])          # (L+1,d)
    mu_img = torch.stack([torch.stack([r[li] for _, r in img_R]).mean(0)
                          for li in range(n_layers)])           # (L+1,d)

    per_layer = []
    for li in range(n_layers):
        wc = {c: F.normalize(word_R[c][li] - mu_word[li], dim=0) for c in concepts}
        matched, mismatched, hits = [], [], 0
        for c, r in img_R:
            iv = F.normalize(r[li] - mu_img[li], dim=0)
            sims = {cc: float(iv @ wc[cc]) for cc in concepts}
            matched.append(sims[c])
            mismatched.extend(sims[cc] for cc in concepts if cc != c)
            if max(sims, key=sims.get) == c:
                hits += 1
        m = sum(matched) / len(matched)
        mm = sum(mismatched) / len(mismatched)
        per_layer.append({"layer": li, "matched_cos": m, "mismatched_cos": mm,
                          "gap": m - mm, "retrieval_at1": hits / len(img_R)})

    result = {"model": MID, "n_images": len(img_R), "n_concepts": len(concepts),
              "concepts": words, "text_layers": L, "per_layer": per_layer,
              "chance_retrieval": 1.0 / len(concepts)}
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print(f"\nchance retrieval@1 = {1/len(concepts):.3f}")
    print("layer   matched  mismatch     gap   retr@1")
    for r in per_layer[:: max(1, n_layers // 20)]:
        print(f"  L{r['layer']:>2}    {r['matched_cos']:+.3f}   {r['mismatched_cos']:+.3f}"
              f"   {r['gap']:+.3f}   {r['retrieval_at1']:.3f}")
    best = max(per_layer, key=lambda r: r["retrieval_at1"])
    print(f"\nbest retrieval@1 = {best['retrieval_at1']:.3f} at L{best['layer']} "
          f"(gap {best['gap']:+.3f})")


if __name__ == "__main__":
    main()
