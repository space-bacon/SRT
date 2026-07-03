"""Cross-modal read-out transfer: apply the TEXT-trained SRT community head to
gemma-4 IMAGE tokens.

The community head in checkpoints/gemma4_readout/readout_selected.pt was trained
only on text (discourse-community SupCon). This is the semiotic payoff test: if
the read-out is modality-agnostic (operating on the interpretant in the residual
stream, not on linguistic form), then feeding it IMAGE-token hidden states should
yield coherent structure with zero image training.

Three measurements (community encoder `encoded` vector at the community layer):
  (a) IMAGE CLASS SEPARABILITY - centroid/kNN top-1 of images by their CIFAR-10
      class, using the text-trained encoder. chance = 1/10. Tests whether the
      read-out organizes images by semantic referent at all.
  (b) IMAGE<->WORD RETRIEVAL - for each image, does its encoded vector rank the
      matching class WORD ("a photo of a {cls}") first among the 10 class words?
      chance = 1/10. Direct cross-modal sign convergence through the trained head.
  (c) IMAGE -> DISCOURSE COMMUNITY - assign each image to the nearest TEXT
      community centroid (built from held-out passages) and report which
      discourse community_label each visual class gets read into. Qualitative
      semiotic mapping.

Usage (venv transformers>=5 + torchvision):
    python scripts/cross_modal_readout.py --ckpt checkpoints/gemma4_readout/readout_selected.pt \
        --heldout data/phase1_heldout.jsonl --per-class 10 \
        --out artifacts/nla/gemma4/cross_modal_readout.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

MID = "google/gemma-4-31B-it"
WORD_FIX = {"automobile": "car"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gemma4_readout/readout_selected.pt")
    p.add_argument("--heldout", default="data/phase1_heldout.jsonl")
    p.add_argument("--out", default="artifacts/nla/gemma4/cross_modal_readout.json")
    p.add_argument("--dataset", default="uoft-cs/cifar10")
    p.add_argument("--split", default="test")
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--text-per-comm", type=int, default=40)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--knn-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(1)

    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from datasets import load_dataset
    from srt.config import SRTConfig
    from srt.modules.community import CommunityDiscoveryHead

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cc = ck["config"]
    comm_L, d, dc = cc["community_layer"], cc["d_backbone"], cc["d_community"]
    print(f"ckpt step {ck.get('step')} | community@{comm_L} d={d}", flush=True)

    proc = AutoProcessor.from_pretrained(MID)
    tok = proc.tokenizer
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    img_tok = model.config.image_token_id

    # load the trained community head (unwrap from the ModuleList state_dict)
    cfg = SRTConfig(backbone_id=MID)
    community = CommunityDiscoveryHead(cfg.community, d).cuda().eval()
    full = ck["heads"]
    sub = {k[len("0."):]: v for k, v in full.items() if k.startswith("0.")}
    community.load_state_dict(sub)

    @torch.no_grad()
    def text_encoded(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_seq_len).to("cuda")
        hs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                   output_hidden_states=True, use_cache=False).hidden_states
        out = community(hs[comm_L].float(), attention_mask=enc.attention_mask)
        return out.encoded[0].float().cpu()

    @torch.no_grad()
    def image_encoded(image) -> torch.Tensor:
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": "."}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        # pool the community encoder over the IMAGE soft-token positions only
        img_mask = (enc["input_ids"] == img_tok).long()
        out = community(hs[comm_L].float(), attention_mask=img_mask)
        return out.encoded[0].float().cpu()

    # ---- (c prep) text community centroids from held-out passages ----
    by_comm_txt: dict[int, list[str]] = defaultdict(list)
    comm_label: dict[int, str] = {}
    with open(args.heldout) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text") and r.get("community_id") is not None:
                c = int(r["community_id"])
                if len(by_comm_txt[c]) < args.text_per_comm:
                    by_comm_txt[c].append(r["text"])
                    comm_label.setdefault(c, r.get("community_label", str(c)))
    comms = sorted(by_comm_txt)
    txt_cent = {}
    for c in comms:
        embs = torch.stack([text_encoded(t) for t in by_comm_txt[c]])
        txt_cent[c] = F.normalize(embs.mean(0), dim=0)
        print(f"  text centroid comm {c} ({comm_label[c]}): {len(by_comm_txt[c])}", flush=True)
    cent_mat = torch.stack([txt_cent[c] for c in comms])  # (n_comm, dc)

    # ---- images ----
    ds = load_dataset(args.dataset, split=args.split)
    label_names = ds.features["label"].names
    img_key = "image" if "image" in ds.features else "img"
    by_label: dict[int, list] = defaultdict(list)
    for row in ds:
        lb = int(row["label"])
        if len(by_label[lb]) < args.per_class:
            by_label[lb].append(row[img_key])
        if all(len(by_label[i]) >= args.per_class for i in range(len(label_names))):
            break
    classes = sorted(by_label)
    words = [WORD_FIX.get(label_names[c], label_names[c]) for c in classes]
    print(f"classes: {words}", flush=True)

    img_emb, img_lbl = [], []
    for c in classes:
        for im in by_label[c]:
            img_emb.append(image_encoded(im)); img_lbl.append(c)
        print(f"  encoded images '{WORD_FIX.get(label_names[c], label_names[c])}'", flush=True)
    img_emb = F.normalize(torch.stack(img_emb), dim=-1)
    img_lbl = torch.tensor(img_lbl)

    # word embeddings for the 10 class words
    word_emb = F.normalize(
        torch.stack([text_encoded("a photo of a " + w) for w in words]), dim=-1)

    # ---- (a) image class separability (ref/query split per class) ----
    refm = torch.zeros(len(img_lbl), dtype=torch.bool)
    for c in classes:
        idx = (img_lbl == c).nonzero(as_tuple=True)[0]
        refm[idx[: len(idx) // 2]] = True
    qrym = ~refm
    ref, reflbl = img_emb[refm], img_lbl[refm]
    qry, qrylbl = img_emb[qrym], img_lbl[qrym]
    cls_index = {c: i for i, c in enumerate(classes)}
    icent = F.normalize(torch.stack(
        [ref[reflbl == c].mean(0) for c in classes]), dim=-1)
    a_top1 = (torch.tensor([classes[i] for i in (qry @ icent.T).argmax(1)]) == qrylbl
              ).float().mean().item()
    sims = qry @ ref.T
    knn = reflbl[sims.topk(min(args.knn_k, ref.shape[0]), dim=1).indices]
    a_knn = (torch.mode(knn, dim=1).values == qrylbl).float().mean().item()

    # ---- (b) image<->word retrieval@1 ----
    iw = img_emb @ word_emb.T  # (n_img, n_class)
    pred = iw.argmax(1)
    b_hits = sum(int(classes[pred[i]] == int(img_lbl[i])) for i in range(len(img_lbl)))
    b_ret = b_hits / len(img_lbl)

    # ---- (c) image -> discourse community ----
    img2comm = (img_emb @ cent_mat.T).argmax(1)
    mapping = {}
    for ci, c in enumerate(classes):
        assigned = [comms[int(img2comm[i])] for i in range(len(img_lbl)) if int(img_lbl[i]) == c]
        top = Counter(comm_label[a] for a in assigned).most_common(3)
        mapping[words[ci]] = top

    chance10 = 1.0 / len(classes)
    print(f"\n(a) image class separability: centroid-top1 {a_top1:.3f} "
          f"kNN {a_knn:.3f} (chance {chance10:.3f}, lift {a_top1/chance10:.1f}x)", flush=True)
    print(f"(b) image<->word retrieval@1: {b_ret:.3f} "
          f"(chance {chance10:.3f}, lift {b_ret/chance10:.1f}x)", flush=True)
    print("(c) image class -> modal discourse community:", flush=True)
    for w, top in mapping.items():
        print(f"    {w:12s} -> {top}", flush=True)

    res = {"model": MID, "ckpt": args.ckpt, "step": ck.get("step"),
           "n_images": len(img_lbl), "classes": words, "chance": chance10,
           "image_class_separability": {"centroid_top1": a_top1, "knn": a_knn},
           "image_word_retrieval_at1": b_ret,
           "image_to_community": {w: [[lbl, n] for lbl, n in top]
                                  for w, top in mapping.items()}}
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
