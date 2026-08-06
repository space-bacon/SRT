#!/usr/bin/env python
"""Generate compositional hard negatives for the 117k train2017 captions.

v4 recipe (SugarCrepe diagnostic, artifacts/nla/q4/
sugarcrepe_diagnostic_20260806.json): the raw L47 states separate
word-order swaps better than object replacements, but InfoNCE with
random in-batch negatives lets the head project that axis away. Fix the
objective: give every training caption a rule-based compositional
perturbation as an explicit hard negative.

Rules per caption (first applicable):
  1. swap two distinct NOUN tokens        (swap_obj analogue)
  2. swap two distinct ADJ tokens         (swap_att analogue)
  3. swap ADJ across two noun chunks
  4. fallback: swap two content words (len > 3)

Alignment: rows are rebuilt exactly as gemma4_encode_pairs.ensure_train2017
does (first annotation per image, sorted by file path), so negative i
pairs with training row i in the chunk concatenation.

    python make_comp_negatives.py --work-dir /root/comp_negs \
        --ann /root/annotations/captions_train2017.json --n-images 120000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", type=Path, required=True,
                   help="captions_train2017.json")
    p.add_argument("--work-dir", type=Path, default=Path("comp_negs"))
    p.add_argument("--n-images", type=int, default=120000)
    p.add_argument("--k-negs", type=int, default=1,
                   help="negatives per caption: 1 = swap only (v4); "
                        "3 = swap + noun-replace + verb/adj-perturb (v5)")
    p.add_argument("--recipe", default="v5", choices=["v5", "v6"],
                   help="v6 = relation/attribute families first "
                        "(preposition replace, adjective transfer between "
                        "noun chunks), then order (noun swap), then "
                        "attribute replace. Reader round-6 mechanism: "
                        "object identity is already enforced by in-batch "
                        "negatives; relations and attributes are the "
                        "discarded axes.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_rows(ann_path: Path, n_images: int) -> list[dict]:
    ann = json.loads(ann_path.read_text())
    caps_by_img: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        caps_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
    rows = []
    for im in ann["images"]:
        caps = caps_by_img.get(im["id"])
        if caps:
            rows.append({"file": im["file_name"], "caption": caps[0]})
    rows.sort(key=lambda r: str(r["file"]))
    return rows[:n_images]


def perturb(doc, rng: random.Random) -> tuple[str, str] | None:
    toks = [t.text for t in doc]

    def swapped(i, j):
        out = toks.copy()
        out[i], out[j] = out[j], out[i]
        return " ".join(out).replace(" ,", ",").replace(" .", ".")

    nouns = [t.i for t in doc if t.pos_ == "NOUN"]
    seen: dict[str, int] = {}
    for i in nouns:
        w = doc[i].lemma_.lower()
        if w in seen and doc[seen[w]].text.lower() != doc[i].text.lower():
            continue
        if w not in seen:
            seen[w] = i
    distinct = list(seen.values())
    if len(distinct) >= 2:
        i, j = rng.sample(distinct, 2)
        if doc[i].text.lower() != doc[j].text.lower():
            return swapped(i, j), "swap_noun"

    adjs = [t.i for t in doc if t.pos_ == "ADJ"]
    adj_distinct = []
    seen_a: set[str] = set()
    for i in adjs:
        w = doc[i].text.lower()
        if w not in seen_a:
            seen_a.add(w)
            adj_distinct.append(i)
    if len(adj_distinct) >= 2:
        i, j = rng.sample(adj_distinct, 2)
        return swapped(i, j), "swap_adj"

    content = [t.i for t in doc
               if len(t.text) > 3 and t.pos_ in ("NOUN", "ADJ", "VERB", "PROPN")]
    uniq = []
    seen_c: set[str] = set()
    for i in content:
        w = doc[i].text.lower()
        if w not in seen_c:
            seen_c.add(w)
            uniq.append(i)
    if len(uniq) >= 2:
        i, j = rng.sample(uniq, 2)
        return swapped(i, j), "swap_content"
    return None


def replace_word(doc, pos: str, vocab: list[str],
                 rng: random.Random) -> tuple[str, str] | None:
    """Replace one token of the given POS with a random vocab word."""
    cands = [t.i for t in doc if t.pos_ == pos]
    if not cands or not vocab:
        return None
    i = rng.choice(cands)
    old = doc[i].text.lower()
    for _ in range(8):
        new = rng.choice(vocab)
        if new != old:
            break
    else:
        return None
    toks = [t.text for t in doc]
    toks[i] = new
    out = " ".join(toks).replace(" ,", ",").replace(" .", ".")
    return out, f"replace_{pos.lower()}"


SPATIAL_PREPS = ["on", "in", "under", "above", "behind", "beside",
                 "over", "near", "below", "inside", "outside", "against",
                 "across", "around", "between", "beneath"]


def prep_replace(doc, rng: random.Random) -> tuple[str, str] | None:
    """Relation negative: swap one spatial preposition for another."""
    cands = [t.i for t in doc
             if t.pos_ == "ADP" and t.text.lower() in SPATIAL_PREPS]
    if not cands:
        return None
    i = rng.choice(cands)
    old = doc[i].text.lower()
    new = rng.choice([p for p in SPATIAL_PREPS if p != old])
    toks = [t.text for t in doc]
    toks[i] = new
    out = " ".join(toks).replace(" ,", ",").replace(" .", ".")
    return out, "prep_replace"


def adj_transfer(doc, rng: random.Random) -> tuple[str, str] | None:
    """Attribute negative: move an adjective from one noun to another
    ("a red car and a blue door" -> "a blue car and a red door" style
    reassignment, or one-way move when only one ADJ exists)."""
    pairs = [(t.i, t.head.i) for t in doc
             if t.pos_ == "ADJ" and t.dep_ == "amod" and t.head.pos_ == "NOUN"]
    nouns = [t.i for t in doc if t.pos_ == "NOUN"]
    if not pairs or len(nouns) < 2:
        return None
    adj_i, noun_i = rng.choice(pairs)
    targets = [n for n in nouns
               if n != noun_i and doc[n].text.lower() != doc[noun_i].text.lower()]
    if not targets:
        return None
    tgt = rng.choice(targets)
    toks = [t.text for t in doc]
    adj = toks[adj_i]
    del toks[adj_i]
    tgt_pos = tgt if tgt < adj_i else tgt - 1
    toks.insert(tgt_pos, adj)
    out = " ".join(toks).replace(" ,", ",").replace(" .", ".")
    return out, "adj_transfer"


def gen_k(doc, caption: str, vocab_noun: list[str], vocab_verb: list[str],
          vocab_adj: list[str], rng: random.Random, k: int,
          kinds: dict, recipe: str = "v5") -> list[str]:
    """k diverse negatives. v5: order-first. v6: relations/attributes
    first (round-6 mechanism: object identity is already enforced by
    in-batch negatives; the discarded axes are relations and attributes)."""
    negs: list[str] = []

    def add(res):
        if res and res[0].lower() != caption.lower() and res[0] not in negs:
            negs.append(res[0])
            kinds[res[1]] = kinds.get(res[1], 0) + 1
            return True
        return False

    if recipe == "v6":
        add(prep_replace(doc, rng))                          # relation
        if len(negs) < k:
            add(adj_transfer(doc, rng))                      # attribute move
        if len(negs) < k:
            add(perturb(doc, rng))                           # order (swap)
        if len(negs) < k:
            if not add(replace_word(doc, "ADJ", vocab_adj, rng)):
                add(replace_word(doc, "VERB", vocab_verb, rng))
    else:
        add(perturb(doc, rng))                               # 1: swap
        if len(negs) < k:
            add(replace_word(doc, "NOUN", vocab_noun, rng))  # 2: noun replace
        if len(negs) < k:
            if not add(replace_word(doc, "VERB", vocab_verb, rng)):
                add(replace_word(doc, "ADJ", vocab_adj, rng))
    while len(negs) < k:                                     # last resort
        if not add(replace_word(doc, "NOUN", vocab_noun, rng)):
            negs.append(caption)                             # identity pad
            kinds["identity"] = kinds.get("identity", 0) + 1
    return negs[:k]


def main():
    import spacy

    args = parse_args()
    rng = random.Random(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.ann, args.n_images)
    print(f"{len(rows)} rows, k={args.k_negs}", flush=True)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.enable_pipe("lemmatizer") if "lemmatizer" in nlp.disabled else None

    caps = [r["caption"] for r in rows]
    kinds: dict[str, int] = {}

    if args.k_negs == 1:
        negatives = []
        for k, doc in enumerate(nlp.pipe(caps, batch_size=256)):
            res = perturb(doc, rng)
            if res is None or res[0].lower() == caps[k].lower():
                negatives.append(caps[k])      # identity: contributes nothing
                kinds["identity"] = kinds.get("identity", 0) + 1
            else:
                negatives.append(res[0])
                kinds[res[1]] = kinds.get(res[1], 0) + 1
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{len(caps)}", flush=True)
        out = args.work_dir / "comp_negatives.json"
        out.write_text(json.dumps({"negatives": negatives, "stats": kinds}))
        sample = list(zip(caps, negatives))[:5]
    else:
        # pass 1: corpus vocab for the replace rules
        vocab_noun: set[str] = set()
        vocab_verb: set[str] = set()
        vocab_adj: set[str] = set()
        docs = list(nlp.pipe(caps, batch_size=256))
        for doc in docs:
            for t in doc:
                if t.pos_ == "NOUN":
                    vocab_noun.add(t.text.lower())
                elif t.pos_ == "VERB":
                    vocab_verb.add(t.text.lower())
                elif t.pos_ == "ADJ":
                    vocab_adj.add(t.text.lower())
        vn, vv, va = sorted(vocab_noun), sorted(vocab_verb), sorted(vocab_adj)
        print(f"vocab: {len(vn)} nouns, {len(vv)} verbs, {len(va)} adjs",
              flush=True)
        flat: list[str] = []
        for k, doc in enumerate(docs):
            flat.extend(gen_k(doc, caps[k], vn, vv, va, rng,
                              args.k_negs, kinds, recipe=args.recipe))
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{len(caps)}", flush=True)
        out = args.work_dir / f"comp_negatives_{args.recipe}_k{args.k_negs}.json"
        out.write_text(json.dumps({"negatives": flat, "k": args.k_negs,
                                   "stats": kinds}))
        sample = [(caps[i], " | ".join(flat[i * args.k_negs:(i + 1) * args.k_negs]))
                  for i in range(5)]

    print(f"stats: {kinds}", flush=True)
    print(f"wrote {out}", flush=True)
    for a, b in sample:
        print(f"  POS: {a}\n  NEG: {b}", flush=True)


if __name__ == "__main__":
    main()
