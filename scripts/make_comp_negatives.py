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


def main():
    import spacy

    args = parse_args()
    rng = random.Random(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.ann, args.n_images)
    print(f"{len(rows)} rows", flush=True)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.enable_pipe("lemmatizer") if "lemmatizer" in nlp.disabled else None

    negatives, kinds = [], {"swap_noun": 0, "swap_adj": 0,
                            "swap_content": 0, "identity": 0}
    caps = [r["caption"] for r in rows]
    for k, doc in enumerate(nlp.pipe(caps, batch_size=256)):
        res = perturb(doc, rng)
        if res is None or res[0].lower() == caps[k].lower():
            negatives.append(caps[k])          # identity: contributes nothing
            kinds["identity"] += 1
        else:
            negatives.append(res[0])
            kinds[res[1]] += 1
        if (k + 1) % 20000 == 0:
            print(f"  {k + 1}/{len(caps)}", flush=True)

    out = args.work_dir / "comp_negatives.json"
    out.write_text(json.dumps({"negatives": negatives, "stats": kinds}))
    print(f"stats: {kinds}", flush=True)
    print(f"wrote {out}", flush=True)
    for a, b in list(zip(caps, negatives))[:5]:
        print(f"  POS: {a}\n  NEG: {b}", flush=True)


if __name__ == "__main__":
    main()
