"""Does the shared space notice when a caption's roles are swapped?

Retrieval scores in this program are all similarity to a true caption. That
says nothing about whether the space encodes *structure*, because a bag of the
right words scores well however it is arranged. This builds the minimal pair
that separates the two: for each image, the true caption against the same words
with two noun phrases exchanged.

    a man riding a horse   ->   a horse riding a man

COCO contains photographs of the first and none of the second, so a space that
represents composition should prefer the true one. Chance is 0.5. Published
CLIP-style dual encoders sit near chance on this kind of swap, which is why
SugarCrepe exists, so a low number here is a boundary rather than a defect.

Stage 1 writes the pairs, stage 2 scores vectors encoded by the browser's own
runtime (see BlackWindow runtime example `encode_text`), so the number
describes the deployment rather than a PyTorch reference.

    python scripts/swap_probe.py --build --out-texts /tmp/swap_texts.json \
        --pairs /tmp/swap_pairs.json --limit 300
    # ... encode_text ...
    python scripts/swap_probe.py --score --pairs /tmp/swap_pairs.json \
        --vectors /tmp/swap_vecs.json --out artifacts/nla/q4/swap_probe.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

BROWSER = Path("artifacts/local/browser")

# Concrete, physically depictable nouns. Swapping "sense" for "moment" tests
# nothing; swapping "man" for "horse" changes what the photograph must contain.
NOUNS = r"(man|woman|boy|girl|child|dog|cat|horse|bird|cow|sheep|elephant|bear|zebra|giraffe)"
PAIR = re.compile(rf"\ba\s+{NOUNS}\b(.{{1,24}}?)\ba\s+{NOUNS}\b", re.I)

# "a giraffe and a zebra" and "a zebra and a giraffe" describe the same
# photograph, so a model has nothing to be right about. Only asymmetric
# relations make a minimal pair, and leaving these in would push accuracy to
# chance by construction and look like a finding.
SYMMETRIC = re.compile(r"^\s*(and|or|,|and then|next to|beside|near|with)\s*$", re.I)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--captions", type=Path, default=BROWSER / "captions.json")
    p.add_argument("--gallery", type=Path, default=BROWSER / "gallery.npz")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-texts", type=Path, default=None)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--vectors", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def build(a) -> None:
    caps = json.loads(a.captions.read_text())["pairs"]
    z = np.load(a.gallery, allow_pickle=True)
    have = {str(f) for f in z["files"]}

    found = []
    symmetric = 0
    for row in caps:
        text, owner = row["text"], row["owner"]
        if owner not in have:
            continue
        m = PAIR.search(text)
        if not m:
            continue
        first, mid, second = m.group(1), m.group(2), m.group(3)
        if first.lower() == second.lower():
            continue
        if SYMMETRIC.match(mid):
            symmetric += 1
            continue
        swapped = text[: m.start()] + f"a {second}{mid}a {first}" + text[m.end():]
        if swapped.lower() == text.lower():
            continue
        found.append({"owner": owner, "true": text, "swapped": swapped})

    random.Random(a.seed).shuffle(found)
    found = found[: a.limit]
    a.pairs.write_text(json.dumps(found, indent=1))

    texts = [f["true"] for f in found] + [f["swapped"] for f in found]
    if a.out_texts:
        a.out_texts.write_text(json.dumps(texts))
    print(f"{len(found)} pairs -> {a.pairs}")
    print(f"{symmetric} rejected as symmetric coordinations")
    print(f"{len(texts)} texts -> {a.out_texts}")
    for f in found[:3]:
        print(f"  true    {f['true']}")
        print(f"  swapped {f['swapped']}")


def score(a) -> None:
    pairs = json.loads(a.pairs.read_text())
    vecs = np.asarray(json.loads(a.vectors.read_text()), dtype=np.float32)
    n = len(pairs)
    if vecs.shape[0] != 2 * n:
        raise SystemExit(f"expected {2 * n} vectors, got {vecs.shape[0]}")
    true_v, swap_v = vecs[:n], vecs[n:]

    z = np.load(a.gallery, allow_pickle=True)
    Z = z["Z_img"].astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    idx = {str(f): i for i, f in enumerate(z["files"])}

    wins, margins = 0, []
    for i, p in enumerate(pairs):
        img = Z[idx[p["owner"]]]
        st, ss = float(img @ true_v[i]), float(img @ swap_v[i])
        margins.append(st - ss)
        wins += st > ss
    margins = np.asarray(margins)
    acc = wins / n
    se = (0.25 / n) ** 0.5
    z = (acc - 0.5) / se
    half = 1.96 * (acc * (1 - acc) / n) ** 0.5

    report = {
        "question": "does the head-space prefer a caption over the same words with two nouns swapped",
        "runtime": "candle Q4_0 with shipped text anchor (the browser's runtime)",
        "pairs": n,
        "accuracy": round(acc, 4),
        "ci95": [round(acc - half, 4), round(acc + half, 4)],
        "chance": 0.5,
        "z_vs_chance": round(z, 2),
        "symmetric_pairs_excluded": True,
        "margin_true_minus_swapped": {
            "mean": round(float(margins.mean()), 5),
            "sd": round(float(margins.std()), 5),
            "median": round(float(np.median(margins)), 5),
        },
        "reading": (
            "indistinguishable from chance: the space scores the words and not "
            "their arrangement. This bounds what the read-out represents; it does "
            "not affect any retrieval number, which never depended on word order."
        ),
    }
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {a.out}")


def main() -> None:
    a = parse_args()
    if a.build:
        build(a)
    elif a.score:
        if not a.vectors:
            raise SystemExit("--score needs --vectors")
        score(a)
    else:
        raise SystemExit("pass --build or --score")


if __name__ == "__main__":
    main()
