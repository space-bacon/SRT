#!/usr/bin/env python3
"""Pick example photographs for the blind-reader Space.

Two constraints. No people, and breadth: six pictures of animals would not show
that the reader handles anything else.

Only captions_val2017.json is on the box, so "no people" is enforced by
rejecting an image if ANY of its five captions uses a person word. That is
strict but not airtight, since a caption can omit someone who is in frame, so
the picks still get looked at before they ship.

Candidates come from outside the images the Space indexes, otherwise the top
retrieval hit is the input picture coming back to meet itself.

    python scripts/pick_space_examples.py --n 8 --skip 1000
"""
import argparse
import json
import os
import re
from collections import defaultdict

from PIL import Image

IMG_DIR = "/root/val2017"
CAPS = "/root/annotations/captions_val2017.json"
OUT = "/root/space_examples"

PEOPLE = """person people man men woman women child children kid kids boy boys
girl girls baby babies lady ladies guy guys someone somebody crowd player
players skier skiers snowboarder surfer surfers rider riders driver pedestrian
pedestrians worker chef cook waiter officer policeman fireman couple family
teenager toddler adult adults male female his her he she hand hands arm face
""".split()

THEMES = {
    "animal": "cat dog bear elephant giraffe zebra horse sheep cow bird cats "
              "dogs birds zebras giraffes elephants".split(),
    "food": "pizza sandwich cake donut banana broccoli food plate meal fruit "
            "vegetables salad dessert".split(),
    "vehicle": "car truck bus train motorcycle airplane plane boat bicycle "
               "cars trucks buses trains".split(),
    "interior": "room bedroom bathroom kitchen living toilet sink couch bed "
                "desk chair table shelf".split(),
    "street": "street city building road intersection sign traffic parking "
              "sidewalk corner".split(),
    "nature": "beach ocean field mountain grass snow forest sky water lake "
              "hill trees flowers".split(),
    "objects": "clock vase laptop computer keyboard phone television remote "
               "scissors umbrella luggage book books bottle".split(),
    "sport": "kite frisbee skateboard surfboard tennis baseball ball racket "
             "court bat glove".split(),
}


def words(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--per-theme", type=int, default=1)
    ap.add_argument("--skip", type=int, default=1000)
    ap.add_argument("--px", type=int, default=640)
    a = ap.parse_args()

    d = json.load(open(CAPS))
    names = {im["id"]: im["file_name"] for im in d["images"]}
    caps = defaultdict(list)
    for c in d["annotations"]:
        caps[c["image_id"]].append(c["caption"].strip())

    indexed = set(sorted(os.listdir(IMG_DIR))[:a.skip])
    people = set(PEOPLE)
    picks, used = [], defaultdict(int)
    for iid in sorted(caps):
        f = names[iid]
        if f in indexed or len(caps[iid]) < 5:
            continue
        w = set().union(*(words(c) for c in caps[iid]))
        if w & people:
            continue
        theme = next((t for t, kw in THEMES.items()
                      if w & set(kw) and used[t] < a.per_theme), None)
        if theme is None:
            continue
        used[theme] += 1
        picks.append((f, theme, caps[iid][0]))
        if len(picks) == a.n:
            break

    os.makedirs(OUT, exist_ok=True)
    for f, theme, cap in picks:
        im = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
        im.thumbnail((a.px, a.px), Image.LANCZOS)
        im.save(os.path.join(OUT, f), "JPEG", quality=88)
        print(f"  {f}  [{theme}]  {cap}")
    print(f"\n{len(picks)} examples -> {OUT}")


if __name__ == "__main__":
    main()
