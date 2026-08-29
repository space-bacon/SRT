#!/usr/bin/env python3
"""Measure caption uniqueness per gallery, which bounds r@1 before any model runs.

Retrieval scores are not comparable across corpora until you know this. If a
caption is shared by k images, those k rows cannot all rank their own image
first, so r@1 carries a ceiling set by the corpus rather than by the models.
RSICD is templated and this matters there; ROCO is not and it does not.

    python scripts/caption_uniqueness.py artifacts/nla/omni/*_manifest.json
"""
import argparse
import collections
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="artifacts/nla/omni/caption_uniqueness.json")
    a = ap.parse_args()

    report = {}
    for path in a.manifests:
        rows = [r for r in json.load(open(path))["rows"]
                if r["modality"] == "image"][:a.n]
        caps = [r["caption"].strip().lower() for r in rows]
        counts = collections.Counter(caps)
        shared = sum(v for v in counts.values() if v > 1)
        top_cap, top_n = counts.most_common(1)[0]
        name = pathlib.Path(path).stem.replace("_manifest", "")
        report[name] = {
            "n": len(caps),
            "unique_captions": len(counts),
            "unique_pct": round(100 * len(counts) / len(caps), 1),
            "rows_sharing_a_caption": shared,
            "shared_pct": round(100 * shared / len(caps), 1),
            "most_repeated_count": top_n,
            "most_repeated_caption": top_cap,
            "r_at_1_ceiling": round(len(counts) / len(caps), 3),
        }
        r = report[name]
        print(f"{name:10s} n={r['n']}  unique {r['unique_pct']}%  "
              f"sharing {r['shared_pct']}%  most repeated x{r['most_repeated_count']}  "
              f"r@1 ceiling ~{r['r_at_1_ceiling']}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
