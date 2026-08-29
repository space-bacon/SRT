#!/usr/bin/env python3
"""Fetch ROCO radiology and emit it in the manifest shape the encoder takes.

Second domain control, after RSICD. Dipankar Sarkar's objection to the
cross-vendor result was that consumer photographs are the one case where every
vendor scraped roughly the same internet, so agreement between four models
could be shared training data rather than shared structure. Satellite tests
that. Radiology tests it again, further out: these are greyscale CT and MRI
slices with clinical captions, which is about as far from web alt-text as
paired image-text data gets.

Streamed rather than downloaded, because only 5000 of the 80k rows are needed
and the full repo is 13 GB the box does not have to spare.

    python scripts/get_roco.py --n 5000 --out-dir /root/roco
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dataset", default="eltorio/ROCO-radiology")
    ap.add_argument("--out-dir", default="/root/roco")
    ap.add_argument("--manifest", default="/root/roco_manifest.json")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/root/.hf_home")
    from datasets import load_dataset

    os.makedirs(a.out_dir, exist_ok=True)
    ds = load_dataset(a.dataset, split="train", streaming=True)
    print(f"streaming {a.dataset}, taking {a.n}", flush=True)

    rows = []
    for i, r in enumerate(ds):
        if len(rows) >= a.n:
            break
        cap = (r.get("caption") or "").strip()
        # A handful of captions are a bare figure label, too short to retrieve on.
        if len(cap) < 12:
            continue
        key = f"roco_{i:06d}.jpg"
        try:
            r["image"].convert("RGB").save(
                os.path.join(a.out_dir, key), "JPEG", quality=92)
        except Exception as e:
            print(f"  skip {key}: {type(e).__name__}", flush=True)
            continue
        rows.append({"modality": "image", "key": key, "caption": cap})
        if len(rows) % 1000 == 0:
            print(f"  {len(rows)}", flush=True)

    json.dump({"counts": {"image": len(rows)}, "rows": rows},
              open(a.manifest, "w"))
    print(f"\n{len(rows)} images -> {a.out_dir}")
    print(f"manifest -> {a.manifest}")
    print("sample:", json.dumps(rows[0], indent=1))


if __name__ == "__main__":
    main()
