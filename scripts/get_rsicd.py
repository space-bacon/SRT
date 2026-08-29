#!/usr/bin/env python3
"""Fetch RSICD and emit it in the manifest shape the omni encoder already takes.

The cross-vendor result was measured on consumer photographs, which is the one
domain where every vendor scraped roughly the same internet. Dipankar Sarkar
put the objection precisely: run the same twelve cross directions over a domain
the four pretraining mixes disagree about, and see whether retention survives.
If it holds the invariance belongs to the readout; if it drops, part of the
original number was four models having seen the same pictures.

Satellite imagery is that domain. RSICD is 10,921 aerial images with five
captions each, so it matches the COCO protocol row for row and only the pixels
change.

    python scripts/get_rsicd.py --n 5000 --out-dir /root/rsicd
"""
import argparse
import io
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out-dir", default="/root/rsicd")
    ap.add_argument("--manifest", default="/root/rsicd_manifest.json")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/root/.hf_home")
    from datasets import load_dataset
    from PIL import Image

    os.makedirs(a.out_dir, exist_ok=True)
    ds = load_dataset("arampacha/rsicd", split="train")
    print(f"loaded {len(ds)} rows, taking {a.n}", flush=True)

    rows = []
    for i, r in enumerate(ds):
        if len(rows) >= a.n:
            break
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        # Five captions per image; take the first so the protocol matches the
        # COCO run, which also used one caption per image.
        caps = r.get("captions") or r.get("caption")
        cap = (caps[0] if isinstance(caps, list) else caps or "").strip()
        if not cap:
            continue
        key = f"rsicd_{i:06d}.jpg"
        img.convert("RGB").save(os.path.join(a.out_dir, key), "JPEG", quality=92)
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
