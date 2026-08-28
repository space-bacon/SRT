#!/usr/bin/env python3
"""Pack thumbnails for the omni manifest images into one file for the demo.

COCO serves val2017 over plain http, which an https Space cannot load without
tripping mixed-content blocking, so the demo needs its own copy. One packed
blob plus an offset index beats 5000 files: the Space fetches it once.
"""
import io
import json
import pathlib
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

MANIFEST = "artifacts/nla/omni/omni_manifest.json"
OUT = pathlib.Path("artifacts/nla/omni/thumbs")
SIZE = 224
URL = "http://images.cocodataset.org/val2017/{}"


def grab(key):
    try:
        r = requests.get(URL.format(key), timeout=30)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((SIZE, SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=78, optimize=True)
        return key, buf.getvalue()
    except Exception:
        return key, None


def main():
    keys = [r["key"] for r in json.load(open(MANIFEST))["rows"]
            if r["modality"] == "image"]
    print(f"{len(keys)} images")
    OUT.mkdir(parents=True, exist_ok=True)

    blobs = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, (k, b) in enumerate(ex.map(grab, keys), 1):
            if b:
                blobs[k] = b
            if i % 500 == 0:
                print(f"  {i}/{len(keys)}  ok={len(blobs)}", flush=True)

    index, offset = {}, 0
    with open(OUT / "thumbs.bin", "wb") as fh:
        for k, b in blobs.items():
            fh.write(b)
            index[k] = (offset, len(b))
            offset += len(b)
    (OUT / "thumbs_index.json").write_text(json.dumps(index))
    mb = offset / 1e6
    print(f"wrote {len(index)} thumbs, {mb:.1f} MB")
    if len(index) < 0.95 * len(keys):
        sys.exit(f"only {len(index)}/{len(keys)} fetched; not enough for the demo")


if __name__ == "__main__":
    main()
