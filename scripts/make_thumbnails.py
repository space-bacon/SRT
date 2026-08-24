"""Thumbnail the gallery so it can be served from a page that uses TLS.

images.cocodataset.org serves HTTP only. A page delivered over HTTPS has those
images blocked as mixed content, silently, so a deployed demo shows a grid of
broken frames while every score reads correctly. The images have to come from
somewhere we control.

Full COCO is 19GB. At 128px the same 123K images are well under a gigabyte,
which is what a gallery of thumbnails actually needs to be.

    python scripts/make_thumbnails.py --src /root/train2017 --split train2017 \
        --out /root/thumbs --size 128 --workers 32
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--split", required=True, help="subdirectory to write under")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--quality", type=int, default=72)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    return p.parse_args()


def one(job):
    src, dst, size, quality = job
    if os.path.exists(dst):
        return 0
    try:
        from PIL import Image

        im = Image.open(src).convert("RGB")
        # Square centre crop: the grid is square, and letterboxing 123K
        # thumbnails wastes both bytes and the reader's attention.
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.LANCZOS)
        im.save(dst, "JPEG", quality=quality, optimize=True)
        return os.path.getsize(dst)
    except Exception as e:
        print(f"skip {src}: {type(e).__name__} {e}", flush=True)
        return 0


def main() -> None:
    a = parse_args()
    dest = a.out / a.split
    dest.mkdir(parents=True, exist_ok=True)
    names = sorted(f for f in os.listdir(a.src) if f.lower().endswith(".jpg"))
    jobs = [(str(a.src / n), str(dest / n), a.size, a.quality) for n in names]
    print(f"{len(jobs)} images -> {dest} at {a.size}px", flush=True)

    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, n in enumerate(ex.map(one, jobs, chunksize=64)):
            total += n
            if i % 10000 == 0:
                print(f"  {i}/{len(jobs)}  {total / 1e6:.0f} MB", flush=True)
    print(f"done: {len(jobs)} thumbnails, {total / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
