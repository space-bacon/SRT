"""Encode COCO train2017 image/caption L47 pairs in resumable chunks.

Feeds the scaled cross-modal alignment runs (§11.6.4b follow-up: how far
does the trained-linear ceiling rise, and does the MLP overtake it, with
an order of magnitude more pairs?). Reuses the Encoder + COCO plumbing
from gemma4_procrustes_xmodal.py so conventions stay bit-identical
(image = mean over image soft-token positions at L47; caption 0 =
BOS-prefixed last-token L47).

Chunks of --chunk-size pairs are written as
  {work_dir}/train_pairs/chunk_{i:03d}.pt  with keys {img, cap0, files}
and existing chunks are skipped, so the job is safe to kill/relaunch.

    python scripts/gemma4_encode_pairs.py \
        --work-dir /workspace/procrustes --n-images 40000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("encode_pairs")

_SPEC = importlib.util.spec_from_file_location(
    "gemma4_procrustes_xmodal", Path(__file__).parent / "gemma4_procrustes_xmodal.py")
_PX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PX)

COCO_TRAIN_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="google/gemma-4-31B-it")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--n-images", type=int, default=40000)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--caption-batch", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--processor-cache", default=None)
    return p.parse_args()


def ensure_train2017(work: Path) -> list[dict]:
    img_dir = work / "train2017"
    ann_json = work / "annotations" / "captions_train2017.json"
    if not ann_json.exists():
        raise SystemExit("annotations missing — run the val2017 chain first "
                         "(annotations_trainval2017.zip contains both)")
    if not img_dir.exists():
        zpath = work / "train2017.zip"
        if not zpath.exists():
            log.info("downloading %s (~19GB) ...", COCO_TRAIN_URL)
            urllib.request.urlretrieve(COCO_TRAIN_URL, zpath)  # noqa: S310
        log.info("extracting %s ...", zpath.name)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(work)
    ann = json.loads(ann_json.read_text())
    caps_by_img: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        caps_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
    rows = []
    for im in ann["images"]:
        caps = caps_by_img.get(im["id"], [])
        if caps:
            rows.append({"file": str(img_dir / im["file_name"]),
                         "caption": caps[0]})
    rows.sort(key=lambda r: str(r["file"]))
    log.info("train2017: %d captioned images", len(rows))
    return rows


def main() -> None:
    args = parse_args()
    rows = ensure_train2017(args.work_dir)[: args.n_images]
    out_dir = args.work_dir / "train_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    enc = None  # lazy: skip backbone load if all chunks already exist
    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        out = out_dir / f"chunk_{ci:03d}.pt"
        if out.exists():
            log.info("chunk %d exists, skipping", ci)
            continue
        if enc is None:
            enc = _PX.Encoder(args)
        chunk = rows[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        img_vs = []
        for i, r in enumerate(chunk):
            img_vs.append(enc.image_v(Image.open(r["file"]).convert("RGB")))
            if (i + 1) % 200 == 0:
                log.info("chunk %d images %d/%d", ci, i + 1, len(chunk))
        cap_vs = []
        for j in range(0, len(chunk), args.caption_batch):
            cap_vs.append(enc.text_vs(
                [r["caption"] for r in chunk[j : j + args.caption_batch]]))
        obj = {"img": torch.stack(img_vs), "cap0": torch.cat(cap_vs),
               "files": [r["file"] for r in chunk]}
        tmp = out.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.rename(out)
        log.info("wrote %s (%d pairs)", out, len(chunk))
    log.info("ALL CHUNKS DONE (%d)", n_chunks)


if __name__ == "__main__":
    main()
