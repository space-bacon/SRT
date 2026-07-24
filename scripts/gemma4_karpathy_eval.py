"""Literature-standard Karpathy-split COCO retrieval eval for the linear head.

The published §11.6.4 numbers use our own protocol (val2017 eval, 5k
caption pool). Before comparing against published tables, this script
runs the standard **Karpathy 5k test** protocol: 5,000 test images
(drawn from COCO val2014), i2t against their 25,000 captions, t2i with
25,000 caption queries against the 5,000 images.

CRITICAL leakage control: COCO's 2017 re-partition moved most of
val2014 INTO train2017, so Karpathy-test images appear in our train2017
pair chunks. This script excludes all Karpathy test+val image ids from
the training pool and RETRAINS the linear head before evaluating.

Stages (all cached/resumable):
  1. Download Karpathy dataset_coco.json (caption_datasets.zip).
  2. Encode the 5k test images + 25k captions at L47 (cached .pt).
  3. Filter train chunks by image id; retrain linear head (InfoNCE).
  4. Report zero-training centered baseline + trained-linear, both
     directions, R@1/5/10 + median rank.

    python scripts/gemma4_karpathy_eval.py \
        --work-dir /workspace/procrustes \
        --train-encodings /workspace/procrustes/train_pairs/chunk_*.pt \
        --out /workspace/procrustes/karpathy_eval.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import urllib.request
import zipfile
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("karpathy")

_HERE = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PX = _load("gemma4_procrustes_xmodal")     # Encoder
_MA = _load("gemma4_mlp_align")             # train_heads, retrieval_eval

KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="google/gemma-4-31B-it")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="dir containing train2017/ and val2017/ image dirs")
    p.add_argument("--train-encodings", type=Path, nargs="+", required=True)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--caption-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--processor-cache", default=None)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_karpathy(work: Path) -> dict:
    """Return {'test': [{'cocoid', 'captions'}], 'excluded_ids': set}."""
    js = work / "dataset_coco.json"
    if not js.exists():
        zpath = work / "caption_datasets.zip"
        if not zpath.exists():
            log.info("downloading %s ...", KARPATHY_URL)
            urllib.request.urlretrieve(KARPATHY_URL, zpath)  # noqa: S310
        with zipfile.ZipFile(zpath) as z:
            z.extract("dataset_coco.json", work)
    data = json.loads(js.read_text())
    test, excluded = [], set()
    for im in data["images"]:
        if im["split"] in ("test", "val"):
            excluded.add(int(im["cocoid"]))
        if im["split"] == "test":
            caps = [s["raw"].strip() for s in im["sentences"]][:5]
            if len(caps) == 5:
                test.append({"cocoid": int(im["cocoid"]), "captions": caps})
    test.sort(key=lambda r: r["cocoid"])
    log.info("karpathy: %d test images, %d excluded ids (test+val)",
             len(test), len(excluded))
    return {"test": test, "excluded_ids": excluded}


def find_image(work: Path, cocoid: int) -> Path:
    for sub in ("train2017", "val2017"):
        p = work / sub / f"{cocoid:012d}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"cocoid {cocoid} not in train2017/ or val2017/")


def encode_test_set(args, test: list[dict]) -> dict:
    cache = args.work_dir / f"karpathy_test_L{args.layer}.pt"
    if cache.exists():
        log.info("loading cached test encodings %s", cache)
        return torch.load(cache, map_location="cpu", weights_only=True)

    from PIL import Image

    enc = _PX.Encoder(args)
    img_vs = []
    for i, r in enumerate(test):
        img_vs.append(enc.image_v(
            Image.open(find_image(args.work_dir, r["cocoid"])).convert("RGB")))
        if (i + 1) % 250 == 0:
            log.info("test images %d/%d", i + 1, len(test))
    flat = [c for r in test for c in r["captions"]]
    cap_vs = []
    for j in range(0, len(flat), args.caption_batch):
        cap_vs.append(enc.text_vs(flat[j : j + args.caption_batch]))
        if (j // args.caption_batch) % 40 == 0:
            log.info("test captions %d/%d", j, len(flat))
    obj = {"img": torch.stack(img_vs), "cap5": torch.cat(cap_vs),
           "cocoids": [r["cocoid"] for r in test]}
    tmp = cache.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(cache)
    log.info("cached -> %s", cache)
    return obj


_ID_RE = re.compile(r"(\d{12})\.jpg$")


def load_train_pool(paths, excluded: set) -> tuple[torch.Tensor, torch.Tensor, int]:
    imgs, caps, dropped = [], [], 0
    for pth in paths:
        ch = torch.load(pth, map_location="cpu", weights_only=True)
        keep = []
        for i, f in enumerate(ch["files"]):
            m = _ID_RE.search(f)
            cocoid = int(m.group(1)) if m else -1
            if cocoid in excluded:
                dropped += 1
            else:
                keep.append(i)
        idx = torch.tensor(keep, dtype=torch.long)
        imgs.append(ch["img"].float()[idx])
        caps.append(ch["cap0"].float()[idx])
    return torch.cat(imgs), torch.cat(caps), dropped


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    k = load_karpathy(args.work_dir)
    test_enc = encode_test_set(args, k["test"])
    img_te = test_enc["img"].float().to(device)          # (5000, d)
    cap_te = test_enc["cap5"].float().to(device)         # (25000, d)
    n_te = img_te.size(0)
    i2t_hits = [set(range(5 * i, 5 * i + 5)) for i in range(n_te)]
    # t2i: 25k caption queries; hit = the caption's own image
    t2i_hits = [{i // 5} for i in range(5 * n_te)]

    img_tr, cap_tr, dropped = load_train_pool(args.train_encodings, k["excluded_ids"])
    n_pool = img_tr.size(0)
    n_train = n_pool - args.n_val
    log.info("train pool %d pairs after dropping %d leaked (karpathy test+val)",
             n_pool, dropped)
    img_tr, cap_tr = img_tr.to(device), cap_tr.to(device)

    results: dict = {
        "protocol": "karpathy 5k test (i2t: 5000 imgs vs 25000 caps; "
                    "t2i: 25000 caps vs 5000 imgs)",
        "backbone": args.backbone, "layer": args.layer,
        "n_train": n_train, "n_dropped_leaked": dropped, "seed": args.seed,
        "runs": {},
    }

    # Zero-training centered baseline (means from the leakage-filtered pool).
    mu_x, mu_y = img_tr[:n_train].mean(0), cap_tr[:n_train].mean(0)
    results["runs"]["baseline_centered"] = {
        "i2t": _MA.retrieval_eval(img_te - mu_x, cap_te - mu_y, i2t_hits),
        "t2i": _MA.retrieval_eval(cap_te - mu_y, img_te - mu_x, t2i_hits),
    }
    log.info("baseline %s", results["runs"]["baseline_centered"])

    # Trained linear head on the leakage-filtered pool.
    heads, fit_info = _MA.train_heads(
        "linear",
        img_tr[:n_train] - mu_x, cap_tr[:n_train] - mu_y,
        img_tr[n_train:] - mu_x, cap_tr[n_train:] - mu_y,
        hidden=args.hidden, proj_dim=args.proj_dim, tau=args.temperature,
        lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, device=device,
    )
    with torch.no_grad():
        i2t = _MA.retrieval_eval(heads["img"](img_te - mu_x),
                                 heads["txt"](cap_te - mu_y), i2t_hits)
        t2i = _MA.retrieval_eval(heads["txt"](cap_te - mu_y),
                                 heads["img"](img_te - mu_x), t2i_hits)
    results["runs"]["linear_leakfree"] = {**fit_info, "i2t": i2t, "t2i": t2i}
    log.info("linear_leakfree %s | t2i %s", i2t, t2i)

    args.out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
