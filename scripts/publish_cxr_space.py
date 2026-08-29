#!/usr/bin/env python3
"""Build the demo subset and push the ChestX-ray14 Space.

The Space runs the probe live, so it needs states. Shipping all 112,120 would
make it slow to boot for no gain, so this cuts a test-split sample, uploads it
next to the states it came from, and pushes the app.

The sample is restricted to films whose image we can actually display. A chest
radiograph demo that shows only a table of probabilities asks the reader to take
the interesting part on trust, which is the habit this whole result is arguing
against. Images come from shard 1 of the source dataset and are downscaled,
since the probe reads the frozen state and never re-reads the pixels.

    python scripts/publish_cxr_space.py
"""
import argparse
import io
import json
import pathlib
import zipfile

import numpy as np
from huggingface_hub import HfApi, get_token, hf_hub_download

SPACE = "RiverRider/srt-cxr14-probe"
DATA_REPO = "RiverRider/srt-cxr14-frozen-probe"
SRC_REPO = "alkzar90/NIH-Chest-X-ray-dataset"

README = """---
title: ChestX-ray14 frozen probe
emoji: 🫁
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
---

A 339 KB `Linear(5376, 14)` probe on frozen `google/gemma-4-31B-it` states,
scoring 0.7590 mean AUROC on the official ChestX-ray14 split against 0.7451 for
the dataset authors' fine-tuned ResNet-50.

The backbone does not run here. The probe runs live on precomputed states.

Research artifact, not a diagnostic device. Detection, not early detection.
"""

REQS = "gradio==4.44.0\ntorch\nnumpy\nhuggingface_hub\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="artifacts/nla/cxr14/cxr14_gemma4.npz")
    ap.add_argument("--manifest", default="artifacts/nla/cxr14/cxr14_manifest.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--px", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from cxr_probe import FINDINGS
    from PIL import Image

    rows = json.load(open(a.manifest))["rows"]
    z = np.load(a.states)
    ok = z["ok"]
    rows = [r for r, k in zip(rows, ok) if k]
    X = z["item"][ok].astype(np.float32)

    print("reading image shard 1")
    zp = hf_hub_download(SRC_REPO, "data/images/images_001.zip", repo_type="dataset")
    zf = zipfile.ZipFile(zp)
    in_shard = {pathlib.PurePosixPath(m).name for m in zf.namelist()
                if m.lower().endswith(".png")}
    print(f"  {len(in_shard)} images available")

    te = np.array([i for i, r in enumerate(rows)
                   if r["split"] == "test" and r["key"] in in_shard])
    print(f"  {len(te)} of them are in the official test split")

    rng = np.random.default_rng(a.seed)
    # Bias toward positives, or a small cut is almost all normal and every
    # slider position reads "nothing found".
    has = np.array([bool(rows[i]["labels"]) for i in te])
    pos, neg = te[has], te[~has]
    n_pos = min(len(pos), int(a.n * 0.7))
    pick = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, min(len(neg), a.n - n_pos),
                                      replace=False)])
    rng.shuffle(pick)

    Y = np.zeros((len(pick), len(FINDINGS)), np.int8)
    for n, i in enumerate(pick):
        for j, f in enumerate(FINDINGS):
            if f in rows[i]["labels"]:
                Y[n, j] = 1

    img_dir = pathlib.Path("/tmp/cxr_demo_images")
    img_dir.mkdir(exist_ok=True)
    names = [m for m in zf.namelist() if m.lower().endswith(".png")]
    by_key = {pathlib.PurePosixPath(m).name: m for m in names}
    for i in pick:
        k = rows[i]["key"]
        with zf.open(by_key[k]) as fh:
            im = Image.open(io.BytesIO(fh.read())).convert("L")
        im.thumbnail((a.px, a.px))
        im.save(img_dir / k.replace(".png", ".jpg"), quality=85)
    mb_img = sum(p.stat().st_size for p in img_dir.iterdir()) / 1e6
    print(f"  {len(pick)} images downscaled to {a.px}px, {mb_img:.1f} MB")

    out = pathlib.Path("/tmp/test_subset.npz")
    np.savez_compressed(
        out, states=X[pick], labels=Y,
        keys=np.array([rows[i]["key"] for i in pick]),
        view=np.array([rows[i]["view"] for i in pick]))
    mb = out.stat().st_size / 1e6
    print(f"  subset: {len(pick)} films ({n_pos} with a finding), {mb:.1f} MB")

    api = HfApi(token=get_token())
    api.upload_file(path_or_fileobj=str(out), path_in_repo="demo/test_subset.npz",
                    repo_id=DATA_REPO, repo_type="dataset")
    print("  demo/test_subset.npz -> dataset repo")
    api.upload_folder(folder_path=str(img_dir), path_in_repo="demo/images",
                      repo_id=DATA_REPO, repo_type="dataset")
    print("  demo/images -> dataset repo")

    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    src = pathlib.Path("demo/cxr_space")
    for name, text in (("README.md", README), ("requirements.txt", REQS)):
        p = pathlib.Path("/tmp") / name
        p.write_text(text)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                        repo_id=SPACE, repo_type="space")
        print(f"  {name}")
    api.upload_file(path_or_fileobj=str(src / "app.py"), path_in_repo="app.py",
                    repo_id=SPACE, repo_type="space")
    print("  app.py")
    print(f"\nhttps://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
